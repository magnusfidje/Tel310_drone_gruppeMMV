"""Fly the course: ring, ladder, then slalom around the three poles to the bush marker.

    python fly_slalom.py          # fly everything, taking off from the start spot
    python fly_slalom.py slalom   # only the slalom: take off 0.5 m in front of pole 1
    python fly_slalom.py look     # do not fly, just show the camera and the estimated position

Flies exactly like fly_course.py, only the route is longer (see slalom_route).

Ctrl+C = emergency stop (motors off).  q in the video window = land.
"""
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from djitellopy import Tello

from src.kalman import PositionFilter
from src.killswitch import install_killswitch
from src.markers import (load_course, find_markers, locate_drone_robust, draw_markers,
                         estimate_camera_tilt)

# How much the camera looks downwards. Found by comparing the marker height with
# the floor distance sensor in five flight logs (with 10 degrees the drone
# thought it was 6-10 cm lower than it really was).
CAMERA_TILT_DEG = 11.8

RING_HEIGHT = 1.30        # centre of the ring (m)
LOWEST_HEIGHT = 0.30      # never aim lower than this, even if a marker hangs lower (m)
CLEARANCE = 0.50          # how far in front of / behind an obstacle we stop (m)


# Slalom
SIDE_STEP = 0.40          # how far out to the side we fly to get past a pole (m)
POLE_DISTANCE = 0.40      # how far in front of a pole we stop (m). Closer than this
                          # and the camera no longer sees the whole marker.
BUSH_DISTANCE = 0.60      # how far in front of the bush marker we stop (m)
ABOVE_MARKER = 0.05       # the camera looks downwards, so we hover a little above a
                          # marker to keep it in the picture when we are close (m)
RIGHT, LEFT = 1, -1       # z is positive to the right

# How close we must be to a waypoint, in [x, y, z].
ring_precise = [0.12, 0.04, 0.04]
height_precise = [0.12, 0.04, 0.10]     # before flying over/under a pole of the ladder
side_precise = [0.12, 0.10, 0.06]       # before flying past a slalom pole
past_precise = [0.08, 0.10, 0.10]       # before turning in behind a slalom pole
rough = [0.15, 0.10, 0.10]


def ring_and_ladder_route(marker_positions):
    """Waypoints (name, target [x, y, z], tolerance [x, y, z]) from take-off to past the ladder.

    The heights are read from the map, so they follow if you move a marker.
    """
    ladder_x = marker_positions[3][0]
    pole_x = marker_positions[15][0]

    over_yellow = marker_positions[5][1]                        # between blue and yellow pole
    under_yellow = marker_positions[8][1]                       # between yellow and red pole
    under_red = max(marker_positions[15][1], LOWEST_HEIGHT)     # between red pole and floor

    before_ladder = ladder_x - CLEARANCE
    after_ladder = ladder_x + CLEARANCE
    before_pole = pole_x - CLEARANCE

    return [
        # Ring
        ("in front of ring",       [0.20,          RING_HEIGHT,  0.0], ring_precise),
        ("behind ring",            [1.60,          RING_HEIGHT,  0.0], rough),
        # Ladder: forward over the yellow pole
        ("up to marker 5 height",  [1.60,          over_yellow,  0.0], height_precise),
        ("over yellow pole",       [after_ladder,  over_yellow,  0.0], rough),
        # Ladder: backwards between the yellow and the red pole
        ("down to marker 8 height", [after_ladder, under_yellow, 0.0], height_precise),
        ("back under yellow pole", [before_ladder, under_yellow, 0.0], rough),
        # Ladder: forward under the red pole
        ("down to marker 15 height", [before_ladder, under_red,  0.0], height_precise),
        ("under red pole",         [before_pole,   under_red,    0.0], rough),
    ]


def around_pole(route, pole_name, side, stop_x, stop_height, stop_name):
    """Add three waypoints to the route: out to the side, forward past the pole,
    and back in to the middle line at stop_x."""
    _, (x, height, _), _ = route[-1]            # where the route ends now
    z = side * SIDE_STEP
    route.append((f"out to the side of {pole_name}", [x,      height,      z],   side_precise))
    route.append((f"forward past {pole_name}",       [stop_x, height,      z],   past_precise))
    route.append((stop_name,                         [stop_x, stop_height, 0.0], rough))


def slalom_route(marker_positions):
    """Waypoints from 0.5 m in front of pole 1, around the three poles, to the bush marker."""
    pole_1 = marker_positions[5]
    pole_2 = marker_positions[6]
    pole_3 = marker_positions[7]
    bush = marker_positions[13]

    route = [("in front of pole 1", [pole_1[0] - CLEARANCE, pole_1[1], 0.0], rough)]

    # Right of pole 1, stop in front of pole 2
    around_pole(route, "pole 1", RIGHT, pole_2[0] - POLE_DISTANCE, pole_2[1] + ABOVE_MARKER,
                "in front of pole 2")
    # Left of pole 2, stop in front of pole 3
    around_pole(route, "pole 2", LEFT, pole_3[0] - POLE_DISTANCE, pole_3[1] + ABOVE_MARKER,
                "in front of pole 3")
    # Right of pole 3, stop in front of the bush marker
    around_pole(route, "pole 3", RIGHT, bush[0] - BUSH_DISTANCE, bush[1] + ABOVE_MARKER,
                "in front of bush marker")
    return route


# All settings with three numbers are for [forward x, up y, sideways z].
RC_TO_SPEED = np.array([0.007, 0.003, 0.005])   # m/s per rc, measured from the logs
APPROACH_RATE = 0.6       # wanted speed = APPROACH_RATE * error. With 0.6 s reaction
                          # time, more than about 1.0 makes it overshoot.
MAX_RC = np.array([15, 25, 15])                 # top command (rc goes from -100 to 100)
MIN_RC = 6                                      # smallest command we bother to send
CLOSE_ENOUGH = np.array([0.06, 0.025, 0.025])   # do not correct errors smaller than this (m).
                                                # Must be smaller than the waypoint tolerances!
FRAMES_ON_TARGET = 40     # must stay on the waypoint this many loops in a row (about 0.8 s)

MAX_HEIGHT = 1.9          # never climb above this (m)
WAYPOINT_TIMEOUT = 40     # give up on a waypoint after this long (s)
BLIND_TIMEOUT = 8         # land if no marker has been seen for this long (s).
                          # Climbing 0.5 m in front of a pole we see nothing for 4-5 s.

# How much we trust each measurement (std. dev. in metres, smaller = trust more).
# The markers are worst at the distance forward, best sideways.
MARKER_NOISE = np.array([0.15, 0.08, 0.05])
FLOOR_SENSOR_NOISE = 0.05
FLOOR_SENSOR_RANGE = (0.2, 3.0)     # readings outside this are not real (m)

LOG_FOLDER = Path(__file__).parent / "logs"
LOG_COLUMNS = ["time", "waypoint",
               "x", "y", "z",                           # where the filter thinks we are
               "marker_x", "marker_y", "marker_z",      # position from the markers alone
               "markers_seen", "marker_accepted",
               "tello_height", "height_accepted", "tello_tof", "tello_pitch",
               "rc_forward", "rc_up", "rc_right"]


def read_camera(frame_reader):
    return cv2.cvtColor(frame_reader.frame, cv2.COLOR_RGB2BGR)


def speed_towards(target, position):
    """P-controller: the further away we are, the faster we fly towards the target."""
    error = np.array(target) - position
    wanted_speed = APPROACH_RATE * error                        # m/s
    rc = np.clip(wanted_speed / RC_TO_SPEED, -MAX_RC, MAX_RC)

    # Very small commands may not move the Tello at all, so send at least MIN_RC ...
    too_weak = np.abs(rc) < MIN_RC
    rc[too_weak] = MIN_RC * np.sign(error[too_weak])
    # ... unless we are close enough: then leave it alone.
    rc[np.abs(error) <= CLOSE_ENOUGH] = 0
    return rc.astype(int)


def show(frame, seen, position, text):
    draw_markers(frame, seen)
    x, y, z = position
    cv2.putText(frame, f"{text}   x={x:.2f} y={y:.2f} z={z:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("tello", frame)
    return cv2.waitKey(1) & 0xFF == ord("q")


def start_log():
    """Open a new log file, logs/flight_<date>_<time>.csv, and write the header."""
    LOG_FOLDER.mkdir(exist_ok=True)
    path = LOG_FOLDER / time.strftime("flight_%Y%m%d_%H%M%S.csv")
    file = open(path, "w", newline="", encoding="utf-8")
    log = csv.writer(file)
    log.writerow(LOG_COLUMNS)
    print(f"Logging to {path}")
    return file, log


def look(frame_reader, marker_positions, marker_size):
    """No flying. Carry the drone around and check that the position makes sense."""
    while True:
        frame = read_camera(frame_reader)
        seen = find_markers(frame, marker_positions)
        position = locate_drone_robust(seen, marker_positions, marker_size, CAMERA_TILT_DEG)
        if position is None:
            position = np.zeros(3)

        # Shows the camera tilt when both ring and ladder markers are visible.
        # Put that number in CAMERA_TILT_DEG in src/markers.py.
        tilt = estimate_camera_tilt(seen, marker_positions, marker_size)
        tilt_text = "tilt ?" if tilt is None else f"tilt {tilt:.1f} deg"
        if show(frame, seen, position, f"markers {sorted(seen)}  {tilt_text}"):
            break


def fly(tello, frame_reader, marker_positions, marker_size, waypoints, start_x):
    """Take off at start_x on the middle line and fly through the waypoints."""
    tello.takeoff()
    start_height = tello.get_distance_tof() / 100
    estimate = PositionFilter([start_x, start_height, 0.0])

    waypoint_number = 0
    frames_on_target = 0
    waypoint_start = time.time()
    last_marker_time = time.time()
    last_loop_time = time.time()
    velocity = np.zeros(3)
    last_print_time = 0
    log_file, log = start_log()
    flight_start = time.time()

    while waypoint_number < len(waypoints):
        name, target, tolerance = waypoints[waypoint_number]
        now = time.time()

        # 1. Predict: we have been flying with `velocity` since the last loop
        estimate.predict(velocity, now - last_loop_time)
        last_loop_time = now

        # 2. Update with the markers we see
        frame = read_camera(frame_reader)
        seen = find_markers(frame, marker_positions)
        marker_position = locate_drone_robust(seen, marker_positions, marker_size,
                                              CAMERA_TILT_DEG)
        marker_accepted = False
        if marker_position is not None:
            marker_accepted = estimate.update(marker_position, MARKER_NOISE)
            last_marker_time = now

        # 3. Update with the distance to the floor from the Tello sensor (ToF).
        #    The filter ignores it if it jumps, for example when the drone is
        #    over the ring or a pole and the sensor sees that instead of the floor.
        tello_tof = tello.get_distance_tof() / 100
        height_accepted = False
        if FLOOR_SENSOR_RANGE[0] < tello_tof < FLOOR_SENSOR_RANGE[1]:
            height_accepted = estimate.update_height(tello_tof, FLOOR_SENSOR_NOISE)
        tello_height = tello.get_height() / 100       # only logged, for comparison

        # 4. Are we there yet?
        error = np.array(target) - estimate.position
        on_target = np.all(np.abs(error) < tolerance)
        frames_on_target = frames_on_target + 1 if on_target else 0
        if frames_on_target >= FRAMES_ON_TARGET:
            print(f"Reached '{name}' at {estimate.position.round(2)}")
            waypoint_number += 1
            frames_on_target = 0
            waypoint_start = now
            continue

        # 5. Safety checks
        if now - waypoint_start > WAYPOINT_TIMEOUT:
            print(f"Did not reach '{name}' in time -> landing")
            break
        if now - last_marker_time > BLIND_TIMEOUT:
            print("No markers seen -> landing")
            break

        # 6. Fly towards the waypoint
        forward, up, right = speed_towards(target, estimate.position)
        if estimate.position[1] > MAX_HEIGHT:
            up = min(up, 0)
        tello.send_rc_control(int(right), int(forward), int(up), 0)
        velocity = np.array([forward, up, right]) * RC_TO_SPEED

        # 7. Write one line in the log
        marker_xyz = ["", "", ""] if marker_position is None else marker_position.round(3)
        log.writerow([round(now - flight_start, 2), name,
                      *estimate.position.round(3), *marker_xyz,
                      " ".join(str(i) for i in sorted(seen)), int(marker_accepted),
                      tello_height, int(height_accepted), tello_tof, tello.get_pitch(),
                      forward, up, right])
        log_file.flush()    # so nothing is lost if we hit the kill switch

        status = f"{name}  markers {sorted(seen)}"
        if now - last_print_time > 1:
            print(f"-> {name}: position {estimate.position.round(2)}, "
                  f"error {error.round(2)}, markers {sorted(seen)}, "
                  f"tello height {tello_height:.2f}, "
                  f"rc forward/up/right = {forward}/{up}/{right}")
            last_print_time = now
        if marker_position is not None and not marker_accepted:
            status += "  MARKERS IGNORED"
        if not height_accepted:
            status += "  HEIGHT IGNORED"
        if show(frame, seen, estimate.position, status):
            print("q pressed -> landing")
            break

    log_file.close()
    tello.send_rc_control(0, 0, 0, 0)
    tello.land()


def main():
    marker_positions, marker_size = load_course()

    tello = Tello()
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")
    install_killswitch(tello)
    tello.streamon()
    frame_reader = tello.get_frame_read()
    time.sleep(1)

    try:
        mode = sys.argv[1] if len(sys.argv) > 1 else "all"
        slalom = slalom_route(marker_positions)
        if mode == "look":
            look(frame_reader, marker_positions, marker_size)
        elif mode == "slalom":
            start_x = slalom[0][1][0]           # the x of "in front of pole 1"
            print(f"Slalom only. The drone must stand at x = {start_x:.2f} m, on the middle line.")
            fly(tello, frame_reader, marker_positions, marker_size, slalom, start_x)
        else:
            everything = ring_and_ladder_route(marker_positions) + slalom
            fly(tello, frame_reader, marker_positions, marker_size, everything, start_x=0.0)
    except KeyboardInterrupt:
        print("Stopped by kill switch")
    finally:
        cv2.destroyAllWindows()
        tello.streamoff()
        tello.end()


if __name__ == "__main__":
    main()
