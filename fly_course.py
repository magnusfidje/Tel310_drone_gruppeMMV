"""Fly the course: through the ring, then weave through the ladder.

    python fly_course.py          # fly
    python fly_course.py look     # do not fly, just show the camera and the estimated position

Same idea as fly_smooth.py, but more careful:
  - slower, and it must rest on a waypoint for a moment before it continues
  - the height comes mainly from the Tello distance sensor that points at the floor
  - near markers count more than far ones

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


def make_waypoints(marker_positions):
    """The route as a list of (name, target [x, y, z], how close we must be in [x, y, z]).

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

    # Before we fly through something the height must be precise.
    # Sideways matters less at the ladder, because the poles are wide.
    ring_precise = [0.12, 0.04, 0.04]
    height_precise = [0.12, 0.04, 0.10]
    rough = [0.15, 0.10, 0.10]

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
        # Up in front of the first slalom pole
        ("up to marker 5 height again", [before_pole, over_yellow, 0.0], rough),
    ]


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


def fly(tello, frame_reader, marker_positions, marker_size):
    tello.takeoff()
    start_height = tello.get_distance_tof() / 100
    estimate = PositionFilter([0.0, start_height, 0.0])

    waypoints = make_waypoints(marker_positions)
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
        if len(sys.argv) > 1 and sys.argv[1] == "look":
            look(frame_reader, marker_positions, marker_size)
        else:
            fly(tello, frame_reader, marker_positions, marker_size)
    except KeyboardInterrupt:
        print("Stopped by kill switch")
    finally:
        cv2.destroyAllWindows()
        tello.streamoff()
        tello.end()


if __name__ == "__main__":
    main()
