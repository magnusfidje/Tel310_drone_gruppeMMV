"""The 360 around the bush: fly one full circle around it with the nose pointing at it.

    python fly_bush.py look      # do not fly: carry the drone around the bush and check
                                 # the position. Start with the nose pointing along +x!
    python fly_bush.py turn      # take off in front of marker 13, turn 30 deg right, back,
                                 # 30 deg left, back. Tests the yaw sign and YAW_RC_TO_DEG_S.
    python fly_bush.py quarter   # only the first quarter: marker 13 -> marker 9
    python fly_bush.py bush      # the whole circle, 13 -> 9 -> 11 -> 16 -> 13
    python fly_bush.py           # everything: ring, ladder, slalom and the bush

For turn, quarter and bush the drone stands 0.6 m in front of marker 13
(x = 5.88, on the middle line) with the nose pointing along the course.

New compared with fly_slalom.py (which is not changed):
  - Every waypoint also has a yaw. The yaw comes from the Tello IMU (get_yaw).
  - Marker vectors and commands are turned with the yaw (src/yaw.py), so the
    Kalman filter for [x, y, z] stays linear. No EKF.
  - Waypoint tolerances are in the drone's own frame [forward, up, sideways].
  - Around the bush only the bush markers are used. In the look video, marker 17
    (the floor mat?) gave a position 3.5 m wrong when the drone was behind the bush.

Ctrl+C = emergency stop (motors off).  q in the video window = land.
"""
import csv
import sys
import time

import cv2
import numpy as np
from djitellopy import Tello

from fly_slalom import (CAMERA_TILT_DEG, ABOVE_MARKER, BUSH_DISTANCE, RIGHT, LEFT,
                        RC_TO_SPEED, FRAMES_ON_TARGET, MAX_HEIGHT, WAYPOINT_TIMEOUT,
                        BLIND_TIMEOUT, MARKER_NOISE, FLOOR_SENSOR_NOISE, FLOOR_SENSOR_RANGE,
                        LOG_FOLDER, read_camera, speed_towards, ring_and_ladder_route,
                        slalom_route)
from src.kalman import PositionFilter
from src.killswitch import install_killswitch
from src.markers import load_course, find_markers, draw_markers
from src.yaw import (body_to_world, world_to_body, heading, start_yaw_tracker,
                     update_yaw_tracker, locate_drone_with_yaw, marker_noise_in_world,
                     update_position)

TURN_DIRECTION = RIGHT    # RIGHT: 13 -> 9 -> 11 -> 16 -> 13.   LEFT: 13 -> 16 -> 11 -> 9 -> 13

# Which bush marker the camera looks at for each yaw (degrees, 0 = along the course).
BUSH_MARKER_AT_YAW = {0: 13, 90: 9, 180: 11, 270: 16}
BUSH_MARKERS = set(BUSH_MARKER_AT_YAW.values())

ARC_STEPS = 4             # waypoints per quarter circle (4 = one every 22.5 deg)

# Tolerances in the drone's own frame [forward, up, sideways] (m) and for the yaw (deg).
station_precise = [0.08, 0.05, 0.05]    # centred in front of a bush marker
on_the_arc = [0.12, 0.10, 0.10]         # just passing, no need to be exact
STATION_YAW_TOLERANCE = 5.0
ARC_YAW_TOLERANCE = 15.0
ARC_FRAMES = 3            # passing points only need to be hit, not held

# Yaw controller. Same idea as speed_towards: wanted turn rate = rate * error.
YAW_RC_TO_DEG_S = 1.0     # deg/s per rc. A GUESS - measure it with `turn` and the log.
YAW_APPROACH_RATE = 0.6   # 1/s
MAX_YAW_RC = 20
MIN_YAW_RC = 6
YAW_CLOSE_ENOUGH = 2.0    # deg. Must be smaller than STATION_YAW_TOLERANCE.

LOG_COLUMNS = ["time", "waypoint",
               "x", "y", "z", "yaw",                    # filter position, IMU yaw
               "target_yaw",
               "marker_x", "marker_y", "marker_z",
               "markers_seen", "marker_accepted",
               "tello_tof", "height_accepted",
               "rc_forward", "rc_up", "rc_right", "rc_yaw"]


# ---------------------------------------------------------------- routes

def waypoint(name, target, tolerance, yaw=0.0, yaw_tolerance=STATION_YAW_TOLERANCE,
             frames=FRAMES_ON_TARGET, markers=None):
    """One point on the route. markers=None means: use every marker in the map."""
    return {"name": name, "target": list(target), "tolerance": np.array(tolerance),
            "yaw": yaw, "yaw_tolerance": yaw_tolerance, "frames": frames,
            "markers": markers}


def from_old_route(route):
    """Turn (name, target, tolerance) tuples from fly_slalom.py into waypoints with yaw 0."""
    return [waypoint(name, target, tolerance) for name, target, tolerance in route]


def bush_station(marker_positions, yaw):
    """[x, z] BUSH_DISTANCE in front of the bush marker the drone faces at this yaw."""
    marker = marker_positions[BUSH_MARKER_AT_YAW[round(yaw) % 360]]
    return marker[[0, 2]] - BUSH_DISTANCE * heading(yaw)


def bush_route(marker_positions, quarters=4, direction=TURN_DIRECTION):
    """Stop in front of marker 13, then go round the bush one quarter at a time.

    Between two stations the drone follows an arc around the centre of the bush
    while it turns, so the nose keeps pointing at the bush. At every station it
    centres itself in front of the marker and holds still before going on.
    """
    centre = np.mean([marker_positions[i][[0, 2]] for i in BUSH_MARKERS], axis=0)
    height = marker_positions[13][1] + ABOVE_MARKER

    def point(xz):
        return [xz[0], height, xz[1]]

    route = [waypoint("bush: in front of marker 13", point(bush_station(marker_positions, 0)),
                      station_precise, yaw=0.0, markers=BUSH_MARKERS)]

    for quarter in range(quarters):
        yaw_from = direction * 90 * quarter
        yaw_to = direction * 90 * (quarter + 1)
        radius_from = np.linalg.norm(bush_station(marker_positions, yaw_from) - centre)
        radius_to = np.linalg.norm(bush_station(marker_positions, yaw_to) - centre)

        for step in range(1, ARC_STEPS):
            share = step / ARC_STEPS
            yaw = yaw_from + share * (yaw_to - yaw_from)
            radius = radius_from + share * (radius_to - radius_from)
            route.append(waypoint(f"bush: arc at {yaw:.0f} deg",
                                  point(centre - radius * heading(yaw)), on_the_arc,
                                  yaw=yaw, yaw_tolerance=ARC_YAW_TOLERANCE,
                                  frames=ARC_FRAMES, markers=BUSH_MARKERS))

        marker_id = BUSH_MARKER_AT_YAW[yaw_to % 360]
        route.append(waypoint(f"bush: in front of marker {marker_id}",
                              point(bush_station(marker_positions, yaw_to)), station_precise,
                              yaw=yaw_to, markers=BUSH_MARKERS))
    return route


def turn_test_route(marker_positions):
    """Stay in front of marker 13 and only turn: +30, 0, -30, 0 degrees."""
    start = bush_route(marker_positions, quarters=0)[0]
    route = []
    for yaw in [0, 30, 0, -30, 0]:
        route.append(waypoint(f"turn test: yaw {yaw}", start["target"], station_precise,
                              yaw=yaw, markers=BUSH_MARKERS))
    return route


# ---------------------------------------------------------------- control

def yaw_rate_towards(target_yaw, yaw):
    """P-controller for the yaw. Returns the rc command (positive = turn right)."""
    error = target_yaw - yaw
    if abs(error) <= YAW_CLOSE_ENOUGH:
        return 0
    rc = np.clip(YAW_APPROACH_RATE * error / YAW_RC_TO_DEG_S, -MAX_YAW_RC, MAX_YAW_RC)
    if abs(rc) < MIN_YAW_RC:
        rc = MIN_YAW_RC * np.sign(error)
    return int(rc)


def markers_to_use(seen, allowed):
    if allowed is None:
        return seen
    return {i: pixels for i, pixels in seen.items() if i in allowed}


def show(frame, seen, position, yaw, text):
    draw_markers(frame, seen)
    x, y, z = position
    cv2.putText(frame, f"{text}   x={x:.2f} y={y:.2f} z={z:.2f} yaw={yaw:.0f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("tello", frame)
    return cv2.waitKey(1) & 0xFF == ord("q")


def start_log():
    LOG_FOLDER.mkdir(exist_ok=True)
    path = LOG_FOLDER / time.strftime("bush_%Y%m%d_%H%M%S.csv")
    file = open(path, "w", newline="", encoding="utf-8")
    log = csv.writer(file)
    log.writerow(LOG_COLUMNS)
    print(f"Logging to {path}")
    return file, log


def look(tello, frame_reader, marker_positions, marker_size):
    """No flying. Hold the nose along +x when you start, then walk around the bush.

    When a bush marker is visible only the bush markers are used (like in flight).
    Turn the drone to the right by hand: yaw must go UP. If it goes down, the
    sign of the Tello yaw is the other way and src/yaw.py must be changed.
    """
    tracker = start_yaw_tracker(tello.get_yaw())
    while True:
        yaw = update_yaw_tracker(tracker, tello.get_yaw())
        frame = read_camera(frame_reader)
        seen = find_markers(frame, marker_positions)
        bush_seen = markers_to_use(seen, BUSH_MARKERS)
        used = bush_seen if bush_seen else seen
        position = locate_drone_with_yaw(used, marker_positions, marker_size,
                                         CAMERA_TILT_DEG, yaw)
        if position is None:
            position = np.zeros(3)
        if show(frame, seen, position, yaw, f"markers {sorted(used)}"):
            break


def fly(tello, frame_reader, marker_positions, marker_size, waypoints, start_x):
    """Take off at start_x on the middle line, nose along +x, and fly the waypoints."""
    tracker = start_yaw_tracker(tello.get_yaw())     # before take-off: nose along +x
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
        point = waypoints[waypoint_number]
        name = point["name"]
        now = time.time()

        # 1. Predict with the speed we commanded (already turned into the world frame)
        estimate.predict(velocity, now - last_loop_time)
        last_loop_time = now
        yaw = update_yaw_tracker(tracker, tello.get_yaw())

        # 2. Update with the markers. The camera points where the nose points.
        frame = read_camera(frame_reader)
        seen = markers_to_use(find_markers(frame, marker_positions), point["markers"])
        marker_position = locate_drone_with_yaw(seen, marker_positions, marker_size,
                                                CAMERA_TILT_DEG, yaw)
        marker_accepted = False
        if marker_position is not None:
            R = marker_noise_in_world(MARKER_NOISE, yaw)
            marker_accepted = update_position(estimate, marker_position, R)
            last_marker_time = now

        # 3. Height from the floor sensor (does not care about the yaw)
        tello_tof = tello.get_distance_tof() / 100
        height_accepted = False
        if FLOOR_SENSOR_RANGE[0] < tello_tof < FLOOR_SENSOR_RANGE[1]:
            height_accepted = estimate.update_height(tello_tof, FLOOR_SENSOR_NOISE)

        # 4. Are we there yet? Position error in the drone's own frame.
        error = world_to_body(np.array(point["target"]) - estimate.position, yaw)
        yaw_error = point["yaw"] - yaw
        on_target = (np.all(np.abs(error) < point["tolerance"])
                     and abs(yaw_error) < point["yaw_tolerance"])
        frames_on_target = frames_on_target + 1 if on_target else 0
        if frames_on_target >= point["frames"]:
            print(f"Reached '{name}' at {estimate.position.round(2)}, yaw {yaw:.0f}")
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

        # 6. Fly: the same P-controller as before, but on the error in the body frame
        forward, up, right = speed_towards(error, np.zeros(3))
        if estimate.position[1] > MAX_HEIGHT:
            up = min(up, 0)
        rc_yaw = yaw_rate_towards(point["yaw"], yaw)
        tello.send_rc_control(int(right), int(forward), int(up), rc_yaw)
        velocity = body_to_world(np.array([forward, up, right]) * RC_TO_SPEED, yaw)

        # 7. Log
        marker_xyz = ["", "", ""] if marker_position is None else marker_position.round(3)
        log.writerow([round(now - flight_start, 2), name,
                      *estimate.position.round(3), round(yaw, 1), round(point["yaw"], 1),
                      *marker_xyz, " ".join(str(i) for i in sorted(seen)),
                      int(marker_accepted), tello_tof, int(height_accepted),
                      forward, up, right, rc_yaw])
        log_file.flush()

        status = f"{name}  markers {sorted(seen)}"
        if now - last_print_time > 1:
            print(f"-> {name}: position {estimate.position.round(2)}, yaw {yaw:.0f} "
                  f"(target {point['yaw']:.0f}), error {error.round(2)}, "
                  f"markers {sorted(seen)}, rc f/u/r/yaw = {forward}/{up}/{right}/{rc_yaw}")
            last_print_time = now
        if marker_position is not None and not marker_accepted:
            status += "  MARKERS IGNORED"
        if show(frame, seen, estimate.position, yaw, status):
            print("q pressed -> landing")
            break

    log_file.close()
    tello.send_rc_control(0, 0, 0, 0)
    tello.land()


def main():
    marker_positions, marker_size = load_course()
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "turn":
        route = turn_test_route(marker_positions)
    elif mode == "quarter":
        route = bush_route(marker_positions, quarters=1)
    elif mode == "bush":
        route = bush_route(marker_positions)
    else:
        route = (from_old_route(ring_and_ladder_route(marker_positions))
                 + from_old_route(slalom_route(marker_positions))
                 + bush_route(marker_positions))

    if mode in ("turn", "quarter", "bush"):
        start_x = route[0]["target"][0]
        print(f"The drone must stand at x = {start_x:.2f} m on the middle line, "
              f"nose along the course.")
    else:
        start_x = 0.0

    tello = Tello()
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")
    install_killswitch(tello)
    tello.streamon()
    frame_reader = tello.get_frame_read()
    time.sleep(1)

    try:
        if mode == "look":
            look(tello, frame_reader, marker_positions, marker_size)
        else:
            fly(tello, frame_reader, marker_positions, marker_size, route, start_x)
    except KeyboardInterrupt:
        print("Stopped by kill switch")
    finally:
        cv2.destroyAllWindows()
        tello.streamoff()
        tello.end()


if __name__ == "__main__":
    main()
