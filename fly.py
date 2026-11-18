"""Fly the course: go from waypoint to waypoint using the estimated position.

    python fly.py          # fly
    python fly.py look     # do not fly, just show the camera and the estimated position

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
from src.markers import (load_course, find_markers, locate_drone, draw_markers,
                         estimate_camera_tilt)

# Where to fly: (name, target [x, y, z], how close we must be in [x, y, z])
# In front of the ring the height (y) and sideways (z) must be precise,
# because the opening is small. How far forward (x) we are matters less.
WAYPOINTS = [
    ("in front of ring", [0.20, 1.30, 0.0], [0.10, 0.04, 0.04]),
    ("behind ring",      [1.60, 1.30, 0.0], [0.15, 0.10, 0.10]),
]

GAIN = 100                # rc per metre of error
MAX_RC = 20               # top speed command (rc goes from -100 to 100)
MIN_RC = 15               # the Tello barely moves on smaller commands than this
CLOSE_ENOUGH = 0.02       # do not correct errors smaller than this (m).
                          # Must be smaller than the waypoint tolerances!
RC_TO_SPEED = 0.01        # rc 100 is roughly 1 m/s. TODO: tune
FRAMES_ON_TARGET = 5      # must stay on the waypoint this many loops in a row

MAX_HEIGHT = 1.9          # never climb above this (m)
WAYPOINT_TIMEOUT = 25     # give up on a waypoint after this long (s)
BLIND_TIMEOUT = 3         # land if no marker has been seen for this long (s)

# How much we trust each height source (std. dev. in metres, smaller = trust more)
MARKER_NOISE = 0.08
TELLO_HEIGHT_NOISE = 0.15

LOG_FOLDER = Path(__file__).parent / "logs"
LOG_COLUMNS = ["time", "waypoint",
               "x", "y", "z",                           # where the filter thinks we are
               "marker_x", "marker_y", "marker_z",      # position from the markers alone
               "markers_seen", "marker_accepted",
               "tello_height", "height_accepted", "tello_tof",
               "rc_forward", "rc_up", "rc_right"]


def read_camera(frame_reader):
    return cv2.cvtColor(frame_reader.frame, cv2.COLOR_RGB2BGR)


def speed_towards(target, position):
    """P-controller: fly towards the target, faster the further away we are."""
    error = np.array(target) - position
    rc = np.clip(GAIN * error, -MAX_RC, MAX_RC)

    # A small error gives a command so small that the Tello does not move at all.
    # Then it would hang just outside the waypoint forever, so push at least MIN_RC.
    too_weak = (np.abs(error) > CLOSE_ENOUGH) & (np.abs(rc) < MIN_RC)
    rc[too_weak] = MIN_RC * np.sign(error[too_weak])
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
        position = locate_drone(seen, marker_positions, marker_size)
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
    start_height = tello.get_height() / 100
    estimate = PositionFilter([0.0, start_height, 0.0])

    waypoint_number = 0
    frames_on_target = 0
    waypoint_start = time.time()
    last_marker_time = time.time()
    last_loop_time = time.time()
    velocity = np.zeros(3)
    last_print_time = 0
    log_file, log = start_log()
    flight_start = time.time()

    while waypoint_number < len(WAYPOINTS):
        name, target, tolerance = WAYPOINTS[waypoint_number]
        now = time.time()

        # 1. Predict: we have been flying with `velocity` since the last loop
        estimate.predict(velocity, now - last_loop_time)
        last_loop_time = now

        # 2. Update with the markers we see
        frame = read_camera(frame_reader)
        seen = find_markers(frame, marker_positions)
        marker_position = locate_drone(seen, marker_positions, marker_size)
        marker_accepted = False
        if marker_position is not None:
            marker_accepted = estimate.update(marker_position, MARKER_NOISE)
            last_marker_time = now

        # 3. Update with the Tello height. The filter ignores it if it jumps,
        #    for example when the drone is over the ring.
        tello_height = tello.get_height() / 100
        height_accepted = estimate.update_height(tello_height, TELLO_HEIGHT_NOISE)
        tello_tof = tello.get_distance_tof() / 100    # only logged, for comparison

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
                      tello_height, int(height_accepted), tello_tof,
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
