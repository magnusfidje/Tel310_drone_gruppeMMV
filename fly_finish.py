"""The last part: up over the bush, towards the board (marker 19), and land 0.5 m in front of it.

    python fly_finish.py finish   # only this part: take off 0.6 m in front of marker 13
                                  # (x = 5.88, middle line, nose along the course)
    python fly_finish.py bush     # the 360 around the bush, then this part
    python fly_finish.py          # the whole course: ring, ladder, slalom, bush, landing

fly() in fly_bush.py lands by itself after the last waypoint, so the route simply
ends at the landing spot. fly_bush.py is not changed.

The landing spot is read from config/course_map.yaml, as a top-level line
(NOT under markers:, then load_course() would crash):

    landing_pad: [7.66, 0.0, 0.0]     # x, y, z of the centre of the pad

Ctrl+C = emergency stop (motors off).  q in the video window = land.
"""
import sys
import time

import cv2
import numpy as np
import yaml
from djitellopy import Tello

from fly_slalom import ABOVE_MARKER, ring_and_ladder_route, slalom_route
from fly_bush import BUSH_MARKERS, waypoint, from_old_route, bush_route, fly
from src.killswitch import install_killswitch
from src.markers import COURSE_FILE, load_course

BUSH_TOP_HEIGHT = 1.10    # highest point of the bush (m). MEASURE IT!
OVER_BUSH = 0.40          # how far above the bush we fly (m)
PAST_BUSH = 0.45          # how far behind the back of the bush we go down again (m)
LAND_DISTANCE = 0.50      # land this far in front of marker 19 (m)

FINISH_MARKERS = BUSH_MARKERS | {19}    # ignore everything else (floor mat)

# Tolerances in the drone's own frame [forward, up, sideways] (m)
height_precise = [0.15, 0.04, 0.08]     # before flying over / after flying down
rough = [0.15, 0.10, 0.10]
land_precise = [0.10, 0.06, 0.06]       # hold still here, then land


def load_landing_pad(marker_positions):
    """[x, z] of the landing pad. Falls back to straight in front of marker 19."""
    with open(COURSE_FILE, encoding="utf-8") as file:
        course = yaml.safe_load(file)
    board = marker_positions[19]
    if "landing_pad" not in course:
        print("No landing_pad in course_map.yaml -> landing on the middle line.")
        return np.array([board[0] - LAND_DISTANCE, board[2]])

    pad = np.array(course["landing_pad"], dtype=float)
    if abs(pad[0] - (board[0] - LAND_DISTANCE)) > 0.15:
        print(f"Note: the landing pad is at x = {pad[0]:.2f}, but we land "
              f"{LAND_DISTANCE} m in front of marker 19, at x = {board[0] - LAND_DISTANCE:.2f}.")
    return pad[[0, 2]]


def finish_route(marker_positions, yaw):
    """From in front of marker 13: up, over the bush, down, and to the landing spot.

    yaw = the yaw the drone has when it starts this part. After a full circle
    to the right that is 360, not 0, otherwise it would turn all the way back.
    """
    bush_front = marker_positions[13]
    bush_back = marker_positions[11]
    board = marker_positions[19]
    pad_x, pad_z = load_landing_pad(marker_positions)

    start_x = bush_front[0] - 0.60
    over_height = BUSH_TOP_HEIGHT + OVER_BUSH
    behind_bush = bush_back[0] + PAST_BUSH
    land_x = board[0] - LAND_DISTANCE
    # Low enough that marker 19 stays in the picture when we are close
    # (the camera looks downwards), but never into the bush.
    approach_height = board[1] + ABOVE_MARKER

    def point(name, target, tolerance, frames=None):
        extra = {} if frames is None else {"frames": frames}
        return waypoint(name, target, tolerance, yaw=yaw, markers=FINISH_MARKERS, **extra)

    return [
        point("finish: up in front of the bush",  [start_x,     over_height,     0.0],   height_precise),
        point("finish: over the bush",            [behind_bush, over_height,     0.0],   rough),
        point("finish: down behind the bush",     [behind_bush, approach_height, 0.0],   height_precise),
        point(f"finish: {LAND_DISTANCE} m in front of marker 19",
              [land_x, approach_height, pad_z], land_precise),
    ]


def main():
    marker_positions, marker_size = load_course()
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "finish":
        route = [waypoint("start: in front of marker 13", [5.88, 0.90, 0.0], rough,
                          markers=FINISH_MARKERS)] + finish_route(marker_positions, yaw=0.0)
    elif mode == "bush":
        circle = bush_route(marker_positions)
        route = circle + finish_route(marker_positions, yaw=circle[-1]["yaw"])
    else:
        circle = bush_route(marker_positions)
        route = (from_old_route(ring_and_ladder_route(marker_positions))
                 + from_old_route(slalom_route(marker_positions))
                 + circle + finish_route(marker_positions, yaw=circle[-1]["yaw"]))

    if mode in ("finish", "bush"):
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
        fly(tello, frame_reader, marker_positions, marker_size, route, start_x)
    except KeyboardInterrupt:
        print("Stopped by kill switch")
    finally:
        cv2.destroyAllWindows()
        tello.streamoff()
        tello.end()


if __name__ == "__main__":
    main()
