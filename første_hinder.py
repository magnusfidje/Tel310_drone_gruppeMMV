"""Test script for the course.

    python test_flight.py detect   # camera only, no flying (hold the drone)
    python test_flight.py step1    # take off, climb until a ring marker is seen, hover, land

Ctrl+C = emergency stop. Press q in the video window = normal landing.
"""
import sys
import time

import cv2
from djitellopy import Tello

from src.aruco_utils import load_course, make_detector, detect_markers, draw_markers
from src.killswitch import install_killswitch

TARGET_OBSTACLE = "ring1"
CLIMB_SPEED = 20        # rc up speed (-100..100)
MAX_HEIGHT_CM = 200     # safety: stop climbing here
CLIMB_TIMEOUT_S = 10
FRAMES_NEEDED = 3       # marker must be seen in this many frames in a row
HOVER_TIME_S = 3


def get_frame(frame_reader):
    # djitellopy gives RGB; OpenCV wants BGR
    return cv2.cvtColor(frame_reader.frame, cv2.COLOR_RGB2BGR)


def show(frame, found, text=""):
    draw_markers(frame, found)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("tello", frame)
    return cv2.waitKey(1) & 0xFF


def run_detect(tello, frame_reader, detector, markers, size):
    while True:
        frame = get_frame(frame_reader)
        found = detect_markers(frame, detector, markers, size)
        for m in found:
            print(f'id {m["id"]:>2} {m["role"]:<6} dist {m["distance"]:.2f} m  tvec {m["tvec"].round(2)}')
        if show(frame, found, f"battery {tello.get_battery()}%") == ord("q"):
            break


def run_step1(tello, frame_reader, detector, markers, size):
    tello.takeoff()
    print(f"Took off, height {tello.get_height()} cm. Climbing...")

    seen_in_a_row = 0
    start = time.time()
    while True:
        frame = get_frame(frame_reader)
        found = [m for m in detect_markers(frame, detector, markers, size)
                 if m["obstacle"] == TARGET_OBSTACLE]
        height = tello.get_height()

        seen_in_a_row = seen_in_a_row + 1 if found else 0
        if seen_in_a_row >= FRAMES_NEEDED:
            tello.send_rc_control(0, 0, 0, 0)
            print(f"Ring seen at {height} cm: ids {[m['id'] for m in found]}")
            break
        if height >= MAX_HEIGHT_CM or time.time() - start > CLIMB_TIMEOUT_S:
            tello.send_rc_control(0, 0, 0, 0)
            print(f"No ring found (height {height} cm) -> landing")
            break

        tello.send_rc_control(0, 0, CLIMB_SPEED, 0)   # left/right, fwd/back, up/down, yaw
        if show(frame, found, f"CLIMB  h={height}cm") == ord("q"):
            break

    # Hover and keep showing the video
    end = time.time() + HOVER_TIME_S
    while time.time() < end:
        tello.send_rc_control(0, 0, 0, 0)
        frame = get_frame(frame_reader)
        show(frame, detect_markers(frame, detector, markers, size), "HOVER")

    tello.land()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "detect"
    config, markers = load_course()
    detector = make_detector(config["dictionary"])
    size = config["marker_size"]

    tello = Tello()
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")
    install_killswitch(tello)

    tello.streamon()
    frame_reader = tello.get_frame_read()
    time.sleep(1)   # let the video stream start

    try:
        if mode == "detect":
            run_detect(tello, frame_reader, detector, markers, size)
        elif mode == "step1":
            run_step1(tello, frame_reader, detector, markers, size)
        else:
            print(f"Unknown mode: {mode}")
    except KeyboardInterrupt:
        print("Aborted by kill switch")
    except Exception as e:
        print(f"Error: {e} -> landing")
        tello.land()
        raise
    finally:
        cv2.destroyAllWindows()
        tello.streamoff()
        tello.end()


if __name__ == "__main__":
    main()