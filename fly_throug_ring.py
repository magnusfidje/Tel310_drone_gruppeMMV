"""Fly through the ring.

Who controls what:
    up/down + left/right -> ring centre (from the ring markers + offsets), when visible
    yaw                  -> marker 3 (ladder, middle pole) centred left/right, when visible
Marker 3 is NOT used for height: at ~1.3 m it hides behind the ring's top marker.

State machine:
    SEARCH  - climb until a ring marker is seen
    ALIGN   - hover and correct until the ring centre is straight ahead
              (and marker 3 is centred, if visible)
    THROUGH - tello.move_forward(distance to ring + PASS_MARGIN_CM). The Tello measures
              this itself with the downward camera (floor texture), so it does not
              matter that the ring markers leave the image.
    HOVER   - hover a few seconds, then land

Ctrl+C = emergency stop. q in the video window = land.
"""
import math
import time

import cv2
import numpy as np
from djitellopy import Tello

from src.aruco_utils import load_course, make_detector, detect_markers, draw_markers
from src.killswitch import install_killswitch

TARGET_ID = 3               # ladder, middle pole (used for yaw only)
RING = "ring1"

# SEARCH
CLIMB_SPEED = 20
MAX_HEIGHT_CM = 150         # safety: never climb above this (set just below the ring top)
SEARCH_TIMEOUT_S = 10

# ALIGN (P-controllers, rc values are -100..100)
YAW_GAIN = 1.5              # rc per degree marker 3 is off-centre
LR_GAIN = 80                # rc per metre the ring centre is left/right of us
UD_GAIN = 80                # rc per metre the ring centre is above/below us
MAX_RC = 25
BEARING_TOL_DEG = 3
RING_TOL_M = 0.05
ALIGNED_FRAMES = 10         # must be inside the tolerances this many frames in a row
ALIGN_TIMEOUT_S = 20

# THROUGH
PASS_MARGIN_CM = 50         # how far past the ring to fly (must be less than ring -> ladder!)
FLY_THROUGH = True          # False = only align, then hover and land (for testing ALIGN)

LOST_TIMEOUT_S = 1.0        # neither ring nor marker 3 seen this long -> land
HOVER_TIME_S = 3


def clip(value):
    return int(np.clip(value, -MAX_RC, MAX_RC))


def bearing_deg(m):
    """Horizontal angle to a marker. > 0: marker is to the right -> yaw right."""
    x, _, z = m["tvec"]
    return math.degrees(math.atan2(x, z))


def ring_centre(found, markers):
    """Ring centre in the camera frame [right, down, forward] (m), averaged over
    the visible ring markers. None if no ring marker is visible."""
    centres = []
    for m in found:
        if m["obstacle"] != RING:
            continue
        _, left, up = markers[m["id"]]["offset"]
        # marker = centre + offset  ->  centre = marker - offset (left = -right, up = -down)
        centres.append(m["tvec"] + np.array([left, up, 0.0]))
    return np.mean(centres, axis=0) if centres else None


def get_frame(frame_reader):
    return cv2.cvtColor(frame_reader.frame, cv2.COLOR_RGB2BGR)


def main():
    config, markers = load_course()
    detector = make_detector(config["dictionary"])

    tello = Tello()
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")
    install_killswitch(tello)
    tello.streamon()
    frame_reader = tello.get_frame_read()
    time.sleep(1)

    try:
        tello.takeoff()
        state, state_start = "SEARCH", time.time()
        aligned_count = 0
        last_seen = time.time()

        while True:
            now = time.time()
            frame = get_frame(frame_reader)
            found = detect_markers(frame, detector, markers)
            target = next((m for m in found if m["id"] == TARGET_ID), None)
            ring = ring_centre(found, markers)
            height = tello.get_height()

            # Corrections from whatever we can see
            lr = fb = ud = yaw = 0
            if target is not None:
                yaw = clip(YAW_GAIN * bearing_deg(target))
            if ring is not None:
                lr = clip(LR_GAIN * ring[0])     # ring centre to the right -> move right
                ud = clip(-UD_GAIN * ring[1])    # ring centre below (y down > 0) -> move down
            if target is not None or ring is not None:
                last_seen = now
            lost_too_long = now - last_seen > LOST_TIMEOUT_S

            if state == "SEARCH":
                lr, ud, yaw = 0, CLIMB_SPEED, 0
                if ring is not None:
                    print(f"Ring found at {height} cm -> ALIGN")
                    state, state_start = "ALIGN", now
                elif height >= MAX_HEIGHT_CM or now - state_start > SEARCH_TIMEOUT_S:
                    print("Ring not found -> landing")
                    break

            elif state == "ALIGN":
                if lost_too_long:
                    print("Lost ring and marker 3 while aligning -> landing")
                    break
                yaw_ok = target is None or abs(bearing_deg(target)) < BEARING_TOL_DEG
                ring_ok = (ring is not None and abs(ring[0]) < RING_TOL_M
                           and abs(ring[1]) < RING_TOL_M)
                aligned_count = aligned_count + 1 if (yaw_ok and ring_ok) else 0
                if aligned_count >= ALIGNED_FRAMES:
                    print(f"Aligned at {height} cm, ring {ring[2]:.2f} m ahead")
                    tello.send_rc_control(0, 0, 0, 0)
                    if FLY_THROUGH:
                        distance_cm = int(np.clip(ring[2] * 100 + PASS_MARGIN_CM, 20, 500))
                        print(f"THROUGH: move_forward({distance_cm})")
                        tello.move_forward(distance_cm)   # blocks until done
                    lr = fb = ud = yaw = 0
                    state, state_start = "HOVER", time.time()
                elif now - state_start > ALIGN_TIMEOUT_S:
                    print("Could not align in time -> landing")
                    break

            elif state == "HOVER":
                lr = fb = ud = yaw = 0
                if now - state_start > HOVER_TIME_S:
                    break

            # Safety: never climb above MAX_HEIGHT_CM, whatever the markers say
            if height >= MAX_HEIGHT_CM:
                ud = min(ud, 0)

            tello.send_rc_control(lr, fb, ud, yaw)

            draw_markers(frame, found)
            info = f"{state} h={height}cm rc=({lr},{fb},{ud},{yaw})"
            if target is not None:
                info += f" b={bearing_deg(target):.1f}"
            if ring is not None:
                info += f" ring=({ring[0]:.2f},{ring[1]:.2f})"
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("tello", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("q pressed -> landing")
                break

        tello.send_rc_control(0, 0, 0, 0)
        tello.land()

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