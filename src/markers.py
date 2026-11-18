"""Find ArUco markers in an image and work out where the drone is in the course."""
from pathlib import Path

import cv2
import numpy as np
import yaml

# The map lies in config/ next to src/, so it is found no matter where you run from.
COURSE_FILE = Path(__file__).parent.parent / "config" / "course_map.yaml"

# Approximate Tello camera (960x720 image).
CAMERA_MATRIX = np.array([[921.0, 0.0, 480.0],
                          [0.0, 921.0, 360.0],
                          [0.0, 0.0, 1.0]])
NO_DISTORTION = np.zeros(5)

# The Tello camera looks a little downwards. Check the value with: python fly.py look
CAMERA_TILT_DEG = 10.0

DETECTOR = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))


def load_course(path=COURSE_FILE):
    """Read the map. Returns (marker_positions, marker_size)."""
    with open(path, encoding="utf-8") as file:
        course = yaml.safe_load(file)

    marker_positions = {}
    for marker_id, position in course["markers"].items():
        marker_positions[int(marker_id)] = np.array(position, dtype=float)
    return marker_positions, course["marker_size"]


def find_markers(frame, marker_positions):
    """Returns {marker_id: 4 corner pixels} for the markers that are in the map.

    Markers that are not in the map (the floor mat) are ignored.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    all_corners, all_ids, _ = DETECTOR.detectMarkers(gray)

    seen = {}
    if all_ids is not None:
        for corners, marker_id in zip(all_corners, all_ids.flatten()):
            if int(marker_id) in marker_positions:
                seen[int(marker_id)] = corners.reshape(4, 2)
    return seen


def camera_to_world(vector, tilt_deg):
    """Turn a vector from the camera frame (right, down, forward) into the world
    frame (x forward, y up, z right) for a camera that looks tilt_deg downwards."""
    tilt = np.radians(tilt_deg)
    right, down, forward = vector
    x = forward * np.cos(tilt) - down * np.sin(tilt)
    y = -forward * np.sin(tilt) - down * np.cos(tilt)
    z = right
    return np.array([x, y, z])


def marker_seen_from_camera(pixels, marker_size):
    """Where the centre of one marker is in the camera frame (right, down, forward)."""
    half = marker_size / 2
    marker_corners = np.array([[-half, half, 0], [half, half, 0],
                               [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    _, _, translation = cv2.solvePnP(marker_corners, pixels.astype(np.float32),
                                     CAMERA_MATRIX, NO_DISTORTION,
                                     flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return translation.flatten()


def locate_drone(seen, marker_positions, marker_size, tilt_deg=CAMERA_TILT_DEG):
    """Drone position [x, y, z] in the world frame. None if no marker is visible.

    The drone never rotates, so we know which way the camera points. Then one
    marker is enough:   drone = marker position - (vector from camera to marker).
    With several markers we take the average.

    (We do NOT let OpenCV work out the camera rotation from the markers. A small
    marker looks almost the same when it is tilted the other way, and that
    mistake moved the drone 30-80 cm up or down.)
    """
    if not seen:
        return None

    positions = []
    for marker_id, pixels in seen.items():
        to_marker = marker_seen_from_camera(pixels, marker_size)
        positions.append(marker_positions[marker_id] - camera_to_world(to_marker, tilt_deg))
    return np.mean(positions, axis=0)


def locate_drone_robust(seen, marker_positions, marker_size, tilt_deg=CAMERA_TILT_DEG):
    """Like locate_drone, but more careful about which markers to believe.

    - Near markers count much more than far ones. A small error in the camera
      tilt moves the height 3-4 cm for every metre of distance to the marker.
    - A marker that disagrees a lot with the others is dropped (wrongly read id).
    """
    if not seen:
        return None

    positions = []
    distances = []
    for marker_id, pixels in seen.items():
        to_marker = marker_seen_from_camera(pixels, marker_size)
        positions.append(marker_positions[marker_id] - camera_to_world(to_marker, tilt_deg))
        distances.append(np.linalg.norm(to_marker))
    positions = np.array(positions)
    distances = np.array(distances)

    if len(positions) >= 3:
        off_from_the_rest = np.linalg.norm(positions - np.median(positions, axis=0), axis=1)
        keep = off_from_the_rest < 0.4
        positions, distances = positions[keep], distances[keep]
        if len(positions) == 0:
            return None

    weights = 1 / distances ** 2
    return np.average(positions, axis=0, weights=weights)


def estimate_camera_tilt(seen, marker_positions, marker_size):
    """Find the camera tilt that makes near and far markers agree on the drone height.

    Only works when markers at different distances are visible (ring + ladder).
    Returns None otherwise. Use it in look mode to set CAMERA_TILT_DEG.
    """
    distances = [marker_positions[marker_id][0] for marker_id in seen]
    if len(seen) < 2 or max(distances) - min(distances) < 0.5:
        return None

    best_tilt, smallest_spread = None, np.inf
    for tilt_deg in np.arange(-5, 25, 0.25):
        heights = [(marker_positions[i] - camera_to_world(
                    marker_seen_from_camera(pixels, marker_size), tilt_deg))[1]
                   for i, pixels in seen.items()]
        if np.std(heights) < smallest_spread:
            best_tilt, smallest_spread = tilt_deg, np.std(heights)
    return best_tilt


def draw_markers(frame, seen):
    for marker_id, pixels in seen.items():
        cv2.polylines(frame, [pixels.astype(int)], True, (0, 255, 0), 2)
        x, y = pixels[0].astype(int)
        cv2.putText(frame, str(marker_id), (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
