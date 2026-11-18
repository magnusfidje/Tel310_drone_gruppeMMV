"""Load the course YAML and detect ArUco markers."""
import cv2
import numpy as np
import yaml

# Approximate Tello camera (960x720). Calibrate later for better distances.
CAMERA_MATRIX = np.array([[921.17, 0.0, 459.90],
                          [0.0, 919.02, 351.24],
                          [0.0, 0.0, 1.0]])
DIST_COEFFS = np.zeros(5)


def load_course(path="Tel310_drone_gruppeMMV/config/course.yaml"):
    """Return (config, markers) where markers = {id: {"obstacle", "role", "offset", "size"}}."""
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    default_size = config.get("marker_size", 0.12)

    markers = {}
    for obstacle_name, obstacle in config["obstacles"].items():
        for marker_id, info in obstacle["markers"].items():
            markers[int(marker_id)] = {
                "obstacle": obstacle_name,
                "role": info["role"],
                "offset": np.array(info["offset"], dtype=float),
                "size": float(info.get("size", default_size)),
            }
    return config, markers


def make_detector(dictionary_name="DICT_4X4_50"):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    return cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())


def detect_markers(frame, detector, markers, marker_size=None):
    """Detect known markers in a BGR frame.

    Returns a list of dicts: id, obstacle, role, corners (4x2 px), center_px,
    rvec, tvec = marker pose in camera frame, tvec = [right, down, forward] (m),
    distance (m). Size comes from the YAML (marker_size is kept for old scripts).
    Markers not in the YAML (e.g. floor markers) are ignored.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return []

    found = []
    for c, marker_id in zip(corners, ids.flatten()):
        marker_id = int(marker_id)
        if marker_id not in markers:
            continue
        pts = c.reshape(4, 2)
        h = markers[marker_id]["size"] / 2
        object_points = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)
        _, rvec, tvec = cv2.solvePnP(object_points, pts, CAMERA_MATRIX, DIST_COEFFS,
                                     flags=cv2.SOLVEPNP_IPPE_SQUARE)
        rvec, tvec = rvec.flatten(), tvec.flatten()
        found.append({
            "id": marker_id,
            "obstacle": markers[marker_id]["obstacle"],
            "role": markers[marker_id]["role"],
            "corners": pts,
            "center_px": pts.mean(axis=0),
            "rvec": rvec,
            "tvec": tvec,
            "distance": float(np.linalg.norm(tvec)),
        })
    return found


def draw_markers(frame, found):
    for m in found:
        cv2.polylines(frame, [m["corners"].astype(int)], True, (0, 255, 0), 2)
        x, y = m["corners"][0].astype(int)
        text = f'{m["id"]} {m["role"]} {m["distance"]:.2f}m'
        cv2.putText(frame, text, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame