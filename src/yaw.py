"""Helpers for flying with the drone turned (yaw is no longer always 0).

Yaw convention (same as the Tello): degrees, positive = turned to the right
(clockwise seen from above). Yaw 0 = nose pointing forward along the course (+x).

    body frame:  [forward, up, right]  - follows the nose of the drone
    world frame: [x, y, z]             - the course map (x forward, y up, z right)

We read the yaw from the Tello IMU and treat it as known. Then the Kalman
filter for [x, y, z] stays linear: we only rotate the commands and the marker
vectors with the yaw before they go into the filter. (No EKF needed.)
"""
import numpy as np

from src.markers import camera_to_world, marker_seen_from_camera

GATE = 3.0      # same gating as src/kalman.py


def body_to_world_matrix(yaw_deg):
    """3x3 matrix that turns a body vector [forward, up, right] into world [x, y, z]."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    return np.array([[c, 0.0, -s],
                     [0.0, 1.0, 0.0],
                     [s, 0.0, c]])


def body_to_world(vector, yaw_deg):
    return body_to_world_matrix(yaw_deg) @ np.asarray(vector, dtype=float)


def world_to_body(vector, yaw_deg):
    return body_to_world_matrix(yaw_deg).T @ np.asarray(vector, dtype=float)


def heading(yaw_deg):
    """Unit vector in the world [x, z] plane that the nose points along."""
    return np.array([np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))])


def wrap_deg(angle):
    """Angle in (-180, 180]."""
    return (angle + 180.0) % 360.0 - 180.0


def start_yaw_tracker(raw_yaw):
    """The Tello yaw jumps from 180 to -180. The tracker keeps counting past it,
    so a full turn to the right ends at 360 and not at 0.

    raw_yaw should be read while the nose points along +x; that becomes yaw 0.
    """
    return {"last_raw": raw_yaw, "yaw": 0.0}


def update_yaw_tracker(tracker, raw_yaw):
    """Feed a new tello.get_yaw() reading. Returns the continuous yaw in degrees."""
    tracker["yaw"] += wrap_deg(raw_yaw - tracker["last_raw"])
    tracker["last_raw"] = raw_yaw
    return tracker["yaw"]


def locate_drone_with_yaw(seen, marker_positions, marker_size, tilt_deg, yaw_deg):
    """Like locate_drone_robust in src/markers.py, but the drone may be turned.

    Only the marker centre (solvePnP translation) is used, not the marker rotation,
    so the map does not need to know which way a marker faces.
    """
    if not seen:
        return None

    positions = []
    distances = []
    for marker_id, pixels in seen.items():
        to_marker = marker_seen_from_camera(pixels, marker_size)
        in_body = camera_to_world(to_marker, tilt_deg)      # as if yaw were 0
        positions.append(marker_positions[marker_id] - body_to_world(in_body, yaw_deg))
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


def marker_noise_in_world(noise_body, yaw_deg):
    """Measurement covariance in the world frame.

    The markers are worst along the camera direction (forward) and best sideways.
    When the drone is turned, 'forward' is no longer x, so the covariance turns too.
    """
    rotation = body_to_world_matrix(yaw_deg)
    return rotation @ np.diag(np.asarray(noise_body) ** 2) @ rotation.T


def update_position(estimate, measured_position, R):
    """Kalman update of a PositionFilter (src/kalman.py) with a full covariance R.

    Same as PositionFilter.update, which only takes one std. dev. per axis.
    Returns False if the measurement was rejected.
    """
    innovation = measured_position - estimate.position
    S = estimate.P + R

    distance = np.sqrt(innovation @ np.linalg.inv(S) @ innovation)
    if distance > GATE:
        return False

    K = estimate.P @ np.linalg.inv(S)
    estimate.position = estimate.position + K @ innovation
    estimate.P = (np.eye(3) - K) @ estimate.P
    return True
