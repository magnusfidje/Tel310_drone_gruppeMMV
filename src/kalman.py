"""Kalman filter for the drone position [x, y, z].

The drone never rotates, so the motion model is linear and the EKF becomes an
ordinary Kalman filter. Same two steps as always:
    predict - move the estimate with the speed we commanded
    update  - correct it with the position from the markers, or the Tello height

A measurement that is far from what the filter expects is thrown away (gating).
While measurements are thrown away the uncertainty keeps growing, so if they
keep saying the same thing the filter starts to believe them again.
"""
import numpy as np

# We only know roughly how fast the drone really moves for a given command,
# so the uncertainty grows quickly when we fly without measurements:
# after 1 second the position is uncertain by about 0.15 m.
DRIFT = 0.15 ** 2       # m^2 per second
GATE = 3.0              # reject measurements more than this many std. dev. away


class PositionFilter:
    def __init__(self, start_position, start_uncertainty=0.5):
        self.position = np.array(start_position, dtype=float)
        self.P = np.eye(3) * start_uncertainty ** 2

    def predict(self, velocity, dt):
        self.position = self.position + np.array(velocity) * dt
        self.P = self.P + np.eye(3) * DRIFT * dt

    def update(self, measured_position, noise_std):
        """Kalman update with a measured [x, y, z]. Returns False if it was rejected."""
        R = np.eye(3) * noise_std ** 2
        innovation = measured_position - self.position
        S = self.P + R

        distance = np.sqrt(innovation @ np.linalg.inv(S) @ innovation)
        if distance > GATE:
            return False

        K = self.P @ np.linalg.inv(S)
        self.position = self.position + K @ innovation
        self.P = (np.eye(3) - K) @ self.P
        return True

    def update_height(self, measured_height, noise_std):
        """Kalman update with only a height (y). Returns False if it was rejected.

        This is where a sudden jump is caught, e.g. when the height sensor
        sees the ring instead of the floor.
        """
        H = np.array([0.0, 1.0, 0.0])           # the measurement is the y in [x, y, z]
        innovation = measured_height - self.position[1]
        S = H @ self.P @ H + noise_std ** 2

        if abs(innovation) > GATE * np.sqrt(S):
            return False

        K = self.P @ H / S
        self.position = self.position + K * innovation
        self.P = self.P - np.outer(K, H @ self.P)
        return True
