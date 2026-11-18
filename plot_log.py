"""Plot a flight log.

    python plot_log.py                      # the newest log in logs/
    python plot_log.py logs/flight_xxx.csv  # a specific log
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.markers import load_course

LOG_FOLDER = Path(__file__).parent / "logs"


def newest_log():
    return sorted(LOG_FOLDER.glob("flight_*.csv"))[-1]


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else newest_log()
    log = pd.read_csv(path)
    marker_positions, _ = load_course()

    figure, (side_view, over_time) = plt.subplots(2, 1, figsize=(9, 8))
    figure.suptitle(path.name)

    # Top: the course seen from the side (x forward, y height)
    for marker_id, (x, y, z) in marker_positions.items():
        side_view.plot(x, y, "ks")
        side_view.annotate(str(marker_id), (x, y), textcoords="offset points", xytext=(6, -3))
    side_view.plot(log["marker_x"], log["marker_y"], ".", color="orange", label="markers alone")
    side_view.plot(log["x"], log["y"], "-", color="tab:blue", label="filter")
    side_view.set_xlabel("x forward (m)")
    side_view.set_ylabel("y height (m)")
    side_view.axis("equal")
    side_view.grid(True)
    side_view.legend()

    # Bottom: height and sideways over time
    over_time.plot(log["time"], log["y"], color="tab:blue", label="height (filter)")
    over_time.plot(log["time"], log["marker_y"], ".", color="orange", label="height (markers alone)")
    over_time.plot(log["time"], log["tello_height"], color="gray", label="height (Tello)")
    if "tello_tof" in log:
        over_time.plot(log["time"], log["tello_tof"], ":", color="black", label="height (Tello ToF)")
    over_time.plot(log["time"], log["z"], color="tab:green", label="sideways z (filter)")
    over_time.set_xlabel("time (s)")
    over_time.set_ylabel("metres")
    over_time.grid(True)
    over_time.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
