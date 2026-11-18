"""Kill switch: Ctrl+C stops the motors immediately (the drone drops!).

In a flight script:   install_killswitch(tello)
Backup in a 2nd terminal:   python killswitch.py   -> press Enter to kill
"""
import signal
import socket
import time

TELLO_ADDRESS = ("192.168.10.1", 8889)


def emergency_stop(tello=None):
    print("\n!!! EMERGENCY STOP !!!", flush=True)
    # Raw UDP so it works even if djitellopy is stuck waiting on a command
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for _ in range(3):
            sock.sendto(b"emergency", TELLO_ADDRESS)
            time.sleep(0.02)
    if tello is not None:
        try:
            tello.emergency()
        except Exception:
            pass


def install_killswitch(tello):
    def handler(signum, frame):
        emergency_stop(tello)
        raise KeyboardInterrupt   # stop the flight script
    signal.signal(signal.SIGINT, handler)


if __name__ == "__main__":
    input("Kill switch armed. Press Enter to KILL the Tello... ")
    emergency_stop()