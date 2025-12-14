"""
DualSense (PS5) controller teleop for the SO-Arm101 via the existing servo driver.

Usage (from repo root):
    uv run python packages/quarto-driver/quarto_driver/gamepad.py --port /dev/ttyUSB0

Control scheme (all values are incremental deltas):
    - Left stick X : base rotation
    - Left stick Y : shoulder
    - Right stick Y: elbow
    - Right stick X: wrist
    - L2 trigger   : open gripper (hold)
    - R2 trigger   : close gripper (hold)
    - Options      : emergency stop (all servos hold current position)
    - Share        : quit

Deadzone and scaling can be tuned via CLI flags.
"""

import argparse
import sys
import time
from typing import Dict

import pygame

from .servo_controller import FeetechSCSController, RobotArmController


class PS5Gamepad:
    """Lightweight wrapper around pygame's joystick API for DualSense."""

    def __init__(self, deadzone: float = 0.1, scale: int = 20):
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad detected. Connect a PS5 controller and try again.")
        self.js = pygame.joystick.Joystick(0)
        self.js.init()
        self.deadzone = deadzone
        self.scale = scale  # servo position delta per full deflection

    def _axis(self, idx: int) -> float:
        val = self.js.get_axis(idx)
        return 0.0 if abs(val) < self.deadzone else val

    def read_deltas(self) -> Dict[str, int]:
        """
        Returns per-servo deltas computed from the current stick/trigger state.
        """
        pygame.event.pump()  # ensure values are fresh
        deltas: Dict[str, int] = {}

        # Axes: DualSense indexes (may differ on other OSes)
        left_x = self._axis(0)
        left_y = self._axis(1)
        right_x = self._axis(2)
        right_y = self._axis(3)
        l2 = (self.js.get_axis(4) + 1) / 2  # 0..1
        r2 = (self.js.get_axis(5) + 1) / 2  # 0..1

        deltas["base_rotation"] = int(left_x * self.scale)
        deltas["shoulder"] = int(-left_y * self.scale)  # invert: up is negative on sticks
        deltas["elbow"] = int(-right_y * self.scale)
        deltas["wrist"] = int(right_x * self.scale)
        deltas["gripper"] = int((r2 - l2) * (self.scale // 2))

        return deltas

    def pressed(self, button_name: str) -> bool:
        mapping = {
            "share": 8,  # Select/Share
            "options": 9,
            "ps": 12,
        }
        idx = mapping.get(button_name)
        if idx is None:
            return False
        return bool(self.js.get_button(idx))


def main():
    parser = argparse.ArgumentParser(description="PS5 controller teleop for SO-Arm101")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial port for Feetech bus")
    parser.add_argument("--baud", type=int, default=1_000_000, help="Baud rate")
    parser.add_argument("--deadzone", type=float, default=0.12, help="Stick deadzone (0-1)")
    parser.add_argument("--scale", type=int, default=25, help="Servo delta per full stick deflection")
    parser.add_argument("--rate", type=float, default=20.0, help="Command rate (Hz)")
    args = parser.parse_args()

    try:
        gamepad = PS5Gamepad(deadzone=args.deadzone, scale=args.scale)
    except RuntimeError as e:
        sys.exit(str(e))

    controller = FeetechSCSController(port=args.port, baudrate=args.baud)
    if not controller.connect():
        sys.exit(f"Failed to open servo bus at {args.port}")

    arm = RobotArmController(controller)
    arm.initialize_positions()
    print("Gamepad ready. Share to quit, Options for emergency stop.")

    dt = 1.0 / args.rate
    try:
        while True:
            if gamepad.pressed("share"):
                print("Share pressed, exiting.")
                break
            if gamepad.pressed("options"):
                print("Emergency stop!")
                arm.emergency_stop()
                time.sleep(0.2)
                continue

            deltas = gamepad.read_deltas()
            for name, delta in deltas.items():
                if delta == 0:
                    continue
                arm.update_position(name, delta)

            time.sleep(dt)
    finally:
        controller.disconnect()
        pygame.quit()


if __name__ == "__main__":
    main()
