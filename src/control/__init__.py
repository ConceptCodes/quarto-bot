"""
Control module for Quarto Bot.
Contains gamepad and servo control functionality.
"""

from .servo_controller import FeetechSCSController, RobotArmController, ServoConfig

__all__ = [
    "FeetechSCSController",
    "RobotArmController",
    "ServoConfig",
]
