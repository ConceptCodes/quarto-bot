#!/usr/bin/env python3
"""
Safe Servo Calibration Script for Quarto Bot
This script helps you safely find the minimum and maximum positions for each servo joint.
"""

import sys
import time
import logging
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quarto_driver.servo_controller import FeetechSCSController, ServoConfig


class SafeServoCalibrator:
    """Safe servo calibration utility."""

    def __init__(self, port: str = None):
        """Initialize the calibrator."""
        # Read port from environment variable if not provided
        if port is None:
            port = os.environ.get("ROBOT_PORT", "/dev/ttyUSB0")
        self.port = port
        self.servo_controller = FeetechSCSController(port)
        self.calibration_data = {}
        self.current_positions = {}

        # Safety parameters
        self.STEP_SIZE = 5  # Small steps for safety
        self.SLOW_SPEED = 30  # Very slow speed
        self.SAFETY_MARGIN = 50  # Stay away from absolute limits

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def connect(self) -> bool:
        """Connect to servo controller."""
        return self.servo_controller.connect()

    def disconnect(self):
        """Disconnect from servo controller."""
        self.servo_controller.disconnect()

    def get_current_position(self, servo_id: int) -> Optional[int]:
        """Get current position of servo."""
        return self.servo_controller.get_servo_position_by_id(servo_id)

    def move_servo_safe(self, servo_id: int, target_position: int) -> bool:
        """Move servo safely to target position."""
        current_pos = self.get_current_position(servo_id)
        if current_pos is None:
            print(f"Cannot read current position of servo {servo_id}")
            return False

        # Move in small steps for safety
        steps = abs(target_position - current_pos) // self.STEP_SIZE + 1
        step_size = (target_position - current_pos) / steps

        for i in range(steps):
            intermediate_pos = int(current_pos + step_size * (i + 1))
            if not self.servo_controller.move_servo_by_id(
                servo_id, intermediate_pos, self.SLOW_SPEED
            ):
                print(f"Failed to move servo {servo_id}")
                return False
            time.sleep(0.1)  # Small delay between steps

        return True

    def find_servo_center(self, servo_id: int) -> Optional[int]:
        """Find the center position of a servo (usually around 512)."""
        print(f"\n--- Finding center position for servo {servo_id} ---")

        # Try the standard center position first
        center_pos = 512
        if self.move_servo_safe(servo_id, center_pos):
            actual_pos = self.get_current_position(servo_id)
            if actual_pos is not None:
                print(f"Servo {servo_id} center position: {actual_pos}")
                return actual_pos

        return None

    def disable_servo_torque(self, servo_id: int) -> bool:
        """Disable torque for a servo so it can be moved manually."""
        try:
            print(f"Disabling torque for servo {servo_id}...")
            # Write 0 to torque enable register to disable torque
            params = [0x28, 0]  # REG_TORQUE_ENABLE = 0x28, disable = 0
            packet = self.servo_controller._create_packet(
                servo_id, 0x03, params
            )  # INST_WRITE = 0x03

            if self.servo_controller._send_packet(packet):
                time.sleep(0.2)  # Give servo time to disable torque
                print(f"✅ Torque disabled for servo {servo_id}")
                return True
            else:
                print(f"❌ Failed to send disable torque command to servo {servo_id}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to disable torque for servo {servo_id}: {e}")
            return False

    def enable_servo_torque(self, servo_id: int) -> bool:
        """Enable torque for a servo."""
        try:
            print(f"Enabling torque for servo {servo_id}...")
            # Write 1 to torque enable register to enable torque
            params = [0x28, 1]  # REG_TORQUE_ENABLE = 0x28, enable = 1
            packet = self.servo_controller._create_packet(
                servo_id, 0x03, params
            )  # INST_WRITE = 0x03

            if self.servo_controller._send_packet(packet):
                time.sleep(0.2)  # Give servo time to enable torque
                print(f"✅ Torque enabled for servo {servo_id}")
                return True
            else:
                print(f"❌ Failed to send enable torque command to servo {servo_id}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to enable torque for servo {servo_id}: {e}")
            return False

    def manual_calibrate_servo(self, servo_id: int, joint_name: str) -> Optional[int]:
        """Manually calibrate only the home position by disabling torque and letting user move the robot."""
        print(
            f"\n=== Manual Home Calibration for {joint_name} (Servo ID: {servo_id}) ==="
        )

        # Get current position
        current_pos = self.get_current_position(servo_id)
        if current_pos is None:
            print(f"Cannot read servo {servo_id}. Skipping.")
            return None

        print(f"Current position: {current_pos}")

        # Disable torque for manual movement
        print("\n🔧 Preparing for manual calibration...")
        print("Disabling torque - you will be able to move the joint manually...")
        if not self.disable_servo_torque(servo_id):
            print(f"❌ Failed to disable torque for servo {servo_id}")
            print("Cannot proceed with manual calibration.")
            return None

        print("✅ Torque disabled! The joint should now move freely.")
        print("   Try gently moving the joint to confirm it moves freely.")
        print("   If it still feels stiff, check your servo connections.")
        print()

        # Set HOME/CENTER position
        print(f"--- Setting HOME position for {joint_name} ---")
        print("🏠 Move the joint to a SAFE NEUTRAL/HOME position:")
        print("   ➤ This should be a comfortable middle position")
        print("   ➤ The robot should be stable and safe in this position")
        print("   ➤ This will be the default position when the robot starts")
        print("   ➤ Press Enter when the joint is at the desired home position")
        input()

        home_pos = self.get_current_position(servo_id)
        if home_pos is None:
            print("   ❌ Failed to read home position")
            return None

        print(f"   ✅ Home position set: {home_pos}")

        # Re-enable torque and move to home
        print(f"\n--- Homing Test for {joint_name} ---")
        print("Re-enabling torque and moving to home position...")
        if self.enable_servo_torque(servo_id):
            time.sleep(0.5)
            self.move_servo_safe(servo_id, home_pos)
            print(f"✅ {joint_name} returned to home position: {home_pos}")
        else:
            print(
                f"⚠️  Failed to re-enable torque for {joint_name} - you may need to do this manually"
            )

        # Store home position for later use
        self.calibration_data[f"{joint_name}_home"] = home_pos

        print(f"\n{joint_name} home calibration complete:")
        print(f"  Home position: {home_pos}")

        return home_pos

    def interactive_position_finder(self, servo_id: int, joint_name: str):
        """Interactive mode to find positions manually."""
        print(
            f"\n=== Interactive Position Finder for {joint_name} (Servo ID: {servo_id}) ==="
        )
        print("Commands:")
        print("  +<num> or -<num> - Move servo by amount (e.g., +10, -5)")
        print("  <num> - Move servo to absolute position (e.g., 512)")
        print("  pos - Show current position")
        print("  save - Save current position as limit")
        print("  quit - Exit interactive mode")
        print()

        saved_positions = []

        while True:
            try:
                current_pos = self.get_current_position(servo_id)
                if current_pos is None:
                    print("Cannot read servo position")
                    break

                command = input(f"{joint_name} [{current_pos}]> ").strip()

                if not command:
                    continue
                elif command.lower() == "quit":
                    break
                elif command.lower() == "pos":
                    print(f"Current position: {current_pos}")
                elif command.lower() == "save":
                    saved_positions.append(current_pos)
                    print(f"Saved position: {current_pos}")
                    print(f"Saved positions: {saved_positions}")
                elif command.startswith("+") or command.startswith("-"):
                    # Relative movement
                    delta = int(command)
                    new_pos = current_pos + delta
                    new_pos = max(0, min(1023, new_pos))  # Clamp to valid range
                    print(f"Moving to {new_pos}...")
                    self.move_servo_safe(servo_id, new_pos)
                else:
                    # Absolute movement
                    target_pos = int(command)
                    target_pos = max(0, min(1023, target_pos))  # Clamp to valid range
                    print(f"Moving to {target_pos}...")
                    self.move_servo_safe(servo_id, target_pos)

            except ValueError:
                print("Invalid command")
            except KeyboardInterrupt:
                break

        if saved_positions:
            print(f"\nSaved positions for {joint_name}: {saved_positions}")
            if len(saved_positions) >= 2:
                min_pos = min(saved_positions)
                max_pos = max(saved_positions)
                print(f"Suggested limits: min={min_pos}, max={max_pos}")
                return min_pos, max_pos

        return None

    def save_calibration_data(self, filename: str = "servo_calibration.json"):
        """Save calibration data to file."""
        calibration_file = Path(__file__).parent.parent / filename

        try:
            with open(calibration_file, "w") as f:
                json.dump(self.calibration_data, f, indent=2)
            print(f"\nCalibration data saved to: {calibration_file}")
        except Exception as e:
            print(f"Failed to save calibration data: {e}")

    def load_calibration_data(self, filename: str = "servo_calibration.json"):
        """Load existing calibration data."""
        calibration_file = Path(__file__).parent.parent / filename

        if calibration_file.exists():
            try:
                with open(calibration_file, "r") as f:
                    self.calibration_data = json.load(f)
                print(f"Loaded existing calibration data from: {calibration_file}")
                return True
            except Exception as e:
                print(f"Failed to load calibration data: {e}")

        return False

    def print_servo_config_code(self):
        """Print Python code for servo configuration."""
        if not self.calibration_data:
            print("No calibration data available")
            return

        print("\n" + "=" * 50)
        print("SERVO CONFIGURATION CODE")
        print("=" * 50)
        print("# Add this to your servo controller setup:")
        print()
        print("def setup_calibrated_servos(self):")
        print('    """Setup servos with calibrated limits."""')
        print("    self.servos = {")

        for joint_name, data in self.calibration_data.items():
            servo_id = data["servo_id"]
            home_pos = data.get(
                "home_position", (data["min_position"] + data["max_position"]) // 2
            )
            print(f'        "{joint_name}": ServoConfig(')
            print(f"            servo_id={servo_id},")
            print(f"            min_position={data['min_position']},")
            print(f"            max_position={data['max_position']},")
            print(f"            speed=100,")
            print(f'            name="{joint_name.replace("_", " ").title()}"')
            print("        ),")

        print("    }")
        print()
        print("# Home positions for easy reference:")
        for joint_name, data in self.calibration_data.items():
            home_pos = data.get("home_position", "unknown")
            print(f"# {joint_name}: home = {home_pos}")
        print("=" * 50)


def main():
    """Main calibration function."""
    import argparse

    # Load environment variables from .env if present
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # dotenv is optional

    parser = argparse.ArgumentParser(description="Safe servo calibration utility")
    parser.add_argument(
        "--port",
        default=None,
        help="Serial port (default: $ROBOT_PORT or /dev/ttyUSB0)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Use interactive mode (manual control via commands)",
    )
    parser.add_argument(
        "--automatic",
        action="store_true",
        help="Use automatic movement mode (NOT recommended - can damage robot)",
    )
    parser.add_argument(
        "--load", action="store_true", help="Load existing calibration data"
    )

    args = parser.parse_args()

    print("=== SAFE SERVO CALIBRATION UTILITY ===")
    print()
    print("🎯 MANUAL CALIBRATION MODE (Recommended)")
    print("This mode disables servo torque so you can manually move the robot")
    print("to find safe limits without any risk of damage.")
    print()
    print("⚠️  SAFETY WARNINGS:")
    print("1. Make sure your robot is properly supported and cannot fall")
    print("2. Move joints slowly and gently by hand")
    print("3. Don't force joints beyond comfortable limits")
    print("4. Leave safety margins from mechanical limits")
    print("5. If a joint feels stuck, don't force it")
    print()
    print("📋 CALIBRATION MODES:")
    print("  Default: Manual calibration (torque disabled, you move robot)")
    print("  --interactive: Command-based control")
    print("  --automatic: Servo moves itself (NOT RECOMMENDED)")
    print()

    # Get user confirmation
    confirm = input("Do you understand the safety warnings? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Calibration cancelled for safety.")
        return

    calibrator = SafeServoCalibrator(args.port)

    try:
        if not calibrator.connect():
            print("Failed to connect to servo controller!")
            return

        if args.load:
            calibrator.load_calibration_data()

        # Scan for servos
        print("Scanning for servos...")
        found_servos = calibrator.servo_controller.scan_servos(1, 10)
        if not found_servos:
            print("No servos found. Check connections and power.")
            return

        print(f"Found servos with IDs: {found_servos}")

        # Define joint names (you can modify these)
        joint_names = {
            1: "base_rotation",
            2: "shoulder",
            3: "elbow",
            4: "wrist",
            5: "gripper",
        }

        # Calibrate each servo
        for servo_id in found_servos:
            joint_name = joint_names.get(servo_id, f"joint_{servo_id}")

            print(f"\n" + "=" * 60)
            print(f"CALIBRATING {joint_name.upper()} (Servo ID: {servo_id})")
            print("=" * 60)

            if args.interactive:
                result = calibrator.interactive_position_finder(servo_id, joint_name)
            elif args.automatic:
                # Use the old automatic method (not recommended)
                result = calibrator.calibrate_servo_limits(servo_id, joint_name)
            else:
                # Use manual calibration by default (safest)
                result = calibrator.manual_calibrate_servo(servo_id, joint_name)

            if result:
                # Only store home position
                home_pos = result if isinstance(result, int) else result[0]
                calibrator.calibration_data[joint_name] = {
                    "servo_id": servo_id,
                    "home_position": home_pos,
                }

            # Ask if user wants to continue
            if servo_id != found_servos[-1]:  # Not the last servo
                continue_cal = (
                    input(f"\nContinue with next servo? (y/n): ").strip().lower()
                )
                if continue_cal != "y":
                    break

        # Save calibration data
        calibrator.save_calibration_data()

        # Print configuration code
        calibrator.print_servo_config_code()

        # Print summary table
        print("\n" + "=" * 80)
        print("HOME CALIBRATION SUMMARY")
        print("=" * 80)
        print("┌─────────────────┬──────────┬───────────────┐")
        print("│   Joint Name    │ Servo ID │  Home Pos.    │")
        print("├─────────────────┼──────────┼───────────────┤")

        for joint_name, data in calibrator.calibration_data.items():
            servo_id = data["servo_id"]
            home_pos = data["home_position"]
            print(f"│ {joint_name:15s} │    {servo_id:2d}    │  {home_pos:7d}     │")

        print("└─────────────────┴──────────┴───────────────┘")
        print("=" * 80)

        print("\n✅ Home calibration complete!")
        print("You can now use the generated home positions in your code.")

    except KeyboardInterrupt:
        print("\nCalibration interrupted by user")
    except Exception as e:
        print(f"Calibration failed: {e}")
    finally:
        calibrator.disconnect()


if __name__ == "__main__":
    main()
