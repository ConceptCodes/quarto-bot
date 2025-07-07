"""
Feetech SCS servo controller module for Quarto Bot.
This module provides direct serial communication with Feetech SCS servos.
"""

import serial
import time
import threading
from typing import Dict, Optional, List
from dataclasses import dataclass
import logging


@dataclass
class ServoConfig:
    """Configuration for a Feetech SCS servo motor."""

    servo_id: int
    min_position: int = 0
    max_position: int = 1023
    speed: int = 100
    acceleration: int = 50
    name: str = ""


class FeetechSCSController:
    """Controller for Feetech SCS servo motors using direct serial communication."""

    # SCS Protocol Constants
    HEADER = 0xFF
    INST_PING = 0x01
    INST_READ = 0x02
    INST_WRITE = 0x03
    INST_REG_WRITE = 0x04
    INST_ACTION = 0x05
    INST_SYNC_WRITE = 0x83

    # Register addresses for SCS servos
    REG_GOAL_POSITION = 0x2A
    REG_GOAL_SPEED = 0x2E
    REG_PRESENT_POSITION = 0x38
    REG_MOVING = 0x42
    REG_TORQUE_ENABLE = 0x28
    REG_LOCK = 0x30

    def __init__(
        self, port: str = "/dev/ttyUSB0", baudrate: int = 1000000, timeout: float = 0.1
    ):
        """
        Initialize the Feetech SCS controller.

        Args:
            port: Serial port for communication
            baudrate: Communication baudrate
            timeout: Serial timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
        self.servos: Dict[str, ServoConfig] = {}
        self.is_connected = False
        self.lock = threading.Lock()

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def add_servo(self, name: str, servo_config: ServoConfig):
        """Add a servo configuration."""
        self.servos[name] = servo_config
        self.logger.info(f"Added servo: {name} (ID: {servo_config.servo_id})")

    def remove_servo(self, name: str):
        """Remove a servo configuration."""
        if name in self.servos:
            del self.servos[name]
            self.logger.info(f"Removed servo: {name}")

    def list_servos(self) -> List[str]:
        """Get list of configured servo names."""
        return list(self.servos.keys())

    def get_servo_config(self, name: str) -> Optional[ServoConfig]:
        """Get servo configuration by name."""
        return self.servos.get(name)

    def _calculate_checksum(self, packet: List[int]) -> int:
        """Calculate checksum for SCS protocol packet."""
        return ~sum(packet[2:]) & 0xFF

    def _create_packet(
        self, servo_id: int, instruction: int, params: List[int]
    ) -> bytes:
        """Create a SCS protocol packet."""
        length = len(params) + 2
        packet = [self.HEADER, self.HEADER, servo_id, length, instruction] + params
        checksum = self._calculate_checksum(packet)
        packet.append(checksum)
        return bytes(packet)

    def _send_packet(self, packet: bytes) -> bool:
        """Send a packet and return success status."""
        try:
            with self.lock:
                if self.serial_conn and self.serial_conn.is_open:
                    self.serial_conn.write(packet)
                    self.serial_conn.flush()
                    return True
        except Exception as e:
            self.logger.error(f"Failed to send packet: {e}")
        return False

    def _read_response(self, expected_length: int = 6) -> Optional[List[int]]:
        """Read response packet from servo."""
        try:
            with self.lock:
                if self.serial_conn and self.serial_conn.is_open:
                    # Wait for header
                    start_time = time.time()
                    while time.time() - start_time < self.timeout:
                        if self.serial_conn.in_waiting >= 2:
                            header = self.serial_conn.read(2)
                            if header == bytes([self.HEADER, self.HEADER]):
                                break
                    else:
                        return None

                    # Read rest of packet
                    remaining = self.serial_conn.read(expected_length - 2)
                    if len(remaining) == expected_length - 2:
                        packet = list(header + remaining)
                        return packet
        except Exception as e:
            self.logger.error(f"Failed to read response: {e}")
        return None

    def connect(self) -> bool:
        """
        Connect to the Feetech servo controller.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
            )

            if self.serial_conn.is_open:
                self.is_connected = True
                self.logger.info(f"Connected to Feetech SCS controller on {self.port}")

                # Test connection by scanning for servos
                connected_servos = self.scan_servos()
                if connected_servos:
                    self.logger.info(f"Found servos with IDs: {connected_servos}")
                else:
                    self.logger.warning(
                        "No servo response - check connections and power"
                    )

                return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Feetech controller: {e}")
        return False

    def disconnect(self):
        """Disconnect from the servo controller."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.is_connected = False
        self.logger.info("Disconnected from Feetech controller")

    def ping_servo(self, servo_id: int) -> bool:
        """Ping a servo to check if it's responding."""
        packet = self._create_packet(servo_id, self.INST_PING, [])
        if self._send_packet(packet):
            response = self._read_response(6)
            return response is not None
        return False

    def scan_servos(self, start_id: int = 1, end_id: int = 10) -> List[int]:
        """
        Scan for connected servos in the given ID range.

        Args:
            start_id: Start of ID range to scan
            end_id: End of ID range to scan

        Returns:
            List of responding servo IDs
        """
        responding_servos = []
        for servo_id in range(start_id, end_id + 1):
            if self.ping_servo(servo_id):
                responding_servos.append(servo_id)
                time.sleep(0.01)  # Small delay between pings
        return responding_servos

    def enable_torque(self, servo_name: str, enable: bool = True) -> bool:
        """
        Enable or disable torque for a servo.

        Args:
            servo_name: Name of the servo
            enable: True to enable torque, False to disable

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected or servo_name not in self.servos:
            return False

        servo_config = self.servos[servo_name]

        try:
            params = [self.REG_TORQUE_ENABLE, 1 if enable else 0]
            packet = self._create_packet(servo_config.servo_id, self.INST_WRITE, params)
            return self._send_packet(packet)
        except Exception as e:
            self.logger.error(f"Failed to set torque for servo {servo_name}: {e}")
            return False

    def move_servo(
        self, servo_name: str, position: int, speed: Optional[int] = None
    ) -> bool:
        """
        Move a servo to a specific position.

        Args:
            servo_name: Name of the servo to move
            position: Target position (0-1023)
            speed: Movement speed (optional)

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected or servo_name not in self.servos:
            return False

        servo_config = self.servos[servo_name]

        # Clamp position to servo limits
        position = max(
            servo_config.min_position, min(servo_config.max_position, position)
        )

        # Use provided speed or default from config
        move_speed = speed if speed is not None else servo_config.speed

        try:
            # Create position and speed parameters (little-endian 16-bit)
            pos_low = position & 0xFF
            pos_high = (position >> 8) & 0xFF
            speed_low = move_speed & 0xFF
            speed_high = (move_speed >> 8) & 0xFF

            # Write goal position and speed
            params = [self.REG_GOAL_POSITION, pos_low, pos_high, speed_low, speed_high]
            packet = self._create_packet(servo_config.servo_id, self.INST_WRITE, params)

            return self._send_packet(packet)

        except Exception as e:
            self.logger.error(f"Failed to move servo {servo_name}: {e}")
            return False

    def move_servo_by_id(self, servo_id: int, position: int, speed: int = 100) -> bool:
        """
        Move a servo by ID directly (without needing to be in servo config).

        Args:
            servo_id: ID of the servo to move
            position: Target position (0-1023)
            speed: Movement speed

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected:
            return False

        try:
            # Clamp position
            position = max(0, min(1023, position))

            # Create position and speed parameters (little-endian 16-bit)
            pos_low = position & 0xFF
            pos_high = (position >> 8) & 0xFF
            speed_low = speed & 0xFF
            speed_high = (speed >> 8) & 0xFF

            # Write goal position and speed
            params = [self.REG_GOAL_POSITION, pos_low, pos_high, speed_low, speed_high]
            packet = self._create_packet(servo_id, self.INST_WRITE, params)

            return self._send_packet(packet)

        except Exception as e:
            self.logger.error(f"Failed to move servo ID {servo_id}: {e}")
            return False

    def get_servo_position(self, servo_name: str) -> Optional[int]:
        """
        Get current position of a servo.

        Args:
            servo_name: Name of the servo

        Returns:
            Current position or None if failed
        """
        if not self.is_connected or servo_name not in self.servos:
            return None

        servo_config = self.servos[servo_name]
        return self.get_servo_position_by_id(servo_config.servo_id)

    def get_servo_position_by_id(self, servo_id: int) -> Optional[int]:
        """
        Get current position of a servo by ID.

        Args:
            servo_id: ID of the servo

        Returns:
            Current position or None if failed
        """
        if not self.is_connected:
            return None

        try:
            # Request present position (2 bytes)
            params = [self.REG_PRESENT_POSITION, 2]
            packet = self._create_packet(servo_id, self.INST_READ, params)

            if self._send_packet(packet):
                response = self._read_response(
                    8
                )  # Header + ID + Length + Error + Data(2) + Checksum
                if response and len(response) >= 8:
                    # Extract position from response (little-endian)
                    position = response[5] | (response[6] << 8)
                    return position

        except Exception as e:
            self.logger.error(f"Failed to read servo ID {servo_id} position: {e}")
        return None

    def is_servo_moving(self, servo_name: str) -> Optional[bool]:
        """
        Check if a servo is currently moving.

        Args:
            servo_name: Name of the servo

        Returns:
            True if moving, False if stopped, None if failed
        """
        if not self.is_connected or servo_name not in self.servos:
            return None

        servo_config = self.servos[servo_name]

        try:
            # Request moving status (1 byte)
            params = [self.REG_MOVING, 1]
            packet = self._create_packet(servo_config.servo_id, self.INST_READ, params)

            if self._send_packet(packet):
                response = self._read_response(
                    7
                )  # Header + ID + Length + Error + Data(1) + Checksum
                if response and len(response) >= 7:
                    return bool(response[5])

        except Exception as e:
            self.logger.error(f"Failed to read servo {servo_name} moving status: {e}")
        return None

    def sync_write_positions(
        self, servo_positions: Dict[str, int], speed: int = 100
    ) -> bool:
        """
        Move multiple servos simultaneously using sync write.

        Args:
            servo_positions: Dictionary mapping servo names to target positions
            speed: Movement speed for all servos

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected:
            return False

        try:
            # Prepare sync write data
            servo_data = []
            for servo_name, position in servo_positions.items():
                if servo_name not in self.servos:
                    continue

                servo_config = self.servos[servo_name]

                # Clamp position
                position = max(
                    servo_config.min_position, min(servo_config.max_position, position)
                )

                # Pack servo ID and position data
                pos_low = position & 0xFF
                pos_high = (position >> 8) & 0xFF
                speed_low = speed & 0xFF
                speed_high = (speed >> 8) & 0xFF

                servo_data.extend(
                    [servo_config.servo_id, pos_low, pos_high, speed_low, speed_high]
                )

            if not servo_data:
                return False

            # Create sync write packet
            params = [
                self.REG_GOAL_POSITION,
                4,
            ] + servo_data  # 4 bytes per servo (pos + speed)
            packet = self._create_packet(
                0xFE, self.INST_SYNC_WRITE, params
            )  # 0xFE = broadcast ID

            return self._send_packet(packet)

        except Exception as e:
            self.logger.error(f"Failed to sync write positions: {e}")
            return False

    def stop_all_servos(self) -> bool:
        """
        Stop all configured servos by setting their goal position to current position.

        Returns:
            True if successful, False otherwise
        """
        current_positions = {}

        # Read current positions
        for servo_name in self.servos.keys():
            position = self.get_servo_position(servo_name)
            if position is not None:
                current_positions[servo_name] = position

        # Set goal positions to current positions
        if current_positions:
            return self.sync_write_positions(current_positions, speed=0)

        return False


class RobotArmController:
    """High-level robot arm controller using Feetech servos."""

    def __init__(self, servo_controller: FeetechSCSController):
        """
        Initialize robot arm controller.

        Args:
            servo_controller: Instance of FeetechSCSController
        """
        self.servo_controller = servo_controller
        self.current_positions = {}
        self.logger = logging.getLogger(__name__)

        # Setup default arm configuration
        self.setup_default_arm()

    def setup_default_arm(self):
        """Setup default robot arm servo configuration."""
        servos = {
            "base_rotation": ServoConfig(
                servo_id=1,
                min_position=100,
                max_position=900,
                speed=150,
                name="Base Rotation",
            ),
            "shoulder": ServoConfig(
                servo_id=2,
                min_position=200,
                max_position=800,
                speed=100,
                name="Shoulder",
            ),
            "elbow": ServoConfig(
                servo_id=3, min_position=150, max_position=850, speed=100, name="Elbow"
            ),
            "wrist": ServoConfig(
                servo_id=4, min_position=100, max_position=900, speed=120, name="Wrist"
            ),
            "gripper": ServoConfig(
                servo_id=5, min_position=300, max_position=700, speed=80, name="Gripper"
            ),
        }

        for name, config in servos.items():
            self.servo_controller.add_servo(name, config)

    def initialize_positions(self):
        """Initialize current servo positions."""
        for servo_name in self.servo_controller.list_servos():
            position = self.servo_controller.get_servo_position(servo_name)
            if position is not None:
                self.current_positions[servo_name] = position
                self.logger.info(f"{servo_name} current position: {position}")
            else:
                # Use middle position as default
                servo_config = self.servo_controller.get_servo_config(servo_name)
                if servo_config:
                    middle_pos = (
                        servo_config.min_position + servo_config.max_position
                    ) // 2
                    self.current_positions[servo_name] = middle_pos
                    self.servo_controller.move_servo(servo_name, middle_pos)
                    self.logger.info(
                        f"{servo_name} initialized to middle position: {middle_pos}"
                    )

    def move_to_home_position(self):
        """Move arm to home position."""
        home_positions = {
            "base_rotation": 512,
            "shoulder": 400,
            "elbow": 500,
            "wrist": 512,
            "gripper": 400,
        }

        self.servo_controller.sync_write_positions(home_positions)
        self.current_positions.update(home_positions)
        self.logger.info("Moving to home position")

    def update_position(self, servo_name: str, delta: int) -> bool:
        """
        Update servo position by a delta amount.

        Args:
            servo_name: Name of the servo
            delta: Amount to change position by

        Returns:
            True if successful, False otherwise
        """
        if servo_name not in self.current_positions:
            return False

        current_pos = self.current_positions[servo_name]
        new_pos = int(current_pos + delta)

        if self.servo_controller.move_servo(servo_name, new_pos):
            self.current_positions[servo_name] = new_pos
            return True

        return False

    def get_positions(self) -> Dict[str, int]:
        """Get current positions of all servos."""
        return self.current_positions.copy()

    def emergency_stop(self):
        """Emergency stop - stop all servo movement."""
        self.servo_controller.stop_all_servos()
        self.logger.warning("Emergency stop activated")
