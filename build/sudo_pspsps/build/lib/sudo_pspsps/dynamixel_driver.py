#
## dynamixel_driver.py
#
#  Controls the two Dynamixel servo motors that move the cat's head.
#
#  This file is NOT a ROS2 node. It is a plain Python class that wraps
#  the Dynamixel SDK so the rest of the code never has to deal with
#  low-level serial communication directly.
#
#  The two motors are:
#    Pan  (ID 1) — rotates the head left and right
#    Tilt (ID 2) — nods the head up and down
#
#  All angles are in degrees, measured from the calibrated zero position.
#  Positive pan  = right
#  Positive tilt = up
#

from dynamixel_sdk import PortHandler, PacketHandler
import time

# ── Dynamixel XL430 control table addresses ───────────────────────────────────
# These are fixed memory locations on the servo. Writing to them changes
# behaviour; reading from them reports current state.
ADDR_TORQUE_ENABLE   = 64   # 1 = motor holds position, 0 = free-spinning
ADDR_GOAL_POSITION   = 116  # write here to move the motor
ADDR_PRESENT_POSITION = 132  # read here to get current position

TORQUE_ENABLE  = 1
TORQUE_DISABLE = 0

PROTOCOL_VERSION = 2.0  # XL430 uses Dynamixel Protocol 2

# The XL430 encoder has 4096 steps over 360 degrees.
TICKS_PER_DEGREE = 4095.0 / 360.0


class DynamixelDriver:

    def __init__(
        self,
        device_name = "/dev/ttyACM0",  # USB serial port
        baudrate    = 1000000,          # 1 Mbps — must match servo settings
        pan_id      = 1,
        tilt_id     = 2,
    ):
        self.pan_id  = pan_id
        self.tilt_id = tilt_id

        # PortHandler manages the serial connection.
        # PacketHandler builds and parses Dynamixel protocol packets.
        self.port_handler   = PortHandler(device_name)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)

        if not self.port_handler.openPort():
            raise RuntimeError(f"Could not open port {device_name}")

        if not self.port_handler.setBaudRate(baudrate):
            raise RuntimeError(f"Could not set baudrate to {baudrate}")

        print(f"Connected to Dynamixel servos on {device_name}")

        # Zero positions are set by calibrate_zero() at startup.
        # All angle commands are offsets from these values.
        self.pan_zero  = 0
        self.tilt_zero = 0


    # ── Hardware limits ───────────────────────────────────────────────────────
    # These prevent the head from rotating into the cat's own body.
    PAN_MIN  = -60   # degrees left
    PAN_MAX  =  60   # degrees right
    TILT_MIN = -15   # degrees down
    TILT_MAX =  20   # degrees up


    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate_zero(self):
        """
        Read the current motor positions and save them as "zero".
        Call this once at startup while the head is pointing straight forward.
        All future angle commands will be relative to this position.
        """
        self.pan_zero  = self._read_position(self.pan_id)
        self.tilt_zero = self._read_position(self.tilt_id)

        print()
        print("=== Zero calibration complete ===")
        print(f"  Pan  zero tick : {self.pan_zero}")
        print(f"  Tilt zero tick : {self.tilt_zero}")
        print("=================================")
        print()


    # ── Torque control ────────────────────────────────────────────────────────

    def enable(self):
        """Lock both motors so they hold their positions."""
        self._set_torque(self.pan_id,  TORQUE_ENABLE)
        self._set_torque(self.tilt_id, TORQUE_ENABLE)

    def disable(self):
        """Release both motors so they can spin freely (safe for power-off)."""
        self._set_torque(self.pan_id,  TORQUE_DISABLE)
        self._set_torque(self.tilt_id, TORQUE_DISABLE)


    # ── Pan control ───────────────────────────────────────────────────────────

    def set_pan(self, angle):
        """
        Move the pan motor to `angle` degrees from the zero position.
        Automatically clamps to PAN_MIN / PAN_MAX so the head can't
        spin into the body.
        """
        angle    = max(self.PAN_MIN, min(self.PAN_MAX, angle))
        position = int(self.pan_zero + angle * TICKS_PER_DEGREE)
        self._write_position(self.pan_id, position)


    # ── Tilt control ──────────────────────────────────────────────────────────

    def set_tilt(self, angle):
        """
        Move the tilt motor to `angle` degrees from the zero position.
        Automatically clamps to TILT_MIN / TILT_MAX.
        """
        angle    = max(self.TILT_MIN, min(self.TILT_MAX, angle))
        position = int(self.tilt_zero + angle * TICKS_PER_DEGREE)
        self._write_position(self.tilt_id, position)


    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self):
        """Close the serial port. Always call this before the program exits."""
        self.port_handler.closePort()


    # ── Private helpers ───────────────────────────────────────────────────────
    # Students don't need to read these — they are just wrappers around
    # the Dynamixel SDK calls.

    def _set_torque(self, dxl_id, value):
        self.packet_handler.write1ByteTxRx(
            self.port_handler, dxl_id, ADDR_TORQUE_ENABLE, value
        )

    def _read_position(self, dxl_id):
        position, comm_result, _ = self.packet_handler.read4ByteTxRx(
            self.port_handler, dxl_id, ADDR_PRESENT_POSITION
        )
        if comm_result != 0:
            print(f"Warning: could not read position from motor ID {dxl_id}")
        return position

    def _write_position(self, dxl_id, position):
        self.packet_handler.write4ByteTxRx(
            self.port_handler, dxl_id, ADDR_GOAL_POSITION, int(position)
        )