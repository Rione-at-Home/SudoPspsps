#
## dynamixel_driver.py
#
#  Controls the two Dynamixel XM430-W250-T servo motors that move the
#  cat's head.
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
#  DESIGN RULE FOR THIS FILE:
#    Every single packet sent to a motor is checked. If a motor does not
#    reply, or replies with an error, you will see a loud message saying
#    exactly which motor and exactly what went wrong. Silent failure is
#    the reason "I sent a command and nothing moved" is so hard to debug.
#

from dynamixel_sdk import PortHandler, PacketHandler
import time


# ── XM430-W250-T control table ────────────────────────────────────────────────
# Fixed memory locations on the servo. Writing to them changes behaviour;
# reading from them reports current state. The number in the comment is the
# size in bytes — you must use the matching read/write call or the packet
# is malformed and the motor ignores it.
ADDR_MODEL_NUMBER        = 0    # 2 bytes — 1020 for XM430-W250
ADDR_OPERATING_MODE      = 11   # 1 byte  — 3 = position control
ADDR_MIN_POSITION_LIMIT  = 52   # 4 bytes — goal positions below this are refused
ADDR_MAX_POSITION_LIMIT  = 48   # 4 bytes — goal positions above this are refused
ADDR_TORQUE_ENABLE       = 64   # 1 byte  — 1 = motor holds position
ADDR_LED                 = 65   # 1 byte
ADDR_HARDWARE_ERROR      = 70   # 1 byte  — non-zero = latched fault
ADDR_PROFILE_ACCELERATION = 108 # 4 bytes — 0 = unlimited (snappy/violent)
ADDR_PROFILE_VELOCITY    = 112  # 4 bytes — 0 = unlimited (snappy/violent)
ADDR_GOAL_POSITION       = 116  # 4 bytes — write here to move the motor
ADDR_PRESENT_POSITION    = 132  # 4 bytes — read here to get current position
ADDR_PRESENT_TEMPERATURE = 146  # 1 byte  — degrees Celsius

TORQUE_ENABLE  = 1
TORQUE_DISABLE = 0

OPERATING_MODE_POSITION = 3

PROTOCOL_VERSION = 2.0  # XM430 uses Dynamixel Protocol 2

# The XM430 encoder has 4096 steps over 360 degrees.
TICKS_PER_DEGREE = 4096.0 / 360.0

# Baudrates tried, in order, when autodetecting. Factory default for a
# brand-new X-series servo is 57600 — this is the single most common reason
# a freshly swapped motor appears completely dead.
CANDIDATE_BAUDRATES = [1000000, 57600, 115200, 2000000, 3000000, 4000000, 9600]

# Bit meanings of ADDR_HARDWARE_ERROR, so a fault prints in English.
HARDWARE_ERROR_BITS = {
    0: "input voltage error (check your power supply)",
    2: "overheating (motor too hot — let it cool)",
    3: "motor encoder error",
    4: "electrical shock / short circuit",
    5: "overload (the head is jammed, or the load is too heavy)",
}


class DynamixelError(RuntimeError):
    """Raised when a motor cannot be reached or refuses a command."""


class DynamixelDriver:

    # ── Hardware limits ───────────────────────────────────────────────────────
    # These prevent the head from rotating into the cat's own body.
    PAN_MIN  = -60   # degrees left
    PAN_MAX  =  60   # degrees right
    TILT_MIN = -15   # degrees down
    TILT_MAX =  20   # degrees up

    def __init__(
        self,
        device_name = "/dev/ttyUSB0",  # USB serial port
        baudrate    = 1000000,         # 1 Mbps — must match servo settings
        pan_id      = 1,
        tilt_id     = 2,
        autodetect_baudrate = True,    # fall back to scanning if nobody answers
        profile_velocity     = 200,    # ~46 RPM ceiling; 0 = unlimited
        profile_acceleration = 50,     # gentle ramp; 0 = unlimited
        verbose = True,
    ):
        self.pan_id  = pan_id
        self.tilt_id = tilt_id
        self.verbose = verbose

        # PortHandler manages the serial connection.
        # PacketHandler builds and parses Dynamixel protocol packets.
        self.port_handler   = PortHandler(device_name)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)

        if not self.port_handler.openPort():
            raise DynamixelError(
                f"Could not open port {device_name}. Is the U2D2 plugged in, "
                f"and do you have permission? Try: sudo usermod -aG dialout $USER"
            )

        self.baudrate = self._establish_baudrate(baudrate, autodetect_baudrate)

        self._log(f"Connected to Dynamixel servos on {device_name} "
                  f"at {self.baudrate} baud")

        # Zero positions are set by calibrate_zero() at startup.
        # All angle commands are offsets from these values.
        self.pan_zero  = 0
        self.tilt_zero = 0

        # Filled in by _prepare_motor(): the tick range each motor will accept.
        self.limits = {}   # motor id -> (min_tick, max_tick)

        for dxl_id, name in ((pan_id, "pan"), (tilt_id, "tilt")):
            self._prepare_motor(dxl_id, name,
                                profile_velocity, profile_acceleration)


    # ── Startup checks ────────────────────────────────────────────────────────

    def _establish_baudrate(self, preferred, autodetect):
        """
        Set the serial baudrate and confirm at least one motor answers.
        If nobody answers and autodetect is on, scan the usual suspects.
        """
        order = [preferred] + [b for b in CANDIDATE_BAUDRATES if b != preferred]
        if not autodetect:
            order = [preferred]

        for baud in order:
            if not self.port_handler.setBaudRate(baud):
                continue
            found = [i for i in (self.pan_id, self.tilt_id) if self._ping(i)]
            if found:
                if baud != preferred:
                    self._log(f"NOTE: motors answered at {baud} baud, not the "
                              f"requested {preferred}. Pass baudrate={baud}, or "
                              f"change the motors in Dynamixel Wizard 2.0.")
                return baud

        raise DynamixelError(
            f"No motor answered on IDs {self.pan_id} and {self.tilt_id} at any "
            f"baudrate. Check: power is on (the LED flashes once at boot), the "
            f"data cable is seated, and each motor has a unique ID. A new "
            f"XM430 ships as ID 1 at 57600 baud — two new motors on one bus "
            f"are both ID 1 and will collide. Set them up one at a time with "
            f"Dynamixel Wizard 2.0."
        )

    def _ping(self, dxl_id):
        """Return the model number if the motor answers, else None."""
        model, comm, err = self.packet_handler.ping(self.port_handler, dxl_id)
        if comm != 0 or err != 0:
            return None
        return model

    def _prepare_motor(self, dxl_id, name, profile_velocity, profile_acceleration):
        """
        Verify one motor is present, healthy, and configured for position
        control, then cache the tick range it will accept.
        """
        model = self._ping(dxl_id)
        if model is None:
            raise DynamixelError(
                f"{name} motor (ID {dxl_id}) did not answer at "
                f"{self.baudrate} baud. Check its ID and baudrate."
            )
        self._log(f"  {name:>4} motor: ID {dxl_id}, model number {model}")

        # A latched hardware error makes the motor refuse torque until it is
        # rebooted. This is the classic "it worked, then it stopped" case.
        self._check_hardware_error(dxl_id, name)

        # Operating mode must be 3. In extended-position or velocity mode the
        # degree maths below is meaningless. Changing it requires torque off.
        mode = self._read1(dxl_id, ADDR_OPERATING_MODE, "operating mode")
        if mode != OPERATING_MODE_POSITION:
            self._log(f"  {name} motor was in operating mode {mode}; "
                      f"switching to position control.")
            self._write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, "torque")
            self._write1(dxl_id, ADDR_OPERATING_MODE,
                         OPERATING_MODE_POSITION, "operating mode")

        # Cache the motor's own position limits. Goal positions outside this
        # range are silently refused with a Data Range Limit error — this is
        # why a head can move one direction but not back.
        lo = self._read4(dxl_id, ADDR_MIN_POSITION_LIMIT, "min position limit")
        hi = self._read4(dxl_id, ADDR_MAX_POSITION_LIMIT, "max position limit")
        self.limits[dxl_id] = (lo, hi)

        # Motion profile: caps speed and acceleration inside the servo itself,
        # so a large jump in goal position is a smooth sweep, not a snap.
        self._write4(dxl_id, ADDR_PROFILE_VELOCITY,
                     profile_velocity, "profile velocity")
        self._write4(dxl_id, ADDR_PROFILE_ACCELERATION,
                     profile_acceleration, "profile acceleration")

    def _check_hardware_error(self, dxl_id, name):
        status = self._read1(dxl_id, ADDR_HARDWARE_ERROR, "hardware error status")
        if status == 0:
            return
        reasons = [text for bit, text in HARDWARE_ERROR_BITS.items()
                   if status & (1 << bit)]
        raise DynamixelError(
            f"{name} motor (ID {dxl_id}) has a latched hardware error "
            f"(0x{status:02X}): " + "; ".join(reasons) +
            f". Fix the cause, then call driver.reboot() or power-cycle."
        )

    def reboot(self, dxl_id=None):
        """
        Clear a latched hardware error. Motors forget their torque state and
        return to their power-on position limits, so call calibrate_zero()
        and enable() again afterwards.
        """
        targets = [dxl_id] if dxl_id is not None else [self.pan_id, self.tilt_id]
        for i in targets:
            self.packet_handler.reboot(self.port_handler, i)
            self._log(f"Rebooted motor ID {i}")
        time.sleep(0.5)


    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate_zero(self):
        """
        Read the current motor positions and save them as "zero".
        Call this once at startup while the head is pointing straight forward.
        All future angle commands will be relative to this position.

        This also checks that the full travel range actually fits inside each
        motor's position limits, and warns you at startup rather than letting
        you discover it when the head refuses to come back.
        """
        self.pan_zero  = self._read_position(self.pan_id)
        self.tilt_zero = self._read_position(self.tilt_id)

        print()
        print("=== Zero calibration complete ===")
        print(f"  Pan  zero tick : {self.pan_zero}")
        print(f"  Tilt zero tick : {self.tilt_zero}")

        self._check_travel(self.pan_id,  "Pan",  self.pan_zero,
                           self.PAN_MIN,  self.PAN_MAX)
        self._check_travel(self.tilt_id, "Tilt", self.tilt_zero,
                           self.TILT_MIN, self.TILT_MAX)
        print("=================================")
        print()

    def _check_travel(self, dxl_id, name, zero, deg_min, deg_max):
        lo, hi = self.limits[dxl_id]
        need_lo = int(zero + deg_min * TICKS_PER_DEGREE)
        need_hi = int(zero + deg_max * TICKS_PER_DEGREE)
        if need_lo < lo or need_hi > hi:
            print(f"  !! {name}: travel {deg_min}°..{deg_max}° needs ticks "
                  f"{need_lo}..{need_hi}, but the motor only accepts {lo}..{hi}.")
            print(f"  !! Commands past the edge will be CLAMPED, so the head "
                  f"will not reach one side. Physically rotate the horn so the "
                  f"head points forward nearer the middle of the range "
                  f"(~{(lo + hi) // 2}) and calibrate again.")


    # ── Torque control ────────────────────────────────────────────────────────

    def enable(self):
        """Lock both motors so they hold their positions."""
        self._set_torque(self.pan_id,  TORQUE_ENABLE)
        self._set_torque(self.tilt_id, TORQUE_ENABLE)

    def disable(self):
        """Release both motors so they can spin freely (safe for power-off)."""
        self._set_torque(self.pan_id,  TORQUE_DISABLE)
        self._set_torque(self.tilt_id, TORQUE_DISABLE)


    # ── Pan / tilt control ────────────────────────────────────────────────────

    def set_pan(self, angle):
        """
        Move the pan motor to `angle` degrees from the zero position.
        Clamps to PAN_MIN / PAN_MAX so the head can't spin into the body,
        and again to the motor's own tick limits so the command is never
        silently refused.
        """
        self._set_angle(self.pan_id, "pan", angle, self.pan_zero,
                        self.PAN_MIN, self.PAN_MAX)

    def set_tilt(self, angle):
        """
        Move the tilt motor to `angle` degrees from the zero position.
        Clamps to TILT_MIN / TILT_MAX.
        """
        self._set_angle(self.tilt_id, "tilt", angle, self.tilt_zero,
                        self.TILT_MIN, self.TILT_MAX)

    def _set_angle(self, dxl_id, name, angle, zero, deg_min, deg_max):
        angle    = max(deg_min, min(deg_max, float(angle)))
        position = int(round(zero + angle * TICKS_PER_DEGREE))

        lo, hi = self.limits[dxl_id]
        if position < lo or position > hi:
            position = max(lo, min(hi, position))

        self._write4(dxl_id, ADDR_GOAL_POSITION, position, f"{name} goal position")


    # ── Feedback ──────────────────────────────────────────────────────────────

    def get_pan(self):
        """Current pan angle in degrees, relative to zero."""
        return (self._read_position(self.pan_id) - self.pan_zero) / TICKS_PER_DEGREE

    def get_tilt(self):
        """Current tilt angle in degrees, relative to zero."""
        return (self._read_position(self.tilt_id) - self.tilt_zero) / TICKS_PER_DEGREE

    def get_temperatures(self):
        """Current temperature of each motor, in degrees Celsius."""
        return {
            "pan":  self._read1(self.pan_id,  ADDR_PRESENT_TEMPERATURE, "temperature"),
            "tilt": self._read1(self.tilt_id, ADDR_PRESENT_TEMPERATURE, "temperature"),
        }


    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self):
        """Close the serial port. Always call this before the program exits."""
        try:
            self.port_handler.closePort()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.disable()
        finally:
            self.close()


    # ── Private helpers ───────────────────────────────────────────────────────
    # Wrappers around the Dynamixel SDK calls. Unlike the SDK, these raise
    # on failure instead of returning a status code nobody reads.

    def _log(self, message):
        if self.verbose:
            print(message)

    def _check(self, comm, err, dxl_id, what):
        if comm != 0:
            raise DynamixelError(
                f"Motor ID {dxl_id}: {what} failed to send/receive — "
                f"{self.packet_handler.getTxRxResult(comm)}"
            )
        if err != 0:
            raise DynamixelError(
                f"Motor ID {dxl_id}: {what} was rejected — "
                f"{self.packet_handler.getRxPacketError(err)}"
            )

    @staticmethod
    def _to_signed_32(value):
        """The SDK returns raw unsigned words; positions can be negative."""
        return value - 4294967296 if value > 2147483647 else value

    def _set_torque(self, dxl_id, value):
        self._write1(dxl_id, ADDR_TORQUE_ENABLE, value, "torque enable")

    def _read_position(self, dxl_id):
        return self._read4(dxl_id, ADDR_PRESENT_POSITION, "present position")

    def _read1(self, dxl_id, address, what):
        value, comm, err = self.packet_handler.read1ByteTxRx(
            self.port_handler, dxl_id, address
        )
        self._check(comm, err, dxl_id, f"read {what}")
        return value

    def _read4(self, dxl_id, address, what):
        value, comm, err = self.packet_handler.read4ByteTxRx(
            self.port_handler, dxl_id, address
        )
        self._check(comm, err, dxl_id, f"read {what}")
        return self._to_signed_32(value)

    def _write1(self, dxl_id, address, value, what):
        comm, err = self.packet_handler.write1ByteTxRx(
            self.port_handler, dxl_id, address, int(value)
        )
        self._check(comm, err, dxl_id, f"write {what}")

    def _write4(self, dxl_id, address, value, what):
        comm, err = self.packet_handler.write4ByteTxRx(
            self.port_handler, dxl_id, address, int(value)
        )
        self._check(comm, err, dxl_id, f"write {what}")


# ── Standalone check ──────────────────────────────────────────────────────────
# Run `python3 dynamixel_driver.py` to confirm the bus is healthy without
# touching ROS2 at all.

if __name__ == "__main__":
    with DynamixelDriver() as driver:
        driver.calibrate_zero()
        print("Temperatures:", driver.get_temperatures())
        driver.enable()
        for angle in (0, 20, 0, -20, 0):
            print(f"Pan -> {angle:+}°")
            driver.set_pan(angle)
            time.sleep(1.5)
            print(f"   reported: {driver.get_pan():+.1f}°")