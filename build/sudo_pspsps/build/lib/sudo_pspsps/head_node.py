#
## head_node.py   ── ACT (movement)
#
#  This is the third step of the see-think-act cycle — for movement.
#
#  What this node does:
#    1. Reads target pan and tilt angles from the brain node.
#    2. Smoothly moves the head toward those angles using a control loop.
#    3. Sends the final angles to the Dynamixel motors via DynamixelDriver.
#
#  The smooth movement comes from exponential smoothing (also called a
#  low-pass filter or EMA). Instead of jumping directly to the target,
#  the head moves a small fraction of the remaining distance every tick.
#  This makes the motion look natural rather than jerky.
#
#  Subscribed topics:
#    /head/pan_target   — Float32  (degrees, from brain node)
#    /head/tilt_target  — Float32  (degrees, from brain node)
#

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from .dynamixel_driver import DynamixelDriver


class HeadNode(Node):

    def __init__(self):
        super().__init__('head_node')

        # ── Hardware setup ────────────────────────────────────────────────────
        self.driver = DynamixelDriver()

        # Read the current motor positions and save them as "zero".
        # The head should be pointing straight forward when this runs.
        self.driver.calibrate_zero()

        # Enable torque so the motors hold their position.
        self.driver.enable()

        # ── State ─────────────────────────────────────────────────────────────
        # current_* : where the head is pointing right now
        # target_*  : where the brain node wants the head to point
        self.current_pan  = 0.0
        self.current_tilt = 0.0

        self.target_pan   = 0.0
        self.target_tilt  = 0.0

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            Float32,
            '/head/pan_target',
            self.pan_callback,
            10,
        )

        self.create_subscription(
            Float32,
            '/head/tilt_target',
            self.tilt_callback,
            10,
        )

        # ── Control loop timer ────────────────────────────────────────────────
        # This calls control_loop() 20 times per second (every 0.05 seconds).
        # The callbacks above only update the *target*; this loop moves toward it.
        self.create_timer(0.05, self.control_loop)

        self.get_logger().info("Head node started.")


    # ── Topic callbacks ───────────────────────────────────────────────────────
    # These just store the latest target. The actual movement happens in
    # control_loop(), which runs on a separate timer.

    def pan_callback(self, msg):
        self.target_pan = msg.data

    def tilt_callback(self, msg):
        self.target_tilt = msg.data


    # ── Control loop ──────────────────────────────────────────────────────────

    def control_loop(self):
        """
        Runs at 20 Hz. Smoothly moves the head toward the target angles.

        Exponential smoothing formula:
            current = current + alpha * (target - current)

        alpha controls how fast the head moves:
          alpha = 1.0  → head jumps directly to target (instant, jerky)
          alpha = 0.1  → head moves 10% of the remaining distance each tick
                         (slow, very smooth)
          alpha = 0.15 → a good balance for this robot

        Think of it like this: each tick the head closes 15% of the gap
        between where it is now and where it wants to be. The gap gets
        smaller and smaller, so movement naturally decelerates as it arrives.
        """

        alpha = 0.15  # smoothing factor — try changing this to see the effect

        self.current_pan  += (self.target_pan  - self.current_pan)  * alpha
        self.current_tilt += (self.target_tilt - self.current_tilt) * alpha

        self.driver.set_pan(self.current_pan)
        self.driver.set_tilt(self.current_tilt)


    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        """
        Called automatically when the node shuts down (e.g. Ctrl+C).
        Disables the motors and closes the serial port cleanly.
        """
        self.driver.disable()
        self.driver.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HeadNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()