#
## brain_node.py   ── THINK
#
#  This is the second step of the see-think-act cycle.
#
#  What this node does:
#    1. Reads the person's pixel position from the camera node.
#    2. Converts that pixel position into pan and tilt angles (degrees).
#    3. Publishes those angles so the head node can move the motors.
#    4. The first time a person is spotted, publishes a greeting.
#
#  The key idea here is the pixel → angle conversion. The camera has a
#  known field of view, so we can work out exactly how many degrees
#  off-centre a person is just from their pixel position.
#
#  Published topics:
#    /head/pan_target      — Float32  (degrees, + = right)
#    /head/tilt_target     — Float32  (degrees, + = up)
#    /cat/robot_actions    — String   (JSON with a "message_to_user" key)
#
#  Subscribed topics:
#    /cat/person_pixel     — Float32MultiArray [pixel_x, pixel_y]
#

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String, Float32MultiArray
import json


class BrainNode(Node):

    # ── Camera settings ───────────────────────────────────────────────────────
    # These must match the resolution your RealSense camera is streaming at.
    # The field of view (FOV) is a physical property of the lens — it does not
    # change when you change resolution, but CAMERA_WIDTH and CAMERA_HEIGHT do.
    CAMERA_WIDTH  = 1280   # pixels
    CAMERA_HEIGHT =  800   # pixels
    HFOV          =  87.0  # horizontal field of view in degrees (D455 spec)
    VFOV          =  58.0  # vertical   field of view in degrees (D455 spec)

    # ── Motor direction ───────────────────────────────────────────────────────
    # Depending on how the camera is mounted relative to the motors, a person
    # on the right side of the image might require a negative pan command, or
    # vice versa. Flip these to -1 if the head moves the wrong way.
    PAN_SIGN  = -1
    TILT_SIGN =  1

    def __init__(self):
        super().__init__('brain_node')

        # Has the cat greeted the person yet this session?
        self.has_greeted = False

        # ── Subscriber ────────────────────────────────────────────────────────
        self.create_subscription(
            Float32MultiArray,
            '/cat/person_pixel',
            self.pixel_callback,
            10,
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self.pan_pub     = self.create_publisher(Float32, '/head/pan_target',  10)
        self.tilt_pub    = self.create_publisher(Float32, '/head/tilt_target', 10)
        self.actions_pub = self.create_publisher(String,  '/cat/robot_actions', 10)

        self.get_logger().info("Brain node started.")


    def pixel_callback(self, msg):
        """
        Called every time the camera node publishes a person's pixel position.

        Steps:
          1. Unpack pixel_x and pixel_y from the message.
          2. Convert pixels → degrees using the camera's field of view.
          3. Publish pan and tilt targets to the head node.
          4. If this is the first detection, publish a greeting.
        """

        pixel_x, pixel_y = msg.data[0], msg.data[1]

        # ── Step 2: pixel → angle conversion ─────────────────────────────────
        #
        # How many pixels represent one degree of angle?
        #   pixels_per_degree = image_width / horizontal_fov
        #
        # The person's distance from the image centre (in pixels) tells us
        # how far off-centre they are. Dividing by pixels_per_degree gives
        # the angular error in degrees.
        #
        # Example:
        #   Image is 1280 px wide, FOV is 87°  →  1280/87 ≈ 14.7 px per degree
        #   Person's nose is at pixel_x = 800
        #   Image centre is at 1280/2 = 640
        #   Error in pixels = 800 - 640 = 160 px to the right
        #   Error in degrees = 160 / 14.7 ≈ 10.9° to the right
        #   So we command the head to pan +10.9° (right).

        px_per_deg_h = self.CAMERA_WIDTH  / self.HFOV
        px_per_deg_v = self.CAMERA_HEIGHT / self.VFOV

        image_centre_x = self.CAMERA_WIDTH  / 2.0
        image_centre_y = self.CAMERA_HEIGHT / 2.0

        # Pixel error: how far is the person from dead-centre?
        error_x = pixel_x - image_centre_x
        error_y = pixel_y - image_centre_y

        # Convert to degrees. PAN_SIGN / TILT_SIGN correct for motor mounting.
        # Note the minus sign on tilt: pixel Y increases downward in images,
        # but a positive tilt angle means "look up", so we flip the sign.
        pan_angle  =  self.PAN_SIGN  * (error_x / px_per_deg_h)
        tilt_angle =  self.TILT_SIGN * (-error_y / px_per_deg_v)

        # ── Step 3: publish angles ────────────────────────────────────────────
        pan_msg  = Float32()
        tilt_msg = Float32()
        pan_msg.data  = pan_angle
        tilt_msg.data = tilt_angle

        self.pan_pub.publish(pan_msg)
        self.tilt_pub.publish(tilt_msg)

        self.get_logger().info(
            f"Pan target: {pan_angle:+.1f}°   Tilt target: {tilt_angle:+.1f}°"
        )

        # ── Step 4: one-shot greeting ─────────────────────────────────────────
        if not self.has_greeted:
            self.has_greeted = True
            self._publish_greeting()


    def _publish_greeting(self):
        """
        Publish a greeting message the first time a person is detected.
        The TTS node subscribes to /cat/robot_actions and will speak this text.

        The message is a JSON string with a "message_to_user" key.
        This format matches what tts_node.py already expects.
        """
        greeting = {
            "message_to_user": "Hello! I can see you. Nice to meet you!"
        }
        msg = String()
        msg.data = json.dumps(greeting)
        self.actions_pub.publish(msg)
        self.get_logger().info("Published greeting.")


def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()