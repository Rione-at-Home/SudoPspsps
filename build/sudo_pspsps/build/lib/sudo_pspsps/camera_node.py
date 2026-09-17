#
## camera_node.py   ── SEE
#
#  This is the first step of the see-think-act cycle.
#
#  What this node does:
#    1. Receives raw images from the RealSense camera.
#    2. Runs MediaPipe on each frame to find a person's body.
#    3. If a person is found, publishes the pixel (x, y) position of
#       their nose so other nodes know where to look.
#
#  Published topics:
#    /cat/person_pixel  — Float32MultiArray [pixel_x, pixel_y]
#                         Only published when a person is actually visible.
#
#  Subscribed topics:
#    /camera/camera/color/image_raw  — sensor_msgs/Image  (from the camera)
#

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import mediapipe as mp


class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')

        # ── MediaPipe pose detector ───────────────────────────────────────────
        # MediaPipe Pose finds 33 landmarks on the human body (shoulders,
        # elbows, nose, etc.). We only use the nose landmark as a simple
        # "where is this person" signal.
        self.pose = mp.solutions.pose.Pose(
            static_image_mode    = False,  # video mode — faster for live frames
            min_detection_confidence = 0.5,
        )

        # CvBridge converts ROS2 Image messages into OpenCV/NumPy arrays
        # that MediaPipe can process.
        self.bridge = CvBridge()

        # ── Subscriber ────────────────────────────────────────────────────────
        # Every time the camera publishes a new frame, our callback runs.
        self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10,  # queue depth — how many unprocessed messages to buffer
        )

        # ── Publisher ─────────────────────────────────────────────────────────
        # We publish [pixel_x, pixel_y] whenever a person is detected.
        # The brain node subscribes to this to decide where to look.
        self.pixel_pub = self.create_publisher(
            Float32MultiArray,
            '/cat/person_pixel',
            10,
        )

        self.get_logger().info("Camera node started — waiting for frames.")


    def image_callback(self, msg):
        """
        Called automatically every time a new camera frame arrives.

        Steps:
          1. Convert the ROS2 image message → NumPy array.
          2. Run MediaPipe pose detection.
          3. If a person is found, publish their nose position in pixels.
        """

        # Step 1: Convert ROS2 Image → BGR NumPy array → RGB NumPy array.
        # MediaPipe expects RGB; the camera publishes BGR.
        bgr_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rgb_image = bgr_image[:, :, ::-1]   # flip channel order BGR → RGB

        # Step 2: Run pose detection.
        results = self.pose.process(rgb_image)

        # If no person was found, do nothing and wait for the next frame.
        if not results.pose_landmarks:
            return

        # Step 3: Extract the nose landmark (landmark index 0).
        # MediaPipe gives us normalised coordinates in the range [0.0, 1.0].
        # Multiply by image dimensions to get actual pixel coordinates.
        image_height, image_width, _ = bgr_image.shape

        nose = results.pose_landmarks.landmark[0]  # index 0 = nose tip

        pixel_x = nose.x * image_width
        pixel_y = nose.y * image_height

        # Publish the pixel position so the brain node can read it.
        out_msg = Float32MultiArray()
        out_msg.data = [float(pixel_x), float(pixel_y)]
        self.pixel_pub.publish(out_msg)

        self.get_logger().info(
            f"Person detected — nose at pixel ({pixel_x:.0f}, {pixel_y:.0f})"
        )


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()

    try:
        rclpy.spin(node)   # keep the node alive and processing callbacks
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()