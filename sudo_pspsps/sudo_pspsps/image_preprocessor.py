import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import datetime
import os


# The preprocessor fires exactly once — on the first good frame after the
# head transitions into TRACKING.  Subsequent frames (and any TALKING frames)
# are ignored so the model server receives a single, clean snapshot per
# "person found" event rather than a continuous stream.
_CAPTURE_STATE = "tracking"


class ImagePreprocessor(Node):
    def __init__(self):
        super().__init__('image_preprocessor')

        # ── One-shot capture flag ─────────────────────────────────────────────
        # Set to True on every SEARCHING → TRACKING transition; cleared after
        # the first good frame is published.  This guarantees exactly one
        # snapshot per "person found" event, taken on a warm, stable frame
        # (the head has already completed acquisition) rather than at an
        # arbitrary counter offset that may land on a dark startup frame.
        self._capture_pending: bool = False
        self._has_captured: bool = False   # latches True after the first snapshot; never resets
        self._head_state: str | None = None

        # Latched QoS to mirror how /head/state is published — ensures we get
        # the current value immediately on connect even if no new message comes.
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String,
            '/head/state',
            self._state_callback,
            latched_qos,
        )

        # ── Camera subscription ───────────────────────────────────────────────
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.listener_callback,
            10
        )

        # ── Publishers ────────────────────────────────────────────────────────
        # Publisher 1: Full image with bounding box drawn on it (spatial context)
        self.full_image_publisher = self.create_publisher(
            Image, '/pre_processed_tracker_frame', 10
        )
        # Publisher 2: Cropped image of just the person (detail context)
        self.crop_image_publisher = self.create_publisher(
            Image, '/pre_processed_person_crop', 10
        )
        # Publisher 3: Bounding box coordinates [xmin, ymin, xmax, ymax]
        self.bbox_publisher = self.create_publisher(
            Float32MultiArray, '/pre_processed_bbox', 10
        )
        # Publisher 4: trigger TALKING mode in the targeting node
        self._behavior_publisher = self.create_publisher(
            String, '/head/behavior', 10
        )

        self.bridge = CvBridge()

        # ── MediaPipe Pose ────────────────────────────────────────────────────
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5
        )

        self.get_logger().info("ImagePreprocessor ready — waiting for head lock.")

    # ── Head-state callback ───────────────────────────────────────────────────

    def _state_callback(self, msg: String):
        new_state = msg.data.strip().lower()
        if new_state == self._head_state:
            return

        self.get_logger().info(f"Head state → {new_state.upper()}")
        prev_state       = self._head_state
        self._head_state = new_state

        # Arm the one-shot capture on every SEARCHING → TRACKING transition.
        # TALKING is excluded: by the time the external node sets TALKING the
        # snapshot has already been taken (or the person was never cleanly
        # acquired), so there is nothing useful to re-capture.
        if new_state == _CAPTURE_STATE and prev_state != _CAPTURE_STATE:
            if self._has_captured:
                self.get_logger().info("Capture already done — ignoring re-acquisition.")
                return
            self._capture_pending = True
            self.get_logger().info("Capture armed — waiting for first stable frame.")

    # ── Main image callback ───────────────────────────────────────────────────

    def listener_callback(self, msg):
        """
        Process incoming images and publish a single preprocessed snapshot to
        the model server — fired once on the first good frame after the head
        transitions SEARCHING → TRACKING.

        Using the state transition (rather than an arbitrary frame-count modulo)
        guarantees:
          • The camera is warm — no dark startup frames.
          • The head has completed acquisition (ACQUIRE_FRAMES consecutive hits).
          • Exactly one snapshot is sent per "person found" event.

        Publishes:
          1. Full image (336×336) with bounding box drawn — spatial context.
          2. Cropped person image (224×224) — close-up detail.
          3. Bounding box coordinates in original pixel space.
        """
        # ── Gate: only process when a fresh capture has been armed ───────────
        if not self._capture_pending:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)

        if not results.pose_landmarks:
            self.get_logger().info("No person detected in frame, skipping.")
            return

        # Disarm the one-shot flag — we have a good frame with a confirmed
        # person.  Do this before publishing so a mid-publish state transition
        # cannot re-arm and fire a second time for this same acquisition event.
        self._capture_pending = False
        self._has_captured = True
        self.get_logger().info("Stable frame captured — publishing snapshot.")

        h, w, _ = cv_image.shape

        # ── Bounding box from pose landmarks ──────────────────────────────────
        x_coords = [int(lm.x * w) for lm in results.pose_landmarks.landmark]
        y_coords = [int(lm.y * h) for lm in results.pose_landmarks.landmark]
        xmin = max(0, min(x_coords) - 20)
        xmax = min(w, max(x_coords) + 20)
        ymin = max(0, min(y_coords) - 20)
        ymax = min(h, max(y_coords) + 20)

        # ── Publish bbox coordinates ───────────────────────────────────────────
        bbox_msg = Float32MultiArray()
        bbox_msg.data = [float(xmin), float(ymin), float(xmax), float(ymax)]
        self.bbox_publisher.publish(bbox_msg)

        # ── Full image: draw bounding box, resize to 336×336 ──────────────────
        annotated = cv_image.copy()
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        cv2.imshow("Detection Preview", annotated)
        cv2.waitKey(1)

        full_resized = cv2.resize(annotated, (336, 336))

        # ── Save debug images to ~/capture_debug/ ─────────────────────────────
        # Saved before publishing so there's always something on disk even if
        # a downstream subscriber crashes.  Timestamp ties each file to the log.
        _ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _save_dir = os.path.expanduser("~/capture_debug")
        os.makedirs(_save_dir, exist_ok=True)
        _full_path = os.path.join(_save_dir, f"capture_{_ts}_full.jpg")
        cv2.imwrite(_full_path, full_resized)
        self.get_logger().info(f"Saved full frame  → {_full_path}")

        # ── Switch to TALKING mode ─────────────────────────────────────────────
        # Published immediately after the snapshot is confirmed good so the
        # head locks into engaged mode while the model server processes the image.
        behavior_msg = String()
        behavior_msg.data = "talking"
        self._behavior_publisher.publish(behavior_msg)
        self.get_logger().info("Published → /head/behavior: talking")

        full_msg = self.bridge.cv2_to_imgmsg(full_resized, encoding='bgr8')
        full_msg.header = msg.header
        self.full_image_publisher.publish(full_msg)

        # ── Cropped person image: slice bbox region, resize to 224×224 ────────
        if xmax > xmin and ymax > ymin:
            person_crop = cv_image[ymin:ymax, xmin:xmax]
            crop_resized = cv2.resize(person_crop, (224, 224))
            _crop_path = os.path.join(_save_dir, f"capture_{_ts}_crop.jpg")
            cv2.imwrite(_crop_path, crop_resized)
            self.get_logger().info(f"Saved person crop → {_crop_path}")
            crop_msg = self.bridge.cv2_to_imgmsg(crop_resized, encoding='bgr8')
            crop_msg.header = msg.header
            self.crop_image_publisher.publish(crop_msg)
        else:
            self.get_logger().warn("Degenerate bounding box, skipping crop publish.")


def main(args=None):
    rclpy.init(args=args)
    node = ImagePreprocessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()