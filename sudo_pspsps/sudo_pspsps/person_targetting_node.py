#
## person_targeting_node.py
#
#  Behaviour states (externally switchable via /head/behavior String topic):
#
#    SEARCHING  — No person committed yet.  Head makes slow deliberate cat-like
#                 glances (discrete saccades to a few look-points, not a sweep).
#                 Requires ACQUIRE_FRAMES consecutive detections before
#                 committing to TRACKING — prevents flickering on a single
#                 noisy frame.
#
#    TRACKING   — Locked onto the closest person.  Cat saccade+hold: only moves
#                 when the smoothed error exceeds a generous dead-zone, then
#                 snaps and freezes.  Occasional curious idle head-tilts.
#                 Only returns to SEARCHING after LOST_TIMEOUT_SEC of *zero*
#                 detections — brief MediaPipe dropouts are ignored.
#
#    TALKING    — Engaged with person.  Pan is nearly locked (only large
#                 corrections allowed).  Tilt nods gently and periodically.
#                 Occasional small curious tilts.  Falls back to SEARCHING
#                 only on sustained person-loss.
#
#  State-switch hysteresis
#  ───────────────────────
#  SEARCHING → TRACKING  requires ACQUIRE_FRAMES (~1s) of consecutive hits.
#                         Transition is triggered by the detection stream, not
#                         a timer.
#  TRACKING  → SEARCHING requires LOST_TIMEOUT_SEC of no detections, enforced
#                         exclusively by the watchdog — single-frame misses are
#                         ignored so brief MediaPipe dropouts don't break lock.
#  TALKING   → SEARCHING same lost-timeout watchdog rule.
#  TALKING   is only entered via the external /head/behavior topic; it is
#             never auto-transitioned into by this node.
#
#  Published topics
#  ────────────────
#  /head/pan_target   Float32  — commanded pan angle (degrees)
#  /head/tilt_target  Float32  — commanded tilt angle (degrees)
#  /head/state        String   — current state: "searching" | "tracking" | "talking"
#                                Latched (transient_local) so late subscribers
#                                get the last value immediately on connect.
#
#  Movement philosophy (cat-like)
#  ───────────────────────────────
#  • Moves are discrete saccades — snap, then hold completely still.
#  • Dead-zones are large so jitter never triggers movement.
#  • EMA smoothing on detections so landmark noise is absorbed before
#    any decision is made.
#  • Smoother never resets to None — decays toward centre when lost,
#    preventing the cold-start giant-saccade bug.
#

import math
import random
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, HistoryPolicy,
    DurabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import os
import tempfile

# ── MediaPipe model ────────────────────────────────────────────────────────────
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
_MODEL_PATH = os.path.join(tempfile.gettempdir(), "pose_landmarker_lite.task")

def _ensure_model() -> str:
    if not os.path.exists(_MODEL_PATH):
        print(f"[PersonTargetingNode] Downloading model → {_MODEL_PATH}")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH

# ── Behaviour names ────────────────────────────────────────────────────────────
BEHAVIOR_SEARCHING = "searching"
BEHAVIOR_TRACKING  = "tracking"
BEHAVIOR_TALKING   = "talking"

# ── Saccade sub-states ─────────────────────────────────────────────────────────
_HOLDING   = "holding"
_SACCADING = "saccading"


class PersonTargetingNode(Node):

    # ── Camera intrinsics — RealSense D455 ───────────────────────────────────
    # D455 colour sensor: 1280x800 native, 87x58 deg FOV.
    # If you stream at a lower resolution (e.g. 640x400) the FOV is identical;
    # only px_per_deg changes.  Update CAMERA_WIDTH/HEIGHT to match your
    # launch file — HFOV/VFOV stay fixed regardless of resolution.
    CAMERA_WIDTH  = 1280
    CAMERA_HEIGHT = 800
    HFOV = 87.0
    VFOV = 58.0


    # ── Motor direction signs ─────────────────────────────────────────────────
    # Set to +1 or -1 to match the physical installation.
    #
    # PAN_SIGN:  +1 = positive motor angle pans RIGHT (from the front).
    #            Flip to -1 if the head moves LEFT when the person is right.
    # TILT_SIGN: +1 = positive motor angle tilts UP.
    #            Flip to -1 if the head tilts DOWN when the person is above centre.
    #
    # Logs show the head moved the wrong way in pan, so PAN_SIGN = -1.
    # Verify TILT_SIGN by standing in front and moving up/down.
    PAN_SIGN  = -1
    TILT_SIGN =  1

    # ── Detection cadence ─────────────────────────────────────────────────────
    PROCESS_EVERY_N_FRAMES = 3        # ~10 Hz at 30 fps

    # ── EMA smoothing ─────────────────────────────────────────────────────────
    # Slower alpha (0.15) means ~7 frames to absorb a position jump —
    # keeps landmark jitter from ever reaching the saccade trigger.
    EMA_ALPHA       = 0.15
    EMA_DECAY_ALPHA = 0.03            # gentle centre-drift when no person

    # ── State-switch hysteresis ───────────────────────────────────────────────
    # Must see this many *consecutive* detection frames before switching
    # SEARCHING → TRACKING.  At ~10 Hz, 10 frames ≈ 1.0 s of stable detection.
    ACQUIRE_FRAMES   = 10

    # Must lose the person for this many seconds before switching back to
    # SEARCHING.  6 s tolerates brief occlusions / pose-detector dropouts.
    LOST_TIMEOUT_SEC = 6.0

    # ── Saccade thresholds ────────────────────────────────────────────────────
    # TRACKING dead-zone is generous — a still person standing in frame
    # should produce near-zero movement.
    TRACKING_SACCADE_TRIGGER_DEG = 12.0
    TRACKING_SETTLE_DEG          =  3.0
    TRACKING_MIN_HOLD_SEC        =  3  # freeze after each saccade

    # TALKING pan corrections are rare — only large misalignments get fixed.
    TALKING_PAN_TRIGGER_DEG  = 15.0
    TALKING_MIN_HOLD_SEC     =  2.0

    # ── SEARCHING glance targets (pan °, tilt °) ──────────────────────────────
    SEARCH_GLANCE_POSITIONS = [
        ( 0.0, -3.0),   # centre
        (+60.0, -3.0),  # right
        ( 0.0, -3.0),   # centre
        (-60.0, -3.0),  # left
    ]
    SEARCH_HOLD_SEC         = 2.2     # seconds to dwell at each glance
    SEARCH_SACCADE_SPEED    = 40.0    # °/s during the glance movement itself

    # ── TRACKING idle tilt ────────────────────────────────────────────────────
    IDLE_TILT_INTERVAL_SEC    =  6.0
    IDLE_TILT_INTERVAL_JITTER =  2.5
    IDLE_TILT_MAG_DEG_SIGMA   =  3.0
    IDLE_TILT_MAG_MAX_DEG     =  5.0

    # ── TALKING motion ────────────────────────────────────────────────────────
    TALKING_NOD_AMP_DEG      =  2.5
    TALKING_NOD_PERIOD_SEC   =  3.8
    TALKING_TILT_INTERVAL_SEC    = 8.0
    TALKING_TILT_INTERVAL_JITTER = 3.0

    # ── Depth sampling ────────────────────────────────────────────────────────
    DEPTH_BBOX_SHRINK = 0.2
    DEPTH_MIN_MM      = 200
    DEPTH_MAX_MM      = 8000

    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__('person_targeting_node')

        self._bridge        = CvBridge()
        self._frame_counter = 0
        self._latest_depth  = None

        self._px_per_deg_h = self.CAMERA_WIDTH  / self.HFOV
        self._px_per_deg_v = self.CAMERA_HEIGHT / self.VFOV
        self._cx = self.CAMERA_WIDTH  / 2.0
        self._cy = self.CAMERA_HEIGHT / 2.0

        # ── EMA — always alive, initialised to image centre ───────────────────
        self._smooth_px = self._cx
        self._smooth_py = self._cy

        # ── Detection state ───────────────────────────────────────────────────
        # _person_visible is ONLY cleared by the watchdog, not by individual
        # missed frames.  This makes TRACKING robust to brief dropout frames.
        # _consecutive_misses counts unbroken miss-frames for the watchdog to
        # use as a secondary signal alongside wall-clock time.
        self._person_visible      = False
        self._last_detection_time = None   # rclpy.Time of last positive frame

        # ── Acquisition counter (hysteresis into TRACKING) ────────────────────
        self._acquire_count = 0

        # ── Published gaze ────────────────────────────────────────────────────
        self._gaze_pan  = 0.0
        self._gaze_tilt = 0.0

        # ── Saccade sub-state (shared by TRACKING & TALKING) ──────────────────
        self._saccade_state = _HOLDING
        self._hold_until    = None        # float seconds
        self._saccade_timeout = None        # rclpy.Time when a saccade should be considered failed if not settled


        # ── Idle / curious tilt ───────────────────────────────────────────────
        self._idle_tilt_offset    = 0.0
        self._next_idle_tilt_time = None

        # ── Top-level behaviour ────────────────────────────────────────────────
        self._behavior = BEHAVIOR_SEARCHING

        # ── Searching glance state ─────────────────────────────────────────────
        self._search_glance_idx   = 0
        self._search_glance_until = None

        # ── Talking clock ─────────────────────────────────────────────────────
        self._talking_start_sec = None

        # ── MediaPipe ─────────────────────────────────────────────────────────
        model_path = _ensure_model()
        base_opts  = mp_python.BaseOptions(model_asset_path=model_path)
        pose_opts  = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=4,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(pose_opts)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Latched QoS for /head/state so late subscribers (e.g. ImagePreprocessor)
        # immediately receive the current state on connect.
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(
            Image, '/camera/camera/color/image_raw',
            self._image_callback, sensor_qos,
        )
        self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw',
            self._depth_callback, sensor_qos,
        )
        self.create_subscription(
            String, '/head/behavior',
            self._behavior_callback, 10,
        )

        self._pan_pub   = self.create_publisher(Float32, '/head/pan_target',  10)
        self._tilt_pub  = self.create_publisher(Float32, '/head/tilt_target', 10)
        self._state_pub = self.create_publisher(String,  '/head/state', latched_qos)

        self.create_timer(0.05, self._motion_tick)   # 20 Hz
        self.create_timer(0.5,  self._watchdog)

        # Kick off the SEARCHING state properly on the first spin iteration.
        # We use a one-shot timer (cancelled inside the callback) rather than
        # calling _enter_searching() directly here because get_clock().now()
        # returns 0 during __init__ in some RMW implementations — the timer
        # fires after the node is fully initialised and the clock is live.
        self._startup_done = False

        self.get_logger().info(
            "PersonTargetingNode ready  "
            f"acquire={self.ACQUIRE_FRAMES} frames  "
            f"lost_timeout={self.LOST_TIMEOUT_SEC}s  "
            f"track_trigger={self.TRACKING_SACCADE_TRIGGER_DEG}°  "
            f"hold={self.TRACKING_MIN_HOLD_SEC}s"
        )

    # ── State publisher ───────────────────────────────────────────────────────

    def _publish_state(self):
        msg = String()
        msg.data = self._behavior
        self._state_pub.publish(msg)

    def _transition_to(self, new_behavior: str, now_sec: float):
        """
        Central choke-point for all state transitions.
        Logs the edge, updates _behavior, calls the enter-hook, and
        publishes /head/state.  All callers must go through here.
        """
        if new_behavior == self._behavior:
            return
        self.get_logger().info(
            f"STATE  {self._behavior.upper()} → {new_behavior.upper()}"
        )
        self._behavior = new_behavior
        if new_behavior == BEHAVIOR_SEARCHING:
            self._enter_searching(now_sec)
        elif new_behavior == BEHAVIOR_TRACKING:
            self._enter_tracking(now_sec, fresh=True)
        elif new_behavior == BEHAVIOR_TALKING:
            self._enter_talking(now_sec)
        self._publish_state()

    # ── External behavior override ────────────────────────────────────────────

    def _behavior_callback(self, msg: String):
        requested = msg.data.strip().lower()
        if requested not in (BEHAVIOR_SEARCHING, BEHAVIOR_TRACKING, BEHAVIOR_TALKING):
            self.get_logger().warn(f"Unknown behavior '{requested}' — ignoring.")
            return
        now_sec = self.get_clock().now().nanoseconds / 1e9
        self._transition_to(requested, now_sec)

    # ── Depth callback ────────────────────────────────────────────────────────

    def _depth_callback(self, msg: Image):
        self._latest_depth = self._bridge.imgmsg_to_cv2(
            msg, desired_encoding='passthrough'
        )

    # ── Image callback ────────────────────────────────────────────────────────

    def _image_callback(self, msg: Image):
        self._frame_counter += 1
        if self._frame_counter % self.PROCESS_EVERY_N_FRAMES != 0:
            return

        cv_image  = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w      = cv_image.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        results  = self._landmarker.detect(mp_image)

        if not results.pose_landmarks:
            # Decay smoother toward centre — prevents cold-start saccade jump.
            # Do NOT clear _person_visible here — that is the watchdog's job.
            # Resetting it per-frame would cause TRACKING to stutter on every
            # brief MediaPipe miss.
            self._smooth_px += self.EMA_DECAY_ALPHA * (self._cx - self._smooth_px)
            self._smooth_py += self.EMA_DECAY_ALPHA * (self._cy - self._smooth_py)
            self._acquire_count = 0   # reset acquisition streak on any miss
            return

        # ── Build bboxes ──────────────────────────────────────────────────────
        # Minimum area filter: a real person at reasonable range occupies at
        # least ~2% of the frame area.  Tiny landmark clusters on wall textures
        # or furniture are rejected here before ever reaching the acquisition
        # counter, preventing false SEARCHING → TRACKING transitions.
        MIN_PERSON_AREA_FRAC = 0.02
        min_area = w * h * MIN_PERSON_AREA_FRAC

        persons = []
        for landmarks in results.pose_landmarks:
            xs = [int(lm.x * w) for lm in landmarks]
            ys = [int(lm.y * h) for lm in landmarks]
            xmin = max(0, min(xs) - 20)
            xmax = min(w, max(xs) + 20)
            ymin = max(0, min(ys) - 20)
            ymax = min(h, max(ys) + 20)
            bbox_area = (xmax - xmin) * (ymax - ymin)
            if xmax > xmin and ymax > ymin and bbox_area >= min_area:
                persons.append((xmin, ymin, xmax, ymax))

        if not persons:
            self._acquire_count = 0   # landmarks present but all degenerate boxes
            return

        # ── Update EMA with closest-person centre ─────────────────────────────
        xmin, ymin, xmax, ymax = self._closest_person(persons)
        raw_px = (xmin + xmax) / 2.0
        raw_py = (ymin + ymax) / 2.0

        a = self.EMA_ALPHA
        self._smooth_px = a * raw_px + (1.0 - a) * self._smooth_px
        self._smooth_py = a * raw_py + (1.0 - a) * self._smooth_py

        was_visible               = self._person_visible
        self._person_visible      = True
        self._last_detection_time = self.get_clock().now()

        now_sec = self.get_clock().now().nanoseconds / 1e9

        if self._behavior == BEHAVIOR_SEARCHING:
            # ── SEARCHING: accumulate consecutive hits before committing ───────
            self._acquire_count += 1
            if self._acquire_count >= self.ACQUIRE_FRAMES:
                self.get_logger().info(
                    f"Person acquired ({self.ACQUIRE_FRAMES} consecutive frames) "
                    "— SEARCHING → TRACKING"
                )
                self._transition_to(BEHAVIOR_TRACKING, now_sec)
        else:
            # ── TRACKING / TALKING: re-acquisition hold after a dropout ────────
            if not was_visible:
                # EMA needs a few frames to warm up after being invisible —
                # hold before allowing any saccade.
                self._hold_until = now_sec + self.TRACKING_MIN_HOLD_SEC

    # ── 20 Hz motion tick ─────────────────────────────────────────────────────

    def _motion_tick(self):
        now_sec = self.get_clock().now().nanoseconds / 1e9

        # First tick after node is fully live: kick off the SEARCHING glance
        # sequence.  We do this here rather than in __init__ because
        # get_clock().now() can return 0 during construction.
        if not self._startup_done:
            self._startup_done = True
            self._enter_searching(now_sec)
            self._publish_state()

        if self._behavior == BEHAVIOR_SEARCHING:
            self._tick_searching(now_sec)
        elif self._behavior == BEHAVIOR_TRACKING:
            self._tick_tracking(now_sec)
        elif self._behavior == BEHAVIOR_TALKING:
            self._tick_talking(now_sec)

    # ── SEARCHING ─────────────────────────────────────────────────────────────

    def _enter_searching(self, now_sec: float):
        self._acquire_count       = 0
        self._person_visible      = False   # safe to reset here: we're leaving lock
        self._search_glance_idx   = 0
        self._search_glance_until = now_sec + self.SEARCH_HOLD_SEC
        self._saccade_state       = _SACCADING
        self._idle_tilt_offset    = 0.0
        pan, tilt = self.SEARCH_GLANCE_POSITIONS[0]
        self._saccade_to(pan, tilt)
        self.get_logger().info(
            f"SEARCHING — glance[0] ({pan:+.0f}°, {tilt:+.0f}°)"
        )

    def _tick_searching(self, now_sec: float):
        if self._search_glance_until is None:
            return
        if now_sec < self._search_glance_until:
            return   # still dwelling — don't move

        self._search_glance_idx = (
            (self._search_glance_idx + 1) % len(self.SEARCH_GLANCE_POSITIONS)
        )
        pan, tilt = self.SEARCH_GLANCE_POSITIONS[self._search_glance_idx]
        self._saccade_to(pan, tilt)
        self._search_glance_until = now_sec + self.SEARCH_HOLD_SEC
        self.get_logger().info(
            f"SEARCHING — glance[{self._search_glance_idx}] "
            f"({pan:+.0f}°, {tilt:+.0f}°)"
        )

    # ── TRACKING ──────────────────────────────────────────────────────────────

    def _enter_tracking(self, now_sec: float, fresh: bool = False):
        self._saccade_state    = _HOLDING
        self._idle_tilt_offset = 0.0
        self._hold_until       = now_sec + self.TRACKING_MIN_HOLD_SEC
        self._saccade_timeout  = None
        self._schedule_next_idle_tilt(now_sec)

    def _tick_tracking(self, now_sec: float):
        if not self._person_visible:
            return

        pan_err, tilt_err, total_err = self._gaze_error()

        if self._saccade_state == _HOLDING:
            hold_expired = (self._hold_until is None or now_sec >= self._hold_until)
            if hold_expired and total_err > self.TRACKING_SACCADE_TRIGGER_DEG:
                target_pan, target_tilt, _ = self._target_angles()
                self.get_logger().info(
                    f"[TRACKING SACCADE] err={total_err:.1f}°  "
                    f"target=({target_pan:+.1f}°, {target_tilt:+.1f}°)"
                )
                self._idle_tilt_offset = 0.0
                self._schedule_next_idle_tilt(now_sec)
                self._saccade_to(target_pan, target_tilt + self._idle_tilt_offset)
                
                self._saccade_state = _SACCADING
                # Allow 1.5 seconds for the smoothed head to physically reach the target
                self._saccade_timeout = now_sec + 1.5
            else:
                self._maybe_idle_tilt(now_sec)
            return

        # SACCADING — Wait for head to physically settle.
        # DO NOT re-issue _saccade_to() here! _gaze_pan is the commanded target;
        # adding current visual error to it while the head is moving causes runaway feedback.
        settled = total_err <= self.TRACKING_SETTLE_DEG
        timed_out = self._saccade_timeout is not None and now_sec >= self._saccade_timeout

        if settled or timed_out:
            self._saccade_state = _HOLDING
            self._hold_until    = now_sec + self.TRACKING_MIN_HOLD_SEC
            reason = "SETTLED" if settled else "TIMED OUT"
            self.get_logger().info(
                f"[TRACKING {reason}]  "
                f"gaze=({self._gaze_pan:+.1f}°, {self._gaze_tilt:+.1f}°)"
            )

    # ── TALKING ───────────────────────────────────────────────────────────────

    def _enter_talking(self, now_sec: float):
        self._talking_start_sec = now_sec
        self._saccade_state     = _HOLDING
        self._idle_tilt_offset  = 0.0
        self._next_idle_tilt_time = (
            now_sec
            + self.TALKING_TILT_INTERVAL_SEC
            + random.uniform(-self.TALKING_TILT_INTERVAL_JITTER,
                              self.TALKING_TILT_INTERVAL_JITTER)
        )
        self._hold_until = now_sec + self.TALKING_MIN_HOLD_SEC

    def _tick_talking(self, now_sec: float):
        if self._talking_start_sec is None:
            self._talking_start_sec = now_sec

        t = now_sec - self._talking_start_sec

        if self._person_visible:
            pan_err, tilt_err, _ = self._gaze_error()
            target_pan, target_tilt, _ = self._target_angles()
        else:
            pan_err    = 0.0
            target_pan = self._gaze_pan
            target_tilt = self._gaze_tilt

        # Pan: only correct for large misalignment (use frame error for threshold,
        # but command the absolute accumulated motor angle)
        hold_expired = (self._hold_until is None or now_sec >= self._hold_until)
        if hold_expired and abs(pan_err) > self.TALKING_PAN_TRIGGER_DEG:
            self._gaze_pan   = target_pan
            self._hold_until = now_sec + self.TALKING_MIN_HOLD_SEC
            self.get_logger().info(
                f"[TALKING PAN CORRECTION]  pan={target_pan:+.1f}°"
            )

        # Tilt: nod gently on top of centred tilt
        nod  = self.TALKING_NOD_AMP_DEG * math.sin(
            2 * math.pi * t / self.TALKING_NOD_PERIOD_SEC
        )
        tilt = target_tilt + nod + self._idle_tilt_offset

        self._gaze_tilt = tilt
        self._publish(self._gaze_pan, tilt)

        self._maybe_talking_tilt(now_sec, target_tilt)

    # ── Idle / curious tilt ───────────────────────────────────────────────────

    def _schedule_next_idle_tilt(self, now_sec: float):
        jitter = random.uniform(
            -self.IDLE_TILT_INTERVAL_JITTER,
             self.IDLE_TILT_INTERVAL_JITTER,
        )
        self._next_idle_tilt_time = now_sec + self.IDLE_TILT_INTERVAL_SEC + jitter

    def _maybe_idle_tilt(self, now_sec: float):
        """Cat curious tilt during TRACKING hold."""
        if self._next_idle_tilt_time is None:
            self._schedule_next_idle_tilt(now_sec)
            return
        if now_sec < self._next_idle_tilt_time:
            return
        mag = min(
            abs(random.gauss(0, self.IDLE_TILT_MAG_DEG_SIGMA)),
            self.IDLE_TILT_MAG_MAX_DEG,
        )
        self._idle_tilt_offset = math.copysign(mag, random.choice([-1, 1]))
        _, target_tilt, _ = self._target_angles()
        self._publish(self._gaze_pan, target_tilt + self._idle_tilt_offset)
        self._gaze_tilt = target_tilt + self._idle_tilt_offset
        self.get_logger().info(f"[IDLE TILT]  offset={self._idle_tilt_offset:+.1f}°")
        self._schedule_next_idle_tilt(now_sec)

    def _maybe_talking_tilt(self, now_sec: float, base_tilt: float):
        """Rarer curious tilt during TALKING."""
        if self._next_idle_tilt_time is None:
            self._next_idle_tilt_time = now_sec + self.TALKING_TILT_INTERVAL_SEC
            return
        if now_sec < self._next_idle_tilt_time:
            return
        mag = min(
            abs(random.gauss(0, self.IDLE_TILT_MAG_DEG_SIGMA)),
            self.IDLE_TILT_MAG_MAX_DEG,
        )
        self._idle_tilt_offset = math.copysign(mag, random.choice([-1, 1]))
        jitter = random.uniform(
            -self.TALKING_TILT_INTERVAL_JITTER,
             self.TALKING_TILT_INTERVAL_JITTER,
        )
        self._next_idle_tilt_time = now_sec + self.TALKING_TILT_INTERVAL_SEC + jitter
        self.get_logger().info(
            f"[TALKING TILT]  offset={self._idle_tilt_offset:+.1f}°"
        )

    # ── Watchdog ──────────────────────────────────────────────────────────────

    def _watchdog(self):
        """
        Sole authority for declaring the person lost and dropping back to
        SEARCHING.  Runs at 0.5 Hz.

        By keeping all 'person lost' logic here (instead of spreading it across
        the per-frame callback), we avoid state thrashing on momentary MediaPipe
        dropouts.  _person_visible is only cleared here, never inside
        _image_callback.
        """
        if self._last_detection_time is None:
            # No detection ever — nothing to time out yet.
            return

        elapsed = (
            self.get_clock().now() - self._last_detection_time
        ).nanoseconds / 1e9

        if elapsed <= self.LOST_TIMEOUT_SEC:
            return   # person still considered present

        # ── Person has been gone long enough ──────────────────────────────────
        if self._person_visible:
            self.get_logger().info(
                f"Person lost for {elapsed:.1f}s — clearing visibility."
            )
            self._person_visible = False
            self._acquire_count  = 0

        if self._behavior != BEHAVIOR_SEARCHING:
            self.get_logger().info(
                f"Person lost for {elapsed:.1f}s — {self._behavior.upper()} → SEARCHING."
            )
            now_sec = self.get_clock().now().nanoseconds / 1e9
            self._transition_to(BEHAVIOR_SEARCHING, now_sec)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _gaze_error(self) -> tuple[float, float, float]:
        """
        Angular error of the person relative to the image centre (degrees),
        already signed to match motor convention.

        Returns (pan_err, tilt_err, magnitude) where positive values mean
        "motor must move in positive direction to centre the person".

        PAN_SIGN / TILT_SIGN account for camera mirror and motor polarity so
        all higher-level logic treats positive = toward the person without
        caring about hardware orientation.

        These are frame-relative offsets — they do NOT include the current
        motor position. Use _target_angles() to get the absolute motor
        commands needed to centre the person.
        """
        pan_err  = self.PAN_SIGN  *  (self._smooth_px - self._cx) / self._px_per_deg_h
        tilt_err = self.TILT_SIGN * -(self._smooth_py - self._cy) / self._px_per_deg_v
        return pan_err, tilt_err, math.hypot(pan_err, tilt_err)

    def _target_angles(self) -> tuple[float, float, float]:
        """
        Absolute motor angles (pan, tilt, error_magnitude) that would centre
        the person in the frame.

        The motor currently points at (_gaze_pan, _gaze_tilt).  The person
        is (pan_err, tilt_err) away from the image centre.  To centre them
        the motor needs to move by that error on top of its current position.
        """
        pan_err, tilt_err, total_err = self._gaze_error()
        return (
            self._gaze_pan  + pan_err,
            self._gaze_tilt + tilt_err,
            total_err,
        )

    def _saccade_to(self, pan_deg: float, tilt_deg: float):
        """Immediately snap gaze to target and publish."""
        self._gaze_pan  = pan_deg
        self._gaze_tilt = tilt_deg
        self._publish(pan_deg, tilt_deg)

    def _closest_person(self, persons):
        if self._latest_depth is not None:
            return self._closest_by_depth(persons)
        return self._closest_by_area(persons)

    def _closest_by_depth(self, persons):
        depth  = self._latest_depth
        dh, dw = depth.shape[:2]
        best_bbox, best_median = None, float('inf')
        shrink = self.DEPTH_BBOX_SHRINK
        for (xmin, ymin, xmax, ymax) in persons:
            bw, bh = xmax - xmin, ymax - ymin
            sx1 = max(0,  int(xmin + bw * shrink / 2))
            sx2 = min(dw, int(xmax - bw * shrink / 2))
            sy1 = max(0,  int(ymin + bh * shrink / 2))
            sy2 = min(dh, int(ymax - bh * shrink / 2))
            sx2 = max(sx1 + 1, sx2)
            sy2 = max(sy1 + 1, sy2)
            roi   = depth[sy1:sy2, sx1:sx2].astype(np.float32)
            valid = roi[(roi >= self.DEPTH_MIN_MM) & (roi <= self.DEPTH_MAX_MM)]
            if valid.size == 0:
                continue
            median = float(np.median(valid))
            if median < best_median:
                best_median, best_bbox = median, (xmin, ymin, xmax, ymax)
        return best_bbox if best_bbox is not None else self._closest_by_area(persons)

    def _closest_by_area(self, persons):
        return max(persons, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

    def _publish(self, pan_deg: float, tilt_deg: float):
        pan_msg, tilt_msg = Float32(), Float32()
        pan_msg.data  = float(pan_deg)
        tilt_msg.data = float(tilt_deg)
        self._pan_pub.publish(pan_msg)
        self._tilt_pub.publish(tilt_msg)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = PersonTargetingNode()
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