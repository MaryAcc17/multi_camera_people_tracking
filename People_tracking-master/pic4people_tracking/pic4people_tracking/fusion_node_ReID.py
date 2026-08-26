"""Global multi-camera people-tracking fusion node.

The node receives person tracks from three camera-specific StrongSORT trackers,
fuses them by assigning and publishing temporally stable global tracks, with unique global IDs.
Association follows a BoT-SORT-inspired multi-cue strategy that combines:

* predicted 2D position in the map frame;
* Re-ID appearance embeddings;
* camera-specific local track identifiers; and
* temporal freshness and lifecycle constraints.

The implementation also takes care of removing duplications across cameras, stale-track
recovery, prevention of duplicate global-ID creation, reliability-based output
filtering, RViz markers, and structured JSON debug events for offline evaluation.

Notes
-----
The local trackers are expected to publish
``pic4people_tracking_msgs/TrackedPeople`` messages on ``/people_center``,
``/people_left``, and ``/people_right``. Global tracks are published as
``people_msgs/People`` on ``/tracked_people``.
"""
import json
from threading import Lock

import numpy as np
import rclpy
from rclpy.node import Node
from scipy.optimize import linear_sum_assignment
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from people_msgs.msg import People, Person
from pic4people_tracking_msgs.msg import TrackedPeople


#NUMERICAL HELPERS

#angle averaging helper
def avg_angle(a, b, alpha=0.7):
    """Return a weighted circular mean of two angles in radians.
    A Cartesian representation is used internally.
    The result is in the range [-pi, pi].
    The weight alpha is in the range [0, 1], where alpha=1 returns a, and alpha=0 returns b.
    """
    x = alpha * np.cos(a) + (1.0 - alpha) * np.cos(b)
    y = alpha * np.sin(a) + (1.0 - alpha) * np.sin(b)
    return np.arctan2(y, x)

#normalize embedding helper
#if the embedding is None, empty, or has zero norm, return None. Otherwise, return the normalized embedding.
def normalize_embedding(embedding):
    if embedding is None:
        return None
    try:
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if emb.size == 0:
        return None
    norm = float(np.linalg.norm(emb))
    if not np.isfinite(norm) or norm < 1e-12:
        return None
    return emb / norm

#compute cosine distance between two embeddings. 
def cosine_distance(a, b):
    """Return cosine distance ``1 - similarity`` between two embeddings.
    
    ``None`` is returned when either embedding is invalid or their dimensions do not
    match.
    """
    a = normalize_embedding(a)
    b = normalize_embedding(b)
    if a is None or b is None or a.shape != b.shape:
        return None
    sim = float(np.dot(a, b))
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim


class GlobalTrack:
    """State associated with a single global track, including its position, velocity, and Re-ID embedding."""
    def __init__(
        self,
        tid,
        x,
        y,
        yaw,
        t,
        cam=None,
        local_id=None,
        min_hits_to_confirm=1,
        velocity_decay_gain=1.2,
        embedding=None,
        reid_ema_alpha=0.85,
    ):
        """Initialize a global track from its first local detection."""
        self.id = tid
        self.x = x
        self.y = y
        self.yaw = yaw if yaw is not None and np.isfinite(yaw) else 0.0

        self.vx = 0.0
        self.vy = 0.0
        self.last_update = t

        # The detection that creates the track counts as the first valid hit.
        # Therefore, if min_hits_to_confirm == 1, the track is confirmed immediately.
        self.hits = 1
        self.missed = 0
        self.min_hits_to_confirm = int(min_hits_to_confirm)
        self.confirmed = self.hits >= self.min_hits_to_confirm
        self.velocity_decay_gain = float(velocity_decay_gain)

        self.local_ids_by_cam = {}
        self._remember_local_id(cam, local_id)

        self.reid_ema_alpha = float(reid_ema_alpha)
        self.embedding = normalize_embedding(embedding)

    def _remember_local_id(self, cam, local_id):
        """Store the latest non-empty local tracker ID observed by a camera."""
        if cam is None or local_id is None:
            return

        local_id = str(local_id)
        if local_id == "":
            return

        self.local_ids_by_cam[str(cam)] = local_id

    def _update_embedding(self, embedding):
        """
        Update the track's Re-ID embedding using an exponential moving average (EMA)."""
        new_embedding = normalize_embedding(embedding)
        if new_embedding is None:
            return False

        if self.embedding is None or self.embedding.shape != new_embedding.shape:
            self.embedding = new_embedding
            return True

        alpha = self.reid_ema_alpha
        smooth = alpha * self.embedding + (1.0 - alpha) * new_embedding
        smooth = normalize_embedding(smooth)
        if smooth is None:
            return False
        self.embedding = smooth
        return True

    def has_embedding(self):
        """Return True if the track has a valid Re-ID embedding."""
        return self.embedding is not None

    def update(self, x, y, yaw, t, cam=None, local_id=None, embedding=None):
        """Update the global track's state from an associated local detection."""
        dt = t - self.last_update

        old_x = self.x
        old_y = self.y
        old_vx = self.vx
        old_vy = self.vy
        old_yaw = self.yaw

        if yaw is not None and np.isfinite(yaw):
            self.yaw = avg_angle(self.yaw, yaw)

        if dt > 0:
            measured_vx = (x - old_x) / dt
            measured_vy = (y - old_y) / dt

            # A newly-created track starts with zero velocity.  Applying the
            # ordinary EMA immediately would retain only 30% of the very first
            # motion estimate and can severely underestimate the displacement
            # during an early camera handover.  Therefore the first available
            # velocity measurement initializes the state directly; subsequent
            # measurements use the original EMA smoothing.
            if self.hits <= 1:
                self.vx = measured_vx
                self.vy = measured_vy
            else:
                self.vx = 0.7 * self.vx + 0.3 * measured_vx
                self.vy = 0.7 * self.vy + 0.3 * measured_vy

        self.x = x
        self.y = y
        self.last_update = t

        self.hits += 1
        self.missed = 0

        self._remember_local_id(cam, local_id)

        old_has_embedding = self.embedding is not None
        embedding_updated = self._update_embedding(embedding)
        new_has_embedding = self.embedding is not None

        became_confirmed = False
        if self.hits >= self.min_hits_to_confirm and not self.confirmed:
            self.confirmed = True
            became_confirmed = True

        return {
            "dt": dt,
            "old_x": old_x,
            "old_y": old_y,
            "old_vx": old_vx,
            "old_vy": old_vy,
            "old_yaw": old_yaw,
            "new_x": self.x,
            "new_y": self.y,
            "new_vx": self.vx,
            "new_vy": self.vy,
            "new_yaw": self.yaw,
            "became_confirmed": became_confirmed,
            "old_has_embedding": bool(old_has_embedding),
            "new_has_embedding": bool(new_has_embedding),
            "embedding_updated": bool(embedding_updated),
        }

    def predicted_position_at(self, t):
        """Predict the track position with an exponentially decaying velocity.

        If v(t) = v0 * exp(-k t), the travelled displacement is the integral
        of that velocity, v0 * (1 - exp(-k*dt)) / k.  The previous
        implementation used the final decayed velocity for the whole interval
        (v0 * exp(-k*dt) * dt), which strongly underestimated motion after even
        a short observation gap.
        """
        dt = t - self.last_update

        if dt <= 0.0:
            return self.x, self.y

        k = float(self.velocity_decay_gain)
        if k <= 1e-9:
            motion_scale = dt
        else:
            motion_scale = (1.0 - np.exp(-k * dt)) / k

        return (
            self.x + self.vx * motion_scale,
            self.y + self.vy * motion_scale,
        )


class FusionNode(Node):
    def __init__(self):
        super().__init__("fusion_node")
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)

        # PARAMS
        # Parameters are declared as ROS 2 parameters so they can be tuned
        # from a launch file or from the command line

        # Dynamic association threshold.
        self.base_dist_thresh = self._declare_and_get_param("base_dist_thresh", 0.45)
        self.min_dist_thresh = self._declare_and_get_param("min_dist_thresh", 0.35)
        self.max_dist_thresh = self._declare_and_get_param("max_dist_thresh", 0.73)
        self.speed_gain = self._declare_and_get_param("speed_gain", 0.9)
        self.stale_track_extra_gate = self._declare_and_get_param("stale_track_extra_gate", 0.10)

        # Track lifecycle.
        self.max_missed = int(self._declare_and_get_param("max_missed", 8))
        self.last_detection_time = 0.0
        self.current_time = 0.0
        self.no_det_grace_time = self._declare_and_get_param("no_det_grace_time", 0.8)#0.6
        self.fusion_dt = self._declare_and_get_param("fusion_dt", 0.3)

        # Publishing.
        self.publish_max_missed = int(self._declare_and_get_param("publish_max_missed", 1))
        self.publish_max_age = self._declare_and_get_param("publish_max_age", 1.8)#1.2
        self.publish_min_reliability = self._declare_and_get_param("publish_min_reliability", 0.40)
        self.debug_publish_min_reliability = self._declare_and_get_param("debug_publish_min_reliability", 0.55)

        # Reliability model.
        self.reliability_hits_norm = self._declare_and_get_param("reliability_hits_norm", 3.0)
        self.reliability_hit_weight = self._declare_and_get_param("reliability_hit_weight", 0.6)
        self.reliability_missed_weight = self._declare_and_get_param("reliability_missed_weight", 0.2)
        self.reliability_age_weight = self._declare_and_get_param("reliability_age_weight", 0.2)

        # Marker visualization.
        self.marker_min_reliability = self._declare_and_get_param("marker_min_reliability", 0.40)
        self.marker_high_reliability = self._declare_and_get_param("marker_high_reliability", 0.70)

        # Matching / deletion.
        self.match_max_age = self._declare_and_get_param("match_max_age", 2.6)#1.8
        self.delete_max_age = self._declare_and_get_param("delete_max_age", 6.0)#5.0

        # Recovery / reactivation.
        self.reactivation_max_age = self._declare_and_get_param("reactivation_max_age", 5.0)#4.0
        self.reactivation_dist_thresh = self._declare_and_get_param("reactivation_dist_thresh", 0.50)

        # New-track blocking: after recovery fails, block the creation of a
        # duplicate global ID if an existing track is still visually/geometrically compatible.
        self.new_track_min_dist_geometry_only = self._declare_and_get_param("new_track_min_dist_geometry_only", 0.35)

        # Ambiguity guards.
        self.recovery_ambiguity_margin = self._declare_and_get_param("recovery_ambiguity_margin", 0.15)

        # Duplicate local detections threshold, only across different cameras.
        self.duplicate_dist_thresh = self._declare_and_get_param("duplicate_dist_thresh", 0.35)

        # Local-ID cue and soft safety penalties.
        self.local_id_extra_gate = self._declare_and_get_param("local_id_extra_gate", 0.15)
        self.max_conflict_match_dist = self._declare_and_get_param("max_conflict_match_dist", 0.20)
        self.max_unknown_cross_camera_match_dist = self._declare_and_get_param("max_unknown_cross_camera_match_dist", 0.30)
        self.max_jump_without_local_id_match = self._declare_and_get_param("max_jump_without_local_id_match", 0.45)

        # Track confirmation / prediction.
        self.min_hits_to_confirm = int(self._declare_and_get_param("min_hits_to_confirm", 1))
        self.velocity_decay_gain = self._declare_and_get_param("velocity_decay_gain", 1.2)

        # Re-ID / appearance association. Embeddings are already embedded in
        # pic4people_tracking_msgs/TrackedPeople messages.
        self.use_reid_features = self._declare_and_get_param("use_reid_features", True)
        self.reid_ema_alpha = self._declare_and_get_param("reid_ema_alpha", 0.85)
        self.no_embedding_cost = self._declare_and_get_param("no_embedding_cost", 0.50)

        # BoT-SORT-inspired MTMC association.
        # Geometry is kept as a candidate gate.
        # The final association cost combines 3D geometry in map, Re-ID appearance,
        # local tracker information and soft penalties for risky cases.
        self.normal_gate_scale = self._declare_and_get_param("normal_gate_scale", 1.35)
        self.normal_gate_extra = self._declare_and_get_param("normal_gate_extra", 0.15)
        self.normal_candidate_max_dist = self._declare_and_get_param("normal_candidate_max_dist", 1.25)

        self.recovery_gate_scale = self._declare_and_get_param("recovery_gate_scale", 1.80) #1.60
        self.recovery_gate_extra = self._declare_and_get_param("recovery_gate_extra", 0.25)
        self.recovery_candidate_max_dist = self._declare_and_get_param("recovery_candidate_max_dist", 2.20)

        self.min_assoc_score = self._declare_and_get_param("min_assoc_score", 0.55)
        self.min_recovery_score = self._declare_and_get_param("min_recovery_score", 0.55)#0.60

        self.assoc_geom_weight = self._declare_and_get_param("assoc_geom_weight", 0.40)
        self.assoc_reid_weight = self._declare_and_get_param("assoc_reid_weight", 0.45)
        self.assoc_local_weight = self._declare_and_get_param("assoc_local_weight", 0.10)
        self.assoc_time_weight = self._declare_and_get_param("assoc_time_weight", 0.05)

        self.no_reid_score = self._declare_and_get_param("no_reid_score", 0.50)
        self.reid_override_score = self._declare_and_get_param("reid_override_score", 0.75)

        # During stale-track recovery, a strong appearance match can modestly
        # enlarge the geometric candidate gate.  This is especially useful for
        # cross-camera handovers where the local-ID relation is "unknown" and
        # the motion model is uncertain.  The candidate must still pass the full
        # multi-cue score and ambiguity checks; Re-ID does not directly force a
        # match.
        self.strong_reid_recovery_extra_gate = self._declare_and_get_param(
            "strong_reid_recovery_extra_gate", 0.30
        )
        
        # Large jumps, conflicting or unknown local ID relations provide soft penalties. A strong Re-ID match can
        # compensate them, while weak Re-ID keeps the association unlikely.
        self.jump_penalty = self._declare_and_get_param("jump_penalty", 0.18)
        self.conflict_penalty_score = self._declare_and_get_param("conflict_penalty_score", 0.22)
        self.unknown_cross_camera_penalty = self._declare_and_get_param("unknown_cross_camera_penalty", 0.14)
        self.outside_dynamic_gate_penalty = self._declare_and_get_param("outside_dynamic_gate_penalty", 0.08)

        self.score_ambiguity_margin = self._declare_and_get_param("score_ambiguity_margin", 0.10)

        # New-track blocking, a separate duplicate-prevention stage.
        self.block_candidate_max_dist = self._declare_and_get_param("block_candidate_max_dist", 2.00)
        self.block_min_score = self._declare_and_get_param("block_min_score", 0.72)
        self.block_geom_weight = self._declare_and_get_param("block_geom_weight", 0.25)
        self.block_reid_weight = self._declare_and_get_param("block_reid_weight", 0.65)
        self.block_time_weight = self._declare_and_get_param("block_time_weight", 0.10)

        # Duplicate-aware one-to-one safeguard.
        # When a Global Track is already assigned in the current fusion cycle,
        # a second unmatched detection is NOT automatically allowed to create
        # a new Global ID. First test whether it is probably another observation
        # of the same physical person that already consumed that track.
        self.assigned_duplicate_same_camera_dist = self._declare_and_get_param(
            "assigned_duplicate_same_camera_dist", 0.50
        )
        self.assigned_duplicate_cross_camera_dist = self._declare_and_get_param(
            "assigned_duplicate_cross_camera_dist", 0.35
        )
        self.assigned_duplicate_reid_distance = self._declare_and_get_param(
            "assigned_duplicate_reid_distance", 0.20
        )

        self.output_frame = self._declare_and_get_param("output_frame", "map")

        self._validate_parameters()

        # Internal state.
        self.tracks = {}
        self.next_id = 1
        self.buffer = []
        self.lock = Lock()
        self.cycle_count = 0
        self.debug_events = []

        # ROS INTERFACES
    
        # Local camera trackers now publish TrackedPeople on the topics:
        # /people_center, /people_left, /people_right.
        # Each TrackedPerson already contains its Re-ID embedding.
        self.create_subscription(TrackedPeople, "/people_center", self.cb_center, 10)
        self.create_subscription(TrackedPeople, "/people_left", self.cb_left, 10)
        self.create_subscription(TrackedPeople, "/people_right", self.cb_right, 10)

        self.pub = self.create_publisher(People, "/tracked_people", 10)
        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/tracked_people_markers",
            10,
        )
        self.debug_pub = self.create_publisher(
            String,
            "/tracked_people_debug",
            10,
        )

        self.create_timer(self.fusion_dt, self.process)

        self.get_logger().info("======================================")
        self.get_logger().info("Global Fusion Tracker READY")
        self.get_logger().info(
            f"dynamic dist thresh: base={self.base_dist_thresh}, "
            f"min={self.min_dist_thresh}, max={self.max_dist_thresh}, "
            f"speed_gain={self.speed_gain}"
        )
        self.get_logger().info(f"max_missed = {self.max_missed}")
        self.get_logger().info(f"publish_max_missed = {self.publish_max_missed}")
        self.get_logger().info(f"publish_max_age = {self.publish_max_age}")
        self.get_logger().info(f"match_max_age = {self.match_max_age}")
        self.get_logger().info(f"delete_max_age = {self.delete_max_age}")
        self.get_logger().info(f"reactivation_max_age = {self.reactivation_max_age}")
        self.get_logger().info(f"reactivation_dist_thresh = {self.reactivation_dist_thresh}")
        self.get_logger().info(
            f"new_track_min_dist_geometry_only = {self.new_track_min_dist_geometry_only}"
        )
        self.get_logger().info(f"recovery_ambiguity_margin = {self.recovery_ambiguity_margin}")
        self.get_logger().info(f"duplicate_dist_thresh = {self.duplicate_dist_thresh}")
        self.get_logger().info(f"local_id_extra_gate = {self.local_id_extra_gate}")
        self.get_logger().info(f"max_conflict_match_dist = {self.max_conflict_match_dist}")
        self.get_logger().info(
            f"max_unknown_cross_camera_match_dist = {self.max_unknown_cross_camera_match_dist}"
        )
        self.get_logger().info(f"min_hits_to_confirm = {self.min_hits_to_confirm}")
        self.get_logger().info(
            f"max_jump_without_local_id_match = {self.max_jump_without_local_id_match}"
        )
        self.get_logger().info(f"velocity_decay_gain = {self.velocity_decay_gain}")
        self.get_logger().info(
            f"strong_reid_recovery_extra_gate = {self.strong_reid_recovery_extra_gate}"
        )
        self.get_logger().info(
            "duplicate-aware one-to-one: "
            f"same_cam_dist={self.assigned_duplicate_same_camera_dist}, "
            f"cross_cam_dist={self.assigned_duplicate_cross_camera_dist}, "
            f"reid_dist={self.assigned_duplicate_reid_distance}"
        )
        self.get_logger().info(f"use_reid_features = {self.use_reid_features}")
        self.get_logger().info(
            "Re-ID policy: embeddings are read directly from TrackedPerson messages; "
            "no separate /people_*_features topics are used"
        )
        self.get_logger().info(
            "BoT-MTMC association: "
            f"normal_gate_scale={self.normal_gate_scale}, "
            f"normal_candidate_max_dist={self.normal_candidate_max_dist}, "
            f"recovery_candidate_max_dist={self.recovery_candidate_max_dist}, "
            f"min_assoc_score={self.min_assoc_score}, "
            f"min_recovery_score={self.min_recovery_score}"
        )
        self.get_logger().info("Subscribed topics: /people_center /people_left /people_right [pic4people_tracking_msgs/TrackedPeople]")
        self.get_logger().info("Publishing topic: /tracked_people")
        self.get_logger().info("======================================")


    # Parameter management

    def _declare_and_get_param(self, name, default_value):
        """Declare a ROS 2 parameter and return its value.

        """
        self.declare_parameter(name, default_value)
        return self.get_parameter(name).value

    def _validate_parameters(self):
        """Apply simple safety checks to avoid inconsistent parameter sets."""
        if self.min_dist_thresh > self.base_dist_thresh:
            self.get_logger().warn(
                "min_dist_thresh > base_dist_thresh; clamping min_dist_thresh to base_dist_thresh"
            )
            self.min_dist_thresh = self.base_dist_thresh

        if self.base_dist_thresh > self.max_dist_thresh:
            self.get_logger().warn(
                "base_dist_thresh > max_dist_thresh; clamping max_dist_thresh to base_dist_thresh"
            )
            self.max_dist_thresh = self.base_dist_thresh

        if self.publish_max_age > self.delete_max_age:
            self.get_logger().warn(
                "publish_max_age > delete_max_age; clamping publish_max_age to delete_max_age"
            )
            self.publish_max_age = self.delete_max_age

        if self.match_max_age > self.delete_max_age:
            self.get_logger().warn(
                "match_max_age > delete_max_age; clamping match_max_age to delete_max_age"
            )
            self.match_max_age = self.delete_max_age

        if self.reactivation_max_age > self.delete_max_age:
            self.get_logger().warn(
                "reactivation_max_age > delete_max_age; clamping reactivation_max_age to delete_max_age"
            )
            self.reactivation_max_age = self.delete_max_age


        weight_sum = (
            float(self.reliability_hit_weight)
            + float(self.reliability_missed_weight)
            + float(self.reliability_age_weight)
        )
        if weight_sum <= 1e-9:
            self.get_logger().warn(
                "Reliability weights sum to zero; restoring default weights 0.6/0.2/0.2"
            )
            self.reliability_hit_weight = 0.6
            self.reliability_missed_weight = 0.2
            self.reliability_age_weight = 0.2
        elif abs(weight_sum - 1.0) > 1e-6:
            self.get_logger().warn(
                f"Reliability weights sum to {weight_sum:.3f}; normalizing them to sum to 1"
            )
            self.reliability_hit_weight /= weight_sum
            self.reliability_missed_weight /= weight_sum
            self.reliability_age_weight /= weight_sum


    # Time conversion helpers
    
    def _stamp_to_sec(self, stamp):
        """Convert a ROS 2 builtin time message to floating-point seconds."""
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _sec_to_stamp(self, t):
        """Convert floating-point seconds to a ROS 2 builtin time message."""
        sec = int(t)
        nanosec = int((t - sec) * 1e9)
        return rclpy.time.Time(seconds=sec, nanoseconds=nanosec).to_msg()


    # Subscription callbacks and input buffer management

    def cb_center(self, msg):
        self._add(msg, "center")

    def cb_left(self, msg):
        self._add(msg, "left")

    def cb_right(self, msg):
        self._add(msg, "right")

    def _add(self, msg, cam):
        """
        Receive one TrackedPeople message from a local camera tracker.

        The message is added to the input buffer for later processing in the main fusion loop. 
        """
        t = self._stamp_to_sec(msg.header.stamp)

        if t <= 0.0:
            self.get_logger().warn(
                f"[RX {cam}] invalid timestamp {t:.3f}, message skipped"
            )
            return

        self.get_logger().info(
            f"[RX {cam}] received TrackedPeople msg with {len(msg.people)} persons"
        )

        if len(msg.people) > 0:
            self.last_detection_time = t

        with self.lock:
            before = len(self.buffer)

            for idx, p in enumerate(msg.people):
                # The local tracker publishes yaw explicitly in TrackedPerson.yaw.
                # Keep a fallback to position.z only for compatibility with older logs/messages.
                yaw = None
                if hasattr(p, "yaw") and np.isfinite(p.yaw):
                    yaw = float(p.yaw)
                elif hasattr(p, "position") and np.isfinite(p.position.z):
                    yaw = float(p.position.z)

                if hasattr(p, "local_id") and p.local_id is not None:
                    local_id = str(p.local_id)
                elif hasattr(p, "name") and p.name is not None:
                    local_id = str(p.name)
                else:
                    local_id = ""

                embedding = normalize_embedding(getattr(p, "embedding", None))

                self.buffer.append({
                    "x": float(p.position.x),
                    "y": float(p.position.y),
                    "yaw": yaw,
                    "t": float(t),
                    "cam": str(getattr(p, "camera", "")) if str(getattr(p, "camera", "")) != "" else cam,
                    "local_id": local_id,
                    "embedding": embedding,
                    "embedding_stamp": float(t) if embedding is not None else None,
                    "embedding_dt": 0.0 if embedding is not None else None,
                    "reliability": float(getattr(p, "reliability", np.nan)),
                })

                self.get_logger().info(
                    f"[RX {cam}]   person[{idx}] local_id={local_id} "
                    f"pos=({p.position.x:.3f}, {p.position.y:.3f}) "
                    f"has_emb={embedding is not None}"
                )

                self._add_debug_event(
                    "EMBEDDED_REID_RECEIVED" if embedding is not None else "EMBEDDED_REID_MISSING",
                    camera=cam,
                    local_id=local_id,
                    det_x=float(p.position.x),
                    det_y=float(p.position.y),
                    stamp=float(t),
                    embedding_dim=0 if embedding is None else int(len(embedding)),
                )

            after = len(self.buffer)

        self.get_logger().info(f"[RX {cam}] buffer size: {before} -> {after}")


    # Re-ID feature helpers and multi-cue scoring

    def _appearance_distance(self, tr, d):
        """Return cosine appearance distance for a track-detection pair when available."""
        if not self.use_reid_features:
            return None
        if not hasattr(tr, "embedding") or tr.embedding is None:
            return None
        return cosine_distance(tr.embedding, d.get("embedding", None))

    def _clamp01(self, value):
        """Clamp a numeric value to the closed interval ``[0, 1]``."""
        return max(0.0, min(1.0, float(value)))

    def _should_use_reid_for_normal_matching(self, tr, d, local_relation, dist=None):
        """
        In the BoT-MTMC like association, appearance is a first-level cue.

        It is used whenever both the global track and the incoming detection have
        an embedding. It is not a hard decision by itself: it contributes to the
        association score together with geometry, local-ID consistency and time.
        """
        if not self.use_reid_features:
            return False
        if d.get("embedding", None) is None:
            return False
        if not hasattr(tr, "embedding") or tr.embedding is None:
            return False
        return True

    def _association_cost_with_appearance(
        self,
        tr,
        d,
        dist,
        allowed_dist,
        recovery_mode=False,
    ):
        """
        BoT-SORT-inspired multi-cue association cost.

        The dynamic/allowed distance is the candidate gate and the geometry
        normalization term. The final cost combines:
        - 3D geometry in the map frame;
        - Re-ID appearance similarity;
        - local-ID relation from StrongSORT;
        - temporal freshness of the global track;
        - soft penalties for risky jumps/conflicts/cross-camera unknown links.
        """
        relation = self._local_id_relation(tr, d)
        allowed_dist = max(1e-6, float(allowed_dist))
        dist = float(dist)

        geom_score = self._clamp01(1.0 - dist / allowed_dist)
        geo_cost = 1.0 - geom_score

        appearance_dist = None
        has_appearance = False
        use_appearance = self._should_use_reid_for_normal_matching(tr, d, relation, dist=dist)

        if use_appearance:
            appearance_dist = self._appearance_distance(tr, d)
            has_appearance = appearance_dist is not None

        if appearance_dist is None:
            appearance_for_log = float(self.no_embedding_cost)
            reid_score = float(self.no_reid_score)
        else:
            appearance_for_log = float(appearance_dist)
            reid_score = self._clamp01(1.0 - appearance_for_log)

        if relation == "match":
            local_score = 1.0
        elif relation == "unknown":
            local_score = 0.55
        elif relation == "none":
            local_score = 0.45
        else:  # conflict
            local_score = 0.10

        age = max(0.0, float(self.current_time) - float(tr.last_update))
        time_norm = max(1e-6, float(self.reactivation_max_age if recovery_mode else self.match_max_age))
        time_score = self._clamp01(1.0 - age / time_norm)

        w_sum = (
            float(self.assoc_geom_weight)
            + float(self.assoc_reid_weight)
            + float(self.assoc_local_weight)
            + float(self.assoc_time_weight)
        )
        if w_sum <= 1e-9:
            w_geom, w_reid, w_local, w_time = 0.40, 0.45, 0.10, 0.05
        else:
            w_geom = float(self.assoc_geom_weight) / w_sum
            w_reid = float(self.assoc_reid_weight) / w_sum
            w_local = float(self.assoc_local_weight) / w_sum
            w_time = float(self.assoc_time_weight) / w_sum

        raw_score = (
            w_geom * geom_score
            + w_reid * reid_score
            + w_local * local_score
            + w_time * time_score
        )

        dyn_thresh = float(self._dynamic_dist_thresh(tr, d.get("t", self.current_time), relation))
        reid_can_override = bool(has_appearance and reid_score >= float(self.reid_override_score))

        penalty = 0.0
        if relation != "match" and dist > float(self.max_jump_without_local_id_match) and not reid_can_override:
            penalty += float(self.jump_penalty)
        if relation == "conflict" and dist > float(self.max_conflict_match_dist) and not reid_can_override:
            penalty += float(self.conflict_penalty_score)
        if relation == "unknown":
            cam = str(d.get("cam", ""))
            local_id = str(d.get("local_id", ""))
            known_other_cameras = [known_cam for known_cam in tr.local_ids_by_cam.keys() if str(known_cam) != cam]
            if cam != "" and local_id != "" and len(known_other_cameras) > 0:
                if dist > float(self.max_unknown_cross_camera_match_dist) and not reid_can_override:
                    penalty += float(self.unknown_cross_camera_penalty)
        if dist > dyn_thresh and not reid_can_override:
            penalty += float(self.outside_dynamic_gate_penalty)

        score = self._clamp01(raw_score - penalty)
        min_score = float(self.min_recovery_score if recovery_mode else self.min_assoc_score)

        if score < min_score:
            cost_value = 1e6
        else:
            cost_value = 1.0 - score

        return (
            float(cost_value),
            relation,
            float(geo_cost),
            float(appearance_for_log),
            bool(has_appearance),
            bool(use_appearance),
        )

    def _add_debug_event(self, event_type, **kwargs):
        ev = {
            "event_type": event_type,
            "time": float(self.current_time),
            "cycle": int(self.cycle_count),
        }

        for k, v in kwargs.items():
            if isinstance(v, np.integer):
                v = int(v)
            elif isinstance(v, np.floating):
                v = float(v)
            elif isinstance(v, np.ndarray):
                v = v.tolist()

            ev[k] = v

        self.debug_events.append(ev)

    def publish_debug(self, event, detections=None, extra=None):
        if detections is None:
            detections = []
        if extra is None:
            extra = {}

        now = self.current_time
        published_ids = []

        data = {
            "time": float(now),
            "cycle": int(self.cycle_count),
            "event": event,
            "num_detections": int(len(detections)),
            "published_ids": published_ids,
            "debug_events": self.debug_events,
            "detections": [],
            "tracks": [],
            "extra": extra,
        }

        for d in detections:
            yaw = d.get("yaw", None)
            data["detections"].append({
                "camera": d.get("cam", ""),
                "local_id": str(d.get("local_id", "")),
                "x": float(d.get("x", 0.0)),
                "y": float(d.get("y", 0.0)),
                "yaw": None if yaw is None else float(yaw),
                "stamp": float(d.get("t", 0.0)),
                "has_embedding": d.get("embedding", None) is not None,
                "embedding_dt": None if d.get("embedding_dt", None) is None else float(d.get("embedding_dt")),
            })

        for tid, tr in self.tracks.items():
            age = now - tr.last_update
            reliability = self._track_reliability(tr, age)

            publishable = (
                tr.confirmed
                and tr.missed <= self.publish_max_missed
                and age <= self.publish_max_age
                and reliability >= self.debug_publish_min_reliability
            )

            if publishable:
                published_ids.append(int(tid))

            data["tracks"].append({
                "global_id": int(tid),
                "x": float(tr.x),
                "y": float(tr.y),
                "yaw": float(tr.yaw),
                "vx": float(tr.vx),
                "vy": float(tr.vy),
                "hits": int(tr.hits),
                "missed": int(tr.missed),
                "confirmed": bool(tr.confirmed),
                "age": float(age),
                "last_update": float(tr.last_update),
                "reliability": float(reliability),
                "publishable": bool(publishable),
                "local_ids_by_cam": tr.local_ids_by_cam,
                "has_embedding": bool(getattr(tr, "embedding", None) is not None),
                "embedding_dim": 0 if getattr(tr, "embedding", None) is None else int(len(tr.embedding)),
            })

        msg = String()
        msg.data = json.dumps(data)
        self.debug_pub.publish(msg)

  
    # Pre-filtering operations
    
    # These filters are applied before the association stage. 
    # They are designed to remove detections that are likely to cause problems during association, 
    # such as duplicates or old detections from the same camera. 
    # They also generate debug events to help understand why certain detections were dropped.
    def _keep_latest_message_per_camera(self, detections):
        """Keep only detections from the latest message per camera, dropping older detections from the same camera in the same fusion cycle."""
        if len(detections) <= 1:
            return detections

        latest_stamp_by_cam = {}
        for d in detections:
            cam = d["cam"]
            latest_stamp_by_cam[cam] = max(
                latest_stamp_by_cam.get(cam, d["t"]),
                d["t"],
            )

        kept = []
        dropped = []
        eps = 1e-6

        for d in detections:
            cam = d["cam"]

            if abs(d["t"] - latest_stamp_by_cam[cam]) <= eps:
                kept.append(d)
            else:
                dropped.append(d)

        for d in dropped:
            cam = d["cam"]
            self._add_debug_event(
                "DROP_OLD_SAME_CAMERA_DETECTION",
                camera=cam,
                local_id=str(d.get("local_id", "")),
                det_x=float(d["x"]),
                det_y=float(d["y"]),
                stamp=float(d["t"]),
                latest_stamp=float(latest_stamp_by_cam[cam]),
            )

        if len(dropped) > 0:
            self.get_logger().warn(
                f"[LATEST MSG FILTER] dropped {len(dropped)} old detections "
                f"from previous messages in the same fusion cycle"
            )

        return kept

    def _filter_duplicate_detections(self, detections):
        """Drop spatially overlapping detections reported by different cameras"""
        if len(detections) <= 1:
            return detections

        kept = []

        for d in detections:
            is_duplicate = False

            for k in kept:
                if d["cam"] == k["cam"]:
                    continue

                dist = np.hypot(d["x"] - k["x"], d["y"] - k["y"])

                if dist < self.duplicate_dist_thresh:
                    self._add_debug_event(
                        "DUPLICATE_DROP",
                        dropped_cam=d["cam"],
                        dropped_local_id=str(d.get("local_id", "")),
                        dropped_x=float(d["x"]),
                        dropped_y=float(d["y"]),
                        kept_cam=k["cam"],
                        kept_local_id=str(k.get("local_id", "")),
                        kept_x=float(k["x"]),
                        kept_y=float(k["y"]),
                        distance=float(dist),
                        threshold=float(self.duplicate_dist_thresh),
                    )

                    self.get_logger().warn(
                        f"[DUP DET DROP] {d['cam']} det at "
                        f"({d['x']:.3f}, {d['y']:.3f}) too close to "
                        f"kept det from {k['cam']} "
                        f"({k['x']:.3f}, {k['y']:.3f}), dist={dist:.3f}"
                    )

                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(d)

        return kept


    # ASSOCIATION STAGE - Multi-cue scoring and candidate selection

  
    def _local_id_relation(self, tr, d):
        """Classify local-ID consistency as ``match``, ``conflict``, ``unknown``, or ``none``."""
        cam = d.get("cam", "")
        local_id = str(d.get("local_id", ""))

        if local_id == "":
            return "none"

        known_id = tr.local_ids_by_cam.get(cam, "")

        if known_id == "":
            return "unknown"

        if known_id == local_id:
            return "match"

        return "conflict"

    def _dynamic_dist_thresh(self, tr, t, local_relation="none"):
        """
        Compute a dynamic distance threshold for a track-detection pair based on the track's speed and age, and the local-ID relation."""
        age = max(0.0, t - tr.last_update)
        speed = np.hypot(tr.vx, tr.vy)

        thresh = self.base_dist_thresh + self.speed_gain * speed * age

        if age > self.publish_max_age:
            thresh += self.stale_track_extra_gate

        thresh = max(self.min_dist_thresh, min(self.max_dist_thresh, thresh))

        if local_relation == "match":
            thresh = min(
                self.max_dist_thresh + self.local_id_extra_gate,
                thresh + self.local_id_extra_gate,
            )

        return thresh

    def _allowed_association_distance(self, tr, d, dynamic_threshold, local_relation):
        """
        Candidate gate used before Hungarian assignment.

        This is not the final identity decision. The final decision is taken by the
        multi-cue cost. This keeps the BoT-SORT idea: motion/geometry selects
        plausible candidates, then appearance and other cues rank them.
        """
        gate = float(dynamic_threshold) * float(self.normal_gate_scale) + float(self.normal_gate_extra)

        if local_relation == "match":
            gate += float(self.local_id_extra_gate)

        # Do not allow unbounded jumps during normal matching.
        gate = min(float(self.normal_candidate_max_dist), gate)
        gate = max(float(self.min_dist_thresh), gate)
        return gate

    def _get_matchable_track_ids(self, now):
        """ Return a list of global track IDs that are still young enough to be considered for normal matching."""
        matchable_ids = []

        for tid, tr in self.tracks.items():
            age = now - tr.last_update

            if age > self.match_max_age:
                self.get_logger().info(
                    f"[MATCH-SKIP] T{tid} too old for normal matching "
                    f"(age={age:.3f}s > {self.match_max_age:.3f}s)"
                )
                continue

            matchable_ids.append(tid)

        return matchable_ids

    def _try_recover_existing_track(self, d, now, assigned_tracks=None):
        """
        Reactivate a stale global track using the same multi-cue association cost
        used by normal matching, but with a slightly wider candidate gate.

        Recovery is applied mainly to tracks outside normal matching age but still
        younger than reactivation_max_age. This keeps the BoT-SORT lifecycle while
        allowing MTMC re-entry based on appearance and motion consistency.
        """
        if assigned_tracks is None:
            assigned_tracks = set()

        candidates = []
        BIG_COST = 1e6

        for tid, tr in self.tracks.items():
            if tid in assigned_tracks:
                continue

            age = float(now) - float(tr.last_update)
            if age < 0.0 or age > float(self.reactivation_max_age):
                continue

            relation = self._local_id_relation(tr, d)

            # For stale tracks, use the last reliable state rather than an overly
            # aggressive velocity prediction. For still-fresh tracks, prediction is OK.
            if age <= float(self.match_max_age):
                pred_x, pred_y = tr.predicted_position_at(d["t"])
            else:
                pred_x, pred_y = tr.x, tr.y

            dist = float(np.hypot(pred_x - d["x"], pred_y - d["y"]))
            dyn_thresh = float(self._dynamic_dist_thresh(tr, d["t"], relation))

            recovery_gate = dyn_thresh * float(self.recovery_gate_scale) + float(self.recovery_gate_extra)
            if relation == "match":
                recovery_gate += float(self.local_id_extra_gate)

            # Strong Re-ID is allowed to widen the recovery candidate gate only
            # modestly and only when the local-ID evidence is not contradictory.
            # This does not accept the association by itself: the pair must
            # still pass _association_cost_with_appearance(), min_recovery_score
            # and the recovery ambiguity test.
            appearance_for_gate = self._appearance_distance(tr, d)
            strong_reid_gate_extension = False
            if appearance_for_gate is not None:
                reid_score_for_gate = self._clamp01(1.0 - appearance_for_gate)
                strong_reid_gate_extension = bool(
                    reid_score_for_gate >= float(self.reid_override_score)
                    and relation in ("match", "unknown")
                )
                if strong_reid_gate_extension:
                    recovery_gate += float(self.strong_reid_recovery_extra_gate)

            recovery_gate = min(float(self.recovery_candidate_max_dist), recovery_gate)
            recovery_gate = max(float(self.reactivation_dist_thresh), recovery_gate)

            (
                cost_value,
                relation,
                geo_cost,
                appearance_dist,
                has_appearance,
                use_appearance,
            ) = self._association_cost_with_appearance(
                tr,
                d,
                dist,
                recovery_gate,
                recovery_mode=True,
            )

            score = 1.0 - cost_value if cost_value < BIG_COST else -1.0

            self._add_debug_event(
                "RECOVER_CANDIDATE_CHECK",
                global_id=int(tid),
                camera=d["cam"],
                local_id=str(d.get("local_id", "")),
                local_id_relation=str(relation),
                det_x=float(d["x"]),
                det_y=float(d["y"]),
                pred_x=float(pred_x),
                pred_y=float(pred_y),
                distance=float(dist),
                recovery_gate=float(recovery_gate),
                dynamic_threshold=float(dyn_thresh),
                strong_reid_gate_extension=bool(strong_reid_gate_extension),
                strong_reid_recovery_extra_gate=(
                    float(self.strong_reid_recovery_extra_gate)
                    if strong_reid_gate_extension else 0.0
                ),
                score=float(score),
                min_score=float(self.min_recovery_score),
                association_cost=float(cost_value),
                geo_cost=float(geo_cost),
                appearance_distance=float(appearance_dist),
                has_appearance=bool(has_appearance),
                use_appearance=bool(use_appearance),
                valid_candidate=bool(cost_value < BIG_COST),
            )

            if cost_value >= BIG_COST:
                continue
            if dist > recovery_gate:
                continue

            candidates.append({
                "tid": tid,
                "track": tr,
                "dist": dist,
                "cost": cost_value,
                "score": score,
                "pred_x": pred_x,
                "pred_y": pred_y,
                "allowed_dist": recovery_gate,
                "strict_allowed_dist": dyn_thresh,
                "relation": relation,
                "appearance_dist": appearance_dist,
                "appearance_cost": appearance_dist,
                "has_appearance": has_appearance,
                "use_appearance": use_appearance,
                "geo_cost": geo_cost,
                "strong_reid_gate_extension": strong_reid_gate_extension,
            })

        if len(candidates) == 0:
            self._add_debug_event(
                "RECOVER_FAILED_NO_VALID_CANDIDATE",
                camera=d["cam"],
                local_id=str(d.get("local_id", "")),
                det_x=float(d["x"]),
                det_y=float(d["y"]),
                has_embedding=bool(d.get("embedding", None) is not None),
                reason="no_candidate_passed_multicue_recovery_score",
            )
            self.get_logger().warn(
                f"[RECOVER-FAILED] detection from {d['cam']} "
                f"local_id={d.get('local_id', '')} at "
                f"({d['x']:.3f}, {d['y']:.3f}) could not recover any existing track"
            )
            return False, None

        candidates.sort(key=lambda c: c["cost"])
        best = candidates[0]

        if len(candidates) >= 2:
            second = candidates[1]
            if best["score"] - second["score"] < float(self.recovery_ambiguity_margin):
                self._add_debug_event(
                    "RECOVER_REJECTED_AMBIGUOUS",
                    best_global_id=int(best["tid"]),
                    second_global_id=int(second["tid"]),
                    camera=d["cam"],
                    local_id=str(d.get("local_id", "")),
                    best_score=float(best["score"]),
                    second_score=float(second["score"]),
                    best_distance=float(best["dist"]),
                    second_distance=float(second["dist"]),
                    best_appearance=float(best["appearance_dist"]),
                    second_appearance=float(second["appearance_dist"]),
                    margin=float(self.recovery_ambiguity_margin),
                )
                self.get_logger().warn(
                    f"[RECOVER-REJECT-AMBIGUOUS] det from {d['cam']} "
                    f"best T{best['tid']} score={best['score']:.3f}, "
                    f"second T{second['tid']} score={second['score']:.3f}"
                )
                return False, "ambiguous"

        best_tid = best["tid"]
        update_info = self.tracks[best_tid].update(
            d["x"],
            d["y"],
            d.get("yaw", None),
            d["t"],
            cam=d.get("cam", None),
            local_id=d.get("local_id", ""),
            embedding=d.get("embedding", None),
        )

        self._add_debug_event(
            "RECOVER_EXISTING_TRACK_MULTICUE",
            global_id=int(best_tid),
            camera=d["cam"],
            local_id=str(d.get("local_id", "")),
            local_id_relation=str(best["relation"]),
            pred_x=float(best["pred_x"]),
            pred_y=float(best["pred_y"]),
            det_x=float(d["x"]),
            det_y=float(d["y"]),
            distance=float(best["dist"]),
            recovery_gate=float(best["allowed_dist"]),
            dynamic_threshold=float(best["strict_allowed_dist"]),
            score=float(best["score"]),
            association_cost=float(best["cost"]),
            appearance_distance=float(best["appearance_dist"]),
            geo_cost=float(best["geo_cost"]),
            strong_reid_gate_extension=bool(best.get("strong_reid_gate_extension", False)),
            has_appearance=bool(best["has_appearance"]),
            use_appearance=bool(best["use_appearance"]),
            embedding_updated=bool(update_info.get("embedding_updated", False)),
            old_x=float(update_info["old_x"]),
            old_y=float(update_info["old_y"]),
            new_x=float(update_info["new_x"]),
            new_y=float(update_info["new_y"]),
            hits=int(self.tracks[best_tid].hits),
            missed=int(self.tracks[best_tid].missed),
        )

        self.get_logger().warn(
            f"[RECOVER-MULTICUE] T{best_tid} recovered from {d['cam']} "
            f"local_id={d.get('local_id', '')} "
            f"score={best['score']:.3f} dist={best['dist']:.3f} "
            f"gate={best['allowed_dist']:.3f} app={best['appearance_dist']:.3f} "
            f"relation={best['relation']}"
        )

        return True, best_tid

    def _is_probable_duplicate_of_assigned_detection(self, d, assigned_detection):
        """
        Return True when ``d`` is probably a duplicate observation of the same
        physical person that already consumed a Global Track in the current
        fusion cycle.

        The test intentionally combines:
        - spatial proximity;
        - same-camera vs cross-camera geometry;
        - Re-ID appearance similarity when available.

        Same-camera duplicates need a slightly wider spatial tolerance because
        local trackers can output two overlapping tracks for the same person.
        Cross-camera duplicates use a tighter threshold because cross-camera
        duplicate suppression is already expected to be spatially strict.
        """
        if assigned_detection is None:
            return False, {}

        dx = float(d["x"]) - float(assigned_detection["x"])
        dy = float(d["y"]) - float(assigned_detection["y"])
        dist = float(np.hypot(dx, dy))

        same_camera = str(d.get("cam", "")) == str(assigned_detection.get("cam", ""))
        spatial_thresh = (
            float(self.assigned_duplicate_same_camera_dist)
            if same_camera
            else float(self.assigned_duplicate_cross_camera_dist)
        )

        emb_a = d.get("embedding", None)
        emb_b = assigned_detection.get("embedding", None)
        appearance_distance = cosine_distance(emb_a, emb_b)
        has_appearance = appearance_distance is not None

        spatial_close = dist <= spatial_thresh
        appearance_close = (
            has_appearance
            and float(appearance_distance) <= float(self.assigned_duplicate_reid_distance)
        )

        # Require geometry plus appearance when appearance is available.
        # If appearance is missing, geometry alone is not enough to suppress a
        # same-camera unmatched detection, because that could hide a true second
        # nearby person.
        if has_appearance:
            probable_duplicate = bool(spatial_close and appearance_close)
        else:
            probable_duplicate = bool((not same_camera) and spatial_close)

        return probable_duplicate, {
            "distance": dist,
            "same_camera": same_camera,
            "spatial_threshold": spatial_thresh,
            "has_appearance": bool(has_appearance),
            "appearance_distance": None if appearance_distance is None else float(appearance_distance),
            "appearance_threshold": float(self.assigned_duplicate_reid_distance),
        }

    def _is_near_existing_track(
        self,
        d,
        now,
        unavailable_track_ids=None,
        assigned_detection_by_track=None,
    ):
        """
        Prevent duplicate Global-ID creation after recovery fails.

        One-to-one safeguard with duplicate-awareness
        ---------------------------------------------
        If a track has already been assigned to another detection in the current
        fusion cycle, that track cannot be used as an ordinary blocker for a
        second distinct person.

        However, before simply skipping the already-assigned track, compare the
        unmatched detection with the detection that consumed it. If the two
        detections are probably duplicate observations of the same physical
        person, suppress the unmatched duplicate instead of creating another
        Global ID.
        """
        if unavailable_track_ids is None:
            unavailable_track_ids = set()
        if assigned_detection_by_track is None:
            assigned_detection_by_track = {}

        best = None

        for tid, tr in self.tracks.items():
            if tid in unavailable_track_ids:
                assigned_detection = assigned_detection_by_track.get(tid)
                is_duplicate, duplicate_info = self._is_probable_duplicate_of_assigned_detection(
                    d,
                    assigned_detection,
                )

                if is_duplicate:
                    self._add_debug_event(
                        "NEW_TRACK_SUPPRESSED_DUPLICATE_OF_ASSIGNED",
                        nearby_global_id=int(tid),
                        camera=d["cam"],
                        local_id=str(d.get("local_id", "")),
                        det_x=float(d["x"]),
                        det_y=float(d["y"]),
                        assigned_camera="" if assigned_detection is None else str(assigned_detection.get("cam", "")),
                        assigned_local_id="" if assigned_detection is None else str(assigned_detection.get("local_id", "")),
                        assigned_det_x=None if assigned_detection is None else float(assigned_detection["x"]),
                        assigned_det_y=None if assigned_detection is None else float(assigned_detection["y"]),
                        distance=float(duplicate_info.get("distance", np.nan)),
                        same_camera=bool(duplicate_info.get("same_camera", False)),
                        spatial_threshold=float(duplicate_info.get("spatial_threshold", np.nan)),
                        has_appearance=bool(duplicate_info.get("has_appearance", False)),
                        appearance_distance=duplicate_info.get("appearance_distance", None),
                        appearance_threshold=float(duplicate_info.get("appearance_threshold", np.nan)),
                        reason="probable_duplicate_of_detection_already_assigned_to_track",
                    )
                    return True

                self._add_debug_event(
                    "NEW_TRACK_BLOCK_SKIP_ALREADY_ASSIGNED",
                    nearby_global_id=int(tid),
                    camera=d["cam"],
                    local_id=str(d.get("local_id", "")),
                    det_x=float(d["x"]),
                    det_y=float(d["y"]),
                    reason="track_already_assigned_in_current_cycle_and_detection_is_distinct",
                )
                continue
            if tid in unavailable_track_ids:
                self._add_debug_event(
                    "NEW_TRACK_BLOCK_SKIP_ALREADY_ASSIGNED",
                    nearby_global_id=int(tid),
                    camera=d["cam"],
                    local_id=str(d.get("local_id", "")),
                    det_x=float(d["x"]),
                    det_y=float(d["y"]),
                    reason="track_already_assigned_in_current_cycle",
                )
                continue
            age = float(now) - float(tr.last_update)
            if age < 0.0 or age > float(self.reactivation_max_age):
                continue

            if age <= float(self.match_max_age):
                px, py = tr.predicted_position_at(d["t"])
            else:
                px, py = tr.x, tr.y

            dist = float(np.hypot(px - d["x"], py - d["y"]))
            if dist > float(self.block_candidate_max_dist):
                continue

            appearance_dist = self._appearance_distance(tr, d)
            has_appearance = appearance_dist is not None
            if appearance_dist is None:
                reid_score = float(self.no_reid_score)
                appearance_for_log = float(self.no_embedding_cost)
            else:
                appearance_for_log = float(appearance_dist)
                reid_score = self._clamp01(1.0 - appearance_for_log)

            geom_score = self._clamp01(1.0 - dist / max(1e-6, float(self.block_candidate_max_dist)))
            time_score = self._clamp01(1.0 - age / max(1e-6, float(self.reactivation_max_age)))

            w_sum = float(self.block_geom_weight) + float(self.block_reid_weight) + float(self.block_time_weight)
            if w_sum <= 1e-9:
                w_geom, w_reid, w_time = 0.25, 0.65, 0.10
            else:
                w_geom = float(self.block_geom_weight) / w_sum
                w_reid = float(self.block_reid_weight) / w_sum
                w_time = float(self.block_time_weight) / w_sum

            score = w_geom * geom_score + w_reid * reid_score + w_time * time_score

            if dist < float(self.new_track_min_dist_geometry_only):
                score = max(score, float(self.block_min_score))

            self._add_debug_event(
                "NEW_TRACK_BLOCK_CANDIDATE",
                nearby_global_id=int(tid),
                camera=d["cam"],
                local_id=str(d.get("local_id", "")),
                det_x=float(d["x"]),
                det_y=float(d["y"]),
                track_x=float(px),
                track_y=float(py),
                distance=float(dist),
                score=float(score),
                min_score=float(self.block_min_score),
                has_appearance=bool(has_appearance),
                appearance_distance=float(appearance_for_log),
                reid_score=float(reid_score),
                geom_score=float(geom_score),
                time_score=float(time_score),
            )

            if best is None or score > best["score"]:
                best = {
                    "tid": tid,
                    "score": float(score),
                    "dist": float(dist),
                    "appearance_dist": float(appearance_for_log),
                    "has_appearance": bool(has_appearance),
                }

        if best is not None and best["score"] >= float(self.block_min_score):
            self._add_debug_event(
                "NEW_TRACK_BLOCKED_NEAR_EXISTING",
                nearby_global_id=int(best["tid"]),
                camera=d["cam"],
                local_id=str(d.get("local_id", "")),
                det_x=float(d["x"]),
                det_y=float(d["y"]),
                distance=float(best["dist"]),
                score=float(best["score"]),
                threshold=float(self.block_min_score),
                appearance_distance=float(best["appearance_dist"]),
                has_appearance=bool(best["has_appearance"]),
                reason="multicue_duplicate_block",
            )
            self.get_logger().warn(
                f"[NEW TRACK BLOCKED] detection from {d['cam']} "
                f"local_id={d.get('local_id', '')} near T{best['tid']}: "
                f"score={best['score']:.3f}, dist={best['dist']:.3f}, "
                f"app={best['appearance_dist']:.3f}"
            )
            return True

        return False

    def _is_ambiguous_match(self, i, j, cost, real_dist):
        """
        Check if the association of track i with detection j is ambiguous.
        Ambiguity is defined as the difference between the best score and the second-best score being smaller than a threshold."""
        BIG_COST = 1e6
        chosen_cost = float(cost[i, j])

        row_costs = [
            float(cost[i, jj])
            for jj in range(cost.shape[1])
            if jj != j and cost[i, jj] < BIG_COST
        ]
        col_costs = [
            float(cost[ii, j])
            for ii in range(cost.shape[0])
            if ii != i and cost[ii, j] < BIG_COST
        ]

        competitors = row_costs + col_costs
        if len(competitors) == 0:
            return False, ""

        second_best_cost = min(competitors)
        chosen_score = 1.0 - chosen_cost
        second_score = 1.0 - second_best_cost

        if chosen_score - second_score < float(self.score_ambiguity_margin):
            return True, "similar_multicue_score"

        return False, ""

    def _create(self, d, t):
        """Create a new global track from an unmatched detection."""
        tid = self.next_id
        self.next_id += 1

        self.tracks[tid] = GlobalTrack(
            tid,
            d["x"],
            d["y"],
            d.get("yaw", 0.0),
            t,
            cam=d.get("cam", None),
            local_id=d.get("local_id", ""),
            min_hits_to_confirm=self.min_hits_to_confirm,
            velocity_decay_gain=self.velocity_decay_gain,
            embedding=d.get("embedding", None),
            reid_ema_alpha=self.reid_ema_alpha,
        )

        self._add_debug_event(
            "NEW_TRACK",
            global_id=int(tid),
            camera=d["cam"],
            local_id=str(d.get("local_id", "")),
            x=float(d["x"]),
            y=float(d["y"]),
            yaw=None if d.get("yaw", None) is None else float(d.get("yaw")),
            has_embedding=d.get("embedding", None) is not None,
        )

        self.get_logger().warn(
            f"[NEW TRACK] T{tid} created from {d['cam']} "
            f"local_id={d.get('local_id', '')} "
            f"at ({d['x']:.3f}, {d['y']:.3f})"
        )

    def _prune_old_tracks(self, now):
        """Delete global tracks that have not been updated for a long time."""
        for tid, tr in list(self.tracks.items()):
            age = now - tr.last_update

            if age > self.delete_max_age:
                self._add_debug_event(
                    "DELETE_TRACK",
                    global_id=int(tid),
                    x=float(tr.x),
                    y=float(tr.y),
                    missed=int(tr.missed),
                    age=float(age),
                    reason="delete_max_age",
                )

                self.get_logger().warn(
                    f"[DELETE-OLD] T{tid} removed by age "
                    f"(age={age:.3f}s > {self.delete_max_age:.3f}s)"
                )

                del self.tracks[tid]

    def _handle_missed(self, now):
        for tid, tr in list(self.tracks.items()):
            age = now - tr.last_update

            if age <= self.match_max_age:
                self.get_logger().info(
                    f"[MISSED-SKIP] T{tid} still fresh "
                    f"(age={age:.3f}s <= {self.match_max_age:.3f}s)"
                )
                continue

            tr.missed += 1

            self._add_debug_event(
                "MISSED_NO_DETECTION",
                global_id=int(tid),
                x=float(tr.x),
                y=float(tr.y),
                missed=int(tr.missed),
                age=float(age),
                reason="no_detections",
            )

            self.get_logger().warn(
                f"[MISSED-NO-DET] T{tid}: missed={tr.missed}/{self.max_missed} "
                f"pos=({tr.x:.3f}, {tr.y:.3f}) age={age:.3f}s confirmed={tr.confirmed}"
            )

            if tr.missed > self.max_missed:
                self._add_debug_event(
                    "DELETE_TRACK",
                    global_id=int(tid),
                    x=float(tr.x),
                    y=float(tr.y),
                    missed=int(tr.missed),
                    age=float(age),
                    reason="max_missed_no_detections",
                )

                self.get_logger().error(
                    f"[DELETE-NO-DET] T{tid} removed after missed={tr.missed}"
                )
                del self.tracks[tid]

    
    # MAIN PROCESS LOOP
    def process(self):
        """ 
        Main processing loop for the fusion node. 
        This method is called periodically to process the buffered detections, 
        to perform Hungarian multi-cue assignment, to try stale-track recovery,
        to update tracks, and to publish results."""
        self.cycle_count += 1
        self.debug_events = []

        with self.lock:
            detections = self.buffer
            self.buffer = []

        if len(detections) > 0:
            det_now = max(d["t"] for d in detections)
            if self.current_time <= 0.0:
                self.current_time = det_now
            else:
                self.current_time = max(self.current_time, det_now)
        else:
            if self.current_time <= 0.0:
                self.get_logger().warn(
                    "[NO DETECTIONS] No detections yet and fusion time not initialized"
                )
                self.publish_debug(
                    event="NO_DETECTIONS_TIME_NOT_INITIALIZED",
                    detections=[],
                    extra={"missed_updated": False},
                )
                self.publish()
                return

            self.current_time += self.fusion_dt

        now = self.current_time
        self._prune_old_tracks(now)

        raw_count = len(detections)
        detections = self._keep_latest_message_per_camera(detections)
        latest_msg_count = len(detections)
        detections = self._filter_duplicate_detections(detections)
        filtered_count = len(detections)
        # Re-ID embeddings are already embedded in each detection message.
        # No feature-buffer pruning or temporal feature attachment is required.

        if latest_msg_count != raw_count:
            self.get_logger().warn(
                f"[LATEST MSG FILTER] detections filtered: {raw_count} -> {latest_msg_count}"
            )

        if filtered_count != latest_msg_count:
            self.get_logger().warn(
                f"[DUP FILTER] detections filtered: {latest_msg_count} -> {filtered_count}"
            )

        self.get_logger().info("")
        self.get_logger().info("============== FUSION CYCLE ==============")
        self.get_logger().info(f"cycle: {self.cycle_count}")
        self.get_logger().info(f"raw detections in buffer: {raw_count}")
        self.get_logger().info(f"after latest-message-per-camera filter: {latest_msg_count}")
        self.get_logger().info(f"after duplicate filter: {filtered_count}")
        self.get_logger().info(f"active tracks: {len(self.tracks)}")
        self.get_logger().info(f"current_time: {self.current_time:.3f}")

        if len(detections) == 0:
            dt_since_last_det = now - self.last_detection_time

            self.get_logger().warn(
                f"[NO DETECTIONS] Buffer empty this cycle. "
                f"Time since last real detection: {dt_since_last_det:.3f}s"
            )

            if dt_since_last_det > self.no_det_grace_time:
                self.get_logger().warn(
                    f"[NO DETECTIONS] Exceeded grace time "
                    f"({self.no_det_grace_time:.3f}s) -> handle missed tracks"
                )
                self._handle_missed(now)
                missed_updated = True
                event_name = "NO_DETECTIONS_MISSED_UPDATE"
            else:
                self.get_logger().info(
                    f"[NO DETECTIONS] Within grace time "
                    f"({self.no_det_grace_time:.3f}s) -> skip missed increment"
                )
                missed_updated = False
                event_name = "NO_DETECTIONS_WITHIN_GRACE_TIME"

            self.publish_debug(
                event=event_name,
                detections=[],
                extra={
                    "dt_since_last_detection": float(dt_since_last_det),
                    "no_det_grace_time": float(self.no_det_grace_time),
                    "missed_updated": missed_updated,
                    "raw_detections": int(raw_count),
                    "latest_message_detections": int(latest_msg_count),
                    "filtered_detections": int(filtered_count),
                },
            )

            self._log_tracks_state("after no-detection cycle")
            self.publish()
            return

        for j, d in enumerate(detections):
            self.get_logger().info(
                f"[DET {j}] cam={d['cam']} local_id={d.get('local_id', '')} "
                f"pos=({d['x']:.3f}, {d['y']:.3f}) stamp={d['t']:.3f} yaw={d['yaw']}"
            )

        if len(self.tracks) == 0:
            self.get_logger().warn("No existing tracks: initializing from detections")

            for d in detections:
                self._create(d, d["t"])

            self._log_tracks_state("after initialization")
            self.publish_debug(
                event="INITIALIZED_TRACKS",
                detections=detections,
                extra={
                    "raw_detections": int(raw_count),
                    "latest_message_detections": int(latest_msg_count),
                    "filtered_detections": int(filtered_count),
                },
            )
            self.publish()
            return

        track_ids = self._get_matchable_track_ids(now)
        assigned_tracks = set()
        assigned_dets = set()
        # Track the exact detection that consumed each Global Track in this
        # fusion cycle. This lets the new-track blocker distinguish a genuine
        # second person from a duplicate observation of the already-assigned one.
        assigned_detection_by_track = {}

        if len(track_ids) == 0:
            self.get_logger().warn(
                "No matchable tracks: trying recovery before creating new tracks"
            )

            for j, d in enumerate(detections):
                recovered, recovered_tid = self._try_recover_existing_track(
                    d,
                    now,
                    assigned_tracks=assigned_tracks,
                )

                if recovered:
                    assigned_tracks.add(recovered_tid)
                    assigned_dets.add(j)
                    assigned_detection_by_track[recovered_tid] = d
                    continue

                if recovered_tid == "ambiguous":
                    assigned_dets.add(j)
                    continue

                if self._is_near_existing_track(
                    d,
                    now,
                    unavailable_track_ids=assigned_tracks,
                    assigned_detection_by_track=assigned_detection_by_track,
                ):
                    assigned_dets.add(j)
                    continue

                self._create(d, d["t"])
                assigned_dets.add(j)

            self._log_tracks_state("after recovery / creation")
            self.publish_debug(
                event="RECOVERY_OR_CREATION",
                detections=detections,
                extra={
                    "raw_detections": int(raw_count),
                    "latest_message_detections": int(latest_msg_count),
                    "filtered_detections": int(filtered_count),
                    "assigned_tracks": [int(x) for x in assigned_tracks],
                    "assigned_detections": [int(x) for x in assigned_dets],
                },
            )
            self.publish()
            return

        
        # COST MATRIX
        BIG_COST = 1e6
        eps = 1e-3

        cost = np.zeros((len(track_ids), len(detections)), dtype=float)
        real_dist = np.zeros((len(track_ids), len(detections)), dtype=float)
        dyn_thresh_mat = np.zeros((len(track_ids), len(detections)), dtype=float)
        # Effective candidate gate used to decide whether a track-detection pair
        # is allowed to enter the Hungarian cost matrix. This is intentionally
        # different from dyn_thresh_mat, which represents the tighter dynamic
        # motion threshold used as a confidence reference and for soft penalties.
        allowed_dist_mat = np.zeros((len(track_ids), len(detections)), dtype=float)
        local_relation_mat = [["none" for _ in detections] for _ in track_ids]
        appearance_dist_mat = np.full((len(track_ids), len(detections)), np.nan, dtype=float)
        geo_cost_mat = np.full((len(track_ids), len(detections)), np.nan, dtype=float)
        has_appearance_mat = np.zeros((len(track_ids), len(detections)), dtype=bool)
        use_appearance_mat = np.zeros((len(track_ids), len(detections)), dtype=bool)

        for i, tid in enumerate(track_ids):
            tr = self.tracks[tid]

            for j, d in enumerate(detections):
                det_time = d["t"]

                pred_x, pred_y = tr.predicted_position_at(det_time)
                dist = np.hypot(pred_x - d["x"], pred_y - d["y"])
                local_relation = self._local_id_relation(tr, d)
                dyn_thresh = self._dynamic_dist_thresh(tr, det_time, local_relation)
                allowed_dist = self._allowed_association_distance(
                    tr,
                    d,
                    dyn_thresh,
                    local_relation,
                )
                (
                    cost_value,
                    local_relation,
                    geo_cost,
                    appearance_dist,
                    has_appearance,
                    use_appearance,
                ) = self._association_cost_with_appearance(
                    tr,
                    d,
                    dist,
                    allowed_dist,
                    recovery_mode=False,
                )

                self._add_debug_event(
                    "MATCH_CANDIDATE",
                    global_id=int(tid),
                    det_index=int(j),
                    camera=d["cam"],
                    local_id=str(d.get("local_id", "")),
                    local_id_relation=str(local_relation),
                    pred_x=float(pred_x),
                    pred_y=float(pred_y),
                    det_x=float(d["x"]),
                    det_y=float(d["y"]),
                    distance=float(dist),
                    association_cost=float(cost_value),
                    geo_cost=float(geo_cost),
                    appearance_distance=float(appearance_dist),
                    has_appearance=bool(has_appearance),
                    use_appearance=bool(use_appearance),
                    dynamic_threshold=float(dyn_thresh),
                    allowed_threshold=float(allowed_dist),
                    valid_candidate=bool(dist <= allowed_dist + eps),
                )


                real_dist[i, j] = dist
                dyn_thresh_mat[i, j] = dyn_thresh
                allowed_dist_mat[i, j] = allowed_dist
                local_relation_mat[i][j] = local_relation
                appearance_dist_mat[i, j] = appearance_dist
                geo_cost_mat[i, j] = geo_cost
                has_appearance_mat[i, j] = has_appearance
                use_appearance_mat[i, j] = use_appearance

                if dist <= allowed_dist + eps and cost_value < BIG_COST:
                    cost[i, j] = cost_value
                else:
                    cost[i, j] = BIG_COST

                if (
                    dist <= dyn_thresh + eps
                    and dist > allowed_dist + eps
                ):
                    self._add_debug_event(
                        "MATCH_CANDIDATE_REJECTED_STRICT_LOCAL_ID_GATE",
                        global_id=int(tid),
                        det_index=int(j),
                        camera=d["cam"],
                        local_id=str(d.get("local_id", "")),
                        local_id_relation=str(local_relation),
                        pred_x=float(pred_x),
                        pred_y=float(pred_y),
                        det_x=float(d["x"]),
                        det_y=float(d["y"]),
                        distance=float(dist),
                        dynamic_threshold=float(dyn_thresh),
                        allowed_threshold=float(allowed_dist),
                        reason="strict_gate_for_unknown_or_conflict_local_id",
                    )

        self.get_logger().info(
            f"Cost matrix shape: {cost.shape[0]} tracks x {cost.shape[1]} detections"
        )

        for i, tid in enumerate(track_ids):
            row_str = ", ".join([
                (
                    f"d={real_dist[i, j]:.3f}"
                    f"/dyn={dyn_thresh_mat[i, j]:.3f}"
                    f"/allow={allowed_dist_mat[i, j]:.3f}"
                    f"/g={geo_cost_mat[i, j]:.2f}"
                    f"/a={appearance_dist_mat[i, j]:.2f}"
                    f"/useA={int(use_appearance_mat[i, j])}"
                    f"/score={1.0 - cost[i, j]:.3f}"
                    f"/c={cost[i, j]:.3f}"
                    f"/{local_relation_mat[i][j]}"
                )
                if cost[i, j] < BIG_COST
                else (
                    f"X(d={real_dist[i, j]:.3f}"
                    f">allow={allowed_dist_mat[i, j]:.3f}"
                    f", dyn={dyn_thresh_mat[i, j]:.3f}"
                    f", relation={local_relation_mat[i][j]})"
                )
                for j in range(len(detections))
            ])
            self.get_logger().info(f"[COST] T{tid} -> [{row_str}]")

        if np.all(cost >= BIG_COST):
            self.get_logger().warn(
                "All track-detection pairs are masked: no valid associations"
            )
            row_ind, col_ind = np.array([], dtype=int), np.array([], dtype=int)
        else:
            row_ind, col_ind = linear_sum_assignment(cost)

        self.get_logger().info(
            f"Hungarian assignments proposed: "
            f"{list(zip(row_ind.tolist(), col_ind.tolist()))}"
        )

     
        # ACCEPT / REJECT MATCHES
        for i, j in zip(row_ind, col_ind):
            tid = track_ids[i]
            d = detections[j]
            dist = real_dist[i, j]
            dyn_thresh = dyn_thresh_mat[i, j]
            local_relation_before_update = local_relation_mat[i][j]

            if cost[i, j] >= BIG_COST:
                self._add_debug_event(
                    "MATCH_REJECTED",
                    global_id=int(tid),
                    det_index=int(j),
                    camera=d["cam"],
                    local_id=str(d.get("local_id", "")),
                    distance=float(dist),
                    dynamic_threshold=float(dyn_thresh),
                    reason="distance_above_threshold_or_guard_rejected",
                )
                self.get_logger().warn(
                    f"[REJECT-MASKED] T{tid} x D{j}: "
                    f"dist={dist:.3f}, dyn={dyn_thresh:.3f}, "
                    f"allow={allowed_dist_mat[i, j]:.3f}"
                )
                continue

            ambiguous, ambiguity_reason = self._is_ambiguous_match(i, j, cost, real_dist)

            if ambiguous:
                self._add_debug_event(
                    "MATCH_REJECTED",
                    global_id=int(tid),
                    det_index=int(j),
                    camera=d["cam"],
                    local_id=str(d.get("local_id", "")),
                    distance=float(dist),
                    dynamic_threshold=float(dyn_thresh),
                    reason=ambiguity_reason,
                )
                self.get_logger().warn(
                    f"[REJECT-AMBIGUOUS] T{tid} x D{j}: "
                    f"dist={dist:.3f}, reason={ambiguity_reason}"
                )
                assigned_dets.add(j)
                continue

            pred_x, pred_y = self.tracks[tid].predicted_position_at(d["t"])
            dist = np.hypot(pred_x - d["x"], pred_y - d["y"])
            dyn_thresh = self._dynamic_dist_thresh(
                self.tracks[tid],
                d["t"],
                local_relation_before_update,
            )
            allowed_dist = self._allowed_association_distance(
                self.tracks[tid],
                d,
                dyn_thresh,
                local_relation_before_update,
            )

            if dist > allowed_dist + eps:
                self._add_debug_event(
                    "MATCH_REJECTED",
                    global_id=int(tid),
                    det_index=int(j),
                    camera=d["cam"],
                    local_id=str(d.get("local_id", "")),
                    local_id_relation=str(local_relation_before_update),
                    distance=float(dist),
                    dynamic_threshold=float(dyn_thresh),
                    allowed_threshold=float(allowed_dist),
                    reason="strict_local_id_gate_after_recheck",
                )
                self.get_logger().warn(
                    f"[REJECT-STRICT-GATE] T{tid} x D{j}: "
                    f"dist={dist:.3f}, allowed={allowed_dist:.3f}, "
                    f"dyn_thresh={dyn_thresh:.3f}, "
                    f"relation={local_relation_before_update}"
                )
                continue

            update_info = self.tracks[tid].update(
                d["x"],
                d["y"],
                d.get("yaw", None),
                d["t"],
                cam=d.get("cam", None),
                local_id=d.get("local_id", ""),
                embedding=d.get("embedding", None),
            )

            assigned_tracks.add(tid)
            assigned_dets.add(j)
            assigned_detection_by_track[tid] = d

            self._add_debug_event(
                "MATCH_ACCEPTED",
                global_id=int(tid),
                det_index=int(j),
                camera=d["cam"],
                local_id=str(d.get("local_id", "")),
                local_id_relation=str(local_relation_before_update),
                pred_x=float(pred_x),
                pred_y=float(pred_y),
                det_x=float(d["x"]),
                det_y=float(d["y"]),
                distance=float(dist),
                dynamic_threshold=float(dyn_thresh),
                allowed_threshold=float(allowed_dist),
                appearance_distance=None if np.isnan(appearance_dist_mat[i, j]) else float(appearance_dist_mat[i, j]),
                has_appearance=bool(has_appearance_mat[i, j]),
                use_appearance=bool(use_appearance_mat[i, j]),
                embedding_updated=bool(update_info.get("embedding_updated", False)),
                old_x=float(update_info["old_x"]),
                old_y=float(update_info["old_y"]),
                new_x=float(update_info["new_x"]),
                new_y=float(update_info["new_y"]),
                new_vx=float(update_info["new_vx"]),
                new_vy=float(update_info["new_vy"]),
                hits=int(self.tracks[tid].hits),
                missed=int(self.tracks[tid].missed),
                confirmed=bool(self.tracks[tid].confirmed),
            )

            self.get_logger().info(
                f"[MATCH] T{tid} <- D{j} cam={d['cam']} "
                f"local_id={d.get('local_id', '')} "
                f"local_relation={local_relation_before_update} "
                f"dist={dist:.3f} dyn={dyn_thresh:.3f} allow={allowed_dist:.3f} "
                f"score={1.0 - cost[i, j]:.3f} app={appearance_dist_mat[i, j]:.3f} "
                f"useA={int(use_appearance_mat[i, j])} "
                f"pos: ({update_info['old_x']:.3f}, {update_info['old_y']:.3f}) -> "
                f"({update_info['new_x']:.3f}, {update_info['new_y']:.3f}) "
                f"yaw: {update_info['old_yaw']:.3f} -> {update_info['new_yaw']:.3f} "
                f"vel: ({update_info['old_vx']:.3f}, {update_info['old_vy']:.3f}) -> "
                f"({update_info['new_vx']:.3f}, {update_info['new_vy']:.3f}) "
                f"hits={self.tracks[tid].hits}"
            )

            if update_info["became_confirmed"]:
                self.get_logger().warn(f"[CONFIRMED] Track T{tid} is now confirmed")

        
        # UNMATCHED DETECTIONS
        for j, d in enumerate(detections):
            if j in assigned_dets:
                continue

            recovered, recovered_tid = self._try_recover_existing_track(
                d,
                now,
                assigned_tracks=assigned_tracks,
            )

            if recovered:
                assigned_tracks.add(recovered_tid)
                assigned_dets.add(j)
                assigned_detection_by_track[recovered_tid] = d
                continue

            if recovered_tid == "ambiguous":
                assigned_dets.add(j)
                continue

            if self._is_near_existing_track(
                d,
                now,
                unavailable_track_ids=assigned_tracks,
                assigned_detection_by_track=assigned_detection_by_track,
            ):
                assigned_dets.add(j)
                continue

            self.get_logger().warn(
                f"[UNMATCHED DET] D{j} from {d['cam']} "
                f"local_id={d.get('local_id', '')} "
                f"-> create new track at ({d['x']:.3f}, {d['y']:.3f})"
            )

            self._create(d, d["t"])
            assigned_dets.add(j)

       
        # HANDLE MISSED TRACKS
        active_cams = set(d["cam"] for d in detections)
        min_cams_for_missed = 2

        if len(active_cams) < min_cams_for_missed:
            self.get_logger().warn(
                f"Only cameras {active_cams} this cycle -> "
                f"skip missed increment for unmatched tracks"
            )
        else:
            for tid in track_ids:
                if tid not in self.tracks:
                    continue

                tr = self.tracks[tid]

                if tid not in assigned_tracks:
                    tr.missed += 1

                    self._add_debug_event(
                        "MISSED",
                        global_id=int(tid),
                        x=float(tr.x),
                        y=float(tr.y),
                        missed=int(tr.missed),
                        age=float(now - tr.last_update),
                        reason="unmatched_track",
                    )

                    self.get_logger().warn(
                        f"[MISSED] T{tid}: missed={tr.missed}/{self.max_missed} "
                        f"pos=({tr.x:.3f}, {tr.y:.3f}) confirmed={tr.confirmed}"
                    )

                    if tr.missed > self.max_missed:
                        self._add_debug_event(
                            "DELETE_TRACK",
                            global_id=int(tid),
                            x=float(tr.x),
                            y=float(tr.y),
                            missed=int(tr.missed),
                            age=float(now - tr.last_update),
                            reason="max_missed",
                        )
                        self.get_logger().error(
                            f"[DELETE] T{tid} removed after missed={tr.missed} "
                            f"last_pos=({tr.x:.3f}, {tr.y:.3f})"
                        )
                        del self.tracks[tid]

        self._log_tracks_state("before publish")

        self.publish_debug(
            event="FUSION_UPDATE",
            detections=detections,
            extra={
                "assigned_tracks": [int(x) for x in assigned_tracks],
                "assigned_detections": [int(x) for x in assigned_dets],
                "active_cameras": list(active_cams),
                "raw_detections": int(raw_count),
                "latest_message_detections": int(latest_msg_count),
                "filtered_detections": int(filtered_count),
            },
        )

        self.publish()

   
    # LOGGING / RELIABILITY
    def _log_tracks_state(self, tag=""):
        """Log the current state of all active tracks."""
        self.get_logger().info(f"---- TRACK STATE {tag} ----")

        if len(self.tracks) == 0:
            self.get_logger().info("No active tracks")
            return

        for tid, tr in self.tracks.items():
            self.get_logger().info(
                f"T{tid}: pos=({tr.x:.3f}, {tr.y:.3f}) yaw={tr.yaw:.3f} "
                f"vel=({tr.vx:.3f}, {tr.vy:.3f}) "
                f"hits={tr.hits} missed={tr.missed} confirmed={tr.confirmed} "
                f"local_ids={tr.local_ids_by_cam} has_emb={getattr(tr, 'embedding', None) is not None}"
            )

    def _track_reliability(self, tr, age):
        """Compute a reliability score for a track based on hits, misses, and age."""
        hit_score = min(1.0, tr.hits / max(1e-6, float(self.reliability_hits_norm)))

        missed_penalty = min(
            1.0,
            tr.missed / max(1.0, float(self.publish_max_missed)),
        )

        age_penalty = min(
            1.0,
            age / max(1e-6, float(self.publish_max_age)),
        )

        reliability = (
            float(self.reliability_hit_weight) * hit_score
            + float(self.reliability_missed_weight) * (1.0 - missed_penalty)
            + float(self.reliability_age_weight) * (1.0 - age_penalty)
        )

        return max(0.0, min(1.0, reliability))

  
    # PUBLISH
    def publish_markers(self):
        marker_array = MarkerArray()
        stamp = self._sec_to_stamp(self.current_time)
        now = self.current_time

        delete_marker = Marker()
        delete_marker.header.frame_id = self.output_frame
        delete_marker.header.stamp = stamp
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        for tid, tr in self.tracks.items():
            age = now - tr.last_update

            if not tr.confirmed:
                continue

            if tr.missed > self.publish_max_missed:
                continue

            if age > self.publish_max_age:
                continue

            reliability = self._track_reliability(tr, age)

            if reliability < self.marker_min_reliability:
                continue

            marker = Marker()
            marker.header.frame_id = self.output_frame
            marker.header.stamp = stamp
            marker.ns = "tracked_people"
            marker.id = int(tid)
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(tr.x)
            marker.pose.position.y = float(tr.y)
            marker.pose.position.z = 0.25
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.45
            marker.scale.y = 0.45
            marker.scale.z = 0.45

            if reliability >= self.marker_high_reliability:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:
                marker.color.r = 1.0
                marker.color.g = 0.5
                marker.color.b = 0.0

            marker.color.a = 1.0
            marker_array.markers.append(marker)

            text = Marker()
            text.header.frame_id = self.output_frame
            text.header.stamp = stamp
            text.ns = "tracked_people_ids"
            text.id = int(tid) + 1000
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(tr.x)
            text.pose.position.y = float(tr.y)
            text.pose.position.z = 0.9
            text.pose.orientation.w = 1.0
            text.scale.z = 0.35
            # Render the global ID label in black instead of white.
            text.color.r = 0.0
            text.color.g = 0.0
            text.color.b = 0.0
            text.color.a = 1.0
            text.text = f"ID {tid}"
            marker_array.markers.append(text)

        self.marker_pub.publish(marker_array)

    def publish(self):
        """Publish reliable confirmed tracks and their RViz visualization markers."""
        msg = People()
        published_ids = []
        now = self.current_time

        msg.header.stamp = self._sec_to_stamp(self.current_time)
        msg.header.frame_id = self.output_frame

        for tid, tr in self.tracks.items():
            if not tr.confirmed:
                self.get_logger().info(
                    f"[PUBLISH-SKIP] T{tid} not confirmed yet "
                    f"(hits={tr.hits}, missed={tr.missed})"
                )
                continue

            age = now - tr.last_update

            if tr.missed > self.publish_max_missed:
                self.get_logger().info(
                    f"[PUBLISH-SKIP] T{tid} stale by missed "
                    f"(missed={tr.missed} > {self.publish_max_missed})"
                )
                continue

            if age > self.publish_max_age:
                self.get_logger().info(
                    f"[PUBLISH-SKIP] T{tid} stale by age "
                    f"(age={age:.3f}s > {self.publish_max_age:.3f}s)"
                )
                continue

            reliability = self._track_reliability(tr, age)

            if reliability < self.publish_min_reliability:
                self.get_logger().info(
                    f"[PUBLISH-SKIP] T{tid} low reliability "
                    f"(rel={reliability:.3f} < {self.publish_min_reliability:.3f})"
                )
                continue

            p = Person()
            p.name = f"id_{tid}"
            p.position.x = float(tr.x)
            p.position.y = float(tr.y)
            p.position.z = float(tr.yaw)
            p.velocity.x = float(tr.vx)
            p.velocity.y = float(tr.vy)
            p.velocity.z = 0.0
            p.reliability = float(reliability)

            msg.people.append(p)
            published_ids.append(tid)

        self.pub.publish(msg)
        self.publish_markers()

        self.get_logger().info(
            f"[PUBLISH] published {len(msg.people)} confirmed tracks -> {published_ids}"
        )


def main():
    rclpy.init()
    node = FusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
