import json
from threading import Lock

import numpy as np
import rclpy
from rclpy.node import Node
from scipy.optimize import linear_sum_assignment
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from people_msgs.msg import People, Person


def avg_angle(a, b, alpha=0.7):
    x = alpha * np.cos(a) + (1.0 - alpha) * np.cos(b)
    y = alpha * np.sin(a) + (1.0 - alpha) * np.sin(b)
    return np.arctan2(y, x)


class GlobalTrack:
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
    ):
        self.id = tid
        self.x = x
        self.y = y
        self.yaw = yaw if yaw is not None and np.isfinite(yaw) else 0.0

        self.vx = 0.0
        self.vy = 0.0
        self.last_update = t

        # The detection that creates the track counts as the first valid hit.
        # Therefore, if min_hits_to_confirm == 1, the track is confirmed immediately.
        # If min_hits_to_confirm == 2, the track will become confirmed only after
        # one additional successful update.
        self.hits = 1
        self.missed = 0
        self.min_hits_to_confirm = int(min_hits_to_confirm)
        self.confirmed = self.hits >= self.min_hits_to_confirm
        self.velocity_decay_gain = float(velocity_decay_gain)

        self.local_ids_by_cam = {}
        self._remember_local_id(cam, local_id)

    def _remember_local_id(self, cam, local_id):
        if cam is None or local_id is None:
            return

        local_id = str(local_id)
        if local_id == "":
            return

        self.local_ids_by_cam[str(cam)] = local_id

    def update(self, x, y, yaw, t, cam=None, local_id=None):
        dt = t - self.last_update

        old_x = self.x
        old_y = self.y
        old_vx = self.vx
        old_vy = self.vy
        old_yaw = self.yaw

        if yaw is not None and np.isfinite(yaw):
            self.yaw = avg_angle(self.yaw, yaw)

        if dt > 0:
            vx = (x - old_x) / dt
            vy = (y - old_y) / dt

            self.vx = 0.7 * self.vx + 0.3 * vx
            self.vy = 0.7 * self.vy + 0.3 * vy

        self.x = x
        self.y = y
        self.last_update = t

        self.hits += 1
        self.missed = 0

        self._remember_local_id(cam, local_id)

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
        }

    def predicted_position_at(self, t):
        dt = t - self.last_update

        if dt <= 0.0:
            return self.x, self.y

        # Damping della velocita': evita che una track stale continui
        # a "camminare da sola" quando non viene piu aggiornata.
        decay = np.exp(-self.velocity_decay_gain * dt)

        vx = self.vx * decay
        vy = self.vy * decay

        return self.x + vx * dt, self.y + vy * dt


class FusionNode(Node):
    def __init__(self):
        super().__init__("fusion_node")
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)

        # PARAMS

        # Dynamic association threshold.
        self.base_dist_thresh = 0.45
        self.min_dist_thresh = 0.30
        self.max_dist_thresh = 0.65
        self.speed_gain = 0.8

        # Track lifecycle
        self.max_missed = 8
        self.last_detection_time = 0.0
        self.current_time = 0.0
        self.no_det_grace_time = 0.6
        self.fusion_dt = 0.3

        # Publishing.
        self.publish_max_missed = 1
        self.publish_max_age = 0.7 #0.6

        # Matching / deletion.
        self.match_max_age = 1.2
        self.delete_max_age = 5.0

        # Recovery / reactivation.
        self.reactivation_max_age = 4.0
        self.reactivation_dist_thresh = 0.50 #0.45

        # Blocking new tracks that are too close to existing ones
        self.new_track_min_dist = 0.60

        # Ambiguity guards.
        self.ambiguity_margin = 0.16
        self.recovery_ambiguity_margin = 0.20

        # Duplicate local detections threshold, only across different cameras.
        self.duplicate_dist_thresh = 0.35

        # Weak local-ID preferences/penalties for association cost.
        self.local_id_match_bonus = 0.08
        self.local_id_conflict_penalty = 0.05

        # Extra gate when local ID is coherent.
        self.local_id_extra_gate = 0.15

        # Even when the local ID is coherent, do not allow very large
        # geometric corrections. This avoids dragging a global track too far
        # only because the local tracker reused/kept the same local ID.
        self.max_same_local_id_match_dist = 0.50

        # If local ID conflicts, accept only close geometric matches.
        self.max_conflict_match_dist = 0.20

        # Cross-camera unknown local-ID guard.
        # A track can be updated by different cameras, but the first association
        # from a new camera with an unknown local ID must be geometrically tight.
        # This prevents cases such as a track jumping from one real person to another
        # just because the dynamic threshold is large.
        self.max_unknown_cross_camera_match_dist = 0.30 #0.25

        # number of hits to consider a track confirmed.
        self.min_hits_to_confirm = 1

        # Maximum allowed jump distance without a local ID match. 
        self.max_jump_without_local_id_match = 0.40 #0.50

        # Velocity damping for prediction.
        self.velocity_decay_gain = 1.2

        self.output_frame = "map"

        # STATE
        self.tracks = {}
        self.next_id = 1
        self.buffer = []
        self.lock = Lock()
        self.cycle_count = 0
        self.debug_events = []

        # ROS INTERFACES
    
        self.create_subscription(People, "/people_center", self.cb_center, 10)
        self.create_subscription(People, "/people_left", self.cb_left, 10)
        self.create_subscription(People, "/people_right", self.cb_right, 10)

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
        self.get_logger().info(f"new_track_min_dist = {self.new_track_min_dist}")
        self.get_logger().info(f"ambiguity_margin = {self.ambiguity_margin}")
        self.get_logger().info(f"recovery_ambiguity_margin = {self.recovery_ambiguity_margin}")
        self.get_logger().info(f"duplicate_dist_thresh = {self.duplicate_dist_thresh}")
        self.get_logger().info(f"local_id_extra_gate = {self.local_id_extra_gate}")
        self.get_logger().info(
            f"max_same_local_id_match_dist = {self.max_same_local_id_match_dist}"
        )
        self.get_logger().info(f"max_conflict_match_dist = {self.max_conflict_match_dist}")
        self.get_logger().info(
            f"max_unknown_cross_camera_match_dist = {self.max_unknown_cross_camera_match_dist}"
        )
        self.get_logger().info(f"min_hits_to_confirm = {self.min_hits_to_confirm}")
        self.get_logger().info(
            f"max_jump_without_local_id_match = {self.max_jump_without_local_id_match}"
        )
        self.get_logger().info(f"velocity_decay_gain = {self.velocity_decay_gain}")
        self.get_logger().info("Subscribed topics: /people_center /people_left /people_right")
        self.get_logger().info("Publishing topic: /tracked_people")
        self.get_logger().info("======================================")


    # TIME HELPERS
    
    def _stamp_to_sec(self, stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _sec_to_stamp(self, t):
        sec = int(t)
        nanosec = int((t - sec) * 1e9)
        return rclpy.time.Time(seconds=sec, nanoseconds=nanosec).to_msg()


    # CALLBACKS

    def cb_center(self, msg):
        self._add(msg, "center")

    def cb_left(self, msg):
        self._add(msg, "left")

    def cb_right(self, msg):
        self._add(msg, "right")

    def _add(self, msg, cam):
        t = self._stamp_to_sec(msg.header.stamp)

        if t <= 0.0:
            self.get_logger().warn(
                f"[RX {cam}] invalid timestamp {t:.3f}, message skipped"
            )
            return

        self.get_logger().info(
            f"[RX {cam}] received People msg with {len(msg.people)} persons"
        )

        if len(msg.people) > 0:
            self.last_detection_time = t

        with self.lock:
            before = len(self.buffer)

            for idx, p in enumerate(msg.people):
                yaw = p.position.z if np.isfinite(p.position.z) else None
                local_id = str(p.name) if hasattr(p, "name") and p.name is not None else ""

                self.buffer.append({
                    "x": float(p.position.x),
                    "y": float(p.position.y),
                    "yaw": yaw,
                    "t": float(t),
                    "cam": cam,
                    "local_id": local_id,
                })

                self.get_logger().info(
                    f"[RX {cam}]   person[{idx}] local_id={local_id} "
                    f"pos=({p.position.x:.3f}, {p.position.y:.3f})"
                )

            after = len(self.buffer)

        self.get_logger().info(f"[RX {cam}] buffer size: {before} -> {after}")


    # DEBUG HELPERS

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
            })

        for tid, tr in self.tracks.items():
            age = now - tr.last_update
            reliability = self._track_reliability(tr, age)

            publishable = (
                tr.confirmed
                and tr.missed <= self.publish_max_missed
                and age <= self.publish_max_age
                and reliability >= 0.4
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
            })

        msg = String()
        msg.data = json.dumps(data)
        self.debug_pub.publish(msg)

  
    # PRE-FILTER HELPERS
    # These filters are applied before the association step, 
    # and they are designed to remove detections that are likely to cause problems during association, 
    # such as duplicates or old detections from the same camera. 
    # They also generate debug events for transparency.
    def _keep_latest_message_per_camera(self, detections):
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


    # ASSOCIATION HELPERS
  
    def _local_id_relation(self, tr, d):
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
        age = max(0.0, t - tr.last_update)
        speed = np.hypot(tr.vx, tr.vy)

        thresh = self.base_dist_thresh + self.speed_gain * speed * age

        if age > self.publish_max_age:
            thresh += 0.10

        thresh = max(self.min_dist_thresh, min(self.max_dist_thresh, thresh))

        if local_relation == "match":
            thresh = min(
                self.max_dist_thresh + self.local_id_extra_gate,
                thresh + self.local_id_extra_gate,
            )

        return thresh

    def _association_cost_from_distance(self, tr, d, dist):
        relation = self._local_id_relation(tr, d)
        cost_value = float(dist)

        if relation == "match":
            cost_value = max(0.0, cost_value - self.local_id_match_bonus)
        elif relation == "conflict":
            cost_value += self.local_id_conflict_penalty

        return cost_value, relation

    def _is_jump_allowed(self, dist, local_relation):
        if local_relation == "match":
            return True

        return dist <= self.max_jump_without_local_id_match

    def _is_conflict_allowed(self, dist, local_relation):
        if local_relation != "conflict":
            return True

        return dist <= self.max_conflict_match_dist

    def _is_unknown_cross_camera_allowed(self, tr, d, dist, local_relation):
        """
        Guard for risky first associations from a different camera.

        Cross-camera matching is allowed and expected, but if the detection local ID
        is unknown for the current camera while the global track already has local
        IDs from other cameras, the match must be very close geometrically.

        This prevents identity jumps such as:
            T30 center/Person_2 -> right/Person_12
        when the association is only geometrically possible because the dynamic
        threshold is too permissive.
        """
        if local_relation != "unknown":
            return True

        cam = str(d.get("cam", ""))
        local_id = str(d.get("local_id", ""))

        if cam == "" or local_id == "":
            return True

        known_other_cameras = [
            known_cam
            for known_cam in tr.local_ids_by_cam.keys()
            if str(known_cam) != cam
        ]

        if len(known_other_cameras) == 0:
            return True

        return dist <= self.max_unknown_cross_camera_match_dist

    def _allowed_association_distance(self, tr, d, dynamic_threshold, local_relation):
        """
        Final geometric gate used before accepting an association.

        The dynamic threshold is useful for normal tracking, but it can become
        too permissive when a track has velocity or age. For risky cases
        involving local-ID uncertainty, we cap it with stricter thresholds.
        """
        allowed_dist = float(dynamic_threshold)

        if local_relation == "match":
            # Local ID consistency helps, but it must not override geometry.
            # If this cap is too high, a track can be pulled toward a wrong
            # detection kept under the same local ID by the local tracker.
            allowed_dist = min(allowed_dist, self.max_same_local_id_match_dist)

        elif local_relation == "conflict":
            allowed_dist = min(allowed_dist, self.max_conflict_match_dist)

        elif local_relation == "unknown":
            cam = str(d.get("cam", ""))
            local_id = str(d.get("local_id", ""))

            known_other_cameras = [
                known_cam
                for known_cam in tr.local_ids_by_cam.keys()
                if str(known_cam) != cam
            ]

            # If this is the first time this track is associated with this
            # camera, and the track already has an ID from another camera,
            # accept only a very close geometric match.
            if cam != "" and local_id != "" and len(known_other_cameras) > 0:
                allowed_dist = min(
                    allowed_dist,
                    self.max_unknown_cross_camera_match_dist,
                )

        return allowed_dist

    def _get_matchable_track_ids(self, now):
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
        if assigned_tracks is None:
            assigned_tracks = set()

        candidates = []

        for tid, tr in self.tracks.items():
            if tid in assigned_tracks:
                continue

            age = now - tr.last_update
            if age < 0.0 or age > self.reactivation_max_age:
                continue

            _, relation = self._association_cost_from_distance(tr, d, 0.0)

            if age <= self.match_max_age:
                pred_x, pred_y = tr.predicted_position_at(d["t"])
                allowed_dist = min(
                    self.reactivation_dist_thresh,
                    self._dynamic_dist_thresh(tr, d["t"], relation),
                )
            else:
                pred_x, pred_y = tr.x, tr.y
                allowed_dist = self.reactivation_dist_thresh

                if relation == "match":
                    allowed_dist = min(
                        self.reactivation_dist_thresh + self.local_id_extra_gate,
                        self.max_dist_thresh + self.local_id_extra_gate,
                    )

            allowed_dist = self._allowed_association_distance(
                tr,
                d,
                allowed_dist,
                relation,
            )

            dist = np.hypot(pred_x - d["x"], pred_y - d["y"])

            if dist < allowed_dist:
                cost_value, relation = self._association_cost_from_distance(tr, d, dist)

                if not self._is_jump_allowed(dist, relation):
                    self._add_debug_event(
                        "RECOVER_REJECTED_JUMP",
                        global_id=int(tid),
                        camera=d["cam"],
                        local_id=str(d.get("local_id", "")),
                        local_id_relation=str(relation),
                        det_x=float(d["x"]),
                        det_y=float(d["y"]),
                        pred_x=float(pred_x),
                        pred_y=float(pred_y),
                        distance=float(dist),
                        max_jump=float(self.max_jump_without_local_id_match),
                    )
                    continue

                if not self._is_conflict_allowed(dist, relation):
                    self._add_debug_event(
                        "RECOVER_REJECTED_LOCAL_ID_CONFLICT",
                        global_id=int(tid),
                        camera=d["cam"],
                        local_id=str(d.get("local_id", "")),
                        local_id_relation=str(relation),
                        det_x=float(d["x"]),
                        det_y=float(d["y"]),
                        pred_x=float(pred_x),
                        pred_y=float(pred_y),
                        distance=float(dist),
                        max_conflict_match_dist=float(self.max_conflict_match_dist),
                    )
                    continue

                if not self._is_unknown_cross_camera_allowed(tr, d, dist, relation):
                    self._add_debug_event(
                        "RECOVER_REJECTED_UNKNOWN_CROSS_CAMERA",
                        global_id=int(tid),
                        camera=d["cam"],
                        local_id=str(d.get("local_id", "")),
                        local_id_relation=str(relation),
                        det_x=float(d["x"]),
                        det_y=float(d["y"]),
                        pred_x=float(pred_x),
                        pred_y=float(pred_y),
                        distance=float(dist),
                        max_unknown_cross_camera_match_dist=float(
                            self.max_unknown_cross_camera_match_dist
                        ),
                    )
                    continue

                candidates.append({
                    "tid": tid,
                    "track": tr,
                    "dist": dist,
                    "cost": cost_value,
                    "pred_x": pred_x,
                    "pred_y": pred_y,
                    "allowed_dist": allowed_dist,
                    "relation": relation,
                })

        if len(candidates) == 0:
            self._add_debug_event(
                "RECOVER_FAILED_NO_VALID_CANDIDATE",
                camera=d["cam"],
                local_id=str(d.get("local_id", "")),
                det_x=float(d["x"]),
                det_y=float(d["y"]),
                reason="no_existing_track_close_enough_or_all_guards_rejected",
            )

            self.get_logger().warn(
                f"[RECOVER-FAILED] detection from {d['cam']} "
                f"local_id={d.get('local_id', '')} at "
                f"({d['x']:.3f}, {d['y']:.3f}) could not recover any existing track: "
                f"no track was close enough or all candidates were rejected by guards"
            )
            return False, None

        candidates.sort(key=lambda c: c["cost"])
        best = candidates[0]

        if len(candidates) >= 2:
            second = candidates[1]

            if abs(second["dist"] - best["dist"]) < self.recovery_ambiguity_margin:
                self._add_debug_event(
                    "RECOVER_REJECTED_AMBIGUOUS",
                    best_global_id=int(best["tid"]),
                    second_global_id=int(second["tid"]),
                    camera=d["cam"],
                    local_id=str(d.get("local_id", "")),
                    det_x=float(d["x"]),
                    det_y=float(d["y"]),
                    best_distance=float(best["dist"]),
                    second_distance=float(second["dist"]),
                    margin=float(self.recovery_ambiguity_margin),
                )

                self.get_logger().warn(
                    f"[RECOVER-REJECT-AMBIGUOUS] det from {d['cam']} "
                    f"near T{best['tid']} dist={best['dist']:.3f} and "
                    f"T{second['tid']} dist={second['dist']:.3f}; "
                    f"skip recovery to avoid ID switch"
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
        )

        self._add_debug_event(
            "RECOVER_EXISTING_TRACK",
            global_id=int(best_tid),
            camera=d["cam"],
            local_id=str(d.get("local_id", "")),
            local_id_relation=str(best["relation"]),
            pred_x=float(best["pred_x"]),
            pred_y=float(best["pred_y"]),
            det_x=float(d["x"]),
            det_y=float(d["y"]),
            distance=float(best["dist"]),
            threshold=float(best["allowed_dist"]),
            old_x=float(update_info["old_x"]),
            old_y=float(update_info["old_y"]),
            new_x=float(update_info["new_x"]),
            new_y=float(update_info["new_y"]),
            hits=int(self.tracks[best_tid].hits),
            missed=int(self.tracks[best_tid].missed),
        )

        self.get_logger().warn(
            f"[RECOVER] T{best_tid} recovered from {d['cam']} "
            f"local_id={d.get('local_id', '')} "
            f"dist={best['dist']:.3f} threshold={best['allowed_dist']:.3f} "
            f"relation={best['relation']} "
            f"pos: ({update_info['old_x']:.3f}, {update_info['old_y']:.3f}) -> "
            f"({update_info['new_x']:.3f}, {update_info['new_y']:.3f})"
        )

        return True, best_tid

    def _is_near_existing_track(self, d, now):
        for tid, tr in self.tracks.items():
            age = now - tr.last_update

            if age < 0.0 or age > self.reactivation_max_age:
                continue

            if age <= self.match_max_age:
                px, py = tr.predicted_position_at(d["t"])
            else:
                px, py = tr.x, tr.y

            dist = np.hypot(px - d["x"], py - d["y"])

            if dist < self.new_track_min_dist:
                self._add_debug_event(
                    "NEW_TRACK_BLOCKED_NEAR_EXISTING",
                    nearby_global_id=int(tid),
                    camera=d["cam"],
                    local_id=str(d.get("local_id", "")),
                    det_x=float(d["x"]),
                    det_y=float(d["y"]),
                    track_x=float(px),
                    track_y=float(py),
                    distance=float(dist),
                    threshold=float(self.new_track_min_dist),
                )

                self.get_logger().warn(
                    f"[NEW TRACK BLOCKED] detection from {d['cam']} "
                    f"local_id={d.get('local_id', '')} at "
                    f"({d['x']:.3f}, {d['y']:.3f}) is close to T{tid} "
                    f"dist={dist:.3f} < {self.new_track_min_dist:.3f}"
                )

                return True

        return False

    def _is_ambiguous_match(self, i, j, cost, real_dist):
        BIG_COST = 1e6
        chosen_dist = real_dist[i, j]

        row_valid = [
            real_dist[i, jj]
            for jj in range(real_dist.shape[1])
            if jj != j and cost[i, jj] < BIG_COST
        ]

        col_valid = [
            real_dist[ii, j]
            for ii in range(real_dist.shape[0])
            if ii != i and cost[ii, j] < BIG_COST
        ]

        if len(row_valid) > 0:
            second_det_dist = min(row_valid)

            if abs(second_det_dist - chosen_dist) < self.ambiguity_margin:
                return True, "track_has_multiple_similar_detections"

        if len(col_valid) > 0:
            second_track_dist = min(col_valid)

            if abs(second_track_dist - chosen_dist) < self.ambiguity_margin:
                return True, "detection_has_multiple_similar_tracks"

        return False, ""

    # ========================================================
    # TRACK LIFECYCLE
    # ========================================================
    def _create(self, d, t):
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
        )

        self._add_debug_event(
            "NEW_TRACK",
            global_id=int(tid),
            camera=d["cam"],
            local_id=str(d.get("local_id", "")),
            x=float(d["x"]),
            y=float(d["y"]),
            yaw=None if d.get("yaw", None) is None else float(d.get("yaw")),
        )

        self.get_logger().warn(
            f"[NEW TRACK] T{tid} created from {d['cam']} "
            f"local_id={d.get('local_id', '')} "
            f"at ({d['x']:.3f}, {d['y']:.3f})"
        )

    def _prune_old_tracks(self, now):
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

    # ========================================================
    # MAIN PROCESS LOOP
    # ========================================================
    def process(self):
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
                    continue

                if recovered_tid == "ambiguous":
                    assigned_dets.add(j)
                    continue

                if self._is_near_existing_track(d, now):
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

        # ----------------------------------------------------
        # COST MATRIX
        # ----------------------------------------------------
        BIG_COST = 1e6
        eps = 1e-3

        cost = np.zeros((len(track_ids), len(detections)), dtype=float)
        real_dist = np.zeros((len(track_ids), len(detections)), dtype=float)
        dyn_thresh_mat = np.zeros((len(track_ids), len(detections)), dtype=float)
        local_relation_mat = [["none" for _ in detections] for _ in track_ids]

        for i, tid in enumerate(track_ids):
            tr = self.tracks[tid]

            for j, d in enumerate(detections):
                det_time = d["t"]

                pred_x, pred_y = tr.predicted_position_at(det_time)
                dist = np.hypot(pred_x - d["x"], pred_y - d["y"])
                cost_value, local_relation = self._association_cost_from_distance(
                    tr,
                    d,
                    dist,
                )
                dyn_thresh = self._dynamic_dist_thresh(tr, det_time, local_relation)
                allowed_dist = self._allowed_association_distance(
                    tr,
                    d,
                    dyn_thresh,
                    local_relation,
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
                    dynamic_threshold=float(dyn_thresh),
                    allowed_threshold=float(allowed_dist),
                    valid_candidate=bool(dist <= allowed_dist + eps),
                )

                jump_allowed = self._is_jump_allowed(dist, local_relation)
                conflict_allowed = self._is_conflict_allowed(dist, local_relation)
                unknown_cross_camera_allowed = self._is_unknown_cross_camera_allowed(
                    tr,
                    d,
                    dist,
                    local_relation,
                )

                real_dist[i, j] = dist
                dyn_thresh_mat[i, j] = dyn_thresh
                local_relation_mat[i][j] = local_relation

                if (
                    dist <= allowed_dist + eps
                    and jump_allowed
                    and conflict_allowed
                    and unknown_cross_camera_allowed
                ):
                    cost[i, j] = cost_value
                else:
                    cost[i, j] = BIG_COST

                if dist <= allowed_dist + eps and not jump_allowed:
                    self._add_debug_event(
                        "MATCH_CANDIDATE_REJECTED_JUMP",
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
                        max_jump=float(self.max_jump_without_local_id_match),
                    )

                if dist <= allowed_dist + eps and jump_allowed and not conflict_allowed:
                    self._add_debug_event(
                        "MATCH_CANDIDATE_REJECTED_LOCAL_ID_CONFLICT",
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
                        max_conflict_match_dist=float(self.max_conflict_match_dist),
                    )

                if (
                    dist <= allowed_dist + eps
                    and jump_allowed
                    and conflict_allowed
                    and not unknown_cross_camera_allowed
                ):
                    self._add_debug_event(
                        "MATCH_CANDIDATE_REJECTED_UNKNOWN_CROSS_CAMERA",
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
                        max_unknown_cross_camera_match_dist=float(
                            self.max_unknown_cross_camera_match_dist
                        ),
                    )

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
                f"{real_dist[i, j]:.3f}/{dyn_thresh_mat[i, j]:.3f}/c={cost[i, j]:.3f}/{local_relation_mat[i][j]}"
                if cost[i, j] < BIG_COST
                else f"X({real_dist[i, j]:.3f}>{dyn_thresh_mat[i, j] + eps:.3f})"
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
                    f"dist={dist:.3f}, dyn_thresh={dyn_thresh:.3f}"
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
            )

            assigned_tracks.add(tid)
            assigned_dets.add(j)

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
                f"dist={dist:.3f} dyn_thresh={dyn_thresh:.3f} "
                f"pos: ({update_info['old_x']:.3f}, {update_info['old_y']:.3f}) -> "
                f"({update_info['new_x']:.3f}, {update_info['new_y']:.3f}) "
                f"yaw: {update_info['old_yaw']:.3f} -> {update_info['new_yaw']:.3f} "
                f"vel: ({update_info['old_vx']:.3f}, {update_info['old_vy']:.3f}) -> "
                f"({update_info['new_vx']:.3f}, {update_info['new_vy']:.3f}) "
                f"hits={self.tracks[tid].hits}"
            )

            if update_info["became_confirmed"]:
                self.get_logger().warn(f"[CONFIRMED] Track T{tid} is now confirmed")

        # ----------------------------------------------------
        # UNMATCHED DETECTIONS
        # ----------------------------------------------------
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
                continue

            if recovered_tid == "ambiguous":
                assigned_dets.add(j)
                continue

            if self._is_near_existing_track(d, now):
                assigned_dets.add(j)
                continue

            self.get_logger().warn(
                f"[UNMATCHED DET] D{j} from {d['cam']} "
                f"local_id={d.get('local_id', '')} "
                f"-> create new track at ({d['x']:.3f}, {d['y']:.3f})"
            )

            self._create(d, d["t"])
            assigned_dets.add(j)

        # ----------------------------------------------------
        # HANDLE MISSED TRACKS
        # ----------------------------------------------------
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

    # ========================================================
    # LOGGING / RELIABILITY
    # ========================================================
    def _log_tracks_state(self, tag=""):
        self.get_logger().info(f"---- TRACK STATE {tag} ----")

        if len(self.tracks) == 0:
            self.get_logger().info("No active tracks")
            return

        for tid, tr in self.tracks.items():
            self.get_logger().info(
                f"T{tid}: pos=({tr.x:.3f}, {tr.y:.3f}) yaw={tr.yaw:.3f} "
                f"vel=({tr.vx:.3f}, {tr.vy:.3f}) "
                f"hits={tr.hits} missed={tr.missed} confirmed={tr.confirmed} "
                f"local_ids={tr.local_ids_by_cam}"
            )

    def _track_reliability(self, tr, age):
        hit_score = min(1.0, tr.hits / 3.0)

        missed_penalty = min(
            1.0,
            tr.missed / max(1.0, float(self.publish_max_missed)),
        )

        age_penalty = min(
            1.0,
            age / max(1e-6, float(self.publish_max_age)),
        )

        reliability = (
            0.6 * hit_score
            + 0.2 * (1.0 - missed_penalty)
            + 0.2 * (1.0 - age_penalty)
        )

        return max(0.0, min(1.0, reliability))

    # ========================================================
    # PUBLISH
    # ========================================================
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

            if reliability < 0.4:
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

            if reliability >= 0.7:
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
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = f"ID {tid}"
            marker_array.markers.append(text)

        self.marker_pub.publish(marker_array)

    def publish(self):
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

            if reliability < 0.40:
                self.get_logger().info(
                    f"[PUBLISH-SKIP] T{tid} low reliability "
                    f"(rel={reliability:.3f} < 0.400)"
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
