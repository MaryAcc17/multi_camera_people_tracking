#!/usr/bin/env python3
"""
extract_tracking_validation_csvs.py

Pure ROS 2 bag -> CSV extractor for thesis validation of the multi-camera
people-tracking fusion node.

This script deliberately DOES NOT:
- associate ground-truth agents with global tracks;
- decide which agents are visible;
- compute visibility segments/episodes;
- compute tracking metrics.

Those operations must be performed by the separate validation script so that
visibility and identity evaluation remain independent from the tracker output.

Expected workflow
-----------------
1. Input bag: original simulation/input topics.
2. Output bag: replay of the input bag while the three local trackers and the
   fusion node are running.
3. Run this extractor.
4. Give the generated CSVs to the visibility-episode validation script.

Main generated files
--------------------
- ground_truth_people.csv
- robot_pose_validation.csv
- global_tracks_validation.csv
- local_detections_validation.csv
- fusion_events_validation.csv
- debug_events_validation.csv
- extraction_summary.json

Recommended topics
------------------
Input bag:
- /human_states or /people                 ground truth
- /jackal/ground_truth                     robot pose in map frame
  OR /tf + /tf_static                      robot pose reconstructed from TF

Output bag:
- /tracked_people                          actual published global tracks
- /people_center
- /people_left
- /people_right                            custom TrackedPeople local outputs
- /tracked_people_debug                    fusion/debug JSON

Important methodological rule
-----------------------------
The GT agent name and the tracker Global ID remain separate:
- agent_id identifies the simulated ground-truth trajectory;
- global_id identifies the trajectory produced by the fusion node.

The extractor never inserts the GT name into a global track. Their association
is built later by the validation script using geometry and temporal continuity.

Example: odometry/ground-truth robot pose
-----------------------------------------
python3 extract_tracking_validation_csvs.py \
  --rosbags-dir /workspaces/hunavsim_devcontainer/src/People_tracking-master/pic4people_tracking/rosbags \
  --input-bag validation_input \
  --output-bag multistream_validation_output \
  --results-dir validation_csv \
  --gt-topic /human_states \
  --robot-pose-source odom \
  --robot-odom-topic /jackal/ground_truth \
  --required-robot-frame map

Example: robot pose reconstructed from TF
-----------------------------------------
python3 extract_tracking_validation_csvs.py \
  --input-bag validation_input \
  --output-bag multistream_validation_output \
  --results-dir validation_csv \
  --gt-topic /human_states \
  --robot-pose-source tf \
  --map-frame map \
  --odom-frame odom \
  --base-frame base_link
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rosbag2_py
from nav_msgs.msg import Odometry
from people_msgs.msg import People
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

try:
    from hunav_msgs.msg import Agents
except Exception:
    Agents = None


DEFAULT_ROSBAGS_DIR = Path(
    "/workspaces/hunavsim_devcontainer/src/People_tracking-master/"
    "pic4people_tracking/rosbags"
)
DEFAULT_INPUT_BAG = "input1"
DEFAULT_OUTPUT_BAG = "output1"
DEFAULT_RESULTS_DIR = "validation_csv"

EPS = 1e-12


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def stamp_to_sec(stamp: Any) -> float:
    if stamp is None:
        return math.nan
    try:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except Exception:
        return math.nan


def valid_message_time(stamp: Any, bag_time: float) -> float:
    value = stamp_to_sec(stamp)
    if not np.isfinite(value) or value <= 0.0:
        return float(bag_time)
    return float(value)


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        result = float(value)
        return result if np.isfinite(result) else default
    except Exception:
        return default


def safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        return str(value)
    except Exception:
        return default


def bool_or_empty(value: Any) -> Any:
    if value is None:
        return ""
    return bool(value)


def get_nested_attr(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    """Return the first available dotted attribute from *names*."""
    for name in names:
        current = obj
        found = True
        for part in name.split("."):
            if not hasattr(current, part):
                found = False
                break
            current = getattr(current, part)
        if found:
            return current
    return default


def vector3_from_any(obj: Any, default: float = math.nan) -> Tuple[float, float, float]:
    if obj is None:
        return default, default, default

    candidates = [
        obj,
        get_nested_attr(obj, ["position", "pose.position", "pose.pose.position"]),
        get_nested_attr(obj, ["linear", "twist.linear", "twist.twist.linear"]),
    ]

    for candidate in candidates:
        if candidate is None:
            continue
        if all(hasattr(candidate, field) for field in ("x", "y", "z")):
            return (
                safe_float(candidate.x),
                safe_float(candidate.y),
                safe_float(candidate.z),
            )

    return default, default, default


def extract_hunav_agent_fields(agent: Any) -> Tuple[str, float, float, float, float, float, float, float]:
    """Extract a HuNav agent robustly across small message-version changes."""
    agent_id = get_nested_attr(agent, ["name", "id", "agent_id"], "")
    if safe_text(agent_id).strip() == "":
        agent_id = get_nested_attr(agent, ["behavior.name"], "")

    position = get_nested_attr(
        agent,
        ["position", "pose.position", "pose.pose.position"],
        None,
    )
    x, y, z = vector3_from_any(position)
    if not np.isfinite(x) or not np.isfinite(y):
        x, y, z = vector3_from_any(agent)

    velocity = get_nested_attr(
        agent,
        ["velocity", "linear_velocity", "twist.linear", "twist.twist.linear"],
        None,
    )
    vx, vy, vz = vector3_from_any(velocity)

    reliability = safe_float(
        get_nested_attr(agent, ["reliability", "confidence"], 1.0),
        1.0,
    )

    return safe_text(agent_id), x, y, z, vx, vy, vz, reliability


# ---------------------------------------------------------------------------
# Quaternion and TF helpers
# ---------------------------------------------------------------------------


def normalize_quaternion(q: Sequence[float]) -> np.ndarray:
    result = np.asarray(q, dtype=float)
    norm = float(np.linalg.norm(result))
    if not np.isfinite(norm) or norm <= EPS:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return result / norm


def quaternion_product_raw(q1: Sequence[float], q2: Sequence[float]) -> np.ndarray:
    """Hamilton product without normalization.

    Products involving a pure-vector quaternion must not be normalized, or the
    magnitude of the rotated translation would be destroyed.
    """
    x1, y1, z1, w1 = np.asarray(q1, dtype=float)
    x2, y2, z2, w2 = np.asarray(q2, dtype=float)
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=float,
    )


def quaternion_multiply(q1: Sequence[float], q2: Sequence[float]) -> np.ndarray:
    return normalize_quaternion(quaternion_product_raw(q1, q2))


def quaternion_to_yaw(q: Sequence[float]) -> float:
    x, y, z, w = normalize_quaternion(q)
    return float(
        math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
    )


def rotate_vector(q: Sequence[float], vector: Sequence[float]) -> np.ndarray:
    q_unit = normalize_quaternion(q)
    q_conjugate = np.array(
        [-q_unit[0], -q_unit[1], -q_unit[2], q_unit[3]],
        dtype=float,
    )
    vector_quaternion = np.array(
        [vector[0], vector[1], vector[2], 0.0],
        dtype=float,
    )
    return quaternion_product_raw(
        quaternion_product_raw(q_unit, vector_quaternion),
        q_conjugate,
    )[:3]


def compose_transform(
    translation_ab: Sequence[float],
    quaternion_ab: Sequence[float],
    translation_bc: Sequence[float],
    quaternion_bc: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    translation_ac = np.asarray(translation_ab, dtype=float) + rotate_vector(
        quaternion_ab,
        translation_bc,
    )
    quaternion_ac = quaternion_multiply(quaternion_ab, quaternion_bc)
    return translation_ac, quaternion_ac


def transform_to_arrays(transform_stamped: Any) -> Tuple[np.ndarray, np.ndarray]:
    translation = transform_stamped.transform.translation
    rotation = transform_stamped.transform.rotation
    return (
        np.array([translation.x, translation.y, translation.z], dtype=float),
        normalize_quaternion([rotation.x, rotation.y, rotation.z, rotation.w]),
    )


def normalize_frame(frame: str) -> str:
    return safe_text(frame).strip().lstrip("/")


@dataclass
class TransformSample:
    time: float
    bag_time: float
    translation: np.ndarray
    quaternion: np.ndarray
    source: str


@dataclass
class RobotPoseRow:
    time: float
    bag_time: float
    frame_id: str
    child_frame_id: str
    x: float
    y: float
    z: float
    yaw: float
    vx: float
    vy: float
    vtheta: float
    source: str


class TfRobotPoseExtractor:
    """Reconstruct map -> base pose from TF messages.

    Supported paths:
    - direct map -> base_link;
    - map -> odom combined with odom -> base_link.

    Direct samples are preferred when a direct and composed pose share the same
    timestamp.
    """

    def __init__(
        self,
        map_frame: str,
        odom_frame: str,
        base_frame: str,
        max_chain_skew_s: float,
    ) -> None:
        self.map_frame = normalize_frame(map_frame)
        self.odom_frame = normalize_frame(odom_frame)
        self.base_frame = normalize_frame(base_frame)
        self.max_chain_skew_s = float(max_chain_skew_s)

        self.latest_map_to_odom: Optional[TransformSample] = None
        self.latest_odom_to_base: Optional[TransformSample] = None
        self.samples: List[RobotPoseRow] = []

    def process_message(self, msg: TFMessage, bag_time: float, topic: str) -> None:
        for transform in msg.transforms:
            parent = normalize_frame(transform.header.frame_id)
            child = normalize_frame(transform.child_frame_id)
            transform_time = valid_message_time(transform.header.stamp, bag_time)
            translation, quaternion = transform_to_arrays(transform)

            if parent == self.map_frame and child == self.base_frame:
                self.samples.append(
                    RobotPoseRow(
                        time=transform_time,
                        bag_time=bag_time,
                        frame_id=self.map_frame,
                        child_frame_id=self.base_frame,
                        x=float(translation[0]),
                        y=float(translation[1]),
                        z=float(translation[2]),
                        yaw=quaternion_to_yaw(quaternion),
                        vx=math.nan,
                        vy=math.nan,
                        vtheta=math.nan,
                        source=f"{topic}:direct",
                    )
                )
                continue

            sample = TransformSample(
                time=transform_time,
                bag_time=bag_time,
                translation=translation,
                quaternion=quaternion,
                source=topic,
            )

            if parent == self.map_frame and child == self.odom_frame:
                self.latest_map_to_odom = sample
            elif parent == self.odom_frame and child == self.base_frame:
                self.latest_odom_to_base = sample
            else:
                continue

            self._try_compose()

    def _try_compose(self) -> None:
        if self.latest_map_to_odom is None or self.latest_odom_to_base is None:
            return

        map_to_odom = self.latest_map_to_odom
        odom_to_base = self.latest_odom_to_base
        skew = abs(map_to_odom.time - odom_to_base.time)
        if skew > self.max_chain_skew_s:
            return

        translation, quaternion = compose_transform(
            map_to_odom.translation,
            map_to_odom.quaternion,
            odom_to_base.translation,
            odom_to_base.quaternion,
        )
        composed_time = max(map_to_odom.time, odom_to_base.time)
        composed_bag_time = max(map_to_odom.bag_time, odom_to_base.bag_time)

        self.samples.append(
            RobotPoseRow(
                time=composed_time,
                bag_time=composed_bag_time,
                frame_id=self.map_frame,
                child_frame_id=self.base_frame,
                x=float(translation[0]),
                y=float(translation[1]),
                z=float(translation[2]),
                yaw=quaternion_to_yaw(quaternion),
                vx=math.nan,
                vy=math.nan,
                vtheta=math.nan,
                source="tf:map_odom_base",
            )
        )

    def finalized_samples(self, dedup_tolerance_s: float) -> List[RobotPoseRow]:
        if not self.samples:
            return []

        # Sort direct samples before composed samples when times are equivalent.
        ordered = sorted(
            self.samples,
            key=lambda row: (
                row.time,
                0 if row.source.endswith(":direct") else 1,
            ),
        )

        result: List[RobotPoseRow] = []
        for sample in ordered:
            if not result:
                result.append(sample)
                continue

            if abs(sample.time - result[-1].time) <= dedup_tolerance_s:
                previous_is_direct = result[-1].source.endswith(":direct")
                current_is_direct = sample.source.endswith(":direct")
                if current_is_direct and not previous_is_direct:
                    result[-1] = sample
                # Otherwise retain the first/preferred sample.
            else:
                result.append(sample)

        return result


# ---------------------------------------------------------------------------
# Bag and CSV infrastructure
# ---------------------------------------------------------------------------


def open_reader(bag_uri: Path, storage_id: str) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_uri),
        storage_id=storage_id,
    )
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader.open(storage_options, converter_options)
    return reader


def available_topics(reader: rosbag2_py.SequentialReader) -> Dict[str, str]:
    return {
        info.name: info.type
        for info in reader.get_all_topics_and_types()
    }


@dataclass
class CsvOutput:
    paths: Dict[str, Path]
    files: Dict[str, Any]
    writers: Dict[str, csv.writer]


def initialize_csv_outputs(results_dir: Path) -> CsvOutput:
    results_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "gt": results_dir / "ground_truth_people.csv",
        "robot": results_dir / "robot_pose_validation.csv",
        "tracks": results_dir / "global_tracks_validation.csv",
        "detections": results_dir / "local_detections_validation.csv",
        "events": results_dir / "fusion_events_validation.csv",
        "debug_events": results_dir / "debug_events_validation.csv",
    }

    files = {
        key: path.open("w", newline="", encoding="utf-8")
        for key, path in paths.items()
    }
    writers = {key: csv.writer(stream) for key, stream in files.items()}

    writers["gt"].writerow(
        [
            "time",
            "bag_time",
            "frame_id",
            "agent_id",
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "reliability",
            "source_topic",
        ]
    )

    writers["robot"].writerow(
        [
            "time",
            "bag_time",
            "frame_id",
            "child_frame_id",
            "x",
            "y",
            "z",
            "yaw",
            "vx",
            "vy",
            "vtheta",
            "source",
        ]
    )

    # Rows in this table come exclusively from the actual published output
    # /tracked_people, not from internal debug track states.
    writers["tracks"].writerow(
        [
            "time",
            "bag_time",
            "frame_id",
            "global_id",
            "x",
            "y",
            "z",
            "yaw",
            "vx",
            "vy",
            "vz",
            "reliability",
            "source_topic",
        ]
    )

    writers["detections"].writerow(
        [
            "time",
            "bag_time",
            "frame_id",
            "camera",
            "local_id",
            "x",
            "y",
            "z",
            "yaw",
            "vx",
            "vy",
            "vz",
            "reliability",
            "source_topic",
            "source_kind",
            "cycle",
            "event",
            "stamp",
            "has_embedding",
            "embedding_dim",
            "embedding_json",
            "embedding_dt",
        ]
    )

    writers["events"].writerow(
        [
            "time",
            "bag_time",
            "cycle",
            "event",
            "num_detections",
            "published_ids",
            "extra",
            "source_topic",
        ]
    )

    writers["debug_events"].writerow(
        [
            "time",
            "bag_time",
            "cycle",
            "main_event",
            "event_type",
            "global_id",
            "det_index",
            "camera",
            "local_id",
            "local_id_relation",
            "association_cost",
            "appearance_distance",
            "appearance_cost",
            "geo_cost",
            "has_appearance",
            "use_appearance",
            "reid_strong",
            "reid_bad",
            "reid_extended_gate",
            "strict_threshold",
            "allowed_threshold",
            "has_embedding",
            "embedding_updated",
            "pred_x",
            "pred_y",
            "det_x",
            "det_y",
            "distance",
            "dynamic_threshold",
            "valid_candidate",
            "old_x",
            "old_y",
            "new_x",
            "new_y",
            "x",
            "y",
            "yaw",
            "hits",
            "missed",
            "age",
            "reason",
            "dropped_cam",
            "dropped_local_id",
            "dropped_x",
            "dropped_y",
            "kept_cam",
            "kept_local_id",
            "kept_x",
            "kept_y",
            "threshold",
            "max_jump",
            "max_conflict_match_dist",
            "best_global_id",
            "second_global_id",
            "best_distance",
            "second_distance",
            "track_x",
            "track_y",
            "nearby_global_id",
            "stamp",
            "latest_stamp",
            "raw_event_json",
        ]
    )

    return CsvOutput(paths=paths, files=files, writers=writers)


def close_csv_outputs(output: CsvOutput) -> None:
    for stream in output.files.values():
        stream.close()


# ---------------------------------------------------------------------------
# Message extraction
# ---------------------------------------------------------------------------


def write_people_ground_truth(
    msg: People,
    bag_time: float,
    source_topic: str,
    writer: csv.writer,
) -> int:
    message_time = valid_message_time(msg.header.stamp, bag_time)
    frame_id = safe_text(msg.header.frame_id) or "map"
    count = 0

    for person in msg.people:
        writer.writerow(
            [
                message_time,
                bag_time,
                frame_id,
                safe_text(person.name),
                safe_float(person.position.x),
                safe_float(person.position.y),
                safe_float(person.position.z),
                safe_float(person.velocity.x),
                safe_float(person.velocity.y),
                safe_float(person.velocity.z),
                safe_float(person.reliability, 1.0),
                source_topic,
            ]
        )
        count += 1

    return count


def write_hunav_ground_truth(
    msg: Any,
    bag_time: float,
    source_topic: str,
    writer: csv.writer,
) -> int:
    header = getattr(msg, "header", None)
    if header is not None:
        message_time = valid_message_time(header.stamp, bag_time)
        frame_id = safe_text(getattr(header, "frame_id", "")) or "map"
    else:
        message_time = bag_time
        frame_id = "map"

    count = 0
    for agent in getattr(msg, "agents", []):
        agent_id, x, y, z, vx, vy, vz, reliability = extract_hunav_agent_fields(agent)
        writer.writerow(
            [
                message_time,
                bag_time,
                frame_id,
                agent_id,
                x,
                y,
                z,
                vx,
                vy,
                vz,
                reliability,
                source_topic,
            ]
        )
        count += 1

    return count


def odometry_to_robot_pose(
    msg: Odometry,
    bag_time: float,
    required_frame: Optional[str],
    source_topic: str,
) -> Optional[RobotPoseRow]:
    frame_id = normalize_frame(msg.header.frame_id)
    child_frame_id = normalize_frame(msg.child_frame_id)

    if required_frame and frame_id != normalize_frame(required_frame):
        return None

    message_time = valid_message_time(msg.header.stamp, bag_time)
    orientation = msg.pose.pose.orientation
    yaw = quaternion_to_yaw(
        [orientation.x, orientation.y, orientation.z, orientation.w]
    )

    return RobotPoseRow(
        time=message_time,
        bag_time=bag_time,
        frame_id=frame_id,
        child_frame_id=child_frame_id,
        x=safe_float(msg.pose.pose.position.x),
        y=safe_float(msg.pose.pose.position.y),
        z=safe_float(msg.pose.pose.position.z),
        yaw=yaw,
        vx=safe_float(msg.twist.twist.linear.x),
        vy=safe_float(msg.twist.twist.linear.y),
        vtheta=safe_float(msg.twist.twist.angular.z),
        source=f"odom:{source_topic}",
    )


def write_robot_rows(rows: Iterable[RobotPoseRow], writer: csv.writer) -> int:
    count = 0
    for row in rows:
        writer.writerow(
            [
                row.time,
                row.bag_time,
                row.frame_id,
                row.child_frame_id,
                row.x,
                row.y,
                row.z,
                row.yaw,
                row.vx,
                row.vy,
                row.vtheta,
                row.source,
            ]
        )
        count += 1
    return count


def write_published_global_tracks(
    msg: People,
    bag_time: float,
    source_topic: str,
    writer: csv.writer,
) -> int:
    message_time = valid_message_time(msg.header.stamp, bag_time)
    frame_id = safe_text(msg.header.frame_id)
    count = 0

    for person in msg.people:
        # In this system Person.name carries the global track ID.
        global_id = safe_text(person.name).strip()
        if global_id == "":
            # Keep the row inspectable, but avoid inventing an identity.
            global_id = ""

        # The current fusion pipeline may encode yaw in position.z. Preserve both
        # z and yaw columns so the downstream script can select the intended one.
        z_value = safe_float(person.position.z)
        yaw_value = z_value

        writer.writerow(
            [
                message_time,
                bag_time,
                frame_id,
                global_id,
                safe_float(person.position.x),
                safe_float(person.position.y),
                z_value,
                yaw_value,
                safe_float(person.velocity.x),
                safe_float(person.velocity.y),
                safe_float(person.velocity.z),
                safe_float(person.reliability, 1.0),
                source_topic,
            ]
        )
        count += 1

    return count


def write_local_people(
    msg: Any,
    bag_time: float,
    source_topic: str,
    default_camera: str,
    writer: csv.writer,
) -> int:
    """Write custom TrackedPeople local-tracker messages.

    Expected element layout:
      string camera
      string local_id
      geometry_msgs/Point position
      geometry_msgs/Point velocity
      float32 yaw
      float32 reliability
      float32[] embedding

    The function is intentionally attribute-based, so it works regardless of
    the ROS package name that owns TrackedPeople/TrackedPerson, provided the
    currently sourced definition matches the one stored in the bag.
    """
    header = getattr(msg, "header", None)
    if header is not None:
        message_time = valid_message_time(getattr(header, "stamp", None), bag_time)
        frame_id = safe_text(getattr(header, "frame_id", ""))
    else:
        message_time = float(bag_time)
        frame_id = ""

    people = getattr(msg, "people", None)
    if people is None:
        raise RuntimeError(
            f"Local topic '{source_topic}' does not expose a 'people' field. "
            "Expected the custom TrackedPeople message."
        )

    count = 0
    for person in people:
        if not hasattr(person, "local_id"):
            raise RuntimeError(
                f"Elements of local topic '{source_topic}' do not expose "
                "'local_id'. Expected the custom TrackedPerson message."
            )

        position = getattr(person, "position", None)
        velocity = getattr(person, "velocity", None)
        if position is None or velocity is None:
            raise RuntimeError(
                f"TrackedPerson on '{source_topic}' is missing position or velocity."
            )

        camera_value = safe_text(getattr(person, "camera", "")).strip()
        if camera_value == "":
            camera_value = default_camera

        embedding_values = list(getattr(person, "embedding", []))
        embedding_clean = [safe_float(value) for value in embedding_values]
        has_embedding = len(embedding_clean) > 0

        writer.writerow(
            [
                message_time,
                bag_time,
                frame_id,
                camera_value,
                safe_text(getattr(person, "local_id", "")),
                safe_float(getattr(position, "x", math.nan)),
                safe_float(getattr(position, "y", math.nan)),
                safe_float(getattr(position, "z", math.nan)),
                safe_float(getattr(person, "yaw", math.nan)),
                safe_float(getattr(velocity, "x", math.nan)),
                safe_float(getattr(velocity, "y", math.nan)),
                safe_float(getattr(velocity, "z", math.nan)),
                safe_float(getattr(person, "reliability", math.nan)),
                source_topic,
                "published_custom_tracked_people",
                "",
                "",
                message_time,
                has_embedding,
                len(embedding_clean),
                json.dumps(embedding_clean),
                "",
            ]
        )
        count += 1

    return count


DEBUG_EVENT_COLUMNS = [
    "event_type",
    "global_id",
    "det_index",
    "camera",
    "local_id",
    "local_id_relation",
    "association_cost",
    "appearance_distance",
    "appearance_cost",
    "geo_cost",
    "has_appearance",
    "use_appearance",
    "reid_strong",
    "reid_bad",
    "reid_extended_gate",
    "strict_threshold",
    "allowed_threshold",
    "has_embedding",
    "embedding_updated",
    "pred_x",
    "pred_y",
    "det_x",
    "det_y",
    "distance",
    "dynamic_threshold",
    "valid_candidate",
    "old_x",
    "old_y",
    "new_x",
    "new_y",
    "x",
    "y",
    "yaw",
    "hits",
    "missed",
    "age",
    "reason",
    "dropped_cam",
    "dropped_local_id",
    "dropped_x",
    "dropped_y",
    "kept_cam",
    "kept_local_id",
    "kept_x",
    "kept_y",
    "threshold",
    "max_jump",
    "max_conflict_match_dist",
    "best_global_id",
    "second_global_id",
    "best_distance",
    "second_distance",
    "track_x",
    "track_y",
    "nearby_global_id",
    "stamp",
    "latest_stamp",
]


def write_debug_payload(
    payload: Dict[str, Any],
    bag_time: float,
    source_topic: str,
    writers: Dict[str, csv.writer],
    include_debug_detections: bool,
) -> Dict[str, int]:
    payload_time = safe_float(payload.get("time"), bag_time)
    cycle = payload.get("cycle", "")
    event = payload.get("event", "")
    published_ids = payload.get("published_ids", [])
    extra = payload.get("extra", {})

    writers["events"].writerow(
        [
            payload_time,
            bag_time,
            cycle,
            event,
            payload.get("num_detections", 0),
            json.dumps(published_ids, ensure_ascii=False),
            json.dumps(extra, ensure_ascii=False),
            source_topic,
        ]
    )

    counts = {"fusion_events": 1, "debug_events": 0, "debug_detections": 0}

    if include_debug_detections:
        for detection in payload.get("detections", []):
            writers["detections"].writerow(
                [
                    payload_time,
                    bag_time,
                    "map",
                    detection.get("camera", ""),
                    detection.get("local_id", ""),
                    detection.get("x", math.nan),
                    detection.get("y", math.nan),
                    detection.get("z", math.nan),
                    detection.get("yaw", math.nan),
                    detection.get("vx", math.nan),
                    detection.get("vy", math.nan),
                    detection.get("vz", math.nan),
                    detection.get("reliability", math.nan),
                    source_topic,
                    "fusion_debug_detection",
                    cycle,
                    event,
                    detection.get("stamp", math.nan),
                    bool_or_empty(detection.get("has_embedding")),
                    detection.get("embedding_dim", ""),
                    json.dumps(detection.get("embedding", []), ensure_ascii=False),
                    detection.get("embedding_dt", math.nan),
                ]
            )
            counts["debug_detections"] += 1

    for debug_event in payload.get("debug_events", []):
        row = [
            payload_time,
            bag_time,
            cycle,
            event,
        ]
        row.extend(debug_event.get(column, "") for column in DEBUG_EVENT_COLUMNS)
        row.append(json.dumps(debug_event, ensure_ascii=False, sort_keys=True))
        writers["debug_events"].writerow(row)
        counts["debug_events"] += 1

    return counts


# ---------------------------------------------------------------------------
# Bag passes
# ---------------------------------------------------------------------------


def extract_input_bag(
    input_bag: Path,
    storage_id: str,
    output: CsvOutput,
    gt_topic: str,
    robot_pose_source: str,
    robot_odom_topic: str,
    required_robot_frame: Optional[str],
    tf_topic: str,
    tf_static_topic: str,
    map_frame: str,
    odom_frame: str,
    base_frame: str,
    tf_max_chain_skew_s: float,
    tf_dedup_tolerance_s: float,
) -> Tuple[Counter, Dict[str, str]]:
    reader = open_reader(input_bag, storage_id)
    topics = available_topics(reader)
    counts: Counter = Counter()

    if gt_topic not in topics:
        raise RuntimeError(
            f"Ground-truth topic '{gt_topic}' not found in input bag. "
            f"Available topics include: {sorted(topics)}"
        )

    if robot_pose_source == "odom" and robot_odom_topic not in topics:
        raise RuntimeError(
            f"Robot odometry topic '{robot_odom_topic}' not found in input bag."
        )

    tf_extractor = TfRobotPoseExtractor(
        map_frame=map_frame,
        odom_frame=odom_frame,
        base_frame=base_frame,
        max_chain_skew_s=tf_max_chain_skew_s,
    )
    odom_rows: List[RobotPoseRow] = []

    while reader.has_next():
        topic, serialized_data, timestamp_ns = reader.read_next()
        bag_time = float(timestamp_ns) * 1e-9

        if topic == gt_topic:
            topic_type = topics.get(topic, "")
            if topic_type == "people_msgs/msg/People" or gt_topic == "/people":
                msg = deserialize_message(serialized_data, People)
                counts["ground_truth_rows"] += write_people_ground_truth(
                    msg,
                    bag_time,
                    topic,
                    output.writers["gt"],
                )
            else:
                if Agents is None:
                    raise RuntimeError(
                        "The selected GT topic is not people_msgs/People, but "
                        "hunav_msgs/msg/Agents is unavailable. Source/install hunav_msgs "
                        "or use a People ground-truth topic."
                    )
                msg = deserialize_message(serialized_data, Agents)
                counts["ground_truth_rows"] += write_hunav_ground_truth(
                    msg,
                    bag_time,
                    topic,
                    output.writers["gt"],
                )
            continue

        if robot_pose_source == "odom" and topic == robot_odom_topic:
            msg = deserialize_message(serialized_data, Odometry)
            row = odometry_to_robot_pose(
                msg,
                bag_time,
                required_frame=required_robot_frame,
                source_topic=topic,
            )
            if row is None:
                counts["robot_odom_rows_rejected_wrong_frame"] += 1
            else:
                odom_rows.append(row)
            continue

        if robot_pose_source == "tf" and topic in {tf_topic, tf_static_topic}:
            msg = deserialize_message(serialized_data, TFMessage)
            tf_extractor.process_message(msg, bag_time, topic)

    if robot_pose_source == "odom":
        robot_rows = sorted(odom_rows, key=lambda row: row.time)
    else:
        robot_rows = tf_extractor.finalized_samples(tf_dedup_tolerance_s)

    if not robot_rows:
        raise RuntimeError(
            "No valid robot poses were extracted. Check --robot-pose-source, "
            "the selected topic/frame names, and bag contents."
        )

    counts["robot_pose_rows"] = write_robot_rows(
        robot_rows,
        output.writers["robot"],
    )
    return counts, topics


def extract_output_bag(
    output_bag: Path,
    storage_id: str,
    output: CsvOutput,
    tracked_people_topic: str,
    local_topic_to_camera: Dict[str, str],
    debug_topic: str,
    include_debug_detections: bool,
) -> Tuple[Counter, Dict[str, str]]:
    reader = open_reader(output_bag, storage_id)
    topics = available_topics(reader)
    counts: Counter = Counter()

    if tracked_people_topic not in topics:
        raise RuntimeError(
            f"Published global-track topic '{tracked_people_topic}' not found in output bag. "
            "The validation must use the actual /tracked_people output, not only internal debug tracks."
        )

    if debug_topic not in topics:
        counts["debug_topic_missing"] = 1

    # Resolve each message class from the type stored in the bag metadata.
    # The local topics are not assumed to be people_msgs/msg/People: in some
    # workspaces they are custom People-like messages with the same fields.
    topic_message_classes: Dict[str, Any] = {}
    topics_to_resolve = {tracked_people_topic, debug_topic, *local_topic_to_camera.keys()}
    for configured_topic in topics_to_resolve:
        recorded_type = topics.get(configured_topic)
        if recorded_type is None:
            continue
        try:
            topic_message_classes[configured_topic] = get_message(recorded_type)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot load ROS message type '{recorded_type}' recorded for "
                f"topic '{configured_topic}'. Source/install the package that "
                f"defines this message before running the extractor. Original error: {exc}"
            ) from exc

    print("Recorded output topic types used by the extractor:")
    for configured_topic in sorted(topics_to_resolve):
        if configured_topic in topics:
            print(f"  {configured_topic}: {topics[configured_topic]}")

    while reader.has_next():
        topic, serialized_data, timestamp_ns = reader.read_next()
        bag_time = float(timestamp_ns) * 1e-9

        if topic == tracked_people_topic:
            msg = deserialize_message(serialized_data, topic_message_classes[topic])
            counts["global_track_rows"] += write_published_global_tracks(
                msg,
                bag_time,
                topic,
                output.writers["tracks"],
            )
            counts["global_track_messages"] += 1
            continue

        if topic in local_topic_to_camera:
            message_class = topic_message_classes.get(topic)
            if message_class is None:
                counts[f"missing_message_class:{topic}"] += 1
                continue
            try:
                msg = deserialize_message(serialized_data, message_class)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to deserialize local topic '{topic}'. The bag reports "
                    f"type '{topics.get(topic)}'. Make sure the same message package "
                    f"definition used while recording the bag is currently sourced. "
                    f"Original error: {exc}"
                ) from exc
            counts["local_detection_rows"] += write_local_people(
                msg,
                bag_time,
                topic,
                local_topic_to_camera[topic],
                output.writers["detections"],
            )
            counts["local_people_messages"] += 1
            continue

        if topic == debug_topic:
            msg = deserialize_message(serialized_data, topic_message_classes[topic])
            try:
                payload = json.loads(msg.data)
            except Exception:
                counts["invalid_debug_json_messages"] += 1
                continue

            debug_counts = write_debug_payload(
                payload,
                bag_time,
                topic,
                output.writers,
                include_debug_detections=include_debug_detections,
            )
            counts.update(debug_counts)
            counts["debug_messages"] += 1

    return counts, topics


# ---------------------------------------------------------------------------
# Validation of extracted tables and summary
# ---------------------------------------------------------------------------


def csv_data_row_count(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return max(0, sum(1 for _ in stream) - 1)


def inspect_frame_values(path: Path) -> List[str]:
    values = set()
    with path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            value = normalize_frame(row.get("frame_id", ""))
            if value:
                values.add(value)
    return sorted(values)


def build_extraction_summary(
    results_dir: Path,
    output: CsvOutput,
    args: argparse.Namespace,
    input_counts: Counter,
    output_counts: Counter,
    input_topics: Dict[str, str],
    output_topics: Dict[str, str],
) -> Path:
    csv_counts = {
        key: csv_data_row_count(path)
        for key, path in output.paths.items()
    }

    summary = {
        "input_bag": str((Path(args.rosbags_dir) / args.input_bag).resolve()),
        "output_bag": str((Path(args.rosbags_dir) / args.output_bag).resolve()),
        "results_dir": str(results_dir.resolve()),
        "storage_id": args.storage_id,
        "configuration": {
            "gt_topic": args.gt_topic,
            "robot_pose_source": args.robot_pose_source,
            "robot_odom_topic": args.robot_odom_topic,
            "required_robot_frame": args.required_robot_frame,
            "map_frame": args.map_frame,
            "odom_frame": args.odom_frame,
            "base_frame": args.base_frame,
            "tracked_people_topic": args.tracked_people_topic,
            "local_topics": {
                "center": args.people_center_topic,
                "left": args.people_left_topic,
                "right": args.people_right_topic,
            },
            "debug_topic": args.debug_topic,
            "include_debug_detections": args.include_debug_detections,
        },
        "input_counts": dict(input_counts),
        "output_counts": dict(output_counts),
        "csv_row_counts": csv_counts,
        "ground_truth_frames": inspect_frame_values(output.paths["gt"]),
        "robot_pose_frames": inspect_frame_values(output.paths["robot"]),
        "global_track_frames": inspect_frame_values(output.paths["tracks"]),
        "input_topics": input_topics,
        "output_topics": output_topics,
        "methodological_notes": [
            "global_tracks_validation.csv contains only actual /tracked_people publications",
            "agent_id and global_id are intentionally kept separate",
            "visibility and GT-track association are not computed by this extractor",
            "local detections are diagnostic and do not define geometric visibility",
        ],
    }

    summary_path = results_dir / "extraction_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    return summary_path


def validate_extracted_outputs(output: CsvOutput, required_robot_frame: Optional[str]) -> List[str]:
    warnings: List[str] = []

    counts = {
        key: csv_data_row_count(path)
        for key, path in output.paths.items()
    }

    for key in ("gt", "robot", "tracks"):
        if counts[key] == 0:
            warnings.append(f"Critical CSV '{output.paths[key].name}' contains no data rows.")

    robot_frames = inspect_frame_values(output.paths["robot"])
    track_frames = inspect_frame_values(output.paths["tracks"])
    gt_frames = inspect_frame_values(output.paths["gt"])

    if len(robot_frames) > 1:
        warnings.append(
            "robot_pose_validation.csv contains multiple frames: "
            + ", ".join(robot_frames)
        )

    if required_robot_frame:
        normalized_required = normalize_frame(required_robot_frame)
        if robot_frames and robot_frames != [normalized_required]:
            warnings.append(
                f"Expected robot pose frame '{normalized_required}', found {robot_frames}."
            )

    if gt_frames and track_frames and set(gt_frames).isdisjoint(track_frames):
        warnings.append(
            "GT and global-track frames do not overlap. Transform them to the same fixed frame before validation."
        )

    return warnings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract raw CSV tables for visibility-episode tracking validation."
    )

    parser.add_argument("--rosbags-dir", default=str(DEFAULT_ROSBAGS_DIR))
    parser.add_argument("--input-bag", default=DEFAULT_INPUT_BAG)
    parser.add_argument("--output-bag", default=DEFAULT_OUTPUT_BAG)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--storage-id",
        default="sqlite3",
        help="rosbag2 storage plugin, normally sqlite3 or mcap.",
    )

    parser.add_argument(
        "--gt-topic",
        default="/human_states",
        help="Ground truth: hunav_msgs/Agents or people_msgs/People.",
    )

    parser.add_argument(
        "--robot-pose-source",
        choices=("odom", "tf"),
        default="odom",
        help="Use one canonical source only; never mix odom-frame and map-frame poses.",
    )
    parser.add_argument(
        "--robot-odom-topic",
        default="/jackal/ground_truth",
        help="Odometry topic used when --robot-pose-source odom.",
    )
    parser.add_argument(
        "--required-robot-frame",
        default="map",
        help="Reject odometry rows not expressed in this frame. Use an empty string to disable the check.",
    )
    parser.add_argument("--tf-topic", default="/tf")
    parser.add_argument("--tf-static-topic", default="/tf_static")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--tf-max-chain-skew-s", type=float, default=0.20)
    parser.add_argument("--tf-dedup-tolerance-s", type=float, default=1e-3)

    parser.add_argument("--tracked-people-topic", default="/tracked_people")
    parser.add_argument("--people-center-topic", default="/people_center")
    parser.add_argument("--people-left-topic", default="/people_left")
    parser.add_argument("--people-right-topic", default="/people_right")
    parser.add_argument("--debug-topic", default="/tracked_people_debug")
    parser.add_argument(
        "--include-debug-detections",
        action="store_true",
        help=(
            "Also append detections embedded in /tracked_people_debug to the local-detection CSV. "
            "By default only /people_center, /people_left and /people_right are used, avoiding duplicates."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rosbags_dir = Path(args.rosbags_dir).expanduser().resolve()
    input_bag = rosbags_dir / args.input_bag
    output_bag = rosbags_dir / args.output_bag
    results_dir = rosbags_dir / args.results_dir

    if not input_bag.exists():
        raise FileNotFoundError(f"Input bag not found: {input_bag}")
    if not output_bag.exists():
        raise FileNotFoundError(f"Output bag not found: {output_bag}")

    required_robot_frame = args.required_robot_frame.strip() or None
    local_topic_to_camera = {
        args.people_center_topic: "center",
        args.people_left_topic: "left",
        args.people_right_topic: "right",
    }

    output = initialize_csv_outputs(results_dir)
    try:
        print(f"Extracting GT and robot pose from: {input_bag}")
        input_counts, input_topics = extract_input_bag(
            input_bag=input_bag,
            storage_id=args.storage_id,
            output=output,
            gt_topic=args.gt_topic,
            robot_pose_source=args.robot_pose_source,
            robot_odom_topic=args.robot_odom_topic,
            required_robot_frame=required_robot_frame,
            tf_topic=args.tf_topic,
            tf_static_topic=args.tf_static_topic,
            map_frame=args.map_frame,
            odom_frame=args.odom_frame,
            base_frame=args.base_frame,
            tf_max_chain_skew_s=args.tf_max_chain_skew_s,
            tf_dedup_tolerance_s=args.tf_dedup_tolerance_s,
        )

        print(f"Extracting published tracks and debug data from: {output_bag}")
        output_counts, output_topics = extract_output_bag(
            output_bag=output_bag,
            storage_id=args.storage_id,
            output=output,
            tracked_people_topic=args.tracked_people_topic,
            local_topic_to_camera=local_topic_to_camera,
            debug_topic=args.debug_topic,
            include_debug_detections=args.include_debug_detections,
        )
    finally:
        close_csv_outputs(output)

    summary_path = build_extraction_summary(
        results_dir=results_dir,
        output=output,
        args=args,
        input_counts=input_counts,
        output_counts=output_counts,
        input_topics=input_topics,
        output_topics=output_topics,
    )

    warnings = validate_extracted_outputs(output, required_robot_frame)

    print("\nExtraction completed.")
    print(f"Results directory: {results_dir}")
    print("\nGenerated files:")
    for path in output.paths.values():
        print(f"  {path}")
    print(f"  {summary_path}")

    print("\nRow counts:")
    for key, path in output.paths.items():
        print(f"  {key}: {csv_data_row_count(path)}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\nBasic frame and content checks passed.")

    print(
        "\nNext step: run the separate visibility-episode validation script "
        "using ground_truth_people.csv, robot_pose_validation.csv and "
        "global_tracks_validation.csv."
    )


if __name__ == "__main__":
    main()
