import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from rclpy.node import Node
import yaml
from hunav_msgs.msg import Agents
import tf_transformations


@dataclass
class AgentObservation:
    key: str
    label: str
    numeric_id: Optional[int]
    position: Tuple[float, float, float]
    yaw: Optional[float] = None
    velocity: Tuple[Optional[float], Optional[float], Optional[float]] = (
        None,
        None,
        None,
    )
    speed: Optional[float] = None


class PeopleTrajectoryRecorder:
    """Capture and serialize per-person trajectories from a ROS topic."""

    def __init__(
        self,
        node: Node,
    ) -> None:
        self._node = node
        self._node.declare_parameter("people_trajectory_sample_period_sec", 0.25)
        self._min_sample_period = (
            self._node.get_parameter("people_trajectory_sample_period_sec")
            .get_parameter_value()
            .double_value
        )
        self._node.declare_parameter("people_trajectory_topic", "human_states")
        topic_param = (
            self._node.get_parameter("people_trajectory_topic")
            .get_parameter_value()
            .string_value
        )
        self._topic_name = topic_param or "human_states"
        self._lock = threading.Lock()
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._start_stamp_sec: Optional[float] = None
        self._last_stamp_sec: Optional[float] = None
        self._creation_wall_time = datetime.now(timezone.utc)
        self._start_wall_time: Optional[datetime] = None
        self._start_wall_monotonic: Optional[float] = None
        self._last_relative_time: float = 0.0
        self._message_count = 0
        self._source_frame: Optional[str] = None
        self._first_sample_event = threading.Event()
        self._sub = node.create_subscription(
            Agents, self._topic_name, self._callback, 10  # type: ignore[arg-type]
        )
        self._logger = node.get_logger()
        self._logger.info(
            f"Recording people trajectories from '{self._topic_name}' "
            f"({self._min_sample_period:.2f} s period)"
        )

    @property
    def elapsed_time_sec(self) -> float:
        if self._start_wall_monotonic is None:
            return 0.0
        return max(0.0, time.monotonic() - self._start_wall_monotonic)

    @property
    def first_sample_received(self) -> bool:
        return self._first_sample_event.is_set()

    @property
    def topic_name(self) -> str:
        return self._topic_name

    def wait_for_first_sample(self, timeout: Optional[float] = None) -> bool:
        return self._first_sample_event.wait(timeout=timeout)

    def _callback(self, msg: Any) -> None:
        observations = self._extract_observations(msg)
        if not observations:
            return
        stamp_sec = self._extract_stamp_sec(msg)
        relative_time = self._compute_relative_time(stamp_sec)
        wall_iso = datetime.now(timezone.utc).isoformat()
        header = getattr(msg, "header", None)
        frame_id = getattr(header, "frame_id", "") if header else ""
        if frame_id and not self._source_frame:
            self._source_frame = frame_id
        self._message_count += 1
        if not self._first_sample_event.is_set():
            self._start_wall_time = datetime.now(timezone.utc)
            self._start_wall_monotonic = time.monotonic()
            self._first_sample_event.set()
            self._logger.info(
                f"First people trajectory sample received on '{self._topic_name}'"
            )
        with self._lock:
            for obs in observations:
                sample = self._build_sample_dict(
                    obs,
                    relative_time=relative_time,
                    stamp_sec=stamp_sec,
                    wall_iso=wall_iso,
                )
                self._last_relative_time = relative_time
                entry = self._agents.setdefault(
                    obs.key,
                    {
                        "key": obs.key,
                        "label": obs.label,
                        "id": obs.numeric_id,
                        "samples": [],
                    },
                )
                entry["samples"].append(sample)

    def _extract_stamp_sec(self, msg: Any) -> Optional[float]:
        header = getattr(msg, "header", None)
        stamp = getattr(header, "stamp", None) if header else None
        if stamp is None:
            return None
        try:
            return float(stamp.sec) + float(stamp.nanosec) / 1e9
        except AttributeError:
            return None

    def _compute_relative_time(self, stamp_sec: Optional[float]) -> float:
        if stamp_sec is not None:
            if self._start_stamp_sec is None:
                self._start_stamp_sec = stamp_sec
            rel = max(stamp_sec - self._start_stamp_sec, 0.0)
            self._last_stamp_sec = stamp_sec
            return rel
        now = time.monotonic()
        if self._start_wall_monotonic is None:
            self._start_wall_monotonic = now
        return max(now - self._start_wall_monotonic, 0.0)

    def _extract_observations(self, msg: Any) -> List[AgentObservation]:
        observations: List[AgentObservation] = []
        agents = getattr(msg, "agents", [])
        for idx, agent in enumerate(agents):
            pose = getattr(agent, "position", None)
            if pose is None or not hasattr(pose, "position"):
                continue
            position = pose.position
            x = float(getattr(position, "x", 0.0))
            y = float(getattr(position, "y", 0.0))
            z = float(getattr(position, "z", 0.0))
            yaw = getattr(agent, "yaw", None)
            if yaw is None and hasattr(pose, "orientation"):
                _, _, yaw = tf_transformations.euler_from_quaternion(
                    pose.orientation,
                )
            velocity = getattr(agent, "velocity", None)
            vx = (
                float(getattr(getattr(velocity, "linear", None), "x", 0.0))
                if velocity
                else None
            )
            vy = (
                float(getattr(getattr(velocity, "linear", None), "y", 0.0))
                if velocity
                else None
            )
            vz = (
                float(getattr(getattr(velocity, "linear", None), "z", 0.0))
                if velocity
                else None
            )
            speed = None
            if vx is not None or vy is not None:
                speed = math.hypot(vx or 0.0, vy or 0.0)
            numeric_id = getattr(agent, "id", None)
            label_base = getattr(agent, "name", "") or ""
            key = (
                f"agent_{numeric_id}"
                if numeric_id is not None
                else (label_base or f"agent_{idx}")
            )
            label = label_base or key
            observations.append(
                AgentObservation(
                    key=key,
                    label=label,
                    numeric_id=int(numeric_id) if numeric_id is not None else None,
                    position=(x, y, z),
                    yaw=float(yaw) if yaw is not None else None,
                    velocity=(vx, vy, vz),
                    speed=speed,
                )
            )
        return observations

    def _build_sample_dict(
        self,
        obs: AgentObservation,
        *,
        relative_time: float,
        stamp_sec: Optional[float],
        wall_iso: str,
    ) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "t_sec": float(relative_time),
            "position": [
                float(obs.position[0]),
                float(obs.position[1]),
                float(obs.position[2]),
            ],
            "wall_time": wall_iso,
        }
        if stamp_sec is not None:
            entry["stamp_sec"] = float(stamp_sec)
        if obs.yaw is not None:
            entry["yaw"] = float(obs.yaw)
        vx, vy, vz = obs.velocity
        if any(component is not None for component in (vx, vy, vz)):
            entry["velocity"] = [
                float(vx) if vx is not None else 0.0,
                float(vy) if vy is not None else 0.0,
                float(vz) if vz is not None else 0.0,
            ]
        if obs.speed is not None:
            entry["speed"] = float(obs.speed)
        return entry

    def export(
        self,
        *,
        destination: Path,
        experiment_tag: str,
        run_id: int,
        stage: str,
    ) -> Optional[Path]:
        data = self._snapshot(experiment_tag, run_id, stage)
        if not data["agents"]:
            self._logger.info(
                f"No people trajectory samples captured from {self._topic_name}; skipping export"
            )
            return None
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, sort_keys=False)
        self._logger.info(f"People trajectories written to {destination}")
        return destination

    def _snapshot(self, experiment_tag: str, run_id: int, stage: str) -> Dict[str, Any]:
        with self._lock:
            agents_payload: List[Dict[str, Any]] = []
            total_samples = 0
            for entry in self._agents.values():
                samples: List[Dict[str, Any]] = entry["samples"]
                if not samples:
                    continue
                payload = {
                    "key": entry["key"],
                    "label": entry["label"],
                    "samples": samples,
                    "sample_count": len(samples),
                }
                if entry.get("id") is not None:
                    payload["id"] = entry["id"]
                agents_payload.append(payload)
                total_samples += len(samples)

            meta = {
                "experiment_tag": experiment_tag,
                "run_id": run_id,
                "stage": stage,
                "start_wall_time": (
                    (self._start_wall_time or self._creation_wall_time).isoformat()
                ),
                "sample_period_sec": self._min_sample_period,
                "message_count": self._message_count,
                "source_frame": self._source_frame,
                "end_relative_time": self._last_relative_time,
                "start_stamp_sec": self._start_stamp_sec,
                "end_stamp_sec": self._last_stamp_sec,
                "total_samples": total_samples,
            }
        return {"meta": meta, "agents": agents_payload}
