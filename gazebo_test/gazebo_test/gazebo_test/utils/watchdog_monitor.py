"""Watchdog support utilities for experiment orchestration.

This module encapsulates the logic for monitoring critical ROS nodes and
reacting to recurring log errors so the main experiment manager can remain
focused on job scheduling concerns.
"""

from __future__ import annotations

import asyncio
import fnmatch
import time
from collections import defaultdict, deque
from typing import Awaitable, Callable, Deque, Dict, Iterable, List, Optional, Set, Tuple, Union

from rcl_interfaces.msg import Log as RosoutLog
from rclpy.node import Node
from rclpy.qos import QoSProfile


LogLevelValue = Union[int, float, str, bytes, bytearray]
MissingNodesCallback = Callable[[Set[str]], Awaitable[None]]
RawPatternInput = Union[str, Iterable[str], None]


def coerce_ros_log_level(value: LogLevelValue) -> int:
    """Normalise ROS log level values to plain integers."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (bytes, bytearray)):
        return int.from_bytes(value, byteorder="little", signed=False)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        cleaned = value.strip().lower()
        lookup = {
            "debug": 10,
            "info": 20,
            "warn": 30,
            "warning": 30,
            "error": 40,
            "fatal": 50,
        }
        if cleaned in lookup:
            return lookup[cleaned]
        if cleaned.isdigit():
            return int(cleaned)
        raise ValueError(f"Unsupported log level string '{value}'")
    raise TypeError(f"Unsupported log level type: {type(value)!r}")


def _default_error_level() -> int:
    try:
        return coerce_ros_log_level(getattr(RosoutLog, "ERROR"))
    except Exception:  # noqa: BLE001
        return 40


class ExperimentWatchdog:
    """Monitor ROS graph health and repeated log errors for experiments."""

    def __init__(
        self,
        *,
        node: Node,
        required_nodes: Iterable[str],
        startup_grace_sec: float,
        raw_error_patterns: RawPatternInput,
        error_threshold: int,
        error_window_sec: float,
        error_min_level: Optional[LogLevelValue],
        on_missing_nodes: MissingNodesCallback,
    ) -> None:
        self._node = node
        self._required_nodes: List[str] = [str(p).strip() for p in required_nodes if str(p).strip()]
        self._startup_grace_sec = max(startup_grace_sec, 0.0)
        self._on_missing_nodes = on_missing_nodes

        self._module_health_event: asyncio.Event = asyncio.Event()
        self._module_health_event.set()
        self._module_failure_active = False
        self._missing_modules: Set[str] = set()
        self._last_reported_missing_nodes: Set[str] = set()
        self._waiting_for_modules_logged = False
        self._monitor_started_at = 0.0
        self._asyncio_loop: Optional[asyncio.AbstractEventLoop] = None

        self._error_patterns = self._parse_watchdog_error_patterns(raw_error_patterns)
        self._error_threshold = max(1, int(error_threshold) if error_threshold is not None else 1)
        self._error_window_sec = max(0.5, float(error_window_sec) if error_window_sec is not None else 5.0)

        if error_min_level is None:
            self._error_min_level = _default_error_level()
        else:
            try:
                self._error_min_level = coerce_ros_log_level(error_min_level)
            except Exception as exc:  # noqa: BLE001
                self._node.get_logger().warning(
                    f"Invalid watchdog_error_min_level value '{error_min_level}': {exc}; using default"
                )
                self._error_min_level = _default_error_level()

        self._log_pattern_windows: Dict[str, Deque[float]] = defaultdict(deque)
        self._active_log_faults: Set[str] = set()
        self._rosout_subscription = None
        if self._error_patterns:
            qos_profile = QoSProfile(depth=100)
            self._rosout_subscription = node.create_subscription(
                RosoutLog,
                "/rosout",
                self._rosout_callback,
                qos_profile=qos_profile,
            )
            pattern_str = ", ".join(
                f"{node_filter or '*'}|{pattern}" for _, node_filter, pattern in self._error_patterns
            )
            self._node.get_logger().info(f"Log watchdog active for patterns: {pattern_str}")

    @property
    def health_event(self) -> asyncio.Event:
        return self._module_health_event

    @property
    def waiting_logged(self) -> bool:
        return self._waiting_for_modules_logged

    def set_waiting_logged(self, value: bool) -> None:
        self._waiting_for_modules_logged = value

    def set_asyncio_loop(self, loop: Optional[asyncio.AbstractEventLoop]) -> None:
        self._asyncio_loop = loop

    def start_monitoring(self) -> None:
        self._monitor_started_at = time.monotonic()

    async def evaluate_module_health(self) -> None:
        if not self._required_nodes:
            if not self._module_health_event.is_set():
                self._module_health_event.set()
                self._waiting_for_modules_logged = False
            return
        present_nodes = self._collect_node_names()
        missing = self._missing_required_nodes(present_nodes)
        if missing:
            self._missing_modules = missing
            if not self._module_failure_active:
                elapsed = 0.0
                if self._monitor_started_at > 0.0:
                    elapsed = time.monotonic() - self._monitor_started_at
                if elapsed < self._startup_grace_sec:
                    return
                await self._handle_missing_required_nodes(missing)
            else:
                if self._module_health_event.is_set():
                    self._module_health_event.clear()
                if missing != self._last_reported_missing_nodes:
                    self._last_reported_missing_nodes = set(missing)
                    self._node.get_logger().error(
                        "Required modules still missing: "
                        + ", ".join(sorted(missing))
                    )
            return
        if self._module_failure_active:
            self._module_failure_active = False
            self._missing_modules.clear()
            self._last_reported_missing_nodes = set()
            if not self._module_health_event.is_set():
                self._node.get_logger().info(
                    "All required modules detected; resuming job dispatch"
                )
                self._module_health_event.set()
            self._waiting_for_modules_logged = False
            self._active_log_faults.clear()
            for window in self._log_pattern_windows.values():
                window.clear()
            return
        if not self._module_health_event.is_set():
            self._module_health_event.set()
            self._waiting_for_modules_logged = False

    async def _handle_missing_required_nodes(self, missing: Set[str]) -> None:
        reason = ", ".join(sorted(missing))
        self._module_failure_active = True
        self._last_reported_missing_nodes = set(missing)
        if self._module_health_event.is_set():
            self._module_health_event.clear()
        self._waiting_for_modules_logged = False
        self._node.get_logger().error(
            f"Critical modules missing from ROS graph: {reason}; pausing job dispatch"
        )
        await self._safe_call_on_missing_nodes(missing)

    async def _safe_call_on_missing_nodes(self, missing: Set[str]) -> None:
        try:
            await self._on_missing_nodes(missing)
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().error(f"Watchdog recovery callback failed: {exc}")

    def _collect_node_names(self) -> Set[str]:
        nodes: Set[str] = set()
        try:
            discovered = self._node.get_node_names_and_namespaces()
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().debug(f"Failed to query node graph: {exc}")
            return nodes
        for name, namespace in discovered:
            full_name = self._fully_qualified_name(namespace, name)
            nodes.add(name)
            nodes.add(full_name)
            nodes.add(full_name.lstrip("/"))
        return nodes

    @staticmethod
    def _fully_qualified_name(namespace: str, name: str) -> str:
        if not namespace or namespace == "/":
            return f"/{name}"
        cleaned = namespace
        if not cleaned.startswith("/"):
            cleaned = f"/{cleaned}"
        cleaned = cleaned.rstrip("/")
        return f"{cleaned}/{name}"

    def _missing_required_nodes(self, present: Set[str]) -> Set[str]:
        missing: Set[str] = set()
        for pattern in self._required_nodes:
            if not pattern:
                continue
            if not any(
                self._node_pattern_matches(pattern, candidate) for candidate in present
            ):
                missing.add(pattern)
        return missing

    @staticmethod
    def _node_pattern_matches(pattern: str, candidate: str) -> bool:
        if fnmatch.fnmatch(candidate, pattern):
            return True
        if pattern.startswith("/") and fnmatch.fnmatch(candidate, pattern.lstrip("/")):
            return True
        if not pattern.startswith("/") and fnmatch.fnmatch(f"/{candidate}", pattern):
            return True
        return False

    def _parse_watchdog_error_patterns(
        self,
        raw_patterns: RawPatternInput,
    ) -> List[Tuple[str, str, str]]:
        patterns: List[Tuple[str, str, str]] = []
        if raw_patterns is None:
            return patterns
        if isinstance(raw_patterns, str):
            entries = [raw_patterns]
        elif isinstance(raw_patterns, (list, tuple)):
            entries = list(raw_patterns)
        else:
            return patterns
        seen: Set[str] = set()
        for entry in entries:
            text = str(entry).strip()
            if not text:
                continue
            if "|" in text:
                node_filter, substring = text.split("|", 1)
            else:
                node_filter, substring = "", text
            normalized_node = node_filter.strip()
            normalized_pattern = substring.strip()
            key = (
                f"{normalized_node}|{normalized_pattern}"
                if normalized_node
                else normalized_pattern
            )
            if key in seen:
                continue
            seen.add(key)
            patterns.append((key, normalized_node, normalized_pattern))
        return patterns

    def _rosout_callback(self, log_msg: RosoutLog) -> None:
        if not self._error_patterns:
            return
        if log_msg.level < self._error_min_level:
            return
        node_name = log_msg.name or ""
        message_text = log_msg.msg or ""
        timestamp = float(log_msg.stamp.sec) + float(log_msg.stamp.nanosec) * 1e-9
        lowered_message = message_text.lower()
        for pattern_key, node_filter, substring in self._error_patterns:
            if node_filter and not self._node_pattern_matches(node_filter, node_name):
                continue
            if substring and substring.lower() not in lowered_message:
                continue
            self._handle_log_pattern_hit(
                pattern_key,
                timestamp,
                node_name,
                message_text,
                substring,
            )

    def _handle_log_pattern_hit(
        self,
        pattern_key: str,
        timestamp: float,
        node_name: str,
        message_text: str,
        pattern_description: str,
    ) -> None:
        window = self._log_pattern_windows[pattern_key]
        window.append(timestamp)
        cutoff = timestamp - self._error_window_sec
        while window and window[0] < cutoff:
            window.popleft()
        if pattern_key in self._active_log_faults:
            return
        if len(window) < self._error_threshold:
            return
        display_pattern = pattern_description or pattern_key
        reason = (
            f"Log errors from '{node_name or 'unknown'}' matched '{display_pattern}'"
            f" ({self._error_threshold} hits in {self._error_window_sec:.1f}s)"
        )
        self._node.get_logger().error(
            f"Watchdog detected repeated log errors from '{node_name or 'unknown'}': {message_text}"
        )
        self._active_log_faults.add(pattern_key)
        window.clear()
        self._schedule_missing_nodes({reason})

    def _schedule_missing_nodes(self, missing: Set[str]) -> None:
        if self._asyncio_loop is None:
            self._node.get_logger().warning(
                "Asyncio loop unavailable to process watchdog fault for pattern(s): "
                + ", ".join(sorted(missing))
            )
            return
        asyncio.run_coroutine_threadsafe(
            self._handle_missing_required_nodes(missing),
            self._asyncio_loop,
        )

    async def stop(self) -> None:
        if self._rosout_subscription is not None:
            self._node.destroy_subscription(self._rosout_subscription)
            self._rosout_subscription = None
        self._log_pattern_windows.clear()
        self._active_log_faults.clear()