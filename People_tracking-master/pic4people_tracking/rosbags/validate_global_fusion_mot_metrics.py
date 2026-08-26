#!/usr/bin/env python3
"""
validate_tracking_visibility_episodes.py

Offline validation of a multi-camera global people-tracking system using:
- robot pose in map frame;
- simulated-agent ground truth;
- published global tracks.

Expected input CSVs (long format)
---------------------------------
Robot pose:
    time, x, y, yaw

Ground truth:
    time, agent_id, x, y

Global tracks:
    time, global_id, x, y

The script:
1. Interpolates robot pose at GT timestamps.
2. Reconstructs the moving union of three camera FOVs.
3. Marks each GT agent as visible/not visible.
4. Associates visible GT agents using the GT agent name as the persistent reference, previous Global-ID continuity, and Hungarian distance.
5. Builds raw visibility segments.
6. Merges segments separated by <= episode_merge_gap_s into one episode.
7. Computes per-sample, per-ID, per-episode, per-agent, and overall metrics.
8. Generates CSV tables and all main validation plots.

Important fusion-component rules
--------------------------------
1. Geometric out-of-range / combined-FOV loss is diagnostic information only:
   it does NOT automatically excuse an identity change.
2. A gap between consecutive VALID LOCAL fusion inputs longer than the fusion
   reactivation horizon (reactivation_max_age_s, default 5.0 s) starts a new
   identity-continuity episode. The old Global ID is no longer recoverable by
   the implemented fusion policy, so a later ID is not scored as a recovery
   failure.
3. If an identity discontinuity is preceded by a local measurement that the
   fusion node ACCEPTED even though that measurement is an outlier with respect
   to every GT person, the event is retained as an observed end-to-end
   discontinuity but is labelled UPSTREAM_LOCALIZATION_INDUCED and is excluded
   from the fusion-attributable switch count.

Example
-------
python3 validate_tracking_visibility_episodes.py \
  --robot-csv robot_pose_validation.csv \
  --gt-csv ground_truth_people.csv \
  --tracks-csv global_tracks_validation.csv \
  --output-dir validation_results \
  --camera-range 4.0 \
  --camera-hfov-deg 69.0 \
  --left-yaw-deg 58.0 \
  --center-yaw-deg 0.0 \
  --right-yaw-deg -58.0 \
  --episode-merge-gap-s 6.0 \
  --association-max-distance 0.75 \
  --track-time-tolerance-s 0.20
"""

from __future__ import annotations
import re

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Wedge
from scipy.optimize import linear_sum_assignment


BIG_COST = 1e9
EPS = 1e-12

COMMON_COLUMNS: Dict[str, Sequence[str]] = {
    "time": (
        "time", "timestamp", "t", "stamp", "time_sec", "timestamp_sec",
        "header_stamp", "sim_time", "ros_time", "sec",
    ),
    "robot_x": (
        "robot_x", "x", "pos_x", "position_x", "map_x", "pose_x",
    ),
    "robot_y": (
        "robot_y", "y", "pos_y", "position_y", "map_y", "pose_y",
    ),
    "robot_yaw": (
        "robot_yaw", "yaw", "theta", "heading", "orientation_yaw", "pose_yaw",
    ),
    "agent_id": (
        "agent_id", "agent_name", "name", "person_id", "human_id", "gt_id",
        "agent", "human_name",
    ),
    "gt_x": (
        "gt_x", "x", "pos_x", "position_x", "map_x", "agent_x",
    ),
    "gt_y": (
        "gt_y", "y", "pos_y", "position_y", "map_y", "agent_y",
    ),
    "global_id": (
        "global_id", "track_id", "id", "person_id", "tracked_id",
        "global_track_id",
    ),
    "track_x": (
        "track_x", "x", "pos_x", "position_x", "map_x", "global_x",
    ),
    "track_y": (
        "track_y", "y", "pos_y", "position_y", "map_y", "global_y",
    ),
    "track_agent_name": (
        "gt_agent_name", "agent_name", "agent_id", "ground_truth_name",
        "ground_truth_id", "sim_agent_name", "human_name", "name",
    ),
}


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den > EPS else 0.0


def normalize_identity(value: object) -> Optional[str]:
    """Normalize optional simulator/GT identity labels."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "unknown"}:
        return None
    return text


def identities_compatible(gt_agent: object, track_agent: object) -> bool:
    """Unknown track identity is allowed; a known one must match the GT name."""
    gt_name = normalize_identity(gt_agent)
    track_name = normalize_identity(track_agent)
    return track_name is None or gt_name == track_name


def detect_column(
    df: pd.DataFrame,
    logical_name: str,
    explicit: Optional[str] = None,
) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(
                f"Column '{explicit}' requested for {logical_name} not found. "
                f"Available columns: {list(df.columns)}"
            )
        return explicit

    normalized = {str(c).strip().lower(): str(c) for c in df.columns}
    for candidate in COMMON_COLUMNS[logical_name]:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    raise ValueError(
        f"Could not auto-detect '{logical_name}'. "
        f"Available columns: {list(df.columns)}. "
        f"Use the corresponding --*-col argument."
    )


def estimate_nominal_dt(times: np.ndarray, fallback: float = 0.1) -> float:
    values = np.sort(np.unique(times[np.isfinite(times)]))
    if len(values) < 2:
        return fallback
    diffs = np.diff(values)
    diffs = diffs[diffs > EPS]
    return float(np.median(diffs)) if len(diffs) else fallback


def sample_weights(
    times: np.ndarray,
    nominal_dt: float,
    max_factor: float = 2.5,
) -> np.ndarray:
    """
    Duration represented by each sample.

    Long logging gaps are clipped so a missing data block is not counted as
    continuous visibility.
    """
    if len(times) == 0:
        return np.array([], dtype=float)
    if len(times) == 1:
        return np.array([nominal_dt], dtype=float)

    diffs = np.diff(times)
    max_weight = max(nominal_dt * max_factor, nominal_dt)
    weights = np.clip(diffs, 0.0, max_weight)
    return np.concatenate([weights, [nominal_dt]])


@dataclass(frozen=True)
class CameraModel:
    name: str
    relative_yaw_rad: float
    hfov_rad: float
    max_range_m: float
    min_range_m: float = 0.0
    offset_x_m: float = 0.0
    offset_y_m: float = 0.0


@dataclass
class VisibilitySegment:
    agent_id: str
    segment_id: int
    start_time: float
    end_time: float
    row_indices: List[int]


@dataclass
class VisibilityEpisode:
    agent_id: str
    episode_id: int
    start_time: float
    end_time: float
    segments: List[VisibilitySegment]


def standardize_inputs(
    robot_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    tracks_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rt = detect_column(robot_df, "time", args.robot_time_col)
    rx = detect_column(robot_df, "robot_x", args.robot_x_col)
    ry = detect_column(robot_df, "robot_y", args.robot_y_col)
    ryaw = detect_column(robot_df, "robot_yaw", args.robot_yaw_col)

    gt_t = detect_column(gt_df, "time", args.gt_time_col)
    gt_id = detect_column(gt_df, "agent_id", args.gt_id_col)
    gt_x = detect_column(gt_df, "gt_x", args.gt_x_col)
    gt_y = detect_column(gt_df, "gt_y", args.gt_y_col)

    tr_t = detect_column(tracks_df, "time", args.track_time_col)
    tr_id = detect_column(tracks_df, "global_id", args.track_id_col)
    tr_x = detect_column(tracks_df, "track_x", args.track_x_col)
    tr_y = detect_column(tracks_df, "track_y", args.track_y_col)

    # Optional simulator identity attached to each global-track row.
    # If absent, evaluation still works with global-ID continuity + geometry.
    tr_agent = None
    if args.track_agent_col:
        tr_agent = detect_column(
            tracks_df, "track_agent_name", args.track_agent_col
        )
    else:
        try:
            candidate = detect_column(tracks_df, "track_agent_name")
            # Avoid accidentally reusing the global-ID column as agent identity.
            if candidate != tr_id:
                tr_agent = candidate
        except ValueError:
            tr_agent = None

    robot = robot_df[[rt, rx, ry, ryaw]].rename(
        columns={rt: "time", rx: "robot_x", ry: "robot_y", ryaw: "robot_yaw"}
    )
    gt = gt_df[[gt_t, gt_id, gt_x, gt_y]].rename(
        columns={gt_t: "time", gt_id: "agent_id", gt_x: "gt_x", gt_y: "gt_y"}
    )

    track_columns = [tr_t, tr_id, tr_x, tr_y]
    if tr_agent is not None:
        track_columns.append(tr_agent)
    tracks = tracks_df[track_columns].copy().rename(
        columns={
            tr_t: "time",
            tr_id: "global_id",
            tr_x: "track_x",
            tr_y: "track_y",
            **({tr_agent: "track_agent_name"} if tr_agent is not None else {}),
        }
    )
    if "track_agent_name" not in tracks.columns:
        tracks["track_agent_name"] = None

    for frame, cols in (
        (robot, ["time", "robot_x", "robot_y", "robot_yaw"]),
        (gt, ["time", "gt_x", "gt_y"]),
        (tracks, ["time", "track_x", "track_y"]),
    ):
        for col in cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    robot = (
        robot.dropna()
        .sort_values("time")
        .drop_duplicates("time", keep="last")
        .reset_index(drop=True)
    )
    gt = (
        gt.dropna(subset=["time", "agent_id", "gt_x", "gt_y"])
        .sort_values(["time", "agent_id"])
        .reset_index(drop=True)
    )
    tracks = (
        tracks.dropna(subset=["time", "global_id", "track_x", "track_y"])
        .sort_values(["time", "global_id"])
        .reset_index(drop=True)
    )

    if args.robot_yaw_unit == "deg":
        robot["robot_yaw"] = np.deg2rad(robot["robot_yaw"].to_numpy(float))

    robot["robot_yaw"] = wrap_angle(robot["robot_yaw"].to_numpy(float))
    gt["agent_id"] = gt["agent_id"].map(lambda v: str(v).strip())
    tracks["global_id"] = tracks["global_id"].map(lambda v: str(v).strip())
    tracks["track_agent_name"] = tracks["track_agent_name"].map(normalize_identity)

    if robot.empty:
        raise ValueError("Robot CSV contains no valid rows.")
    if gt.empty:
        raise ValueError("Ground-truth CSV contains no valid rows.")

    # Never extrapolate the robot pose indefinitely. Keep only GT samples for
    # which robot pose interpolation is supported by the recorded interval.
    robot_t_min = float(robot["time"].min())
    robot_t_max = float(robot["time"].max())
    before = len(gt)
    gt = gt[(gt["time"] >= robot_t_min) & (gt["time"] <= robot_t_max)].copy()
    if gt.empty:
        raise ValueError(
            "No GT samples overlap the robot-pose recording interval."
        )
    dropped = before - len(gt)
    if dropped:
        print(
            f"Warning: discarded {dropped} GT rows outside robot-pose "
            f"interval [{robot_t_min:.3f}, {robot_t_max:.3f}] s."
        )

    return robot, gt.reset_index(drop=True), tracks

def interpolate_robot_pose(
    robot: pd.DataFrame,
    query_times: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = robot["time"].to_numpy(float)
    x = robot["robot_x"].to_numpy(float)
    y = robot["robot_y"].to_numpy(float)
    yaw = np.unwrap(robot["robot_yaw"].to_numpy(float))

    if len(times) == 1:
        return (
            np.full_like(query_times, x[0], dtype=float),
            np.full_like(query_times, y[0], dtype=float),
            np.full_like(query_times, wrap_angle(yaw[0]), dtype=float),
        )

    return (
        np.interp(query_times, times, x),
        np.interp(query_times, times, y),
        wrap_angle(np.interp(query_times, times, yaw)),
    )


def camera_world_pose(
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    camera: CameraModel,
) -> Tuple[float, float, float]:
    c = math.cos(robot_yaw)
    s = math.sin(robot_yaw)

    cam_x = robot_x + c * camera.offset_x_m - s * camera.offset_y_m
    cam_y = robot_y + s * camera.offset_x_m + c * camera.offset_y_m
    cam_yaw = float(wrap_angle(robot_yaw + camera.relative_yaw_rad))
    return cam_x, cam_y, cam_yaw


def point_visible_in_camera(
    point_x: float,
    point_y: float,
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    camera: CameraModel,
) -> Tuple[bool, float, float]:
    cam_x, cam_y, cam_yaw = camera_world_pose(
        robot_x, robot_y, robot_yaw, camera
    )

    dx = point_x - cam_x
    dy = point_y - cam_y
    distance = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx)
    angular_error = float(wrap_angle(bearing - cam_yaw))

    visible = (
        camera.min_range_m <= distance <= camera.max_range_m
        and abs(angular_error) <= camera.hfov_rad / 2.0
    )
    return visible, distance, angular_error


def annotate_visibility(
    gt: pd.DataFrame,
    robot: pd.DataFrame,
    cameras: Sequence[CameraModel],
) -> pd.DataFrame:
    result = gt.copy()
    qx, qy, qyaw = interpolate_robot_pose(
        robot,
        result["time"].to_numpy(float),
    )
    result["robot_x"] = qx
    result["robot_y"] = qy
    result["robot_yaw"] = qyaw

    visible_values = []
    camera_values = []
    nearest_values = []

    for row in result.itertuples(index=False):
        names = []
        distances = []

        for camera in cameras:
            visible, distance, _ = point_visible_in_camera(
                float(row.gt_x),
                float(row.gt_y),
                float(row.robot_x),
                float(row.robot_y),
                float(row.robot_yaw),
                camera,
            )
            if visible:
                names.append(camera.name)
                distances.append(distance)

        visible_values.append(bool(names))
        camera_values.append("|".join(names))
        nearest_values.append(min(distances) if distances else np.nan)

    result["visible"] = visible_values
    result["visible_cameras"] = camera_values
    result["nearest_visible_camera_distance_m"] = nearest_values
    return result


def nearest_snapshot_time(
    sorted_times: np.ndarray,
    query_time: float,
    tolerance: float,
) -> Optional[float]:
    if len(sorted_times) == 0:
        return None

    idx = int(np.searchsorted(sorted_times, query_time))
    candidates = []

    if idx < len(sorted_times):
        candidates.append(float(sorted_times[idx]))
    if idx > 0:
        candidates.append(float(sorted_times[idx - 1]))

    if not candidates:
        return None

    nearest = min(candidates, key=lambda t: abs(t - query_time))
    return nearest if abs(nearest - query_time) <= tolerance else None


def associate_visible_gt_to_tracks(
    visibility_table: pd.DataFrame,
    tracks: pd.DataFrame,
    time_tolerance_s: float,
    max_distance_m: float,
    continuity_max_distance_m: float,
    identity_mode: str,
) -> pd.DataFrame:
    """
    Identity-aware frame association.

    Priority:
    1. preserve the previous GT -> global-ID assignment when that ID is still
       present, identity-compatible, and within continuity_max_distance_m;
    2. assign remaining rows using Hungarian on Euclidean distance;
    3. when a track carries a simulator/GT agent name, reject incompatible
       GT-track pairs (or require names in identity_mode='required').
    """
    result = visibility_table.copy()
    result["matched_global_id"] = pd.Series([None] * len(result), dtype="object")
    result["matched_track_agent_name"] = pd.Series(
        [None] * len(result), dtype="object"
    )
    result["association_distance_m"] = np.nan
    result["track_snapshot_time"] = np.nan
    result["track_time_error_s"] = np.nan
    result["association_method"] = pd.Series([None] * len(result), dtype="object")
    result["identity_verified"] = False

    if tracks.empty:
        return result

    track_times = np.sort(tracks["time"].unique().astype(float))
    tracks_by_time = {
        float(t): group.reset_index(drop=True)
        for t, group in tracks.groupby("time", sort=True)
    }

    previous_id_by_agent: Dict[str, str] = {}

    def pair_allowed(gt_agent: str, track_agent: object) -> bool:
        track_name = normalize_identity(track_agent)
        if identity_mode == "ignore":
            return True
        if identity_mode == "required":
            return track_name is not None and track_name == gt_agent
        return identities_compatible(gt_agent, track_name)

    for gt_time, gt_group in result[result["visible"]].groupby("time", sort=True):
        snapshot_time = nearest_snapshot_time(
            track_times, float(gt_time), time_tolerance_s
        )
        if snapshot_time is None:
            continue

        track_group = tracks_by_time[snapshot_time]
        if gt_group.empty or track_group.empty:
            continue

        gt_indices = gt_group.index.to_list()
        gt_xy = gt_group[["gt_x", "gt_y"]].to_numpy(float)
        tr_xy = track_group[["track_x", "track_y"]].to_numpy(float)
        distances = np.linalg.norm(gt_xy[:, None, :] - tr_xy[None, :, :], axis=2)

        assigned_gt: set[int] = set()
        assigned_tr: set[int] = set()

        def commit(r: int, c: int, method: str) -> None:
            idx = gt_indices[r]
            gt_agent = str(gt_group.iloc[r]["agent_id"])
            tr_id = str(track_group.iloc[c]["global_id"])
            tr_agent = normalize_identity(track_group.iloc[c]["track_agent_name"])
            result.at[idx, "matched_global_id"] = tr_id
            result.at[idx, "matched_track_agent_name"] = tr_agent
            result.at[idx, "association_distance_m"] = float(distances[r, c])
            result.at[idx, "track_snapshot_time"] = snapshot_time
            result.at[idx, "track_time_error_s"] = abs(snapshot_time - float(gt_time))
            result.at[idx, "association_method"] = method
            result.at[idx, "identity_verified"] = (
                tr_agent is not None and tr_agent == gt_agent
            )
            previous_id_by_agent[gt_agent] = tr_id
            assigned_gt.add(r)
            assigned_tr.add(c)

        # Stage 1: preserve the previous global ID when it is still valid.
        for r in range(len(gt_group)):
            gt_agent = str(gt_group.iloc[r]["agent_id"])
            previous_id = previous_id_by_agent.get(gt_agent)
            if previous_id is None:
                continue
            candidates = [
                c for c in range(len(track_group))
                if c not in assigned_tr
                and str(track_group.iloc[c]["global_id"]) == previous_id
                and pair_allowed(gt_agent, track_group.iloc[c]["track_agent_name"])
                and distances[r, c] <= continuity_max_distance_m
            ]
            if candidates:
                c = min(candidates, key=lambda j: distances[r, j])
                commit(r, c, "global_id_continuity")

        # Stage 2: Hungarian on all remaining compatible pairs.
        rem_gt = [r for r in range(len(gt_group)) if r not in assigned_gt]
        rem_tr = [c for c in range(len(track_group)) if c not in assigned_tr]
        if rem_gt and rem_tr:
            cost = np.full((len(rem_gt), len(rem_tr)), BIG_COST, dtype=float)
            for i, r in enumerate(rem_gt):
                gt_agent = str(gt_group.iloc[r]["agent_id"])
                for j, c in enumerate(rem_tr):
                    if (
                        distances[r, c] <= max_distance_m
                        and pair_allowed(
                            gt_agent, track_group.iloc[c]["track_agent_name"]
                        )
                    ):
                        cost[i, j] = distances[r, c]

            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] >= BIG_COST:
                    continue
                r, c = rem_gt[i], rem_tr[j]
                tr_agent = normalize_identity(
                    track_group.iloc[c]["track_agent_name"]
                )
                method = (
                    "agent_name_and_distance"
                    if tr_agent is not None
                    else "hungarian_distance"
                )
                commit(r, c, method)

    return result

def build_visibility_segments(
    agent_rows: pd.DataFrame,
    nominal_dt: float,
    continuity_factor: float,
) -> List[VisibilitySegment]:
    visible_rows = agent_rows[agent_rows["visible"]].sort_values("time")
    if visible_rows.empty:
        return []

    max_gap = nominal_dt * continuity_factor
    segments = []
    current_indices = []
    previous_time = None

    for index, row in visible_rows.iterrows():
        current_time = float(row["time"])

        if previous_time is None or current_time - previous_time <= max_gap:
            current_indices.append(int(index))
        else:
            selected = agent_rows.loc[current_indices]
            segments.append(
                VisibilitySegment(
                    agent_id=str(selected.iloc[0]["agent_id"]),
                    segment_id=len(segments) + 1,
                    start_time=float(selected["time"].min()),
                    end_time=float(selected["time"].max()),
                    row_indices=current_indices.copy(),
                )
            )
            current_indices = [int(index)]

        previous_time = current_time

    if current_indices:
        selected = agent_rows.loc[current_indices]
        segments.append(
            VisibilitySegment(
                agent_id=str(selected.iloc[0]["agent_id"]),
                segment_id=len(segments) + 1,
                start_time=float(selected["time"].min()),
                end_time=float(selected["time"].max()),
                row_indices=current_indices.copy(),
            )
        )

    return segments


def merge_segments_into_episodes(
    agent_id: str,
    segments: Sequence[VisibilitySegment],
    max_gap_s: float,
) -> List[VisibilityEpisode]:
    if not segments:
        return []

    episodes = []
    current = [segments[0]]

    for segment in segments[1:]:
        gap = segment.start_time - current[-1].end_time

        if gap <= max_gap_s:
            current.append(segment)
        else:
            episodes.append(
                VisibilityEpisode(
                    agent_id=agent_id,
                    episode_id=len(episodes) + 1,
                    start_time=current[0].start_time,
                    end_time=current[-1].end_time,
                    segments=current.copy(),
                )
            )
            current = [segment]

    episodes.append(
        VisibilityEpisode(
            agent_id=agent_id,
            episode_id=len(episodes) + 1,
            start_time=current[0].start_time,
            end_time=current[-1].end_time,
            segments=current.copy(),
        )
    )
    return episodes



def _cumulative_wrapped_rotation_deg(yaws: np.ndarray) -> float:
    """Total absolute robot yaw rotation along a time interval, in degrees."""
    if len(yaws) < 2:
        return 0.0
    diffs = wrap_angle(np.diff(yaws.astype(float)))
    return float(np.degrees(np.sum(np.abs(diffs))))


def classify_visibility_gap(
    table: pd.DataFrame,
    agent_id: str,
    gap_start_time: float,
    gap_end_time: float,
    cameras: Sequence[CameraModel],
    ego_rotation_threshold_deg: float,
) -> Dict[str, Any]:
    """Classify an invisible interval using range, combined FOV and robot yaw."""
    rows = table[
        (table["agent_id"].astype(str) == str(agent_id))
        & (table["time"] > float(gap_start_time))
        & (table["time"] < float(gap_end_time))
    ].sort_values("time")

    if rows.empty:
        return {
            "gap_cause": "unknown_no_samples",
            "out_of_range_fraction": np.nan,
            "out_of_combined_fov_fraction": np.nan,
            "robot_cumulative_rotation_deg": 0.0,
            "sample_count": 0,
        }

    out_of_range = []
    out_of_fov = []
    for row in rows.itertuples(index=False):
        any_in_range = False
        any_in_angular_fov = False
        for camera in cameras:
            _, distance, angular_error = point_visible_in_camera(
                float(row.gt_x), float(row.gt_y),
                float(row.robot_x), float(row.robot_y),
                float(row.robot_yaw), camera,
            )
            if camera.min_range_m <= distance <= camera.max_range_m:
                any_in_range = True
            if abs(angular_error) <= camera.hfov_rad / 2.0:
                any_in_angular_fov = True
        out_of_range.append(not any_in_range)
        out_of_fov.append(not any_in_angular_fov)

    out_of_range_fraction = float(np.mean(out_of_range))
    out_of_fov_fraction = float(np.mean(out_of_fov))
    cumulative_rotation = _cumulative_wrapped_rotation_deg(
        rows["robot_yaw"].to_numpy(float)
    )

    if out_of_range_fraction >= 0.5:
        cause = "out_of_range"
    elif out_of_fov_fraction >= 0.5 and cumulative_rotation >= float(ego_rotation_threshold_deg):
        cause = "ego_rotation_out_of_fov"
    else:
        cause = "other_temporary_invisibility"

    return {
        "gap_cause": cause,
        "out_of_range_fraction": out_of_range_fraction,
        "out_of_combined_fov_fraction": out_of_fov_fraction,
        "robot_cumulative_rotation_deg": cumulative_rotation,
        "sample_count": int(len(rows)),
    }


def merge_segments_into_episodes_scope_aware(
    agent_id: str,
    segments: Sequence[VisibilitySegment],
    table: pd.DataFrame,
    cameras: Sequence[CameraModel],
    max_gap_s: float,
    structural_split_gap_s: float,
    ego_rotation_threshold_deg: float,
) -> Tuple[List[VisibilityEpisode], List[Dict[str, Any]]]:
    """Split long structural visibility losses into distinct evaluation episodes."""
    if not segments:
        return [], []

    episodes = []
    gap_rows = []
    current = [segments[0]]

    for segment in segments[1:]:
        previous = current[-1]
        gap_s = float(segment.start_time - previous.end_time)
        diag = classify_visibility_gap(
            table, agent_id, previous.end_time, segment.start_time, cameras,
            ego_rotation_threshold_deg,
        )
        structural = diag["gap_cause"] in {"out_of_range", "ego_rotation_out_of_fov"}
        forced_structural_split = gap_s > float(structural_split_gap_s) and structural
        standard_split = gap_s > float(max_gap_s)
        split = standard_split or forced_structural_split

        gap_rows.append({
            "agent_id": str(agent_id),
            "previous_segment_id": int(previous.segment_id),
            "next_segment_id": int(segment.segment_id),
            "gap_start_time": float(previous.end_time),
            "gap_end_time": float(segment.start_time),
            "gap_duration_s": gap_s,
            **diag,
            "structural_split_gap_s": float(structural_split_gap_s),
            "ego_rotation_threshold_deg": float(ego_rotation_threshold_deg),
            "forced_structural_episode_split": bool(forced_structural_split),
            "standard_max_gap_episode_split": bool(standard_split),
            "episode_split": bool(split),
        })

        if split:
            episodes.append(VisibilityEpisode(
                agent_id=agent_id, episode_id=len(episodes)+1,
                start_time=current[0].start_time, end_time=current[-1].end_time,
                segments=current.copy(),
            ))
            current = [segment]
        else:
            current.append(segment)

    episodes.append(VisibilityEpisode(
        agent_id=agent_id, episode_id=len(episodes)+1,
        start_time=current[0].start_time, end_time=current[-1].end_time,
        segments=current.copy(),
    ))

    for row in gap_rows:
        row["previous_episode_id"] = next((ep.episode_id for ep in episodes if any(s.segment_id == row["previous_segment_id"] for s in ep.segments)), None)
        row["next_episode_id"] = next((ep.episode_id for ep in episodes if any(s.segment_id == row["next_segment_id"] for s in ep.segments)), None)
    return episodes, gap_rows


def split_episodes_on_local_support_gaps(
    episodes: Sequence[VisibilityEpisode],
    table: pd.DataFrame,
    max_local_support_gap_s: float,
) -> Tuple[List[VisibilityEpisode], List[Dict[str, Any]]]:
    """
    Further split visibility episodes when consecutive VALID LOCAL SUPPORT
    samples for the same GT agent are separated by more than the configured
    fusion recovery horizon.

    This makes the fusion-ID episode definition consistent with the implemented
    component: if no valid local observation is available for longer than the
    reactivation horizon, the old Global ID is no longer recoverable by design.
    A later valid local observation starts a new identity-continuity episode.

    The split is defined from the input available to the fusion node, not from
    the global track lifecycle (DELETE_TRACK, missed count, etc.), so the
    evaluation remains independent of the tracker output being scored.
    """
    if max_local_support_gap_s <= 0:
        return list(episodes), []

    split_episodes: List[VisibilityEpisode] = []
    diagnostics: List[Dict[str, Any]] = []

    for source_ep in episodes:
        rows = episode_rows(table, source_ep).sort_values("time")
        local_rows = rows[rows["local_observation_available"]].sort_values("time")

        if len(local_rows) < 2:
            split_episodes.append(source_ep)
            continue

        local_times = local_rows["time"].to_numpy(float)
        split_positions = np.where(np.diff(local_times) > float(max_local_support_gap_s))[0]

        if len(split_positions) == 0:
            split_episodes.append(source_ep)
            continue

        # Boundaries are placed halfway between the last local-supported sample
        # of one block and the first local-supported sample of the next block.
        boundaries = []
        for pos in split_positions:
            t_before = float(local_times[pos])
            t_after = float(local_times[pos + 1])
            boundaries.append(0.5 * (t_before + t_after))
            diagnostics.append({
                "agent_id": str(source_ep.agent_id),
                "source_episode_id": int(source_ep.episode_id),
                "last_local_support_time_before_gap": t_before,
                "first_local_support_time_after_gap": t_after,
                "local_support_gap_s": t_after - t_before,
                "episode_merge_gap_s": float(max_local_support_gap_s),
                "forced_local_support_episode_split": True,
            })

        window_edges = [-np.inf] + boundaries + [np.inf]

        for wi in range(len(window_edges) - 1):
            lo = window_edges[wi]
            hi = window_edges[wi + 1]
            new_segments: List[VisibilitySegment] = []

            for original_seg in source_ep.segments:
                seg_rows = table.loc[original_seg.row_indices].sort_values("time")
                mask = (seg_rows["time"] > lo) & (seg_rows["time"] <= hi)
                part = seg_rows[mask]
                if part.empty:
                    continue

                new_segments.append(VisibilitySegment(
                    agent_id=str(source_ep.agent_id),
                    segment_id=len(new_segments) + 1,
                    start_time=float(part["time"].iloc[0]),
                    end_time=float(part["time"].iloc[-1]),
                    row_indices=part.index.to_list(),
                ))

            if not new_segments:
                continue

            # Keep only windows that actually contain local fusion input.
            new_rows_idx = [idx for seg in new_segments for idx in seg.row_indices]
            new_rows = table.loc[new_rows_idx]
            if not bool(new_rows["local_observation_available"].any()):
                continue

            split_episodes.append(VisibilityEpisode(
                agent_id=str(source_ep.agent_id),
                episode_id=0,  # renumbered below per agent
                start_time=float(new_segments[0].start_time),
                end_time=float(new_segments[-1].end_time),
                segments=new_segments,
            ))

    # Renumber episodes monotonically per physical agent after all split rules.
    by_agent: Dict[str, List[VisibilityEpisode]] = {}
    for ep in split_episodes:
        by_agent.setdefault(str(ep.agent_id), []).append(ep)

    final: List[VisibilityEpisode] = []
    for agent_id, agent_eps in by_agent.items():
        agent_eps = sorted(agent_eps, key=lambda ep: (ep.start_time, ep.end_time))
        for new_id, ep in enumerate(agent_eps, start=1):
            final.append(VisibilityEpisode(
                agent_id=agent_id,
                episode_id=new_id,
                start_time=ep.start_time,
                end_time=ep.end_time,
                segments=ep.segments,
            ))

    final.sort(key=lambda ep: (str(ep.agent_id), ep.start_time, ep.episode_id))
    return final, diagnostics


def episode_rows(
    association_table: pd.DataFrame,
    episode: VisibilityEpisode,
) -> pd.DataFrame:
    indices = []
    for segment in episode.segments:
        indices.extend(segment.row_indices)

    return association_table.loc[indices].sort_values("time").copy()


def matched_state_sequence(rows: pd.DataFrame) -> List[Tuple[float, Optional[str]]]:
    """Run-length encoded matched-ID state, preserving unmatched intervals."""
    states: List[Tuple[float, Optional[str]]] = []
    previous = object()
    for row in rows.sort_values("time").itertuples(index=False):
        current = normalize_identity(row.matched_global_id)
        if current != previous:
            states.append((float(row.time), current))
            previous = current
    return states


def segment_sample_weights(
    association_table: pd.DataFrame,
    episode: VisibilityEpisode,
    nominal_dt: float,
) -> pd.DataFrame:
    """Weight every visibility segment independently; invisible gaps get zero."""
    parts = []
    for segment in episode.segments:
        part = association_table.loc[segment.row_indices].sort_values("time").copy()
        part["episode_sample_weight_s"] = sample_weights(
            part["time"].to_numpy(float), nominal_dt
        )
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=0).sort_values("time")



def detect_any_column(df: pd.DataFrame, candidates: Sequence[str], explicit: Optional[str], logical: str) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Column '{explicit}' requested for {logical} not found. Available: {list(df.columns)}")
        return explicit
    normalized = {str(c).strip().lower(): str(c) for c in df.columns}
    for name in candidates:
        if name.lower() in normalized:
            return normalized[name.lower()]
    raise ValueError(f"Could not detect {logical}. Available columns: {list(df.columns)}")


def standardize_local_detections(local_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    t = detect_any_column(local_df, COMMON_COLUMNS['time'], args.local_time_col, 'local detection time')
    x = detect_any_column(local_df, ('x','local_x','det_x','position_x','map_x'), args.local_x_col, 'local detection x')
    y = detect_any_column(local_df, ('y','local_y','det_y','position_y','map_y'), args.local_y_col, 'local detection y')
    camera = detect_any_column(local_df, ('camera','cam','camera_name','source_camera'), args.local_camera_col, 'local detection camera')
    local_id = detect_any_column(local_df, ('local_id','track_id','id','name','person_id'), args.local_id_col, 'local detection local_id')
    cols=[t,x,y,camera,local_id]
    reliability=None
    for c in ('reliability','confidence','score'):
        if c in local_df.columns:
            reliability=c; cols.append(c); break
    out=local_df[cols].copy().rename(columns={t:'time',x:'local_x',y:'local_y',camera:'camera',local_id:'local_id'})
    if reliability:
        out=out.rename(columns={reliability:'local_reliability'})
    else:
        out['local_reliability']=np.nan
    for c in ('time','local_x','local_y','local_reliability'):
        out[c]=pd.to_numeric(out[c],errors='coerce')
    out=out.dropna(subset=['time','local_x','local_y']).sort_values(['time','camera','local_id']).reset_index(drop=True)
    out['camera']=out['camera'].astype(str)
    out['local_id']=out['local_id'].astype(str)
    return out


def associate_visible_gt_to_local_detections(
    visibility_table: pd.DataFrame,
    local: pd.DataFrame,
    time_tolerance_s: float,
    max_distance_m: float,
) -> pd.DataFrame:
    """Mark when the fusion node had a local observation available for each visible GT agent."""
    result=visibility_table.copy()
    result['local_observation_available']=False
    result['matched_local_id']=pd.Series([None]*len(result),dtype='object')
    result['matched_local_camera']=pd.Series([None]*len(result),dtype='object')
    result['local_association_distance_m']=np.nan
    result['local_snapshot_time']=np.nan
    result['local_time_error_s']=np.nan
    result['local_reliability']=np.nan
    result['matched_local_x']=np.nan
    result['matched_local_y']=np.nan
    if local.empty:
        return result
    times=np.sort(local['time'].unique().astype(float))
    by_time={float(t):g.reset_index(drop=True) for t,g in local.groupby('time',sort=True)}
    for gt_time,gt_group in result[result['visible']].groupby('time',sort=True):
        snap=nearest_snapshot_time(times,float(gt_time),time_tolerance_s)
        if snap is None: continue
        dets=by_time[snap]
        if dets.empty: continue
        gt_xy=gt_group[['gt_x','gt_y']].to_numpy(float)
        det_xy=dets[['local_x','local_y']].to_numpy(float)
        dist=np.linalg.norm(gt_xy[:,None,:]-det_xy[None,:,:],axis=2)
        cost=dist.copy(); cost[cost>max_distance_m]=BIG_COST
        rr,cc=linear_sum_assignment(cost)
        indices=gt_group.index.to_list()
        for r,c in zip(rr,cc):
            if cost[r,c]>=BIG_COST: continue
            idx=indices[r]
            result.at[idx,'local_observation_available']=True
            result.at[idx,'matched_local_id']=str(dets.iloc[c]['local_id'])
            result.at[idx,'matched_local_camera']=str(dets.iloc[c]['camera'])
            result.at[idx,'local_association_distance_m']=float(dist[r,c])
            result.at[idx,'local_snapshot_time']=snap
            result.at[idx,'local_time_error_s']=abs(snap-float(gt_time))
            result.at[idx,'local_reliability']=float(dets.iloc[c]['local_reliability']) if np.isfinite(dets.iloc[c]['local_reliability']) else np.nan
            result.at[idx,'matched_local_x']=float(dets.iloc[c]['local_x'])
            result.at[idx,'matched_local_y']=float(dets.iloc[c]['local_y'])
    return result


def associate_evaluable_gt_to_global_tracks(
    table: pd.DataFrame,
    tracks: pd.DataFrame,
    time_tolerance_s: float,
    max_distance_m: float,
    continuity_max_distance_m: float,
) -> pd.DataFrame:
    """Associate only samples for which a local observation exists; local misses are excluded from fusion evaluation."""
    result=table.copy()
    result['matched_global_id']=pd.Series([None]*len(result),dtype='object')
    result['association_distance_m']=np.nan
    result['track_snapshot_time']=np.nan
    result['track_time_error_s']=np.nan
    result['association_method']=pd.Series([None]*len(result),dtype='object')
    if tracks.empty: return result
    times=np.sort(tracks['time'].unique().astype(float))
    by_time={float(t):g.reset_index(drop=True) for t,g in tracks.groupby('time',sort=True)}
    previous: Dict[str,str]={}
    for gt_time,gt_group in result[result['local_observation_available']].groupby('time',sort=True):
        snap=nearest_snapshot_time(times,float(gt_time),time_tolerance_s)
        if snap is None: continue
        tr=by_time[snap]
        if tr.empty: continue
        gt_xy=gt_group[['gt_x','gt_y']].to_numpy(float)
        tr_xy=tr[['track_x','track_y']].to_numpy(float)
        dist=np.linalg.norm(gt_xy[:,None,:]-tr_xy[None,:,:],axis=2)
        gt_indices=gt_group.index.to_list(); used_g=set(); used_t=set()
        def commit(r,c,method):
            idx=gt_indices[r]; aid=str(gt_group.iloc[r]['agent_id']); gid=str(tr.iloc[c]['global_id'])
            result.at[idx,'matched_global_id']=gid
            result.at[idx,'association_distance_m']=float(dist[r,c])
            result.at[idx,'track_snapshot_time']=snap
            result.at[idx,'track_time_error_s']=abs(snap-float(gt_time))
            result.at[idx,'association_method']=method
            previous[aid]=gid; used_g.add(r); used_t.add(c)
        for r in range(len(gt_group)):
            aid=str(gt_group.iloc[r]['agent_id']); gid=previous.get(aid)
            if gid is None: continue
            cand=[c for c in range(len(tr)) if c not in used_t and str(tr.iloc[c]['global_id'])==gid and dist[r,c]<=continuity_max_distance_m]
            if cand: commit(r,min(cand,key=lambda c:dist[r,c]),'global_id_continuity')
        rg=[r for r in range(len(gt_group)) if r not in used_g]
        rt=[c for c in range(len(tr)) if c not in used_t]
        if rg and rt:
            cost=np.full((len(rg),len(rt)),BIG_COST)
            for i,r in enumerate(rg):
                for j,c in enumerate(rt):
                    if dist[r,c]<=max_distance_m: cost[i,j]=dist[r,c]
            rr,cc=linear_sum_assignment(cost)
            for i,j in zip(rr,cc):
                if cost[i,j]<BIG_COST: commit(rg[i],rt[j],'hungarian_distance')
    return result


def evaluable_state_sequence(rows: pd.DataFrame) -> List[Tuple[float, Optional[str]]]:
    """Run-length encoded global-ID sequence over locally observable samples only."""
    rows=rows[rows['local_observation_available']].sort_values('time')
    states=[]; sentinel=object(); prev=sentinel
    for row in rows.itertuples(index=False):
        cur=normalize_identity(row.matched_global_id)
        if cur!=prev:
            states.append((float(row.time),cur)); prev=cur
    return states


def extract_global_events(table: pd.DataFrame, episode: VisibilityEpisode):
    rows=episode_rows(table,episode)
    states=evaluable_state_sequence(rows)
    switches=[]; interruptions=[]
    last_id=None; gap_start=None; id_before=None
    for t,state in states:
        if state is None:
            if gap_start is None:
                gap_start=t; id_before=last_id
            continue
        if gap_start is not None:
            kind='initial_acquisition_delay' if id_before is None else 'global_association_interruption'
            interruptions.append({
                'agent_id':episode.agent_id,'episode_id':episode.episode_id,'episode_label':f'{episode.agent_id}-E{episode.episode_id}',
                'event_type':kind,'loss_start_time':gap_start,'reacquisition_time':t,
                'interruption_duration_s':max(0.0,t-gap_start),'global_id_before_loss':id_before,
                'global_id_after_reacquisition':state,'same_id_recovered':id_before is not None and id_before==state,
            })
            gap_start=None; id_before=None
        if last_id is not None and state!=last_id:
            switches.append({'agent_id':episode.agent_id,'episode_id':episode.episode_id,'episode_label':f'{episode.agent_id}-E{episode.episode_id}',
                'switch_time':t,'previous_global_id':last_id,'new_global_id':state})
        last_id=state
    if gap_start is not None:
        interruptions.append({'agent_id':episode.agent_id,'episode_id':episode.episode_id,'episode_label':f'{episode.agent_id}-E{episode.episode_id}',
            'event_type':'terminal_global_loss' if id_before is not None else 'never_globally_acquired',
            'loss_start_time':gap_start,'reacquisition_time':np.nan,
            'interruption_duration_s':max(0.0,episode.end_time-gap_start),'global_id_before_loss':id_before,
            'global_id_after_reacquisition':None,'same_id_recovered':False})
    return switches,interruptions




# ===========================================================================
# Recovery-horizon and causal switch diagnostics
# ===========================================================================

_ACCEPTED_FUSION_UPDATE_EVENTS = {
    "MATCH_ACCEPTED",
    "RECOVER_EXISTING_TRACK_MULTICUE",
}


def _nearest_gt_snapshot(
    gt_table: pd.DataFrame,
    query_time: float,
    tolerance_s: float,
) -> pd.DataFrame:
    """Return the nearest GT snapshot within tolerance."""
    if gt_table.empty:
        return pd.DataFrame()
    times = np.sort(gt_table["time"].dropna().unique().astype(float))
    snap = nearest_snapshot_time(times, float(query_time), float(tolerance_s))
    if snap is None:
        return pd.DataFrame()
    return gt_table[np.isclose(gt_table["time"].to_numpy(float), snap)].copy()


def load_debug_events_optional(path: Optional[Path]) -> pd.DataFrame:
    """Load the fusion debug CSV if available; otherwise return an empty table."""
    if path is None or not path.exists():
        return pd.DataFrame()
    debug = pd.read_csv(path)
    if "time" in debug.columns:
        debug["time"] = pd.to_numeric(debug["time"], errors="coerce")
    if "global_id" in debug.columns:
        debug["global_id"] = debug["global_id"].map(normalize_identity)
    for col in ("det_x", "det_y", "distance", "appearance_distance", "age"):
        if col in debug.columns:
            debug[col] = pd.to_numeric(debug[col], errors="coerce")
    return debug


def classify_switch_root_causes(
    switches_df: pd.DataFrame,
    debug: pd.DataFrame,
    gt_table: pd.DataFrame,
    local_valid_distance_m: float,
    gt_time_tolerance_s: float,
    lookback_s: float,
) -> pd.DataFrame:
    """
    Classify observed within-episode Global-ID switches.

    The key upstream-localization test is deliberately conservative:
    - inspect only fusion updates that were actually ACCEPTED for the previous
      Global ID shortly before the switch;
    - compare the accepted local measurement to ALL GT people at that time;
    - call it an upstream localization outlier only when the measurement is
      farther than local_valid_distance_m from every GT target.

    Thus, a measurement that is close to another real person is NOT excused as
    an upstream localization outlier; that remains a possible fusion
    misassociation.
    """
    if switches_df.empty:
        out = switches_df.copy()
        for col in (
            "switch_root_cause",
            "fusion_attributable_switch",
            "upstream_outlier_event_time",
            "upstream_outlier_global_id",
            "upstream_outlier_local_id",
            "upstream_outlier_camera",
            "upstream_outlier_target_gt_distance_m",
            "upstream_outlier_nearest_any_gt_distance_m",
            "upstream_outlier_appearance_distance",
        ):
            out[col] = pd.Series(dtype="object")
        return out

    out = switches_df.copy()
    out["switch_root_cause"] = "fusion_or_unresolved"
    out["fusion_attributable_switch"] = True
    out["upstream_outlier_event_time"] = np.nan
    out["upstream_outlier_global_id"] = None
    out["upstream_outlier_local_id"] = None
    out["upstream_outlier_camera"] = None
    out["upstream_outlier_target_gt_distance_m"] = np.nan
    out["upstream_outlier_nearest_any_gt_distance_m"] = np.nan
    out["upstream_outlier_appearance_distance"] = np.nan

    if debug.empty:
        out["switch_root_cause"] = "fusion_or_unresolved_no_debug"
        return out

    required = {"time", "event_type", "global_id", "det_x", "det_y"}
    if not required.issubset(debug.columns):
        out["switch_root_cause"] = "fusion_or_unresolved_incomplete_debug"
        return out

    accepted = debug[
        debug["event_type"].astype(str).isin(_ACCEPTED_FUSION_UPDATE_EVENTS)
    ].copy()
    accepted = accepted.dropna(subset=["time", "global_id", "det_x", "det_y"])

    for idx, sw in out.iterrows():
        switch_t = float(sw["switch_time"])
        previous_gid = normalize_identity(sw["previous_global_id"])
        agent_id = str(sw["agent_id"])

        cand = accepted[
            (accepted["global_id"].map(normalize_identity) == previous_gid)
            & (accepted["time"] <= switch_t + EPS)
            & (accepted["time"] >= switch_t - float(lookback_s) - EPS)
        ].sort_values("time", ascending=False)

        found = None
        for _, ev in cand.iterrows():
            snap = _nearest_gt_snapshot(
                gt_table,
                float(ev["time"]),
                gt_time_tolerance_s,
            )
            if snap.empty:
                continue

            det_x = float(ev["det_x"])
            det_y = float(ev["det_y"])
            all_dist = np.hypot(
                snap["gt_x"].to_numpy(float) - det_x,
                snap["gt_y"].to_numpy(float) - det_y,
            )
            if len(all_dist) == 0:
                continue

            nearest_any = float(np.min(all_dist))
            target_rows = snap[snap["agent_id"].astype(str) == agent_id]
            target_dist = np.nan
            if not target_rows.empty:
                target_dist = float(np.min(np.hypot(
                    target_rows["gt_x"].to_numpy(float) - det_x,
                    target_rows["gt_y"].to_numpy(float) - det_y,
                )))

            # Conservative upstream-outlier definition:
            # the accepted local input is not spatially compatible with ANY GT.
            if nearest_any > float(local_valid_distance_m):
                found = {
                    "time": float(ev["time"]),
                    "gid": previous_gid,
                    "local_id": normalize_identity(ev.get("local_id")),
                    "camera": normalize_identity(ev.get("camera")),
                    "target_dist": target_dist,
                    "nearest_any": nearest_any,
                    "appearance_distance": (
                        float(ev["appearance_distance"])
                        if "appearance_distance" in ev and pd.notna(ev["appearance_distance"])
                        else np.nan
                    ),
                }
                break

        if found is not None:
            out.at[idx, "switch_root_cause"] = "upstream_localization_induced"
            out.at[idx, "fusion_attributable_switch"] = False
            out.at[idx, "upstream_outlier_event_time"] = found["time"]
            out.at[idx, "upstream_outlier_global_id"] = found["gid"]
            out.at[idx, "upstream_outlier_local_id"] = found["local_id"]
            out.at[idx, "upstream_outlier_camera"] = found["camera"]
            out.at[idx, "upstream_outlier_target_gt_distance_m"] = found["target_dist"]
            out.at[idx, "upstream_outlier_nearest_any_gt_distance_m"] = found["nearest_any"]
            out.at[idx, "upstream_outlier_appearance_distance"] = found["appearance_distance"]

    return out


def add_causal_switch_aggregates(
    agent_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    classified_switches: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add transparent raw-vs-fusion-attributable switch counts."""
    adf = agent_df.copy()
    odf = overall_df.copy()

    if classified_switches.empty:
        if not adf.empty:
            adf["observed_global_id_switches"] = adf.get("total_global_id_switches", 0)
            adf["fusion_attributable_global_id_switches"] = 0
            adf["upstream_induced_identity_discontinuities"] = 0
        if not odf.empty:
            odf["observed_global_id_switches"] = odf.get("total_global_id_switches", 0)
            odf["fusion_attributable_global_id_switches"] = 0
            odf["upstream_induced_identity_discontinuities"] = 0
        return adf, odf

    classified_switches = classified_switches.copy()
    classified_switches["agent_id"] = classified_switches["agent_id"].astype(str)

    if not adf.empty:
        adf["agent_id"] = adf["agent_id"].astype(str)
        observed = classified_switches.groupby("agent_id").size()
        attributable = (
            classified_switches[classified_switches["fusion_attributable_switch"].astype(bool)]
            .groupby("agent_id").size()
        )
        upstream = (
            classified_switches[
                classified_switches["switch_root_cause"] == "upstream_localization_induced"
            ].groupby("agent_id").size()
        )
        adf["observed_global_id_switches"] = adf["agent_id"].map(observed).fillna(0).astype(int)
        adf["fusion_attributable_global_id_switches"] = adf["agent_id"].map(attributable).fillna(0).astype(int)
        adf["upstream_induced_identity_discontinuities"] = adf["agent_id"].map(upstream).fillna(0).astype(int)

    if not odf.empty:
        odf["observed_global_id_switches"] = int(len(classified_switches))
        odf["fusion_attributable_global_id_switches"] = int(
            classified_switches["fusion_attributable_switch"].astype(bool).sum()
        )
        odf["upstream_induced_identity_discontinuities"] = int(
            (classified_switches["switch_root_cause"] == "upstream_localization_induced").sum()
        )

    return adf, odf


def compute_episode_metrics(table: pd.DataFrame, episode: VisibilityEpisode, nominal_dt: float):
    """
    Compute fusion-node metrics for one geometric visibility episode.

    Important:
    - geometric visibility defines the episode boundaries;
    - only samples with a matched local observation are evaluable for the fusion node;
    - episodes with zero evaluable time keep NaN metrics, not 0%.
    """
    rows = segment_sample_weights(table, episode, nominal_dt)
    w = rows["episode_sample_weight_s"].to_numpy(float)

    visible = float(w.sum())
    local_mask = rows["local_observation_available"].to_numpy(bool)
    evaluable = float(w[local_mask].sum())

    global_mask = local_mask & rows["matched_global_id"].notna().to_numpy()
    assigned = float(w[global_mask].sum())

    id_durations: Dict[str, float] = {}
    for gid in rows.loc[global_mask, "matched_global_id"].astype(str).unique():
        mask = global_mask & (
            rows["matched_global_id"].astype(object) == gid
        ).to_numpy()
        id_durations[gid] = float(w[mask].sum())

    dominant = max(id_durations, key=id_durations.get) if id_durations else None
    domdur = id_durations.get(dominant, 0.0) if dominant is not None else 0.0

    switches, interruptions = extract_global_events(table, episode)
    errors = rows.loc[global_mask, "association_distance_m"].dropna()
    local_errors = rows.loc[local_mask, "local_association_distance_m"].dropna()
    seq = [s for _, s in evaluable_state_sequence(rows) if s is not None]

    coverage = assigned / evaluable if evaluable > EPS else np.nan
    purity = domdur / assigned if assigned > EPS else np.nan
    stability = domdur / evaluable if evaluable > EPS else np.nan

    metric = {
        "agent_id": episode.agent_id,
        "episode_id": episode.episode_id,
        "episode_label": f"{episode.agent_id}-E{episode.episode_id}",
        "episode_start_time": episode.start_time,
        "episode_end_time": episode.end_time,
        "episode_elapsed_s": episode.end_time - episode.start_time,
        "geometric_visible_duration_s": visible,
        "local_observation_duration_s": evaluable,
        "is_fusion_evaluable": bool(evaluable > EPS),
        "local_observation_fraction_of_visible": (
            evaluable / visible if visible > EPS else np.nan
        ),
        "global_id_assigned_duration_s": assigned,
        "has_global_id_assignment": bool(assigned > EPS),
        "global_association_coverage": coverage,
        "dominant_global_id": dominant,
        "dominant_global_id_duration_s": domdur,
        "global_id_purity_when_assigned": purity,
        "global_id_stability_evaluable": stability,
        "unique_global_ids": len(set(seq)),
        "global_id_switches": len(switches),
        "global_association_interruptions": sum(
            1 for x in interruptions
            if x["event_type"] == "global_association_interruption"
        ),
        "initial_acquisition_delays": sum(
            1 for x in interruptions
            if x["event_type"] == "initial_acquisition_delay"
        ),
        "terminal_global_losses": sum(
            1 for x in interruptions
            if x["event_type"] == "terminal_global_loss"
        ),
        "raw_visibility_segments": len(episode.segments),
        "short_visibility_gaps": max(0, len(episode.segments) - 1),
        "global_id_sequence": " -> ".join(seq),
        "mean_global_position_error_m": (
            float(errors.mean()) if not errors.empty else np.nan
        ),
        "median_global_position_error_m": (
            float(errors.median()) if not errors.empty else np.nan
        ),
        "max_global_position_error_m": (
            float(errors.max()) if not errors.empty else np.nan
        ),
        "mean_local_gt_distance_m": (
            float(local_errors.mean()) if not local_errors.empty else np.nan
        ),
    }

    breakdown = []
    for gid, dur in sorted(id_durations.items(), key=lambda x: (-x[1], x[0])):
        breakdown.append({
            "agent_id": episode.agent_id,
            "episode_id": episode.episode_id,
            "episode_label": metric["episode_label"],
            "global_id": gid,
            "id_duration_s": dur,
            "fraction_of_evaluable_time": (
                dur / evaluable if evaluable > EPS else np.nan
            ),
            "fraction_of_assigned_time": (
                dur / assigned if assigned > EPS else np.nan
            ),
            "is_dominant_id": gid == dominant,
        })

    return metric, breakdown, switches, interruptions


def compute_reentry_events_fusion(table: pd.DataFrame, episode: VisibilityEpisode):
    events=[]
    if len(episode.segments)<2: return events
    for i in range(len(episode.segments)-1):
        before=table.loc[episode.segments[i].row_indices].sort_values('time')
        after=table.loc[episode.segments[i+1].row_indices].sort_values('time')
        before_eval=before[before['local_observation_available']]
        after_eval=after[after['local_observation_available']]
        before_global=before_eval[before_eval['matched_global_id'].notna()]
        after_global=after_eval[after_eval['matched_global_id'].notna()]
        id_before=str(before_global.iloc[-1]['matched_global_id']) if not before_global.empty else None
        first_local_time=float(after_eval.iloc[0]['time']) if not after_eval.empty else None
        id_after=str(after_global.iloc[0]['matched_global_id']) if not after_global.empty else None
        first_global_time=float(after_global.iloc[0]['time']) if not after_global.empty else None
        if id_before is None: status='not_evaluable_no_global_id_before_exit'; success=None
        elif after_eval.empty: status='not_evaluable_no_local_detection_after_reentry'; success=None
        elif id_after is None: status='not_globally_reacquired'; success=False
        elif id_after==id_before: status='same_global_id_after_reentry'; success=True
        else: status='different_global_id_after_reentry'; success=False
        events.append({'agent_id':episode.agent_id,'episode_id':episode.episode_id,'episode_label':f'{episode.agent_id}-E{episode.episode_id}',
            'reentry_event':i+1,'exit_time':episode.segments[i].end_time,'geometric_reentry_time':episode.segments[i+1].start_time,
            'invisible_gap_s':episode.segments[i+1].start_time-episode.segments[i].end_time,
            'first_local_observation_time_after_reentry':first_local_time,
            'global_id_before_exit':id_before,'global_id_after_reentry':id_after,
            'same_global_id_after_reentry':success,'reentry_status':status,
            'global_reacquisition_latency_from_local_s':(first_global_time-first_local_time) if first_global_time is not None and first_local_time is not None else np.nan})
    return events


def build_segment_table_new(episodes,table,nominal):
    rows=[]
    for ep in episodes:
        for seg in ep.segments:
            s=table.loc[seg.row_indices].sort_values('time'); w=sample_weights(s['time'].to_numpy(float),nominal[ep.agent_id])
            local=s['local_observation_available'].to_numpy(bool); glob=local & s['matched_global_id'].notna().to_numpy()
            rows.append({'agent_id':ep.agent_id,'episode_id':ep.episode_id,'episode_label':f'{ep.agent_id}-E{ep.episode_id}',
                'segment_id':seg.segment_id,'segment_start_time':seg.start_time,'segment_end_time':seg.end_time,
                'geometric_visible_duration_s':float(w.sum()),'local_observation_duration_s':float(w[local].sum()),
                'global_id_assigned_duration_s':float(w[glob].sum()),'sample_count':len(s)})
    return pd.DataFrame(rows)


def build_episode_table_new(episodes):
    rows=[]
    for ep in episodes:
        gaps=[ep.segments[i+1].start_time-ep.segments[i].end_time for i in range(len(ep.segments)-1)]
        rows.append({'agent_id':ep.agent_id,'episode_id':ep.episode_id,'episode_label':f'{ep.agent_id}-E{ep.episode_id}',
            'episode_start_time':ep.start_time,'episode_end_time':ep.end_time,'episode_elapsed_s':ep.end_time-ep.start_time,
            'segment_count':len(ep.segments),'short_gap_count':len(gaps),'maximum_short_gap_s':max(gaps) if gaps else 0.0,'total_short_gap_s':sum(gaps)})
    return pd.DataFrame(rows)



def aggregate_metrics(epm: pd.DataFrame, reentry: pd.DataFrame):
    """
    Aggregate separately:
    - geometric population;
    - fusion-evaluable population;
    - globally assigned population.

    Non-evaluable agents/episodes are counted but never converted to 0% performance.
    """
    if epm.empty:
        return pd.DataFrame(), pd.DataFrame()

    agents = []
    for aid, g in epm.groupby("agent_id", sort=True):
        ev = float(g["local_observation_duration_s"].sum())
        ass = float(g["global_id_assigned_duration_s"].sum())
        dom = float(g["dominant_global_id_duration_s"].sum())

        r = (
            reentry[reentry["agent_id"] == aid]
            if not reentry.empty
            else pd.DataFrame()
        )
        valid = (
            r["same_global_id_after_reentry"].dropna()
            if not r.empty
            else pd.Series(dtype=bool)
        )

        agents.append({
            "agent_id": aid,
            "geometric_episode_count": len(g),
            "evaluable_episode_count": int(
                (g["local_observation_duration_s"] > EPS).sum()
            ),
            "assigned_episode_count": int(
                (g["global_id_assigned_duration_s"] > EPS).sum()
            ),
            "total_geometric_visible_duration_s": float(
                g["geometric_visible_duration_s"].sum()
            ),
            "total_local_observation_duration_s": ev,
            "total_global_id_assigned_duration_s": ass,
            "is_fusion_evaluable": bool(ev > EPS),
            "global_association_coverage": ass / ev if ev > EPS else np.nan,
            "global_id_purity_when_assigned": dom / ass if ass > EPS else np.nan,
            "global_id_stability_evaluable": dom / ev if ev > EPS else np.nan,
            "total_global_id_switches": int(g["global_id_switches"].sum()),
            "total_global_association_interruptions": int(
                g["global_association_interruptions"].sum()
            ),
            "reentry_event_count": len(r),
            "evaluable_reentry_event_count": len(valid),
            "same_global_id_after_reentry_rate": (
                float(valid.astype(bool).mean()) if len(valid) else np.nan
            ),
        })

    agent_df = pd.DataFrame(agents)

    ev = float(epm["local_observation_duration_s"].sum())
    ass = float(epm["global_id_assigned_duration_s"].sum())
    dom = float(epm["dominant_global_id_duration_s"].sum())
    valid = (
        reentry["same_global_id_after_reentry"].dropna()
        if not reentry.empty
        else pd.Series(dtype=bool)
    )

    overall = pd.DataFrame([{
        "geometrically_visible_agent_count": int(epm["agent_id"].nunique()),
        "geometric_visibility_episode_count": len(epm),
        "fusion_evaluable_agent_count": int(
            agent_df["is_fusion_evaluable"].sum()
        ),
        "fusion_evaluable_episode_count": int(
            (epm["local_observation_duration_s"] > EPS).sum()
        ),
        "globally_assigned_agent_count": int(
            (agent_df["total_global_id_assigned_duration_s"] > EPS).sum()
        ),
        "globally_assigned_episode_count": int(
            (epm["global_id_assigned_duration_s"] > EPS).sum()
        ),
        "total_geometric_visible_duration_s": float(
            epm["geometric_visible_duration_s"].sum()
        ),
        "total_local_observation_duration_s": ev,
        "total_global_id_assigned_duration_s": ass,
        "global_association_coverage": ass / ev if ev > EPS else np.nan,
        "global_id_purity_when_assigned": dom / ass if ass > EPS else np.nan,
        "global_id_stability_evaluable": dom / ev if ev > EPS else np.nan,
        "total_global_id_switches": int(epm["global_id_switches"].sum()),
        "global_id_switches_per_evaluable_minute": (
            epm["global_id_switches"].sum() / (ev / 60.0)
            if ev > EPS else np.nan
        ),
        "total_global_association_interruptions": int(
            epm["global_association_interruptions"].sum()
        ),
        "total_initial_acquisition_delays": int(
            epm["initial_acquisition_delays"].sum()
        ),
        "total_terminal_global_losses": int(
            epm["terminal_global_losses"].sum()
        ),
        "reentry_event_count": len(reentry),
        "evaluable_reentry_event_count": len(valid),
        "same_global_id_after_reentry_rate": (
            float(valid.astype(bool).mean()) if len(valid) else np.nan
        ),
    }])

    return agent_df, overall


def savefig(path: Path):
    plt.tight_layout(); plt.savefig(path,dpi=220,bbox_inches='tight'); plt.close()






def _clean_previous_plot_files(output_dir: Path) -> None:
    """Remove obsolete PNG figures before generating the new thesis set."""
    for path in output_dir.glob("*.png"):
        try:
            path.unlink()
        except OSError:
            pass


def _apply_thesis_style() -> None:
    """Readable and restrained defaults for engineering-thesis figures."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _global_id_color_map(global_ids: Sequence[str]) -> Dict[str, object]:
    """Use the same color for each Global ID in every figure."""
    cmap = plt.get_cmap("tab10")
    ids = sorted(set(map(str, global_ids)))
    return {gid: cmap(i % 10) for i, gid in enumerate(ids)}


def _agent_stability_map(agents: pd.DataFrame) -> Dict[str, float]:
    result = {}
    if agents.empty:
        return result
    for row in agents.itertuples(index=False):
        value = getattr(row, "global_id_stability_evaluable", np.nan)
        result[str(row.agent_id)] = float(value) if pd.notna(value) else np.nan
    return result


def _segment_duration(
    table: pd.DataFrame,
    segment: VisibilitySegment,
    nominal_dt: float,
) -> float:
    rows = table.loc[segment.row_indices].sort_values("time")
    if rows.empty:
        return 0.0
    return float(sample_weights(rows["time"].to_numpy(float), nominal_dt).sum())


def _state_intervals(
    rows: pd.DataFrame,
    nominal_dt: float,
) -> List[Tuple[float, float, Optional[str]]]:
    """Convert evaluable samples into continuous Global-ID intervals."""
    rows = rows.sort_values("time")
    if rows.empty:
        return []

    times = rows["time"].to_numpy(float)
    states = [normalize_identity(v) for v in rows["matched_global_id"]]
    max_gap = max(2.5 * nominal_dt, nominal_dt)

    intervals: List[Tuple[float, float, Optional[str]]] = []
    start = times[0]
    previous_time = times[0]
    previous_state = states[0]

    for current_time, current_state in zip(times[1:], states[1:]):
        if (
            current_state != previous_state
            or current_time - previous_time > max_gap
        ):
            intervals.append((
                float(start),
                float(previous_time + nominal_dt),
                previous_state,
            ))
            start = current_time
            previous_state = current_state
        previous_time = current_time

    intervals.append((
        float(start),
        float(previous_time + nominal_dt),
        previous_state,
    ))
    return intervals



def clean_identity_for_display(value):
    """Return a clean scalar identity label for plot annotations."""
    identity = normalize_identity(value)
    if identity is None:
        return "None"

    text = str(identity).strip()

    # Handle stringified singleton containers such as "['id_2']".
    match = re.fullmatch(r"[\[\(]\s*['\"]?([^'\"\],\)]+)['\"]?\s*[\]\)]", text)
    if match:
        text = match.group(1).strip()

    return text


def _build_episode_summary_table(
    epm: pd.DataFrame,
    episodes: Sequence[VisibilityEpisode],
    table: pd.DataFrame,
    reentry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the episode-level thesis table using only fusion-evaluable segments.

    A visibility segment is included only when it contains at least one sample
    with a local observation associated to the GT agent. Consequently:
    - purely geometric segments without local tracker input are excluded;
    - gaps are reported only between two retained fusion-evaluable segments.
    """
    episode_lookup = {
        (ep.agent_id, ep.episode_id): ep
        for ep in episodes
    }

    reentry_lookup = {}
    if not reentry.empty:
        for key, group in reentry.groupby(["agent_id", "episode_id"], sort=True):
            valid = group["same_global_id_after_reentry"].dropna()
            if len(valid) == 0:
                status = "N/A"
            elif valid.astype(bool).all():
                status = "Maintained"
            else:
                status = "Failed"
            reentry_lookup[(str(key[0]), int(key[1]))] = status

    evaluable = epm[
        epm["global_id_assigned_duration_s"] > EPS
    ].copy()

    rows = []

    for metric in evaluable.sort_values(
        ["agent_id", "episode_id"]
    ).itertuples(index=False):

        episode = episode_lookup[
            (str(metric.agent_id), int(metric.episode_id))
        ]

        agent_rows = table[
            table["agent_id"].astype(str) == str(metric.agent_id)
        ]
        nominal_dt = estimate_nominal_dt(
            agent_rows["time"].to_numpy(float)
        )

        retained_segments = []

        for segment in episode.segments:
            segment_rows = table.loc[segment.row_indices].sort_values("time")
            evaluable_rows = segment_rows[
                segment_rows["local_observation_available"]
            ].copy()

            if evaluable_rows.empty:
                continue

            # The segment is fusion-evaluable because local input exists.
            # Report the full visibility-segment boundaries/duration, not the
            # sparse duration covered by local samples. The latter is already
            # reported separately as "Evaluable time [s]".
            retained_segments.append({
                "start_time": float(segment.start_time),
                "end_time": float(segment.end_time),
                "duration_s": _segment_duration(table, segment, nominal_dt),
            })

        # Gaps are computed only between two retained evaluable segments.
        retained_gaps = []
        for index in range(len(retained_segments) - 1):
            current = retained_segments[index]
            nxt = retained_segments[index + 1]
            retained_gaps.append(
                max(0.0, nxt["start_time"] - current["end_time"])
            )

        no_id_fraction = (
            1.0 - float(metric.global_association_coverage)
            if pd.notna(metric.global_association_coverage)
            else np.nan
        )

        rows.append({
            "GT agent": metric.agent_id,
            "Episode": f"E{metric.episode_id}",
            "Evaluable segments [s]": " | ".join(
                f"S{i + 1}={segment['duration_s']:.2f}"
                for i, segment in enumerate(retained_segments)
            ) if retained_segments else "—",
            "Short evaluable gaps [s]": " | ".join(
                f"G{i + 1}={gap:.2f}"
                for i, gap in enumerate(retained_gaps)
            ) if retained_gaps else "—",
            "Evaluable time [s]": round(
                float(metric.local_observation_duration_s), 3
            ),
            "Dominant Global ID": (
                metric.dominant_global_id
                if normalize_identity(metric.dominant_global_id) is not None
                else "None"
            ),
            "Stability [%]": (
                round(
                    100.0 * float(metric.global_id_stability_evaluable),
                    2,
                )
                if pd.notna(metric.global_id_stability_evaluable)
                else np.nan
            ),
            "No-ID time [%]": (
                round(100.0 * no_id_fraction, 2)
                if pd.notna(no_id_fraction)
                else np.nan
            ),
            "ID switches": int(metric.global_id_switches),
            "Re-entry": reentry_lookup.get(
                (str(metric.agent_id), int(metric.episode_id)),
                "—",
            ),
        })

    return pd.DataFrame(rows)


def _save_summary_table_figure(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render a compact landscape table with controlled spacing."""
    if summary.empty:
        return

    display = summary.copy()
    for column in ("Stability [%]", "No-ID time [%]"):
        display[column] = display[column].map(
            lambda value: "N/A" if pd.isna(value) else f"{value:.1f}"
        )
    display["Evaluable time [s]"] = display["Evaluable time [s]"].map(
        lambda value: f"{value:.2f}"
    )

    headers = [
        "GT agent", "Episode", "Segments [s]", "Short gaps [s]",
        "Evaluable time [s]", "Dominant Global ID", "Stability [%]",
        "No-ID time [%]", "ID switches", "Re-entry",
    ]
    widths = [0.07, 0.06, 0.18, 0.15, 0.10, 0.11, 0.08, 0.09, 0.07, 0.09]

    fig_height = max(3.4, 0.62 * len(display) + 1.8)
    fig, ax = plt.subplots(figsize=(17.0, fig_height))
    ax.axis("off")

    artist = ax.table(
        cellText=display.values,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        colWidths=widths,
        bbox=[0.01, 0.04, 0.98, 0.78],
    )
    artist.auto_set_font_size(False)
    artist.set_fontsize(8.3)

    for (row, _), cell in artist.get_celld().items():
        cell.set_linewidth(0.45)
        cell.get_text().set_wrap(True)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("0.91")
        elif row % 2 == 0:
            cell.set_facecolor("0.97")

    ax.set_title(
        "Global-ID temporal stability by visibility episode",
        y=0.94,
        fontweight="bold",
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def plot_results(table, episodes, epm, breakdown, agents, reentry, output_dir):
    """
    Generate thesis-ready plots for temporal Global-ID stability.

    Figures
    -------
    1. Clean Global-ID timeline with non-overlapping gap callouts.
    2. Stability per visibility episode with overall reference.
    3. Episode composition over all fusion-evaluable time.
    4. Re-entry gap duration and ID preservation in aligned panels.
    5. Spatio-temporal trajectory progress colored by Global ID.
    6. Thesis summary table.
    """
    if epm.empty:
        return

    _clean_previous_plot_files(output_dir)
    _apply_thesis_style()

    assigned_ep = (
        epm[epm["global_id_assigned_duration_s"] > EPS]
        .sort_values(["agent_id", "episode_id"])
        .copy()
    )
    assigned_agents = (
        agents[agents["total_global_id_assigned_duration_s"] > EPS]
        .sort_values("agent_id")
        .copy()
    )

    assigned_agent_ids = sorted(assigned_agents["agent_id"].astype(str))
    global_ids = sorted(
        table["matched_global_id"].dropna().astype(str).unique()
    )
    gid_colors = _global_id_color_map(global_ids)
    no_id_color = "0.72"
    stability_by_agent = _agent_stability_map(assigned_agents)

    # ==============================================================
    # FIGURE 1 — Timeline with gap bands and callouts in separate lanes
    # ==============================================================
    if assigned_agent_ids:
        row_spacing = 2.4
        y_map = {
            agent_id: index * row_spacing
            for index, agent_id in enumerate(assigned_agent_ids)
        }
        nominal_by_agent = {
            agent_id: estimate_nominal_dt(
                table[table["agent_id"] == agent_id]["time"].to_numpy(float)
            )
            for agent_id in assigned_agent_ids
        }

        fig_height = max(7.2, 1.55 * len(assigned_agent_ids))
        fig, ax = plt.subplots(figsize=(16.2, fig_height))
        used_ids = set()

        for episode in episodes:
            agent_id = str(episode.agent_id)
            if agent_id not in y_map:
                continue

            metric = assigned_ep[
                (assigned_ep["agent_id"].astype(str) == agent_id)
                & (assigned_ep["episode_id"] == episode.episode_id)
            ]
            if metric.empty:
                continue

            y = y_map[agent_id]
            nominal_dt = nominal_by_agent[agent_id]

            episode_rows_eval = table[
                (table["agent_id"].astype(str) == agent_id)
                & (table["time"] >= episode.start_time)
                & (table["time"] <= episode.end_time)
                & table["local_observation_available"]
            ].copy()

            for start, end, state in _state_intervals(
                episode_rows_eval,
                nominal_dt,
            ):
                if state is None:
                    color = no_id_color
                    linestyle = "--"
                else:
                    color = gid_colors[str(state)]
                    linestyle = "-"
                    used_ids.add(str(state))

                ax.hlines(
                    y,
                    start,
                    end,
                    linewidth=9,
                    color=color,
                    linestyles=linestyle,
                    zorder=4,
                )

            # Episode label close to the first evaluable interval.
            if not episode_rows_eval.empty:
                first_eval_time = float(episode_rows_eval["time"].min())
            else:
                first_eval_time = episode.start_time
            ax.text(
                first_eval_time,
                y + 0.34,
                f"E{episode.episode_id}",
                ha="left",
                va="bottom",
                fontsize=8.8,
                fontweight="bold",
            )

            # Show a short gap only when it lies between two actually
            # displayed fusion-evaluable intervals. A geometric segment with
            # no local observation does not qualify as the second side.
            visible_gap_index = 0
            for gap_index in range(len(episode.segments) - 1):
                current = episode.segments[gap_index]
                nxt = episode.segments[gap_index + 1]

                current_rows = table.loc[current.row_indices]
                next_rows = table.loc[nxt.row_indices]

                current_is_displayed = bool(
                    current_rows["local_observation_available"].any()
                )
                next_is_displayed = bool(
                    next_rows["local_observation_available"].any()
                )

                if not (current_is_displayed and next_is_displayed):
                    continue

                gap_start = current.end_time
                gap_end = nxt.start_time
                gap = gap_end - gap_start
                midpoint = (gap_start + gap_end) / 2.0

                ax.axvspan(
                    gap_start,
                    gap_end,
                    ymin=max(
                        0.0,
                        (y - 0.18) /
                        max(row_spacing * len(assigned_agent_ids), 1.0),
                    ),
                    ymax=min(
                        1.0,
                        (y + 0.18) /
                        max(row_spacing * len(assigned_agent_ids), 1.0),
                    ),
                    facecolor="0.90",
                    edgecolor="0.45",
                    hatch="///",
                    alpha=0.8,
                    zorder=1,
                )

                lane = visible_gap_index % 3
                visible_gap_index += 1
                label_y = y - 0.52 - 0.26 * lane

                ax.annotate(
                    f"gap {gap:.2f} s",
                    xy=(midpoint, y - 0.05),
                    xytext=(midpoint, label_y),
                    ha="center",
                    va="top",
                    fontsize=7.6,
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": "white",
                        "edgecolor": "0.65",
                        "linewidth": 0.7,
                    },
                    arrowprops={
                        "arrowstyle": "-",
                        "color": "0.45",
                        "linewidth": 0.8,
                    },
                    zorder=6,
                )

        right_limit = max(
            ep.end_time for ep in episodes
            if str(ep.agent_id) in y_map
        )
        left_limit = min(
            ep.start_time for ep in episodes
            if str(ep.agent_id) in y_map
        )
        span = max(right_limit - left_limit, 1.0)

        for agent_id, y in y_map.items():
            value = stability_by_agent[agent_id]
            ax.text(
                right_limit + 0.025 * span,
                y,
                f"Stability {100.0 * value:.1f}%",
                ha="left",
                va="center",
                fontsize=9.7,
                fontweight="bold",
            )

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        handles = [
            Line2D(
                [0], [0],
                color=gid_colors[gid],
                linewidth=8,
                label=f"Global ID {gid}",
            )
            for gid in sorted(used_ids)
        ]
        handles.extend([
            Line2D(
                [0], [0],
                color=no_id_color,
                linewidth=8,
                linestyle="--",
                label="Fusion input available, no Global ID",
            ),
            Patch(
                facecolor="0.90",
                edgecolor="0.45",
                hatch="///",
                label="Short invisibility gap within the same episode",
            ),
        ])

        ax.legend(
            handles=handles,
            title="Legend",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
        )
        ax.set_yticks([y_map[a] for a in assigned_agent_ids])
        ax.set_yticklabels(assigned_agent_ids)
        ax.set_xlabel("Simulation time [s]")
        ax.set_ylabel("GT agent")
        ax.set_title(
            "Temporal Global-ID assignment during fusion-evaluable intervals",
            pad=18,
        )
        ax.set_xlim(left_limit, right_limit + 0.20 * span)
        ax.set_ylim(-1.05, max(y_map.values()) + 0.9)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        fig.subplots_adjust(right=0.77, left=0.09, top=0.88, bottom=0.12)
        plt.savefig(
            output_dir / "01_global_id_timeline_clean.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # ==============================================================
    # FIGURE 2 — Stability by episode with overall reference
    # ==============================================================
    if not assigned_ep.empty:
        labels = assigned_ep["episode_label"].tolist()
        x = np.arange(len(labels))
        values = (
            assigned_ep["global_id_stability_evaluable"].to_numpy(float)
            * 100.0
        )

        total_eval = float(assigned_ep["local_observation_duration_s"].sum())
        total_dom = float(assigned_ep["dominant_global_id_duration_s"].sum())
        overall = 100.0 * total_dom / total_eval if total_eval > EPS else np.nan

        fig, ax = plt.subplots(
            figsize=(max(11.0, 1.45 * len(labels)), 5.8)
        )
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=2.2,
            markersize=7,
            label="Episode stability",
        )
        ax.axhline(
            overall,
            linestyle="--",
            linewidth=1.6,
            label=f"Overall = {overall:.1f}%",
        )

        for i, value in enumerate(values):
            ax.text(
                i,
                value + 0.9,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(max(0.0, min(values.min() - 6.0, 85.0)), 102.5)
        ax.set_xlabel("Visibility episode")
        ax.set_ylabel("Global-ID stability [%]")
        ax.set_title(
            "Global-ID temporal stability by visibility episode",
            pad=16,
        )
        ax.legend(
            title="Legend",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
        )
        ax.grid(axis="both")
        fig.subplots_adjust(right=0.80, bottom=0.20, top=0.88)
        plt.savefig(
            output_dir / "02_stability_per_visibility_episode.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # ==============================================================
    # FIGURE 3 — Stability composition over all fusion-evaluable time
    # ==============================================================
    if not assigned_ep.empty:
        labels = assigned_ep["episode_label"].tolist()
        y = np.arange(len(labels))
        fig, ax = plt.subplots(
            figsize=(12.6, max(5.0, 0.78 * len(labels)))
        )
        left = np.zeros(len(labels), dtype=float)

        for gid in global_ids:
            vals = []
            for label in labels:
                row = breakdown[
                    (breakdown["episode_label"] == label)
                    & (breakdown["global_id"].astype(str) == gid)
                ]
                vals.append(
                    100.0 * float(
                        row["fraction_of_evaluable_time"].iloc[0]
                    ) if not row.empty else 0.0
                )

            vals = np.asarray(vals)
            ax.barh(
                y,
                vals,
                left=left,
                color=gid_colors[gid],
                label=f"Global ID {gid}",
            )
            for i, value in enumerate(vals):
                if value >= 6.0:
                    ax.text(
                        left[i] + value / 2,
                        i,
                        f"{gid}\n{value:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=8.5,
                    )
            left += vals

        no_id = np.maximum(0.0, 100.0 - left)
        ax.barh(
            y,
            no_id,
            left=left,
            color=no_id_color,
            hatch="//",
            edgecolor="white",
            label="No Global ID",
        )

        # Keep every No-ID percentage outside the stacked bar. This gives
        # all episodes the same annotation convention and avoids text inside
        # narrow hatched regions.
        for i, value in enumerate(no_id):
            if value > 0.0:
                ax.annotate(
                    f"No ID {value:.1f}%",
                    xy=(100.0, i),
                    xytext=(102.0, i),
                    ha="left",
                    va="center",
                    fontsize=8.2,
                    arrowprops={
                        "arrowstyle": "-",
                        "color": "0.45",
                        "linewidth": 0.8,
                    },
                )

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 112)
        ax.set_xlabel("Fusion-evaluable episode time [%]")
        ax.set_ylabel("Visibility episode")
        ax.set_title(
            "Composition of temporal Global-ID stability within each episode",
            pad=16,
        )
        ax.legend(
            title="Legend",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
        )
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        fig.subplots_adjust(right=0.78, top=0.88)
        plt.savefig(
            output_dir / "03_episode_stability_composition.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # ==============================================================
    # FIGURE 5 — Re-entry plot with aligned duration and ID columns
    # ==============================================================
    if not reentry.empty:
        valid = reentry[
            reentry["same_global_id_after_reentry"].notna()
        ].copy()
        valid = valid[valid["global_id_before_exit"].notna()].copy()

        if not valid.empty:
            valid["event_label"] = (
                valid["agent_id"].astype(str)
                + "-E"
                + valid["episode_id"].astype(str)
                + "-R"
                + valid["reentry_event"].astype(str)
            )
            # Order by GT agent and then by the chronological re-entry
            # event number. This yields, for the current data:
            # agent7-R1, agent11-R1, agent11-R2.
            valid["_agent_number"] = (
                valid["agent_id"]
                .astype(str)
                .str.extract(r"(\d+)", expand=False)
                .astype(float)
            )
            valid = valid.sort_values(
                ["_agent_number", "episode_id", "reentry_event"],
                ascending=[True, True, True],
            ).reset_index(drop=True)
            y = np.arange(len(valid))

            fig = plt.figure(
                figsize=(14.2, max(5.2, 0.86 * len(valid)))
            )
            grid = fig.add_gridspec(
                1, 2,
                width_ratios=[3.0, 1.65],
                wspace=0.05,
            )
            ax = fig.add_subplot(grid[0, 0])
            ax_info = fig.add_subplot(grid[0, 1], sharey=ax)

            success_color = plt.get_cmap("tab10")(2)
            failure_color = plt.get_cmap("tab10")(3)

            for i, row in enumerate(valid.itertuples(index=False)):
                success = bool(row.same_global_id_after_reentry)
                color = success_color if success else failure_color
                gap = float(row.invisible_gap_s)

                ax.hlines(i, 0, gap, linewidth=7, color=color)
                ax.scatter(gap, i, s=72, color=color, zorder=4)
                ax.text(
                    gap + 0.10,
                    i,
                    f"{gap:.2f} s",
                    ha="left",
                    va="center",
                    fontsize=8.8,
                    bbox={
                        "boxstyle": "round,pad=0.14",
                        "facecolor": "white",
                        "edgecolor": color,
                        "linewidth": 0.8,
                    },
                )

                ax_info.text(
                    0.02,
                    i,
                    (
                        f"{clean_identity_for_display(row.global_id_before_exit)} → "
                        f"{clean_identity_for_display(row.global_id_after_reentry)}"
                    ),
                    ha="left",
                    va="center",
                    fontsize=9.2,
                    fontweight="bold",
                )
                ax_info.text(
                    0.66,
                    i,
                    "Maintained" if success else "Failed",
                    ha="left",
                    va="center",
                    fontsize=9.2,
                    color=color,
                )
            ax.set_yticks(y)
            ax.set_yticklabels(valid["event_label"])
            ax.set_xlim(
                0,
                max(1.0, float(valid["invisible_gap_s"].max()) + 1.3),
            )
            ax.set_xlabel("Invisible gap before re-entry [s]")
            ax.set_ylabel("Re-entry event")
            ax.grid(axis="x")
            ax.grid(axis="y", visible=False)

            ax_info.set_xlim(0, 1)
            ax_info.grid(False)

            # The right-hand panel contains annotations only. Disable the
            # complete axis frame so no residual spine/tick can look like a
            # bracket next to the Global-ID transition.
            ax_info.set_axis_off()
            # Column headers are placed below the main title and directly
            # above the annotation columns.
            ax_info.text(
                0.02,
                1.035,
                "Global-ID transition",
                transform=ax_info.transAxes,
                fontsize=9,
                fontweight="bold",
                ha="left",
                va="bottom",
                clip_on=False,
            )
            ax_info.text(
                0.66,
                1.035,
                "Outcome",
                transform=ax_info.transAxes,
                fontsize=9,
                fontweight="bold",
                ha="left",
                va="bottom",
                clip_on=False,
            )
            fig.suptitle(
                "Global-ID preservation after temporary invisibility",
                y=0.985,
                fontsize=14,
            )
            fig.subplots_adjust(
                right=0.96,
                left=0.13,
                top=0.80,
                bottom=0.13,
            )
            plt.savefig(
                output_dir / "05_reentry_preservation_events.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

    # ==============================================================
    # FIGURE 6 — Spatio-temporal trajectory progress colored by ID
    # ==============================================================
    if assigned_agent_ids:
        fig, axes = plt.subplots(
            len(assigned_agent_ids),
            1,
            figsize=(15.2, max(7.0, 2.8 * len(assigned_agent_ids))),
            sharex=True,
        )
        if len(assigned_agent_ids) == 1:
            axes = [axes]

        used_ids = set()

        # Annotation settings. IDs remain repeated to reproduce the visual
        # continuity of RViz markers, but they are not printed at every sample.
        annotation_min_time_s = 0.65
        annotation_edge_margin_s = 0.12
        annotation_offset_pt = 9

        for ax, agent_id in zip(axes, assigned_agent_ids):
            rows = table[
                (table["agent_id"].astype(str) == agent_id)
                & table["visible"]
            ].sort_values("time").copy()

            if rows.empty:
                continue

            # ----------------------------------------------------------
            # Cumulative GT path distance
            # ----------------------------------------------------------
            xy = rows[["gt_x", "gt_y"]].to_numpy(float)
            step = np.zeros(len(rows), dtype=float)

            nominal_dt = estimate_nominal_dt(
                rows["time"].to_numpy(float)
            )

            if len(rows) > 1:
                diffs = np.linalg.norm(np.diff(xy, axis=0), axis=1)
                time_diffs = np.diff(rows["time"].to_numpy(float))

                # Never infer travelled distance across an unobserved/logging
                # gap. The cumulative path resumes from its previous value.
                diffs[time_diffs > 2.5 * nominal_dt] = 0.0
                step[1:] = diffs

            rows["path_progress_m"] = np.cumsum(step)

            # Only locally observable samples are fusion-evaluable.
            eval_rows = rows[
                rows["local_observation_available"]
            ].copy()

            if eval_rows.empty:
                continue

            # ----------------------------------------------------------
            # Draw Global-ID intervals and repeated, spaced annotations
            # ----------------------------------------------------------
            annotation_counter = 0
            last_annotation_time = -np.inf
            last_annotation_state = None

            for interval_start, interval_end, state in _state_intervals(
                eval_rows,
                nominal_dt,
            ):
                segment = eval_rows[
                    (eval_rows["time"] >= interval_start - EPS)
                    & (eval_rows["time"] <= interval_end + EPS)
                ].copy()

                if segment.empty:
                    continue

                if state is None:
                    color = no_id_color
                    linestyle = "--"
                else:
                    state = clean_identity_for_display(state)
                    color = gid_colors[str(state)]
                    linestyle = "-"
                    used_ids.add(str(state))

                segment_times = segment["time"].to_numpy(float)
                segment_path = segment["path_progress_m"].to_numpy(float)

                ax.plot(
                    segment_times,
                    segment_path,
                    color=color,
                    linewidth=2.8,
                    linestyle=linestyle,
                    solid_capstyle="round",
                    zorder=3,
                )

                if state is None:
                    continue

                # Candidate annotations are generated inside the interval,
                # but accepted using one spacing rule across the whole agent.
                duration = float(segment_times[-1] - segment_times[0])
                candidate_indices = []

                if duration <= annotation_min_time_s:
                    candidate_indices = [len(segment) // 2]
                else:
                    target_time = (
                        segment_times[0] + annotation_edge_margin_s
                    )
                    latest_allowed = (
                        segment_times[-1] - annotation_edge_margin_s
                    )

                    while target_time <= latest_allowed + EPS:
                        nearest_index = int(
                            np.argmin(np.abs(segment_times - target_time))
                        )
                        if (
                            not candidate_indices
                            or nearest_index != candidate_indices[-1]
                        ):
                            candidate_indices.append(nearest_index)
                        target_time += annotation_min_time_s

                    if not candidate_indices:
                        candidate_indices = [len(segment) // 2]

                selected_indices = []
                for candidate_index in candidate_indices:
                    candidate_time = float(
                        segment.iloc[candidate_index]["time"]
                    )
                    state_changed = state != last_annotation_state

                    if (
                        state_changed
                        or candidate_time - last_annotation_time
                        >= annotation_min_time_s
                    ):
                        selected_indices.append(candidate_index)
                        last_annotation_time = candidate_time
                        last_annotation_state = state

                for selected_index in selected_indices:
                    selected_row = segment.iloc[selected_index]
                    offset_y = annotation_offset_pt
                    annotation_counter += 1

                    ax.annotate(
                        str(state),
                        xy=(
                            float(selected_row["time"]),
                            float(selected_row["path_progress_m"]),
                        ),
                        xytext=(0, offset_y),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7.4,
                        color=color,
                        fontweight="bold",
                        zorder=6,
                    )

            # ----------------------------------------------------------
            # Mark valid short invisibility gaps within the same episode
            # ----------------------------------------------------------
            agent_reentries = pd.DataFrame()
            if not reentry.empty:
                agent_reentries = reentry[
                    (reentry["agent_id"].astype(str) == agent_id)
                    & reentry["same_global_id_after_reentry"].notna()
                ].copy()

            if not agent_reentries.empty:
                agent_reentries = agent_reentries.sort_values(
                    ["episode_id", "reentry_event"]
                )

                y_min, y_max = ax.get_ylim()
                y_span = max(y_max - y_min, 1e-6)

                for gap_number, event in enumerate(
                    agent_reentries.itertuples(index=False)
                ):
                    gap_start = float(event.exit_time)
                    gap_end = float(event.geometric_reentry_time)
                    gap_duration = float(event.invisible_gap_s)

                    before_rows = eval_rows[
                        eval_rows["time"] <= gap_start + nominal_dt
                    ]
                    after_rows = eval_rows[
                        eval_rows["time"] >= gap_end - nominal_dt
                    ]

                    # Draw a gap only when there are fusion-evaluable samples
                    # both before and after the invisible interval.
                    if before_rows.empty or after_rows.empty:
                        continue

                    midpoint = (gap_start + gap_end) / 2.0

                    # Position the gap marker at the level of the neighbouring
                    # trajectory portions. This keeps the temporal gap visually
                    # tied to the curve instead of placing it at the subplot base.
                    before_point = before_rows.iloc[-1]
                    after_point = after_rows.iloc[0]

                    before_y = float(before_point["path_progress_m"])
                    after_y = float(after_point["path_progress_m"])
                    gap_y = 0.5 * (before_y + after_y)

                    ax.plot(
                        [gap_start, gap_end],
                        [gap_y, gap_y],
                        color="0.30",
                        linewidth=1.1,
                        linestyle=(0, (3, 2)),
                        zorder=2,
                    )

                    ax.annotate(
                        f"gap {gap_duration:.2f} s",
                        xy=(midpoint, gap_y),
                        xytext=(0, -13),
                        textcoords="offset points",
                        ha="center",
                        va="top",
                        fontsize=8.8,
                        color="black",
                        bbox={
                            "boxstyle": "round,pad=0.16",
                            "facecolor": "white",
                            "edgecolor": "0.60",
                            "linewidth": 0.7,
                            "alpha": 0.92,
                        },
                        zorder=7,
                    )

            ax.set_ylabel(
                f"{agent_id}\nCumulative GT\npath distance [m]",
                rotation=0,
                ha="right",
                va="center",
                labelpad=42,
                fontsize=9.5,
            )
            ax.grid(axis="both")

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        handles = [
            Line2D(
                [0], [0],
                color=gid_colors[gid],
                linewidth=3,
                label=f"Global ID {gid}",
            )
            for gid in sorted(used_ids)
        ]
        handles.extend([
            Line2D(
                [0], [0],
                color=no_id_color,
                linewidth=3,
                linestyle="--",
                label="Fusion input available, no Global ID",
            ),
            Line2D(
                [0], [0],
                color="0.55",
                linewidth=1.0,
                linestyle=(0, (3, 2)),
                label="Short non-evaluable gap within the same episode",
            ),
        ])

        axes[0].legend(
            handles=handles,
            title="Legend",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
        )
        axes[-1].set_xlabel("Simulation time [s]")
        fig.suptitle(
            "Agent path progression over time with associated Global IDs",
            y=0.99,
            fontsize=15,
        )
        fig.subplots_adjust(
            right=0.77,
            left=0.17,
            top=0.92,
            bottom=0.08,
            hspace=0.38,
        )
        plt.savefig(
            output_dir / "06_spatiotemporal_agent_trajectories.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # ==============================================================
    # TABLE
    # ==============================================================
    summary = _build_episode_summary_table(
        epm,
        episodes,
        table,
        reentry,
    )
    summary.to_csv(
        output_dir / "thesis_global_id_stability_table.csv",
        index=False,
    )
    _save_summary_table_figure(
        summary,
        output_dir / "07_thesis_global_id_stability_table.png",
    )



# ============================================================================
# STANDARD MOT METRICS FOR THE GLOBAL FUSION FILTER
# ============================================================================
#
# Evaluation philosophy:
#   - visibility episodes are the SAME episodes already built above;
#   - only GT samples with local_observation_available == True are evaluable;
#   - one evaluation GT sample is kept per (local_snapshot_time, episode ID),
#     preventing dense GT logging from counting the same local input repeatedly;
#   - global outputs are evaluated only in the spatial neighbourhood supported
#     by the currently available local inputs. This is a COMPONENT-LEVEL
#     evaluation of the global fusion filter, not an end-to-end detector test.
#
# HOTA and CLEAR follow the TrackEval mathematical formulation. The only
# adaptation is the detection similarity because this system outputs metric
# map-frame points rather than image bounding boxes.
#
# HOTA point similarity:
#     s(d) = max(0, 1 - d / D_HOTA)
#
# CLEAR point similarity is defined so that the standard TrackEval threshold
# 0.5 corresponds exactly to CLEAR_MATCH_DISTANCE:
#     s_clear(d) = max(0, 1 - d / (2 * CLEAR_MATCH_DISTANCE))
#
# Therefore:
#     s_clear >= 0.5  <=>  d <= CLEAR_MATCH_DISTANCE
# ============================================================================


def _assign_episode_identity_to_table(
    table: pd.DataFrame,
    episodes: Sequence[VisibilityEpisode],
) -> pd.DataFrame:
    result = table.copy()
    result["mot_episode_id"] = pd.Series([pd.NA] * len(result), dtype="Int64")
    result["mot_evaluation_gt_id"] = pd.Series(
        [None] * len(result), dtype="object"
    )

    for ep in episodes:
        eval_id = f"{ep.agent_id}-E{ep.episode_id}"
        for seg in ep.segments:
            if not seg.row_indices:
                continue
            result.loc[seg.row_indices, "mot_episode_id"] = int(ep.episode_id)
            result.loc[seg.row_indices, "mot_evaluation_gt_id"] = eval_id

    return result


def _build_fusion_evaluable_samples(
    table: pd.DataFrame,
    episodes: Sequence[VisibilityEpisode],
) -> pd.DataFrame:
    """
    Build one MOT GT sample per actual local-input snapshot and evaluation
    trajectory.

    Several GT rows can point to the same local snapshot because GT may be
    logged faster than the local tracker. Keeping all of them would artificially
    inflate TP/FN counts. We therefore retain the GT row temporally closest to
    each matched local observation.
    """
    t = _assign_episode_identity_to_table(table, episodes)

    ev = t[
        t["visible"]
        & t["local_observation_available"]
        & t["mot_evaluation_gt_id"].notna()
        & t["local_snapshot_time"].notna()
    ].copy()

    if ev.empty:
        return ev

    # The row with minimum local_time_error_s is the best GT representation of
    # that actual local-input observation.
    ev["_local_abs_error"] = ev["local_time_error_s"].abs()

    ev = (
        ev.sort_values(
            [
                "local_snapshot_time",
                "mot_evaluation_gt_id",
                "_local_abs_error",
                "time",
            ]
        )
        .drop_duplicates(
            subset=["local_snapshot_time", "mot_evaluation_gt_id"],
            keep="first",
        )
        .drop(columns=["_local_abs_error"])
        .sort_values(["local_snapshot_time", "mot_evaluation_gt_id"])
        .reset_index(drop=True)
    )

    return ev


def _nearest_snapshot_time_mot(
    sorted_times: np.ndarray,
    query_time: float,
    tolerance: float,
) -> Optional[float]:
    if len(sorted_times) == 0:
        return None

    idx = int(np.searchsorted(sorted_times, query_time))
    candidates = []

    if idx < len(sorted_times):
        candidates.append(float(sorted_times[idx]))
    if idx > 0:
        candidates.append(float(sorted_times[idx - 1]))

    if not candidates:
        return None

    nearest = min(candidates, key=lambda x: abs(x - query_time))
    return (
        nearest
        if abs(nearest - query_time) <= tolerance + EPS
        else None
    )


def _metric_similarity(
    distances: np.ndarray,
    distance_scale_m: float,
) -> np.ndarray:
    if distance_scale_m <= 0:
        raise ValueError("distance scale must be > 0.")
    return np.clip(
        1.0 - distances / float(distance_scale_m),
        0.0,
        1.0,
    )


def _build_component_mot_frames(
    evaluable: pd.DataFrame,
    tracks: pd.DataFrame,
    track_time_tolerance_s: float,
    output_scope_distance_m: float,
) -> Tuple[List[dict], pd.DataFrame]:
    """
    Build MOT frames at ACTUAL local-input timestamps.

    Global outputs are included in the component-level evaluation scope when
    they lie within output_scope_distance_m from at least one current local
    input. The scope decision uses the filter INPUT, not GT, so GT is not used
    to cherry-pick tracker outputs.

    Duplicate global outputs around a valid local input are retained and may
    therefore become false positives.
    """
    if output_scope_distance_m <= 0:
        raise ValueError("output_scope_distance_m must be > 0.")

    track_times = np.sort(tracks["time"].unique().astype(float))
    tracks_by_time = {
        float(t): g.reset_index(drop=True)
        for t, g in tracks.groupby("time", sort=True)
    }

    frames: List[dict] = []
    scope_diagnostics: List[dict] = []

    for local_time, gt_group in evaluable.groupby(
        "local_snapshot_time", sort=True
    ):
        gt_group = gt_group.sort_values(
            "mot_evaluation_gt_id"
        ).reset_index(drop=True)

        gt_ids = gt_group["mot_evaluation_gt_id"].astype(str).tolist()
        physical_ids = gt_group["agent_id"].astype(str).tolist()

        gt_xy = gt_group[["gt_x", "gt_y"]].to_numpy(float)
        local_xy = gt_group[
            ["matched_local_x", "matched_local_y"]
        ].to_numpy(float)

        snap = _nearest_snapshot_time_mot(
            track_times,
            float(local_time),
            track_time_tolerance_s,
        )

        if snap is None:
            scoped_tracks = tracks.iloc[0:0].copy()
        else:
            raw_tracks = tracks_by_time[snap].copy()

            if raw_tracks.empty:
                scoped_tracks = raw_tracks
            else:
                raw_xy = raw_tracks[
                    ["track_x", "track_y"]
                ].to_numpy(float)

                dist_to_inputs = np.linalg.norm(
                    raw_xy[:, None, :] - local_xy[None, :, :],
                    axis=2,
                )
                min_dist = dist_to_inputs.min(axis=1)
                keep = (
                    min_dist
                    <= output_scope_distance_m + EPS
                )

                for row_index, row in raw_tracks.iterrows():
                    scope_diagnostics.append({
                        "evaluation_time": float(local_time),
                        "track_snapshot_time": float(snap),
                        "global_id": str(row["global_id"]),
                        "track_x": float(row["track_x"]),
                        "track_y": float(row["track_y"]),
                        "nearest_current_local_input_distance_m":
                            float(min_dist[row_index]),
                        "included_in_component_mot_scope":
                            bool(keep[row_index]),
                    })

                scoped_tracks = (
                    raw_tracks.loc[keep]
                    .copy()
                    .reset_index(drop=True)
                )

        tracker_ids = (
            scoped_tracks["global_id"].astype(str).tolist()
            if not scoped_tracks.empty
            else []
        )
        tracker_xy = (
            scoped_tracks[["track_x", "track_y"]].to_numpy(float)
            if not scoped_tracks.empty
            else np.zeros((0, 2), dtype=float)
        )

        if len(gt_xy) and len(tracker_xy):
            distances = np.linalg.norm(
                gt_xy[:, None, :] - tracker_xy[None, :, :],
                axis=2,
            )
        else:
            distances = np.zeros(
                (len(gt_ids), len(tracker_ids)),
                dtype=float,
            )

        frames.append({
            "time": float(local_time),
            "track_snapshot_time": (
                float(snap) if snap is not None else np.nan
            ),
            "track_time_error_s": (
                abs(float(snap) - float(local_time))
                if snap is not None
                else np.nan
            ),
            "gt_ids": gt_ids,
            "physical_agent_ids": physical_ids,
            "gt_xy": gt_xy,
            "local_xy": local_xy,
            "tracker_ids": tracker_ids,
            "tracker_xy": tracker_xy,
            "distances": distances,
        })

    return frames, pd.DataFrame(scope_diagnostics)


def _evaluate_hota_trackeval_formulation(
    frames: Sequence[dict],
    distance_scale_m: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    HOTA following the TrackEval sequence formulation:
      1) normalized potential matches;
      2) global Jaccard alignment;
      3) Hungarian on global_alignment * frame similarity;
      4) alpha-specific TP/FN/FP and association metrics.
    """
    alphas = np.arange(0.05, 0.99, 0.05)

    all_gt_ids = sorted({
        gid for frame in frames for gid in frame["gt_ids"]
    })
    all_tr_ids = sorted({
        tid for frame in frames for tid in frame["tracker_ids"]
    })

    gt_to_int = {gid: i for i, gid in enumerate(all_gt_ids)}
    tr_to_int = {tid: i for i, tid in enumerate(all_tr_ids)}

    n_gt_ids = len(all_gt_ids)
    n_tr_ids = len(all_tr_ids)

    potential_matches_count = np.zeros(
        (n_gt_ids, n_tr_ids), dtype=float
    )
    gt_id_count = np.zeros((n_gt_ids, 1), dtype=float)
    tracker_id_count = np.zeros((1, n_tr_ids), dtype=float)

    # Pass 1: global association information.
    for frame in frames:
        gt_ids_t = np.array(
            [gt_to_int[x] for x in frame["gt_ids"]],
            dtype=int,
        )
        tracker_ids_t = np.array(
            [tr_to_int[x] for x in frame["tracker_ids"]],
            dtype=int,
        )

        similarity = _metric_similarity(
            frame["distances"],
            distance_scale_m,
        )

        if len(gt_ids_t) and len(tracker_ids_t):
            sim_iou_denom = (
                similarity.sum(0)[np.newaxis, :]
                + similarity.sum(1)[:, np.newaxis]
                - similarity
            )
            sim_iou = np.zeros_like(similarity)
            mask = sim_iou_denom > np.finfo(float).eps
            sim_iou[mask] = (
                similarity[mask] / sim_iou_denom[mask]
            )

            potential_matches_count[
                gt_ids_t[:, np.newaxis],
                tracker_ids_t[np.newaxis, :],
            ] += sim_iou

        if len(gt_ids_t):
            gt_id_count[gt_ids_t] += 1
        if len(tracker_ids_t):
            tracker_id_count[0, tracker_ids_t] += 1

    if n_gt_ids and n_tr_ids:
        denom = (
            gt_id_count
            + tracker_id_count
            - potential_matches_count
        )
        global_alignment_score = np.divide(
            potential_matches_count,
            denom,
            out=np.zeros_like(potential_matches_count),
            where=denom > np.finfo(float).eps,
        )
    else:
        global_alignment_score = np.zeros(
            (n_gt_ids, n_tr_ids), dtype=float
        )

    hota_tp = np.zeros(len(alphas), dtype=int)
    hota_fn = np.zeros(len(alphas), dtype=int)
    hota_fp = np.zeros(len(alphas), dtype=int)
    loca_sum = np.zeros(len(alphas), dtype=float)

    matches_counts = [
        np.zeros_like(potential_matches_count)
        for _ in alphas
    ]

    match_debug = []

    # Pass 2: frame matching.
    for frame in frames:
        gt_ids_t = np.array(
            [gt_to_int[x] for x in frame["gt_ids"]],
            dtype=int,
        )
        tracker_ids_t = np.array(
            [tr_to_int[x] for x in frame["tracker_ids"]],
            dtype=int,
        )

        n_gt = len(gt_ids_t)
        n_tr = len(tracker_ids_t)

        if n_gt == 0:
            hota_fp += n_tr
            continue

        if n_tr == 0:
            hota_fn += n_gt
            continue

        similarity = _metric_similarity(
            frame["distances"],
            distance_scale_m,
        )

        score_mat = (
            global_alignment_score[
                gt_ids_t[:, np.newaxis],
                tracker_ids_t[np.newaxis, :],
            ]
            * similarity
        )

        match_rows, match_cols = linear_sum_assignment(
            -score_mat
        )

        for a, alpha in enumerate(alphas):
            actually_matched = (
                similarity[match_rows, match_cols]
                >= alpha - np.finfo(float).eps
            )
            alpha_rows = match_rows[actually_matched]
            alpha_cols = match_cols[actually_matched]

            num_matches = len(alpha_rows)

            hota_tp[a] += num_matches
            hota_fn[a] += n_gt - num_matches
            hota_fp[a] += n_tr - num_matches

            if num_matches == 0:
                continue

            loca_sum[a] += float(np.sum(
                similarity[alpha_rows, alpha_cols]
            ))

            matches_counts[a][
                gt_ids_t[alpha_rows],
                tracker_ids_t[alpha_cols],
            ] += 1

            for r, c in zip(alpha_rows, alpha_cols):
                match_debug.append({
                    "alpha": float(alpha),
                    "time": frame["time"],
                    "evaluation_gt_id": frame["gt_ids"][r],
                    "global_id": frame["tracker_ids"][c],
                    "distance_m": float(
                        frame["distances"][r, c]
                    ),
                    "similarity": float(
                        similarity[r, c]
                    ),
                })

    rows = []

    for a, alpha in enumerate(alphas):
        matches_count = matches_counts[a]

        if n_gt_ids and n_tr_ids:
            ass_a_matrix = (
                matches_count
                / np.maximum(
                    1.0,
                    gt_id_count
                    + tracker_id_count
                    - matches_count,
                )
            )
            ass_re_matrix = (
                matches_count
                / np.maximum(1.0, gt_id_count)
            )
            ass_pr_matrix = (
                matches_count
                / np.maximum(1.0, tracker_id_count)
            )

            ass_a = float(
                np.sum(matches_count * ass_a_matrix)
                / max(1, hota_tp[a])
            )
            ass_re = float(
                np.sum(matches_count * ass_re_matrix)
                / max(1, hota_tp[a])
            )
            ass_pr = float(
                np.sum(matches_count * ass_pr_matrix)
                / max(1, hota_tp[a])
            )
        else:
            ass_a = 0.0
            ass_re = 0.0
            ass_pr = 0.0

        det_re = (
            hota_tp[a]
            / max(1, hota_tp[a] + hota_fn[a])
        )
        det_pr = (
            hota_tp[a]
            / max(1, hota_tp[a] + hota_fp[a])
        )
        det_a = (
            hota_tp[a]
            / max(
                1,
                hota_tp[a]
                + hota_fn[a]
                + hota_fp[a],
            )
        )
        loc_a = (
            loca_sum[a]
            / max(np.finfo(float).eps, float(hota_tp[a]))
        )
        hota = math.sqrt(det_a * ass_a)
        owta = math.sqrt(det_re * ass_a)

        rows.append({
            "alpha": float(alpha),
            "HOTA": hota,
            "DetA": det_a,
            "AssA": ass_a,
            "DetRe": det_re,
            "DetPr": det_pr,
            "AssRe": ass_re,
            "AssPr": ass_pr,
            "LocA": loc_a,
            "OWTA": owta,
            "HOTA_TP": int(hota_tp[a]),
            "HOTA_FN": int(hota_fn[a]),
            "HOTA_FP": int(hota_fp[a]),
        })

    hota_df = pd.DataFrame(rows)

    if not hota_df.empty:
        average = {
            "alpha": "AVERAGE",
            "HOTA": float(hota_df["HOTA"].mean()),
            "DetA": float(hota_df["DetA"].mean()),
            "AssA": float(hota_df["AssA"].mean()),
            "DetRe": float(hota_df["DetRe"].mean()),
            "DetPr": float(hota_df["DetPr"].mean()),
            "AssRe": float(hota_df["AssRe"].mean()),
            "AssPr": float(hota_df["AssPr"].mean()),
            "LocA": float(hota_df["LocA"].mean()),
            "OWTA": float(hota_df["OWTA"].mean()),
            "HOTA_TP": np.nan,
            "HOTA_FN": np.nan,
            "HOTA_FP": np.nan,
        }
        hota_df = pd.concat(
            [hota_df, pd.DataFrame([average])],
            ignore_index=True,
        )

    return hota_df, pd.DataFrame(match_debug)


def _evaluate_clear_trackeval_formulation(
    frames: Sequence[dict],
    clear_match_distance_m: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    CLEAR MOT adapted consistently to the input-conditioned evaluation domain.

    Important detail:
    The benchmark contains only samples where a GT target is visible AND a
    valid local observation is available to the global fusion filter.

    Therefore, temporal continuity for one GT trajectory must be evaluated
    between consecutive EVALUABLE samples of that same trajectory, not between
    consecutive global timestamps where that GT may be absent from the
    benchmark entirely.

    Matching:
      - previous evaluable tracker ID for the same GT gets 1000x priority;
      - standard similarity threshold = 0.5;
      - IDSW compares the current tracker ID against the last tracker ID matched
        to that GT on a previous evaluable sample;
      - Fragmentation counts a new tracked segment only when a GT trajectory
        was unmatched at its previous evaluable sample and becomes matched
        again later. Non-evaluable gaps do not create fragmentations.
    """
    if clear_match_distance_m <= 0:
        raise ValueError(
            "clear_match_distance_m must be > 0."
        )

    similarity_scale = 2.0 * clear_match_distance_m
    threshold = 0.5

    all_gt_ids = sorted({
        gid for frame in frames for gid in frame["gt_ids"]
    })
    all_tr_ids = sorted({
        tid for frame in frames for tid in frame["tracker_ids"]
    })

    gt_to_int = {gid: i for i, gid in enumerate(all_gt_ids)}
    tr_to_int = {tid: i for i, tid in enumerate(all_tr_ids)}

    num_gt_ids = len(all_gt_ids)

    gt_id_count = np.zeros(num_gt_ids, dtype=float)
    gt_matched_count = np.zeros(num_gt_ids, dtype=float)

    # Previous tracker ID matched to the GT on its previous evaluable sample.
    prev_evaluable_tracker_id = np.full(num_gt_ids, np.nan)

    # Last tracker ID ever matched to the GT (for IDSW definition).
    last_matched_tracker_id = np.full(num_gt_ids, np.nan)

    # For fragmentation: state only advances when the GT is evaluable.
    has_been_evaluable = np.zeros(num_gt_ids, dtype=bool)
    prev_evaluable_was_matched = np.zeros(num_gt_ids, dtype=bool)
    tracked_segment_count = np.zeros(num_gt_ids, dtype=int)

    tp = fn = fp = idsw = 0
    motp_sum = 0.0
    distance_sum = 0.0

    matches = []
    events = []

    for frame in frames:
        gt_ids_t = np.array(
            [gt_to_int[x] for x in frame["gt_ids"]],
            dtype=int,
        )
        tracker_ids_t = np.array(
            [tr_to_int[x] for x in frame["tracker_ids"]],
            dtype=int,
        )

        n_gt = len(gt_ids_t)
        n_tr = len(tracker_ids_t)

        # Frames should normally contain at least one evaluable GT, but keep
        # the generic case correct.
        if n_gt == 0:
            fp += n_tr
            for c in range(n_tr):
                events.append({
                    "time": frame["time"],
                    "event_type": "FP",
                    "agent_id": None,
                    "evaluation_gt_id": None,
                    "previous_global_id": None,
                    "current_global_id": frame["tracker_ids"][c],
                })
            continue

        gt_id_count[gt_ids_t] += 1

        if n_tr == 0:
            fn += n_gt

            # Every present GT has an evaluable sample here, and it is unmatched.
            for r, gt_idx in enumerate(gt_ids_t):
                events.append({
                    "time": frame["time"],
                    "event_type": "FN",
                    "agent_id": frame["physical_agent_ids"][r],
                    "evaluation_gt_id": frame["gt_ids"][r],
                    "previous_global_id": (
                        all_tr_ids[int(last_matched_tracker_id[gt_idx])]
                        if np.isfinite(last_matched_tracker_id[gt_idx])
                        else None
                    ),
                    "current_global_id": None,
                })

                has_been_evaluable[gt_idx] = True
                prev_evaluable_was_matched[gt_idx] = False
                prev_evaluable_tracker_id[gt_idx] = np.nan

            continue

        similarity = _metric_similarity(
            frame["distances"],
            similarity_scale,
        )

        # Continuity priority is based on the previous EVALUABLE sample of each
        # GT trajectory, not the previous global timestamp.
        continuity = (
            tracker_ids_t[np.newaxis, :]
            == prev_evaluable_tracker_id[
                gt_ids_t[:, np.newaxis]
            ]
        ).astype(float)

        score_mat = 1000.0 * continuity + similarity
        score_mat[
            similarity < threshold - np.finfo(float).eps
        ] = 0.0

        match_rows, match_cols = linear_sum_assignment(
            -score_mat
        )

        valid = (
            score_mat[match_rows, match_cols]
            > np.finfo(float).eps
        )
        match_rows = match_rows[valid]
        match_cols = match_cols[valid]

        matched_gt_ids = gt_ids_t[match_rows]
        matched_tracker_ids = tracker_ids_t[match_cols]

        matched_by_gt = {
            int(gt_idx): int(tr_idx)
            for gt_idx, tr_idx in zip(
                matched_gt_ids,
                matched_tracker_ids,
            )
        }

        # ID switches: compare against the last previously matched ID for this
        # evaluation trajectory, even if an unmatched evaluable sample occurred
        # in between.
        for k, gt_idx in enumerate(matched_gt_ids):
            current_tr_idx = int(matched_tracker_ids[k])
            previous_tr_idx = last_matched_tracker_id[gt_idx]

            if (
                np.isfinite(previous_tr_idx)
                and current_tr_idx != int(previous_tr_idx)
            ):
                idsw += 1
                r = int(match_rows[k])
                c = int(match_cols[k])
                events.append({
                    "time": frame["time"],
                    "event_type": "IDSW",
                    "agent_id": frame["physical_agent_ids"][r],
                    "evaluation_gt_id": frame["gt_ids"][r],
                    "previous_global_id": all_tr_ids[int(previous_tr_idx)],
                    "current_global_id": frame["tracker_ids"][c],
                })

            last_matched_tracker_id[gt_idx] = current_tr_idx

        gt_matched_count[matched_gt_ids] += 1

        # Fragmentation:
        # A new tracked segment starts when a GT is matched now but was
        # unmatched at its previous EVALUABLE sample. A GT being absent from
        # other frames does not modify this state.
        for gt_idx in gt_ids_t:
            gt_idx = int(gt_idx)
            current_matched = gt_idx in matched_by_gt

            if current_matched:
                if (
                    not has_been_evaluable[gt_idx]
                    or not prev_evaluable_was_matched[gt_idx]
                ):
                    tracked_segment_count[gt_idx] += 1

                prev_evaluable_was_matched[gt_idx] = True
                prev_evaluable_tracker_id[gt_idx] = matched_by_gt[gt_idx]
            else:
                prev_evaluable_was_matched[gt_idx] = False
                prev_evaluable_tracker_id[gt_idx] = np.nan

            has_been_evaluable[gt_idx] = True

        num_matches = len(matched_gt_ids)

        tp += num_matches
        fn += n_gt - num_matches
        fp += n_tr - num_matches

        if num_matches > 0:
            motp_sum += float(np.sum(
                similarity[match_rows, match_cols]
            ))
            distance_sum += float(np.sum(
                frame["distances"][match_rows, match_cols]
            ))

        matched_row_set = set(match_rows.tolist())
        matched_col_set = set(match_cols.tolist())

        for r, c in zip(match_rows, match_cols):
            matches.append({
                "time": frame["time"],
                "track_snapshot_time": frame["track_snapshot_time"],
                "track_time_error_s": frame["track_time_error_s"],
                "agent_id": frame["physical_agent_ids"][r],
                "evaluation_gt_id": frame["gt_ids"][r],
                "global_id": frame["tracker_ids"][c],
                "gt_x": float(frame["gt_xy"][r, 0]),
                "gt_y": float(frame["gt_xy"][r, 1]),
                "track_x": float(frame["tracker_xy"][c, 0]),
                "track_y": float(frame["tracker_xy"][c, 1]),
                "distance_m": float(frame["distances"][r, c]),
                "clear_similarity": float(similarity[r, c]),
            })

        for r in range(n_gt):
            if r in matched_row_set:
                continue

            gt_idx = int(gt_ids_t[r])
            events.append({
                "time": frame["time"],
                "event_type": "FN",
                "agent_id": frame["physical_agent_ids"][r],
                "evaluation_gt_id": frame["gt_ids"][r],
                "previous_global_id": (
                    all_tr_ids[int(last_matched_tracker_id[gt_idx])]
                    if np.isfinite(last_matched_tracker_id[gt_idx])
                    else None
                ),
                "current_global_id": None,
            })

        for c in range(n_tr):
            if c in matched_col_set:
                continue
            events.append({
                "time": frame["time"],
                "event_type": "FP",
                "agent_id": None,
                "evaluation_gt_id": None,
                "previous_global_id": None,
                "current_global_id": frame["tracker_ids"][c],
            })

    # Number of tracked segments minus the first one for each GT trajectory.
    frag = int(np.sum(
        np.maximum(0, tracked_segment_count - 1)
    ))

    total_gt = tp + fn

    tracked_ratio = (
        gt_matched_count[gt_id_count > 0]
        / gt_id_count[gt_id_count > 0]
        if np.any(gt_id_count > 0)
        else np.array([], dtype=float)
    )

    mt = int(np.sum(tracked_ratio > 0.8))
    pt = int(
        np.sum((tracked_ratio >= 0.2) & (tracked_ratio <= 0.8))
    )
    ml = int(np.sum(tracked_ratio < 0.2))

    mota = (
        (tp - fp - idsw)
        / max(1.0, float(total_gt))
    )
    motp_similarity = (
        motp_sum / max(1.0, float(tp))
    )
    mean_distance = (
        distance_sum / max(1.0, float(tp))
    )
    recall = tp / max(1.0, float(tp + fn))
    precision = tp / max(1.0, float(tp + fp))

    clear_df = pd.DataFrame([{
        "MOTA": mota,
        "MOTP_similarity": motp_similarity,
        "MOTP_mean_distance_m": mean_distance,
        "CLR_TP": int(tp),
        "CLR_FN": int(fn),
        "CLR_FP": int(fp),
        "IDSW": int(idsw),
        "Frag": int(frag),
        "MT": mt,
        "PT": pt,
        "ML": ml,
        "CLR_Recall": recall,
        "CLR_Precision": precision,
        "CLR_Frames": int(len(frames)),
    }])

    return (
        clear_df,
        pd.DataFrame(matches),
        pd.DataFrame(events),
    )

def _compute_position_metrics(
    matches: pd.DataFrame,
) -> Dict[str, float]:
    if matches.empty:
        return {
            "mean_position_error_m": np.nan,
            "median_position_error_m": np.nan,
            "rmse_position_error_m": np.nan,
            "p95_position_error_m": np.nan,
            "max_position_error_m": np.nan,
        }

    e = matches["distance_m"].to_numpy(float)

    return {
        "mean_position_error_m": float(np.mean(e)),
        "median_position_error_m": float(np.median(e)),
        "rmse_position_error_m":
            float(np.sqrt(np.mean(e ** 2))),
        "p95_position_error_m":
            float(np.percentile(e, 95)),
        "max_position_error_m": float(np.max(e)),
    }


def _compute_error_jitter(
    matches: pd.DataFrame,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Jitter = temporal variation of the localization ERROR VECTOR:
        e_t = p_track(t) - p_GT(t)

        J_t = ||e_t - e_(t-1)|| / dt

    Only consecutive samples of the same evaluation GT ID AND same Global ID
    are used. This prevents ID switches from being misinterpreted as jitter.
    """
    if matches.empty:
        return {
            "mean_error_jitter_mps": np.nan,
            "rms_error_jitter_mps": np.nan,
            "p95_error_jitter_mps": np.nan,
        }, pd.DataFrame()

    rows = []

    for (eval_id, gid), group in matches.groupby(
        ["evaluation_gt_id", "global_id"],
        sort=True,
    ):
        g = group.sort_values("time").copy()
        if len(g) < 2:
            continue

        times = g["time"].to_numpy(float)
        nominal_dt = estimate_nominal_dt(
            times,
            fallback=np.nan,
        )

        if not np.isfinite(nominal_dt):
            continue

        max_gap = max(
            2.5 * nominal_dt,
            nominal_dt,
        )

        ex = (
            g["track_x"].to_numpy(float)
            - g["gt_x"].to_numpy(float)
        )
        ey = (
            g["track_y"].to_numpy(float)
            - g["gt_y"].to_numpy(float)
        )

        for i in range(1, len(g)):
            dt = float(times[i] - times[i - 1])

            if (
                dt <= EPS
                or dt > max_gap + EPS
            ):
                continue

            dex = float(ex[i] - ex[i - 1])
            dey = float(ey[i] - ey[i - 1])

            jitter = math.hypot(dex, dey) / dt

            rows.append({
                "time": float(times[i]),
                "evaluation_gt_id": str(eval_id),
                "global_id": str(gid),
                "dt_s": dt,
                "error_delta_x_m": dex,
                "error_delta_y_m": dey,
                "error_jitter_mps": float(jitter),
            })

    jitter_df = pd.DataFrame(rows)

    if jitter_df.empty:
        return {
            "mean_error_jitter_mps": np.nan,
            "rms_error_jitter_mps": np.nan,
            "p95_error_jitter_mps": np.nan,
        }, jitter_df

    values = jitter_df[
        "error_jitter_mps"
    ].to_numpy(float)

    return {
        "mean_error_jitter_mps":
            float(np.mean(values)),
        "rms_error_jitter_mps":
            float(np.sqrt(np.mean(values ** 2))),
        "p95_error_jitter_mps":
            float(np.percentile(values, 95)),
    }, jitter_df


def _frames_to_tables(
    frames: Sequence[dict],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame_rows = []
    gt_rows = []
    tracker_rows = []

    for frame in frames:
        frame_rows.append({
            "time": frame["time"],
            "track_snapshot_time":
                frame["track_snapshot_time"],
            "track_time_error_s":
                frame["track_time_error_s"],
            "num_fusion_evaluable_gt":
                len(frame["gt_ids"]),
            "num_scoped_global_tracks":
                len(frame["tracker_ids"]),
        })

        for i, gt_id in enumerate(frame["gt_ids"]):
            gt_rows.append({
                "time": frame["time"],
                "agent_id":
                    frame["physical_agent_ids"][i],
                "evaluation_gt_id": gt_id,
                "gt_x": float(frame["gt_xy"][i, 0]),
                "gt_y": float(frame["gt_xy"][i, 1]),
                "local_x":
                    float(frame["local_xy"][i, 0]),
                "local_y":
                    float(frame["local_xy"][i, 1]),
            })

        for j, gid in enumerate(frame["tracker_ids"]):
            tracker_rows.append({
                "time": frame["time"],
                "track_snapshot_time":
                    frame["track_snapshot_time"],
                "global_id": gid,
                "track_x":
                    float(frame["tracker_xy"][j, 0]),
                "track_y":
                    float(frame["tracker_xy"][j, 1]),
            })

    return (
        pd.DataFrame(frame_rows),
        pd.DataFrame(gt_rows),
        pd.DataFrame(tracker_rows),
    )


def _save_table_figure(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    column_labels: Optional[Sequence[str]] = None,
) -> None:
    """
    Save a compact publication-oriented table as PNG.
    The CSV remains the authoritative numeric output; this image is only a
    readable thesis/report visualization.
    """
    if df.empty:
        return

    display_df = df.copy()

    fig_height = max(2.2, 0.55 * (len(display_df) + 2))
    fig_width = max(7.5, 1.8 * len(display_df.columns))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, pad=16, fontsize=14, fontweight="bold")

    labels = (
        list(column_labels)
        if column_labels is not None
        else list(display_df.columns)
    )

    table = ax.table(
        cellText=display_df.astype(str).values,
        colLabels=labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.5)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def _save_mot_plots(
    output_dir: Path,
    hota_df: pd.DataFrame,
    clear_df: pd.DataFrame,
    matches: pd.DataFrame,
    jitter_df: pd.DataFrame,
) -> None:
    """
    Thesis-oriented visual outputs.

    Deliberately avoids:
      - CLEAR count bar charts,
      - position-error histograms,
      - per-trajectory position-error time curves.

    Those quantities are clearer as compact tables. The figures retained are:
      1) horizontal headline MOT score bars;
      2) selected HOTA-alpha table;
      3) CLEAR summary table;
      4) localization summary table;
      5) jitter summary table;
      6) jitter Median / Mean / P95 horizontal bars.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for old in output_dir.glob("*.png"):
        try:
            old.unlink()
        except OSError:
            pass

    # ------------------------------------------------------------------
    # 1. Headline MOT metrics: horizontal bars + percentages
    # ------------------------------------------------------------------
    avg = hota_df[
        hota_df["alpha"].astype(str) == "AVERAGE"
    ]

    if not avg.empty and not clear_df.empty:
        h = avg.iloc[0]
        c = clear_df.iloc[0]

        labels = np.array(
            ["HOTA", "DetA", "AssA", "MOTA"],
            dtype=object,
        )
        values = np.array(
            [
                float(h["HOTA"]),
                float(h["DetA"]),
                float(h["AssA"]),
                float(c["MOTA"]),
            ],
            dtype=float,
        )

        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        y = np.arange(len(labels))
        bars = ax.barh(y, values)

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlim(0.0, 1.08)
        ax.set_xlabel("Score [%]")
        ax.set_title(
            "Global fusion standard MOT performance",
            fontsize=15,
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.22)

        # Use percentage ticks for readability.
        ticks = np.arange(0.0, 1.01, 0.2)
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [f"{100*t:.0f}" for t in ticks]
        )

        for bar, value in zip(bars, values):
            ax.text(
                min(value + 0.015, 1.055),
                bar.get_y() + bar.get_height() / 2.0,
                f"{100.0 * value:.1f}%",
                va="center",
                ha="left",
                fontsize=11.5,
                fontweight="bold",
            )

        fig.tight_layout()
        fig.savefig(
            output_dir / "01_standard_mot_performance.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Selected HOTA thresholds as a TABLE, not a curve
    # ------------------------------------------------------------------
    numeric = hota_df[
        hota_df["alpha"].astype(str) != "AVERAGE"
    ].copy()

    if not numeric.empty:
        numeric["alpha_num"] = pd.to_numeric(
            numeric["alpha"]
        )

        requested = [0.50, 0.75, 0.85, 0.90, 0.95]
        selected_rows = []

        for alpha in requested:
            idx = (
                numeric["alpha_num"] - alpha
            ).abs().idxmin()
            row = numeric.loc[idx]
            selected_rows.append({
                "alpha": f"{float(row['alpha_num']):.2f}",
                "Max dist. [m]": (
                    f"{1.0 * (1.0 - float(row['alpha_num'])):.3f}"
                    if False else ""
                ),
                "HOTA [%]": f"{100.0 * float(row['HOTA']):.1f}",
                "DetA [%]": f"{100.0 * float(row['DetA']):.1f}",
                "AssA [%]": f"{100.0 * float(row['AssA']):.1f}",
            })

        selected_df = pd.DataFrame(selected_rows)

        # The metric-distance scale is not directly passed here. Leave the
        # distance column out of the rendered table to avoid hiding assumptions.
        selected_df = selected_df[
            ["alpha", "HOTA [%]", "DetA [%]", "AssA [%]"]
        ]

        _save_table_figure(
            selected_df,
            output_dir / "02_hota_selected_thresholds_table.png",
            "HOTA at representative similarity thresholds",
            ["α", "HOTA [%]", "DetA [%]", "AssA [%]"],
        )

    # ------------------------------------------------------------------
    # 3. CLEAR summary as table
    # ------------------------------------------------------------------
    if not clear_df.empty:
        c = clear_df.iloc[0]

        clear_table = pd.DataFrame([
            ["TP", int(c["CLR_TP"])],
            ["FN", int(c["CLR_FN"])],
            ["FP", int(c["CLR_FP"])],
            ["ID switches", int(c["IDSW"])],
            ["Fragmentations", int(c["Frag"])],
            ["Recall", f"{100.0 * float(c['CLR_Recall']):.1f}%"],
            ["Precision", f"{100.0 * float(c['CLR_Precision']):.1f}%"],
            ["MOTA", f"{100.0 * float(c['MOTA']):.1f}%"],
        ], columns=["Metric", "Result"])

        _save_table_figure(
            clear_table,
            output_dir / "03_clear_mot_summary_table.png",
            "CLEAR MOT summary",
        )

    # ------------------------------------------------------------------
    # 4. Localization metrics as table
    # ------------------------------------------------------------------
    if not matches.empty:
        e = matches["distance_m"].to_numpy(float)

        loc_table = pd.DataFrame([
            ["Mean error", f"{np.mean(e):.3f} m"],
            ["Median error", f"{np.median(e):.3f} m"],
            ["RMSE", f"{np.sqrt(np.mean(e**2)):.3f} m"],
            ["95th percentile", f"{np.percentile(e, 95):.3f} m"],
            ["Maximum error", f"{np.max(e):.3f} m"],
        ], columns=["Localization metric", "Result"])

        _save_table_figure(
            loc_table,
            output_dir / "04_localization_summary_table.png",
            "Global-track localization accuracy",
        )

    # ------------------------------------------------------------------
    # 5. Jitter summary as table + 6. intuitive jitter bar chart
    # ------------------------------------------------------------------
    if not jitter_df.empty:
        values = jitter_df[
            "error_jitter_mps"
        ].to_numpy(float)

        median_jitter = float(np.median(values))
        mean_jitter = float(np.mean(values))
        rms_jitter = float(np.sqrt(np.mean(values**2)))
        p95_jitter = float(np.percentile(values, 95))
        max_jitter = float(np.max(values))

        jitter_table = pd.DataFrame([
            ["Median jitter", f"{median_jitter:.3f} m/s"],
            ["Mean jitter", f"{mean_jitter:.3f} m/s"],
            ["RMS jitter", f"{rms_jitter:.3f} m/s"],
            ["95th percentile", f"{p95_jitter:.3f} m/s"],
            ["Maximum jitter", f"{max_jitter:.3f} m/s"],
        ], columns=["Jitter metric", "Result"])

        _save_table_figure(
            jitter_table,
            output_dir / "05_jitter_summary_table.png",
            "Global-track localization jitter",
        )

        # A simple horizontal comparison is more intuitive than a boxplot.
        # We deliberately show Median, Mean and P95:
        #   Median -> typical behaviour
        #   Mean   -> average behaviour, affected by spikes
        #   P95    -> upper bound for 95% of samples
        labels = np.array(
            ["Median", "Mean", "P95"],
            dtype=object,
        )
        bar_values = np.array(
            [median_jitter, mean_jitter, p95_jitter],
            dtype=float,
        )

        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        y = np.arange(len(labels))
        bars = ax.barh(y, bar_values)

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Error-vector jitter [m/s]")
        ax.set_title(
            "Global-track localization jitter summary",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.22)

        xmax = max(1.0, 1.18 * float(np.max(bar_values)))
        ax.set_xlim(0.0, xmax)

        for bar, value in zip(bars, bar_values):
            ax.text(
                min(value + 0.02 * xmax, 0.97 * xmax),
                bar.get_y() + bar.get_height() / 2.0,
                f"{value:.3f} m/s",
                va="center",
                ha="left",
                fontsize=11,
                fontweight="bold",
            )

        fig.tight_layout()
        fig.savefig(
            output_dir / "06_jitter_summary_bars.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

def evaluate_standard_mot_metrics(
    table: pd.DataFrame,
    tracks: pd.DataFrame,
    episodes: Sequence[VisibilityEpisode],
    track_time_tolerance_s: float,
    clear_match_distance_m: float,
    hota_distance_scale_m: float,
    output_scope_distance_m: float,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Run the standard MOT block on exactly the input-supported portion of the
    global fusion problem.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluable = _build_fusion_evaluable_samples(
        table,
        episodes,
    )

    if evaluable.empty:
        raise RuntimeError(
            "No fusion-evaluable local-input samples are available "
            "for standard MOT evaluation."
        )

    frames, scope_df = _build_component_mot_frames(
        evaluable,
        tracks,
        track_time_tolerance_s,
        output_scope_distance_m,
    )

    hota_df, hota_matches_df = (
        _evaluate_hota_trackeval_formulation(
            frames,
            hota_distance_scale_m,
        )
    )

    clear_df, matches_df, events_df = (
        _evaluate_clear_trackeval_formulation(
            frames,
            clear_match_distance_m,
        )
    )

    position = _compute_position_metrics(matches_df)
    jitter, jitter_df = _compute_error_jitter(
        matches_df
    )

    frame_df, gt_df, tracker_df = _frames_to_tables(
        frames
    )

    hota_avg = hota_df[
        hota_df["alpha"].astype(str) == "AVERAGE"
    ]
    hota_row = (
        hota_avg.iloc[0]
        if not hota_avg.empty
        else pd.Series(dtype=float)
    )

    clear_row = (
        clear_df.iloc[0]
        if not clear_df.empty
        else pd.Series(dtype=float)
    )

    overall = pd.DataFrame([{
        "evaluation_scope":
            "GT_visible_AND_local_input_available",
        "fusion_evaluable_physical_agents":
            int(evaluable["agent_id"].nunique()),
        "fusion_evaluable_evaluation_trajectories":
            int(
                evaluable[
                    "mot_evaluation_gt_id"
                ].nunique()
            ),
        "fusion_evaluable_agent_frames":
            int(len(evaluable)),
        "evaluation_frame_count":
            int(len(frames)),
        "HOTA":
            float(hota_row.get("HOTA", np.nan)),
        "DetA":
            float(hota_row.get("DetA", np.nan)),
        "AssA":
            float(hota_row.get("AssA", np.nan)),
        "DetRe":
            float(hota_row.get("DetRe", np.nan)),
        "DetPr":
            float(hota_row.get("DetPr", np.nan)),
        "AssRe":
            float(hota_row.get("AssRe", np.nan)),
        "AssPr":
            float(hota_row.get("AssPr", np.nan)),
        "LocA":
            float(hota_row.get("LocA", np.nan)),
        "MOTA":
            float(clear_row.get("MOTA", np.nan)),
        "MOTP_similarity":
            float(
                clear_row.get(
                    "MOTP_similarity", np.nan
                )
            ),
        "MOTP_mean_distance_m":
            float(
                clear_row.get(
                    "MOTP_mean_distance_m",
                    np.nan,
                )
            ),
        "TP":
            int(clear_row.get("CLR_TP", 0)),
        "FN":
            int(clear_row.get("CLR_FN", 0)),
        "FP":
            int(clear_row.get("CLR_FP", 0)),
        "IDSW":
            int(clear_row.get("IDSW", 0)),
        "Frag":
            int(clear_row.get("Frag", 0)),
        **position,
        **jitter,
    }])

    overall.to_csv(
        output_dir / "overall_mot_metrics.csv",
        index=False,
    )
    hota_df.to_csv(
        output_dir / "hota_metrics.csv",
        index=False,
    )

    # Compact representative HOTA thresholds for tables in the thesis.
    numeric_hota = hota_df[
        hota_df["alpha"].astype(str) != "AVERAGE"
    ].copy()
    if not numeric_hota.empty:
        numeric_hota["alpha_num"] = pd.to_numeric(
            numeric_hota["alpha"]
        )
        selected_hota_rows = []
        for alpha in (0.50, 0.75, 0.85, 0.90, 0.95):
            idx = (
                numeric_hota["alpha_num"] - alpha
            ).abs().idxmin()
            row = numeric_hota.loc[idx]
            selected_hota_rows.append({
                "alpha": float(row["alpha_num"]),
                "HOTA": float(row["HOTA"]),
                "DetA": float(row["DetA"]),
                "AssA": float(row["AssA"]),
                "DetRe": float(row["DetRe"]),
                "DetPr": float(row["DetPr"]),
                "AssRe": float(row["AssRe"]),
                "AssPr": float(row["AssPr"]),
                "LocA": float(row["LocA"]),
            })
        pd.DataFrame(selected_hota_rows).to_csv(
            output_dir / "hota_selected_thresholds.csv",
            index=False,
        )
    clear_df.to_csv(
        output_dir / "clear_mot_metrics.csv",
        index=False,
    )
    matches_df.to_csv(
        output_dir / "matched_pairs.csv",
        index=False,
    )
    events_df.to_csv(
        output_dir / "mot_events.csv",
        index=False,
    )
    jitter_df.to_csv(
        output_dir / "jitter_samples.csv",
        index=False,
    )
    frame_df.to_csv(
        output_dir / "evaluation_frames.csv",
        index=False,
    )
    gt_df.to_csv(
        output_dir / "fusion_evaluable_gt_samples.csv",
        index=False,
    )
    tracker_df.to_csv(
        output_dir / "scoped_global_track_samples.csv",
        index=False,
    )
    scope_df.to_csv(
        output_dir / "global_output_scope_diagnostics.csv",
        index=False,
    )
    hota_matches_df.to_csv(
        output_dir / "hota_matches_by_alpha.csv",
        index=False,
    )

    # Human-readable compact numeric tables.
    localization_summary = pd.DataFrame([{
        **position
    }])
    localization_summary.to_csv(
        output_dir / "localization_summary.csv",
        index=False,
    )

    jitter_summary = pd.DataFrame([{
        **jitter,
        "median_error_jitter_mps": (
            float(
                np.median(
                    jitter_df["error_jitter_mps"].to_numpy(float)
                )
            )
            if not jitter_df.empty
            else np.nan
        ),
        "max_error_jitter_mps": (
            float(
                np.max(
                    jitter_df["error_jitter_mps"].to_numpy(float)
                )
            )
            if not jitter_df.empty
            else np.nan
        ),
    }])
    jitter_summary.to_csv(
        output_dir / "jitter_summary.csv",
        index=False,
    )

    # One-row output designed for future multi-bag aggregation.
    run_summary = overall.copy()
    run_summary.insert(0, "run_id", "run_01")
    run_summary.to_csv(
        output_dir / "run_summary.csv",
        index=False,
    )

    config = {
        "methodology": (
            "Component-level standard MOT evaluation of the global "
            "fusion filter. Only geometrically visible GT targets for "
            "which a valid local observation was available are "
            "evaluated. Repeated dense GT rows referring to the same "
            "local input are collapsed to one agent-frame."
        ),
        "visibility_episode_policy": (
            "The exact visibility episodes already built by the "
            "fusion-specific validator are reused. A long absence that "
            "starts a new episode also starts a new MOT GT identity."
        ),
        "track_time_tolerance_s":
            track_time_tolerance_s,
        "clear_match_distance_m":
            clear_match_distance_m,
        "clear_similarity": (
            "max(0, 1 - d/(2*clear_match_distance)); "
            "TrackEval threshold 0.5"
        ),
        "hota_distance_scale_m":
            hota_distance_scale_m,
        "hota_similarity": (
            "max(0, 1 - d/hota_distance_scale)"
        ),
        "hota_alphas": [
            round(float(a), 2)
            for a in np.arange(0.05, 0.99, 0.05)
        ],
        "component_output_scope_distance_m":
            output_scope_distance_m,
        "component_output_scope_definition": (
            "A global output is included if it is within the scope "
            "distance of at least one current valid local fusion input. "
            "The scope is defined from local input positions, not GT."
        ),
    }

    (
        output_dir / "evaluation_config.json"
    ).write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    _save_mot_plots(
        output_dir,
        hota_df,
        clear_df,
        matches_df,
        jitter_df,
    )

    return overall


def parse_args():
    p=argparse.ArgumentParser(description='Validate global-ID stability of the fusion node without penalizing local-tracker misses.')
    p.add_argument('--robot-csv',type=Path,required=True); p.add_argument('--gt-csv',type=Path,required=True); p.add_argument('--tracks-csv',type=Path,required=True); p.add_argument('--local-csv',type=Path,required=True); p.add_argument('--output-dir',type=Path,default=Path('fusion_id_validation_results'))
    p.add_argument('--camera-range',type=float,default=4.0); p.add_argument('--camera-min-range',type=float,default=0.0); p.add_argument('--camera-hfov-deg',type=float,default=69.0); p.add_argument('--left-yaw-deg',type=float,default=58.0); p.add_argument('--center-yaw-deg',type=float,default=0.0); p.add_argument('--right-yaw-deg',type=float,default=-58.0); p.add_argument('--camera-offset-x',type=float,default=0.0); p.add_argument('--camera-offset-y',type=float,default=0.0)
    p.add_argument('--episode-merge-gap-s',type=float,default=6.0); p.add_argument('--structural-reentry-split-gap-s',type=float,default=4.0,help='Deprecated for scoring: structural FOV/range gaps are diagnostic only.'); p.add_argument('--ego-rotation-threshold-deg',type=float,default=15.0); p.add_argument('--reactivation-max-age-s',type=float,default=5.0,help='Fusion-node recovery horizon. A valid-local-support gap longer than this starts a new identity-continuity episode.'); p.add_argument('--association-max-distance',type=float,default=.75); p.add_argument('--continuity-max-distance',type=float,default=None); p.add_argument('--track-time-tolerance-s',type=float,default=.20); p.add_argument('--local-association-max-distance',type=float,default=.75); p.add_argument('--local-time-tolerance-s',type=float,default=.20); p.add_argument('--upstream-outlier-lookback-s',type=float,default=2.0,help='Look-back window before an observed ID switch for accepted invalid local measurements.'); p.add_argument('--debug-events-csv',type=Path,default=None,help='Optional fusion debug CSV. If omitted, debug_events_validation.csv next to --local-csv is used when present.'); p.add_argument('--visibility-continuity-factor',type=float,default=2.5); p.add_argument('--robot-yaw-unit',choices=('rad','deg'),default='rad')
    p.add_argument('--mot-output-dir',type=Path,default=Path('global_fusion_metrics'))
    p.add_argument('--clear-match-distance',type=float,default=.75)
    p.add_argument('--hota-distance-scale',type=float,default=1.50)
    p.add_argument('--mot-output-scope-distance',type=float,default=None)
    for prefix in ('robot','gt','track'):
        pass
    p.add_argument('--robot-time-col'); p.add_argument('--robot-x-col'); p.add_argument('--robot-y-col'); p.add_argument('--robot-yaw-col'); p.add_argument('--gt-time-col'); p.add_argument('--gt-id-col'); p.add_argument('--gt-x-col'); p.add_argument('--gt-y-col'); p.add_argument('--track-time-col'); p.add_argument('--track-id-col'); p.add_argument('--track-x-col'); p.add_argument('--track-y-col'); p.add_argument('--track-agent-col'); p.add_argument('--local-time-col'); p.add_argument('--local-x-col'); p.add_argument('--local-y-col'); p.add_argument('--local-camera-col'); p.add_argument('--local-id-col')
    return p.parse_args()


def main():
    args=parse_args()
    for p in (args.robot_csv,args.gt_csv,args.tracks_csv,args.local_csv):
        if not p.exists(): raise FileNotFoundError(p)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    robot,gt,tracks=standardize_inputs(pd.read_csv(args.robot_csv),pd.read_csv(args.gt_csv),pd.read_csv(args.tracks_csv),args)
    local=standardize_local_detections(pd.read_csv(args.local_csv),args)

    debug_path=args.debug_events_csv
    if debug_path is None:
        auto_debug=args.local_csv.parent/'debug_events_validation.csv'
        debug_path=auto_debug if auto_debug.exists() else None
    debug_events=load_debug_events_optional(debug_path)
    if debug_path is not None:
        print(f"Fusion causal diagnostics source: {debug_path}")
    else:
        print("Fusion causal diagnostics source: unavailable (switches will remain unresolved)")
    cameras=[CameraModel('center',math.radians(args.center_yaw_deg),math.radians(args.camera_hfov_deg),args.camera_range,args.camera_min_range,args.camera_offset_x,args.camera_offset_y),CameraModel('left',math.radians(args.left_yaw_deg),math.radians(args.camera_hfov_deg),args.camera_range,args.camera_min_range,args.camera_offset_x,args.camera_offset_y),CameraModel('right',math.radians(args.right_yaw_deg),math.radians(args.camera_hfov_deg),args.camera_range,args.camera_min_range,args.camera_offset_x,args.camera_offset_y)]
    table=annotate_visibility(gt,robot,cameras)
    table=associate_visible_gt_to_local_detections(table,local,args.local_time_tolerance_s,args.local_association_max_distance)
    cont=args.continuity_max_distance if args.continuity_max_distance is not None else args.association_max_distance
    table=associate_evaluable_gt_to_global_tracks(table,tracks,args.track_time_tolerance_s,args.association_max_distance,cont)
    geometric_episodes=[]; metrics=[]; breakdown=[]; switches=[]; interruptions=[]; reentries=[]; nominal={}; visibility_gap_diagnostics=[]
    for aid,rows in table.groupby('agent_id',sort=True):
        dt=estimate_nominal_dt(rows['time'].to_numpy(float)); nominal[str(aid)]=dt
        seg=build_visibility_segments(rows.sort_values('time'),dt,args.visibility_continuity_factor)
        # Keep FOV/range/robot-rotation information as diagnostics, but do NOT
        # automatically excuse an identity change because of those causes.
        eps,gap_diag=merge_segments_into_episodes_scope_aware(
            str(aid),seg,table,cameras,args.episode_merge_gap_s,
            float('inf'),args.ego_rotation_threshold_deg
        )
        visibility_gap_diagnostics.extend(gap_diag); geometric_episodes.extend(eps)

    # Identity-continuity scope is tied to the REAL recovery capability of the
    # implemented fusion node, not to a generic geometric visibility threshold.
    # If no valid local input for the same GT person is available for longer
    # than reactivation_max_age_s, the old Global ID is outside the designed
    # recovery horizon. The later support block therefore starts a new
    # fusion-evaluable identity-continuity episode.
    episodes, local_support_gap_diagnostics = split_episodes_on_local_support_gaps(
        geometric_episodes, table, args.reactivation_max_age_s
    )

    for ep in episodes:
        dt=nominal[str(ep.agent_id)]
        m,b,s,i=compute_episode_metrics(table,ep,dt); metrics.append(m); breakdown.extend(b); switches.extend(s); interruptions.extend(i); reentries.extend(compute_reentry_events_fusion(table,ep))

    epm=pd.DataFrame(metrics); bdf=pd.DataFrame(breakdown); sdf=pd.DataFrame(switches); idf=pd.DataFrame(interruptions); rdf=pd.DataFrame(reentries); gapdf=pd.DataFrame(visibility_gap_diagnostics); localgapdf=pd.DataFrame(local_support_gap_diagnostics)
    segdf=build_segment_table_new(episodes,table,nominal); epdf=build_episode_table_new(episodes); adf,odf=aggregate_metrics(epm,rdf)

    # Causal attribution of the switches that remain WITHIN the actual recovery
    # horizon. An accepted local measurement that is farther than the local-GT
    # validity threshold from every GT target is an upstream localization
    # outlier, not a clean-input fusion recovery failure.
    sdf=classify_switch_root_causes(
        sdf,
        debug_events,
        gt,
        args.local_association_max_distance,
        args.local_time_tolerance_s,
        args.upstream_outlier_lookback_s,
    )
    adf,odf=add_causal_switch_aggregates(adf,odf,sdf)

    outputs={'per_sample_fusion_evaluation.csv':table,'visibility_segments.csv':segdf,'visibility_episodes.csv':epdf,'episode_fusion_id_metrics.csv':epm,'episode_global_id_breakdown.csv':bdf,'global_id_switch_events.csv':sdf,'global_association_interruptions.csv':idf,'reentry_global_id_events.csv':rdf,'visibility_gap_diagnostics.csv':gapdf,'local_support_gap_diagnostics.csv':localgapdf,'agent_fusion_id_metrics.csv':adf,'overall_fusion_id_metrics.csv':odf}
    for name,df in outputs.items(): df.to_csv(args.output_dir/name,index=False)
    cfg={'methodology':'Recovery-aware fusion-node Global-ID validation: score identity continuity only while valid local input remains within the fusion reactivation horizon; retain FOV/range/rotation as diagnostics; causally label switches preceded by accepted upstream localization outliers.','camera_range_m':args.camera_range,'camera_hfov_deg':args.camera_hfov_deg,'camera_relative_yaw_deg':{'left':args.left_yaw_deg,'center':args.center_yaw_deg,'right':args.right_yaw_deg},'episode_merge_gap_s':args.episode_merge_gap_s,'reactivation_max_age_s':args.reactivation_max_age_s,'structural_reentry_split_gap_s_legacy':args.structural_reentry_split_gap_s,'ego_rotation_threshold_deg':args.ego_rotation_threshold_deg,'structural_gap_policy':'Out-of-range and ego-rotation-associated FOV losses are diagnostic only and do not automatically excuse a new identity. Identity continuity is reset only when consecutive valid local-support blocks are separated by more than reactivation_max_age_s.','causal_switch_policy':'Observed switches preceded by an accepted local measurement that is farther than local_association_max_distance_m from every GT target are labelled upstream_localization_induced and excluded from the fusion-attributable switch count, while remaining visible as observed end-to-end discontinuities.','debug_events_csv':str(debug_path) if debug_path is not None else None,'upstream_outlier_lookback_s':args.upstream_outlier_lookback_s,'local_association_max_distance_m':args.local_association_max_distance,'local_time_tolerance_s':args.local_time_tolerance_s,'global_association_max_distance_m':args.association_max_distance,'global_track_time_tolerance_s':args.track_time_tolerance_s,'continuity_max_distance_m':cont}
    (args.output_dir/'evaluation_config.json').write_text(json.dumps(cfg,indent=2))

    # --------------------------------------------------------------
    # Standard MOT evaluation on the SAME fusion-evaluable population
    # --------------------------------------------------------------
    # The scope radius is derived conservatively from the maximum allowed
    # GT-local input error plus the CLEAR GT-global matching distance unless
    # explicitly overridden.
    mot_scope_distance = (
        args.mot_output_scope_distance
        if args.mot_output_scope_distance is not None
        else (
            args.local_association_max_distance
            + args.clear_match_distance
        )
    )

    mot_overall = evaluate_standard_mot_metrics(
        table=table,
        tracks=tracks,
        episodes=episodes,
        track_time_tolerance_s=args.track_time_tolerance_s,
        clear_match_distance_m=args.clear_match_distance,
        hota_distance_scale_m=args.hota_distance_scale,
        output_scope_distance_m=mot_scope_distance,
        output_dir=args.mot_output_dir,
    )

    plot_results(table,episodes,epm,bdf,adf,rdf,args.output_dir)
    print("\nFusion Global-ID validation complete.")
    print(f"Results directory: {args.output_dir.resolve()}")

    if not odf.empty:
        summary = odf.iloc[0]
        print("\nEvaluation scope:")
        print(
            f"  Geometrically visible GT agents: "
            f"{int(summary['geometrically_visible_agent_count'])}"
        )
        print(
            f"  Geometric visibility episodes: "
            f"{int(summary['geometric_visibility_episode_count'])}"
        )
        print(
            f"  Fusion-evaluable GT agents: "
            f"{int(summary['fusion_evaluable_agent_count'])}"
        )
        print(
            f"  Fusion-evaluable episodes: "
            f"{int(summary['fusion_evaluable_episode_count'])}"
        )
        print(
            f"  Agents with a Global ID: "
            f"{int(summary['globally_assigned_agent_count'])}"
        )
        print(
            f"  Episodes with a Global ID: "
            f"{int(summary['globally_assigned_episode_count'])}"
        )

        print("\nOverall fusion-ID metrics:")
        for key, label in (
            ("global_association_coverage", "Association coverage"),
            ("global_id_purity_when_assigned", "ID purity when assigned"),
            ("global_id_stability_evaluable", "Global-ID stability"),
            ("same_global_id_after_reentry_rate", "Re-entry preservation"),
        ):
            value = summary[key]
            print(
                f"  {label}: {100.0 * value:.2f}%"
                if pd.notna(value)
                else f"  {label}: N/A"
            )
        observed_switches=int(summary['observed_global_id_switches']) if 'observed_global_id_switches' in summary.index else int(summary['total_global_id_switches'])
        fusion_switches=int(summary['fusion_attributable_global_id_switches']) if 'fusion_attributable_global_id_switches' in summary.index else observed_switches
        upstream_switches=int(summary['upstream_induced_identity_discontinuities']) if 'upstream_induced_identity_discontinuities' in summary.index else 0
        print(f"  Observed Global-ID switches: {observed_switches}")
        print(f"  Fusion-attributable Global-ID switches: {fusion_switches}")
        print(f"  Upstream-induced identity discontinuities: {upstream_switches}")
        print(
            f"  Global association interruptions: "
            f"{int(summary['total_global_association_interruptions'])}"
        )

    if not mot_overall.empty:
        m = mot_overall.iloc[0]
        print("\nStandard MOT metrics for the global fusion filter:")
        print(
            f"  Fusion-evaluable agent-frames: "
            f"{int(m['fusion_evaluable_agent_frames'])}"
        )
        print(f"  HOTA: {float(m['HOTA']):.6f}")
        print(f"  DetA: {float(m['DetA']):.6f}")
        print(f"  AssA: {float(m['AssA']):.6f}")
        print(f"  MOTA: {float(m['MOTA']):.6f}")
        print(
            f"  TP / FN / FP: "
            f"{int(m['TP'])} / {int(m['FN'])} / {int(m['FP'])}"
        )
        print(
            f"  IDSW / Frag: "
            f"{int(m['IDSW'])} / {int(m['Frag'])}"
        )
        print(
            f"  Mean position error: "
            f"{float(m['mean_position_error_m']):.4f} m"
        )
        print(
            f"  RMSE position error: "
            f"{float(m['rmse_position_error_m']):.4f} m"
        )
        print(
            f"  P95 position error: "
            f"{float(m['p95_position_error_m']):.4f} m"
        )
        print(
            f"  RMS error jitter: "
            f"{float(m['rms_error_jitter_mps']):.4f} m/s"
            if pd.notna(m['rms_error_jitter_mps'])
            else "  RMS error jitter: N/A"
        )
        print(
            f"  MOT results directory: "
            f"{args.mot_output_dir.resolve()}"
        )


if __name__=='__main__':
    main()
