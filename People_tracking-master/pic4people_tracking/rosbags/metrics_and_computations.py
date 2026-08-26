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

Important rule
--------------
A gap > episode_merge_gap_s starts a NEW visibility episode. Therefore, a new
global ID after a long absence is not counted as an ID switch.

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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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

            ax.axvline(
                6.0,
                linestyle="--",
                linewidth=1.3,
                color="black",
                label="Episode threshold = 6 s",
            )
            ax.set_yticks(y)
            ax.set_yticklabels(valid["event_label"])
            ax.set_xlim(
                0,
                max(6.8, float(valid["invisible_gap_s"].max()) + 1.3),
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

            # Put the threshold legend outside the complete plotting area,
            # below the right-hand information panel.
            handles, legend_labels = ax.get_legend_handles_labels()
            fig.legend(
                handles,
                legend_labels,
                title="Legend",
                loc="center left",
                bbox_to_anchor=(0.89, 0.50),
                frameon=True,
            )
            fig.suptitle(
                "Global-ID preservation after temporary invisibility",
                y=0.985,
                fontsize=14,
            )
            fig.subplots_adjust(
                right=0.88,
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


def parse_args():
    p=argparse.ArgumentParser(description='Validate global-ID stability of the fusion node without penalizing local-tracker misses.')
    p.add_argument('--robot-csv',type=Path,required=True); p.add_argument('--gt-csv',type=Path,required=True); p.add_argument('--tracks-csv',type=Path,required=True); p.add_argument('--local-csv',type=Path,required=True); p.add_argument('--output-dir',type=Path,default=Path('fusion_id_validation_results'))
    p.add_argument('--camera-range',type=float,default=4.0); p.add_argument('--camera-min-range',type=float,default=0.0); p.add_argument('--camera-hfov-deg',type=float,default=69.0); p.add_argument('--left-yaw-deg',type=float,default=58.0); p.add_argument('--center-yaw-deg',type=float,default=0.0); p.add_argument('--right-yaw-deg',type=float,default=-58.0); p.add_argument('--camera-offset-x',type=float,default=0.0); p.add_argument('--camera-offset-y',type=float,default=0.0)
    p.add_argument('--episode-merge-gap-s',type=float,default=6.0); p.add_argument('--association-max-distance',type=float,default=.75); p.add_argument('--continuity-max-distance',type=float,default=None); p.add_argument('--track-time-tolerance-s',type=float,default=.20); p.add_argument('--local-association-max-distance',type=float,default=.75); p.add_argument('--local-time-tolerance-s',type=float,default=.20); p.add_argument('--visibility-continuity-factor',type=float,default=2.5); p.add_argument('--robot-yaw-unit',choices=('rad','deg'),default='rad')
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
    cameras=[CameraModel('center',math.radians(args.center_yaw_deg),math.radians(args.camera_hfov_deg),args.camera_range,args.camera_min_range,args.camera_offset_x,args.camera_offset_y),CameraModel('left',math.radians(args.left_yaw_deg),math.radians(args.camera_hfov_deg),args.camera_range,args.camera_min_range,args.camera_offset_x,args.camera_offset_y),CameraModel('right',math.radians(args.right_yaw_deg),math.radians(args.camera_hfov_deg),args.camera_range,args.camera_min_range,args.camera_offset_x,args.camera_offset_y)]
    table=annotate_visibility(gt,robot,cameras)
    table=associate_visible_gt_to_local_detections(table,local,args.local_time_tolerance_s,args.local_association_max_distance)
    cont=args.continuity_max_distance if args.continuity_max_distance is not None else args.association_max_distance
    table=associate_evaluable_gt_to_global_tracks(table,tracks,args.track_time_tolerance_s,args.association_max_distance,cont)
    episodes=[]; metrics=[]; breakdown=[]; switches=[]; interruptions=[]; reentries=[]; nominal={}
    for aid,rows in table.groupby('agent_id',sort=True):
        dt=estimate_nominal_dt(rows['time'].to_numpy(float)); nominal[str(aid)]=dt
        seg=build_visibility_segments(rows.sort_values('time'),dt,args.visibility_continuity_factor); eps=merge_segments_into_episodes(str(aid),seg,args.episode_merge_gap_s); episodes.extend(eps)
        for ep in eps:
            m,b,s,i=compute_episode_metrics(table,ep,dt); metrics.append(m); breakdown.extend(b); switches.extend(s); interruptions.extend(i); reentries.extend(compute_reentry_events_fusion(table,ep))
    epm=pd.DataFrame(metrics); bdf=pd.DataFrame(breakdown); sdf=pd.DataFrame(switches); idf=pd.DataFrame(interruptions); rdf=pd.DataFrame(reentries)
    segdf=build_segment_table_new(episodes,table,nominal); epdf=build_episode_table_new(episodes); adf,odf=aggregate_metrics(epm,rdf)
    outputs={'per_sample_fusion_evaluation.csv':table,'visibility_segments.csv':segdf,'visibility_episodes.csv':epdf,'episode_fusion_id_metrics.csv':epm,'episode_global_id_breakdown.csv':bdf,'global_id_switch_events.csv':sdf,'global_association_interruptions.csv':idf,'reentry_global_id_events.csv':rdf,'agent_fusion_id_metrics.csv':adf,'overall_fusion_id_metrics.csv':odf}
    for name,df in outputs.items(): df.to_csv(args.output_dir/name,index=False)
    cfg={'methodology':'Fusion-node Global-ID stability evaluated only when a local observation is available','camera_range_m':args.camera_range,'camera_hfov_deg':args.camera_hfov_deg,'camera_relative_yaw_deg':{'left':args.left_yaw_deg,'center':args.center_yaw_deg,'right':args.right_yaw_deg},'episode_merge_gap_s':args.episode_merge_gap_s,'local_association_max_distance_m':args.local_association_max_distance,'local_time_tolerance_s':args.local_time_tolerance_s,'global_association_max_distance_m':args.association_max_distance,'global_track_time_tolerance_s':args.track_time_tolerance_s,'continuity_max_distance_m':cont}
    (args.output_dir/'evaluation_config.json').write_text(json.dumps(cfg,indent=2))
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
        print(
            f"  Global-ID switches: "
            f"{int(summary['total_global_id_switches'])}"
        )
        print(
            f"  Global association interruptions: "
            f"{int(summary['total_global_association_interruptions'])}"
        )


if __name__=='__main__':
    main()
