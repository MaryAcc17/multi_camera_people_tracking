#!/usr/bin/env python3
"""
fusion_identity_validation.py

Valutazione specifica del fusion node multi-camera, senza HOTA/MOTA/MOTP.

La Ground Truth serve per etichettare gli input locali. Il fusion node viene
valutato rispetto alle track locali effettivamente disponibili.

Metriche:
- Fusion IDF1, ID Precision, ID Recall
- Fusion coverage
- ID switches e fragmentations nei frame local-supported
- MT / PT / ML per visibility episode
- Re-entry preservation per gap <= episode_merge_gap_s
- Multi-camera consolidation rate
- Duplicate Global-ID rate
- Diagnostic decomposition of unresolved/unmatched global outputs
- Lifecycle persistence events
- Spatial mismatch local-supported, separato dagli errori d'identità

Le Global Track vengono associate alla mediana delle posizioni locali dello
stesso agente, non direttamente alla GT. L'errore Global-GT viene registrato
separatamente.

Esempio:
python3 fusion_identity_validation.py \
  --robot-csv validation_csv/robot_pose_validation.csv \
  --gt-csv validation_csv/ground_truth_people.csv \
  --global-csv validation_csv/global_tracks_validation.csv \
  --local-csv validation_csv/local_detections_validation.csv \
  --output-dir fusion_identity_results \
  --camera-range 4.0 --camera-hfov-deg 69.0 \
  --left-yaw-deg 58.0 --center-yaw-deg 0.0 --right-yaw-deg -58.0 \
  --episode-merge-gap-s 6.0 --publish-max-age-s 1.8 \
  --local-gt-max-distance 0.75 --global-local-max-distance 0.75 \
  --spatial-error-threshold 0.75 --local-time-tolerance-s 0.20
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

EPS = 1e-12
SCRIPT_VERSION = "FUSION_IDENTITY_VALIDATION_V4_FRAGMENTATION_AWARE"


@dataclass(frozen=True)
class Segment:
    start: float
    end: float


@dataclass(frozen=True)
class Episode:
    agent_id: str
    index: int
    start: float
    end: float

    @property
    def identity(self) -> str:
        return f"{self.agent_id}-E{self.index}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fusion-specific identity validation without HOTA/MOTA."
    )
    p.add_argument("--robot-csv", required=True)
    p.add_argument("--gt-csv", required=True)
    p.add_argument("--global-csv", required=True)
    p.add_argument("--local-csv", required=True)
    p.add_argument("--output-dir", default="fusion_identity_results")

    p.add_argument("--camera-range", type=float, default=4.0)
    p.add_argument("--camera-min-range", type=float, default=0.0)
    p.add_argument("--camera-hfov-deg", type=float, default=69.0)
    p.add_argument("--left-yaw-deg", type=float, default=58.0)
    p.add_argument("--center-yaw-deg", type=float, default=0.0)
    p.add_argument("--right-yaw-deg", type=float, default=-58.0)
    p.add_argument("--camera-offset-x", type=float, default=0.0)
    p.add_argument("--camera-offset-y", type=float, default=0.0)

    p.add_argument("--episode-merge-gap-s", type=float, default=6.0)
    p.add_argument("--publish-max-age-s", type=float, default=1.8)
    p.add_argument("--visibility-sample-gap-factor", type=float, default=2.5)
    p.add_argument("--local-gt-max-distance", type=float, default=0.75)
    p.add_argument("--global-local-max-distance", type=float, default=0.75)
    p.add_argument("--duplicate-global-distance", type=float, default=0.75)
    p.add_argument("--spatial-error-threshold", type=float, default=0.75)
    p.add_argument("--local-time-tolerance-s", type=float, default=0.20)
    p.add_argument(
        "--global-time-tolerance-s",
        type=float,
        default=0.20,
        help="Maximum time offset between a local-supported frame and the nearest Global Track frame.",
    )
    p.add_argument("--assignment-time-bin-s", type=float, default=0.03)

    # Episode validity and re-entry reconstruction.
    p.add_argument(
        "--min-evaluable-frames",
        type=int,
        default=3,
        help=(
            "Episodes with fewer local-supported frames are reported as "
            "INSUFFICIENT_DATA and excluded from MT/PT/ML percentages."
        ),
    )
    p.add_argument(
        "--local-support-gap-factor",
        type=float,
        default=2.5,
        help=(
            "A new local-support segment starts when the interval between two "
            "successive supported local frames exceeds this factor times the "
            "nominal local-frame period."
        ),
    )
    p.add_argument(
        "--local-support-min-gap-s",
        type=float,
        default=0.35,
        help="Minimum interval used to split local-supported segments.",
    )
    p.add_argument(
        "--reentry-min-gap-s",
        type=float,
        default=0.55,
        help="Minimum local-input absence duration considered a real re-entry gap.",
    )
    p.add_argument(
        "--reentry-min-segment-frames",
        type=int,
        default=2,
        help="Minimum number of supported frames required before and after a re-entry gap.",
    )
    p.add_argument(
        "--reentry-id-time-tolerance-s",
        type=float,
        default=0.25,
        help="Tolerance for reading the Global ID at the two sides of a local-input gap.",
    )

    # Diagnostic classification of formerly generic unsupported outputs.
    p.add_argument(
        "--unsupported-local-time-tolerance-s",
        type=float,
        default=0.35,
        help="Time tolerance used to search raw local support for an unmatched Global Track.",
    )
    p.add_argument(
        "--unsupported-local-distance",
        type=float,
        default=0.90,
        help="Distance gate for raw-local support diagnostics.",
    )
    p.add_argument(
        "--extended-lifecycle-age-s",
        type=float,
        default=6.0,
        help=(
            "Diagnostic age up to which an unmatched Global ID previously "
            "associated to an episode is labelled extended lifecycle candidate. "
            "It is not penalized as IDFP."
        ),
    )

    p.add_argument("--robot-yaw-unit", choices=["rad", "deg"], default="rad")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def clean_id(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def natural_key(value: object) -> Tuple[str, int, str]:
    text = clean_id(value)
    i = len(text) - 1
    while i >= 0 and text[i].isdigit():
        i -= 1
    return (
        (text[: i + 1], int(text[i + 1 :]), text)
        if i < len(text) - 1
        else (text, -1, text)
    )


def norm_angle(value):
    array = np.asarray(value, dtype=float)
    return (array + np.pi) % (2.0 * np.pi) - np.pi


def require(df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}; available={list(df.columns)}")


def nearest_index(times: np.ndarray, query: float) -> Optional[int]:
    if len(times) == 0:
        return None
    i = int(np.searchsorted(times, query))
    candidates = ([i] if i < len(times) else []) + ([i - 1] if i > 0 else [])
    return min(candidates, key=lambda j: abs(float(times[j]) - query)) if candidates else None


def nominal_dt(times: Sequence[float], fallback: float = 0.1) -> float:
    values = np.sort(np.asarray(times, dtype=float))
    differences = np.diff(values)
    differences = differences[np.isfinite(differences) & (differences > EPS)]
    return float(np.median(differences)) if len(differences) else fallback


def group_frames(df: pd.DataFrame, bin_s: float) -> List[pd.DataFrame]:
    if df.empty:
        return []
    ordered = df.sort_values("time").reset_index(drop=True).copy()
    ids = np.zeros(len(ordered), dtype=int)
    frame_id = 0
    previous = float(ordered.iloc[0]["time"])
    for i in range(1, len(ordered)):
        current = float(ordered.iloc[i]["time"])
        if current - previous > bin_s:
            frame_id += 1
        ids[i] = frame_id
        previous = current
    ordered["_frame"] = ids
    return [g.drop(columns="_frame").copy() for _, g in ordered.groupby("_frame", sort=True)]


def load_inputs(args):
    robot = pd.read_csv(args.robot_csv)
    gt = pd.read_csv(args.gt_csv)
    global_tracks = pd.read_csv(args.global_csv)
    local = pd.read_csv(args.local_csv)

    require(robot, ["time", "x", "y", "yaw"], "robot")
    require(gt, ["time", "agent_id", "x", "y"], "GT")
    require(global_tracks, ["time", "global_id", "x", "y"], "global")
    require(local, ["time", "camera", "local_id", "x", "y"], "local")

    for df in [robot, gt, global_tracks, local]:
        for c in ["time", "x", "y"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    robot["yaw"] = pd.to_numeric(robot["yaw"], errors="coerce")

    robot.dropna(subset=["time", "x", "y", "yaw"], inplace=True)
    gt.dropna(subset=["time", "x", "y"], inplace=True)
    global_tracks.dropna(subset=["time", "x", "y"], inplace=True)
    local.dropna(subset=["time", "x", "y"], inplace=True)

    if args.robot_yaw_unit == "deg":
        robot["yaw"] = np.deg2rad(robot["yaw"].to_numpy(float))

    gt["agent_id"] = gt["agent_id"].map(clean_id)
    global_tracks["global_id"] = global_tracks["global_id"].map(clean_id)
    local["camera"] = local["camera"].map(clean_id)
    local["local_id"] = local["local_id"].map(clean_id)

    gt = gt[gt["agent_id"] != ""].copy()
    global_tracks = global_tracks[global_tracks["global_id"] != ""].copy()
    local = local[(local["camera"] != "") & (local["local_id"] != "")].copy()

    for df in [robot, gt, global_tracks, local]:
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)

    start = max(float(robot.time.min()), float(gt.time.min()))
    end = min(float(robot.time.max()), float(gt.time.max()))
    original = len(gt)
    gt = gt[(gt.time >= start) & (gt.time <= end)].copy()
    if original - len(gt):
        print(f"Warning: discarded {original-len(gt)} GT rows outside [{start:.3f}, {end:.3f}] s.")
    return robot, gt, global_tracks, local


def visibility_table(robot: pd.DataFrame, gt: pd.DataFrame, args) -> pd.DataFrame:
    robot = robot.sort_values("time").drop_duplicates("time", keep="last")
    rt = robot.time.to_numpy(float)
    q = gt.time.to_numpy(float)
    rx = np.interp(q, rt, robot.x.to_numpy(float))
    ry = np.interp(q, rt, robot.y.to_numpy(float))
    yaw = norm_angle(np.interp(q, rt, np.unwrap(robot.yaw.to_numpy(float))))

    ox = rx + np.cos(yaw) * args.camera_offset_x - np.sin(yaw) * args.camera_offset_y
    oy = ry + np.sin(yaw) * args.camera_offset_x + np.cos(yaw) * args.camera_offset_y
    dx = gt.x.to_numpy(float) - ox
    dy = gt.y.to_numpy(float) - oy
    distance = np.hypot(dx, dy)
    bearing = np.arctan2(dy, dx)
    half = math.radians(args.camera_hfov_deg) / 2.0

    table = gt.copy()
    visible_any = np.zeros(len(table), dtype=bool)
    for name, degrees in {
        "left": args.left_yaw_deg,
        "center": args.center_yaw_deg,
        "right": args.right_yaw_deg,
    }.items():
        angular = np.abs(norm_angle(bearing - (yaw + math.radians(degrees))))
        visible = (
            (distance >= args.camera_min_range)
            & (distance <= args.camera_range)
            & (angular <= half + EPS)
        )
        table[f"visible_{name}"] = visible
        visible_any |= visible
    table["geometrically_visible"] = visible_any
    return table


def make_episodes(table: pd.DataFrame, args) -> List[Episode]:
    episodes: List[Episode] = []
    for agent_id, group in table.groupby("agent_id", sort=False):
        visible = group[group.geometrically_visible].sort_values("time")
        if visible.empty:
            continue
        times = visible.time.to_numpy(float)
        limit = max(
            nominal_dt(group.time.to_numpy(float)) * args.visibility_sample_gap_factor,
            nominal_dt(group.time.to_numpy(float)) + 1e-6,
        )
        segments: List[Segment] = []
        start = previous = float(times[0])
        for current in times[1:]:
            current = float(current)
            if current - previous > limit:
                segments.append(Segment(start, previous))
                start = current
            previous = current
        segments.append(Segment(start, previous))

        merged: List[Segment] = []
        current = segments[0]
        for segment in segments[1:]:
            if segment.start - current.end <= args.episode_merge_gap_s + EPS:
                current = Segment(current.start, segment.end)
            else:
                merged.append(current)
                current = segment
        merged.append(current)

        for index, segment in enumerate(merged, 1):
            episodes.append(Episode(str(agent_id), index, segment.start, segment.end))
    return episodes


def episode_at(lookup: Dict[str, List[Episode]], agent_id: str, time_value: float) -> Optional[Episode]:
    for episode in lookup.get(agent_id, []):
        if episode.start - EPS <= time_value <= episode.end + EPS:
            return episode
    return None


def build_tracks(df: pd.DataFrame, identity: str) -> Dict[str, pd.DataFrame]:
    return {
        str(key): group.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
        for key, group in df.groupby(identity, sort=False)
    }


def interpolate_position(track: pd.DataFrame, time_value: float, tolerance: float) -> Optional[Tuple[float, float]]:
    times = track.time.to_numpy(float)
    index = nearest_index(times, time_value)
    if index is None or abs(float(times[index]) - time_value) > tolerance:
        return None
    if len(times) >= 2 and times[0] <= time_value <= times[-1]:
        return (
            float(np.interp(time_value, times, track.x.to_numpy(float))),
            float(np.interp(time_value, times, track.y.to_numpy(float))),
        )
    return float(track.iloc[index].x), float(track.iloc[index].y)


def visible_gt_now(tracks: Dict[str, pd.DataFrame], time_value: float, tolerance: float) -> pd.DataFrame:
    rows = []
    for agent_id, track in tracks.items():
        times = track.time.to_numpy(float)
        index = nearest_index(times, time_value)
        if index is None or abs(float(times[index]) - time_value) > tolerance:
            continue
        if not bool(track.iloc[index].geometrically_visible):
            continue
        position = interpolate_position(track[["time", "x", "y"]], time_value, tolerance)
        if position:
            rows.append({"agent_id": agent_id, "gt_x": position[0], "gt_y": position[1]})
    return pd.DataFrame(rows)


def locals_near(
    local_frames: Sequence[pd.DataFrame],
    frame_times: np.ndarray,
    time_value: float,
    tolerance: float,
) -> pd.DataFrame:
    candidates = np.where(np.abs(frame_times - time_value) <= tolerance + EPS)[0]
    if len(candidates) == 0:
        return pd.DataFrame(columns=["time", "camera", "local_id", "x", "y"])
    by_camera = defaultdict(list)
    for i in candidates:
        frame = local_frames[int(i)]
        for camera, rows in frame.groupby("camera", sort=False):
            by_camera[str(camera)].append((abs(float(frame_times[int(i)]) - time_value), rows.copy()))
    selected = [min(items, key=lambda item: item[0])[1] for _, items in sorted(by_camera.items())]
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()



def build_local_support_timeline(
    local_frames: Sequence[pd.DataFrame],
    visibility_tracks: Dict[str, pd.DataFrame],
    episode_lookup: Dict[str, List[Episode]],
    args,
) -> pd.DataFrame:
    """
    Build the actual local-input support timeline from every local frame,
    independently of the timestamps at which Global Tracks are published.
    """
    rows = []
    for frame in local_frames:
        if frame.empty:
            continue
        time_value = float(frame.time.median())
        visible_gt = visible_gt_now(
            visibility_tracks,
            time_value,
            args.local_time_tolerance_s,
        )
        local_assoc = associate_local(
            frame,
            visible_gt,
            args.local_gt_max_distance,
        )
        supported = supported_agents(
            local_assoc,
            episode_lookup,
            time_value,
        )
        for row in supported.itertuples(index=False):
            rows.append({
                "time": time_value,
                **row._asdict(),
            })
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["episode_identity", "time"])
        .drop_duplicates(["episode_identity", "time"], keep="last")
        .reset_index(drop=True)
    )


def raw_local_support_for_global(
    local_frames: Sequence[pd.DataFrame],
    frame_times: np.ndarray,
    time_value: float,
    global_x: float,
    global_y: float,
    args,
) -> dict:
    """
    Search the raw local CSV around an unmatched Global Track timestamp.
    This does not require the local observation to have been associated to GT.
    """
    candidates = np.where(
        np.abs(frame_times - time_value)
        <= args.unsupported_local_time_tolerance_s + EPS
    )[0]

    best = {
        "nearest_raw_local_distance_m": math.nan,
        "nearest_raw_local_time_difference_s": math.nan,
        "nearest_raw_local_camera": "",
        "nearest_raw_local_id": "",
        "raw_local_support_found": False,
    }

    for frame_index in candidates:
        frame = local_frames[int(frame_index)]
        if frame.empty:
            continue
        frame_time = float(frame.time.median())
        for row in frame.itertuples(index=False):
            distance = float(math.hypot(
                float(row.x) - global_x,
                float(row.y) - global_y,
            ))
            previous = best["nearest_raw_local_distance_m"]
            if not np.isfinite(previous) or distance < previous:
                best = {
                    "nearest_raw_local_distance_m": distance,
                    "nearest_raw_local_time_difference_s": abs(frame_time - time_value),
                    "nearest_raw_local_camera": str(row.camera),
                    "nearest_raw_local_id": str(row.local_id),
                    "raw_local_support_found": (
                        distance <= args.unsupported_local_distance + EPS
                    ),
                }
    return best


def state_global_id_near_time(
    states: pd.DataFrame,
    episode_identity: str,
    query_time: float,
    tolerance: float,
) -> str:
    """Return the Global ID from the nearest local-supported state."""
    if states.empty:
        return ""
    valid = states[
        (states.episode_identity == episode_identity)
        & states.matched.astype(bool)
        & (states.global_id.astype(str) != "")
    ].copy()
    if valid.empty:
        return ""
    valid["time_error"] = np.abs(valid.time.to_numpy(float) - float(query_time))
    row = valid.sort_values("time_error").iloc[0]
    return (
        clean_id(row.global_id)
        if float(row.time_error) <= tolerance + EPS
        else ""
    )


def reentry_events_from_local_support(
    support_timeline: pd.DataFrame,
    states: pd.DataFrame,
    max_gap: float,
    args,
) -> pd.DataFrame:
    """
    Re-entry is evaluated only between two substantial local-supported segments.

    Small sampling holes are ignored. Identity is read from the nearest
    local-supported state at the two segment boundaries using a real time
    tolerance rather than exact floating-point comparisons.
    """
    if support_timeline.empty:
        return pd.DataFrame()

    rows = []
    for episode_identity, group in support_timeline.groupby(
        "episode_identity", sort=False
    ):
        times = np.sort(group.time.unique().astype(float))
        if len(times) < 2:
            continue

        base_dt = nominal_dt(times, fallback=0.1)
        split_threshold = max(
            args.local_support_min_gap_s,
            args.local_support_gap_factor * base_dt,
            args.reentry_min_gap_s,
        )

        segments = []
        start = previous = float(times[0])
        count = 1
        for current in times[1:]:
            current = float(current)
            if current - previous > split_threshold + EPS:
                segments.append({
                    "start": start,
                    "end": previous,
                    "frames": count,
                })
                start = current
                count = 1
            else:
                count += 1
            previous = current
        segments.append({"start": start, "end": previous, "frames": count})

        for index in range(len(segments) - 1):
            before = segments[index]
            after = segments[index + 1]
            gap = float(after["start"] - before["end"])

            if gap < args.reentry_min_gap_s - EPS or gap > max_gap + EPS:
                continue
            if (
                before["frames"] < args.reentry_min_segment_frames
                or after["frames"] < args.reentry_min_segment_frames
            ):
                continue

            before_id = state_global_id_near_time(
                states,
                episode_identity,
                before["end"],
                args.reentry_id_time_tolerance_s,
            )
            after_id = state_global_id_near_time(
                states,
                episode_identity,
                after["start"],
                args.reentry_id_time_tolerance_s,
            )

            if not before_id:
                outcome = "not_evaluable_before"
            elif not after_id:
                outcome = "not_reacquired"
            elif before_id == after_id:
                outcome = "preserved"
            else:
                outcome = "changed"

            rows.append({
                "episode_identity": episode_identity,
                "support_segment_before_start": before["start"],
                "support_segment_before_end": before["end"],
                "support_segment_before_frames": before["frames"],
                "support_segment_after_start": after["start"],
                "support_segment_after_end": after["end"],
                "support_segment_after_frames": after["frames"],
                "gap_start_time": before["end"],
                "gap_end_time": after["start"],
                "gap_duration_s": gap,
                "global_id_before": before_id,
                "global_id_after": after_id,
                "outcome": outcome,
                "support_split_threshold_s": split_threshold,
            })
    return pd.DataFrame(rows)


def associate_local(local_rows: pd.DataFrame, visible_gt: pd.DataFrame, max_distance: float) -> pd.DataFrame:
    if local_rows.empty or visible_gt.empty:
        return pd.DataFrame()
    rows = []
    gt_xy = visible_gt[["gt_x", "gt_y"]].to_numpy(float)
    for camera, camera_rows in local_rows.groupby("camera", sort=False):
        local_xy = camera_rows[["x", "y"]].to_numpy(float)
        cost = np.linalg.norm(gt_xy[:, None, :] - local_xy[None, :, :], axis=2)
        gi, li = linear_sum_assignment(cost)
        for g, l in zip(gi, li):
            distance = float(cost[g, l])
            if distance > max_distance:
                continue
            gt_row = visible_gt.iloc[int(g)]
            local_row = camera_rows.iloc[int(l)]
            rows.append({
                "agent_id": str(gt_row.agent_id),
                "gt_x": float(gt_row.gt_x),
                "gt_y": float(gt_row.gt_y),
                "camera": str(camera),
                "local_id": str(local_row.local_id),
                "local_x": float(local_row.x),
                "local_y": float(local_row.y),
                "local_gt_distance_m": distance,
            })
    return pd.DataFrame(rows)


def supported_agents(local_assoc: pd.DataFrame, lookup, time_value: float) -> pd.DataFrame:
    if local_assoc.empty:
        return pd.DataFrame()
    rows = []
    for agent_id, group in local_assoc.groupby("agent_id", sort=False):
        episode = episode_at(lookup, str(agent_id), time_value)
        if episode is None:
            continue
        rows.append({
            "episode_identity": episode.identity,
            "agent_id": str(agent_id),
            "episode_index": episode.index,
            "gt_x": float(group.gt_x.median()),
            "gt_y": float(group.gt_y.median()),
            "local_x": float(group.local_x.median()),
            "local_y": float(group.local_y.median()),
            "local_camera_count": int(group.camera.nunique()),
            "local_observation_count": int(len(group)),
            "local_cameras": ", ".join(sorted(group.camera.unique())),
            "local_ids": ", ".join(sorted(group.local_id.unique(), key=natural_key)),
        })
    return pd.DataFrame(rows)


def global_matches(supported: pd.DataFrame, global_rows: pd.DataFrame, max_distance: float):
    if supported.empty:
        return [], [], list(range(len(global_rows)))
    if global_rows.empty:
        return [], list(range(len(supported))), []
    axy = supported[["local_x", "local_y"]].to_numpy(float)
    gxy = global_rows[["x", "y"]].to_numpy(float)
    cost = np.linalg.norm(axy[:, None, :] - gxy[None, :, :], axis=2)
    gated = cost.copy()
    gated[gated > max_distance] = 1e9
    ai, gi = linear_sum_assignment(gated)
    matches, ma, mg = [], set(), set()
    for a, g in zip(ai, gi):
        distance = float(cost[a, g])
        if distance > max_distance:
            continue
        agent = supported.iloc[int(a)]
        prediction = global_rows.iloc[int(g)]
        matches.append({
            "agent_index": int(a),
            "global_index": int(g),
            "episode_identity": str(agent.episode_identity),
            "agent_id": str(agent.agent_id),
            "global_id": str(prediction.global_id),
            "global_local_distance_m": distance,
            "global_gt_error_m": float(math.hypot(
                float(prediction.x) - float(agent.gt_x),
                float(prediction.y) - float(agent.gt_y),
            )),
            "global_x": float(prediction.x),
            "global_y": float(prediction.y),
        })
        ma.add(int(a)); mg.add(int(g))
    return matches, [i for i in range(len(supported)) if i not in ma], [i for i in range(len(global_rows)) if i not in mg]


def optimal_identity_mapping(matched: pd.DataFrame):
    if matched.empty:
        return {}, 0
    episodes = sorted(matched.episode_identity.unique(), key=natural_key)
    gids = sorted(matched.global_id.unique(), key=natural_key)
    counts = matched.groupby(["episode_identity", "global_id"]).size().to_dict()
    matrix = np.zeros((len(episodes), len(gids)), dtype=float)
    for i, episode in enumerate(episodes):
        for j, gid in enumerate(gids):
            matrix[i, j] = counts.get((episode, gid), 0)
    ri, ci = linear_sum_assignment(-matrix)
    mapping, idtp = {}, 0
    for r, c in zip(ri, ci):
        if matrix[r, c] > 0:
            mapping[episodes[r]] = gids[c]
            idtp += int(matrix[r, c])
    return mapping, idtp


def episode_events(
    states: pd.DataFrame,
    local_support_gap_factor: float,
    local_support_min_gap_s: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Count ID switches and fragmentations only inside contiguous local-supported
    segments.

    Fragmentation:
        matched -> one or more unmatched local-supported frames -> matched

    A real gap in the local input starts a new segment. It is evaluated by the
    re-entry metric and is not counted as a fusion fragmentation.
    """
    summary_rows = []
    event_rows = []

    for episode, group in states.groupby("episode_identity", sort=False):
        group = group.sort_values("time").reset_index(drop=True)
        times = group.time.to_numpy(float)
        if len(times) == 0:
            continue

        nominal = nominal_dt(times, fallback=0.1)
        split_threshold = max(
            float(local_support_min_gap_s),
            float(local_support_gap_factor) * nominal,
        )

        segments = []
        start = 0
        for index in range(1, len(group)):
            if float(times[index] - times[index - 1]) > split_threshold + EPS:
                segments.append((start, index))
                start = index
        segments.append((start, len(group)))

        switches = 0
        fragments = 0

        for segment_index, (start, end) in enumerate(segments, start=1):
            segment = group.iloc[start:end].reset_index(drop=True)
            previous_id = None
            last_matched_id = None
            missing_start = None
            missing_count = 0

            for row in segment.itertuples(index=False):
                current_id = clean_id(row.global_id)

                if current_id:
                    if previous_id is not None and current_id != previous_id:
                        switches += 1
                        event_rows.append({
                            "episode_identity": episode,
                            "segment_index": segment_index,
                            "event_type": "id_switch",
                            "start_time": float(row.time),
                            "end_time": float(row.time),
                            "missing_frame_count": 0,
                            "global_id_before": previous_id,
                            "global_id_after": current_id,
                        })

                    if missing_start is not None:
                        fragments += 1
                        event_rows.append({
                            "episode_identity": episode,
                            "segment_index": segment_index,
                            "event_type": "fragmentation",
                            "start_time": float(missing_start),
                            "end_time": float(row.time),
                            "missing_frame_count": int(missing_count),
                            "global_id_before": last_matched_id or "",
                            "global_id_after": current_id,
                        })
                        missing_start = None
                        missing_count = 0

                    previous_id = current_id
                    last_matched_id = current_id

                elif last_matched_id is not None:
                    if missing_start is None:
                        missing_start = float(row.time)
                    missing_count += 1

            # A missing tail is not a fragmentation because the track never
            # resumes inside the same local-supported segment.

        summary_rows.append({
            "episode_identity": episode,
            "id_switches": switches,
            "fragmentations": fragments,
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(event_rows)


def reentry_events(states: pd.DataFrame, max_gap: float) -> pd.DataFrame:
    if states.empty:
        return pd.DataFrame()
    times_all = np.sort(states.time.unique().astype(float))
    threshold = max(1.5 * nominal_dt(times_all, 0.3), nominal_dt(times_all, 0.3) + 1e-6)
    rows = []
    for episode, group in states.groupby("episode_identity", sort=False):
        group = group.sort_values("time").reset_index(drop=True)
        times = group.time.to_numpy(float)
        for i in range(len(group) - 1):
            gap = float(times[i + 1] - times[i])
            if gap <= threshold or gap > max_gap + EPS:
                continue
            before = clean_id(group.iloc[i].global_id)
            after = clean_id(group.iloc[i + 1].global_id)
            if not before:
                outcome = "not_evaluable_before"
            elif not after:
                outcome = "not_reacquired"
            elif before == after:
                outcome = "preserved"
            else:
                outcome = "changed"
            rows.append({
                "episode_identity": episode,
                "gap_start_time": float(times[i]),
                "gap_end_time": float(times[i + 1]),
                "gap_duration_s": gap,
                "global_id_before": before,
                "global_id_after": after,
                "outcome": outcome,
            })
    return pd.DataFrame(rows)


def evaluate(robot, gt, globals_df, local, args):
    """
    Two complementary timelines are evaluated separately:

    1. Local-supported timeline:
       coverage, misses, identity consistency, MT/PT/ML and re-entry.

    2. Global-output timeline:
       lifecycle persistence, duplicates and unmatched-output diagnostics.
    """
    vis = visibility_table(robot, gt, args)
    episodes = make_episodes(vis, args)
    lookup = defaultdict(list)
    for episode in episodes:
        lookup[episode.agent_id].append(episode)

    vis_tracks = build_tracks(vis, "agent_id")
    global_frames = group_frames(globals_df, args.assignment_time_bin_s)
    global_times = np.asarray(
        [float(frame.time.median()) for frame in global_frames],
        dtype=float,
    )
    local_frames = group_frames(local, args.assignment_time_bin_s)
    local_times = np.asarray(
        [float(frame.time.median()) for frame in local_frames],
        dtype=float,
    )

    local_support_timeline = build_local_support_timeline(
        local_frames,
        vis_tracks,
        lookup,
        args,
    )
    if local_support_timeline.empty:
        raise RuntimeError("No local-supported agent frames found.")

    # ------------------------------------------------------------------
    # A. LOCAL-SUPPORTED TIMELINE
    # ------------------------------------------------------------------
    state_rows = []
    consolidation_rows = []
    local_match_rows = []

    for time_value, supported in local_support_timeline.groupby("time", sort=True):
        time_value = float(time_value)
        supported = supported.reset_index(drop=True)

        global_frame = pd.DataFrame(columns=["time", "global_id", "x", "y"])
        if len(global_times):
            nearest = nearest_index(global_times, time_value)
            if (
                nearest is not None
                and abs(float(global_times[nearest]) - time_value)
                <= args.global_time_tolerance_s + EPS
            ):
                global_frame = global_frames[int(nearest)][
                    ["time", "global_id", "x", "y"]
                ].reset_index(drop=True)

        matches, _, unmatched_globals = global_matches(
            supported,
            global_frame,
            args.global_local_max_distance,
        )
        match_by_agent = {m["agent_index"]: m for m in matches}
        duplicate_counts = defaultdict(int)

        # Any extra Global Track close to an already represented supported
        # agent is a duplicate for the local-supported evaluation frame.
        for global_index in unmatched_globals:
            prediction = global_frame.iloc[int(global_index)]
            if supported.empty:
                continue
            distances = np.linalg.norm(
                supported[["local_x", "local_y"]].to_numpy(float)
                - np.asarray([float(prediction.x), float(prediction.y)])[None, :],
                axis=1,
            )
            nearest_agent = int(np.argmin(distances))
            if float(distances[nearest_agent]) <= args.duplicate_global_distance + EPS:
                duplicate_counts[nearest_agent] += 1

        for agent_index, agent in supported.iterrows():
            match = match_by_agent.get(int(agent_index))
            duplicate_count = duplicate_counts.get(int(agent_index), 0)
            global_id = match["global_id"] if match else ""

            state_rows.append({
                "time": time_value,
                "episode_identity": str(agent.episode_identity),
                "agent_id": str(agent.agent_id),
                "local_camera_count": int(agent.local_camera_count),
                "local_observation_count": int(agent.local_observation_count),
                "global_id": global_id,
                "matched": match is not None,
                "duplicate_global_count": duplicate_count,
                "global_local_distance_m": (
                    match["global_local_distance_m"] if match else math.nan
                ),
                "global_gt_error_m": (
                    match["global_gt_error_m"] if match else math.nan
                ),
            })

            if match is not None:
                local_match_rows.append({
                    "time": time_value,
                    "episode_identity": str(agent.episode_identity),
                    "agent_id": str(agent.agent_id),
                    "global_id": global_id,
                    "global_local_distance_m": match["global_local_distance_m"],
                    "global_gt_error_m": match["global_gt_error_m"],
                })

            outcome = (
                "fusion_miss"
                if match is None
                else "duplicate_global"
                if duplicate_count > 0
                else "single_global_id"
            )
            consolidation_rows.append({
                "time": time_value,
                "episode_identity": str(agent.episode_identity),
                "agent_id": str(agent.agent_id),
                "local_camera_count": int(agent.local_camera_count),
                "outcome": outcome,
                "global_id": global_id,
                "duplicate_global_count": duplicate_count,
            })

    states = pd.DataFrame(state_rows)
    consolidation = pd.DataFrame(consolidation_rows)
    local_matches = pd.DataFrame(local_match_rows)

    # Episode-local dominant identity. No global one-to-one constraint is
    # imposed across different visibility episodes.
    dominant_mapping = {}
    idtp = 0
    wrong_identity_events = 0
    for episode_identity, group in states.groupby("episode_identity", sort=False):
        matched_ids = group.loc[group.matched, "global_id"].map(clean_id)
        matched_ids = matched_ids[matched_ids != ""]
        if matched_ids.empty:
            continue
        counts = matched_ids.value_counts()
        dominant_id = str(counts.index[0])
        dominant_mapping[episode_identity] = dominant_id
        idtp += int(counts.iloc[0])
        wrong_identity_events += int((matched_ids != dominant_id).sum())

    total_evaluable = len(states)
    idfn = total_evaluable - idtp

    events, identity_event_details = episode_events(
        states,
        args.local_support_gap_factor,
        args.local_support_min_gap_s,
    )
    episode_rows = []
    for episode_identity, group in states.groupby("episode_identity", sort=False):
        evaluable = len(group)
        matched_frames = int(group.matched.sum())
        coverage_ratio = matched_frames / evaluable if evaluable else 0.0
        dominant = dominant_mapping.get(episode_identity, "")
        dominant_count = int(
            (
                group.loc[group.matched, "global_id"].map(clean_id)
                == dominant
            ).sum()
        ) if dominant else 0
        identity_purity_when_tracked = (
            dominant_count / matched_frames
            if matched_frames else math.nan
        )

        if evaluable < args.min_evaluable_frames:
            status = "INSUFFICIENT_DATA"
        else:
            status = (
                "MT" if coverage_ratio >= 0.8
                else "ML" if coverage_ratio <= 0.2
                else "PT"
            )

        ev = events[events.episode_identity == episode_identity]
        episode_rows.append({
            "episode_identity": episode_identity,
            "evaluable_frames": evaluable,
            "matched_frames": matched_frames,
            "fusion_coverage": coverage_ratio,
            "dominant_global_id": dominant,
            "global_id_purity_when_tracked": identity_purity_when_tracked,
            "id_switches": int(ev.iloc[0].id_switches) if not ev.empty else 0,
            "fragmentations": int(ev.iloc[0].fragmentations) if not ev.empty else 0,
            "status": status,
            "fusion_misses": int((~group.matched).sum()),
            "duplicate_agent_frames": int(
                (group.duplicate_global_count > 0).sum()
            ),
        })
    episode_summary = pd.DataFrame(episode_rows)

    # ------------------------------------------------------------------
    # B. GLOBAL-OUTPUT TIMELINE
    # ------------------------------------------------------------------
    assignment_rows = []
    last_match_by_global: Dict[str, Tuple[str, float]] = {}

    for global_frame in global_frames:
        time_value = float(global_frame.time.median())
        global_frame = global_frame[
            ["time", "global_id", "x", "y"]
        ].reset_index(drop=True)

        visible_gt = visible_gt_now(
            vis_tracks,
            time_value,
            args.local_time_tolerance_s,
        )
        local_rows = locals_near(
            local_frames,
            local_times,
            time_value,
            args.local_time_tolerance_s,
        )
        local_assoc = associate_local(
            local_rows,
            visible_gt,
            args.local_gt_max_distance,
        )
        supported = supported_agents(local_assoc, lookup, time_value)

        matches, _, unmatched_globals = global_matches(
            supported,
            global_frame,
            args.global_local_max_distance,
        )
        current_by_episode = {
            m["episode_identity"]: m["global_id"] for m in matches
        }

        for match in matches:
            classification = (
                "spatial_mismatch_local_supported"
                if match["global_gt_error_m"]
                > args.spatial_error_threshold + EPS
                else "correct_association"
            )
            assignment_rows.append({
                "time": time_value,
                "global_id": match["global_id"],
                "classification": classification,
                "episode_identity": match["episode_identity"],
                "agent_id": match["agent_id"],
                "global_x": match["global_x"],
                "global_y": match["global_y"],
                "global_local_distance_m": match["global_local_distance_m"],
                "global_gt_error_m": match["global_gt_error_m"],
                "age_since_last_match_s": 0.0,
                "nearest_raw_local_distance_m": match["global_local_distance_m"],
                "nearest_raw_local_time_difference_s": 0.0,
                "nearest_raw_local_camera": "",
                "nearest_raw_local_id": "",
                "raw_local_support_found": True,
                "inside_geometric_scope": True,
                "classification_reason": (
                    "Assigned to a GT-labelled local-supported agent."
                ),
                "penalized_identity_error": False,
                "identity_adjudicated": True,
            })
            last_match_by_global[match["global_id"]] = (
                match["episode_identity"],
                time_value,
            )

        for global_index in unmatched_globals:
            prediction = global_frame.iloc[int(global_index)]
            gid = str(prediction.global_id)
            global_x = float(prediction.x)
            global_y = float(prediction.y)
            previous = last_match_by_global.get(gid)

            raw_support = raw_local_support_for_global(
                local_frames,
                local_times,
                time_value,
                global_x,
                global_y,
                args,
            )

            inside_scope = not visible_gt.empty
            previous_episode = ""
            age = math.nan

            if previous is not None:
                previous_episode, previous_time = previous
                age = time_value - previous_time
                if age <= args.publish_max_age_s + EPS:
                    current_id = current_by_episode.get(previous_episode)
                    classification = (
                        "duplicate_residual_output"
                        if current_id is not None and current_id != gid
                        else "lifecycle_persistence"
                    )
                    assignment_rows.append({
                        "time": time_value,
                        "global_id": gid,
                        "classification": classification,
                        "episode_identity": previous_episode,
                        "agent_id": previous_episode.split("-E")[0],
                        "global_x": global_x,
                        "global_y": global_y,
                        "global_local_distance_m": math.nan,
                        "global_gt_error_m": math.nan,
                        "age_since_last_match_s": age,
                        **raw_support,
                        "inside_geometric_scope": inside_scope,
                        "classification_reason": (
                            "Same Global ID remains inside publish_max_age."
                            if classification == "lifecycle_persistence"
                            else "Another Global ID already represents the previous episode."
                        ),
                        "penalized_identity_error": (
                            classification == "duplicate_residual_output"
                        ),
                        "identity_adjudicated": True,
                    })
                    continue

            duplicate_agent = None
            duplicate_distance = math.inf
            if not supported.empty:
                distances = np.linalg.norm(
                    supported[["local_x", "local_y"]].to_numpy(float)
                    - np.asarray([global_x, global_y])[None, :],
                    axis=1,
                )
                nearest = int(np.argmin(distances))
                if float(distances[nearest]) <= args.duplicate_global_distance + EPS:
                    duplicate_agent = nearest
                    duplicate_distance = float(distances[nearest])

            if duplicate_agent is not None:
                agent = supported.iloc[duplicate_agent]
                assignment_rows.append({
                    "time": time_value,
                    "global_id": gid,
                    "classification": "duplicate_global_output",
                    "episode_identity": str(agent.episode_identity),
                    "agent_id": str(agent.agent_id),
                    "global_x": global_x,
                    "global_y": global_y,
                    "global_local_distance_m": duplicate_distance,
                    "global_gt_error_m": float(math.hypot(
                        global_x - float(agent.gt_x),
                        global_y - float(agent.gt_y),
                    )),
                    "age_since_last_match_s": age,
                    **raw_support,
                    "inside_geometric_scope": inside_scope,
                    "classification_reason": (
                        "An extra Global ID is close to an already represented agent."
                    ),
                    "penalized_identity_error": True,
                    "identity_adjudicated": True,
                })
                continue

            if raw_support["raw_local_support_found"]:
                classification = "unresolved_raw_local_supported"
                reason = (
                    "A nearby raw local observation exists, but its GT identity "
                    "cannot be established reliably."
                )
                penalized = False
                adjudicated = False
            elif (
                previous is not None
                and np.isfinite(age)
                and age <= args.extended_lifecycle_age_s + EPS
            ):
                classification = "extended_lifecycle_candidate"
                reason = (
                    "Same Global ID has recent history beyond publish_max_age; "
                    "kept diagnostic and not automatically penalized."
                )
                penalized = False
                adjudicated = False
            elif not inside_scope:
                classification = "out_of_scope_global_output"
                reason = (
                    "No GT agent is inside the reconstructed multi-camera FOV."
                )
                penalized = False
                adjudicated = True
            else:
                classification = "true_unsupported_global_output"
                reason = (
                    "No local support, lifecycle explanation, duplicate explanation "
                    "or out-of-scope condition was found."
                )
                penalized = True
                adjudicated = True

            assignment_rows.append({
                "time": time_value,
                "global_id": gid,
                "classification": classification,
                "episode_identity": previous_episode,
                "agent_id": (
                    previous_episode.split("-E")[0]
                    if previous_episode else ""
                ),
                "global_x": global_x,
                "global_y": global_y,
                "global_local_distance_m": math.nan,
                "global_gt_error_m": math.nan,
                "age_since_last_match_s": age,
                **raw_support,
                "inside_geometric_scope": inside_scope,
                "classification_reason": reason,
                "penalized_identity_error": penalized,
                "identity_adjudicated": adjudicated,
            })

    assignments = pd.DataFrame(assignment_rows)

    output_identity_errors = int(
        assignments["penalized_identity_error"]
        .fillna(False)
        .astype(bool)
        .sum()
    )
    idfp = wrong_identity_events + output_identity_errors

    idr = idtp / (idtp + idfn) if idtp + idfn else 0.0
    idp = idtp / (idtp + idfp) if idtp + idfp else 0.0
    idf1 = (
        2 * idtp / (2 * idtp + idfp + idfn)
        if 2 * idtp + idfp + idfn else 0.0
    )
    coverage = float(states.matched.mean())

    reentries = reentry_events_from_local_support(
        local_support_timeline,
        states,
        args.episode_merge_gap_s,
        args,
    )
    evaluable_reentries = (
        reentries[
            reentries.outcome.isin(
                ["preserved", "changed", "not_reacquired"]
            )
        ]
        if not reentries.empty
        else pd.DataFrame()
    )
    preserved = (
        int((evaluable_reentries.outcome == "preserved").sum())
        if not evaluable_reentries.empty else 0
    )
    reentry_rate = (
        preserved / len(evaluable_reentries)
        if len(evaluable_reentries) else math.nan
    )

    multi = consolidation[consolidation.local_camera_count >= 2]
    correct_multi = int((multi.outcome == "single_global_id").sum())
    consolidation_rate = (
        correct_multi / len(multi) if len(multi) else math.nan
    )

    duplicate_frames = int((states.duplicate_global_count > 0).sum())
    duplicate_rate = duplicate_frames / len(states)

    true_unsupported = int(
        (
            assignments.classification
            == "true_unsupported_global_output"
        ).sum()
    )
    unresolved = int(
        (
            assignments.classification
            == "unresolved_raw_local_supported"
        ).sum()
    )
    extended = int(
        (
            assignments.classification
            == "extended_lifecycle_candidate"
        ).sum()
    )
    out_of_scope = int(
        (
            assignments.classification
            == "out_of_scope_global_output"
        ).sum()
    )


    episodes_df = pd.DataFrame([{
        "episode_identity": episode.identity,
        "agent_id": episode.agent_id,
        "episode_index": episode.index,
        "start_time": episode.start,
        "end_time": episode.end,
        "duration_s": episode.end - episode.start,
    } for episode in episodes])

    summary = pd.DataFrame([{
        "local_supported_agent_frames": total_evaluable,
        "matched_agent_frames": int(states.matched.sum()),
        "fusion_coverage": coverage,
        "idtp": idtp,
        "idfp_fusion": idfp,
        "idfn_fusion": idfn,
        "fusion_id_precision": idp,
        "fusion_id_recall": idr,
        "fusion_idf1": idf1,
        "id_switches": int(episode_summary.id_switches.sum()),
        "fragmentations": int(episode_summary.fragmentations.sum()),
        "reentry_events_evaluable": len(evaluable_reentries),
        "reentry_preserved": preserved,
        "reentry_preservation_rate": reentry_rate,
        "multicamera_agent_frames": len(multi),
        "multicamera_correctly_consolidated": correct_multi,
        "multicamera_consolidation_rate": consolidation_rate,
        "duplicate_agent_frames": duplicate_frames,
        "duplicate_global_rate": duplicate_rate,
        "wrong_identity_events": wrong_identity_events,
        "true_unsupported_global_outputs": true_unsupported,
        "true_unsupported_output_rate": (
            true_unsupported / len(assignments)
            if len(assignments) else 0.0
        ),
        "unresolved_raw_local_supported_outputs": unresolved,
        "extended_lifecycle_candidates": extended,
        "out_of_scope_global_outputs": out_of_scope,
        "lifecycle_persistence_events": int(
            (
                assignments.classification
                == "lifecycle_persistence"
            ).sum()
        ),
        "spatial_mismatch_local_supported_events": int(
            (
                assignments.classification
                == "spatial_mismatch_local_supported"
            ).sum()
        ),
        "mostly_tracked_episodes": int(
            (episode_summary.status == "MT").sum()
        ),
        "partially_tracked_episodes": int(
            (episode_summary.status == "PT").sum()
        ),
        "mostly_lost_episodes": int(
            (episode_summary.status == "ML").sum()
        ),
        "insufficient_data_episodes": int(
            (
                episode_summary.status
                == "INSUFFICIENT_DATA"
            ).sum()
        ),
    }])

    consolidation_summary = (
        consolidation.groupby(["local_camera_count", "outcome"])
        .size()
        .reset_index(name="count")
    )
    return (
        episodes_df,
        local_support_timeline,
        assignments,
        states,
        episode_summary,
        consolidation_summary,
        reentries,
        identity_event_details,
        summary,
    )


def paper_axes(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, alpha=0.25, linewidth=0.8)
    ax.tick_params(labelsize=9.5)


def plot_summary(summary, path, dpi):
    row = summary.iloc[0]
    labels = [
        "Fusion IDF1",
        "ID precision",
        "ID recall",
        "Fusion coverage",
        "Re-entry\npreservation",
    ]
    values = [
        100 * row.fusion_idf1,
        100 * row.fusion_id_precision,
        100 * row.fusion_id_recall,
        100 * row.fusion_coverage,
        (
            100 * row.reentry_preservation_rate
            if np.isfinite(row.reentry_preservation_rate)
            else math.nan
        ),
    ]

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        np.nan_to_num(values, nan=0.0),
        width=0.66,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score [%]")
    ax.set_title("Fusion-specific identity performance", pad=12)
    ax.set_ylim(0, 108)
    paper_axes(ax)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5 if np.isfinite(value) else 2,
            "N/A" if not np.isfinite(value) else f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.20,
        top=0.88,
    )
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_episode_coverage(table, path, dpi):
    ordered = table.sort_values("fusion_coverage")
    labels = ordered.episode_identity.tolist()
    coverage = 100 * ordered.fusion_coverage.to_numpy(float)
    purity = 100 * ordered.global_id_purity_when_tracked.to_numpy(float)

    y = np.arange(len(labels))
    h = 0.34
    fig, ax = plt.subplots(
        figsize=(9.0, max(4.0, 0.62 * len(labels) + 1.8))
    )

    coverage_bars = ax.barh(
        y - h / 2,
        coverage,
        h,
        label="Fusion coverage",
    )
    purity_bars = ax.barh(
        y + h / 2,
        np.nan_to_num(purity, nan=0.0),
        h,
        label="ID purity when tracked",
    )
    ax.axvline(
        80,
        linestyle="--",
        linewidth=1.1,
        label="MT coverage threshold = 80%",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Episode score [%]")
    ax.set_title("Fusion coverage and ID purity by episode", pad=12)
    paper_axes(ax, "x")

    for bars, values in (
        (coverage_bars, coverage),
        (purity_bars, purity),
    ):
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(
                    min(value + 1.0, 102.5),
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}%",
                    va="center",
                    fontsize=8.5,
                )

    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.10, 0.5),
        borderaxespad=0.0,
    )
    fig.subplots_adjust(
        left=0.18,
        right=0.64,
        bottom=0.14,
        top=0.88,
    )
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_consolidation(summary, path, dpi):
    camera_counts = [1, 2, 3]
    outcomes = ["single_global_id", "fusion_miss", "duplicate_global"]
    labels = ["Correct single Global ID", "Fusion miss", "Duplicate Global IDs"]
    values = {outcome: [] for outcome in outcomes}
    for count in camera_counts:
        subset = summary[summary.local_camera_count == count]
        total = int(subset["count"].sum())
        for outcome in outcomes:
            n = int(subset.loc[subset.outcome == outcome, "count"].sum())
            values[outcome].append(100*n/total if total else 0.0)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    x = np.arange(3); width = 0.24
    for i, (outcome, label) in enumerate(zip(outcomes, labels)):
        ax.bar(x+(i-1)*width, values[outcome], width, label=label)
    ax.set_xticks(x, ["1 local camera", "2 local cameras", "3 local cameras"])
    ax.set_ylabel("Agent-frame outcomes [%]")
    ax.set_title("Local-to-global multi-camera consolidation", pad=12)
    ax.set_ylim(0, 105); paper_axes(ax)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.subplots_adjust(left=0.11, right=0.70, bottom=0.16, top=0.88)
    fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)


def plot_classification(assignments, path, dpi):
    order = [
        "correct_association", "lifecycle_persistence",
        "spatial_mismatch_local_supported", "duplicate_global_output",
        "duplicate_residual_output", "unresolved_raw_local_supported",
        "extended_lifecycle_candidate", "out_of_scope_global_output",
        "true_unsupported_global_output",
    ]
    names = {
        "correct_association": "Correct association",
        "lifecycle_persistence": "Lifecycle persistence",
        "spatial_mismatch_local_supported": "Local-supported spatial mismatch",
        "duplicate_global_output": "Duplicate Global ID",
        "duplicate_residual_output": "Residual duplicate ID",
        "unresolved_raw_local_supported": "Unresolved raw-local support",
        "extended_lifecycle_candidate": "Extended lifecycle candidate",
        "out_of_scope_global_output": "Out-of-scope global output",
        "true_unsupported_global_output": "True unsupported global output",
    }
    counts = assignments.classification.value_counts()
    values = [int(counts.get(category, 0)) for category in order]
    labels = [names[category] for category in order]
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    y = np.arange(len(labels)); bars = ax.barh(y, values)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlabel("Frame-level global-output events")
    ax.set_title("Fusion-output classification", pad=12); paper_axes(ax, "x")
    maximum = max(values+[1])
    for bar, value in zip(bars, values):
        ax.text(value+0.02*maximum, bar.get_y()+bar.get_height()/2, str(value), va="center", fontsize=9)
    ax.set_xlim(0, maximum*1.20)
    fig.subplots_adjust(left=0.34, right=0.96, bottom=0.14, top=0.88)
    fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)


def plot_errors(table, path, dpi):
    labels = table.episode_identity.tolist()
    y = np.arange(len(labels)); width = 0.20
    columns = [
        ("id_switches", "ID switches"),
        ("fragmentations", "Fragmentations"),
        ("fusion_misses", "Fusion misses"),
        ("duplicate_agent_frames", "Duplicate-ID frames"),
    ]
    fig, ax = plt.subplots(figsize=(9.0, max(4.0, 0.60*len(labels)+1.8)))
    for i, (column, label) in enumerate(columns):
        ax.barh(y+(i-1.5)*width, table[column].to_numpy(float), width, label=label)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Number of local-supported frames/events")
    ax.set_title("Fusion identity errors by episode", pad=12)
    paper_axes(ax, "x")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.subplots_adjust(left=0.18, right=0.72, bottom=0.14, top=0.88)
    fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)


def plot_reentry(events, max_gap, path, dpi):
    if events.empty:
        return

    data = events[
        events.outcome.isin(["preserved", "changed", "not_reacquired"])
    ].copy()
    if data.empty:
        return

    data.sort_values(
        ["episode_identity", "gap_start_time"],
        inplace=True,
    )
    labels = [
        (
            f"{row.episode_identity}: "
            f"{row.global_id_before or 'none'} → "
            f"{row.global_id_after or 'none'}"
        )
        for row in data.itertuples(index=False)
    ]
    values = data.gap_duration_s.to_numpy(float)
    y = np.arange(len(labels))

    fig, ax = plt.subplots(
        figsize=(9.8, max(3.8, 0.58 * len(labels) + 1.8))
    )
    bars = ax.barh(y, values)
    ax.axvline(
        max_gap,
        linestyle="--",
        linewidth=1.1,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Gap duration [s]")
    ax.set_title(
        "Global-ID preservation after local-input gaps",
        pad=12,
    )
    paper_axes(ax, "x")

    for bar, outcome in zip(bars, data.outcome.tolist()):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            outcome.replace("_", " "),
            va="center",
            fontsize=9,
        )

    # Keep visible space beyond the threshold and place the threshold label
    # on the right-hand side of the vertical line.
    data_max = float(np.nanmax(values)) if len(values) else 0.0
    right_limit = max(
        data_max + 0.7,
        float(max_gap) + 1.55,
    )
    ax.set_xlim(0.0, right_limit)

    ax.text(
        float(max_gap) + 0.12,
        0.08,
        f"Episode lifetime = {max_gap:g} s",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=9,
    )

    fig.subplots_adjust(
        left=0.35,
        right=0.95,
        bottom=0.14,
        top=0.88,
    )
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_table(summary, path, dpi):
    row = summary.iloc[0]
    pct = lambda value: "N/A" if not np.isfinite(value) else f"{100*value:.2f}%"
    rows = [
        ["Fusion IDF1", pct(row.fusion_idf1)],
        ["Fusion ID precision", pct(row.fusion_id_precision)],
        ["Fusion ID recall", pct(row.fusion_id_recall)],
        ["Fusion coverage", pct(row.fusion_coverage)],
        ["Re-entry preservation", pct(row.reentry_preservation_rate)],
        ["Multi-camera consolidation", ("N/A" if not np.isfinite(row.multicamera_consolidation_rate) else f"{100*row.multicamera_consolidation_rate:.2f}% ({int(row.multicamera_correctly_consolidated)}/{int(row.multicamera_agent_frames)})")],
        ["Duplicate Global-ID rate", pct(row.duplicate_global_rate)],
        ["True unsupported-output rate", pct(row.true_unsupported_output_rate)],
        ["Unresolved raw-local support", str(int(row.unresolved_raw_local_supported_outputs))],
        ["Extended lifecycle candidates", str(int(row.extended_lifecycle_candidates))],
        ["Out-of-scope global outputs", str(int(row.out_of_scope_global_outputs))],
        ["ID switches", str(int(row.id_switches))],
        ["Fragmentations", str(int(row.fragmentations))],
        ["Wrong-identity events", str(int(row.wrong_identity_events))],
        ["Lifecycle persistence events", str(int(row.lifecycle_persistence_events))],
        ["Local-supported spatial mismatches", str(int(row.spatial_mismatch_local_supported_events))],
        ["MT / PT / ML / insufficient episodes",
         f"{int(row.mostly_tracked_episodes)} / {int(row.partially_tracked_episodes)} / "
         f"{int(row.mostly_lost_episodes)} / {int(row.insufficient_data_episodes)}"],
    ]
    fig, ax = plt.subplots(figsize=(8.4, 6.5)); ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Fusion-specific metric", "Result"],
                     cellLoc="left", colLoc="center", loc="center", colWidths=[0.68, 0.24])
    table.auto_set_font_size(False); table.set_fontsize(9.2); table.scale(1.0, 1.45)
    for (r, _), cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_text_props(weight="bold", ha="center"); cell.set_facecolor("0.92")
        elif r % 2 == 0:
            cell.set_facecolor("0.975")
    ax.set_title("Fusion-specific identity and trajectory-integrity summary",
                 fontsize=13, weight="bold", pad=14)
    fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)



def plot_unsupported_timeline(assignments: pd.DataFrame, path: Path, dpi: int) -> None:
    categories = [
        "unresolved_raw_local_supported",
        "extended_lifecycle_candidate",
        "out_of_scope_global_output",
        "true_unsupported_global_output",
    ]
    data = assignments[assignments.classification.isin(categories)].copy()
    if data.empty:
        return

    data.sort_values(["global_id", "time"], inplace=True)
    global_ids = sorted(data.global_id.unique(), key=natural_key)
    y_map = {gid: index for index, gid in enumerate(global_ids)}

    fig, ax = plt.subplots(
        figsize=(10.0, max(3.8, 0.55 * len(global_ids) + 1.8))
    )

    markers = {
        "unresolved_raw_local_supported": "o",
        "extended_lifecycle_candidate": "s",
        "out_of_scope_global_output": "^",
        "true_unsupported_global_output": "X",
    }
    labels = {
        "unresolved_raw_local_supported": "Unresolved raw-local support",
        "extended_lifecycle_candidate": "Extended lifecycle candidate",
        "out_of_scope_global_output": "Out-of-scope output",
        "true_unsupported_global_output": "True unsupported output",
    }

    for category in categories:
        subset = data[data.classification == category]
        if subset.empty:
            continue
        ax.scatter(
            subset.time.to_numpy(float),
            [y_map[gid] for gid in subset.global_id],
            marker=markers[category],
            s=42,
            label=labels[category],
        )

    ax.set_yticks(np.arange(len(global_ids)))
    ax.set_yticklabels(global_ids)
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("Global ID")
    ax.set_title("Diagnostic timeline of unmatched global outputs", pad=12)
    paper_axes(ax, "x")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.subplots_adjust(left=0.12, right=0.72, bottom=0.14, top=0.88)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_unsupported_diagnostics(assignments: pd.DataFrame, path: Path, dpi: int) -> None:
    data = assignments[assignments.classification.isin([
        "unresolved_raw_local_supported",
        "extended_lifecycle_candidate",
        "out_of_scope_global_output",
        "true_unsupported_global_output",
    ])].copy()
    if data.empty:
        return

    order = [
        "unresolved_raw_local_supported",
        "extended_lifecycle_candidate",
        "out_of_scope_global_output",
        "true_unsupported_global_output",
    ]
    names = [
        "Unresolved\nraw-local",
        "Extended\nlifecycle",
        "Out of\nscope",
        "True\nunsupported",
    ]
    counts = [int((data.classification == category).sum()) for category in order]

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    bars = ax.bar(np.arange(len(order)), counts, width=0.62)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(names)
    ax.set_ylabel("Global-output events")
    ax.set_title("Diagnostic decomposition of unmatched global outputs", pad=12)
    paper_axes(ax)

    for bar, value in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(counts + [1]) * 0.025,
            str(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, max(counts + [1]) * 1.18)
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.20, top=0.86)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"[fusion_identity_validation] version: {SCRIPT_VERSION}")
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    robot, gt, globals_df, local = load_inputs(args)

    (
        episodes,
        supported,
        assignments,
        states,
        episode_summary,
        consolidation,
        reentries,
        identity_event_details,
        summary,
    ) = evaluate(robot, gt, globals_df, local, args)

    episodes.to_csv(out/"episode_definitions.csv", index=False)
    supported.to_csv(out/"local_supported_agent_frames.csv", index=False)
    assignments.to_csv(out/"global_assignment_events.csv", index=False)
    states.to_csv(out/"agent_frame_states.csv", index=False)
    episode_summary.to_csv(out/"episode_identity_summary.csv", index=False)
    consolidation.to_csv(out/"multicamera_consolidation_summary.csv", index=False)
    reentries.to_csv(out/"reentry_events.csv", index=False)
    identity_event_details.to_csv(
        out/"identity_switch_and_fragmentation_events.csv",
        index=False,
    )
    summary.to_csv(out/"fusion_identity_summary.csv", index=False)
    assignments[assignments.classification.isin([
        "unresolved_raw_local_supported",
        "extended_lifecycle_candidate",
        "out_of_scope_global_output",
        "true_unsupported_global_output",
    ])].to_csv(out/"unmatched_output_diagnostics.csv", index=False)

    plot_summary(summary, out/"01_fusion_identity_summary.png", args.dpi)
    plot_episode_coverage(
        episode_summary,
        out/"02_episode_coverage.png",
        args.dpi,
    )
    plot_classification(
        assignments,
        out/"03_global_output_classification.png",
        args.dpi,
    )
    plot_reentry(
        reentries,
        args.episode_merge_gap_s,
        out/"04_reentry_preservation.png",
        args.dpi,
    )
    plot_table(summary, out/"05_fusion_identity_table.png", args.dpi)
    plot_unsupported_timeline(
        assignments,
        out/"06_unsupported_output_timeline.png",
        args.dpi,
    )

    row = summary.iloc[0]
    print("\nFusion-specific identity validation complete.")
    print(f"Results directory: {out.resolve()}")
    print(f"\nLocal-supported agent frames: {int(row.local_supported_agent_frames)}")
    print(f"Fusion-evaluable episodes: {episode_summary.episode_identity.nunique()}")
    print(f"Multi-camera agent frames: {int(row.multicamera_agent_frames)}")
    print(f"\nFusion IDF1: {100*row.fusion_idf1:.2f}%")
    print(f"Fusion ID Precision: {100*row.fusion_id_precision:.2f}%")
    print(f"Fusion ID Recall: {100*row.fusion_id_recall:.2f}%")
    print(f"Fusion coverage: {100*row.fusion_coverage:.2f}%")
    print(f"ID switches: {int(row.id_switches)}")
    print(f"Fragmentations: {int(row.fragmentations)}")
    print("Re-entry preservation: " + (
        f"{100*row.reentry_preservation_rate:.2f}%"
        if np.isfinite(row.reentry_preservation_rate) else "N/A"
    ))
    print("Multi-camera consolidation: " + (
        f"{100*row.multicamera_consolidation_rate:.2f}%"
        if np.isfinite(row.multicamera_consolidation_rate) else "N/A"
    ))
    print(f"Duplicate Global-ID rate: {100*row.duplicate_global_rate:.2f}%")
    print(
        f"True unsupported-output rate: "
        f"{100*row.true_unsupported_output_rate:.2f}%"
    )
    print(
        f"Unresolved raw-local-supported outputs: "
        f"{int(row.unresolved_raw_local_supported_outputs)}"
    )
    print(
        "  Nearby raw local input exists, but the validation cannot assign "
        "that local observation reliably to a GT identity."
    )
    print(
        f"Extended lifecycle candidates: "
        f"{int(row.extended_lifecycle_candidates)}"
    )
    print(
        f"Out-of-scope global outputs: "
        f"{int(row.out_of_scope_global_outputs)}"
    )
    print(f"Wrong-identity events: {int(row.wrong_identity_events)}")
    print(f"Lifecycle persistence events: {int(row.lifecycle_persistence_events)}")
    print(f"Local-supported spatial mismatches: {int(row.spatial_mismatch_local_supported_events)}")
    print(
        "\nMethodological note: coverage and fragmentations are evaluated "
        "only inside contiguous local-supported segments. Real local-input gaps "
        "are evaluated separately through the re-entry metric."
    )


if __name__ == "__main__":
    main()
