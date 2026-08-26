import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse


# ============================================================
# USER CONFIG
# ============================================================

WINDOW_SIZE = 6.0
WINDOW_STEP = 6.0

# GIF lunga
LONG_GIF_START = 636.0
LONG_GIF_END = 672.0

GIF_FPS = 1
GT_TIME_TOLERANCE = 0.20
TRAIL_SECONDS = 2.0

# Robot and camera FOV visualization in XY animations/window maps.
SHOW_ROBOT_AND_FOV = True
ROBOT_POSE_TIME_TOLERANCE = 0.25
ROBOT_MARKER_SIZE = 180
ROBOT_ARROW_LENGTH = 0.7
ROBOT_PATH_TRAIL_SECONDS = 8.0

CENTER_CAM_YAW_OFFSET = np.deg2rad(0.0)
LEFT_CAM_YAW_OFFSET = np.deg2rad(58.0)
RIGHT_CAM_YAW_OFFSET = np.deg2rad(-58.0)

CAMERA_FOV_DEG = 69.0
CAMERA_FOV_LENGTH = 5.0

SHOW_INTERNAL_TRACKS_IN_ANIMATION = False
SHOW_INTERNAL_TRACKS_IN_WINDOW_SUMMARY = False

CREATE_PUBLISHED_ONLY_WINDOW_GIF = False
CREATE_PUBLISHED_ONLY_LONG_GIF = False
CREATE_LIFECYCLE_TIMELINE_LONG_GIF = False
CREATE_NEAREST_DETECTION_DISTANCE_LONG_GIF = False


# ============================================================
# PLOT STYLE CONFIG
# ============================================================

DETECTION_STYLES = {
    "center": {
        "marker": "o",
        "color": "yellow",
        "edgecolor": "black",
        "label": "det center",
    },
    "left": {
        "marker": "o",
        "color": "red",
        "edgecolor": "black",
        "label": "det left",
    },
    "right": {
        "marker": "o",
        "color": "orange",
        "edgecolor": "black",
        "label": "det right",
    },
    "unknown": {
        "marker": "o",
        "color": "gray",
        "edgecolor": "black",
        "label": "det unknown",
    },
}

GLOBAL_TRACK_STYLE = {
    "marker": "o",
    "color": "green",
    "edgecolor": "black",
    "label": "global track",
}

UNPUBLISHED_TRACK_STYLE = {
    "marker": "o",
    "color": "none",
    "edgecolor": "green",
    "label": "internal/unpublished track",
}

DUPLICATE_DROPPED_STYLE = {
    "marker": "x",
    "color": "purple",
    "label": "dropped duplicate",
}

GROUND_TRUTH_STYLE = {
    # Ground truth is drawn as a pair of small footprints instead of a star.
    # The marker entry is kept only for compatibility with older legend helpers.
    "marker": "o",
    "color": "blue",
    "edgecolor": "black",
    "label": "ground truth footprint",
}

# Ground-truth agent visualization.
# Footprints are generated directly with Matplotlib ellipses, so no external PNG is required.
GT_FOOTPRINT_COLOR = "blue"
GT_FOOTPRINT_LENGTH = 0.34  # larger footprint for presentation readability
GT_FOOTPRINT_WIDTH = 0.155
GT_FOOTPRINT_FORWARD_OFFSET = 0.085
GT_FOOTPRINT_LATERAL_OFFSET = 0.105
GT_FOOTPRINT_ALPHA = 0.88
GT_TRAIL_SECONDS = 3.0
GT_TRAIL_LINEWIDTH = 2.1
GT_VELOCITY_ARROW_SCALE = 0.62
GT_SHOW_VELOCITY_ARROW = True
GT_MIN_SPEED_FOR_ORIENTATION = 0.03

# Global-track presentation visualization.
# The fusion estimate is drawn as a green tracked-person symbol rather than a simple dot:
# body marker + uncertainty halo, without velocity arrow.
GLOBAL_TRACK_HALO_BASE_RADIUS = 0.20
GLOBAL_TRACK_HALO_MAX_EXTRA_RADIUS = 0.32
GLOBAL_TRACK_HALO_MISSED_GAIN = 0.045
LOCAL_DETECTION_LABEL_FONT_SIZE = 8
LOCAL_DETECTION_MARKER_SIZE = 85

ROBOT_STYLE = {
    "marker": "o",
    "color": "black",
    "edgecolor": "black",
    "label": "robot",
}

FOV_STYLES = {
    "center": {"color": "yellow", "label": "FOV center"},
    "left": {"color": "red", "label": "FOV left"},
    "right": {"color": "orange", "label": "FOV right"},
}

ASSOCIATION_LINE_COLOR = "black"
DUPLICATE_LINK_COLOR = "purple"
JUMP_REJECT_COLOR = "red"
LOCAL_ID_CONFLICT_REJECT_COLOR = "orange"


# ============================================================
# PATHS
# ============================================================

IN_DIR = Path(__file__).parent
OUT_DIR = IN_DIR / "fusion_analysis_plots"
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# PAPER TABLE OUTPUT CONFIG
# ============================================================
# Keep only the tables that are actually useful to explain whether the
# multi-camera tracking system behaves correctly. This avoids generating many
# low-level debug tables that are less suitable for a thesis/paper.
GENERATE_ONLY_KEY_PAPER_TABLES = True
KEY_PAPER_TABLE_NAMES = {
    "key_table_local_tracker_behavior",
    "key_table_fusion_matching_behavior",
    "key_table_track_lifecycle_behavior",
    "key_table_system_behavior_compact",
}

def resolve_csv(base_name):
    """
    Return the CSV path for base_name.

    Priority:
    1) exact file name, e.g. global_tracks_debug.csv
    2) exported variants, e.g. global_tracks_debug(10).csv
    """
    exact = IN_DIR / base_name
    if exact.exists():
        return exact

    stem = Path(base_name).stem
    matches = sorted(IN_DIR.glob(f"{stem}*.csv"))
    if matches:
        chosen = matches[-1]
        print(f"INFO: using {chosen.name} for expected file {base_name}")
        return chosen

    raise FileNotFoundError(f"Required CSV not found: {base_name}")

tracks = pd.read_csv(resolve_csv("global_tracks_debug.csv"))
detections = pd.read_csv(resolve_csv("local_detections_debug.csv"))
events = pd.read_csv(resolve_csv("fusion_events.csv"))
debug = pd.read_csv(resolve_csv("debug_events_expanded.csv"))

robot_path = resolve_csv("robot_pose_debug.csv") if (IN_DIR / "robot_pose_debug.csv").exists() or list(IN_DIR.glob("robot_pose_debug*.csv")) else IN_DIR / "robot_pose_debug.csv"
if robot_path.exists():
    robot = pd.read_csv(robot_path)
else:
    robot = pd.DataFrame(columns=[
        "time", "bag_time", "frame_id", "x", "y", "z", "yaw", "vx", "vy", "vtheta"
    ])
    print("WARNING: robot_pose_debug.csv not found. Animations will be generated without robot/FOV.")

gt_path = resolve_csv("ground_truth_people.csv") if (IN_DIR / "ground_truth_people.csv").exists() or list(IN_DIR.glob("ground_truth_people*.csv")) else IN_DIR / "ground_truth_people.csv"
if gt_path.exists():
    ground_truth = pd.read_csv(gt_path)
else:
    ground_truth = pd.DataFrame(columns=[
        "time", "bag_time", "frame_id", "agent_id", "x", "y", "z", "vx", "vy", "vz", "reliability"
    ])
    print("WARNING: ground_truth_people.csv not found. Animations will be generated without GT.")

local_tracker_path = resolve_csv("local_tracker_3d_debug.csv") if (IN_DIR / "local_tracker_3d_debug.csv").exists() or list(IN_DIR.glob("local_tracker_3d_debug*.csv")) else IN_DIR / "local_tracker_3d_debug.csv"
if local_tracker_path.exists():
    local_tracker_3d = pd.read_csv(local_tracker_path)
else:
    local_tracker_3d = pd.DataFrame()
    print("WARNING: local_tracker_3d_debug.csv not found. Local 3D diagnostic tables will be skipped.")


# ============================================================
# CLEANING
# ============================================================

numeric_cols = [
    "time", "cycle", "x", "y", "yaw", "vx", "vy", "vz", "z",
    "age", "last_update", "reliability", "stamp", "latest_stamp",
    "distance", "dynamic_threshold", "association_cost",
    "pred_x", "pred_y", "det_x", "det_y",
    "old_x", "old_y", "new_x", "new_y",
    "dropped_x", "dropped_y", "kept_x", "kept_y",
    "threshold", "max_jump", "max_conflict_match_dist",
    "global_id", "best_global_id", "second_global_id",
    "best_distance", "second_distance", "track_x", "track_y",
    "nearby_global_id", "bag_time",
    # Local tracker 3D debug CSV columns
    "det_index", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_score",
    "num_valid_keypoints", "fallback_valid_count", "fallback_valid_fraction",
    "fallback_depth_std", "camera_x", "camera_y", "camera_z",
    "map_x", "map_y", "map_z", "velocity_camera_x", "velocity_camera_y",
    "velocity_camera_z", "image_stamp", "transform_stamp",
]

for df in [tracks, detections, events, debug, ground_truth, robot, local_tracker_3d]:
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

if "publishable" in tracks.columns:
    tracks["publishable"] = tracks["publishable"].astype(str).str.lower() == "true"
else:
    tracks["publishable"] = True

if "confirmed" in tracks.columns:
    tracks["confirmed"] = tracks["confirmed"].astype(str).str.lower() == "true"
else:
    tracks["confirmed"] = False

for col in ["event_type", "local_id_relation", "local_id"]:
    if col in debug.columns:
        debug[col] = debug[col].astype(str)
    else:
        debug[col] = ""

published_tracks = tracks[(tracks["publishable"]) & (tracks["confirmed"])].copy()
tracks_for_debug_viz = tracks.copy() if SHOW_INTERNAL_TRACKS_IN_ANIMATION else published_tracks.copy()
tracks_for_window_summary = tracks.copy() if SHOW_INTERNAL_TRACKS_IN_WINDOW_SUMMARY else published_tracks.copy()

matches = debug[debug["event_type"] == "MATCH_ACCEPTED"].copy()
duplicates = debug[debug["event_type"] == "DUPLICATE_DROP"].copy()
new_tracks = debug[debug["event_type"] == "NEW_TRACK"].copy()
reactivations = debug[debug["event_type"] == "RECOVER_EXISTING_TRACK"].copy()
deleted_tracks = debug[debug["event_type"] == "DELETE_TRACK"].copy()

jump_rejects = debug[debug["event_type"] == "MATCH_CANDIDATE_REJECTED_JUMP"].copy()
conflict_rejects = debug[debug["event_type"] == "MATCH_CANDIDATE_REJECTED_LOCAL_ID_CONFLICT"].copy()
recover_jump_rejects = debug[debug["event_type"] == "RECOVER_REJECTED_JUMP"].copy()
recover_conflict_rejects = debug[debug["event_type"] == "RECOVER_REJECTED_LOCAL_ID_CONFLICT"].copy()
recover_ambiguous_rejects = debug[debug["event_type"] == "RECOVER_REJECTED_AMBIGUOUS"].copy()
new_track_blocked = debug[debug["event_type"] == "NEW_TRACK_BLOCKED_NEAR_EXISTING"].copy()
match_rejected = debug[debug["event_type"] == "MATCH_REJECTED"].copy()


# ============================================================
# HELPERS
# ============================================================

def normalize_camera_name(camera):
    cam = str(camera).lower()
    if "center" in cam or "centre" in cam:
        return "center"
    if "left" in cam:
        return "left"
    if "right" in cam:
        return "right"
    return "unknown"


def get_detection_style(camera):
    return DETECTION_STYLES.get(normalize_camera_name(camera), DETECTION_STYLES["unknown"])


def scatter_detection_group(ax, group, camera, size=LOCAL_DETECTION_MARKER_SIZE, alpha=0.78, zorder=3, label_prefix="det"):
    """Draw local camera detections as colored camera-observation symbols.

    For presentation clarity all local detections use the same circular marker,
    while the color and the letter identify the originating camera:
    L = left, C = center, R = right.
    """
    if len(group) == 0:
        return

    style = get_detection_style(camera)
    cam_name = normalize_camera_name(camera)
    letter = {"left": "L", "center": "C", "right": "R"}.get(cam_name, "?")

    ax.scatter(
        group["x"].to_numpy(),
        group["y"].to_numpy(),
        s=size,
        marker="o",
        c=style["color"],
        edgecolors=style["edgecolor"],
        linewidths=1.1,
        alpha=alpha,
        label=f"{label_prefix} {cam_name}",
        zorder=zorder,
    )

    # Small camera letter inside the detection marker. This avoids relying only on color.
    for _, row in group.iterrows():
        x = _safe_float(row.get("x", np.nan))
        y = _safe_float(row.get("y", np.nan))
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        ax.text(
            x,
            y,
            letter,
            ha="center",
            va="center",
            fontsize=LOCAL_DETECTION_LABEL_FONT_SIZE,
            fontweight="bold",
            color="black",
            zorder=zorder + 1,
        )


def _safe_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _gt_yaw_from_row(row):
    """Estimate ground-truth heading from velocity if available, otherwise yaw."""
    vx = _safe_float(row.get("vx", np.nan))
    vy = _safe_float(row.get("vy", np.nan))
    if np.isfinite(vx) and np.isfinite(vy) and np.hypot(vx, vy) >= GT_MIN_SPEED_FOR_ORIENTATION:
        return float(np.arctan2(vy, vx))

    yaw = _safe_float(row.get("yaw", np.nan))
    if np.isfinite(yaw):
        return float(yaw)

    return 0.0


def draw_gt_footprint(ax, x, y, yaw=0.0, color=GT_FOOTPRINT_COLOR, alpha=GT_FOOTPRINT_ALPHA,
                      zorder=6, label=None):
    """Draw a top-view pedestrian footprint using two small oriented ellipses."""
    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(yaw)):
        return

    forward = np.array([np.cos(yaw), np.sin(yaw)], dtype=float)
    lateral = np.array([-np.sin(yaw), np.cos(yaw)], dtype=float)

    # Two feet: slightly forward and separated laterally.
    centers = [
        np.array([x, y]) + GT_FOOTPRINT_FORWARD_OFFSET * forward + GT_FOOTPRINT_LATERAL_OFFSET * lateral,
        np.array([x, y]) - GT_FOOTPRINT_FORWARD_OFFSET * forward - GT_FOOTPRINT_LATERAL_OFFSET * lateral,
    ]

    for i, c in enumerate(centers):
        foot = Ellipse(
            xy=(float(c[0]), float(c[1])),
            width=GT_FOOTPRINT_WIDTH,
            height=GT_FOOTPRINT_LENGTH,
            angle=np.rad2deg(yaw),
            facecolor=color,
            edgecolor="black",
            linewidth=0.7,
            alpha=alpha,
            zorder=zorder,
            label=label if i == 0 else "_nolegend_",
        )
        ax.add_patch(foot)


def draw_gt_velocity_arrow(ax, row, color=GT_FOOTPRINT_COLOR, zorder=7):
    """Draw a small velocity arrow for a GT agent when vx/vy are available."""
    if not GT_SHOW_VELOCITY_ARROW:
        return
    x = _safe_float(row.get("x", np.nan))
    y = _safe_float(row.get("y", np.nan))
    vx = _safe_float(row.get("vx", np.nan))
    vy = _safe_float(row.get("vy", np.nan))
    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(vx) and np.isfinite(vy)):
        return
    speed = float(np.hypot(vx, vy))
    if speed < GT_MIN_SPEED_FOR_ORIENTATION:
        return

    # Normalize arrow length for readability, independent of absolute speed scale.
    dx = GT_VELOCITY_ARROW_SCALE * vx / speed
    dy = GT_VELOCITY_ARROW_SCALE * vy / speed
    ax.arrow(
        x, y, dx, dy,
        head_width=0.08,
        head_length=0.10,
        length_includes_head=True,
        color=color,
        linewidth=1.4,
        alpha=0.75,
        zorder=zorder,
    )


def draw_ground_truth_trails(ax, gt_window, frame_time, trail_seconds=GT_TRAIL_SECONDS,
                             color=GT_FOOTPRINT_COLOR, zorder=2):
    """Draw dashed recent trajectories of GT agents."""
    if gt_window is None or len(gt_window) == 0 or pd.isna(frame_time):
        return
    required = {"time", "agent_id", "x", "y"}
    if not required.issubset(gt_window.columns):
        return

    trail = gt_window[
        (gt_window["time"] >= float(frame_time) - float(trail_seconds))
        & (gt_window["time"] <= float(frame_time))
    ].copy()
    if len(trail) == 0:
        return

    for _, group in trail.groupby("agent_id"):
        group = group.sort_values("time")
        if len(group) < 2:
            continue
        ax.plot(
            group["x"].to_numpy(),
            group["y"].to_numpy(),
            linestyle="--",
            linewidth=GT_TRAIL_LINEWIDTH,
            color=color,
            alpha=0.28,
            zorder=zorder,
            label="GT trail" if _ == trail["agent_id"].iloc[0] else "_nolegend_",
        )


def scatter_ground_truth(ax, group, size=160, alpha=1.0, zorder=5, label=None):
    """Draw GT agents as footprints rather than stars.

    The function name is kept for compatibility with the rest of the script.
    """
    if len(group) == 0:
        return

    used_label = False
    for _, row in group.iterrows():
        x = _safe_float(row.get("x", np.nan))
        y = _safe_float(row.get("y", np.nan))
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        yaw = _gt_yaw_from_row(row)
        draw_gt_footprint(
            ax, x, y, yaw=yaw,
            color=GT_FOOTPRINT_COLOR,
            alpha=alpha,
            zorder=zorder,
            label=(label if label is not None else GROUND_TRUTH_STYLE["label"]) if not used_label else "_nolegend_",
        )
        used_label = True
        draw_gt_velocity_arrow(ax, row, color=GT_FOOTPRINT_COLOR, zorder=zorder + 1)

def _global_track_uncertainty_radius(row):
    """Small visual halo radius for the global track.

    This is not a formal covariance. It is a presentation-oriented cue: lower
    reliability and more missed cycles produce a larger dashed halo.
    """
    rel = _safe_float(row.get("reliability", np.nan))
    missed = _safe_float(row.get("missed", 0.0), default=0.0)
    if not np.isfinite(rel):
        rel = 0.5
    if not np.isfinite(missed):
        missed = 0.0
    rel = float(np.clip(rel, 0.0, 1.0))
    radius = GLOBAL_TRACK_HALO_BASE_RADIUS + (1.0 - rel) * GLOBAL_TRACK_HALO_MAX_EXTRA_RADIUS + missed * GLOBAL_TRACK_HALO_MISSED_GAIN
    return float(np.clip(radius, GLOBAL_TRACK_HALO_BASE_RADIUS, GLOBAL_TRACK_HALO_BASE_RADIUS + GLOBAL_TRACK_HALO_MAX_EXTRA_RADIUS + 0.25))


def draw_global_track_symbol(ax, row, size=135, alpha=1.0, zorder=5, label=None, publishable=True):
    """Draw a fused global track as a tracked-person symbol.

    Components:
    - dashed green halo: visual uncertainty / confidence cue;
    - filled or open green body marker: published vs internal track;
    """
    x = _safe_float(row.get("x", np.nan))
    y = _safe_float(row.get("y", np.nan))
    if not (np.isfinite(x) and np.isfinite(y)):
        return

    radius = _global_track_uncertainty_radius(row)
    halo = Ellipse(
        xy=(x, y),
        width=2.0 * radius,
        height=2.0 * radius,
        angle=0.0,
        facecolor="none",
        edgecolor=GLOBAL_TRACK_STYLE["color"],
        linewidth=1.2,
        linestyle="--",
        alpha=0.35 if publishable else 0.25,
        zorder=zorder - 1,
        label="uncertainty halo" if label == "published global track" else "_nolegend_",
    )
    ax.add_patch(halo)

    if publishable:
        ax.scatter(
            [x], [y],
            s=size,
            marker="o",
            c=GLOBAL_TRACK_STYLE["color"],
            edgecolors=GLOBAL_TRACK_STYLE["edgecolor"],
            linewidths=1.4,
            alpha=alpha,
            label=label,
            zorder=zorder,
        )
    else:
        ax.scatter(
            [x], [y],
            s=size * 0.82,
            marker="o",
            facecolors="none",
            edgecolors=UNPUBLISHED_TRACK_STYLE["edgecolor"],
            linewidths=1.8,
            alpha=0.85,
            label="_nolegend_",
            zorder=zorder,
        )



def scatter_global_tracks(ax, group, size=135, alpha=1.0, zorder=4, label=None, show_publish_state=True):
    if len(group) == 0:
        return

    label_used = False
    for _, row in group.iterrows():
        publishable = True
        if show_publish_state and "publishable" in row.index:
            publishable = bool(row.get("publishable", True))

        current_label = None
        if publishable and not label_used:
            current_label = label if label is not None else GLOBAL_TRACK_STYLE["label"]
            label_used = True

        draw_global_track_symbol(
            ax,
            row,
            size=size,
            alpha=alpha,
            zorder=zorder,
            label=current_label,
            publishable=publishable,
        )


def scatter_duplicate_dropped(ax, group, x_col="dropped_x", y_col="dropped_y", size=75, alpha=1.0, zorder=4, label=None):
    if len(group) == 0:
        return
    ax.scatter(
        group[x_col].to_numpy(), group[y_col].to_numpy(),
        s=size, marker=DUPLICATE_DROPPED_STYLE["marker"], c=DUPLICATE_DROPPED_STYLE["color"],
        linewidths=1.8, alpha=alpha,
        label=label if label is not None else DUPLICATE_DROPPED_STYLE["label"], zorder=zorder,
    )


def parse_published_ids(x):
    try:
        return json.loads(x)
    except Exception:
        return []


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()


def set_equal_xy(ax, xs, ys, margin=0.5):
    xs = pd.to_numeric(pd.Series(xs), errors="coerce").dropna()
    ys = pd.to_numeric(pd.Series(ys), errors="coerce").dropna()
    if len(xs) == 0 or len(ys) == 0:
        return
    ax.set_xlim(float(xs.min()) - margin, float(xs.max()) + margin)
    ax.set_ylim(float(ys.min()) - margin, float(ys.max()) + margin)
    ax.set_aspect("equal", adjustable="box")


def annotate_last_point(ax, group, text, fontsize=8):
    group = group.sort_values("time")
    if len(group) == 0:
        return
    last = group.iloc[-1]
    ax.text(float(last["x"]), float(last["y"]), text, fontsize=fontsize, fontweight="bold")


def get_gt_near_time(gt_window, frame_time, tolerance=0.20):
    if len(gt_window) == 0 or pd.isna(frame_time):
        return gt_window.iloc[0:0].copy()
    gt_tmp = gt_window.copy()
    gt_tmp["dt"] = (gt_tmp["time"] - frame_time).abs()
    gt_tmp = gt_tmp[gt_tmp["dt"] <= tolerance].copy()
    if len(gt_tmp) == 0:
        return gt_tmp
    return gt_tmp.sort_values("dt").groupby("agent_id", as_index=False).first()


def get_robot_pose_at_time(robot_df, frame_time, tolerance=0.25):
    if robot_df is None or len(robot_df) == 0 or "time" not in robot_df.columns or pd.isna(frame_time):
        return None
    times = robot_df["time"].to_numpy(dtype=float)
    if len(times) == 0 or not np.isfinite(times).any():
        return None
    idx = int(np.nanargmin(np.abs(times - float(frame_time))))
    pose = robot_df.iloc[idx]
    if pd.isna(pose.get("x", np.nan)) or pd.isna(pose.get("y", np.nan)) or pd.isna(pose.get("yaw", np.nan)):
        return None
    if abs(float(pose["time"]) - float(frame_time)) > tolerance:
        return None
    return pose


def draw_robot(ax, x, y, yaw, label="robot"):
    """Draw robot as a clean top-view dot plus heading arrow.

    This is more presentation-friendly than a triangle because the robot body
    position and the yaw direction are separated visually.
    """
    ax.scatter(
        [x], [y],
        s=ROBOT_MARKER_SIZE,
        marker=ROBOT_STYLE["marker"],
        c=ROBOT_STYLE["color"],
        edgecolors=ROBOT_STYLE["edgecolor"],
        linewidths=1.2,
        label=label,
        zorder=14,
    )
    dx = ROBOT_ARROW_LENGTH * np.cos(yaw)
    dy = ROBOT_ARROW_LENGTH * np.sin(yaw)
    ax.arrow(
        x, y, dx, dy,
        head_width=0.20,
        head_length=0.22,
        length_includes_head=True,
        color=ROBOT_STYLE["color"],
        linewidth=2.0,
        zorder=15,
    )


def draw_robot_path(ax, robot_df, frame_time, trail_seconds=ROBOT_PATH_TRAIL_SECONDS):
    """Draw the recent robot path as a thin grey trail."""
    if robot_df is None or len(robot_df) == 0 or "time" not in robot_df.columns:
        return
    if pd.isna(frame_time):
        return
    path = robot_df[
        (robot_df["time"] >= float(frame_time) - float(trail_seconds))
        & (robot_df["time"] <= float(frame_time))
    ].copy()
    if len(path) < 2 or "x" not in path.columns or "y" not in path.columns:
        return
    path = path.sort_values("time")
    ax.plot(
        path["x"].to_numpy(),
        path["y"].to_numpy(),
        color="gray",
        linewidth=1.6,
        alpha=0.45,
        label="robot path",
        zorder=0,
    )


def draw_camera_fov(ax, x, y, yaw, fov_deg=CAMERA_FOV_DEG, length=CAMERA_FOV_LENGTH, color="gray", label=None):
    """Draw a camera FOV as a lightly filled cone plus dashed boundaries."""
    half = np.deg2rad(fov_deg / 2.0)
    a1 = yaw - half
    a2 = yaw + half
    x1 = x + length * np.cos(a1)
    y1 = y + length * np.sin(a1)
    x2 = x + length * np.cos(a2)
    y2 = y + length * np.sin(a2)

    ax.fill(
        [x, x1, x2],
        [y, y1, y2],
        facecolor=color,
        edgecolor="none",
        alpha=0.10,
        zorder=0,
    )
    ax.plot([x, x1], [y, y1], linestyle="--", color=color, linewidth=1.5, alpha=0.75, label=label, zorder=1)
    ax.plot([x, x2], [y, y2], linestyle="--", color=color, linewidth=1.5, alpha=0.75, zorder=1)
    ax.plot([x1, x2], [y1, y2], linestyle="--", color=color, linewidth=1.2, alpha=0.55, zorder=1)


def draw_robot_and_fovs(ax, robot_df, frame_time, draw_labels=True):
    if not SHOW_ROBOT_AND_FOV:
        return
    pose = get_robot_pose_at_time(robot_df, frame_time, tolerance=ROBOT_POSE_TIME_TOLERANCE)
    if pose is None:
        return
    rx = float(pose["x"])
    ry = float(pose["y"])
    ryaw = float(pose["yaw"])

    draw_robot_path(ax, robot_df, frame_time, trail_seconds=ROBOT_PATH_TRAIL_SECONDS)

    for cam_name, offset in [
        ("center", CENTER_CAM_YAW_OFFSET),
        ("left", LEFT_CAM_YAW_OFFSET),
        ("right", RIGHT_CAM_YAW_OFFSET),
    ]:
        style = FOV_STYLES[cam_name]
        draw_camera_fov(ax, rx, ry, ryaw + offset,
                        fov_deg=CAMERA_FOV_DEG, length=CAMERA_FOV_LENGTH,
                        color=style["color"], label=style["label"] if draw_labels else None)

    draw_robot(ax, rx, ry, ryaw, label="robot" if draw_labels else None)


def get_robot_window_for_time_range(robot_df, start_time, end_time):
    if robot_df is None or len(robot_df) == 0 or "time" not in robot_df.columns:
        return pd.DataFrame(columns=["x", "y", "yaw", "time"])
    if pd.isna(start_time) or pd.isna(end_time):
        return robot_df.iloc[0:0].copy()
    margin = max(ROBOT_POSE_TIME_TOLERANCE, 0.5)
    return robot_df[(robot_df["time"] >= float(start_time) - margin) & (robot_df["time"] <= float(end_time) + margin)].copy()


def robot_fov_limit_points(robot_window):
    if robot_window is None or len(robot_window) == 0:
        return pd.DataFrame(columns=["x", "y"])
    rows = []
    for _, pose in robot_window.iterrows():
        if pd.isna(pose.get("x", np.nan)) or pd.isna(pose.get("y", np.nan)) or pd.isna(pose.get("yaw", np.nan)):
            continue
        rx = float(pose["x"])
        ry = float(pose["y"])
        ryaw = float(pose["yaw"])
        rows.append({"x": rx, "y": ry})
        for offset in [CENTER_CAM_YAW_OFFSET, LEFT_CAM_YAW_OFFSET, RIGHT_CAM_YAW_OFFSET]:
            half = np.deg2rad(CAMERA_FOV_DEG / 2.0)
            for a in [ryaw + offset - half, ryaw + offset + half]:
                rows.append({"x": rx + CAMERA_FOV_LENGTH * np.cos(a), "y": ry + CAMERA_FOV_LENGTH * np.sin(a)})
    return pd.DataFrame(rows)


def get_time_range_from_dfs(*dfs):
    t_min_candidates = []
    t_max_candidates = []
    for df in dfs:
        if df is None or len(df) == 0 or "time" not in df.columns:
            continue
        vals = pd.to_numeric(df["time"], errors="coerce").dropna()
        vals = vals[vals > 0.0]
        if len(vals) == 0:
            continue
        t_min_candidates.append(float(vals.min()))
        t_max_candidates.append(float(vals.max()))
    if len(t_min_candidates) == 0 or len(t_max_candidates) == 0:
        return np.nan, np.nan
    return min(t_min_candidates), max(t_max_candidates)


def compute_window_limits(*dfs, margin=0.8):
    xs = []
    ys = []
    for df in dfs:
        if df is None or len(df) == 0:
            continue
        for x_col, y_col in [
            ("x", "y"), ("pred_x", "pred_y"), ("det_x", "det_y"),
            ("kept_x", "kept_y"), ("dropped_x", "dropped_y"), ("track_x", "track_y"),
        ]:
            if x_col in df.columns and y_col in df.columns:
                xs.extend(pd.to_numeric(df[x_col], errors="coerce").dropna().tolist())
                ys.extend(pd.to_numeric(df[y_col], errors="coerce").dropna().tolist())
    if len(xs) == 0 or len(ys) == 0:
        return (-1, 1), (-1, 1)
    return (float(np.nanmin(xs)) - margin, float(np.nanmax(xs)) + margin), (float(np.nanmin(ys)) - margin, float(np.nanmax(ys)) + margin)


def slice_time_window(df, start_time, end_time, include_end=True):
    if len(df) == 0 or "time" not in df.columns:
        return df.iloc[0:0].copy()
    if include_end:
        return df[(df["time"] >= start_time) & (df["time"] <= end_time)].copy()
    return df[(df["time"] >= start_time) & (df["time"] < end_time)].copy()


def deduplicate_legend(ax, fontsize=8, loc="best", bbox_to_anchor=None):
    handles, labels = ax.get_legend_handles_labels()
    valid = []
    for h, lab in zip(handles, labels):
        if lab is None:
            continue
        lab = str(lab)
        if lab == "" or lab.startswith("_"):
            continue
        if lab in {"internal/unpublished track", "internal / not published", "association link", "duplicate link"}:
            continue
        valid.append((h, lab))
    if len(valid) == 0:
        return
    unique = {}
    for h, lab in valid:
        if lab not in unique:
            unique[lab] = h
    if bbox_to_anchor is None:
        ax.legend(unique.values(), unique.keys(), fontsize=fontsize, loc=loc)
    else:
        ax.legend(
            unique.values(),
            unique.keys(),
            fontsize=fontsize,
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            borderaxespad=0.0,
            frameon=True,
        )


def add_fixed_xy_legend_outside(ax, fontsize=7):
    handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=GT_FOOTPRINT_COLOR, markeredgecolor="black",
               markersize=11, linestyle="None", label="GT agent footprints"),
        Line2D([0], [0], marker=DETECTION_STYLES["center"]["marker"], color="none",
               markerfacecolor=DETECTION_STYLES["center"]["color"], markeredgecolor=DETECTION_STYLES["center"]["edgecolor"],
               markersize=8, linestyle="None", label="det center"),
        Line2D([0], [0], marker=DETECTION_STYLES["left"]["marker"], color="none",
               markerfacecolor=DETECTION_STYLES["left"]["color"], markeredgecolor=DETECTION_STYLES["left"]["edgecolor"],
               markersize=8, linestyle="None", label="det left"),
        Line2D([0], [0], marker=DETECTION_STYLES["right"]["marker"], color="none",
               markerfacecolor=DETECTION_STYLES["right"]["color"], markeredgecolor=DETECTION_STYLES["right"]["edgecolor"],
               markersize=8, linestyle="None", label="det right"),
        Line2D([0], [0], marker=GLOBAL_TRACK_STYLE["marker"], color="none",
               markerfacecolor=GLOBAL_TRACK_STYLE["color"], markeredgecolor=GLOBAL_TRACK_STYLE["edgecolor"],
               markersize=10, linestyle="None", label="global track + uncertainty"),
        Line2D([0], [0], marker=DUPLICATE_DROPPED_STYLE["marker"], color=DUPLICATE_DROPPED_STYLE["color"],
               markersize=8, linestyle="None", label="dropped duplicate"),
        Line2D([0], [0], marker=ROBOT_STYLE["marker"], color="none",
               markerfacecolor=ROBOT_STYLE["color"], markeredgecolor=ROBOT_STYLE["edgecolor"],
               markersize=10, linestyle="None", label="robot"),
        Line2D([0], [0], color="gray", linewidth=1.6, linestyle="-", alpha=0.6, label="robot path"),
        Line2D([0], [0], color=FOV_STYLES["center"]["color"], linewidth=1.4, linestyle="--", label="FOV center 69°"),
        Line2D([0], [0], color=FOV_STYLES["left"]["color"], linewidth=1.4, linestyle="--", label="FOV left 69°"),
        Line2D([0], [0], color=FOV_STYLES["right"]["color"], linewidth=1.4, linestyle="--", label="FOV right 69°"),
    ]
    ax.legend(handles=handles, fontsize=fontsize, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=True)


def plot_cycle_associations(matches_window, out_dir, gt_window=None):
    """
    Generate one static association plot for each fusion cycle in a temporal window.

    In each cycle plot it shows:
    - predicted global track position used for the accepted association;
    - matched local detection, with the camera-specific marker/color;
    - association segment between prediction and detection;
    - ground-truth agents close to the current cycle time.

    Ground truth is selected with GT_TIME_TOLERANCE using get_gt_near_time().
    This means that for every frame/cycle we show the nearest GT sample for each
    agent, if its timestamp is close enough to the fusion cycle time.
    """

    out_dir.mkdir(exist_ok=True)

    if gt_window is None:
        gt_window = pd.DataFrame(columns=["time", "agent_id", "x", "y"])

    if len(matches_window) == 0:
        return 0

    num_plots = 0

    for cycle, group in matches_window.groupby("cycle"):
        group = group.sort_values("time")
        frame_time = float(group["time"].iloc[0])

        gt_now = get_gt_near_time(
            gt_window,
            frame_time,
            tolerance=GT_TIME_TOLERANCE,
        )

        fig, ax = plt.subplots(figsize=(11, 8))
        fig.subplots_adjust(right=0.72)

        # ----------------------------------------------------
        # Ground truth near current cycle time
        # ----------------------------------------------------
        draw_ground_truth_trails(ax, gt_window, frame_time, trail_seconds=GT_TRAIL_SECONDS)

        if len(gt_now) > 0:
            scatter_ground_truth(
                ax,
                gt_now,
                size=160,
                alpha=0.95,
                zorder=6,
                label="ground truth",
            )

            for _, gt_row in gt_now.iterrows():
                agent_id = gt_row.get("agent_id", "")
                ax.text(
                    float(gt_row["x"]) + 0.12,
                    float(gt_row["y"]) + 0.12,
                    f"GT {agent_id}",
                    fontsize=8,
                    fontweight="bold",
                    zorder=7,
                )

        # ----------------------------------------------------
        # Predicted global track positions
        # ----------------------------------------------------
        ax.scatter(
            group["pred_x"].to_numpy(),
            group["pred_y"].to_numpy(),
            s=95,
            marker=GLOBAL_TRACK_STYLE["marker"],
            c=GLOBAL_TRACK_STYLE["color"],
            edgecolors=GLOBAL_TRACK_STYLE["edgecolor"],
            linewidths=0.8,
            label="predicted global track",
            zorder=4,
        )

        # ----------------------------------------------------
        # Matched local detections, using camera-specific style
        # ----------------------------------------------------
        if "camera" in group.columns:
            for cam, cam_group in group.groupby("camera"):
                style = get_detection_style(cam)
                cam_name = normalize_camera_name(cam)

                ax.scatter(
                    cam_group["det_x"].to_numpy(),
                    cam_group["det_y"].to_numpy(),
                    s=90,
                    marker=style["marker"],
                    c=style["color"],
                    edgecolors=style["edgecolor"],
                    linewidths=0.8,
                    label=f"matched local detection {cam_name}",
                    zorder=3,
                )

        # ----------------------------------------------------
        # Association links and labels
        # ----------------------------------------------------
        for _, row in group.iterrows():
            if (
                pd.isna(row["pred_x"])
                or pd.isna(row["pred_y"])
                or pd.isna(row["det_x"])
                or pd.isna(row["det_y"])
            ):
                continue

            ax.plot(
                [float(row["pred_x"]), float(row["det_x"])],
                [float(row["pred_y"]), float(row["det_y"])],
                linewidth=2,
                color=ASSOCIATION_LINE_COLOR,
                alpha=0.75,
                zorder=1,
            )

            gid = int(row["global_id"]) if pd.notna(row["global_id"]) else -1
            cam = str(row["camera"]) if "camera" in row and pd.notna(row["camera"]) else "unknown"
            dist = float(row["distance"]) if pd.notna(row["distance"]) else np.nan
            relation = str(row.get("local_id_relation", "none"))
            local_id = str(row.get("local_id", ""))

            ax.text(
                float(row["det_x"]),
                float(row["det_y"]),
                f"T{gid} | {normalize_camera_name(cam)} | {relation} | {local_id} | d={dist:.2f}m",
                fontsize=7,
                fontweight="bold",
                zorder=7,
            )

        ax.set_title(
            f"Accepted associations + GT | cycle {int(cycle)} | t={frame_time:.2f}s"
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.5)
        # Legend outside the plot area, so labels and trajectories remain readable.
        deduplicate_legend(
            ax,
            fontsize=8,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
        )

        set_equal_xy(
            ax,
            list(group["pred_x"].to_numpy())
            + list(group["det_x"].to_numpy())
            + list(gt_now["x"].to_numpy()),
            list(group["pred_y"].to_numpy())
            + list(group["det_y"].to_numpy())
            + list(gt_now["y"].to_numpy()),
        )

        savefig(out_dir / f"cycle_{int(cycle):04d}_associations.png")
        num_plots += 1

    return num_plots


def create_cycle_associations_animation(matches_window, gt_window, output_file, fps=1, draw_robot_fov=False):
    """
    Create a GIF for the accepted associations inside one temporal window.

    If draw_robot_fov=True, the GIF also shows robot pose and the three camera FOVs.
    """
    if matches_window is None or len(matches_window) == 0 or "cycle" not in matches_window.columns:
        return False

    cycles = sorted(matches_window["cycle"].dropna().unique())
    if len(cycles) == 0:
        return False

    if gt_window is None:
        gt_window = pd.DataFrame(columns=["time", "agent_id", "x", "y"])

    t_min, t_max = get_time_range_from_dfs(matches_window, gt_window)
    robot_window = get_robot_window_for_time_range(robot, t_min, t_max) if draw_robot_fov else pd.DataFrame(columns=["x", "y", "yaw", "time"])
    robot_fov_points = robot_fov_limit_points(robot_window) if draw_robot_fov else pd.DataFrame(columns=["x", "y"])
    xlim, ylim = compute_window_limits(matches_window, gt_window, robot_window, robot_fov_points, margin=0.8)

    fig, ax = plt.subplots(figsize=(13, 8.5))
    fig.subplots_adjust(right=0.72)

    def update(frame_idx):
        cycle = cycles[frame_idx]
        ax.clear()

        group = matches_window[matches_window["cycle"] == cycle].copy().sort_values("time")
        if len(group) == 0:
            return

        frame_time = float(group["time"].iloc[0])
        gt_now = get_gt_near_time(gt_window, frame_time, tolerance=GT_TIME_TOLERANCE)

        if draw_robot_fov:
            draw_robot_and_fovs(ax, robot_window, frame_time, draw_labels=True)

        draw_ground_truth_trails(ax, gt_window, frame_time, trail_seconds=GT_TRAIL_SECONDS)

        if len(gt_now) > 0:
            scatter_ground_truth(
                ax, gt_now, size=170, alpha=0.95,
                zorder=6, label="ground truth",
            )
            for _, gt_row in gt_now.iterrows():
                agent_id = gt_row.get("agent_id", "")
                ax.text(
                    float(gt_row["x"]) + 0.12, float(gt_row["y"]) + 0.12,
                    f"GT {agent_id}", fontsize=8, fontweight="bold", zorder=7,
                )

        ax.scatter(
            group["pred_x"].to_numpy(), group["pred_y"].to_numpy(),
            s=105, marker=GLOBAL_TRACK_STYLE["marker"],
            c=GLOBAL_TRACK_STYLE["color"], edgecolors=GLOBAL_TRACK_STYLE["edgecolor"],
            linewidths=0.8, label="predicted global track", zorder=4,
        )

        if "camera" in group.columns:
            for cam, cam_group in group.groupby("camera"):
                style = get_detection_style(cam)
                cam_name = normalize_camera_name(cam)
                ax.scatter(
                    cam_group["det_x"].to_numpy(), cam_group["det_y"].to_numpy(),
                    s=95, marker=style["marker"], c=style["color"],
                    edgecolors=style["edgecolor"], linewidths=0.8,
                    label=f"matched local detection {cam_name}", zorder=3,
                )

        for _, row in group.iterrows():
            if (
                pd.isna(row.get("pred_x", np.nan))
                or pd.isna(row.get("pred_y", np.nan))
                or pd.isna(row.get("det_x", np.nan))
                or pd.isna(row.get("det_y", np.nan))
            ):
                continue

            ax.plot(
                [float(row["pred_x"]), float(row["det_x"])],
                [float(row["pred_y"]), float(row["det_y"])],
                linewidth=2, color=ASSOCIATION_LINE_COLOR,
                alpha=0.75, zorder=1,
            )

            gid = int(row["global_id"]) if pd.notna(row.get("global_id", np.nan)) else -1
            cam = str(row.get("camera", "unknown"))
            dist = float(row.get("distance", np.nan)) if pd.notna(row.get("distance", np.nan)) else np.nan
            relation = str(row.get("local_id_relation", "none"))
            local_id = str(row.get("local_id", ""))
            ax.text(
                float(row["det_x"]), float(row["det_y"]),
                f"T{gid} | {normalize_camera_name(cam)} | {relation} | {local_id} | d={dist:.2f}m",
                fontsize=7, fontweight="bold", zorder=7,
            )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        suffix = " + robot/FOV" if draw_robot_fov else ""
        ax.set_title(f"Accepted associations + GT{suffix} | cycle {int(cycle)} | t={frame_time:.2f}s")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.5)
        deduplicate_legend(
            ax, fontsize=8, loc="center left",
            bbox_to_anchor=(1.02, 0.5),
        )

    animation = FuncAnimation(fig, update, frames=len(cycles), interval=int(1000 / fps), repeat=True)
    animation.save(output_file, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return True

def create_matplotlib_fusion_animation(
    tracks_window,
    detections_window,
    matches_window,
    duplicates_window,
    gt_window,
    output_file,
    fps=2,
    trail_seconds=2.0,
    show_publish_state=True,
    title_suffix="* = internal not published",
    draw_associations=True,
):
    if len(matches_window) > 0 and "cycle" in matches_window.columns:
        cycles = sorted(matches_window["cycle"].dropna().unique())
    elif len(tracks_window) > 0 and "cycle" in tracks_window.columns:
        cycles = sorted(tracks_window["cycle"].dropna().unique())
    elif len(detections_window) > 0 and "cycle" in detections_window.columns:
        cycles = sorted(detections_window["cycle"].dropna().unique())
    else:
        return False
    if len(cycles) == 0:
        return False

    t_min, t_max = get_time_range_from_dfs(tracks_window, detections_window, matches_window, duplicates_window, gt_window)
    robot_window = get_robot_window_for_time_range(robot, t_min, t_max)
    robot_fov_points = robot_fov_limit_points(robot_window)
    xlim, ylim = compute_window_limits(tracks_window, detections_window, matches_window, duplicates_window, gt_window,
                                       robot_window, robot_fov_points, margin=0.8)
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.subplots_adjust(right=0.72)

    def update(frame_idx):
        cycle = cycles[frame_idx]
        ax.clear()
        if len(matches_window[matches_window["cycle"] == cycle]) > 0:
            frame_time = float(matches_window[matches_window["cycle"] == cycle]["time"].iloc[0])
        elif len(tracks_window[tracks_window["cycle"] == cycle]) > 0:
            frame_time = float(tracks_window[tracks_window["cycle"] == cycle]["time"].iloc[0])
        elif len(detections_window[detections_window["cycle"] == cycle]) > 0:
            frame_time = float(detections_window[detections_window["cycle"] == cycle]["time"].iloc[0])
        else:
            frame_time = np.nan

        tw_now = tracks_window[tracks_window["cycle"] == cycle].copy()
        dw_now = detections_window[detections_window["cycle"] == cycle].copy()
        mw_now = matches_window[matches_window["cycle"] == cycle].copy()
        dup_now = duplicates_window[duplicates_window["cycle"] == cycle].copy()
        tw_trail = tracks_window[(tracks_window["time"] >= frame_time - trail_seconds) & (tracks_window["time"] <= frame_time)].copy()
        gt_now = get_gt_near_time(gt_window, frame_time, tolerance=GT_TIME_TOLERANCE)

        draw_robot_and_fovs(ax, robot_window, frame_time, draw_labels=True)
        draw_ground_truth_trails(ax, gt_window, frame_time, trail_seconds=GT_TRAIL_SECONDS)
        scatter_ground_truth(ax, gt_now, size=170, alpha=1.0, zorder=6, label="GT agent footprints")
        for _, row in gt_now.iterrows():
            ax.text(float(row["x"]) + 0.12, float(row["y"]) + 0.12, f"GT {row['agent_id']}", fontsize=8, fontweight="bold", zorder=7)

        if len(dw_now) > 0 and "camera" in dw_now.columns:
            for cam, group in dw_now.groupby("camera"):
                scatter_detection_group(ax, group, cam, size=LOCAL_DETECTION_MARKER_SIZE, alpha=0.78, zorder=3, label_prefix="det")

        for tid, group in tw_trail.groupby("global_id"):
            group = group.sort_values("time")
            if len(group) < 2:
                continue
            ax.plot(group["x"].to_numpy(), group["y"].to_numpy(), linewidth=1.5,
                    color=GLOBAL_TRACK_STYLE["color"], alpha=0.35, zorder=2)

        scatter_global_tracks(ax, tw_now, size=145, alpha=1.0, zorder=5,
                              label="published global track", show_publish_state=show_publish_state)

        for _, row in tw_now.iterrows():
            gid = int(row["global_id"]) if pd.notna(row["global_id"]) else -1
            rel = float(row["reliability"]) if "reliability" in row and pd.notna(row["reliability"]) else np.nan
            hits = int(row["hits"]) if "hits" in row and pd.notna(row["hits"]) else -1
            missed = int(row["missed"]) if "missed" in row and pd.notna(row["missed"]) else -1
            pub = bool(row["publishable"]) if "publishable" in row else True
            label = f"T{gid}\nr={rel:.2f}\nh={hits} m={missed}"
            if show_publish_state and not pub:
                label = f"T{gid}*\nr={rel:.2f}\nh={hits} m={missed}"
            ax.text(float(row["x"]), float(row["y"]), label, fontsize=7, fontweight="bold", zorder=7)

        if draw_associations:
            for _, row in mw_now.iterrows():
                if pd.isna(row["pred_x"]) or pd.isna(row["pred_y"]) or pd.isna(row["det_x"]) or pd.isna(row["det_y"]):
                    continue
                ax.plot([float(row["pred_x"]), float(row["det_x"])],
                        [float(row["pred_y"]), float(row["det_y"])],
                        linewidth=2.0, linestyle="-", color=ASSOCIATION_LINE_COLOR, alpha=0.65, zorder=1)
                gid = int(row["global_id"]) if pd.notna(row["global_id"]) else -1
                dist = float(row["distance"]) if pd.notna(row["distance"]) else np.nan
                relation = str(row.get("local_id_relation", ""))
                ax.text(float(row["det_x"]), float(row["det_y"]), f"T{gid} {relation} d={dist:.2f}", fontsize=7, zorder=7)

        scatter_duplicate_dropped(ax, dup_now, x_col="dropped_x", y_col="dropped_y", size=90, alpha=1.0, zorder=6, label="dropped duplicate")
        for _, row in dup_now.iterrows():
            if pd.notna(row["kept_x"]) and pd.notna(row["dropped_x"]):
                ax.plot([float(row["kept_x"]), float(row["dropped_x"])],
                        [float(row["kept_y"]), float(row["dropped_y"])],
                        linestyle="--", linewidth=1.0, color=DUPLICATE_LINK_COLOR, alpha=0.55, zorder=1)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        title = f"Fusion animation | cycle {int(cycle)} | t={frame_time:.2f}s"
        if title_suffix:
            title += f" | {title_suffix}"
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.4)
        add_fixed_xy_legend_outside(ax, fontsize=7)

    animation = FuncAnimation(fig, update, frames=len(cycles), interval=int(1000 / fps), repeat=True)
    animation.save(output_file, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return True


def create_published_only_animation(published_tracks_window, detections_window, matches_window, duplicates_window, gt_window,
                                    output_file, fps=2, trail_seconds=2.0):
    return create_matplotlib_fusion_animation(
        published_tracks_window, detections_window, matches_window, duplicates_window, gt_window,
        output_file, fps=fps, trail_seconds=trail_seconds,
        show_publish_state=False, title_suffix="published tracks only", draw_associations=True,
    )


def create_lifecycle_timeline_animation(tracks_window, output_file, fps=2):
    if len(tracks_window) == 0 or "cycle" not in tracks_window.columns:
        return False
    cycles = sorted(tracks_window["cycle"].dropna().unique())
    ids = sorted(tracks_window["global_id"].dropna().unique())
    if len(cycles) == 0 or len(ids) == 0:
        return False
    id_to_y = {tid: i for i, tid in enumerate(ids)}
    t_min = float(tracks_window["time"].min())
    t_max = float(tracks_window["time"].max())
    fig, ax = plt.subplots(figsize=(14, max(6, 0.45 * len(ids) + 3)))
    fig.subplots_adjust(right=0.78)

    def update(frame_idx):
        cycle = cycles[frame_idx]
        ax.clear()
        tw_now = tracks_window[tracks_window["cycle"] == cycle].copy()
        frame_time = float(tw_now["time"].iloc[0]) if len(tw_now) > 0 else float(tracks_window[tracks_window["cycle"] <= cycle]["time"].max())
        tw_past = tracks_window[tracks_window["time"] <= frame_time].copy()
        labels_used = set()
        for tid in ids:
            g = tw_past[tw_past["global_id"] == tid].sort_values("time")
            if len(g) == 0:
                continue
            y = id_to_y[tid]
            ax.plot(g["time"].to_numpy(), np.full(len(g), y), linewidth=1.0, alpha=0.35, color="black", zorder=1)
            pub = g[g["publishable"] == True] if "publishable" in g.columns else g
            unpub = g[g["publishable"] == False] if "publishable" in g.columns else g.iloc[0:0]
            if len(pub) > 0:
                lab = "published" if "published" not in labels_used else None
                labels_used.add("published")
                ax.scatter(pub["time"].to_numpy(), np.full(len(pub), y), s=30, marker="s", color="green", label=lab, zorder=3)
            if len(unpub) > 0:
                lab = "internal / not published" if "internal / not published" not in labels_used else None
                labels_used.add("internal / not published")
                ax.scatter(unpub["time"].to_numpy(), np.full(len(unpub), y), s=30, marker="s", facecolors="none", edgecolors="orange", linewidths=1.3, label=lab, zorder=4)
            if "missed" in g.columns:
                missed_g = g[g["missed"] > 0]
                if len(missed_g) > 0:
                    lab = "missed > 0" if "missed > 0" not in labels_used else None
                    labels_used.add("missed > 0")
                    ax.scatter(missed_g["time"].to_numpy(), np.full(len(missed_g), y), s=42, marker="x", color="gray", linewidths=1.5, label=lab, zorder=5)
        for source_df, marker, size, color, label in [
            (new_tracks, "*", 100, "blue", "new track"),
            (reactivations, "D", 60, "cyan", "reactivated"),
            (deleted_tracks, "X", 75, "red", "deleted"),
        ]:
            if len(source_df) == 0 or "time" not in source_df.columns or "global_id" not in source_df.columns:
                continue
            ev = source_df[(source_df["time"] >= t_min) & (source_df["time"] <= frame_time) & (source_df["global_id"].isin(ids))].copy()
            for _, row in ev.iterrows():
                gid = row.get("global_id", np.nan)
                event_time = row.get("time", np.nan)
                if pd.isna(gid) or pd.isna(event_time) or gid not in id_to_y:
                    continue
                lab = label if label not in labels_used else None
                labels_used.add(label)
                ax.scatter(float(event_time), id_to_y[gid], marker=marker, s=size, color=color, label=lab, zorder=6)
        ax.axvline(frame_time, color="black", linewidth=1.5, linestyle="--", label="current time" if "current time" not in labels_used else None, zorder=2)
        ax.set_xlim(t_min, t_max)
        ax.set_yticks(list(id_to_y.values()))
        ax.set_yticklabels([f"T{int(tid)}" for tid in ids])
        ax.set_xlabel("time [s]")
        ax.set_ylabel("global ID")
        ax.set_title(f"Track lifecycle timeline | cycle {int(cycle)} | t={frame_time:.2f}s")
        ax.grid(True, alpha=0.4)
        deduplicate_legend(ax, fontsize=8, loc="center left")

    animation = FuncAnimation(fig, update, frames=len(cycles), interval=int(1000 / fps), repeat=True)
    animation.save(output_file, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return True


def compute_nearest_detection_distances(tracks_window, detections_window, matches_window):
    rows = []
    if len(tracks_window) == 0 or len(detections_window) == 0:
        return pd.DataFrame(columns=["time", "cycle", "global_id", "track_x", "track_y", "det_x", "det_y", "camera", "distance", "dynamic_threshold"])
    threshold_lookup = {}
    if len(matches_window) > 0 and {"cycle", "global_id", "dynamic_threshold"}.issubset(matches_window.columns):
        tmp = matches_window.dropna(subset=["cycle", "global_id", "dynamic_threshold"]).copy()
        for (cy, gid), g in tmp.groupby(["cycle", "global_id"]):
            threshold_lookup[(cy, gid)] = float(g["dynamic_threshold"].max())
    for _, tr in tracks_window.iterrows():
        cycle = tr.get("cycle", np.nan)
        gid = tr.get("global_id", np.nan)
        tx = tr.get("x", np.nan)
        ty = tr.get("y", np.nan)
        t = tr.get("time", np.nan)
        if pd.isna(cycle) or pd.isna(gid) or pd.isna(tx) or pd.isna(ty) or pd.isna(t):
            continue
        dets = detections_window[detections_window["cycle"] == cycle].copy().dropna(subset=["x", "y"])
        if len(dets) == 0:
            continue
        dx = dets["x"].to_numpy(dtype=float) - float(tx)
        dy = dets["y"].to_numpy(dtype=float) - float(ty)
        distances = np.sqrt(dx * dx + dy * dy)
        if len(distances) == 0 or not np.isfinite(distances).any():
            continue
        idx = int(np.nanargmin(distances))
        det = dets.iloc[idx]
        rows.append({
            "time": float(t), "cycle": cycle, "global_id": gid,
            "track_x": float(tx), "track_y": float(ty),
            "det_x": float(det["x"]), "det_y": float(det["y"]),
            "camera": det.get("camera", "unknown"),
            "distance": float(distances[idx]),
            "dynamic_threshold": threshold_lookup.get((cycle, gid), np.nan),
        })
    return pd.DataFrame(rows)


def create_nearest_detection_distance_animation(tracks_window, detections_window, matches_window, output_file, fps=2):
    nearest = compute_nearest_detection_distances(tracks_window, detections_window, matches_window)
    if len(nearest) == 0 or "cycle" not in nearest.columns:
        return False
    cycles = sorted(nearest["cycle"].dropna().unique())
    if len(cycles) == 0:
        return False
    t_min = float(nearest["time"].min())
    t_max = float(nearest["time"].max())
    max_distance = float(nearest["distance"].max())
    finite_thresholds = nearest["dynamic_threshold"].dropna()
    max_threshold = float(finite_thresholds.max()) if len(finite_thresholds) > 0 else 0.0
    y_max = max(0.5, 1.15 * max(max_distance, max_threshold))
    ids = sorted(nearest["global_id"].dropna().unique())
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.subplots_adjust(right=0.78)

    def update(frame_idx):
        cycle = cycles[frame_idx]
        ax.clear()
        now_rows = nearest[nearest["cycle"] == cycle].copy()
        if len(now_rows) == 0:
            return
        frame_time = float(now_rows["time"].max())
        past = nearest[nearest["time"] <= frame_time].copy()
        for gid in ids:
            g = past[past["global_id"] == gid].sort_values("time")
            if len(g) == 0:
                continue
            ax.plot(g["time"].to_numpy(), g["distance"].to_numpy(), marker="o", markersize=3, linewidth=1.4, label=f"T{int(gid)} nearest detection distance")
            g_now = now_rows[now_rows["global_id"] == gid]
            if len(g_now) > 0:
                r = g_now.iloc[0]
                ax.scatter([float(r["time"])], [float(r["distance"])], s=60, marker="o", edgecolors="black", linewidths=0.8, zorder=5)
        current_thresholds = now_rows["dynamic_threshold"].dropna()
        if len(current_thresholds) > 0:
            threshold = float(current_thresholds.median())
        elif len(finite_thresholds) > 0:
            threshold = float(finite_thresholds.median())
        else:
            threshold = np.nan
        if np.isfinite(threshold):
            ax.axhline(threshold, color="black", linestyle="--", linewidth=1.8, label=f"dynamic threshold = {threshold:.2f} m")
        ax.axvline(frame_time, color="gray", linestyle=":", linewidth=1.4, label="current time")
        ax.set_xlim(t_min, t_max)
        ax.set_ylim(0.0, y_max)
        ax.set_title(f"Nearest local detection distance | cycle {int(cycle)} | t={frame_time:.2f}s")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("distance track - nearest detection [m]")
        ax.grid(True, alpha=0.4)
        deduplicate_legend(ax, fontsize=7, loc="center left")

    animation = FuncAnimation(fig, update, frames=len(cycles), interval=int(1000 / fps), repeat=True)
    animation.save(output_file, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return True


# ============================================================
# REMAINING GLOBAL PLOTS
# ============================================================

# Plot kept: local detections by camera.
det_counts = detections.groupby(["time", "camera"]).size().reset_index(name="count")
plt.figure(figsize=(12, 5))
for cam, group in det_counts.groupby("camera"):
    group = group.sort_values("time")
    style = get_detection_style(cam)
    cam_name = normalize_camera_name(cam)
    plt.plot(group["time"].to_numpy(), group["count"].to_numpy(), marker=style["marker"], color=style["color"],
             markeredgecolor=style["edgecolor"], markersize=4, label=cam_name)
plt.title("Local detections per camera")
plt.xlabel("Time [s]")
plt.ylabel("Number of detections")
plt.grid(True, alpha=0.5)
plt.legend(title="Camera")
savefig(OUT_DIR / "local_detections_by_camera.png")

# Plot kept: accepted/rejected distances and dynamic threshold.
plt.figure(figsize=(12, 5))
if len(matches) > 0:
    plt.scatter(matches["time"].to_numpy(), matches["distance"].to_numpy(), s=15, label="accepted match distance")
    plt.scatter(matches["time"].to_numpy(), matches["dynamic_threshold"].to_numpy(), s=15, label="accepted dynamic threshold")
if len(jump_rejects) > 0:
    plt.scatter(jump_rejects["time"].to_numpy(), jump_rejects["distance"].to_numpy(), marker="x", s=35, label="jump reject")
if len(conflict_rejects) > 0:
    plt.scatter(conflict_rejects["time"].to_numpy(), conflict_rejects["distance"].to_numpy(), marker="^", s=35, label="local-id conflict reject")
if len(match_rejected) > 0:
    plt.scatter(match_rejected["time"].to_numpy(), match_rejected["distance"].to_numpy(), marker="v", s=25, label="ambiguous/masked reject")
plt.title("Match distance, dynamic threshold, and rejected candidates")
plt.xlabel("Time [s]")
plt.ylabel("Distance [m]")
plt.grid(True, alpha=0.5)
plt.legend()
savefig(OUT_DIR / "match_distance_vs_threshold_and_rejects.png")

# Plot kept: accepted match distance by local-id relation.
plt.figure(figsize=(12, 5))
if len(matches) > 0 and "local_id_relation" in matches.columns:
    for relation, group in matches.groupby("local_id_relation"):
        plt.scatter(group["time"].to_numpy(), group["distance"].to_numpy(), s=18, label=str(relation))
plt.title("Accepted match distance by local-id relation")
plt.xlabel("Time [s]")
plt.ylabel("Distance [m]")
plt.grid(True, alpha=0.5)
plt.legend(title="local_id_relation")
savefig(OUT_DIR / "match_distance_by_local_id_relation.png")

# Plot kept: global reliability curve over time, filtered on confirmed tracks.
# This plot aggregates reliability across confirmed tracks at each fusion time.
# It intentionally does NOT draw a separate line for the number of published tracks:
# only the mean reliability curve is shown, with a shaded min-max reliability band.
confirmed_tracks_for_reliability = tracks[
    (tracks["confirmed"] == True)
    & tracks["reliability"].notna()
    & tracks["time"].notna()
    & tracks["global_id"].notna()
].copy() if {"confirmed", "reliability", "time", "global_id"}.issubset(tracks.columns) else pd.DataFrame()

if len(confirmed_tracks_for_reliability) > 0:
    reliability_curve_df = confirmed_tracks_for_reliability[[
        "time", "global_id", "reliability"
    ]].copy()
    reliability_curve_df = reliability_curve_df.dropna(
        subset=["time", "global_id", "reliability"]
    )

    # Round timestamps so rows belonging to the same fusion cycle collapse into
    # one readable sample while preserving the temporal trend.
    reliability_curve_df["time_plot"] = reliability_curve_df["time"].round(2)

    reliability_curve = (
        reliability_curve_df
        .groupby("time_plot", as_index=False)
        .agg(
            mean_reliability=("reliability", "mean"),
            min_reliability=("reliability", "min"),
            max_reliability=("reliability", "max"),
            num_confirmed_tracks=("global_id", "nunique"),
        )
        .sort_values("time_plot")
    )

    reliability_curve.to_csv(
        OUT_DIR / "global_reliability_curve_over_time.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.fill_between(
        reliability_curve["time_plot"].to_numpy(),
        reliability_curve["min_reliability"].to_numpy(),
        reliability_curve["max_reliability"].to_numpy(),
        alpha=0.22,
        label="min-max reliability range",
    )

    ax.plot(
        reliability_curve["time_plot"].to_numpy(),
        reliability_curve["mean_reliability"].to_numpy(),
        marker="o",
        markersize=3,
        linewidth=2.2,
        label="mean reliability",
    )

    ax.set_title("Global tracking reliability over time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Reliability")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.5)
    ax.legend(fontsize=8, loc="best")
    savefig(OUT_DIR / "global_reliability_over_time.png")
else:
    print("WARNING: no confirmed tracks with reliability found. Reliability curve skipped.")

# Reliability heatmap for confirmed tracks.
# Rows are global IDs, columns are time/cycle samples, and the color encodes
# the reliability value. This makes track duration, fragmentation, and reliability
# drops easier to inspect than a dense multi-line plot.
if len(confirmed_tracks_for_reliability) > 0:
    heatmap_df = confirmed_tracks_for_reliability[["time", "global_id", "reliability"]].copy()
    heatmap_df = heatmap_df.dropna(subset=["time", "global_id", "reliability"])
    heatmap_df["global_id"] = heatmap_df["global_id"].astype(int)

    # Round time only for plotting columns, so very close timestamps from the
    # same fusion cycle collapse into one readable heatmap column.
    heatmap_df["time_plot"] = heatmap_df["time"].round(2)

    reliability_pivot = (
        heatmap_df
        .groupby(["global_id", "time_plot"], as_index=False)["reliability"]
        .max()
        .pivot(index="global_id", columns="time_plot", values="reliability")
        .sort_index()
    )

    if reliability_pivot.shape[0] > 0 and reliability_pivot.shape[1] > 0:
        fig_width = max(12.0, min(26.0, 0.18 * reliability_pivot.shape[1] + 6.0))
        fig_height = max(5.5, min(18.0, 0.28 * reliability_pivot.shape[0] + 3.0))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        matrix = reliability_pivot.to_numpy(dtype=float)
        masked_matrix = np.ma.masked_invalid(matrix)

        im = ax.imshow(
            masked_matrix,
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )

        ax.set_title("Confirmed track reliability heatmap")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Global ID")

        ax.set_yticks(np.arange(len(reliability_pivot.index)))
        ax.set_yticklabels([f"T{int(tid)}" for tid in reliability_pivot.index])

        n_cols = reliability_pivot.shape[1]
        max_ticks = 12
        tick_step = max(1, int(np.ceil(n_cols / max_ticks)))
        xtick_positions = np.arange(0, n_cols, tick_step)
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(
            [f"{float(reliability_pivot.columns[i]):.1f}" for i in xtick_positions],
            rotation=45,
            ha="right",
        )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Reliability")

        plt.tight_layout()
        plt.savefig(OUT_DIR / "reliability_heatmap_confirmed_tracks.png", dpi=250, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# LONG GIF WINDOW
# ============================================================

LONG_WINDOW_DIR = OUT_DIR / f"long_window_{LONG_GIF_START:.1f}_{LONG_GIF_END:.1f}"
LONG_WINDOW_DIR.mkdir(exist_ok=True)

tracks_long = slice_time_window(tracks_for_debug_viz, LONG_GIF_START, LONG_GIF_END)
published_tracks_long = slice_time_window(published_tracks, LONG_GIF_START, LONG_GIF_END)
detections_long = slice_time_window(detections, LONG_GIF_START, LONG_GIF_END)
matches_long = slice_time_window(matches, LONG_GIF_START, LONG_GIF_END)
duplicates_long = slice_time_window(duplicates, LONG_GIF_START, LONG_GIF_END)
gt_long = slice_time_window(ground_truth, LONG_GIF_START, LONG_GIF_END)

long_animation_created = create_matplotlib_fusion_animation(
    tracks_long, detections_long, matches_long, duplicates_long, gt_long,
    LONG_WINDOW_DIR / "long_fusion_gt_detections_tracks_animation.gif",
    fps=GIF_FPS, trail_seconds=TRAIL_SECONDS,
)

long_published_only_animation_created = False
if CREATE_PUBLISHED_ONLY_LONG_GIF:
    long_published_only_animation_created = create_published_only_animation(
        published_tracks_long, detections_long, matches_long, duplicates_long, gt_long,
        LONG_WINDOW_DIR / "long_fusion_published_tracks_only_animation.gif",
        fps=GIF_FPS, trail_seconds=TRAIL_SECONDS,
    )

long_lifecycle_animation_created = False
if CREATE_LIFECYCLE_TIMELINE_LONG_GIF:
    long_lifecycle_animation_created = create_lifecycle_timeline_animation(
        tracks_long,
        LONG_WINDOW_DIR / "long_fusion_lifecycle_timeline_animation.gif",
        fps=GIF_FPS,
    )

long_nearest_distance_animation_created = False
if CREATE_NEAREST_DETECTION_DISTANCE_LONG_GIF:
    long_nearest_distance_animation_created = create_nearest_detection_distance_animation(
        tracks_long, detections_long, matches_long,
        LONG_WINDOW_DIR / "long_fusion_nearest_detection_distance_animation.gif",
        fps=GIF_FPS,
    )


# ============================================================
# SLIDING WINDOWS OVER WHOLE BAG
# ============================================================

ALL_WINDOWS_DIR = OUT_DIR / "all_windows"
ALL_WINDOWS_DIR.mkdir(exist_ok=True)

time_candidates = []
for df in [tracks_for_window_summary, detections, matches, ground_truth]:
    if len(df) > 0 and "time" in df.columns:
        vals = pd.to_numeric(df["time"], errors="coerce").dropna()
        vals = vals[vals > 0.0]
        if len(vals) > 0:
            time_candidates.append(vals.min())

if len(time_candidates) == 0:
    raise RuntimeError("No valid timestamps found in input CSVs.")

bag_start = float(np.nanmin(time_candidates))

time_candidates = []
for df in [tracks_for_window_summary, detections, matches, ground_truth]:
    if len(df) > 0 and "time" in df.columns:
        vals = pd.to_numeric(df["time"], errors="coerce").dropna()
        vals = vals[vals > 0.0]
        if len(vals) > 0:
            time_candidates.append(vals.max())

bag_end = float(np.nanmax(time_candidates))

window_rows = []
w_start = bag_start
window_index = 0

while w_start < bag_end:
    w_end = min(w_start + WINDOW_SIZE, bag_end)

    tw = slice_time_window(tracks_for_window_summary, w_start, w_end, include_end=False)
    tw_pub = slice_time_window(published_tracks, w_start, w_end, include_end=False)
    dw = slice_time_window(detections, w_start, w_end, include_end=False)
    mw = slice_time_window(matches, w_start, w_end, include_end=False)
    dupw = slice_time_window(duplicates, w_start, w_end, include_end=False)
    gtw = slice_time_window(ground_truth, w_start, w_end, include_end=False)
    jumpw = slice_time_window(jump_rejects, w_start, w_end, include_end=False)
    conflictw = slice_time_window(conflict_rejects, w_start, w_end, include_end=False)
    blockw = slice_time_window(new_track_blocked, w_start, w_end, include_end=False)

    if len(tw) == 0 and len(dw) == 0 and len(mw) == 0 and len(gtw) == 0:
        w_start += WINDOW_STEP
        window_index += 1
        continue

    window_folder = ALL_WINDOWS_DIR / f"window_{window_index:03d}_{w_start:.2f}_{w_end:.2f}"
    window_folder.mkdir(exist_ok=True)

    cycle_folder = window_folder / "cycle_associations"
    num_cycle_plots = plot_cycle_associations(mw, cycle_folder, gtw)

    # Animated cycle associations for this 6-second window.
    cycle_association_animation_file = cycle_folder / "cycle_associations_animation.gif"
    cycle_association_animation_created = create_cycle_associations_animation(
        mw, gtw, cycle_association_animation_file, fps=GIF_FPS, draw_robot_fov=False
    )

    cycle_association_robot_fov_animation_file = cycle_folder / "cycle_associations_robot_fov_animation.gif"
    cycle_association_robot_fov_animation_created = create_cycle_associations_animation(
        mw, gtw, cycle_association_robot_fov_animation_file, fps=GIF_FPS, draw_robot_fov=True
    )

    # Per-window 4-panel summary plot intentionally disabled/removed.
    # The useful per-window outputs are now:
    # - static cycle_associations PNGs;
    # - animated cycle_associations GIF;
    # - all_windows_summary.csv/global summary;
    # - key PNG tables generated later.
    animation_created = bool(cycle_association_animation_created)
    published_only_animation_created = False

    ids_w = sorted(tw["global_id"].dropna().unique()) if len(tw) > 0 and "global_id" in tw.columns else []
    out_file = ""

    window_rows.append({
        "window_index": window_index,
        "start_time": w_start,
        "end_time": w_end,
        "num_ground_truth_samples": int(len(gtw)),
        "num_ground_truth_agents": int(gtw["agent_id"].nunique()) if len(gtw) > 0 else 0,
        "num_internal_ids": int(tw["global_id"].nunique()) if len(tw) > 0 else 0,
        "num_published_ids": int(tw_pub["global_id"].nunique()) if len(tw_pub) > 0 else 0,
        "num_local_detections": int(len(dw)),
        "num_accepted_matches": int(len(mw)),
        "num_duplicate_drops": int(len(dupw)),
        "num_jump_rejects": int(len(jumpw)),
        "num_local_id_conflict_rejects": int(len(conflictw)),
        "num_new_track_blocked": int(len(blockw)),
        "mean_match_distance": float(mw["distance"].mean()) if len(mw) > 0 else np.nan,
        "max_match_distance": float(mw["distance"].max()) if len(mw) > 0 else np.nan,
        "mean_reliability": float(tw["reliability"].mean()) if len(tw) > 0 and "reliability" in tw.columns else np.nan,
        "min_reliability": float(tw["reliability"].min()) if len(tw) > 0 and "reliability" in tw.columns else np.nan,
        "max_reliability": float(tw["reliability"].max()) if len(tw) > 0 and "reliability" in tw.columns else np.nan,
        "ids": ",".join([str(int(x)) for x in ids_w]),
        "published_ids": ",".join([str(int(x)) for x in sorted(tw_pub["global_id"].dropna().unique())]) if len(tw_pub) > 0 else "",
        "window_folder": window_folder.name,
        "summary_plot_file": "",
        "cycle_associations_folder": "cycle_associations",
        "num_cycle_association_plots": int(num_cycle_plots),
        "cycle_association_animation_created": bool(cycle_association_animation_created),
        "cycle_association_animation_file": "cycle_associations/cycle_associations_animation.gif" if cycle_association_animation_created else "",
        "cycle_association_robot_fov_animation_created": bool(cycle_association_robot_fov_animation_created),
        "cycle_association_robot_fov_animation_file": "cycle_associations/cycle_associations_robot_fov_animation.gif" if cycle_association_robot_fov_animation_created else "",
        "animation_created": bool(animation_created),
        "animation_file": "cycle_associations/cycle_associations_animation.gif" if animation_created else "",
        "published_only_animation_created": bool(published_only_animation_created),
        "published_only_animation_file": "fusion_published_tracks_only_animation.gif" if published_only_animation_created else "",
    })

    w_start += WINDOW_STEP
    window_index += 1


# ============================================================
# WINDOW SUMMARY CSV
# ============================================================

window_summary = pd.DataFrame(window_rows)
window_summary.to_csv(ALL_WINDOWS_DIR / "all_windows_summary.csv", index=False)


# ============================================================
# GLOBAL WINDOW SUMMARY PLOT
# ============================================================

if len(window_summary) > 0:
    fig, axes = plt.subplots(7, 1, figsize=(14, 23), sharex=True)

    axes[0].plot(window_summary["start_time"].to_numpy(), window_summary["num_ground_truth_agents"].to_numpy(), marker="o")
    axes[0].set_ylabel("GT agents")
    axes[0].set_title("Global window-based summary over whole bag")
    axes[0].grid(True, alpha=0.5)

    axes[1].plot(window_summary["start_time"].to_numpy(), window_summary["num_published_ids"].to_numpy(), marker="o", color=GLOBAL_TRACK_STYLE["color"], label="published IDs")
    axes[1].plot(window_summary["start_time"].to_numpy(), window_summary["num_internal_ids"].to_numpy(), marker="x", label="internal IDs")
    axes[1].set_ylabel("IDs")
    axes[1].grid(True, alpha=0.5)
    axes[1].legend()

    axes[2].plot(window_summary["start_time"].to_numpy(), window_summary["num_accepted_matches"].to_numpy(), marker="o")
    axes[2].set_ylabel("Accepted matches")
    axes[2].grid(True, alpha=0.5)

    axes[3].plot(window_summary["start_time"].to_numpy(), window_summary["num_duplicate_drops"].to_numpy(), marker="x", color=DUPLICATE_DROPPED_STYLE["color"], label="duplicate drops")
    axes[3].plot(window_summary["start_time"].to_numpy(), window_summary["num_new_track_blocked"].to_numpy(), marker="o", label="new track blocked")
    axes[3].set_ylabel("Duplicate/block")
    axes[3].grid(True, alpha=0.5)
    axes[3].legend()

    axes[4].plot(window_summary["start_time"].to_numpy(), window_summary["num_jump_rejects"].to_numpy(), marker="x", label="jump rejects")
    axes[4].plot(window_summary["start_time"].to_numpy(), window_summary["num_local_id_conflict_rejects"].to_numpy(), marker="^", label="local-id conflict rejects")
    axes[4].set_ylabel("Rejects")
    axes[4].grid(True, alpha=0.5)
    axes[4].legend()

    axes[5].plot(window_summary["start_time"].to_numpy(), window_summary["mean_match_distance"].to_numpy(), marker="o", label="mean")
    axes[5].plot(window_summary["start_time"].to_numpy(), window_summary["max_match_distance"].to_numpy(), marker="o", label="max")
    axes[5].set_ylabel("Distance [m]")
    axes[5].grid(True, alpha=0.5)
    axes[5].legend()

    axes[6].plot(window_summary["start_time"].to_numpy(), window_summary["mean_reliability"].to_numpy(), marker="o", label="mean reliability")
    axes[6].plot(window_summary["start_time"].to_numpy(), window_summary["min_reliability"].to_numpy(), marker="o", label="min reliability")
    axes[6].plot(window_summary["start_time"].to_numpy(), window_summary["max_reliability"].to_numpy(), marker="o", label="max reliability")
    axes[6].set_ylabel("Reliability")
    axes[6].set_xlabel("Window start time [s]")
    axes[6].set_ylim(-0.05, 1.05)
    axes[6].grid(True, alpha=0.5)
    axes[6].legend()

    savefig(ALL_WINDOWS_DIR / "all_windows_global_summary.png")


# ============================================================
# DIAGNOSTIC TABLES
# ============================================================

def safe_count_by(df, col, name_col=None, count_col="count"):
    if df is None or len(df) == 0 or col not in df.columns:
        if name_col is None:
            name_col = col
        return pd.DataFrame(columns=[name_col, count_col])
    out = df[col].astype(str).value_counts(dropna=False).reset_index()
    out.columns = [name_col or col, count_col]
    return out


def numeric_summary(df, columns):
    rows = []
    for col in columns:
        if df is None or len(df) == 0 or col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        rows.append({
            "metric": col,
            "count": int(len(vals)),
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "std": float(vals.std()) if len(vals) > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def format_table_value(value):
    """Format numbers/booleans/NaNs for readable paper-style tables."""
    try:
        if pd.isna(value):
            return "-"
    except Exception:
        pass
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if bool(value) else "No"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if not np.isfinite(v):
            return "-"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 100:
            return f"{v:.1f}"
        if abs(v) >= 10:
            return f"{v:.2f}"
        return f"{v:.3f}"
    return str(value)


def _wrap_table_text(value, width=30):
    text = format_table_value(value)
    if len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, replace_whitespace=False))


def _infer_paper_col_widths(columns):
    """Column width presets tuned for thesis/paper result tables."""
    cols = [str(c).lower() for c in columns]
    n = len(cols)
    if n == 1:
        return [1.0]
    if cols == ["metric", "value"]:
        return [0.62, 0.38]
    if n == 3 and "unit" in cols:
        return [0.48, 0.24, 0.28]
    if n == 4 and "information" in cols:
        return [0.27, 0.18, 0.12, 0.43]
    if n == 5 and "information" in cols:
        return [0.16, 0.28, 0.16, 0.10, 0.30]
    if n == 3:
        return [0.38, 0.22, 0.40]
    if n == 4:
        return [0.25, 0.25, 0.18, 0.32]
    return [1.0 / n] * n


def save_dataframe_as_table_plot(df, png_path, title=None, max_rows=32):
    """Save a dataframe as a clean paper/thesis-style PNG table.

    This renderer does not rely on matplotlib's automatic table scaling. It draws
    every cell manually, with row height computed from wrapped text. This avoids
    the typical overlap problems when table cells contain explanatory text.
    """
    if df is None or len(df) == 0:
        df = pd.DataFrame([{"Metric": "No data available", "Value": "-"}])
    else:
        df = df.copy()

    if len(df) > max_rows:
        df = df.head(max_rows).copy()
        df.loc[len(df)] = ["..." for _ in df.columns]

    # Paper-friendly column names: keep original CSV names in saved CSV, but make
    # the PNG headers readable.
    display_df = df.copy()
    display_df.columns = [str(c).replace("_", " ").title() for c in display_df.columns]

    n_cols = len(display_df.columns)
    col_widths = _infer_paper_col_widths(display_df.columns)

    # Wrap width depends on the available visual column width.
    wrap_widths = []
    for w in col_widths:
        wrap_widths.append(max(10, int(54 * w)))

    formatted_rows = []
    row_line_counts = []
    for _, row in display_df.iterrows():
        values = []
        max_lines = 1
        for value, wrap_width in zip(row.tolist(), wrap_widths):
            wrapped = _wrap_table_text(value, width=wrap_width)
            values.append(wrapped)
            max_lines = max(max_lines, wrapped.count("\n") + 1)
        formatted_rows.append(values)
        row_line_counts.append(max_lines)

    header_values = []
    header_lines = 1
    for c, wrap_width in zip(display_df.columns, wrap_widths):
        wrapped = _wrap_table_text(c, width=max(8, wrap_width))
        header_values.append(wrapped)
        header_lines = max(header_lines, wrapped.count("\n") + 1)

    # Layout sizing. Large enough for slide/paper inclusion, but not enormous.
    title_space = 0.95 if title else 0.35
    footer_space = 0.20
    line_height = 0.34
    header_height = max(0.65, header_lines * line_height + 0.28)
    row_heights = [max(0.70, lc * line_height + 0.30) for lc in row_line_counts]
    table_height = header_height + sum(row_heights)
    fig_height = min(28.0, max(5.2, title_space + table_height + footer_space))
    fig_width = 15.5 if n_cols <= 4 else 18.0

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if title:
        ax.text(0.5, 0.985, str(title), ha="center", va="top", fontsize=18, fontweight="bold")

    top = 0.925 if title else 0.975
    bottom = 0.035
    available_h = top - bottom
    scale = available_h / table_height

    x_edges = [0.02]
    usable_w = 0.96
    acc = 0.02
    for w in col_widths:
        acc += usable_w * w
        x_edges.append(acc)

    y = top

    def draw_cell(x0, x1, y_top, h, text, facecolor, weight="normal", fontsize=10.5, ha="left"):
        rect = plt.Rectangle((x0, y_top - h), x1 - x0, h, facecolor=facecolor, edgecolor="#5F6B73", linewidth=0.75)
        ax.add_patch(rect)
        pad = 0.007
        tx = x0 + pad if ha == "left" else (x0 + x1) / 2.0
        ax.text(tx, y_top - h / 2.0, text, ha=ha, va="center", fontsize=fontsize, fontweight=weight, linespacing=1.18, color="#111111")

    # Header
    h = header_height * scale
    for j, txt in enumerate(header_values):
        draw_cell(x_edges[j], x_edges[j + 1], y, h, txt, "#DDEBF7", weight="bold", fontsize=11.0)
    y -= h

    # Body
    for i, row in enumerate(formatted_rows):
        h = row_heights[i] * scale
        face = "#FFFFFF" if i % 2 == 0 else "#F7F9FB"
        for j, txt in enumerate(row):
            # Values/units are easier to read centered, prose left-aligned.
            col_name = str(display_df.columns[j]).lower()
            ha = "center" if col_name in {"value", "unit", "count", "percentage", "status"} else "left"
            draw_cell(x_edges[j], x_edges[j + 1], y, h, txt, face, fontsize=10.2, ha=ha)
        y -= h

    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return png_path


def save_table(df, name, title=None, latex=True):
    """Save table data as CSV, PNG, and optionally LaTeX for thesis/paper use.

    With GENERATE_ONLY_KEY_PAPER_TABLES=True, only the final key behavior
    tables are rendered/saved. Intermediate diagnostic dataframes are still
    computed internally, but not exported as paper-style tables.
    """
    if GENERATE_ONLY_KEY_PAPER_TABLES and name not in KEY_PAPER_TABLE_NAMES:
        return None, None

    csv_path = OUT_DIR / f"{name}.csv"
    png_path = OUT_DIR / f"{name}.png"
    tex_path = OUT_DIR / f"{name}.tex"

    df = df.copy() if df is not None else pd.DataFrame()
    # Make thesis/paper tables use the requested wording.
    df = df.rename(columns={"why_it_matters": "information", "interpretation": "information"})

    df.to_csv(csv_path, index=False)
    save_dataframe_as_table_plot(df, png_path, title=title or name.replace("_", " "))
    if latex:
        try:
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(df.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.3f}"))
        except Exception as exc:
            print(f"WARNING: could not save LaTeX table {tex_path.name}: {exc}")
    return csv_path, png_path


# Event distribution from debug_events_expanded.csv
event_type_counts = safe_count_by(debug, "event_type", "event_type", "count")
save_table(event_type_counts, "table_event_type_counts")

# Detection distribution from local_detections_debug.csv
if len(detections) > 0 and "camera" in detections.columns:
    detections_by_camera = safe_count_by(detections, "camera", "camera", "num_detections")
else:
    detections_by_camera = pd.DataFrame(columns=["camera", "num_detections"])
save_table(detections_by_camera, "table_detections_by_camera")

# Matching quality from MATCH_ACCEPTED rows
match_quality = numeric_summary(matches, ["distance", "dynamic_threshold", "association_cost"])
save_table(match_quality, "table_match_quality")

# Recovery quality from RECOVER_EXISTING_TRACK rows
recovery_quality = numeric_summary(reactivations, ["distance", "threshold", "best_distance"])
save_table(recovery_quality, "table_recovery_quality")

# New-track blocking distances
new_track_blocked_quality = numeric_summary(new_track_blocked, ["distance", "threshold", "best_distance"])
save_table(new_track_blocked_quality, "table_new_track_blocked_quality")

# Per-global-track lifecycle summary from global_tracks_debug.csv
if len(tracks) > 0 and "global_id" in tracks.columns:
    agg_dict = {
        "time": ["min", "max", "count"],
    }
    for col in ["hits", "missed", "age", "reliability"]:
        if col in tracks.columns:
            agg_dict[col] = ["max", "mean"]
    if "confirmed" in tracks.columns:
        agg_dict["confirmed"] = "max"
    if "publishable" in tracks.columns:
        agg_dict["publishable"] = "max"

    per_track = tracks.groupby("global_id").agg(agg_dict)
    per_track.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in per_track.columns]
    per_track = per_track.reset_index()
    if "time_min" in per_track.columns and "time_max" in per_track.columns:
        per_track["duration_s"] = per_track["time_max"] - per_track["time_min"]
    if "time_count" in per_track.columns:
        per_track = per_track.rename(columns={"time_count": "num_logged_states"})
else:
    per_track = pd.DataFrame()
save_table(per_track, "table_per_global_track_lifecycle")

# Delete reasons from DELETE_TRACK rows
if len(deleted_tracks) > 0 and "reason" in deleted_tracks.columns:
    delete_reasons = safe_count_by(deleted_tracks, "reason", "delete_reason", "count")
elif len(deleted_tracks) > 0 and "delete_reason" in deleted_tracks.columns:
    delete_reasons = safe_count_by(deleted_tracks, "delete_reason", "delete_reason", "count")
else:
    delete_reasons = pd.DataFrame(columns=["delete_reason", "count"])
save_table(delete_reasons, "table_delete_reasons")

# Local 3D tracker diagnostic table, if the optional CSV exists
if len(local_tracker_3d) > 0:
    local_3d_tables = []
    for col in ["method", "status", "reason", "camera"]:
        if col in local_tracker_3d.columns:
            tmp = safe_count_by(local_tracker_3d, col, col, "count")
            tmp.insert(0, "source_column", col)
            local_3d_tables.append(tmp)
    local_3d_counts = pd.concat(local_3d_tables, ignore_index=True) if local_3d_tables else pd.DataFrame()
else:
    local_3d_counts = pd.DataFrame()
save_table(local_3d_counts, "table_local_tracker_3d_counts")


# ============================================================
# LOCAL TRACKER 3D DIAGNOSTICS
# ============================================================

LOCAL_TRACKER_DIR = OUT_DIR / "local_tracker_3d_diagnostics"
LOCAL_TRACKER_DIR.mkdir(exist_ok=True)


def _value_counts_table(df, column, out_name, normalized=True):
    """Save count and percentage table for one categorical local-tracker column."""
    if df is None or len(df) == 0 or column not in df.columns:
        out = pd.DataFrame(columns=[column, "count", "percentage"])
        save_table(out, out_name)
        return out

    counts = (
        df[column]
        .fillna("nan")
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name="count")
    )
    if normalized and int(counts["count"].sum()) > 0:
        counts["percentage"] = 100.0 * counts["count"] / counts["count"].sum()
    else:
        counts["percentage"] = np.nan

    save_table(counts, out_name)
    return counts


def _bar_from_counts(counts, label_col, value_col, title, ylabel, out_file, rotation=25):
    """Create a simple count bar plot with readable labels."""
    if counts is None or len(counts) == 0 or label_col not in counts.columns or value_col not in counts.columns:
        return False

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(counts))
    ax.bar(x, counts[value_col].to_numpy())
    ax.set_xticks(x)
    ax.set_xticklabels(counts[label_col].astype(str).tolist(), rotation=rotation, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.35)

    for i, v in enumerate(counts[value_col].to_numpy()):
        try:
            ax.text(i, float(v), f"{int(v)}", ha="center", va="bottom", fontsize=8)
        except Exception:
            pass

    plt.tight_layout()
    plt.savefig(out_file, dpi=250)
    plt.close()
    return True


def generate_local_tracker_3d_diagnostics(local_df):
    """
    Generate CSV/Markdown tables and PNG plots for the local RGB-D tracker debug CSV.

    Expected input file: local_tracker_3d_debug.csv, or exported variants such as
    local_tracker_3d_debug(3).csv. The function is intentionally optional: if the
    CSV is missing, the rest of the fusion analysis still runs normally.
    """
    if local_df is None or len(local_df) == 0:
        empty = pd.DataFrame([
            {"metric": "local_tracker_3d_debug_available", "value": False},
            {"metric": "local_tracker_3d_total_rows", "value": 0},
        ])
        save_table(empty, "table_local_tracker_3d_summary")
        return empty

    df = local_df.copy()

    # Normalize key categorical columns if present.
    for col in ["row_type", "method", "reason", "source_topic", "camera_frame", "published_by_local_tracker", "used_transform_available"]:
        if col in df.columns:
            df[col] = df[col].fillna("nan").astype(str)

    # Split accepted/dropped rows in a robust way.
    row_type = df["row_type"].astype(str).str.lower() if "row_type" in df.columns else pd.Series([""] * len(df), index=df.index)
    reason = df["reason"].astype(str).str.lower() if "reason" in df.columns else pd.Series([""] * len(df), index=df.index)
    method = df["method"].astype(str).str.lower() if "method" in df.columns else pd.Series([""] * len(df), index=df.index)

    accepted_mask = (
        row_type.str.contains("detection", na=False)
        & ~method.str.contains("dropped|reject|invalid", na=False)
        & ~reason.str.contains("reject|invalid|insufficient|dropped", na=False)
    )
    dropped_mask = (
        row_type.str.contains("drop|reject", na=False)
        | method.str.contains("dropped|reject|invalid", na=False)
        | reason.str.contains("reject|invalid|insufficient|dropped", na=False)
    )

    accepted_df = df[accepted_mask].copy()
    dropped_df = df[dropped_mask].copy()

    total_rows = int(len(df))
    num_accepted = int(len(accepted_df))
    num_dropped = int(len(dropped_df))

    # Method/reason/source tables.
    method_counts = _value_counts_table(df, "method", "table_local_tracker_3d_method_counts")
    reason_counts = _value_counts_table(df, "reason", "table_local_tracker_3d_reason_counts")
    row_type_counts = _value_counts_table(df, "row_type", "table_local_tracker_3d_row_type_counts")
    source_counts = _value_counts_table(df, "source_topic", "table_local_tracker_3d_source_topic_counts")
    camera_frame_counts = _value_counts_table(df, "camera_frame", "table_local_tracker_3d_camera_frame_counts")

    # Numeric statistics for quality indicators.
    numeric_quality_cols = [
        "bbox_score", "num_valid_keypoints", "fallback_valid_count",
        "fallback_valid_fraction", "fallback_depth_std", "camera_z", "map_z",
        "reliability",
    ]
    local_numeric_stats = numeric_summary(df, numeric_quality_cols)
    save_table(local_numeric_stats, "table_local_tracker_3d_numeric_statistics")

    # Per camera/source summary.
    source_col = "source_topic" if "source_topic" in df.columns else ("camera_frame" if "camera_frame" in df.columns else None)
    if source_col is not None:
        per_source_rows = []
        for source, g in df.groupby(source_col):
            g_reason = g["reason"].astype(str).str.lower() if "reason" in g.columns else pd.Series([""] * len(g), index=g.index)
            g_method = g["method"].astype(str).str.lower() if "method" in g.columns else pd.Series([""] * len(g), index=g.index)
            g_row_type = g["row_type"].astype(str).str.lower() if "row_type" in g.columns else pd.Series([""] * len(g), index=g.index)
            g_dropped = (
                g_row_type.str.contains("drop|reject", na=False)
                | g_method.str.contains("dropped|reject|invalid", na=False)
                | g_reason.str.contains("reject|invalid|insufficient|dropped", na=False)
            )
            per_source_rows.append({
                "source": source,
                "total_rows": int(len(g)),
                "accepted_rows": int(len(g) - g_dropped.sum()),
                "dropped_rows": int(g_dropped.sum()),
                "drop_percentage": float(100.0 * g_dropped.sum() / len(g)) if len(g) > 0 else np.nan,
                "mean_bbox_score": float(g["bbox_score"].mean()) if "bbox_score" in g.columns else np.nan,
                "mean_valid_keypoints": float(g["num_valid_keypoints"].mean()) if "num_valid_keypoints" in g.columns else np.nan,
                "median_valid_keypoints": float(g["num_valid_keypoints"].median()) if "num_valid_keypoints" in g.columns else np.nan,
                "mean_camera_depth_m": float(g["camera_z"].mean()) if "camera_z" in g.columns else np.nan,
            })
        per_source_summary = pd.DataFrame(per_source_rows).sort_values("source")
    else:
        per_source_summary = pd.DataFrame()
    save_table(per_source_summary, "table_local_tracker_3d_per_source_summary")

    # Compact summary metrics.
    def _count_method_contains(name):
        return int(method.str.contains(name, na=False).sum())

    summary_rows = [
        {"metric": "local_tracker_3d_debug_available", "value": True},
        {"metric": "local_tracker_3d_total_rows", "value": total_rows},
        {"metric": "local_tracker_3d_accepted_rows", "value": num_accepted},
        {"metric": "local_tracker_3d_dropped_or_rejected_rows", "value": num_dropped},
        {"metric": "local_tracker_3d_drop_percentage", "value": float(100.0 * num_dropped / total_rows) if total_rows > 0 else np.nan},
        {"metric": "local_tracker_keypoints_median_rows", "value": _count_method_contains("keypoints_median")},
        {"metric": "local_tracker_bbox_center_fallback_rows", "value": _count_method_contains("bbox_center_fallback")},
        {"metric": "local_tracker_dropped_method_rows", "value": _count_method_contains("dropped")},
        {"metric": "local_tracker_mean_valid_keypoints", "value": float(df["num_valid_keypoints"].mean()) if "num_valid_keypoints" in df.columns else np.nan},
        {"metric": "local_tracker_median_valid_keypoints", "value": float(df["num_valid_keypoints"].median()) if "num_valid_keypoints" in df.columns else np.nan},
        {"metric": "local_tracker_min_valid_keypoints", "value": float(df["num_valid_keypoints"].min()) if "num_valid_keypoints" in df.columns else np.nan},
        {"metric": "local_tracker_max_valid_keypoints", "value": float(df["num_valid_keypoints"].max()) if "num_valid_keypoints" in df.columns else np.nan},
        {"metric": "local_tracker_mean_bbox_score", "value": float(df["bbox_score"].mean()) if "bbox_score" in df.columns else np.nan},
        {"metric": "local_tracker_mean_camera_depth_m", "value": float(df["camera_z"].mean()) if "camera_z" in df.columns else np.nan},
        {"metric": "local_tracker_mean_fallback_valid_fraction", "value": float(df["fallback_valid_fraction"].mean()) if "fallback_valid_fraction" in df.columns else np.nan},
        {"metric": "local_tracker_mean_fallback_depth_std", "value": float(df["fallback_depth_std"].mean()) if "fallback_depth_std" in df.columns else np.nan},
    ]
    local_summary = pd.DataFrame(summary_rows)
    save_table(local_summary, "table_local_tracker_3d_summary")

    # Plots: method, reject reasons, source/camera contribution.
    _bar_from_counts(
        method_counts,
        "method",
        "count",
        "Local RGB-D tracker: 3D estimation method counts",
        "Rows",
        LOCAL_TRACKER_DIR / "local_tracker_3d_method_counts.png",
        rotation=20,
    )
    _bar_from_counts(
        reason_counts.head(15),
        "reason",
        "count",
        "Local RGB-D tracker: top reject/status reasons",
        "Rows",
        LOCAL_TRACKER_DIR / "local_tracker_3d_reason_counts.png",
        rotation=35,
    )
    _bar_from_counts(
        source_counts,
        "source_topic",
        "count",
        "Local RGB-D tracker: rows per camera/source topic",
        "Rows",
        LOCAL_TRACKER_DIR / "local_tracker_3d_source_topic_counts.png",
        rotation=20,
    )

    # Valid-keypoints histogram intentionally disabled to keep the local tracker diagnostics lighter.

    # Bbox score distribution.
    if "bbox_score" in df.columns:
        vals = pd.to_numeric(df["bbox_score"], errors="coerce").dropna()
        if len(vals) > 0:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.hist(vals.to_numpy(), bins=20)
            ax.set_title("Local RGB-D tracker: YOLO bbox score distribution")
            ax.set_xlabel("BBox score")
            ax.set_ylabel("Rows")
            ax.grid(True, axis="y", alpha=0.35)
            plt.tight_layout()
            plt.savefig(LOCAL_TRACKER_DIR / "local_tracker_3d_bbox_score_histogram.png", dpi=250)
            plt.close()

    # Accepted/dropped timeline per second.
    if "time" in df.columns:
        tmp = df[["time"]].copy()
        tmp["accepted"] = accepted_mask.astype(int)
        tmp["dropped"] = dropped_mask.astype(int)
        tmp = tmp.dropna(subset=["time"])
        if len(tmp) > 0:
            tmp["time_bin"] = np.floor(tmp["time"].astype(float))
            timeline = tmp.groupby("time_bin")[["accepted", "dropped"]].sum().reset_index()
            save_table(timeline, "table_local_tracker_3d_timeline_per_second")

            fig, ax = plt.subplots(figsize=(11, 5))
            ax.plot(timeline["time_bin"].to_numpy(), timeline["accepted"].to_numpy(), marker="o", label="accepted")
            ax.plot(timeline["time_bin"].to_numpy(), timeline["dropped"].to_numpy(), marker="x", label="dropped/rejected")
            ax.set_title("Local RGB-D tracker: accepted vs dropped rows over time")
            ax.set_xlabel("time [s]")
            ax.set_ylabel("Rows per second")
            ax.grid(True, alpha=0.35)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
            plt.tight_layout(rect=[0, 0, 0.82, 1])
            plt.savefig(LOCAL_TRACKER_DIR / "local_tracker_3d_accepted_dropped_timeline.png", dpi=250)
            plt.close()

    # Fallback-specific diagnostic scatter/histogram, if fallback rows exist.
    if "method" in df.columns:
        fallback_df = df[df["method"].astype(str).str.lower().str.contains("fallback", na=False)].copy()
    else:
        fallback_df = df.iloc[0:0].copy()

    if len(fallback_df) > 0:
        fallback_stats = numeric_summary(fallback_df, ["fallback_valid_count", "fallback_valid_fraction", "fallback_depth_std", "camera_z", "bbox_score"])
        save_table(fallback_stats, "table_local_tracker_3d_fallback_statistics")

        if "fallback_valid_fraction" in fallback_df.columns:
            vals = pd.to_numeric(fallback_df["fallback_valid_fraction"], errors="coerce").dropna()
            if len(vals) > 0:
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.hist(vals.to_numpy(), bins=15)
                ax.set_title("Local RGB-D tracker: fallback valid depth fraction")
                ax.set_xlabel("Valid depth fraction")
                ax.set_ylabel("Fallback rows")
                ax.grid(True, axis="y", alpha=0.35)
                plt.tight_layout()
                plt.savefig(LOCAL_TRACKER_DIR / "local_tracker_3d_fallback_valid_fraction_histogram.png", dpi=250)
                plt.close()

        if "fallback_depth_std" in fallback_df.columns:
            vals = pd.to_numeric(fallback_df["fallback_depth_std"], errors="coerce").dropna()
            if len(vals) > 0:
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.hist(vals.to_numpy(), bins=15)
                ax.set_title("Local RGB-D tracker: fallback depth standard deviation")
                ax.set_xlabel("Depth std [m]")
                ax.set_ylabel("Fallback rows")
                ax.grid(True, axis="y", alpha=0.35)
                plt.tight_layout()
                plt.savefig(LOCAL_TRACKER_DIR / "local_tracker_3d_fallback_depth_std_histogram.png", dpi=250)
                plt.close()

    return local_summary


local_tracker_3d_summary_table = generate_local_tracker_3d_diagnostics(local_tracker_3d)

# Compact overview table useful for reports/presentations
overview_rows = [
    {"metric": "debug_events_total", "value": int(len(debug))},
    {"metric": "fusion_cycles", "value": int(events["cycle"].nunique()) if "cycle" in events.columns else 0},
    {"metric": "local_detections_total", "value": int(len(detections))},
    {"metric": "gt_agents", "value": int(ground_truth["agent_id"].nunique()) if len(ground_truth) > 0 and "agent_id" in ground_truth.columns else 0},
    {"metric": "global_ids_logged", "value": int(tracks["global_id"].nunique()) if len(tracks) > 0 and "global_id" in tracks.columns else 0},
    {"metric": "published_global_ids", "value": int(published_tracks["global_id"].nunique()) if len(published_tracks) > 0 and "global_id" in published_tracks.columns else 0},
    {"metric": "accepted_matches", "value": int(len(matches))},
    {"metric": "recoveries", "value": int(len(reactivations))},
    {"metric": "new_tracks", "value": int(len(new_tracks))},
    {"metric": "deleted_tracks", "value": int(len(deleted_tracks))},
    {"metric": "duplicate_drops", "value": int(len(duplicates))},
    {"metric": "new_track_blocked", "value": int(len(new_track_blocked))},
    {"metric": "mean_match_distance_m", "value": float(matches["distance"].mean()) if len(matches) > 0 and "distance" in matches.columns else np.nan},
    {"metric": "median_match_distance_m", "value": float(matches["distance"].median()) if len(matches) > 0 and "distance" in matches.columns else np.nan},
    {"metric": "max_match_distance_m", "value": float(matches["distance"].max()) if len(matches) > 0 and "distance" in matches.columns else np.nan},
]
overview_table = pd.DataFrame(overview_rows)
save_table(overview_table, "table_system_behavior_overview", title="System behavior overview")


def percent(part, total):
    return float(100.0 * part / total) if total and total > 0 else np.nan


def first_available_numeric_mean(df, cols):
    for col in cols:
        if df is not None and len(df) > 0 and col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) > 0:
                return float(vals.mean())
    return np.nan


def first_available_numeric_median(df, cols):
    for col in cols:
        if df is not None and len(df) > 0 and col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) > 0:
                return float(vals.median())
    return np.nan


def count_contains(series, pattern):
    if series is None:
        return 0
    return int(series.astype(str).str.lower().str.contains(pattern, na=False).sum())


def generate_key_behavior_table_plots():
    """Generate PNG table plots with the key values used to judge system behavior."""
    # --------------------------------------------------------
    # Local RGB-D tracker quality
    # --------------------------------------------------------
    if len(local_tracker_3d) > 0:
        method_series = local_tracker_3d["method"] if "method" in local_tracker_3d.columns else pd.Series([], dtype=str)
        reason_series = local_tracker_3d["reason"] if "reason" in local_tracker_3d.columns else pd.Series([], dtype=str)
        total_local_3d = int(len(local_tracker_3d))
        keypoint_rows = count_contains(method_series, "keypoints_median")
        fallback_rows = count_contains(method_series, "bbox_center_fallback")
        dropped_rows = count_contains(method_series, "dropped|reject|invalid")
        if dropped_rows == 0:
            dropped_rows = count_contains(reason_series, "reject|invalid|insufficient|dropped")
        accepted_rows = max(total_local_3d - dropped_rows, 0)
        local_rows = [
            {"indicator": "Total local RGB-D rows", "value": total_local_3d, "information": "Amount of local tracker evidence available for the test."},
            {"indicator": "Accepted 3D estimates", "value": accepted_rows, "information": "High accepted count means the fusion node receives usable detections."},
            {"indicator": "Dropped / rejected estimates", "value": f"{dropped_rows} ({percent(dropped_rows, total_local_3d):.1f}%)", "information": "Rejected noisy estimates are not injected into fusion."},
            {"indicator": "Keypoint median estimates", "value": f"{keypoint_rows} ({percent(keypoint_rows, total_local_3d):.1f}%)", "information": "Main robust 3D estimation path; should dominate over fallback."},
            {"indicator": "BBox-center fallback estimates", "value": f"{fallback_rows} ({percent(fallback_rows, total_local_3d):.1f}%)", "information": "Fallback should be rare and only used when valid."},
            {"indicator": "Mean valid keypoints", "value": first_available_numeric_mean(local_tracker_3d, ["num_valid_keypoints"]), "information": "Higher values mean the 3D person position is based on more body evidence."},
            {"indicator": "Median valid keypoints", "value": first_available_numeric_median(local_tracker_3d, ["num_valid_keypoints"]), "information": "Robust central quality indicator for pose-based 3D estimation."},
            {"indicator": "Mean bbox confidence", "value": first_available_numeric_mean(local_tracker_3d, ["bbox_score"]), "information": "Detection quality before fusion."},
            {"indicator": "Mean depth / camera_z [m]", "value": first_available_numeric_mean(local_tracker_3d, ["camera_z"]), "information": "Shows the typical sensing distance of accepted local estimates."},
        ]
    else:
        local_rows = [{"indicator": "local_tracker_3d_debug.csv", "value": "missing", "information": "Local RGB-D diagnostics could not be generated."}]
    local_key_table = pd.DataFrame(local_rows)
    save_table(local_key_table, "key_table_local_tracker_behavior", title="Local RGB-D tracker quality")

    # --------------------------------------------------------
    # Fusion matching and association quality
    # --------------------------------------------------------
    mean_match_distance = first_available_numeric_mean(matches, ["distance"])
    median_match_distance = first_available_numeric_median(matches, ["distance"])
    max_match_distance = float(matches["distance"].max()) if len(matches) > 0 and "distance" in matches.columns else np.nan
    mean_dyn_threshold = first_available_numeric_mean(matches, ["dynamic_threshold"])
    mean_recovery_distance = first_available_numeric_mean(reactivations, ["distance", "best_distance"])
    mean_recovery_threshold = first_available_numeric_mean(reactivations, ["threshold", "dynamic_threshold"])
    mean_block_distance = first_available_numeric_mean(new_track_blocked, ["distance", "best_distance"])

    fusion_rows = [
        {"indicator": "Accepted matches", "value": int(len(matches)), "information": "Successful local-to-global associations."},
        {"indicator": "Mean match distance [m]", "value": mean_match_distance, "information": "Low values mean detections are geometrically close to predicted tracks."},
        {"indicator": "Median match distance [m]", "value": median_match_distance, "information": "Robust typical association error."},
        {"indicator": "Max accepted match distance [m]", "value": max_match_distance, "information": "Checks whether accepted associations are close to the gating boundary."},
        {"indicator": "Mean dynamic threshold [m]", "value": mean_dyn_threshold, "information": "If this is much larger than match distance, gating is not overly strict."},
        {"indicator": "Recoveries", "value": int(len(reactivations)), "information": "Number of stale/global tracks correctly reactivated."},
        {"indicator": "Mean recovery distance [m]", "value": mean_recovery_distance, "information": "Low value means recovery is conservative."},
        {"indicator": "Mean recovery threshold [m]", "value": mean_recovery_threshold, "information": "Useful to see if recovery is tighter than normal matching."},
        {"indicator": "Duplicate drops", "value": int(len(duplicates)), "information": "Low count means few overlapping-camera duplicates reached fusion."},
        {"indicator": "New-track blocked near existing", "value": int(len(new_track_blocked)), "information": "Prevents duplicate global IDs close to existing tracks."},
        {"indicator": "Mean blocked-new-track distance [m]", "value": mean_block_distance, "information": "Shows how close blocked detections were to existing tracks."},
    ]
    fusion_key_table = pd.DataFrame(fusion_rows)
    save_table(fusion_key_table, "key_table_fusion_matching_behavior", title="Fusion matching and data association quality")

    # --------------------------------------------------------
    # Global track lifecycle / fragmentation
    # --------------------------------------------------------
    total_global_ids = int(tracks["global_id"].nunique()) if len(tracks) > 0 and "global_id" in tracks.columns else 0
    if len(tracks) > 0 and {"global_id", "confirmed"}.issubset(tracks.columns):
        confirmed_ids = int(tracks.groupby("global_id")["confirmed"].max().sum())
    else:
        confirmed_ids = 0
    if len(tracks) > 0 and {"global_id", "hits"}.issubset(tracks.columns):
        per_id_max_hits = tracks.groupby("global_id")["hits"].max()
        mean_max_hits = float(per_id_max_hits.mean())
        median_max_hits = float(per_id_max_hits.median())
        max_hits = float(per_id_max_hits.max())
    else:
        mean_max_hits = median_max_hits = max_hits = np.nan
    mean_pub_reliability = first_available_numeric_mean(published_tracks, ["reliability"])
    max_pub_reliability = float(published_tracks["reliability"].max()) if len(published_tracks) > 0 and "reliability" in published_tracks.columns else np.nan

    delete_reason_text = "-"
    if len(deleted_tracks) > 0:
        reason_col = "reason" if "reason" in deleted_tracks.columns else ("delete_reason" if "delete_reason" in deleted_tracks.columns else None)
        if reason_col is not None:
            top_reasons = deleted_tracks[reason_col].astype(str).value_counts().head(3)
            delete_reason_text = "; ".join([f"{k}: {int(v)}" for k, v in top_reasons.items()])

    lifecycle_rows = [
        {"indicator": "Global IDs observed", "value": total_global_ids, "information": "High value relative to GT agents can indicate fragmentation."},
        {"indicator": "Confirmed global IDs", "value": f"{confirmed_ids} ({percent(confirmed_ids, total_global_ids):.1f}%)", "information": "Tracks that survived enough hits to become reliable."},
        {"indicator": "Published global IDs", "value": int(published_tracks["global_id"].nunique()) if len(published_tracks) > 0 and "global_id" in published_tracks.columns else 0, "information": "IDs actually visible on /tracked_people."},
        {"indicator": "New tracks", "value": int(len(new_tracks)), "information": "Track creation rate."},
        {"indicator": "Deleted tracks", "value": int(len(deleted_tracks)), "information": "High deletion count suggests stale/fragmented tracks."},
        {"indicator": "Main delete reasons", "value": delete_reason_text, "information": "Explains whether tracks die from age, missed detections, or other logic."},
        {"indicator": "Mean max hits per ID", "value": mean_max_hits, "information": "Higher means IDs persist longer."},
        {"indicator": "Median max hits per ID", "value": median_max_hits, "information": "Low median means many short-lived tracks."},
        {"indicator": "Maximum hits on one ID", "value": max_hits, "information": "Shows best-case temporal continuity."},
        {"indicator": "Mean published reliability", "value": mean_pub_reliability, "information": "Quality score of tracks that are actually published."},
        {"indicator": "Max published reliability", "value": max_pub_reliability, "information": "Shows whether stable tracks can reach high confidence."},
    ]
    lifecycle_key_table = pd.DataFrame(lifecycle_rows)
    save_table(lifecycle_key_table, "key_table_track_lifecycle_behavior", title="Global track continuity and reliability")

    # --------------------------------------------------------
    # One compact table that combines the most presentation-relevant values
    # --------------------------------------------------------
    compact_rows = [
        {"area": "Local RGB-D", "key_value": "Keypoint median dominates", "value": f"{keypoint_rows if len(local_tracker_3d) > 0 else 0} / {len(local_tracker_3d)}", "information": "Most 3D estimates come from robust keypoint median, not fallback."},
        {"area": "Local RGB-D", "key_value": "Fallback usage", "value": f"{fallback_rows if len(local_tracker_3d) > 0 else 0}", "information": "Fallback is available but should remain rare."},
        {"area": "Local RGB-D", "key_value": "Dropped estimates", "value": f"{dropped_rows if len(local_tracker_3d) > 0 else 0}", "information": "Noisy/invalid 3D estimates are filtered before fusion."},
        {"area": "Fusion", "key_value": "Accepted matches", "value": int(len(matches)), "information": "Number of successful associations."},
        {"area": "Fusion", "key_value": "Mean match distance", "value": f"{mean_match_distance:.3f} m" if np.isfinite(mean_match_distance) else "-", "information": "Low distance means stable geometric association."},
        {"area": "Fusion", "key_value": "Mean dynamic threshold", "value": f"{mean_dyn_threshold:.3f} m" if np.isfinite(mean_dyn_threshold) else "-", "information": "Compare with match distance to understand gating margin."},
        {"area": "Fusion", "key_value": "Recoveries", "value": int(len(reactivations)), "information": "Recovered IDs after temporary loss."},
        {"area": "Fusion", "key_value": "Duplicate drops", "value": int(len(duplicates)), "information": "Low value means limited multi-camera duplicate noise."},
        {"area": "Lifecycle", "key_value": "Global IDs / confirmed", "value": f"{total_global_ids} / {confirmed_ids}", "information": "Shows how much track fragmentation remains."},
        {"area": "Lifecycle", "key_value": "Deleted tracks", "value": int(len(deleted_tracks)), "information": "High value can indicate stale tracks or fragmented IDs."},
        {"area": "Lifecycle", "key_value": "Mean published reliability", "value": f"{mean_pub_reliability:.3f}" if np.isfinite(mean_pub_reliability) else "-", "information": "Reliability of tracks reaching output."},
    ]
    compact_key_table = pd.DataFrame(compact_rows)
    save_table(compact_key_table, "key_table_system_behavior_compact", title="Overall multi-camera tracking behavior")


generate_key_behavior_table_plots()

# ============================================================
# SUMMARY METRICS
# ============================================================

summary = {
    "num_cycles": int(events["cycle"].nunique()) if "cycle" in events.columns else 0,
    "num_fusion_updates": int((events["event"] == "FUSION_UPDATE").sum()) if "event" in events.columns else 0,
    "num_no_detection_cycles": int(events["event"].astype(str).str.contains("NO_DETECTIONS").sum()) if "event" in events.columns else 0,
    "num_gt_agents": int(ground_truth["agent_id"].nunique()) if len(ground_truth) > 0 else 0,
    "num_gt_samples": int(len(ground_truth)),
    "num_global_tracks_created": int(new_tracks["global_id"].nunique()) if len(new_tracks) > 0 and "global_id" in new_tracks.columns else 0,
    "num_tracks_published": int(published_tracks["global_id"].nunique()) if len(published_tracks) > 0 else 0,
    "num_internal_tracks": int(tracks["global_id"].nunique()) if len(tracks) > 0 and "global_id" in tracks.columns else 0,
    "num_match_candidates": int((debug["event_type"] == "MATCH_CANDIDATE").sum()),
    "num_match_accepted": int((debug["event_type"] == "MATCH_ACCEPTED").sum()),
    "num_match_rejected": int((debug["event_type"] == "MATCH_REJECTED").sum()),
    "num_jump_rejects": int((debug["event_type"] == "MATCH_CANDIDATE_REJECTED_JUMP").sum()),
    "num_local_id_conflict_rejects": int((debug["event_type"] == "MATCH_CANDIDATE_REJECTED_LOCAL_ID_CONFLICT").sum()),
    "num_recover_jump_rejects": int((debug["event_type"] == "RECOVER_REJECTED_JUMP").sum()),
    "num_recover_local_id_conflict_rejects": int((debug["event_type"] == "RECOVER_REJECTED_LOCAL_ID_CONFLICT").sum()),
    "num_recover_ambiguous_rejects": int((debug["event_type"] == "RECOVER_REJECTED_AMBIGUOUS").sum()),
    "num_new_track_blocked": int((debug["event_type"] == "NEW_TRACK_BLOCKED_NEAR_EXISTING").sum()),
    "num_duplicate_drops": int((debug["event_type"] == "DUPLICATE_DROP").sum()),
    "num_reactivations": int((debug["event_type"] == "RECOVER_EXISTING_TRACK").sum()),
    "num_deleted_tracks": int((debug["event_type"] == "DELETE_TRACK").sum()),
    "mean_match_distance": float(matches["distance"].mean()) if len(matches) > 0 else np.nan,
    "median_match_distance": float(matches["distance"].median()) if len(matches) > 0 else np.nan,
    "max_match_distance": float(matches["distance"].max()) if len(matches) > 0 else np.nan,
    "mean_published_reliability": float(published_tracks["reliability"].mean()) if len(published_tracks) > 0 and "reliability" in published_tracks.columns else np.nan,
    "median_published_reliability": float(published_tracks["reliability"].median()) if len(published_tracks) > 0 and "reliability" in published_tracks.columns else np.nan,
    "min_published_reliability": float(published_tracks["reliability"].min()) if len(published_tracks) > 0 and "reliability" in published_tracks.columns else np.nan,
    "max_published_reliability": float(published_tracks["reliability"].max()) if len(published_tracks) > 0 and "reliability" in published_tracks.columns else np.nan,
    "mean_published_track_age": float(published_tracks["age"].mean()) if len(published_tracks) > 0 and "age" in published_tracks.columns else np.nan,
    "accepted_local_id_match": int((matches["local_id_relation"] == "match").sum()) if "local_id_relation" in matches.columns else 0,
    "accepted_local_id_conflict": int((matches["local_id_relation"] == "conflict").sum()) if "local_id_relation" in matches.columns else 0,
    "accepted_local_id_unknown": int((matches["local_id_relation"] == "unknown").sum()) if "local_id_relation" in matches.columns else 0,
    "accepted_local_id_none": int((matches["local_id_relation"] == "none").sum()) if "local_id_relation" in matches.columns else 0,
    "local_tracker_3d_debug_available": bool(len(local_tracker_3d) > 0),
    "local_tracker_3d_total_rows": int(len(local_tracker_3d)),
    "local_tracker_3d_methods": int(local_tracker_3d["method"].nunique()) if len(local_tracker_3d) > 0 and "method" in local_tracker_3d.columns else 0,
    "local_tracker_3d_reasons": int(local_tracker_3d["reason"].nunique()) if len(local_tracker_3d) > 0 and "reason" in local_tracker_3d.columns else 0,
    "local_tracker_mean_valid_keypoints": float(local_tracker_3d["num_valid_keypoints"].mean()) if len(local_tracker_3d) > 0 and "num_valid_keypoints" in local_tracker_3d.columns else np.nan,
    "local_tracker_median_valid_keypoints": float(local_tracker_3d["num_valid_keypoints"].median()) if len(local_tracker_3d) > 0 and "num_valid_keypoints" in local_tracker_3d.columns else np.nan,
    "local_tracker_mean_bbox_score": float(local_tracker_3d["bbox_score"].mean()) if len(local_tracker_3d) > 0 and "bbox_score" in local_tracker_3d.columns else np.nan,
    "long_gif_start": float(LONG_GIF_START),
    "long_gif_end": float(LONG_GIF_END),
    "long_gif_animation_created": bool(long_animation_created),
    "long_published_only_animation_created": bool(long_published_only_animation_created),
    "long_lifecycle_animation_created": bool(long_lifecycle_animation_created),
    "long_nearest_distance_animation_created": bool(long_nearest_distance_animation_created),
    "bag_start_used_for_windows": float(bag_start),
    "bag_end_used_for_windows": float(bag_end),
    "window_size": float(WINDOW_SIZE),
    "window_step": float(WINDOW_STEP),
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(OUT_DIR / "summary_metrics.csv", index=False)

print(f"\nSaved plots in: {OUT_DIR}")
print(f"Long GIF window plots in: {LONG_WINDOW_DIR}")
print(f"All-window plots in: {ALL_WINDOWS_DIR}")
print("\nLegend used in plots/GIFs:")
print("- center camera detection: yellow circle")
print("- left camera detection: red circle")
print("- right camera detection: orange circle")
print("- published global track: green body marker with dashed uncertainty halo")
print("- internal/unpublished global track: empty green circle, marked with * in debug GIF text")
print("- duplicate dropped: purple x")
print("- ground truth: large blue footprints with dashed trail and velocity arrow")
print("- robot: black circle with heading arrow")
print("- FOV center/left/right: semi-transparent yellow/red/orange cones")
print("\nGenerated GIF file:")
print(f"- Long debug GIF: {LONG_WINDOW_DIR / 'long_fusion_gt_detections_tracks_animation.gif'}")
print("\nGenerated key paper-style CSV/PNG/LaTeX tables in:", OUT_DIR)
print("\nSUMMARY METRICS")
print(summary_df.T)
