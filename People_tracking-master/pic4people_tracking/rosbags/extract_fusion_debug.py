import json
import csv
from pathlib import Path

import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message

from std_msgs.msg import String
from people_msgs.msg import People
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage


bag_path = "tracking_fusion_result"  # Change this to the path of your rosbag file
out_dir = Path("fusion_analysis")
out_dir.mkdir(exist_ok=True)

storage_options = rosbag2_py.StorageOptions(
    uri=bag_path,
    storage_id="sqlite3"
)

converter_options = rosbag2_py.ConverterOptions("", "")

reader = rosbag2_py.SequentialReader()
reader.open(storage_options, converter_options)

tracks_csv = out_dir / "global_tracks_debug.csv"
detections_csv = out_dir / "local_detections_debug.csv"
events_csv = out_dir / "fusion_events.csv"
debug_events_csv = out_dir / "debug_events_expanded.csv"
gt_csv = out_dir / "ground_truth_people.csv"
robot_csv = out_dir / "robot_pose_debug.csv"
local_debug_csv = out_dir / "local_tracker_3d_debug.csv"


def stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def normalize_quat(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n


def quat_multiply(q1, q2):
    # q = [x, y, z, w]
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return normalize_quat([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def quat_to_yaw(q):
    x, y, z, w = normalize_quat(q)
    return float(np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z)
    ))


def rotate_vector(q, v):
    # Rotate v by quaternion q=[x,y,z,w].
    q = normalize_quat(q)
    vx, vy, vz = v
    qv = np.array([vx, vy, vz, 0.0], dtype=float)
    q_conj = np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)
    return quat_multiply(quat_multiply(q, qv), q_conj)[:3]


def compose_transform(t_ab, q_ab, t_bc, q_bc):
    # Compose T_ab and T_bc -> T_ac.
    t_ab = np.asarray(t_ab, dtype=float)
    t_bc = np.asarray(t_bc, dtype=float)
    q_ab = normalize_quat(q_ab)
    q_bc = normalize_quat(q_bc)
    t_ac = t_ab + rotate_vector(q_ab, t_bc)
    q_ac = quat_multiply(q_ab, q_bc)
    return t_ac, q_ac


def transform_to_arrays(transform):
    tr = transform.transform.translation
    qr = transform.transform.rotation
    t_vec = np.array([tr.x, tr.y, tr.z], dtype=float)
    q_vec = normalize_quat([qr.x, qr.y, qr.z, qr.w])
    return t_vec, q_vec



with open(tracks_csv, "w", newline="") as ft, \
     open(detections_csv, "w", newline="") as fd, \
     open(events_csv, "w", newline="") as fe, \
     open(debug_events_csv, "w", newline="") as fde, \
     open(gt_csv, "w", newline="") as fgt, \
     open(robot_csv, "w", newline="") as fr, \
     open(local_debug_csv, "w", newline="") as fld:

    tracks_writer = csv.writer(ft)
    det_writer = csv.writer(fd)
    event_writer = csv.writer(fe)
    debug_event_writer = csv.writer(fde)
    gt_writer = csv.writer(fgt)
    robot_writer = csv.writer(fr)
    local_debug_writer = csv.writer(fld)

    # ====================================================
    # TRACKS CSV
    # ====================================================

    tracks_writer.writerow([
        "time",
        "cycle",
        "event",
        "global_id",
        "x",
        "y",
        "yaw",
        "vx",
        "vy",
        "hits",
        "missed",
        "confirmed",
        "age",
        "last_update",
        "reliability",
        "publishable",
        "local_ids_by_cam"
    ])

    # ====================================================
    # DETECTIONS CSV
    # ====================================================

    det_writer.writerow([
        "time",
        "cycle",
        "event",
        "camera",
        "local_id",
        "x",
        "y",
        "yaw",
        "stamp"
    ])

    # ====================================================
    # EVENTS CSV
    # ====================================================

    event_writer.writerow([
        "time",
        "cycle",
        "event",
        "num_detections",
        "published_ids",
        "extra"
    ])

    # ====================================================
    # DEBUG EVENTS CSV
    # ====================================================

    debug_event_writer.writerow([
        "time",
        "cycle",
        "main_event",
        "event_type",

        "global_id",
        "det_index",
        "camera",
        "local_id",
        "local_id_relation",
        "association_cost",

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
        "latest_stamp"
    ])

    # ====================================================
    # GROUND TRUTH CSV
    # ====================================================

    gt_writer.writerow([
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
        "reliability"
    ])

    # ====================================================
    # ROBOT POSE CSV
    # ====================================================

    robot_writer.writerow([
        "time",
        "bag_time",
        "frame_id",
        "x",
        "y",
        "z",
        "yaw",
        "vx",
        "vy",
        "vtheta"
    ])

    # ====================================================
    # LOCAL TRACKER 3D DEBUG CSV
    # ====================================================

    local_debug_writer.writerow([
        "time",
        "bag_time",
        "frame_id",
        "camera_frame",
        "source_topic",
        "row_type",
        "det_index",
        "local_id",
        "method",
        "reason",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "bbox_score",
        "num_valid_keypoints",
        "valid_keypoint_indices",
        "fallback_valid_count",
        "fallback_valid_fraction",
        "fallback_depth_std",
        "camera_x",
        "camera_y",
        "camera_z",
        "map_x",
        "map_y",
        "map_z",
        "yaw",
        "velocity_camera_x",
        "velocity_camera_y",
        "velocity_camera_z",
        "reliability",
        "published_by_local_tracker",
        "image_stamp",
        "transform_stamp",
        "used_transform_available"
    ])

    # ====================================================
    # TF CACHE FOR ROBOT POSE EXTRACTION
    # ====================================================

    latest_map_to_odom = None
    latest_odom_to_base = None
    last_written_tf_time = None

    # ====================================================
    # MAIN LOOP
    # ====================================================

    while reader.has_next():

        topic, data, t = reader.read_next()

        bag_time = float(t) * 1e-9

        # ====================================================
        # ROBOT POSE FROM TF
        # ====================================================

        # This works when the second rosbag contains /tf and /tf_static but not odometry.
        # Preferred pose is map -> base_link. If only map -> odom and odom -> base_link
        # are available, the script composes them into map -> base_link.
        if topic in ["/tf", "/tf_static"]:

            msg = deserialize_message(data, TFMessage)

            for tf in msg.transforms:
                parent = tf.header.frame_id.lstrip("/")
                child = tf.child_frame_id.lstrip("/")

                tf_time = stamp_to_sec(tf.header.stamp)
                if tf_time <= 0.0:
                    tf_time = bag_time

                t_vec, q_vec = transform_to_arrays(tf)

                # Direct transform: map -> base_link.
                if parent == "map" and child == "base_link":
                    yaw = quat_to_yaw(q_vec)
                    robot_writer.writerow([
                        tf_time, bag_time, "map",
                        t_vec[0], t_vec[1], t_vec[2],
                        yaw, np.nan, np.nan, np.nan,
                    ])
                    last_written_tf_time = tf_time
                    continue

                # Store transforms for composition.
                if parent == "map" and child == "odom":
                    latest_map_to_odom = (tf_time, bag_time, t_vec, q_vec)

                if parent == "odom" and child == "base_link":
                    latest_odom_to_base = (tf_time, bag_time, t_vec, q_vec)

                # Fallback: if the bag only contains odom -> base_link, write it in odom frame.
                # This is less ideal if detections are in map, but it is better than an empty CSV.
                if parent == "odom" and child == "base_link" and latest_map_to_odom is None:
                    yaw = quat_to_yaw(q_vec)
                    if last_written_tf_time is None or abs(tf_time - last_written_tf_time) > 1e-3:
                        robot_writer.writerow([
                            tf_time, bag_time, "odom",
                            t_vec[0], t_vec[1], t_vec[2],
                            yaw, np.nan, np.nan, np.nan,
                        ])
                        last_written_tf_time = tf_time

                # Compose map -> odom and odom -> base_link when both are available.
                if latest_map_to_odom is not None and latest_odom_to_base is not None:
                    t_mo_time, t_mo_bag, t_mo, q_mo = latest_map_to_odom
                    t_ob_time, t_ob_bag, t_ob, q_ob = latest_odom_to_base
                    composed_time = max(t_mo_time, t_ob_time)

                    if last_written_tf_time is None or abs(composed_time - last_written_tf_time) > 1e-3:
                        t_mb, q_mb = compose_transform(t_mo, q_mo, t_ob, q_ob)
                        yaw = quat_to_yaw(q_mb)
                        robot_writer.writerow([
                            composed_time, bag_time, "map",
                            t_mb[0], t_mb[1], t_mb[2],
                            yaw, np.nan, np.nan, np.nan,
                        ])
                        last_written_tf_time = composed_time

            continue

        # ====================================================
        # ROBOT POSE FROM ODOMETRY
        # ====================================================

        # Use the odometry topic recorded in the bag.
        # If your bag uses another topic, change this string accordingly,
        # for example: "/jackal/ground_truth".
        if topic == "/jackal_velocity_controller/odom":

            msg = deserialize_message(data, Odometry)

            msg_time = stamp_to_sec(msg.header.stamp)
            if msg_time <= 0.0:
                msg_time = bag_time

            q = msg.pose.pose.orientation
            yaw = np.arctan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )

            robot_writer.writerow([
                msg_time,
                bag_time,
                msg.header.frame_id,

                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,

                yaw,

                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.angular.z,
            ])

            continue

        # ====================================================
        # GROUND TRUTH
        # ====================================================

        if topic == "/people":

            msg = deserialize_message(data, People)

            msg_time = stamp_to_sec(msg.header.stamp)

            if msg_time <= 0.0:
                msg_time = bag_time

            frame_id = msg.header.frame_id

            for person in msg.people:

                gt_writer.writerow([
                    msg_time,
                    bag_time,
                    frame_id,

                    person.name,

                    person.position.x,
                    person.position.y,
                    person.position.z,

                    person.velocity.x,
                    person.velocity.y,
                    person.velocity.z,

                    person.reliability
                ])

            continue

        # ====================================================
        # LOCAL TRACKER DEBUG JSON
        # ====================================================

        if topic == "/local_tracker_debug":

            msg = deserialize_message(data, String)

            try:
                payload = json.loads(msg.data)
            except Exception:
                continue

            msg_time = payload.get("time", np.nan)
            frame_id = payload.get("frame_id", "")
            camera_frame = payload.get("camera_frame", "")
            source_topic = payload.get("source_topic", "")
            image_stamp = payload.get("image_stamp", np.nan)
            transform_stamp = payload.get("transform_stamp", np.nan)
            used_transform_available = payload.get("used_transform_available", False)

            # Detection rows: raw 3D estimate before StrongSORT local tracking.
            for d in payload.get("detections", []):
                local_debug_writer.writerow([
                    msg_time,
                    bag_time,
                    frame_id,
                    camera_frame,
                    source_topic,
                    "detection",
                    d.get("det_index"),
                    "",
                    d.get("method"),
                    d.get("reason"),
                    d.get("bbox_x1"),
                    d.get("bbox_y1"),
                    d.get("bbox_x2"),
                    d.get("bbox_y2"),
                    d.get("bbox_score"),
                    d.get("num_valid_keypoints"),
                    json.dumps(d.get("valid_keypoint_indices", [])),
                    d.get("fallback_valid_count"),
                    d.get("fallback_valid_fraction"),
                    d.get("fallback_depth_std"),
                    d.get("camera_x"),
                    d.get("camera_y"),
                    d.get("camera_z"),
                    d.get("map_x"),
                    d.get("map_y"),
                    d.get("map_z"),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    image_stamp,
                    transform_stamp,
                    used_transform_available,
                ])

            # Track rows: StrongSORT local output, linked to original det_index.
            for tr in payload.get("tracks", []):
                local_debug_writer.writerow([
                    msg_time,
                    bag_time,
                    frame_id,
                    camera_frame,
                    source_topic,
                    "track",
                    tr.get("det_index"),
                    tr.get("local_id"),
                    tr.get("method"),
                    tr.get("reason"),
                    "",
                    "",
                    "",
                    "",
                    tr.get("bbox_score"),
                    tr.get("num_valid_keypoints"),
                    "",
                    "",
                    "",
                    "",
                    tr.get("camera_x"),
                    tr.get("camera_y"),
                    tr.get("camera_z"),
                    tr.get("map_x"),
                    tr.get("map_y"),
                    tr.get("map_z"),
                    tr.get("yaw"),
                    tr.get("velocity_camera_x"),
                    tr.get("velocity_camera_y"),
                    tr.get("velocity_camera_z"),
                    tr.get("reliability"),
                    tr.get("published_by_local_tracker"),
                    image_stamp,
                    transform_stamp,
                    used_transform_available,
                ])

            continue

        # ====================================================
        # FUSION DEBUG JSON
        # ====================================================

        if topic != "/tracked_people_debug":
            continue

        msg = deserialize_message(data, String)

        try:
            payload = json.loads(msg.data)

        except Exception:
            continue

        time = payload.get("time")
        cycle = payload.get("cycle")
        event = payload.get("event")

        published_ids = payload.get("published_ids", [])
        extra = payload.get("extra", {})

        # ====================================================
        # EVENTS
        # ====================================================

        event_writer.writerow([
            time,
            cycle,
            event,

            payload.get("num_detections", 0),

            json.dumps(published_ids),

            json.dumps(extra)
        ])

        # ====================================================
        # DETECTIONS
        # ====================================================

        for d in payload.get("detections", []):

            det_writer.writerow([
                time,
                cycle,
                event,

                d.get("camera"),
                d.get("local_id"),

                d.get("x"),
                d.get("y"),
                d.get("yaw"),

                d.get("stamp")
            ])

        # ====================================================
        # TRACKS
        # ====================================================

        for tr in payload.get("tracks", []):

            tracks_writer.writerow([
                time,
                cycle,
                event,

                tr.get("global_id"),

                tr.get("x"),
                tr.get("y"),
                tr.get("yaw"),

                tr.get("vx"),
                tr.get("vy"),

                tr.get("hits"),
                tr.get("missed"),

                tr.get("confirmed"),

                tr.get("age"),
                tr.get("last_update"),

                tr.get("reliability"),
                tr.get("publishable"),

                json.dumps(tr.get("local_ids_by_cam", {}))
            ])

        # ====================================================
        # DEBUG EVENTS
        # ====================================================

        for ev in payload.get("debug_events", []):

            debug_event_writer.writerow([

                time,
                cycle,
                event,

                ev.get("event_type"),

                ev.get("global_id"),
                ev.get("det_index"),

                ev.get("camera"),
                ev.get("local_id"),

                ev.get("local_id_relation"),
                ev.get("association_cost"),

                ev.get("pred_x"),
                ev.get("pred_y"),

                ev.get("det_x"),
                ev.get("det_y"),

                ev.get("distance"),
                ev.get("dynamic_threshold"),
                ev.get("valid_candidate"),

                ev.get("old_x"),
                ev.get("old_y"),

                ev.get("new_x"),
                ev.get("new_y"),

                ev.get("x"),
                ev.get("y"),
                ev.get("yaw"),

                ev.get("hits"),
                ev.get("missed"),
                ev.get("age"),

                ev.get("reason"),

                ev.get("dropped_cam"),
                ev.get("dropped_local_id"),
                ev.get("dropped_x"),
                ev.get("dropped_y"),

                ev.get("kept_cam"),
                ev.get("kept_local_id"),
                ev.get("kept_x"),
                ev.get("kept_y"),

                ev.get("threshold"),

                ev.get("max_jump"),
                ev.get("max_conflict_match_dist"),

                ev.get("best_global_id"),
                ev.get("second_global_id"),

                ev.get("best_distance"),
                ev.get("second_distance"),

                ev.get("track_x"),
                ev.get("track_y"),

                ev.get("nearby_global_id"),

                ev.get("stamp"),
                ev.get("latest_stamp"),
            ])


print(f"Saved CSV files in: {out_dir}")

print(f"Tracks CSV: {tracks_csv}")
print(f"Detections CSV: {detections_csv}")
print(f"Events CSV: {events_csv}")
print(f"Debug events CSV: {debug_events_csv}")
print(f"Ground truth CSV: {gt_csv}")
print(f"Robot CSV: {robot_csv}")
print(f"Local tracker 3D debug CSV: {local_debug_csv}")