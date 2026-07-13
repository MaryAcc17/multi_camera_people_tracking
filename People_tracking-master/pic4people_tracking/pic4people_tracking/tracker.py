import traceback
import json
import time
import math
import cv2
import numpy as np
import torch
from pathlib import Path
from nav_msgs import msg
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
import rclpy.time
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from realsense2_camera_msgs.msg import RGBD
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
from message_filters import Subscriber, ApproximateTimeSynchronizer
from pic4people_tracking.StrongSORT.trackers.strongsort.strong_sort_pose import StrongSORT
from pic4people_tracking.utils.visual import *
from pic4people_tracking.utils.yolov8pose_openvino import yolov8pose_OpenVINO
from tf2_geometry_msgs import do_transform_pose, do_transform_vector3
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from people_msgs.msg import People, Person
from geometry_msgs.msg import Point, Pose, Vector3Stamped, TransformStamped, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
from rclpy.duration import Duration

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

def avg_angles(a, b, c=None):
    x1, y1 = np.cos(a), np.sin(a)
    x2, y2 = np.cos(b), np.sin(b)

    if c:
        x3, y3 = np.cos(c), np.sin(c)
        x = (x1 + x2 + x3)/3
        y = (y1 + y2 + y3)/3
        return np.arctan2(y, x)

    x = (x1 + x2)/2
    y = (y1 + y2)/2

    return np.arctan2(y, x)


class PeopleTracker(Node):
    def __init__(self):
        super().__init__('people_tracker')
        #forcing debug level to INFO (10 is DEBUG, 20 is INFO, 30 is WARN, 40 is ERROR)
        #self.get_logger().set_level(10)
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)
        self.define_parameters()
        self.initialize_variables()
        self.define_publishers()
        self.define_subscribers()
        self.define_timers()
        self.new_data_available = False
        
        self.yolo_ov = yolov8pose_OpenVINO(
            model_filepath='/workspaces/hunavsim_devcontainer/src/People_tracking-master/pic4people_tracking/models/yolov8s-pose_openvino_int8_model/yolov8s-pose.xml',
            min_conf_threshold=self.yolo_min_conf_threshold,
            ros_logger=self.get_logger()
        )
        
    
        self.yolo_ov.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        self.get_logger().info("YOLO warmup done")

        self.get_logger().debug("Initializing process")

    def define_timers(self):
        main_thread     = MutuallyExclusiveCallbackGroup()

        self.main = self.create_timer(
            0.05,
            self.main_callback,
            callback_group=main_thread
            )
        
    def define_publishers(self):
        self.tracked_people_publisher = self.create_publisher(People, self.tracked_people_topic, 10)
        self.image_pub = self.create_publisher(Image, self.output_image_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/people_points', 10)

        # Debug topic used for offline analysis of the local tracker.
        # It publishes one JSON message per processed frame with:
        # - 3D source of each detection: keypoints_median / bbox_center_fallback / dropped
        # - camera-frame 3D position
        # - map-frame position after TF
        # - local StrongSORT track ID and reliability when available
        self.local_debug_pub = self.create_publisher(String, '/local_tracker_debug', 10)
    
    def define_subscribers(self):
        sensor_thread   = MutuallyExclusiveCallbackGroup()
        if self.compressed:
            self.color_type = CompressedImage
            self.bridge_func = self.bridge.compressed_imgmsg_to_cv2
        else:
            self.color_type = Image
            self.bridge_func = self.bridge.imgmsg_to_cv2

        
        # --- buffer for data---
        #self.latest_color = None
        #self.latest_depth = None
        #self.latest_info = None

        # --- RGBD case ---
        if self.rgbd_option:
            self.rgbd_sub = self.create_subscription(
                RGBD,
                self.rgbd_topic,
                self.rgbd_callback,
                qos_profile=qos_profile_sensor_data,
                callback_group=sensor_thread
            )

        # --- standard case ---
        else:
            self.color_sub = Subscriber(
                self,
                self.color_type,
                self.color_topic,
                qos_profile=qos_profile_sensor_data,
                callback_group=sensor_thread
            )

            self.depth_sub = Subscriber(
                self,
                Image,
                self.depth_topic,
                qos_profile=qos_profile_sensor_data,
                callback_group=sensor_thread
            )

            self.info_sub = Subscriber(
                self,
                CameraInfo,
                self.info_topic,
                #qos_profile=qos_profile_sensor_data,   # same qos
                qos_profile=QoSProfile(depth=10),
                callback_group=sensor_thread
            )

            self.ts = ApproximateTimeSynchronizer(
                [self.color_sub, self.depth_sub, self.info_sub],
                queue_size=20,
                slop=0.1   # augmented
            )

            self.ts.registerCallback(self.sync_msgs_callback)
        # ODOM
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            qos_profile=qos_profile_sensor_data
        )
        
    def color_callback(self, msg):
         self.latest_color = msg

    def depth_callback(self, msg):
        self.latest_depth = msg

    def info_callback(self, msg):
        self.latest_info = msg
        
    def define_parameters(self):
        self.declare_parameter('visualize', True)
        self.declare_parameter('rgbd_option', False)
        self.declare_parameter('compressed', False)
        self.declare_parameter('rgbd_topic', '/camera/rgbd')
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('info_topic', '/camera/color/camera_info')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('tracked_people_topic', '/tracked_people')
        self.declare_parameter('output_image_topic', '/result_image')
        self.declare_parameter('fixed_frame','map')
        self.declare_parameter('camera_frame','camera_color_optical_frame')
        self.declare_parameter('robot_frame','base_link')
        self.declare_parameter('yolo_model', '/root/people_tracking_proj/models/yolo_models/yolov8s-pose_openvino_int8_model/yolov8s-pose.xml')
        self.declare_parameter('reid_model', '/root/people_tracking_proj/models/reid_models/osnet_x0_25_market1501_int8_openvino_mixed_model')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('max_distance', 5.0)
        self.declare_parameter('yolo_min_conf_threshold', 0.45)
        self.declare_parameter('max_shoulder_hip_ratio', 0.65)
        self.declare_parameter('max_hip_shoulder_hip_ratio', 0.42)
        self.declare_parameter('fallback_enable', True)
        
        #fallback parameters
        self.declare_parameter('fallback_min_valid_keypoints', 3)
        self.declare_parameter('fallback_min_box_score', 0.65)
        self.declare_parameter('fallback_min_box_width', 20)
        self.declare_parameter('fallback_min_box_height', 40)
        self.declare_parameter('fallback_patch_radius', 5)
        self.declare_parameter('fallback_min_valid_points', 8)
        self.declare_parameter('fallback_min_valid_fraction', 0.20)
        self.declare_parameter('fallback_max_depth_std', 0.35)

        # Maximum allowed difference between RGB-D image timestamp and TF timestamp.
        # If the transform at the exact image time is not available or is too far,
        # the frame is dropped instead of using Time(0), because Time(0) can create
        # map positions that are shifted by several seconds of robot motion.
        self.declare_parameter('max_tf_time_diff', 0.15)

        self.W = self.get_parameter('width').value
        self.H = self.get_parameter('height').value
        self.fps = self.get_parameter('fps').value
        self.visualize = self.get_parameter('visualize').value
        self.max_distance = self.get_parameter('max_distance').value 
        self.rgbd_option = self.get_parameter('rgbd_option').value 
        self.compressed = self.get_parameter('compressed').value 
        self.rgbd_topic = self.get_parameter('rgbd_topic').value
        self.color_topic = self.get_parameter('color_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.info_topic = self.get_parameter('info_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.tracked_people_topic = self.get_parameter('tracked_people_topic').value
        self.output_image_topic = self.get_parameter('output_image_topic').value
        self.fixed_frame = self.get_parameter('fixed_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.robot_frame = self.get_parameter('robot_frame').value
        self.yolo_model = self.get_parameter('yolo_model').value
        self.reid_model = self.get_parameter('reid_model').value
        self.yolo_min_conf_threshold = self.get_parameter('yolo_min_conf_threshold').value
        self.max_shoulder_hip_ratio = self.get_parameter('max_shoulder_hip_ratio').value
        self.max_hip_shoulder_hip_ratio = self.get_parameter('max_hip_shoulder_hip_ratio').value
        
        #reading fallback parameters        
        self.fallback_enable = self.get_parameter('fallback_enable').value
        self.fallback_min_valid_keypoints = self.get_parameter('fallback_min_valid_keypoints').value
        self.fallback_min_box_score = self.get_parameter('fallback_min_box_score').value
        self.fallback_min_box_width = self.get_parameter('fallback_min_box_width').value
        self.fallback_min_box_height = self.get_parameter('fallback_min_box_height').value
        self.fallback_patch_radius = self.get_parameter('fallback_patch_radius').value
        self.fallback_min_valid_points = self.get_parameter('fallback_min_valid_points').value
        self.fallback_min_valid_fraction = self.get_parameter('fallback_min_valid_fraction').value
        self.fallback_max_depth_std = self.get_parameter('fallback_max_depth_std').value
        self.max_tf_time_diff = float(self.get_parameter('max_tf_time_diff').value)

        self.get_logger().info(f"Yolo model: {self.yolo_model}")
        self.get_logger().info(f"Re-ID model: {self.reid_model}")

    def initialize_variables(self):
        self.main_timestamp = self.get_clock().now()
        self.prev_main_timestamp = self.get_clock().now()
        self.bridge = CvBridge()
        self.data = None
        self.display_info = None
        self.map_camera_transform = None
        self.robot_velocity = None
        self.robot_info = None

        # Stores detection-level debug info for the last processed frame.
        self.last_3d_debug = []
        self.last_local_debug_payload = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.tracker = StrongSORT(model_weights=Path(self.reid_model), device=torch.device("cpu"), fp16=True)
        

    def sync_msgs_callback(self, msg1, msg2, msg3):
        self.get_logger().info("Synchronized messages received\n")
        color_image            = self.bridge_func(msg1, 'bgr8')
        frame_timestamp        = rclpy.time.Time.from_msg(msg1.header.stamp)
        
        self.new_data_available = True

        aligned_depth_image = self.bridge.imgmsg_to_cv2(msg2, msg2.encoding)
        aligned_depth_image = np.nan_to_num(aligned_depth_image, nan=0.0, posinf=0.0, neginf=0.0)  
        aligned_depth_image[aligned_depth_image > self.max_distance] = 0.0   #depth in m
        aligned_depth_image = np.array(aligned_depth_image, dtype=np.float32) 
        
        self.get_logger().debug(f"Depth encoding: {msg2.encoding}")
        self.get_logger().debug(
            f"Depth stats raw: min={np.min(aligned_depth_image)}, max={np.max(aligned_depth_image)}"
        )
        
        self.get_logger().debug(f"Depth min: {aligned_depth_image.min()}, max: {aligned_depth_image.max()}")
        self.get_logger().debug(f"Depth sample [240,320]: {aligned_depth_image[240,320]}")
        
        fx = msg3.k[0]
        fy = msg3.k[4]
        cx = msg3.k[2]
        cy = msg3.k[5]
        
        valid = np.count_nonzero(aligned_depth_image > 0)
        total = aligned_depth_image.size

        self.get_logger().debug(f"Valid depth: {valid}/{total} ({valid/total*100:.2f}%)")

        self.data = [color_image, frame_timestamp, aligned_depth_image, fx, fy, cx, cy]

    def odom_callback(self, msg):
        self.robot_velocity = msg.twist.twist.linear.x

    def get_camera_transform(self):
        """
        Look up fixed_frame <- camera_frame transform at the RGB-D image timestamp.

        IMPORTANT:
        We do NOT fall back to rclpy.time.Time(seconds=0) anymore.
        Time(0) returns the latest available TF, which can be several seconds away
        from the image timestamp during rosbag replay. That creates local detections
        in map that are shifted far from the ground truth.

        If the timestamped TF is unavailable, or if the returned TF timestamp is too
        far from the image timestamp, this function sets self.map_camera_transform
        to None. The current frame will then be dropped.
        """
        self.map_camera_transform = None

        if not hasattr(self, "frame_timestamp"):
            self.get_logger().warn("[TF DROP] frame_timestamp not available")
            return

        try:
            image_time = self.frame_timestamp

            tf = self.tf_buffer.lookup_transform(
                self.fixed_frame,
                self.camera_frame,
                image_time,
                timeout=Duration(seconds=0.10),
            )

            image_stamp_sec = image_time.nanoseconds * 1e-9
            tf_stamp_sec = self._stamp_to_float(tf.header.stamp)
            tf_dt = abs(tf_stamp_sec - image_stamp_sec)

            if not np.isfinite(tf_stamp_sec):
                self.get_logger().warn(
                    f"[TF DROP] invalid TF stamp for {self.fixed_frame} <- {self.camera_frame}"
                )
                return

            if tf_dt > self.max_tf_time_diff:
                self.get_logger().warn(
                    f"[TF DROP] TF too far from image stamp: "
                    f"image={image_stamp_sec:.3f}, tf={tf_stamp_sec:.3f}, "
                    f"dt={tf_dt:.3f}s > max={self.max_tf_time_diff:.3f}s"
                )
                return

            self.map_camera_transform = tf
            tr = tf.transform.translation
            self.get_logger().info(
                f"[TF OK] {self.fixed_frame} <- {self.camera_frame}: "
                f"image={image_stamp_sec:.3f}, tf={tf_stamp_sec:.3f}, "
                f"dt={tf_dt:.3f}s, "
                f"trans=({tr.x:.3f},{tr.y:.3f},{tr.z:.3f})"
            )

        except TransformException as e:
            self.get_logger().warn(
                f"[TF DROP] no timestamped transform "
                f"{self.fixed_frame} <- {self.camera_frame} "
                f"at image stamp: {e}"
            )
            self.map_camera_transform = None

    def get_robot_transform(self):
        """
        Used to take the robot velocities w.r.t. map (and not just odom).
        """
        try:
            map_robot_transform = self.tf_buffer.lookup_transform(
                self.fixed_frame, 
                self.robot_frame,
                rclpy.time.Time(seconds=0)
            )
            q = map_robot_transform.transform.rotation
            orientation_q = [q.x, q.y, q.z, q.w]

            _, _, robot_yaw = euler_from_quaternion(orientation_q)
            if self.robot_velocity is None:
                robot_vel = 0.
            else:
                robot_vel = self.robot_velocity
            self.robot_info = [robot_vel, robot_yaw]

        except TransformException as e:
            self.get_logger().warn(f"Failed to look up transform: {e}")
            self.robot_info = None

    def get_robot_velocities(self):
        v, theta = self.robot_info

        return v*math.cos(theta), v*math.sin(theta)

    def get_distance(self, pixel):
        '''
        The output is given in a reference frame with z forward (exiting the camera), x on the left, and y upward.
        '''
        px, py = pixel
        try:
            depth = self.aligned_depth_image[py, px]
        except:
            return None
        
        #filtering out invalid depth values
        if depth == 0.0 or np.isnan(depth) or np.isinf(depth):
            return None

        # --- DEBUG DEPTH FOR PIXEL ---
        #self.get_logger().debug(f"Pixel {pixel} -> Depth {depth}")
        
        return depth
    
    def deproject_pixel_to_point(self, px, py):
        
        z = self.get_distance((px, py))
        #correction formula 
        x = ((px - self.cx)*z)/self.fx
        y = ((py - self.cy)*z)/self.fy
        
        # --- DEBUG 3D POSITION ---
        #self.get_logger().debug(f"Pixel ({px},{py}) -> 3D point: x={x:.3f}, y={y:.3f}, z={z:.3f}")
        
        return [x, y, z]

    def get_keypoint_pos(self, px, py):
        """
        This method deproject the pixel (px, py) if it is a valid point of the segmentation mask.
        """
        
        if not (0 <= px < self.W and 0 <= py < self.H):
          return None

        z = self.get_distance((px, py))

        # invalid depth
        if z is None:
            return None

        # out of range depth
        if z <= 0.1 or z > self.max_distance:
            return None
       
        # deprojection only for valid depth points 
        point = self.deproject_pixel_to_point(px, py)
            
        return np.array(point)
            
    def get_bbox_center_fallback(self, xyxy, score):
        """
        Controlled fallback around bbox center.
        It is accepted only if bbox confidence, bbox size and local depth support
        satisfy minimum validity constraints.
        """
        x1, y1, x2, y2 = xyxy

        # 1. Check bbox confidence
        if score < self.fallback_min_box_score:
            self.get_logger().warn(
                f"[FALLBACK REJECT] low bbox score: {score:.3f}"
            )
            return None

        # 2. Check bbox size
        bw = int(x2 - x1)
        bh = int(y2 - y1)

        if bw < self.fallback_min_box_width or bh < self.fallback_min_box_height:
            self.get_logger().warn(
                f"[FALLBACK REJECT] bbox too small: "
                f"w={bw}, h={bh}, "
                f"xyxy=({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}), "
                f"limits=({self.fallback_min_box_width}, {self.fallback_min_box_height})"
            )
            return None

        # 3. Compute bbox center
        cx_bb = int((x1 + x2) / 2)
        cy_bb = int((y1 + y2) / 2)

        radius = self.fallback_patch_radius
        valid_points = []
        valid_depths = []

        # 4. Search valid 3D points in local patch
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                px = cx_bb + dx
                py = cy_bb + dy

                point = self.get_keypoint_pos(px, py)
                if point is not None:
                    valid_points.append(point)
                    valid_depths.append(point[2])

        patch_size = (2 * radius + 1) ** 2
        valid_count = len(valid_points)
        valid_fraction = valid_count / float(patch_size)

        # 5. Check enough valid points
        if valid_count < self.fallback_min_valid_points:
            self.get_logger().warn(
                f"[FALLBACK REJECT] too few valid points: {valid_count}"
            )
            return None

        # 6. Check enough valid fraction
        if valid_fraction < self.fallback_min_valid_fraction:
            self.get_logger().warn(
                f"[FALLBACK REJECT] valid fraction too low: {valid_fraction:.3f}"
            )
            return None

        valid_points = np.array(valid_points, dtype=np.float32)
        valid_depths = np.array(valid_depths, dtype=np.float32)

        # 7. Check depth consistency
        depth_std = float(np.std(valid_depths))
        if depth_std > self.fallback_max_depth_std:
            self.get_logger().warn(
                f"[FALLBACK REJECT] depth std too high: {depth_std:.3f}"
            )
            return None

        # 8. Final robust estimate
        pos3D = np.median(valid_points, axis=0)

        self.get_logger().warn(
            f"[FALLBACK OK] center=({cx_bb},{cy_bb}) "
            f"valid={valid_count}/{patch_size} "
            f"frac={valid_fraction:.2f} "
            f"depth_std={depth_std:.3f} -> {pos3D}"
        )

        return pos3D  

    
    def get_3Dpositions(self, xyxys, keypoints, boxes_with_scores=None):
        """
        Estimate one 3D position for each 2D detection and store rich debug
        metadata in self.last_3d_debug.

        Each position is either:
        - np.array([x, y, z]) in camera optical frame
        - None if the detection is dropped

        Debug fields include:
        - method: keypoints_median / bbox_center_fallback / dropped
        - number of valid keypoints
        - bbox and score
        - camera-frame 3D position
        """
        self.get_logger().info(f"get_3Dpositions called - dets: {len(xyxys)}")

        self.last_3d_debug = []

        if len(xyxys) == 0 or keypoints is None:
            self.get_logger().warn("No detections or null keypoints")
            return []

        positions3D = []

        for det_idx, (xyxy, kpoints) in enumerate(zip(xyxys, keypoints)):
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            score = np.nan
            if boxes_with_scores is not None and det_idx < len(boxes_with_scores):
                score = float(boxes_with_scores[det_idx][4])

            debug_row = {
                "det_index": int(det_idx),
                "method": "dropped",
                "reason": "not_processed",
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "bbox_score": score,
                "num_valid_keypoints": 0,
                "fallback_valid_count": np.nan,
                "fallback_valid_fraction": np.nan,
                "fallback_depth_std": np.nan,
                "camera_x": np.nan,
                "camera_y": np.nan,
                "camera_z": np.nan,
            }

            valid_points = []
            valid_kp_indices = []

            # 1. Primary 3D estimate from valid keypoints
            for kp_idx, kpoint in enumerate(kpoints):
                px = int(kpoint[0])
                py = int(kpoint[1])
                conf = float(kpoint[2])

                if conf < 0.45:
                    continue

                point3D = self.get_keypoint_pos(px, py)
                if point3D is not None:
                    valid_points.append(point3D)
                    valid_kp_indices.append(int(kp_idx))

            debug_row["num_valid_keypoints"] = int(len(valid_points))
            debug_row["valid_keypoint_indices"] = valid_kp_indices

            pos3D = None

            # 2. Use primary estimate if enough valid keypoints
            if len(valid_points) >= self.fallback_min_valid_keypoints:
                valid_points_np = np.array(valid_points, dtype=np.float32)
                pos3D = np.median(valid_points_np, axis=0)

                debug_row.update({
                    "method": "keypoints_median",
                    "reason": "ok",
                    "camera_x": float(pos3D[0]),
                    "camera_y": float(pos3D[1]),
                    "camera_z": float(pos3D[2]),
                })

                self.get_logger().warn(
                    f"[3D DEBUG][KEYPOINTS] det={det_idx} "
                    f"valid_kpts={len(valid_points)} idx={valid_kp_indices} "
                    f"bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) "
                    f"score={score:.3f} "
                    f"camera=({pos3D[0]:.3f},{pos3D[1]:.3f},{pos3D[2]:.3f})"
                )

            # 3. Otherwise try controlled fallback
            elif self.fallback_enable:
                fallback = self.get_bbox_center_fallback(xyxy, score if np.isfinite(score) else 1.0)

                if fallback is not None:
                    pos3D = fallback
                    debug_row.update({
                        "method": "bbox_center_fallback",
                        "reason": "ok",
                        "camera_x": float(pos3D[0]),
                        "camera_y": float(pos3D[1]),
                        "camera_z": float(pos3D[2]),
                    })
                    self.get_logger().warn(
                        f"[3D DEBUG][FALLBACK_USED] det={det_idx} "
                        f"valid_kpts={len(valid_points)} "
                        f"bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) "
                        f"score={score:.3f} "
                        f"camera=({pos3D[0]:.3f},{pos3D[1]:.3f},{pos3D[2]:.3f})"
                    )
                else:
                    debug_row.update({
                        "method": "dropped",
                        "reason": "insufficient_keypoints_and_fallback_rejected",
                    })
                    self.get_logger().warn(
                        f"[3D DEBUG][DROP] det={det_idx}: insufficient keypoints "
                        f"({len(valid_points)}) and fallback rejected"
                    )
            else:
                debug_row.update({
                    "method": "dropped",
                    "reason": "insufficient_keypoints_and_fallback_disabled",
                })
                self.get_logger().warn(
                    f"[3D DEBUG][DROP] det={det_idx}: insufficient keypoints "
                    f"({len(valid_points)}) and fallback disabled"
                )

            positions3D.append(pos3D)
            self.last_3d_debug.append(debug_row)

        valid_count = sum(p is not None for p in positions3D)
        self.get_logger().info(
            f"positions3D total: {len(positions3D)}, valid: {valid_count}"
        )

        return positions3D


    def estimate_orientation_from_points(self, left_point, right_point):
        if left_point[2] < 0.6 or right_point[2] < 0.6:
            return None
        
        left_3D = self.get_keypoint_pos(int(left_point[0]), int(left_point[1]))
        right_3D = self.get_keypoint_pos(int(right_point[0]), int(right_point[1]))
        
        if left_3D is None or right_3D is None:
            return None
        
        vector = right_3D - left_3D
        if np.linalg.norm(vector) < 0.1:  # points not sufficiently apart
            return None
        
        orientation_vector = np.array([-vector[2], 0.0, vector[0]])
        orientation = np.arctan2(-orientation_vector[0], orientation_vector[2])
        
        return orientation

    def estimate_orientation_from_ratio(self, l_eye, r_eye, l_ear, r_ear, l_sh, r_sh, l_hip, r_hip):
        l_eye2D = [int(l_eye[0]), int(l_eye[1])] if l_eye[2] > 0.6 else None
        r_eye2D = [int(r_eye[0]), int(r_eye[1])] if r_eye[2] > 0.6 else None
        l_ear2D = [int(l_ear[0]), int(l_ear[1])] if l_ear[2] > 0.6 else None
        r_ear2D = [int(r_ear[0]), int(r_ear[1])] if r_ear[2] > 0.6 else None 
        l_sh2D = [int(l_sh[0]), int(l_sh[1])] if l_sh[2] > 0.5 else None
        r_sh2D =  [int(r_sh[0]), int(r_sh[1])] if r_sh[2] > 0.5 else None
        l_hip2D = [int(l_hip[0]), int(l_hip[1])] if l_hip[2] > 0.5 else None
        r_hip2D = [int(r_hip[0]), int(r_hip[1])] if r_hip[2] > 0.5 else None
        
        if any(point is None for point in [l_sh2D, r_sh2D, l_hip2D, r_hip2D]):
            return None, None
        
        shoulder_distance = r_sh2D[0] - l_sh2D[0]
        hip_distance = r_hip2D[0] - l_hip2D[0]
        shoulder_hip_distance = -(l_sh2D[1] - l_hip2D[1])
        
        if shoulder_hip_distance < 0.1:  # Avoid division by zero
            return None, None
        
        shoulder_hip_ratio = shoulder_distance / shoulder_hip_distance
        hip_shoulder_hip_ratio = hip_distance / shoulder_hip_distance
        
        normalized_shoulder_ratio = max(min(shoulder_hip_ratio / self.max_shoulder_hip_ratio, 1.0), -1.0)
        normalized_hip_ratio = max(min(hip_shoulder_hip_ratio / self.max_hip_shoulder_hip_ratio, 1.0), -1.0)
        
        # Convert ratios to angles
        shoulder_angle = np.arccos(normalized_shoulder_ratio)
        hip_angle = np.arccos(normalized_hip_ratio)
        
        # Average the two angles. The obtained orientation has a +- ambiguity
        orientation = avg_angles(shoulder_angle, hip_angle)

        if (l_eye2D or l_ear2D) and not r_ear2D:
            sign = 1.0 
        elif (r_eye2D or r_ear2D) and not l_ear2D:
            sign = -1.0
        else:
            sign = None

        return orientation, sign
    
    def combine_orientations(self, o1, o2, o_ratio, sign_ratio):
        
        if o1 is None and o2 is None:
            return sign_ratio * o_ratio if sign_ratio is not None else None
        
        if o1 is None or o2 is None:
            o = o2 if o1 is None else o1
            return avg_angles(o, (sign_ratio or np.sign(o)) * o_ratio) if o_ratio is not None else o
        
        return avg_angles(o1, o2, (np.sign(avg_angles(o1, o2))) * o_ratio) if o_ratio is not None else avg_angles(o1, o2)

    def get_orientations(self, keypoints):
        orientations = []

        if keypoints is None:
            return []

        for idx, kpoints in enumerate(keypoints):
            indices = [
                1,   # left eye
                2,   # right eye
                3,   # left ear
                4,   # right ear
                5,   # left shoulder
                6,   # right shoulder
                11,  # left hip
                12   # right hip
            ]

            l_eye, r_eye, l_ear, r_ear, l_sh, r_sh, l_hip, r_hip = kpoints[indices]

            shoulder_orientation = self.estimate_orientation_from_points(l_sh, r_sh)
            hip_orientation = self.estimate_orientation_from_points(l_hip, r_hip)
            ratio_orientation, ratio_sign = self.estimate_orientation_from_ratio(
                l_eye, r_eye, l_ear, r_ear, l_sh, r_sh, l_hip, r_hip
            )

            orientation = self.combine_orientations(
                shoulder_orientation, hip_orientation, ratio_orientation, ratio_sign
            )
            orientations.append(orientation)

        return orientations
   
       
      
    def get_dets(self, boxes, positions3D, orientations):
        """
        This method concatenates bounding boxes and 3D positions to be used by the SortRGBD tracking algorithm

        return: 
        dets - a numpy array of detections 

        Output format:
        [[x1, y1, x2, y2, x, y, z, yaw, score], [x1, y1, x2, y2, x, y, z, yaw, score],...]
        """
        dets = []

        n_boxes = len(boxes)
        n_pos = len(positions3D)
        n_ori = len(orientations)

        if n_boxes != n_pos or n_boxes != n_ori:
            self.get_logger().error(
                f"[MISMATCH] boxes={n_boxes}, positions3D={n_pos}, orientations={n_ori}"
            )

        for i in range(n_boxes):
            box = boxes[i]
            pos = positions3D[i] if i < n_pos else None
            yaw = orientations[i] if i < n_ori else None

            if pos is None:
                self.get_logger().warn(f"[DET DROP] det {i}: missing 3D position")
                continue

            x1, y1, x2, y2, score = box
            x, y, z = pos
            yaw = yaw if yaw is not None else np.nan

            dets.append([x1, y1, x2, y2, x, y, z, yaw, score])

        if len(dets) == 0:
            return np.empty((0, 9), dtype=float)

        return np.array(dets, dtype=float)

    def yolov8_inference(self, frame):

        results = self.yolo_ov.detect(frame)[0]
        return results

    def track(self, dets, img):

        if len(dets) == 0:
            dets = np.empty((0,9))

        # self.get_logger().info(f"dets: {dets[:, 6]}")

        dt = 1./23
        if self.main_timestamp != self.prev_main_timestamp:
            dt = self.main_timestamp - self.prev_main_timestamp
            dt = dt.nanoseconds/10**9
        # self.get_logger().info(f"dt: {dt}")
        tracks = self.tracker.update(dets, img, dt=dt)
        velocities = self.tracker.get_3Dvelocities()
        reliabilities = self.tracker.get_reliabilities()
        yaws = self.tracker.get_yaws()

        print("BB tracked: ", tracks)

        return tracks, velocities, yaws, reliabilities
    
    def get_absolute_velocities(self, vel_vector):

        vx, vy = self.get_robot_velocities()
        # self.get_logger().info(f"Robot velocities: Vx: {vx}, Vy: {vy}")

        return [vel_vector.vector.x + vx, vel_vector.vector.y + vy]
    
    def get_color(self, ID):
        n = ID % 5
        if n == 0:
            return [1.,0.,0.]
        if n == 1:
            return [0.,1.,0.]
        if n == 2:
            return [0.,0.,1.]
        if n == 3:
            return [1.,1.,0.]
        if n == 4:
            return [1.,0.,1.]
    
    def _stamp_to_float(self, stamp_msg):
        if stamp_msg is None:
            return np.nan
        return float(stamp_msg.sec) + float(stamp_msg.nanosec) * 1e-9

    def publish_local_tracker_debug(self, tracks=None, yaws=None, velocities=None, reliabilities=None):
        """
        Publish debug information from the local tracker as JSON.

        The message is intended to be recorded in the second rosbag and later
        extracted into local_tracker_3d_debug.csv. It allows us to diagnose
        whether the error comes from:
        - 3D estimation in camera frame
        - camera->map TF
        - StrongSORT local tracking
        """
        if not hasattr(self, 'local_debug_pub'):
            return

        payload = {
            "time": self._stamp_to_float(self.frame_timestamp.to_msg()) if hasattr(self, 'frame_timestamp') else np.nan,
            "frame_id": self.fixed_frame,
            "camera_frame": self.camera_frame,
            "source_topic": self.tracked_people_topic,
            "image_stamp": self._stamp_to_float(self.frame_timestamp.to_msg()) if hasattr(self, 'frame_timestamp') else np.nan,
            "transform_stamp": np.nan,
            "used_transform_available": self.map_camera_transform is not None,
            "detections": [],
            "tracks": [],
        }

        if self.map_camera_transform is not None:
            payload["transform_stamp"] = self._stamp_to_float(self.map_camera_transform.header.stamp)

        # Detection-level debug: project each 3D camera point to map when possible.
        for d in getattr(self, 'last_3d_debug', []):
            row = dict(d)
            row["map_x"] = np.nan
            row["map_y"] = np.nan
            row["map_z"] = np.nan

            if self.map_camera_transform is not None and np.isfinite(row.get("camera_x", np.nan)):
                pose = Pose()
                pose.position.x = float(row["camera_x"])
                pose.position.y = float(row["camera_y"])
                pose.position.z = float(row["camera_z"])
                pose.orientation.w = 1.0
                try:
                    pose_t = do_transform_pose(pose, self.map_camera_transform)
                    row["map_x"] = float(pose_t.position.x)
                    row["map_y"] = float(pose_t.position.y)
                    row["map_z"] = float(pose_t.position.z)
                except Exception as e:
                    row["reason"] = f"map_transform_failed: {e}"

            payload["detections"].append(row)

        # Track-level debug: local StrongSORT output after filtering/reliability.
        if tracks is not None and yaws is not None and velocities is not None and reliabilities is not None:
            for track, yaw, vel, rel in zip(tracks, yaws, velocities, reliabilities):
                try:
                    local_id = int(track[7])
                    det_index = int(track[8])
                    camera_x = float(track[4])
                    camera_y = float(track[5])
                    camera_z = float(track[6])
                    rel_value = float(rel[0])

                    pose = Pose()
                    pose.position.x = camera_x
                    pose.position.y = camera_y
                    pose.position.z = camera_z
                    pose.orientation.w = 1.0

                    map_x = map_y = map_z = np.nan
                    if self.map_camera_transform is not None:
                        pose_t = do_transform_pose(pose, self.map_camera_transform)
                        map_x = float(pose_t.position.x)
                        map_y = float(pose_t.position.y)
                        map_z = float(pose_t.position.z)

                    method = "unknown"
                    reason = "unknown"
                    bbox_score = np.nan
                    num_valid_keypoints = np.nan
                    if 0 <= det_index < len(getattr(self, 'last_3d_debug', [])):
                        det_dbg = self.last_3d_debug[det_index]
                        method = det_dbg.get("method", "unknown")
                        reason = det_dbg.get("reason", "unknown")
                        bbox_score = det_dbg.get("bbox_score", np.nan)
                        num_valid_keypoints = det_dbg.get("num_valid_keypoints", np.nan)

                    payload["tracks"].append({
                        "local_id": local_id,
                        "det_index": det_index,
                        "method": method,
                        "reason": reason,
                        "bbox_score": bbox_score,
                        "num_valid_keypoints": num_valid_keypoints,
                        "camera_x": camera_x,
                        "camera_y": camera_y,
                        "camera_z": camera_z,
                        "map_x": map_x,
                        "map_y": map_y,
                        "map_z": map_z,
                        "yaw": float(yaw[0]) if len(yaw) > 0 else np.nan,
                        "velocity_camera_x": float(vel[0]),
                        "velocity_camera_y": float(vel[1]) if len(vel) > 1 else np.nan,
                        "velocity_camera_z": float(vel[2]) if len(vel) > 2 else np.nan,
                        "reliability": rel_value,
                        "published_by_local_tracker": bool(rel_value >= 0.4),
                    })
                except Exception as e:
                    self.get_logger().warn(f"[LOCAL DEBUG] failed to serialize track debug: {e}")

        msg = String()
        # Convert numpy values to native Python/JSON values.
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [clean(v) for v in obj]
            if isinstance(obj, tuple):
                return [clean(v) for v in obj]
            if isinstance(obj, np.ndarray):
                return clean(obj.tolist())
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            if isinstance(obj, float) and not np.isfinite(obj):
                return None
            return obj

        msg.data = json.dumps(clean(payload))
        self.local_debug_pub.publish(msg)



    def publish_tracking_results(self, tracks, yaws, velocities, reliabilities=[]):
        """
        Publish results expressed in the self.fixed_frame        
        """
        self.get_logger().debug("=== publish_tracking_results called ===")
        if self.map_camera_transform is None or self.robot_info is None:
            self.get_logger().error("publish aborted: map_camera_transform is None")
            return
        
        if self.robot_info is None:
            self.get_logger().error("publish aborted: robot_info is None")
            return

        self.get_logger().debug(f"len(tracks)={len(tracks) if tracks is not None else 'None'}")
        self.get_logger().debug(f"len(yaws)={len(yaws) if yaws is not None else 'None'}")
        self.get_logger().debug(f"len(velocities)={len(velocities) if velocities is not None else 'None'}")
        self.get_logger().debug(f"len(reliabilities)={len(reliabilities) if reliabilities is not None else 'None'}")

        transform_vectors = TransformStamped()
        transform_vectors.transform.translation.x = 0.0
        transform_vectors.transform.translation.y = 0.0
        transform_vectors.transform.translation.z = 0.0
        transform_vectors.transform.rotation.x = self.map_camera_transform.transform.rotation.x
        transform_vectors.transform.rotation.y = self.map_camera_transform.transform.rotation.y
        transform_vectors.transform.rotation.z = self.map_camera_transform.transform.rotation.z
        transform_vectors.transform.rotation.w = self.map_camera_transform.transform.rotation.w

        # self.get_logger().info(f"Transform: {self.map_camera_transform}")

        msg = People()
        marray = MarkerArray()
        # delta_t_timestamp = rclpy.time.Duration(seconds=0, nanoseconds=int(delta_t*10**9))
        
        msg.header.stamp = self.frame_timestamp.to_msg() # (self.frame_timestamp + delta_t_timestamp).to_msg() # 
        
        # msg.header.stamp = self.get_clock().now().to_msg()
        
        msg.header.frame_id = self.fixed_frame
        
        #publish_stamp = self.get_clock().now().to_msg()

        #msg.header.stamp = publish_stamp
        #msg.header.frame_id = self.fixed_frame
        
        count_loop = 0
      
        for track, yaw, vel, rel in zip(tracks, yaws, velocities, reliabilities):
            count_loop += 1

            self.get_logger().info(
                f"[publish loop] track={track}, yaw={yaw}, vel={vel}, rel={rel}"
            )
            
            rel_value = float(rel[0])

            if rel_value < 0.4:
                self.get_logger().warn(
                    f"[LOCAL PUBLISH SKIP] low reliability: "
                    f"ID={int(track[7])} rel={rel_value:.3f} < 0.400"
                )
                continue

            ID = track[7]
            ind = track[8].astype('int')
            x, z = float(track[4]), float(track[6])
            vx, vz = float(vel[0]), float(vel[2])
            yaw = yaw[0] 
            pose = Pose()
           
            pose.position.x = x
            pose.position.y = 0.0
            pose.position.z = z
            q = quaternion_from_euler(math.pi/2, 0.0, math.pi/2 + yaw, axes='rxyz')
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]

            vel_vector = Vector3Stamped()
            vel_vector.vector.x = vx
            vel_vector.vector.y = 0.0
            vel_vector.vector.z = vz
            
            pose_t = do_transform_pose(pose, self.map_camera_transform)
            self.get_logger().warn(
                f"[LOCAL MAP DEBUG] ID={int(ID)} det_index={int(ind)} "
                f"camera=({x:.3f},{z:.3f}) -> "
                f"map=({pose_t.position.x:.3f},{pose_t.position.y:.3f}) "
                f"rel={rel_value:.3f}"
            )
            vel_vector_t = do_transform_vector3(vel_vector, transform_vectors)
            yaw_t = euler_from_quaternion((pose_t.orientation.x,
                                           pose_t.orientation.y,
                                           pose_t.orientation.z,
                                           pose_t.orientation.w))[2]
            
            absolute_vel = self.get_absolute_velocities(vel_vector_t)

            # self.get_logger().info(f"ID: {int(ID)}")            
            # self.get_logger().info(f"Relative velocities: Vx: {vel_vector_t.vector.x}, Vy: {vel_vector_t.vector.y}")
            # self.get_logger().info(f"Absolute velocities: Vx: {absolute_vel[0]}, Vy: {absolute_vel[1]}\n")

            tracked_person = Person()
            tracked_person.name = f"Person_{int(ID)}"
            tracked_person.tagnames = ["behaviour", "group_id", "id"]
            tracked_person.tags = ["0", "-1", f"{int(ID)}"]
            tracked_person.position = Point(x=pose_t.position.x,
                                            y=pose_t.position.y, 
                                            z=yaw_t)
            tracked_person.velocity = Point(x=absolute_vel[0],
                                            y=absolute_vel[1],
                                            z=0.0)
            tracked_person.reliability = rel_value
            msg.people.append(tracked_person)
            
            self.get_logger().info(
                f"[publish ok] Person_{int(ID)} map=({pose_t.position.x:.3f},{pose_t.position.y:.3f}) "
                f"vel=({absolute_vel[0]:.3f},{absolute_vel[1]:.3f}) rel={rel_value:.3f}"
            )

            color = self.get_color(ID)
            person_point = Marker()
            person_point.header.stamp = self.frame_timestamp.to_msg()
            person_point.header.frame_id = self.fixed_frame
            person_point.id = int(ID)
            person_point.type = 2
            person_point.scale.x = 0.2
            person_point.scale.y = 0.2
            person_point.scale.z = 0.2
            person_point.lifetime.nanosec = int(5e8)
            person_point.pose.position.x = pose_t.position.x
            person_point.pose.position.y = pose_t.position.y
            person_point.pose.position.z = pose_t.position.z
            person_point.color.r = color[0]
            person_point.color.g = color[1]
            person_point.color.b = color[2]
            person_point.color.a = 1.

            marray.markers.append(person_point)
        
        self.get_logger().debug(f"zip loop iterations: {count_loop}")
        self.get_logger().debug(f"msg.people final len: {len(msg.people)}")

        self.tracked_people_publisher.publish(msg)
        self.marker_pub.publish(marray)

        # Publish local debug even if no person passed the local reliability filter.
        self.publish_local_tracker_debug(tracks, yaws, velocities, reliabilities)
        
        self.get_logger().debug("=== publish_tracking_results end ===")

    def display(self, frame, keypoints, tracks):
        tracked_keypoints = None
        tracked_points = None
        
        try:
            if keypoints is not None and len(tracks) > 0:
                inds = tracks[:, 8].astype('int')
                tracked_keypoints = [keypoints[i] for i in inds] 
            img = plot_tracked_BB_pose3(frame, tracks, keypoints=tracked_keypoints, points=tracked_points)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            image_message = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            try:
                self.image_pub.publish(image_message)
            except CvBridgeError as e:
                print(e)
        except:
            pass

    def main_callback(self):
        # --- use the flag to avoid empty loop ---
        if not self.new_data_available:
            return

        # reset flag
        self.new_data_available = False

        # --- verify that synchronized data is available ---
        if self.data is None:
            self.get_logger().warn("Waiting for synchronized data...")
            return

        self.get_logger().info("Processing frame")

        # --- unpack synchronized data ---
        try:
            color_image, self.frame_timestamp, self.aligned_depth_image, self.fx, self.fy, self.cx, self.cy = self.data
        except Exception as e:
            self.get_logger().error(f"Data unpack failed: {e}")
            return

        self.main_timestamp = self.frame_timestamp

        # --- check for duplicate frames ---
        if self.main_timestamp == self.prev_main_timestamp:
            self.get_logger().warn("Duplicate frame skipped")
            return

        # --- TF ---
        self.get_camera_transform()
        self.get_robot_transform()

        if self.map_camera_transform is None:
            self.get_logger().warn(
                "[FRAME DROP] camera transform unavailable or temporally inconsistent; "
                "skipping this RGB-D frame"
            )
            self.prev_main_timestamp = self.main_timestamp
            return

        if self.robot_info is None:
            self.get_logger().warn(
                "[FRAME DROP] robot transform unavailable; skipping this RGB-D frame"
            )
            self.prev_main_timestamp = self.main_timestamp
            return

        # --- YOLO CALL INFO ---
        #self.get_logger().info("Running YOLO inference")
        #self.get_logger().info(f"TIPO OGGETTO YOLO: {type(self.yolo_ov)}")
        #self.get_logger().info(f"METODO DETECT: {self.yolo_ov.detect}")

        try:
            results = self.yolo_ov.detect(color_image)
        except Exception as e:
            self.get_logger().error(f"YOLO inference failed: {e}")
            return

        # --- debug results ---
        #self.get_logger().info(f"RESULTS RAW TYPE: {type(results)}")
        #self.get_logger().info(f"RESULTS LEN: {len(results)}")

        if len(results) == 0:
            boxes = np.empty((0, 6))
            keypoints = None
        else:
            results = results[0]
            boxes = results['box']
            keypoints = results.get('kpt', None)

        # --- DEBUG detections ---
        self.get_logger().info(f"Detections: {len(boxes)}")

        # --- 3D + tracking ---
        try:
            positions3D = self.get_3Dpositions(
                boxes[:, 0:4],
                keypoints,
                boxes_with_scores=boxes[:, 0:5]
            )
            orientations = self.get_orientations(keypoints)
            dets = self.get_dets(boxes[:, 0:5], positions3D, orientations)
            
            self.get_logger().info(f"boxes len: {len(boxes)}")
            self.get_logger().info(f"positions3D len: {len(positions3D)}")
            self.get_logger().info(f"orientations len: {len(orientations)}")
            self.get_logger().info(f"dets shape: {dets.shape if isinstance(dets, np.ndarray) else 'NOT_NP'}")

            if isinstance(dets, np.ndarray) and len(dets) > 0:
                self.get_logger().info(f"dets content:\n{dets}")
            else:
                self.get_logger().warn("dets is empty before StrongSORT")

            tracks, velocities, yaws, reliabilities = self.track(dets, color_image)
            
            self.get_logger().debug(f"tracks len: {len(tracks) if tracks is not None else 'None'}")
            self.get_logger().debug(f"velocities len: {len(velocities) if velocities is not None else 'None'}")
            self.get_logger().debug(f"yaws len: {len(yaws) if yaws is not None else 'None'}")
            self.get_logger().debug(f"reliabilities len: {len(reliabilities) if reliabilities is not None else 'None'}")

            self.get_logger().debug(f"tracks content: {tracks}")
            self.get_logger().debug(f"velocities content: {velocities}")
            self.get_logger().debug(f"yaws content: {yaws}")
            self.get_logger().debug(f"reliabilities content: {reliabilities}")
        except Exception as e:
            self.get_logger().error(f"Tracking pipeline failed: {e}")
            return

        # --- publish ---
        try:
            self.publish_tracking_results(tracks, yaws, velocities, reliabilities)
        except Exception as e:
            self.get_logger().error(f"Publish failed: {e}")

        # --- visualization ---
        if self.visualize:
            try:
                self.display(color_image, keypoints, tracks)
            except Exception as e:
                self.get_logger().error(f"Display failed: {e}")

        # --- timestamp update ---
        self.prev_main_timestamp = self.main_timestamp

def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    people_tracker = PeopleTracker()
    executor.add_node(people_tracker)

    try:
        executor.spin()
        # rclpy.spin(people_tracker)
    except KeyboardInterrupt:
        people_tracker.stop()
        traceback.print_exc()

    people_tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()