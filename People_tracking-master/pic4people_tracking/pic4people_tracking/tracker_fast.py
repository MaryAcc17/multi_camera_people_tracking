import traceback
import time
import math
import cv2
import numpy as np
import torch
from pathlib import Path
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
        self.get_logger().set_level(10)
        self.define_parameters()
        self.initialize_variables()
        self.define_publishers()
        self.define_subscribers()
        self.define_timers()

        self.get_logger().info("Initializing process")

    def define_timers(self):
        main_thread     = MutuallyExclusiveCallbackGroup()
        pub_thread      = MutuallyExclusiveCallbackGroup()

        self.main = self.create_timer(
            0.01,
            self.main_callback,
            callback_group=main_thread,
            # clock=ROSTime
            )

        self.publisher = self.create_timer(
            0.01,
            self.publisher_callback,
            callback_group=pub_thread
            )
        
    def define_publishers(self):
        self.tracked_people_publisher = self.create_publisher(People, self.tracked_people_topic, qos_profile_sensor_data)
        self.image_pub = self.create_publisher(Image, self.output_image_topic, qos_profile_sensor_data)
        self.marker_pub = self.create_publisher(MarkerArray, '/people_points', qos_profile_sensor_data)
    
    def define_subscribers(self):
        sensor_thread   = MutuallyExclusiveCallbackGroup()
        if self.compressed:
            self.color_type = CompressedImage
            self.bridge_func = self.bridge.compressed_imgmsg_to_cv2
        else:
            self.color_type = Image
            self.bridge_func = self.bridge.imgmsg_to_cv2

        if self.rgbd_option:
            self.rgbd_sub = self.create_subscription(
                RGBD,
                self.rgbd_topic,
                self.rgbd_callback,
                qos_profile=qos_profile_sensor_data,
                callback_group=sensor_thread
                )
        else:
            self.color_sub = Subscriber(
                self,
                self.color_type,
                self.color_topic,
                qos_profile=qos_profile_sensor_data,
                callback_group=sensor_thread
                )
            
            self.aligned_depth_sub = Subscriber(
                self,
                Image,
                self.depth_topic,
                callback_group=sensor_thread,
                qos_profile=qos_profile_sensor_data,
                )
            
            self.cameraInfo_sub = Subscriber(
                self,
                CameraInfo,
                self.info_topic,
                qos_profile=QoSProfile(depth=10),
                callback_group=sensor_thread
            )

            self.ts = ApproximateTimeSynchronizer([self.color_sub,
                                                self.aligned_depth_sub,
                                                self.cameraInfo_sub],
                                                queue_size=10,
                                                slop=0.01)
        
            self.ts.registerCallback(self.sync_msgs_callback)

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            qos_profile=qos_profile_sensor_data
        )

    def define_parameters(self):
        self.declare_parameter('visualize', True)
        self.declare_parameter('rgbd_option', False)
        self.declare_parameter('compressed', False)
        self.declare_parameter('rgbd_topic', '/camera/rgbd')
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('info_topic', '/camera/camera/color/camera_info')
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

        self.get_logger().info(f"Yolo model: {self.yolo_model}")
        self.get_logger().info(f"Re-ID model: {self.reid_model}")

    def initialize_variables(self):
        self.main_timestamp = self.get_clock().now()
        self.prev_main_timestamp = self.get_clock().now()
        self.pub_time = self.get_clock().now()
        self.bridge = CvBridge()
        self.data = None
        self.display_info = None
        self.map_camera_transform = None
        self.robot_velocity = None
        self.robot_info = None
        self.dets = None
        self.tracks = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.ov_model = yolov8pose_OpenVINO(model_filepath=self.yolo_model, min_conf_threshold=self.yolo_min_conf_threshold)
        self.tracker = StrongSORT(model_weights=Path(self.reid_model), device=torch.device("cpu"), fp16=True)
        
        self.yolov8_inference(np.zeros((480, 640, 3), dtype=np.uint8)) # to warm up yolov8 

    def sync_msgs_callback(self, msg1, msg2, msg3):
        color_image            = self.bridge_func(msg1, 'rgb8')
        frame_timestamp        = rclpy.time.Time.from_msg(msg1.header.stamp)

        aligned_depth_image = self.bridge.imgmsg_to_cv2(msg2, msg2.encoding)
        aligned_depth_image = np.nan_to_num(aligned_depth_image, nan=0.0, posinf=0.0, neginf=0.0)  
        aligned_depth_image[aligned_depth_image > self.max_distance*1000] = 0.0  
        aligned_depth_image = np.array(aligned_depth_image, dtype=np.float32) 
        
        fx = msg3.k[0]
        fy = msg3.k[4]
        cx = msg3.k[2]
        cy = msg3.k[5]

        self.data = [color_image, frame_timestamp, aligned_depth_image, fx, fy, cx, cy]

    def odom_callback(self, msg):
        self.robot_velocity = msg.twist.twist.linear.x

    def get_camera_transform(self):
        try:
            self.map_camera_transform = self.tf_buffer.lookup_transform(
                self.fixed_frame, 
                self.camera_frame,
                rclpy.time.Time(seconds=0))
        except TransformException as e:
            self.get_logger().warn(f"Failed to look up transform: {e}")
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
            depth = 0.0
        return depth
    
    def deproject_pixel_to_point(self, px, py):
        
        z = self.get_distance((px, py))/1000
        x = ((px - self.cx)*z)/self.fx
        y = ((px - self.cy)*z)/self.fy
        
        return [x, y, z]

    def get_keypoint_pos(self, px, py):
        """
        This method deproject the pixel (px, py) if it is a valid point of the segmentation mask.
        """
        if self.W>px>=0 and self.H>py>=0 and self.fx is not None:
            
            point = self.deproject_pixel_to_point(px, py)
            if self.max_distance>point[2]>0.0:
                return np.array(point)
            
        return None

    def get_3Dpositions(self, xyxys, keypoints):
        """
        This method calculate the 3D position of each detected bounding box by averaging over main
        body keypoints
        """
        positions3D = []

        if len(xyxys)>0 and keypoints is not None:
    
            for xyxy, kpoints in zip(xyxys, keypoints):
             
                mean_pos = np.zeros([1,3])
                count = 0
                main_kpoints = kpoints
                for kpoint in main_kpoints: #kpoints[kpoints[:,0] != 0]
                    px = int(kpoint[0])
                    py = int(kpoint[1])
                    conf = kpoint[2]
                    if conf<0.45: 
                        continue
                    point3D = self.get_keypoint_pos(px, py)
                    if point3D is not None:
                        mean_pos += point3D
                        count += 1

                if count > 0:
                    mean_pos /= count
                    positions3D.append(mean_pos[0])
                else:
                    positions3D.append([None, None, None])
            
            return np.array(positions3D)

        else:
            return np.empty((0,3),dtype=int)

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
        if keypoints is not None:
            for idx, kpoints in enumerate(keypoints):

                indices = [
                    1, # left eye
                    2, # right eye
                    3, # left ear
                    4, # right ear
                    5, # left shoulder
                    6, # right shoulder
                    11, # left hip
                    12, # right hip
                ] 
                
                l_eye, r_eye, l_ear, r_ear, l_sh, r_sh, l_hip, r_hip = kpoints[indices]

                shoulder_orientation = self.estimate_orientation_from_points(l_sh, r_sh)
                hip_orientation = self.estimate_orientation_from_points(l_hip, r_hip)

                ratio_orientation, ratio_sign = self.estimate_orientation_from_ratio(l_eye, r_eye, l_ear, r_ear, l_sh, r_sh, l_hip, r_hip)
                
                orientation = self.combine_orientations(None, hip_orientation, ratio_orientation, ratio_sign)
                orientations.append(orientation)
            return orientations
        else:
            return np.empty((0,1),dtype=int)

    def get_dets(self, boxes, positions3D, orientations):
        """
        This method concatenates bounding boxes and 3D positions to be used by the SortRGBD tracking algorithm

        return: 
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,x,y,z,score],[x1,y1,x2,y2,x,y,z,score],...]
        """
        dets = []

        for box, pos, yaw in zip(boxes, positions3D, orientations):
            x1, y1, x2, y2, score = box
            x, y, z = pos
            yaw = yaw if yaw is not None else np.nan
            if x is not None:
                dets.append([x1,y1,x2,y2,x,y,z,yaw,score])

        return np.array(dets)

    def yolov8_inference(self, frame):

        results = self.ov_model.detect(frame)[0]
        return results
    
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
    
    
    def publish_tracking_results(self, tracks, yaws, velocities, reliabilities=[]):
        """
        Publish results expressed in the self.fixed_frame        
        """
        if self.map_camera_transform is None or self.robot_info is None:
            return
        
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
      
        for track, yaw, vel, rel in zip(tracks, yaws, velocities, reliabilities):

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
            tracked_person.reliability = rel[0]
            msg.people.append(tracked_person)

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

        self.tracked_people_publisher.publish(msg)
        self.marker_pub.publish(marray)

    def display(self, frame, keypoints):
        tracked_keypoints = None
        tracked_points = None
        try:
            if keypoints is not None and len(self.tracks) > 0:
                inds = self.tracks[:, 8].astype('int')
                # self.get_logger().info(f"IDs: {self.tracks[:, 7]}")
                #tracked_keypoints = [keypoints[i] for i in inds]
            img = plot_tracked_BB_pose3(frame, self.tracks, keypoints=tracked_keypoints, points=tracked_points)
            image_message = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            try:
                self.image_pub.publish(image_message)
            except CvBridgeError as e:
                print(e)
        except Exception as e:
            self.get_logger().info(f"Unable to publish result image due to error: {e}")
            pass

    def main_callback(self):
        if self.data is None:
            return

        self.color_image, self.frame_timestamp, self.aligned_depth_image, self.fx, self.fy, self.cx, self.cy = self.data
        self.main_timestamp = self.frame_timestamp

        if self.main_timestamp == self.prev_main_timestamp:
            return
            
        self.get_camera_transform()
        self.get_robot_transform()
        
        results = self.yolov8_inference(self.color_image)
        boxes = results['box']
        keypoints = results.get('kpt', None)
        positions3D = self.get_3Dpositions(boxes[:,0:4], keypoints)
        orientations = self.get_orientations(keypoints)
        self.dets = self.get_dets(boxes[:,0:5], positions3D, orientations)
        if self.visualize and self.tracks is not None:
            self.display(self.color_image, keypoints)

        self.prev_main_timestamp = self.main_timestamp

    def publisher_callback(self):
        if self.data is None:
            return
        
        if self.dets is not None:
            if len(self.dets) == 0:
                self.dets = np.empty((0,9))

            # self.get_logger().info(f"dets: {dets[:, 6]}")

            dt = 1./23
            dt = self.get_clock().now() - self.pub_time
            dt = dt.nanoseconds/10**9
            # self.get_logger().info(f"dt: {dt}")
            self.tracks = self.tracker.update(self.dets, self.color_image, dt=dt)
            velocities = self.tracker.get_3Dvelocities()
            reliabilities = self.tracker.get_reliabilities()
            yaws = self.tracker.get_yaws()
            # self.get_logger().info(f"Tracks with dets: {dt, self.tracks}")
            self.dets = None

        else:
            dt = self.get_clock().now() - self.pub_time
            dt = dt.nanoseconds/10**9
            self.tracks = self.tracker.estimate(dt)
            velocities = self.tracker.get_3Dvelocities()
            reliabilities = self.tracker.get_reliabilities()
            yaws = self.tracker.get_yaws()
            # self.get_logger().info(f"Tracks with No dets: {dt, self.tracks}")

        if self.tracks is not None:
            self.publish_tracking_results(self.tracks, yaws, velocities, reliabilities)
            self.pub_time = self.get_clock().now()


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