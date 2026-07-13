from typing import Optional
import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message
from std_msgs.msg import String
from gazebo_collision_msgs.msg import Collision
from geometry_msgs.msg import Pose, PoseStamped, Twist
from people_msgs.msg import People
from nav_msgs.msg import Odometry, Path as NavPath
from sensor_msgs.msg import LaserScan
from hunav_msgs.msg import Agents
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import OccupancyGrid

import shutil
from pathlib import Path

import rosbag2_py

topic_dict = {
    "/goal_pose": PoseStamped,
    "/experiment_result": String,
    "/cmd_vel": Twist,
    "/front/scan": LaserScan,
    "/human_states": Agents,
    "/jackal/ground_truth": Odometry,
    "/jackal_velocity_controller/cmd_vel_unstamped": Twist,
    "/jackal_velocity_controller/odom": Odometry,
    "/jackal/collision": Collision,
    "/people": People,
    "/plan": NavPath,
    "/tf": TFMessage,
    "/tf_static": TFMessage,
}

maps_topic_dict = {
    "/map": OccupancyGrid,
    "/global_costmap/costmap": OccupancyGrid,
    "/local_costmap/costmap": OccupancyGrid,
}


class BagRecorder:
    def __init__(
        self,
        node: Node,
        algorithm: str = "test",
        base_path: Optional[Path] = None,
        record_maps: bool = False,
    ):
        """
        Initialize the BagRecorder class.
        Args:
            node (Node): The ROS2 node to which the BagRecorder is attached.
            algorithm (str): The name of the algorithm being tested.
            base_path (Optional[Path]): The base path for storing bag files.
        """
        if base_path:
            self.base_path = Path(base_path).expanduser()
        else:
            self.base_path = Path("results/gazebo_test") / algorithm
            self.base_path = self.base_path.expanduser()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.node = node
        self.writer = rosbag2_py.SequentialWriter()
        self.recording = False
        self.logger = rclpy.logging.get_logger("bag_recorder")  # type: ignore
        self.get_clock = node.get_clock
        self.algorithm = algorithm
        self.topics_metadata = []
        # build a local copy of topics to record (avoid mutating the module-level dict)
        active_topics = dict(topic_dict)
        if record_maps:
            active_topics.update(maps_topic_dict)
        self._active_topics = active_topics
        for topic_name, msg_type in active_topics.items():
            # Create a TopicMetadata object for each topic
            self.topics_metadata.append(
                rosbag2_py.TopicMetadata(
                    name=topic_name,
                    type=f"{msg_type.__module__.split('.')[0]}/msg/{msg_type.__name__}",
                    serialization_format="cdr",
                )
            )
            if topic_name == "/experiment_result" or topic_name == "/goal_pose":
                continue
            self.node.create_subscription(
                msg_type=msg_type,
                topic=topic_name,
                callback=lambda msg, topic_name=topic_name: self.topic_callback(
                    msg, topic_name
                ),
                qos_profile=10,
            )
        # set the logger level of rosbag2_storage to warn
        rosbag2_logger = rclpy.logging.get_logger("rosbag2_storage")  # type: ignore
        rosbag2_logger.set_level(rclpy.logging.LoggingSeverity.WARN)  # type: ignore
        self.logger.info(
            f"BagRecorder initialized with base path: {self.base_path}, algorithm: {self.algorithm}"
        )

    def start_recording(self, experiment_name: str, run_id: str):
        if self.recording:
            self.logger.warn("Recording is already in progress.")
            return
        self.logger.debug("Starting recording...")
        run_identifier = str(run_id)
        run_suffix = (
            f"experiment_{run_identifier}"
            if run_identifier.isdigit()
            else f"experiment_{run_identifier}"
        )
        algorithm_root = self.base_path / experiment_name / self.algorithm
        algorithm_root.mkdir(parents=True, exist_ok=True)
        self.bag_path = algorithm_root / run_suffix
        if self.bag_path.exists():
            self.logger.warning(
                f"Bag path {self.bag_path} already exists; replacing previous recording."
            )
            shutil.rmtree(self.bag_path)
        self.logger.debug(f"Bag path: {self.bag_path}")

        # Create a fresh writer for each recording session
        self.writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(
            uri=str(self.bag_path), storage_id="sqlite3"
        )
        converter_options = rosbag2_py.ConverterOptions("", "")
        self.writer.open(storage_options, converter_options)

        # Create topics
        for topic_info in self.topics_metadata:
            self.writer.create_topic(topic_info)

        self.recording = True
        self.logger.debug(f"Started recording to {self.bag_path}")

    def stop_recording(self):
        if not self.recording:
            self.logger.warn("No recording is in progress.")
            return

        self.writer.close()
        self.recording = False
        self.logger.debug(f"Stopped recording to {self.bag_path}")

    def topic_callback(self, msg, topic_name):
        """
        Callback function for the topic subscription.
        This function is called whenever a message is received on the subscribed topic.
        It serializes the message and writes it to the bag file.
        Args:
            msg: The message received on the topic.
            topic_name: The name of the topic.
        """
        # Check if recording is in progress
        # and if the topic is in the topic_dict
        if topic_name not in self._active_topics:
            self.logger.warn(f"Topic {topic_name} is not in the topic_dict.")
            return
        if not self.recording:
            return
        # Serialize the message
        serialized_msg = serialize_message(msg)
        # Write the message to the bag file
        self.writer.write(
            topic_name, serialized_msg, self.get_clock().now().nanoseconds
        )

    def set_experiment_result(self, result: str):
        """
        Set the result of the experiment.
        Args:
            result: The result of the experiment.
        """
        self.writer.write(
            "/experiment_result",
            serialize_message(String(data=result)),
            self.get_clock().now().nanoseconds,
        )
        self.logger.debug(f"Experiment result: {result}")

    def set_result_and_stop_recording(self, result: str):
        """
        Set the result of the experiment and stop recording.
        Args:
            result: The result of the experiment.
        """
        self.set_experiment_result(result)
        self.stop_recording()

    def set_goal(self, goal: PoseStamped):
        """
        Set the goal of the experiment.
        Args:
            goal: The goal of the experiment.
        """
        self.writer.write(
            "/goal_pose",
            serialize_message(goal),
            self.get_clock().now().nanoseconds,
        )
        self.logger.debug(f"Goal set to: {goal}")

    def start_recording_and_set_goal(
        self, experiment_name: str, run_id: str, goal: PoseStamped
    ):
        """
        Start recording and set the goal of the experiment.
        Args:
            goal: The goal of the experiment.
        """
        self.start_recording(experiment_name, run_id)
        self.set_goal(goal)
        self.logger.debug(f"Started recording and set goal to: {goal}")
