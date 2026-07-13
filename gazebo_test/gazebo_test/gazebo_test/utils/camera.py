"""Utility to drive a Gazebo camera and save bird-eye snapshots."""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import cv2
from cv_bridge import CvBridge, CvBridgeError
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from geometry_msgs.msg import Pose
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from tf_transformations import quaternion_from_euler
from gazebo_test.utils.gazebo_env_handler import GazeboEnvironmentHandler


class CameraCapturer:
    """Handles positioning and saving outputs from a Gazebo camera."""

    def __init__(
        self,
        node: Node,
        *,
        base_output_dir: Path,
        entity_name: str,
        image_topic: str,
        image_encoding: str,
        camera_position: Sequence[float],
        camera_orientation_rpy: Sequence[float],
        camera_model_path: Path | str,
        reference_frame: str = "",
        image_timeout_sec: float = 5.0,
    ) -> None:
        self._node = node
        self._entity_name = entity_name
        self._reference_frame = reference_frame
        self._base_output_dir = Path(base_output_dir)
        self._base_output_dir.mkdir(parents=True, exist_ok=True)
        self._bridge = CvBridge()
        self._image_timeout_sec = max(image_timeout_sec, 0.1)

        self._gazebo_handler = GazeboEnvironmentHandler(node)
        self._image_sub = node.create_subscription(
            Image,
            image_topic,
            self._image_callback,
            10,
        )

        self._pose = Pose()
        self._pose.position.x = (
            float(camera_position[0]) if len(camera_position) > 0 else 0.0
        )
        self._pose.position.y = (
            float(camera_position[1]) if len(camera_position) > 1 else 0.0
        )
        self._pose.position.z = (
            float(camera_position[2]) if len(camera_position) > 2 else 0.0
        )
        roll = (
            float(camera_orientation_rpy[0]) if len(camera_orientation_rpy) > 0 else 0.0
        )
        pitch = (
            float(camera_orientation_rpy[1]) if len(camera_orientation_rpy) > 1 else 0.0
        )
        yaw = (
            float(camera_orientation_rpy[2]) if len(camera_orientation_rpy) > 2 else 0.0
        )
        quat = quaternion_from_euler(roll, pitch, yaw)
        self._pose.orientation.x = quat[0]
        self._pose.orientation.y = quat[1]
        self._pose.orientation.z = quat[2]
        self._pose.orientation.w = quat[3]

        camera_model_path = Path(camera_model_path).expanduser()
        if not camera_model_path.is_file():
            raise FileNotFoundError(f"Camera model file not found: {camera_model_path}")
        try:
            self._camera_model_xml = camera_model_path.read_text()
        except OSError as exc:
            raise RuntimeError(
                f"Failed to read camera model file {camera_model_path}: {exc}"
            ) from exc

        self._image_encoding = image_encoding or "bgr8"
        self._gazebo_services_ready = False
        self._image_event = threading.Event()
        self._latest_image = None
        self._camera_ready = False

    def capture_snapshot(
        self,
        *,
        experiment_tag: str,
        run_id: int,
        stage: str,
        target_dir: Optional[Path] = None,
        episode: Optional[str] = None,
    ) -> Optional[Path]:
        """Persist the latest frame to disk.

        Assumes :meth:`prepare_for_capture` was awaited before this call.
        """

        if not self._camera_ready:
            self._node.get_logger().warning(
                "Camera not prepared; call prepare_for_capture() first"
            )
            return None

        self._image_event.clear()
        self._latest_image = None
        self._node.get_logger().info(
            f"Waiting up to {self._image_timeout_sec:.1f} seconds for image)"
        )

        # Wait for the image callback to receive a new frame
        # wait for the image event to be set by the callback, with a timeout to avoid hanging indefinitely
        if not self._image_event.wait(timeout=self._image_timeout_sec):
            self._node.get_logger().warning(
                f"Timed out waiting for camera image after {self._image_timeout_sec:.1f} seconds"
            )
            return None

        image = self._latest_image
        if image is None:
            self._node.get_logger().warning("Camera image callback returned no data")
            return None

        self._node.get_logger().info("Camera image received, saving snapshot...")

        output_dir = target_dir or self._base_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        sanitized_stage = stage.replace("/", "-")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / f"{experiment_tag}" / "images"
        output_path.mkdir(parents=True, exist_ok=True)
        if episode:
            episode = episode.replace("/", "-")
            output_path = output_path / episode
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = (
                output_path / f"run{run_id:03d}_{sanitized_stage}_{timestamp}.jpg"
            )
        else:
            output_path = output_path / f"{experiment_tag}_{sanitized_stage}_raw.jpg"

        if cv2.imwrite(str(output_path), image):
            self._node.get_logger().info(f"Saved camera image to {output_path}")
            return output_path

        self._node.get_logger().warning(
            f"Failed to write camera image to {output_path}"
        )
        return None

    async def prepare_for_capture(self) -> bool:
        """Ensure the camera is spawned and ready before capturing."""

        if self._camera_ready:
            return True

        self._node.get_logger().info("Waiting for camera to publish first image...")
        ready = await self._spawn_camera()
        if ready:
            self._node.get_logger().info(
                "CameraCapturer is ready to capture snapshots (skipping spawn in Gazebo)"
            )
            self._camera_ready = True
        else:
            self._node.get_logger().error(
                "Failed to prepare camera via Gazebo services"
            )
        return ready

    async def _spawn_camera(self) -> bool:
        if not self._ensure_spawn_services_ready():
            return False

        request = SpawnEntity.Request()
        request.name = self._entity_name
        request.xml = self._camera_model_xml
        request.robot_namespace = ""
        request.initial_pose = self._pose
        request.reference_frame = self._reference_frame

        # pause gazebo physics to help ensure the camera spawns in the correct pose
        pause_success = await self._gazebo_handler.pause_gazebo()

        if not pause_success:
            self._node.get_logger().error("Failed to pause Gazebo environment")
            return False

        self._node.get_logger().info(
            f"Spawning camera entity '{self._entity_name}' in Gazebo at position {self._pose.position} and orientation {self._pose.orientation}"
            f" in reference frame '{self._reference_frame or 'world'}'"
        )
        await self._gazebo_handler.spawn_entity(
            entity_name=self._entity_name,
            entity_xml=self._camera_model_xml,
            initial_pose=self._pose,
            reference_frame=self._reference_frame,
        )
        resume_success = await self._gazebo_handler.resume_gazebo()
        if not resume_success:
            self._node.get_logger().error("Failed to resume Gazebo environment")
            return False
        return True

    def _ensure_spawn_services_ready(self) -> bool:
        if self._gazebo_services_ready:
            return True

        clients = [
            self._gazebo_handler.spawn_entity_client,
            self._gazebo_handler.delete_entity_client,
        ]
        for client in clients:
            if client is None:
                continue
            self._node.get_logger().debug(
                f"Waiting for {client.srv_name} service to manage camera"
            )
            while not client.wait_for_service(timeout_sec=1.0):
                self._node.get_logger().info(
                    f"Waiting for {client.srv_name} service..."
                )
                if not self._node.context.ok():
                    return False
        self._gazebo_services_ready = True
        return True

    def _image_callback(self, msg: Image) -> None:
        try:
            cv_image = self._bridge.imgmsg_to_cv2(
                msg, desired_encoding=self._image_encoding
            )
        except CvBridgeError as exc:  # pragma: no cover - cv bridge errors
            self._node.get_logger().warning(f"Failed to convert camera image: {exc}")
            return
        self._latest_image = cv_image
        self._image_event.set()
