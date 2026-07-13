# for eache experiment, this class will trigger the evaluation of the experiment by using the hunav_evaluator class.
import os
from typing import Optional
from rclpy.node import Node
import rclpy
from hunav_msgs.srv import StartEvaluation
from std_srvs.srv import Empty
from rclpy.logging import get_logger
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.srv import SetParameters
from pathlib import Path
import pandas as pd


class HunavEvaluatorHandler:
    def __init__(self, node: Node, algorithm: str = "test", base_path: Optional[Path] = None):
        """
        Initialize the HunavEvaluatorHandler class.
        Args:
            node (Node): The ROS2 node to which the handler is attached.
            algorithm (str): The name of the algorithm being tested.
            base_path (Path): The base path for storing evaluation results."""
        self.hunav_start_client = node.create_client(
            StartEvaluation, "hunav_start_recording"
        )
        self.hunav_stop_client = node.create_client(Empty, "hunav_stop_recording")
        self.set_result_file_path_client = node.create_client(
            SetParameters, "hunav_evaluator_node/set_parameters"
        )
        self.algorithm = algorithm
        self.base_path = base_path
        self.date = node.get_clock().now().to_msg().sec
        if not self.base_path:
            self.base_path = Path(f"bags/gazebo_test/exp_{self.date}/")
        self.logger = get_logger("hunav_evaluator_handler")

    async def wait_for_hunav_evaluator(self):
        """
        Wait for the hunav evaluator service to be available.
        This method will block until the service is available.
        """

        self.logger.debug("Waiting for hunav evaluator service to be available...")
        await self.wait_for_services()
        self.logger.debug("Hunav evaluator service is available.")

    async def wait_for_services(self):
        """
        Wait for the hunav start and stop services to be available.
        This method will block until both services are available.
        """
        services = [
            self.hunav_start_client,
            self.hunav_stop_client,
            self.set_result_file_path_client,
        ]
        for client in services:
            while not client.wait_for_service(timeout_sec=1.0):
                self.logger.debug(f"Waiting for {client.srv_name} service...")

    async def start_recording(
        self, goal: PoseStamped, experiment_tag: str, run_id: int
    ) -> bool:
        """
        Start the hunav evaluator recording.
        This method will call the hunav_start_recording service.

        Returns:
            bool: True if the service call was successful, False otherwise.
        """
        self.logger.debug("Starting hunav evaluator recording...")
        req = StartEvaluation.Request()
        req.robot_goal = goal
        req.experiment_tag = experiment_tag
        req.run_id = run_id
        self.metrics_path = Path(
            f"{self.base_path}/{experiment_tag}/{self.algorithm}/experiment_{run_id}/metrics.csv"
        )
        self.logger.debug(f"Metrics path: {self.metrics_path}")
        parameters_future = await self.set_result_file_path(str(self.metrics_path))
        if not parameters_future:
            self.logger.error("Failed to set result file path.")
            return False

        try:
            response = await self.hunav_start_client.call_async(req)
            if response.success: # type: ignore
                self.logger.debug("Hunav evaluator recording started successfully.")
                return True
            else:
                self.logger.error("Failed to start hunav evaluator recording.")
                return False
        except Exception as e:
            self.logger.error(f"Failed to pause Gazebo environment: {e}")
            return False

    async def stop_recording(self) -> bool:
        """
        Stop the hunav evaluator recording.
        This method will call the hunav_stop_recording service.

        Returns:
            bool: True if the service call was successful, False otherwise.
        """
        self.logger.debug("Stopping hunav evaluator recording...")
        try:
            future = await self.hunav_stop_client.call_async(Empty.Request())
            self.logger.debug("Hunav evaluator recording stopped successfully.")
        except Exception as e:
            self.logger.error(f"Failed to stop hunav evaluator recording: {e}")
            return False
        return True

    async def set_result_file_path(self, file_path: str) -> bool:
        """
        Set the path for the result file in the hunav evaluator.
        This method will call the set_parameters service to set the result file path.
        Args:
            file_path (str): The path to the result file.
        Returns:
            bool: True if the service call was successful, False otherwise.
        """
        self.logger.debug(f"Setting result file path to {file_path}...")
        req = SetParameters.Request()
        req.parameters = [
            rclpy.Parameter(
                "result_file", rclpy.Parameter.Type.STRING, file_path
            ).to_parameter_msg()
        ]
        try:
            response = await self.set_result_file_path_client.call_async(req)
            for result in response.results: # type: ignore
                if result.successful:
                    self.logger.debug("Result file path set successfully.")
                    continue
                self.logger.error("Failed to set result file path.")
                self.logger.error(f"Reason: {result.reason}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Failed to set result file path: {e}")
            return False

    def get_result_df(self):
        """
        Get the result DataFrame from the metrics CSV file.
        This method reads the metrics CSV file and returns it as a pandas DataFrame.
        Returns:
            pd.DataFrame: The DataFrame containing the evaluation metrics.
        """
        if not self.metrics_path.exists():
            self.logger.error(f"Metrics file {self.metrics_path} does not exist.")
            return pd.DataFrame()
        return pd.read_csv(self.metrics_path)


# Note: This class is designed to be used in conjunction with the hunav evaluator
# and is expected to be called from a ROS2 node. It provides methods to start and stop
# the hunav evaluator recording, as well as to set the result file path for the evaluation metrics.
