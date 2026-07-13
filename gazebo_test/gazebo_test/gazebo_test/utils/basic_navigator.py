from enum import Enum
import time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration

from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import ComputePathToPose
from nav2_msgs.action import (
    FollowPath,
    NavigateToPose,
)
from nav2_msgs.action import SmoothPath
from nav2_msgs.srv import ClearEntireCostmap, GetCostmap, LoadMap, ManageLifecycleNodes

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration as rclpyDuration
from rclpy.node import Node
import asyncio
from rclpy.logging import get_logger
from typing import Optional


class TaskResult(Enum):
    UNKNOWN = 0
    SUCCEEDED = 1
    CANCELED = 2
    FAILED = 3


class BasicNavigator:

    def __init__(self, node: Node):

        self.node = node
        self._loop = None

        self.follow_path_goal_handle = None
        self.go_to_pose_goal_handle = None
        # self.compute_path_to_pose_goal_handle = None
        # self.smooth_path_goal_handle = None
        self.go_to_pose_result = None
        self.follow_path_result = None
        # self.compute_path_to_pose_result = None
        # self.smooth_path_result = None
        self.feedback = None
        self.go_to_pose_status = GoalStatus.STATUS_UNKNOWN
        self.follow_path_status = GoalStatus.STATUS_UNKNOWN

        self.go_to_pose_event = asyncio.Event()
        self.follow_path_event = asyncio.Event()

        # for now allow only the nav_to_pose action
        self.nav_to_pose_client = ActionClient(
            self.node, NavigateToPose, "navigate_to_pose"
        )
        self.follow_path_client = ActionClient(self.node, FollowPath, "follow_path")
        self.compute_path_to_pose_client = ActionClient(
            self.node, ComputePathToPose, "compute_path_to_pose"
        )
        self.smoother_client = ActionClient(self.node, SmoothPath, "smooth_path")
        # todo: explore map changing

        self.clear_costmap_global_srv = node.create_client(
            ClearEntireCostmap, "global_costmap/clear_entirely_global_costmap"
        )
        self.clear_costmap_local_srv = node.create_client(
            ClearEntireCostmap, "local_costmap/clear_entirely_local_costmap"
        )
        self.logger = get_logger("basic_navigator")
        # self.logger.set_level(rclpy.logging.LoggingSeverity.DEBUG)

    def destroyNode(self):
        """Destroy the node and all action clients."""
        self.destroy_node()

    def destroy_node(self):
        """Destroy the node and all action clients."""
        self.nav_to_pose_client.destroy()
        self.follow_path_client.destroy()
        self.compute_path_to_pose_client.destroy()
        self.smoother_client.destroy()
        super().destroy_node()

    def clearEvents(self):
        """Clear the events."""
        self.go_to_pose_event.clear()
        self.follow_path_event.clear()

    def getEvents(self):
        """Get the events."""
        return self.go_to_pose_event, self.follow_path_event

    async def goToPose(self, pose: PoseStamped, behavior_tree: str = "") -> bool:
        """Send a `NavToPose` action request.
        Args:
            pose (PoseStamped): The target pose to navigate to.
            behavior_tree (str): The behavior tree to use for navigation.
        Returns:
            bool: True if the goal was accepted, False otherwise."""

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        goal_msg.behavior_tree = behavior_tree

        self.logger.debug(
            "Navigating to goal: "
            + str(pose.pose.position.x)
            + " "
            + str(pose.pose.position.y)
            + "..."
        )
        send_goal_future = self.nav_to_pose_client.send_goal_async(
            goal_msg, self._feedbackCallback
        )
        self.go_to_pose_goal_handle = await send_goal_future
        if not self.go_to_pose_goal_handle.accepted:
            self.logger.error(
                "Goal to "
                + str(pose.pose.position.x)
                + " "
                + str(pose.pose.position.y)
                + " was rejected!"
            )
            return False

        self.go_to_pose_result = self.go_to_pose_goal_handle.get_result_async()
        self.go_to_pose_result.add_done_callback(self._go_to_pose_result_callback)
        return True

    async def followPath(
        self, path: Path, controller_id: str = "", goal_checker_id: str = ""
    ) -> bool:
        """Send a `FollowPath` action request.
        Args:
            path (Path): The path to follow.
            controller_id (str): The controller ID to use.
            goal_checker_id (str): The goal checker ID to use.
        Returns:
            bool: True if the goal was accepted, False otherwise."""
        goal_msg = FollowPath.Goal()
        goal_msg.path = path
        goal_msg.controller_id = controller_id
        goal_msg.goal_checker_id = goal_checker_id

        self.logger.info("Executing path...")
        send_goal_future = self.follow_path_client.send_goal_async(
            goal_msg, self._feedbackCallback
        )
        self.follow_path_goal_handle = await send_goal_future
        if not self.follow_path_goal_handle.accepted:
            self.logger.error("Follow path was rejected!")
            return False
        self.follow_path_result = self.follow_path_goal_handle.get_result_async()
        self.follow_path_result.add_done_callback(self._follow_path_result_callback)
        return True

    async def cancelGoToPose(self):
        """Cancel pending task of the `NavToPose` action.
        This will cancel the current task and set the status to CANCELED.
        Returns:
            None
        """
        self.logger.debug("Canceling current task.")
        if self.go_to_pose_result:
            future = await self.go_to_pose_goal_handle.cancel_goal_async()

    def getFeedback(self):
        """Get the pending action feedback message."""
        return self.feedback

    def getResult(self, goal_result: GoalStatus) -> TaskResult:
        """Get the pending action result message
        Returns:
            TaskResult: The result of the task.
        """
        match goal_result.status:
            case GoalStatus.STATUS_SUCCEEDED:
                return TaskResult.SUCCEEDED
            case GoalStatus.STATUS_ABORTED:
                return TaskResult.FAILED
            case GoalStatus.STATUS_CANCELED:
                return TaskResult.CANCELED
            case _:
                return TaskResult.UNKNOWN

    async def waitUntilNav2Active(self, navigator="bt_navigator"):
        """Block until the full navigation system is up and running."""
        await self._waitForNodeToActivate(navigator)
        await self._waitForServer()
        self.logger.info("Nav2 is ready for use!", once=True)
        return

    def setLoop(self, loop):
        """Set the loop for the navigator."""
        self._loop = loop
        self.logger.debug("Set loop for navigator")
        return

    async def _getPathImpl(self, start, goal, planner_id="", use_start=False):
        """
        Send a `ComputePathToPose` action request.

        Internal implementation to get the full result, not just the path.
        """
        goal_msg = ComputePathToPose.Goal()
        goal_msg.start = start
        goal_msg.goal = goal
        goal_msg.planner_id = planner_id
        goal_msg.use_start = use_start
        self.logger.info("Getting path...")
        send_goal_future = self.compute_path_to_pose_client.send_goal_async(goal_msg)
        goal_handle = await send_goal_future

        if not goal_handle.accepted:
            self.logger.error("Get path was rejected!")
            return None

        result = await goal_handle.get_result_async()
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            self.logger.error(f"Getting path failed with status code: {result.status}")
            return None

        return result.result

    def getPath(self, start, goal, planner_id="", use_start=False):
        """Send a `ComputePathToPose` action request."""
        rtn = self._getPathImpl(start, goal, planner_id, use_start)
        if not rtn:
            return None
        else:
            return rtn.path

    async def _smoothPathImpl(
        self, path, smoother_id="", max_duration=2.0, check_for_collision=False
    ):
        """
        Send a `SmoothPath` action request.

        Internal implementation to get the full result, not just the path.
        """

        goal_msg = SmoothPath.Goal()
        goal_msg.path = path
        goal_msg.max_smoothing_duration = rclpyDuration(seconds=max_duration).to_msg()
        goal_msg.smoother_id = smoother_id
        goal_msg.check_for_collisions = check_for_collision
        self.logger.info("Smoothing path...")
        goal_handle = await self.smoother_client.send_goal_async(goal_msg)

        if not goal_handle.accepted:
            self.logger.error("Smooth path was rejected!")
            return None

        result = await goal_handle.get_result_async()

        if result.status != GoalStatus.STATUS_SUCCEEDED:
            self.logger.error(f"Getting path failed with status code: {result.status}")
            return None

        return result.result

    def smoothPath(
        self, path, smoother_id="", max_duration=2.0, check_for_collision=False
    ):
        """Send a `SmoothPath` action request."""
        rtn = self._smoothPathImpl(path, smoother_id, max_duration, check_for_collision)
        if not rtn:
            return None
        else:
            return rtn.path

    async def clearAllCostmaps(self):
        """Clear all costmaps."""

        clear_local_future = asyncio.create_task(self.clearLocalCostmap())
        clear_global_future = asyncio.create_task(self.clearGlobalCostmap())
        await asyncio.gather(clear_local_future, clear_global_future)
        self.logger.info("Cleared all costmaps")
        return

    async def clearLocalCostmap(self):
        """Clear local costmap."""
        req = ClearEntireCostmap.Request()
        result = await self.clear_costmap_local_srv.call_async(req)
        if not result:
            self.logger.error("Failed to clear local costmap")
            return
        self.logger.debug("Cleared local costmap")
        return

    async def clearGlobalCostmap(self):
        """Clear global costmap."""
        req = ClearEntireCostmap.Request()
        result = await self.clear_costmap_global_srv.call_async(req)
        if not result:
            self.logger.error("Failed to clear global costmap")
            return
        self.logger.debug("Cleared global costmap")
        return

    async def lifecycleStartup(self):
        """Startup nav2 lifecycle system."""
        self.logger.debug("Starting up lifecycle nodes based on lifecycle_manager.")
        for srv_name, srv_type in self.node.get_service_names_and_types():
            if srv_type[0] == "nav2_msgs/srv/ManageLifecycleNodes":
                self.logger.debug(f"Starting up {srv_name}")
                mgr_client = self.node.create_client(ManageLifecycleNodes, srv_name)
                while not mgr_client.wait_for_service(timeout_sec=1.0):
                    self.logger.debug(f"{srv_name} service not available, waiting...")
                req = ManageLifecycleNodes.Request()
                req.command = ManageLifecycleNodes.Request().STARTUP
                result = await mgr_client.call_async(req)
                if not result:
                    self.logger.error(f"Failed to start up {srv_name}")  # handle error
                    return
                self.logger.debug(f"Started up {srv_name}")
        self.logger.debug("Nav2 is ready for use!")
        return

    async def lifecycleReset(self):
        """Reset nav2 lifecycle system."""
        self.logger.debug("Resetting lifecycle nodes based on lifecycle_manager.")
        for srv_name, srv_type in self.node.get_service_names_and_types():
            if srv_type[0] == "nav2_msgs/srv/ManageLifecycleNodes":
                self.logger.debug(f"Resetting {srv_name}")
                mgr_client = self.node.create_client(ManageLifecycleNodes, srv_name)
                while not mgr_client.wait_for_service(timeout_sec=1.0):
                    self.logger.debug(f"{srv_name} service not available, waiting...")
                req = ManageLifecycleNodes.Request()
                req.command = ManageLifecycleNodes.Request().RESET
                result = await mgr_client.call_async(req)
                if not result:
                    self.logger.error(f"Failed to reset {srv_name}")
                    return
                self.logger.debug(f"Reset {srv_name}")
        return

    async def lifecycleShutdown(self):
        """Shutdown nav2 lifecycle system."""
        self.logger.debug("Shutting down lifecycle nodes based on lifecycle_manager.")
        for srv_name, srv_type in self.node.get_service_names_and_types():
            if srv_type[0] == "nav2_msgs/srv/ManageLifecycleNodes":
                self.logger.debug(f"Shutting down {srv_name}")
                mgr_client = self.node.create_client(ManageLifecycleNodes, srv_name)
                while not mgr_client.wait_for_service(timeout_sec=1.0):
                    self.logger.debug(f"{srv_name} service not available, waiting...")
                req = ManageLifecycleNodes.Request()
                req.command = ManageLifecycleNodes.Request().SHUTDOWN
                result = await mgr_client.call_async(req)
                if not result:
                    self.logger.error(f"Failed to shut down {srv_name}")  # handle error
                    return
                self.logger.debug(f"Shut down {srv_name}")
        self.logger.info("Nav2 is shut down!")
        return

    async def checkNodeState(self, node_name):
        """Check the state of a node.
        Args:
            node_name (str): The name of the node to check.
        """
        self.logger.debug(f"Checking state of {node_name}...")
        node_service = f"{node_name}/get_state"
        state_client = self.node.create_client(GetState, node_service)
        while not state_client.wait_for_service(timeout_sec=1.0):
            self.logger.debug(f"{node_service} service not available, waiting...")

        req = GetState.Request()
        future = await state_client.call_async(req)
        return future.current_state.label

    async def _waitForNodeToActivate(self, node_name):
        """Wait for a node to become active.
        Args:
            node_name (str): The name of the node to wait for.
        """
        self.logger.debug(f"Waiting for {node_name} to become active..")
        node_service = f"{node_name}/get_state"
        state_client = self.node.create_client(GetState, node_service)
        while not state_client.wait_for_service(timeout_sec=1.0):
            self.logger.debug(f"{node_service} service not available, waiting...")

        req = GetState.Request()
        state = "unknown"
        while state != "active":
            self.logger.debug(f"Waiting for {node_name} to become active...")
            future = await state_client.call_async(req)
            state = future.current_state.label
            self.logger.debug(f"Result of get_state: {state}")
        return

    def _feedbackCallback(self, msg):
        # self.logger.debug("Received action feedback message")
        self.feedback = msg.feedback
        return

    async def _waitForServer(self):
        for client in [
            self.nav_to_pose_client,
            self.follow_path_client,
            self.compute_path_to_pose_client,
            self.smoother_client,
        ]:
            while not client.wait_for_server(timeout_sec=1.0):
                self.logger.debug(
                    f"{client.action_name} action server not available, waiting..."
                )
        self.logger.debug("All action servers are available.")
        for srv in [
            self.clear_costmap_global_srv,
            self.clear_costmap_local_srv,
        ]:
            while not srv.wait_for_service(timeout_sec=1.0):
                self.logger.debug(f"{srv.srv_name} service not available, waiting...")
        self.logger.debug("All services are available.")
        return

    def _go_to_pose_result_callback(self, future):
        self.logger.debug("Received action result message")
        self.go_to_pose_result = future.result()
        self.go_to_pose_status = self.getResult(self.go_to_pose_result)
        self.logger.debug(f"Go to pose status: {self.go_to_pose_status}")
        self._loop.call_soon_threadsafe(self.go_to_pose_event.set)

    def _follow_path_result_callback(self, future):
        self.logger.debug("Received action result message")
        self.follow_path_result = future.result()
        self.follow_path_status = self.getResult(self.follow_path_result)
        self.logger.debug(f"Follow path status: {self.follow_path_status}")
        self._loop.call_soon_threadsafe(self.follow_path_event.set)
