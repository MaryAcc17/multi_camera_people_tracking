from rclpy.node import Node
from gazebo_test.utils.basic_navigator import BasicNavigator, TaskResult
import rclpy
from geometry_msgs.msg import PoseStamped, Pose
from typing import Optional, Callable, Awaitable
from rclpy.logging import LoggingSeverity
import asyncio
from contextlib import suppress


class NavigationHandler:
    """
    A class to handle the navigation stack in a ROS2 simulation environment.
    It provides methods to reset the navigation stack and manage its lifecycle.
    """

    def __init__(self, node: Node, navigation_result_callback=None) -> None:
        self.node = node
        self._loop = None
        self.navigator = BasicNavigator(
            node=self.node,
        )
        self.logger = rclpy.logging.get_logger("navigation_handler")
        # self.logger.set_level(rclpy.logging.LoggingSeverity.DEBUG)
        self.navigation_task = None
        self._default_navigation_result_callback = navigation_result_callback
        self._lifecycle_lock = asyncio.Lock()

    async def initialize_navigation(self) -> None:
        """
        Initializes the navigation stack and verifies its status.
        """
        # add get status of the nodes
        loop = asyncio.get_running_loop()
        self._loop = loop
        self.navigator.setLoop(loop)
        node_state = await self.navigator.checkNodeState("bt_navigator")
        self.logger.debug(f"Node state: {node_state}")
        if node_state != "active":
            self.logger.debug("Nav2 is not active, starting it ...")
            await self.navigator.lifecycleStartup()
            self.logger.debug("Waiting for Nav2 to be active ...")
        await self.navigator.waitUntilNav2Active()
        self.logger.debug("NavigationHandler initialized")

    def pause_navigation(self) -> None:
        """
        Pauses the navigation stack by sending a request to the lifecycle manager.

        This method creates a client for the ManageLifecycleNodes service and sends
        a request to pause the navigation stack. It waits for the response and logs
        the result.
        """
        pass

    async def reset_navigation(self) -> None:
        """
        Resets the navigation stack by sending a request to the lifecycle manager.

        This method creates a client for the ManageLifecycleNodes service and sends
        a request to reset the navigation stack. It waits for the response and logs
        the result.
        """
        self.logger.debug("Resetting navigation stack ...")
        async with self._lifecycle_lock:
            await self.cancel_navigation()
            self.navigator.clearEvents()

            reset_ok = await self._call_lifecycle_step(
                "reset Nav2", self.navigator.lifecycleReset
            )
            if not reset_ok:
                self.logger.warning(
                    "Nav2 reset failed; attempting shutdown/startup cycle instead"
                )
                await self._call_lifecycle_step(
                    "shutdown Nav2", self.navigator.lifecycleShutdown
                )
            await self._call_lifecycle_step(
                "startup Nav2", self.navigator.lifecycleStartup
            )
            await self.navigator.waitUntilNav2Active()
            await self._clear_costmaps_safe()

    def start_navigation_task(
        self,
        goal_pose: PoseStamped,
        navigation_result_callback: Optional[callable] = None,
    ) -> None:
        """
        Starts the navigation task by sending a goal to the navigator.
        This method creates a PoseStamped message with the goal pose and sends it to the navigator.
        It also creates a task to handle the navigation process.

        Parameters:
        - goal_pose (PoseStamped): The goal pose to navigate to.
        - navigation_result_callback (callable): Optional callback function to be called on success.
        """

        # launch the navigation task
        self.navigation_task = asyncio.create_task(self.navigator.goToPose(goal_pose))
        self.logger.debug("Navigation task started")

        # add a callback whenever one of the events is triggered
        self.wait_for_events_task = asyncio.create_task(
            self.wait_for_events(navigation_result_callback)
        )

    async def wait_for_events(self, callback: Optional[callable] = None) -> None:
        """
        Waits for navigation events to be triggered.
        This method subscribes to the navigation events and waits for them to be triggered.
        """

        # Wrap all event.wait() calls into tasks

        # Wait until any one of them is triggered
        done, pending = await asyncio.wait(
            [asyncio.ensure_future(event.wait()) for event in self.navigator.getEvents()],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # # Cancel the remaining tasks
        for task in pending:
            task.cancel()
        self.logger.debug("Event triggered")
        # Call the callback function with the triggered event
        success = self.navigator.go_to_pose_status == TaskResult.SUCCEEDED
        if callback:
            callback(success)
        else:
            self._default_navigation_result_callback(success)

    async def shutdown_navigation(self) -> None:
        """
        Shuts down the navigation stack by sending a request to the lifecycle manager.

        This method creates a client for the ManageLifecycleNodes service and sends
        a request to shut down the navigation stack. It waits for the response and logs
        the result.
        """
        self.logger.debug("Shutting down navigation stack ...")
        async with self._lifecycle_lock:
            await self.cancel_navigation()
            self.navigator.clearEvents()
            await self._call_lifecycle_step(
                "shutdown Nav2", self.navigator.lifecycleShutdown
            )

    async def cancel_navigation(self) -> None:
        """
        Cancels the navigation task if it is running.
        This method checks if the navigation task is running and cancels it.
        """
        if self.navigation_task:
            self.logger.debug("Cancelling navigation task ...")
            try:
                await self.navigator.cancelGoToPose()
            except Exception as exc:  # noqa: BLE001
                self.logger.debug(f"cancelGoToPose failed or already complete: {exc}")
            self.navigation_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.navigation_task
            self.navigation_task = None

        if task := getattr(self, "wait_for_events_task", None):
            self.logger.debug("Cancelling wait_for_events task ...")
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            self.wait_for_events_task = None

    def resume_navigation(self) -> None:
        """
        Resumes the navigation stack by sending a request to the lifecycle manager.

        This method creates a client for the ManageLifecycleNodes service and sends
        a request to resume the navigation stack. It waits for the response and logs
        the result.
        """
        pass

    async def _call_lifecycle_step(
        self, description: str, action: Callable[[], Awaitable[None]]
    ) -> bool:
        """Call a lifecycle action and log failures without stopping the loop."""
        try:
            await action()
            return True
        except Exception as exc:  # noqa: BLE001
            self.logger.error(f"Failed to {description}: {exc}")
            return False

    async def _clear_costmaps_safe(self, retries: int = 3) -> None:
        for attempt in range(1, retries + 1):
            try:
                await self.navigator.clearAllCostmaps()
                return
            except Exception as exc:  # noqa: BLE001
                if attempt < retries:
                    self.logger.warning(
                        f"Costmap clear attempt {attempt}/{retries} failed: {exc}; retrying..."
                    )
                    await asyncio.sleep(0.5)
                else:
                    self.logger.warning(
                        f"Unable to clear costmaps after {retries} attempts; continuing anyway: {exc}"
                    )
