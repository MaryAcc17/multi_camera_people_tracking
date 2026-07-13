import asyncio
from rclpy.node import Node
from rclpy.qos import QoSProfile

from gazebo_msgs.srv import (
    SetEntityState,
    SpawnEntity,
    DeleteEntity,
    GetEntityState,
    GetModelList,
)
from gazebo_msgs.msg import EntityState
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose
from typing import Optional, List, Dict, Callable
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.logging import get_logger, LoggingSeverity


class GazeboEnvironmentHandler:
    """
    A class to handle the communication with the Gazebo environment using asyncio.
    """

    def __init__(self, node: Node):
        self.reset_world_client = node.create_client(
            Empty, "reset_simulation", callback_group=MutuallyExclusiveCallbackGroup()
        )
        self.pause_physics_client = node.create_client(
            Empty, "pause_physics", callback_group=MutuallyExclusiveCallbackGroup()
        )
        self.unpause_physics_client = node.create_client(
            Empty, "unpause_physics", callback_group=MutuallyExclusiveCallbackGroup()
        )
        self.set_entity_state_client = node.create_client(
            SetEntityState,
            "test/set_entity_state",
            callback_group=ReentrantCallbackGroup(),
        )
        self.spawn_entity_client = node.create_client(
            SpawnEntity,
            "/spawn_entity",
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.delete_entity_client = node.create_client(
            DeleteEntity,
            "/delete_entity",
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        self.get_entity_state_client = node.create_client(
            GetEntityState,
            "test/get_entity_state",
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        self.get_model_list_client = node.create_client(
            GetModelList,
            "/get_model_list",
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.logger = get_logger("gazebo_env_handler")
        # self.logger.set_level(LoggingSeverity.DEBUG)
        self.logger.info("GazeboEnvironmentHandler initialized")

    async def wait_for_gazebo_ready(self):
        """
        Wait for the Gazebo environment to be ready.
        """
        self.logger.debug("Waiting for Gazebo environment to be ready...")
        await self.wait_for_services()
        self.logger.debug("Gazebo environment is ready")

    async def wait_for_services(self):
        """
        Wait for all required services to be available.
        """
        services = [
            self.reset_world_client,
            self.pause_physics_client,
            self.unpause_physics_client,
            self.set_entity_state_client,
            self.spawn_entity_client,
            self.delete_entity_client,
            self.get_entity_state_client,
        ]
        for client in services:
            while not client.wait_for_service(timeout_sec=1.0):
                self.logger.debug(f"Waiting for {client.srv_name} service...")

    async def pause_gazebo(self) -> bool:
        """
        Pauses the Gazebo environment.
        returns:
            bool: True if the environment was paused successfully, False otherwise.
        """
        self.logger.debug("Pausing Gazebo environment")
        req = Empty.Request()
        try:
            future = await self.pause_physics_client.call_async(req)
            self.logger.debug("Gazebo environment paused successfully")
        except Exception as e:
            self.logger.error(f"Failed to pause Gazebo environment: {e}")
            return False
        return True

    async def resume_gazebo(self) -> bool:
        """
        Resumes the Gazebo environment.
        returns:
            bool: True if the environment was resumed successfully, False otherwise.
        """
        self.logger.debug("Resuming Gazebo environment")
        req = Empty.Request()
        try:
            future = await self.unpause_physics_client.call_async(req)
            self.logger.debug("Gazebo environment resumed successfully")
        except Exception as e:
            self.logger.error(f"Failed to resume Gazebo environment: {e}")
            return False
        return True

    async def reset_the_world(self) -> bool:
        """
        Resets the Gazebo environment.
        returns:
            bool: True if the environment was reset successfully, False otherwise.
        """
        self.logger.debug("Resetting Gazebo environment")
        req = Empty.Request()
        try:
            # await self.reset_world_client.call_async(req)
            future = await self.reset_world_client.call_async(req)
            self.logger.debug("Gazebo environment reset successfully")
        except Exception as e:
            self.logger.error(f"Failed to reset Gazebo environment: {e}")
            return False
        return True

    async def spawn_entity(
        self,
        entity_name: str,
        entity_xml: str,
        robot_namespace: Optional[str] = None,
        initial_pose: Optional[Pose] = None,
        reference_frame: Optional[str] = None,
    ):
        """
        Spawns an entity in the Gazebo environment.
        """
        self.logger.debug(f"Spawning {entity_name} in Gazebo")
        req = SpawnEntity.Request()
        req.name = entity_name
        req.xml = entity_xml
        req.robot_namespace = robot_namespace if robot_namespace else ""
        req.initial_pose = initial_pose if initial_pose else Pose()
        req.reference_frame = reference_frame if reference_frame else ""
        try:
            response = await self.spawn_entity_client.call_async(req)
            if response.success:
                self.logger.debug("Entity spawned successfully")
            else:
                self.logger.error("Failed to spawn entity")
        except Exception as e:
            self.logger.error(f"Failed to spawn entity: {e}")

    async def delete_entity(self, entity_name: str):
        """
        Deletes an entity from the Gazebo environment.
        """
        self.logger.debug(f"Deleting {entity_name} from Gazebo")
        req = DeleteEntity.Request()
        req.name = entity_name
        try:
            response = await self.delete_entity_client.call_async(req)
            if response.success:
                self.logger.debug("Entity deleted successfully")
            else:
                self.logger.error("Failed to delete entity")
        except Exception as e:
            self.logger.error(f"Failed to delete entity: {e}")

    async def set_entities_state(
        self,
        entities: List[EntityState],
        on_done: Optional[Callable[[Dict[str, bool]], None]] = None,
    ) -> Dict[str, bool]:
        """
        Asynchronously set the state of multiple entities in Gazebo.
        Args:
            entities (List[EntityState]): List of EntityState objects to set.
            on_done (Optional[Callable[[Dict[str, bool]], None]]): Callback function
                to be called when the operation is done. It receives a dictionary
                with entity names as keys and success status as values.
        Returns:
            dict: A dictionary with entity names as keys and success status as values.
        Raises:
            ValueError: If the entities list is empty.
        """
        if not entities:
            raise ValueError("Entity list must not be empty.")

        self.logger.debug(f"Setting state for {len(entities)} entities in Gazebo...")
        results = {}

        async def set_state(entity: EntityState):
            try:
                req = SetEntityState.Request()
                req._state = entity
                response = await self.set_entity_state_client.call_async(req)
                results[entity.name] = response.success
                msg = "✔" if response.success else "✖"
                self.logger.debug(
                    f"{msg} Entity '{entity.name}' set: success={response.success}"
                )
            except Exception as e:
                results[entity.name] = False
                self.logger.error(f"⚠ Exception setting entity '{entity.name}': {e}")

        for entity in entities:
            await set_state(entity)

        if on_done:
            self.logger.debug("Calling on_done callback")
            try:
                on_done(results)
            except Exception as e:
                self.logger.error(f"on_done callback raised an exception: {e}")
        return results

    async def reset_environment_for_experiment(
        self, entities: List[EntityState], goal_entity: EntityState, goal_xml: str
    ) -> bool:
        """
        Resets the Gazebo environment.
        """
        self.logger.debug("Resetting Gazebo environment")
        pause_success = await self.pause_gazebo()
        if not pause_success:
            self.logger.error("Failed to pause Gazebo environment")
            return False

        # delete the goal entity if it exists
        while await self.check_entities_in_world([goal_entity.name]):
            await self.delete_entity(goal_entity.name)
        # spawn the goal entity
        reset_success = await self.reset_the_world()
        if not reset_success:
            self.logger.error("Failed to reset Gazebo environment")
            return False
        set_entity_success = await self.set_entities_state(entities)
        if not set_entity_success:
            self.logger.error("Failed to set entities state")
            return False
        while not await self.check_entities_in_world([goal_entity.name]):
            await self.spawn_entity(
                entity_name=goal_entity.name,
                entity_xml=goal_xml,
                initial_pose=goal_entity.pose,
            )
        resume_success = await self.resume_gazebo()
        if not resume_success:
            self.logger.error("Failed to resume Gazebo environment")
            return False

        # Allow physics to settle for a few simulation steps after unpause
        await asyncio.sleep(0.5)

        self.logger.debug("Experiment reset successfully")

        return True

    async def get_entity_state(
        self,
        entity_name: str,
    ) -> EntityState:
        """
        Asynchronously get the state of an entity in Gazebo.
        Args:
            entity_name (str): Name of the entity to get state for.
        Returns:
            EntityState: The state of the entity.
        Raises:
            ValueError: If the entities list is empty.
        """

        if not entity_name:
            raise ValueError("Entity name must not be empty.")
        self.logger.debug(f"Getting state for {entity_name} in Gazebo")
        req = GetEntityState.Request()
        req.name = entity_name
        try:
            response = await self.get_entity_state_client.call_async(req)
            if response.success:
                self.logger.debug("Entity state retrieved successfully")
                return response.state
            else:
                self.logger.error("Failed to get entity state")
        except Exception as e:
            self.logger.error(f"Failed to get entity state: {e}")
            return None
        return None

    async def check_entities_in_world(self, entities_name: List[str]) -> bool:
        """
        Check if multiple entities are in the Gazebo world.
        Args:
            entities_name (List[str]): List of entity names to check.
        Returns:
            bool: True if the entity is in the world, False otherwise.
        """
        if not entities_name or not isinstance(entities_name, list):
            raise ValueError("Entity name must not be empty and must be a list.")
        self.logger.debug(f"Checking if {entities_name} are in the world")
        req = GetModelList.Request()

        try:
            response = await self.get_model_list_client.call_async(req)
            if response.success:
                for entity_name in entities_name:
                    if entity_name not in response.model_names:
                        self.logger.debug(f"Entity '{entity_name}' is not in the world")
                        return False
                self.logger.debug("All entities are in the world")
                return True
            else:
                self.logger.error("Failed to get model list")
        except Exception as e:
            self.logger.error(f"Failed to check entity in world: {e}")
            return False
        return False
