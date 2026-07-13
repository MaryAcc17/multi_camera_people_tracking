import os
from pathlib import Path
from typing import Dict
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, PoseStamped
from tf_transformations import quaternion_from_euler
import yaml


def parse_entity_state_yaml(yaml_path: Path) -> Dict[str, Dict[str, EntityState]]:
    """
    Parse data to create a list of EntityState objects.
    This method reads the yaml file from the provided path and converts it into
    a list of EntityState objects.
    The data should contain the necessary information to create EntityState
    objects, such as name, pose, and other attributes.


    Args:
        yaml_path (Path): Path to the YAML file containing the entity state data.
    Returns:
        Dict[str, Dict[str, EntityState]]: A dictionary with two keys:
            'initial_state_entities' and 'goal_entities', each containing a
            dictionary mapping episode names to their corresponding EntityState
            objects.
    """

    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)

    episodes = data.get("episodes", [])

    goals = data.get("goals", {})
    poses = data.get("poses", {})

    robot_name = data.get("robot_name", "robot")
    goal_name = data.get("goal_name", "goal_box")

    initial_state_entities = {}
    goal_entities = {}

    for episode in episodes:
        goal = goals.get(episode, [])
        pose = poses.get(episode, [])

        goal_entity = EntityState()
        goal_entity.name = goal_name
        goal_entity.pose = Pose()
        goal_entity.pose.position.x = goal[0]
        goal_entity.pose.position.y = goal[1]

        initial_state_entity = EntityState()
        initial_state_entity.name = robot_name
        initial_state_entity.pose = Pose()

        # Position
        initial_state_entity.pose.position.x = pose[0]
        initial_state_entity.pose.position.y = pose[1]
        initial_state_entity.pose.position.z = 0.07
        # Orientation
        quaternion = quaternion_from_euler(0, 0, pose[2])
        initial_state_entity.pose.orientation.x = quaternion[0]
        initial_state_entity.pose.orientation.y = quaternion[1]
        initial_state_entity.pose.orientation.z = quaternion[2]
        initial_state_entity.pose.orientation.w = quaternion[3]
        # Reference frame

        goal_entities[episode] = goal_entity
        initial_state_entities[episode] = initial_state_entity
    return {
        "initial_state_entities": initial_state_entities,
        "goal_entities": goal_entities,
    }


def get_posestamped_from_entity(entity: EntityState, frame_id: str) -> PoseStamped:
    """
    Get the PoseStamped object from the entity state.
    This method takes an EntityState object and returns a PoseStamped object
    with the pose of the entity.
    Args:
        entity (EntityState): The entity state object.
        frame_id (str): The frame ID for the pose stamped.
    Returns:
        PoseStamped: The pose stamped object with the goal pose.
    """
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = frame_id
    pose_stamped.pose = entity.pose
    return pose_stamped
