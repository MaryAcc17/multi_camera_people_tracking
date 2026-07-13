from typing import List, Optional, Union
from launch.substitutions import PathJoinSubstitution
from launch import SomeSubstitutionsType
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def path_to_file_in_pkg(
    pkg_name: SomeSubstitutionsType, paths: List[SomeSubstitutionsType], **kwargs
) -> PathJoinSubstitution:
    """
    Return a PathJoinSubstitution object that can be used in a launch file to be used
    as a path to a file in a package.

    Example:
    -------
    .. code-block:: python

        path_to_file_in_pkg(
            pkg_name="my_package",
            paths=["launch", "my_launch_file.launch.py"]
        )
        # This will return file IncludeLaunchDescription from PATH_TO_MY_PKG_SHARE/launch/my_launch_file.launch.py

    """

    return PathJoinSubstitution([FindPackageShare(pkg_name)] + paths)


def map_to_odom_identity(use_sim_time: SomeSubstitutionsType) -> Node:
    """
    Return a Node that publishes the identity transform from map to odom.
    This is used to make the map frame the same as the odom frame.
    """
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_odom",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
        arguments=["--frame-id", "map", "--child-frame-id", "odom"],
    )


def spawn_goal_entity(
    x: Optional[SomeSubstitutionsType] = 0.0, y: Optional[SomeSubstitutionsType] = 0.0
) -> Node:
    """
    Spawn a goal entity in the simulation.
    This is used to indicate the goal position for the robot.
    """

    goal_entity = PathJoinSubstitution(
        [FindPackageShare("gazebo_sim")] + ["models", "goal_box", "model.sdf"]
    )

    return Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        output="screen",
        arguments=["-entity", "goal", "-file", goal_entity, "-x", x, "-y", y],
    )


def parse_launch_config_value(value: str) -> Union[str, bool, int, float, List[str]]:
    """
    Parse a LaunchConfiguration string into an appropriate Python type:
    - "true" / "false" → bool
    - numeric strings → int or float
    - comma-separated strings → list
    - fallback → string
    """
    lowered = value.lower()

    if lowered in ("true", "1", "yes"):
        return True
    if lowered in ("false", "0", "no"):
        return False

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    # Try parsing as list
    if "," in value:
        return [item.strip() for item in value.split(",")]

    return value  # fallback: return as string
