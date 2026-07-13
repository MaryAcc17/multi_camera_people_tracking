from launch.actions import DeclareLaunchArgument
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class HunavArgs:
    """This class contains a collection of frequently used LaunchArguments for the hunav_sim launch file."""

    config_pkg_name: DeclareLaunchArgument = DeclareLaunchArgument(
        "config_pkg_name",
        default_value="gazebo_sim",
        description="Package name where the world and configuration files are located.",
    )

    world_pkg_name: DeclareLaunchArgument = DeclareLaunchArgument(
        "world_pkg_name",
        default_value="gazebo_sim",
        description="Package name where the world and configuration files are located.",
    )
    ignore_models: DeclareLaunchArgument = DeclareLaunchArgument(
        "ignore_models",
        default_value="ground_plane cafe",
        description="list of Gazebo models that the agents should ignore as obstacles as the ground_plane. Indicate the models with a blank space between them",
    )
    base_world: DeclareLaunchArgument = DeclareLaunchArgument(
        "base_world",
        default_value="social_nav.world",
        description="Base world to be used. If empty, the world will be the one defined in the robot_description.",
    )

    update_rate: DeclareLaunchArgument = DeclareLaunchArgument(
        "update_rate",
        default_value="40.0",
        description="Update rate of the hunav plugin in Hz",
    )

    global_frame_to_publish: DeclareLaunchArgument = DeclareLaunchArgument(
        "global_frame_to_publish",
        default_value="map",
        description="Name of the global frame in which the position of the agents are provided",
    )
    agents_configuration_file = DeclareLaunchArgument(
        "agents_configuration_file",
        default_value="agents_envs/social_nav.yaml",
        description="Specify configuration file name in the config directory",
    )

    use_gazebo_obs = DeclareLaunchArgument(
        "use_gazebo_obs",
        default_value="true",
        description="Whether to fill the agents obstacles with closest Gazebo obstacle or not",
    )
    use_navgoal_to_start = DeclareLaunchArgument(
        "use_navgoal_to_start",
        default_value="false",
        description="Whether to start the agents movements when a navigation goal is received or not",
    )

    use_collision = DeclareLaunchArgument(
        "use_collision",
        default_value="true",
        description="Whether to use the collision detection or not",
    )
