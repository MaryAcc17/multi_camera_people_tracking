from launch.actions import DeclareLaunchArgument
from dataclasses import dataclass
from ament_index_python.packages import get_package_share_directory
import os


@dataclass(frozen=True, kw_only=True)
class NavigationArgs:
    bringup_dir = get_package_share_directory("gazebo_experiments")
    """This class contains a collection of frequently used LaunchArguments for the navigation launch file."""

    namespace: DeclareLaunchArgument = DeclareLaunchArgument(
        "namespace",
        default_value="",
        description="Top-level namespace",
    )

    use_namespace: DeclareLaunchArgument = DeclareLaunchArgument(
        "use_namespace",
        default_value="false",
        choices=["true", "false", "True", "False"],
        description="Whether to apply a namespace to the navigation stack",
    )

    map: DeclareLaunchArgument = DeclareLaunchArgument(
        "map",
        default_value=os.path.join(bringup_dir, "maps", "social_nav_map.yaml"),
        description="Full path to map yaml file to load",
    )

    params_file: DeclareLaunchArgument = DeclareLaunchArgument(
        "params_file",
        default_value=os.path.join(bringup_dir, "config", "nav2_params.yaml"),
        description="Full path to the ROS2 parameters file to use for all launched nodes",
    )
    autostart: DeclareLaunchArgument = DeclareLaunchArgument(
        "autostart",
        default_value="true",
        choices=["true", "false"],
        description="Automatically startup the nav2 stack",
    )

    use_composition: DeclareLaunchArgument = DeclareLaunchArgument(
        "use_composition",
        default_value="True",
        choices=["True", "False"],
        description="Use composed bringup if true",
    )

    use_respawn: DeclareLaunchArgument = DeclareLaunchArgument(
        "use_respawn",
        default_value="False",
        choices=["True", "False"],
        description="Whether to respawn if a node crashes. Applied when composition is disabled.",
    )
    container_name: DeclareLaunchArgument = DeclareLaunchArgument(
        "container_name",
        default_value="nav2_container",
        description="The name of the container that nodes will load in if use composition",
    )

    log_level: DeclareLaunchArgument = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        choices=["debug", "info", "warn", "error", "fatal"],
        description="Log level for the nodes",
    )
