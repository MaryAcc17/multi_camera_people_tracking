from dataclasses import dataclass

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    SetEnvironmentVariable,
)
from launch.conditions import UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.substitutions import FindPackageShare

from gazebo_sim.launch.launch_params_subs import LaunchArgumentsBaseParam
from gazebo_sim.launch.launch_utils import path_to_file_in_pkg
from gazebo_sim.launch_arguments.common import GazeboCommonArgs
from gazebo_sim.launch_arguments.hunav import HunavArgs


@dataclass(frozen=True)
class LaunchArguments(LaunchArgumentsBaseParam):
    """Common arguments required to bring up the Gazebo world."""

    world_name: DeclareLaunchArgument = GazeboCommonArgs.world_name
    world_pkg_name: DeclareLaunchArgument = HunavArgs.world_pkg_name
    use_sim_time: DeclareLaunchArgument = GazeboCommonArgs.use_sim_time
    headless: DeclareLaunchArgument = GazeboCommonArgs.headless


def generate_launch_description() -> LaunchDescription:
    ld = LaunchDescription()

    launch_arguments = LaunchArguments()
    launch_arguments.add_to_launch_description(ld)

    world_path = path_to_file_in_pkg(
        pkg_name=LaunchConfiguration("world_pkg_name"),
        paths=["worlds", LaunchConfiguration("world_name")],
    )

    gz_server_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("gazebo_ros"), "launch", "gzserver.launch.py"]
            )
        ),
        launch_arguments={
            "world": world_path,
            "verbose": "true",
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }.items(),
    )

    gz_client_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("gazebo_ros"), "launch", "gzclient.launch.py"]
            )
        ),
        launch_arguments={"use_sim_time": LaunchConfiguration("use_sim_time")}.items(),
        condition=UnlessCondition(LaunchConfiguration("headless")),
    )

    ld.add_action(LogInfo(msg="Launching Gazebo server..."))
    ld.add_action(gz_server_launch)
    ld.add_action(gz_client_launch)

    return ld
