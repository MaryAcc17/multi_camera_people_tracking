from launch.actions import DeclareLaunchArgument
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class RobotArgs:
    """This class contains a collection of frequently used LaunchArguments for the robot launch file."""

    use_gazebo_controllers: DeclareLaunchArgument = DeclareLaunchArgument(
        "use_gazebo_controllers",
        default_value="false",
        choices=["true", "false", "True", "False"],
        description="Whether to use Gazebo controllers or not",
    )
    use_collision_sensor: DeclareLaunchArgument = DeclareLaunchArgument(
        "use_collision_sensor",
        default_value="true",
        choices=["true", "false", "True", "False"],
        description="Whether to use collision sensor or not",
    )
    use_lidar_gpu: DeclareLaunchArgument = DeclareLaunchArgument(
        "use_lidar_gpu",
        default_value="true",
        choices=["true", "false", "True", "False"],
        description="Whether to use GPU for LIDAR processing or not",
    )
    spawn_robot: DeclareLaunchArgument = DeclareLaunchArgument(
        "spawn_robot",
        default_value="true",
        choices=["true", "false", "True", "False"],
        description="Whether to spawn the robot in Gazebo",
    )
