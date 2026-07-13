from launch.actions import DeclareLaunchArgument
from dataclasses import dataclass
from ament_index_python.packages import get_package_share_directory
import os


@dataclass(frozen=True, kw_only=True)
class ExperimentManagerArgs:
    """This class contains a collection of frequently used LaunchArguments for the experiment manager launch file."""

    algorithm_name: DeclareLaunchArgument = DeclareLaunchArgument(
        "algorithm_name",
        default_value="",
        description="Specify the name of the algorithm to be used",
    )
    base_path: DeclareLaunchArgument = DeclareLaunchArgument(
        "base_path",
        default_value="",
        description="Base path for the experiment manager configuration files",
    )
    timeout_duration: DeclareLaunchArgument = DeclareLaunchArgument(
        "timeout_duration",
        default_value="",
        description="Timeout duration for the experiment manager",
    )
    use_recorder: DeclareLaunchArgument = DeclareLaunchArgument(
        "use_recorder",
        choices=["true", "false", "True", "False"],
        default_value="false",
        description="Specify whether to use the bag recorder",
    )
    use_evaluator: DeclareLaunchArgument = DeclareLaunchArgument(
        "use_evaluator",
        choices=["true", "false", "True", "False"],
        default_value="false",
        description="Specify whether to use the hunav evaluator",
    )
    record_maps: DeclareLaunchArgument = DeclareLaunchArgument(
        "record_maps",
        choices=["true", "false", "True", "False"],
        default_value="false",
        description="Specify whether to record map topics in the bag files",
    )

    wait_before_start: DeclareLaunchArgument = DeclareLaunchArgument(
        "wait_before_start",
        default_value="0",
        description="Number of seconds to wait before starting the experiment (to let the environment settle)",
    )

    repetitions: DeclareLaunchArgument = DeclareLaunchArgument(
        "repetitions",
        default_value="",
        description="Number of repetitions for the experiment",
    )
    goals_and_poses_file: DeclareLaunchArgument = DeclareLaunchArgument(
        "goals_and_poses_file",
        default_value="",
        description="Path to the YAML file containing goals and poses",
    )
    exp_config_pkg: DeclareLaunchArgument = DeclareLaunchArgument(
        "exp_config_pkg",
        default_value="gazebo_experiments",
        description="Package containing the experiment manager configuration files",
    )
    checkpoint_dsn: DeclareLaunchArgument = DeclareLaunchArgument(
        "checkpoint_dsn",
        default_value="",
        description="PostgreSQL DSN for the shared checkpoint store",
    )
    resume_checkpoint: DeclareLaunchArgument = DeclareLaunchArgument(
        "resume_checkpoint",
        choices=["true", "false", "True", "False"],
        default_value="false",
        description="Resume experiments by skipping runs already marked in the checkpoint",
    )
