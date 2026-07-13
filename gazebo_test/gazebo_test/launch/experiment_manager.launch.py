from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.descriptions import ComposableNode, ParameterFile
from nav2_common.launch import RewrittenYaml
from gazebo_sim.launch.launch_params_subs import LaunchArgumentsBaseParam
from dataclasses import dataclass
from launch.actions import OpaqueFunction
from gazebo_sim.launch_arguments.common import GazeboCommonArgs
from gazebo_test.launch_arguments.exp_manager import ExperimentManagerArgs


@dataclass(frozen=True)
class LaunchArguments(LaunchArgumentsBaseParam):
    use_sim_time: DeclareLaunchArgument = GazeboCommonArgs.use_sim_time
    algorithm_name: DeclareLaunchArgument = ExperimentManagerArgs.algorithm_name
    base_path: DeclareLaunchArgument = ExperimentManagerArgs.base_path
    timeout_duration: DeclareLaunchArgument = ExperimentManagerArgs.timeout_duration
    use_recorder: DeclareLaunchArgument = ExperimentManagerArgs.use_recorder
    use_evaluator: DeclareLaunchArgument = ExperimentManagerArgs.use_evaluator
    record_maps: DeclareLaunchArgument = ExperimentManagerArgs.record_maps
    repetitions: DeclareLaunchArgument = ExperimentManagerArgs.repetitions
    goals_and_poses_file: DeclareLaunchArgument = (
        ExperimentManagerArgs.goals_and_poses_file
    )
    wait_before_start: DeclareLaunchArgument = ExperimentManagerArgs.wait_before_start
    exp_config_pkg: DeclareLaunchArgument = ExperimentManagerArgs.exp_config_pkg
    checkpoint_dsn: DeclareLaunchArgument = ExperimentManagerArgs.checkpoint_dsn
    resume_checkpoint: DeclareLaunchArgument = ExperimentManagerArgs.resume_checkpoint


def launch_setup(context, *args, **kwargs):
    """Launch setup function to be used with OpaqueFunction."""
    parameters = ParameterFile(
        RewrittenYaml(
            source_file=PathJoinSubstitution(
                [
                    FindPackageShare(
                        LaunchConfiguration("exp_config_pkg").perform(context)
                    ),
                    "config",
                    "experiment_config.yaml",
                ]
            ),
            param_rewrites=LaunchArguments().param_rewrites_dict(context),
            convert_types=True,
        ),
        allow_substs=True,
    )

    node = Node(
        package="gazebo_test",
        executable="experiment_manager",
        output="screen",
        parameters=[parameters],
    )
    return [node]


def generate_launch_description():
    launch_arguments = LaunchArguments()
    # Specify the actions
    ld = LaunchDescription()

    launch_arguments.add_to_launch_description(ld)
    ld.add_action(OpaqueFunction(function=launch_setup))
    return ld
