from dataclasses import dataclass

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    SetLaunchConfiguration,
    LogInfo,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile

from launch_pal.include_utils import include_scoped_launch_py_description
from gazebo_sim.launch_arguments.hunav import HunavArgs
from gazebo_sim.launch_arguments.robot import RobotArgs
from gazebo_sim.launch_arguments.common import GazeboCommonArgs

from gazebo_sim.launch.launch_params_subs import LaunchArgumentsBaseParam
from gazebo_test.script_utils.parsing_utils import parse_experiment_name
from nav2_common.launch import RewrittenYaml


@dataclass(frozen=True)
class SimulationLaunchArguments(LaunchArgumentsBaseParam):
    agents_configuration_file: DeclareLaunchArgument = (
        HunavArgs.agents_configuration_file
    )
    config_pkg_name: DeclareLaunchArgument = HunavArgs.config_pkg_name
    world_pkg_name: DeclareLaunchArgument = HunavArgs.world_pkg_name
    use_sim_time: DeclareLaunchArgument = GazeboCommonArgs.use_sim_time
    base_world: DeclareLaunchArgument = HunavArgs.base_world
    use_gazebo_obs: DeclareLaunchArgument = HunavArgs.use_gazebo_obs
    update_rate: DeclareLaunchArgument = HunavArgs.update_rate
    robot_name: DeclareLaunchArgument = GazeboCommonArgs.robot_name
    global_frame_to_publish: DeclareLaunchArgument = HunavArgs.global_frame_to_publish
    ignore_models: DeclareLaunchArgument = HunavArgs.ignore_models
    x: DeclareLaunchArgument = GazeboCommonArgs.x
    y: DeclareLaunchArgument = GazeboCommonArgs.y
    yaw: DeclareLaunchArgument = GazeboCommonArgs.yaw
    use_navgoal_to_start: DeclareLaunchArgument = HunavArgs.use_navgoal_to_start
    use_collision: DeclareLaunchArgument = HunavArgs.use_collision
    use_gazebo_controllers: DeclareLaunchArgument = RobotArgs.use_gazebo_controllers
    use_lidar_gpu: DeclareLaunchArgument = RobotArgs.use_lidar_gpu
    use_collision_sensor: DeclareLaunchArgument = RobotArgs.use_collision_sensor
    goal_x: DeclareLaunchArgument = GazeboCommonArgs.goal_x
    goal_y: DeclareLaunchArgument = GazeboCommonArgs.goal_y
    headless: DeclareLaunchArgument = GazeboCommonArgs.headless
    spawn_robot: DeclareLaunchArgument = DeclareLaunchArgument(
        "spawn_robot",
        default_value="false",
        description="Whether to spawn the robot in the simulation. Set to false if the robot is already present in the world file or will be spawned externally.",
    )


@dataclass(frozen=True)
class SnapshotExperimentArgs(LaunchArgumentsBaseParam):
    experiment: DeclareLaunchArgument = DeclareLaunchArgument(
        "experiment",
        default_value="",
        description=(
            "Experiment identifier defined in gazebo_experiments. "
            "When set, launch arguments and snapshot goals will be populated using the experiment configuration."
        ),
    )


@dataclass(frozen=True)
class SnapshotCameraArgs(LaunchArgumentsBaseParam):
    camera_config_file: DeclareLaunchArgument = DeclareLaunchArgument(
        "camera_config_file",
        default_value="",
        description="Absolute path to the YAML file providing environment_camera parameters.",
    )
    output_dir: DeclareLaunchArgument = DeclareLaunchArgument(
        "output_dir",
        default_value="",
        description="Directory where the captured snapshot will be stored.",
    )
    experiment_tag: DeclareLaunchArgument = DeclareLaunchArgument(
        "experiment_tag",
        default_value="",
        description="Identifier used in the generated filename.",
    )
    stage_label: DeclareLaunchArgument = DeclareLaunchArgument(
        "stage_label",
        default_value="",
        description="Label describing the snapshot stage.",
    )
    run_id: DeclareLaunchArgument = DeclareLaunchArgument(
        "run_id",
        default_value="",
        description="Run id embedded in the snapshot filename.",
    )
    capture_delay_sec: DeclareLaunchArgument = DeclareLaunchArgument(
        "capture_delay_sec",
        default_value="",
        description="Delay in seconds before taking the first capture.",
    )
    camera_position: DeclareLaunchArgument = DeclareLaunchArgument(
        "camera_position",
        default_value="",
        description="Camera position (x, y, z).",
    )
    camera_orientation_rpy: DeclareLaunchArgument = DeclareLaunchArgument(
        "camera_orientation_rpy",
        default_value="",
        description="Camera orientation as roll, pitch, yaw (radians).",
    )
    camera_model_path: DeclareLaunchArgument = DeclareLaunchArgument(
        "camera_model_path",
        default_value="",
        description="Override path to the camera SDF model.",
    )
    record_people_trajectories: DeclareLaunchArgument = DeclareLaunchArgument(
        "record_people_trajectories",
        default_value="",
        description="Enable recording of people trajectories while capturing the snapshot.",
    )
    people_trajectory_sample_period_sec: DeclareLaunchArgument = DeclareLaunchArgument(
        "people_trajectory_sample_period_sec",
        default_value="",
        description="Minimum period (seconds) between samples per agent while recording trajectories.",
    )
    people_trajectory_duration_sec: DeclareLaunchArgument = DeclareLaunchArgument(
        "people_trajectory_duration_sec",
        default_value="",
        description="Minimum total duration in seconds to record people trajectories before exporting.",
    )
    image_timeout_sec: DeclareLaunchArgument = DeclareLaunchArgument(
        "image_timeout_sec",
        default_value="",
        description="Maximum time to wait for the camera image (seconds).",
    )


def _apply_experiment_overrides(context, *args, **kwargs):
    experiment_id = LaunchConfiguration("experiment").perform(context).strip()
    if not experiment_id:
        return []
    try:
        experiment_dict = parse_experiment_name(experiment_id)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to resolve experiment '{experiment_id}': {exc}"
        ) from exc

    return [
        SetLaunchConfiguration("base_world", experiment_dict.get("world", "")),
        SetLaunchConfiguration(
            "world_pkg_name", experiment_dict.get("world_pkg_name", "")
        ),
        SetLaunchConfiguration(
            "config_pkg_name", experiment_dict.get("agents_pkg_name", "")
        ),
        SetLaunchConfiguration(
            "agents_configuration_file",
            experiment_dict.get("agents_configuration_file", ""),
        ),
        SetLaunchConfiguration(
            "experiment_tag",
            experiment_id,
        ),
        SetLaunchConfiguration(
            "camera_config_file",
            experiment_dict.get("camera_config_file", ""),
        ),
    ]


def generate_launch_description():
    ld = LaunchDescription()

    experiment_args = SnapshotExperimentArgs()
    experiment_args.add_to_launch_description(ld)

    sim_args = SimulationLaunchArguments()
    sim_args.add_to_launch_description(ld)

    camera_args = SnapshotCameraArgs()
    camera_args.add_to_launch_description(ld)

    def _include_simulation(context, *args, **kwargs):
        return [
            include_scoped_launch_py_description(
                pkg_name="gazebo_sim",
                paths=["launch", "simulation.launch.py"],
                launch_arguments=sim_args.launch_configurations_dict_with_context(
                    context
                ),
            )
        ]

    ld.add_action(OpaqueFunction(function=_apply_experiment_overrides))
    ld.add_action(OpaqueFunction(function=_include_simulation))

    def _launch_snapshot_node(context, *args, **kwargs):
        camera_config_path = (
            LaunchConfiguration("camera_config_file").perform(context).strip()
        )
        if not camera_config_path:
            raise RuntimeError(
                "camera_config_file must be provided via the experiment "
                "configuration or launch argument."
            )
        LogInfo(msg=f"Using camera config file: {camera_config_path}").execute(context)
        param_rewrites = SnapshotCameraArgs().param_rewrites_dict(context)
        LogInfo(msg=f"Camera param rewrites: {param_rewrites}").execute(context)

        camera_parameters = ParameterFile(
            RewrittenYaml(
                source_file=camera_config_path,
                param_rewrites=param_rewrites,
                convert_types=True,
            ),
            allow_substs=True,
        )

        snapshot_node = Node(
            package="gazebo_test",
            executable="environment_camera",
            name="environment_camera",
            output="screen",
            parameters=[camera_parameters],
        )
        return [snapshot_node]

    ld.add_action(OpaqueFunction(function=_launch_snapshot_node))

    return ld
