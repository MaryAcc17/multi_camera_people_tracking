from launch.actions import (
    RegisterEventHandler,
    TimerAction,
    LogInfo,
    DeclareLaunchArgument,
    SetLaunchConfiguration,
    OpaqueFunction,
)

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.event_handlers import OnProcessIO, OnProcessStart
from gazebo_sim.launch_arguments.common import GazeboCommonArgs
from gazebo_sim.launch_arguments.hunav import HunavArgs
from gazebo_sim.launch.launch_utils import (
    path_to_file_in_pkg,
    parse_launch_config_value,
)

from dataclasses import dataclass
from launch_pal.arg_utils import LaunchArgumentsBase
from launch.substitutions import LaunchConfiguration

# import execute_process
from launch.actions import ExecuteProcess


@dataclass(frozen=True)
class LaunchArguments(LaunchArgumentsBase):
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
    use_navgoal_to_start: DeclareLaunchArgument = HunavArgs.use_navgoal_to_start
    use_collision: DeclareLaunchArgument = HunavArgs.use_collision


def _event_contains(event, text: str) -> bool:
    event_text = (
        event.text.decode() if isinstance(event.text, (bytes, bytearray)) else event.text
    )
    return text in event_text


def _launch_hunav_world_generator(context, *args, **kwargs):

    def _resolve_config(name: str):
        return parse_launch_config_value(LaunchConfiguration(name).perform(context))

    agent_conf_file = path_to_file_in_pkg(
        pkg_name=LaunchConfiguration("config_pkg_name"),
        paths=["config", LaunchConfiguration("agents_configuration_file")],
    ).perform(context)

    hunav_loader_node = Node(
        package="hunav_agent_manager",
        executable="hunav_loader",
        output="screen",
        parameters=[agent_conf_file],
    )

    world_file = path_to_file_in_pkg(
        pkg_name=LaunchConfiguration("world_pkg_name"),
        paths=["worlds", LaunchConfiguration("base_world")],
    ).perform(context)

    hunav_gazebo_worldgen_node = Node(
        package="hunav_gazebo_wrapper",
        executable="hunav_gazebo_world_generator",
        output="screen",
        parameters=[
            {"base_world": world_file},
            {"use_gazebo_obs": _resolve_config("use_gazebo_obs")},
            {"update_rate": _resolve_config("update_rate")},
            {"robot_name": _resolve_config("robot_name")},
            {"global_frame_to_publish": _resolve_config("global_frame_to_publish")},
            {"use_navgoal_to_start": _resolve_config("use_navgoal_to_start")},
            {"ignore_models": _resolve_config("ignore_models")},
            {"use_collision": _resolve_config("use_collision")},
        ],
    )

    ordered_launch_event = RegisterEventHandler(
        OnProcessStart(
            target_action=hunav_loader_node,
            on_start=[
                LogInfo(
                    msg="HunNavLoader started, launching HuNav_Gazebo_world_generator after 2 seconds..."
                ),
                TimerAction(
                    period=5.0,
                    actions=[hunav_gazebo_worldgen_node],
                ),
            ],
        )
    )

    ordered_launch_event_2 = RegisterEventHandler(
        OnProcessIO(
            target_action=hunav_gazebo_worldgen_node,
            on_stdout=lambda event: [
                ExecuteProcess(
                    cmd=["echo", "Hunav world generation finished"],
                    output="screen",
                )
            ]
            if _event_contains(event, "New world file created!")
            else [],
            on_stderr=lambda event: [
                ExecuteProcess(
                    cmd=["echo", "Hunav world generation finished"],
                    output="screen",
                )
            ]
            if _event_contains(event, "New world file created!")
            else [],
        )
    )

    hunav_manager_node = Node(
        package="hunav_agent_manager",
        executable="hunav_agent_manager",
        name="hunav_agent_manager",
        output="screen",
        parameters=[{"use_sim_time": _resolve_config("use_sim_time")}],
    )

    return [
        hunav_loader_node,
        ordered_launch_event,
        hunav_manager_node,
        ordered_launch_event_2,
    ]


def generate_launch_description():

    launch_arguments = LaunchArguments()
    ld = LaunchDescription()
    launch_arguments.add_to_launch_description(ld)

    ld.add_action(OpaqueFunction(function=_launch_hunav_world_generator))
    return ld
