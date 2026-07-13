from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    SetLaunchConfiguration,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessIO
from launch.substitutions import LaunchConfiguration, PythonExpression

from launch_pal.include_utils import include_scoped_launch_py_description
from gazebo_sim.launch.launch_params_subs import LaunchArgumentsBaseParam
from dataclasses import dataclass
from gazebo_sim.launch_arguments.common import GazeboCommonArgs
from gazebo_sim.launch_arguments.hunav import HunavArgs
from gazebo_sim.launch_arguments.robot import RobotArgs
from gazebo_sim.launch.launch_utils import map_to_odom_identity
from typing import List


@dataclass(frozen=True)
class LaunchArguments(LaunchArgumentsBaseParam):
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
    spawn_robot: DeclareLaunchArgument = RobotArgs.spawn_robot
    start_hunav: DeclareLaunchArgument = GazeboCommonArgs.start_hunav


def launch_gazebo_world(context, *args, **kwargs):
    base_world = LaunchConfiguration("base_world").perform(context)
    if UnlessCondition(LaunchConfiguration("start_hunav")).evaluate(context):
        world_name = base_world
    elif Path(base_world).is_absolute():
        world_name = str(Path(base_world).with_name("generatedWorld.world"))
    else:
        world_name = "generatedWorld.world"
    LogInfo(msg="Launching Gazebo world: " + world_name).execute(context)
    return [
        include_scoped_launch_py_description(
            pkg_name="gazebo_sim",
            paths=["launch", "gazebo_world.launch.py"],
            launch_arguments={
                "world_pkg_name": LaunchConfiguration("world_pkg_name"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "headless": LaunchConfiguration("headless"),
                "world_name": world_name,
            },
        ),
        # Give gzserver time to advertise /spawn_entity before spawning the robot.
        TimerAction(period=5.0, actions=[OpaqueFunction(function=spawn_robot)]),
    ]


def launch_hunav_generation(context, *args, **kwargs):
    configs = LaunchArguments().launch_configurations_dict_with_context(context)
    LogInfo(msg="HuNav include robot_name = " + configs["robot_name"]).execute(context)

    return [
        include_scoped_launch_py_description(
            pkg_name="gazebo_sim",
            paths=["launch", "hunav_generation.launch.py"],
            # be sure that the robot_name argument is included in the launch_arguments and is updated before including the hunav_generation.launch.py
            # this is required because the hunav_generation.launch.py uses the robot_name argument to decide whether to spawn the robot in the world generation or not
            launch_arguments=configs,
            # Only generate the world if the start_hunav argument is true
            condition=IfCondition(LaunchConfiguration("start_hunav")),
        )
    ]


def spawn_robot(context, *args, **kwargs):
    LogInfo(msg="Checking if robot should be spawned...").execute(context)
    if IfCondition(LaunchConfiguration("spawn_robot")).evaluate(context) or IfCondition(
        LaunchConfiguration("start_hunav")
    ).evaluate(context):
        LogInfo(msg="Robot spawn enabled. Spawning robot...").execute(context)
        LogInfo(
            msg="robot_name: " + LaunchConfiguration("robot_name").perform(context)
        ).execute(context)
        LogInfo(
            msg="spawn_robot: " + LaunchConfiguration("spawn_robot").perform(context)
        ).execute(context)
        LogInfo(
            msg="start_hunav: " + LaunchConfiguration("start_hunav").perform(context)
        ).execute(context)
        return [
            include_scoped_launch_py_description(
                pkg_name="gazebo_sim",
                paths=["launch", "spawn_robot.launch.py"],
                launch_arguments=LaunchArguments().launch_configurations_dict_with_context(
                    context
                ),
            )
        ]
    else:
        return [
            LogInfo(
                msg="Robot spawn disabled. Launch `spawn_robot.launch.py` when ready."
            )
        ]


def generate_launch_description():

    ld = LaunchDescription()
    # Declare launch arguments
    launch_arguments = LaunchArguments()

    launch_arguments.add_to_launch_description(ld)

    robot_name_arg = SetLaunchConfiguration(
        name="robot_name",
        value="ghost_robot",  # Set a default robot name to avoid issues with the hunav_generation.launch.py when spawn_robot is false
        condition=UnlessCondition(LaunchConfiguration("spawn_robot")),
    )

    ld.add_action(robot_name_arg)

    hunav_world_generation = OpaqueFunction(
        function=launch_hunav_generation,
    )

    launch_simulation_without_hunav = OpaqueFunction(
        function=launch_gazebo_world,
        condition=UnlessCondition(LaunchConfiguration("start_hunav")),
    )

    launch_simulation = RegisterEventHandler(
        OnProcessIO(
            on_stdout=lambda event: (
                [
                    LogInfo(msg="Gazebo starting..."),
                    OpaqueFunction(function=launch_gazebo_world),
                    LogInfo(msg="Gazebo started"),
                ]
                if b"generation finished" in event.text
                else []
            ),
        ),
        condition=IfCondition(LaunchConfiguration("start_hunav")),
    )
    ld.add_action(hunav_world_generation)
    ld.add_action(launch_simulation_without_hunav)
    ld.add_action(launch_simulation)
    ld.add_action(
        map_to_odom_identity(use_sim_time=LaunchConfiguration("use_sim_time"))
    )

    return ld
