from dataclasses import dataclass

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import (
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch_pal.include_utils import include_scoped_launch_py_description

from gazebo_sim.launch.launch_params_subs import LaunchArgumentsBaseParam
from gazebo_sim.launch_arguments.common import GazeboCommonArgs
from gazebo_sim.launch_arguments.robot import RobotArgs


@dataclass(frozen=True)
class LaunchArguments(LaunchArgumentsBaseParam):
    """Arguments required to spawn a supported robot and its gazebo controllers."""

    robot_name: DeclareLaunchArgument = GazeboCommonArgs.robot_name
    x: DeclareLaunchArgument = GazeboCommonArgs.x
    y: DeclareLaunchArgument = GazeboCommonArgs.y
    z: DeclareLaunchArgument = GazeboCommonArgs.z
    yaw: DeclareLaunchArgument = GazeboCommonArgs.yaw
    use_sim_time: DeclareLaunchArgument = GazeboCommonArgs.use_sim_time
    use_gazebo_controllers: DeclareLaunchArgument = RobotArgs.use_gazebo_controllers
    use_collision_sensor: DeclareLaunchArgument = RobotArgs.use_collision_sensor
    use_lidar_gpu: DeclareLaunchArgument = RobotArgs.use_lidar_gpu


def generate_launch_description() -> LaunchDescription:
    ld = LaunchDescription()
    launch_arguments = LaunchArguments()
    launch_arguments.add_to_launch_description(ld)

    ld.add_action(
        OpaqueFunction(
            function=_spawn_robot_with_controllers,
            args=[
                launch_arguments.launch_configurations_dict(),
            ],
        )
    )

    return ld


def _spawn_robot_with_controllers(context, launch_configurations):
    use_gazebo_controllers = LaunchConfiguration("use_gazebo_controllers").perform(
        context
    )
    use_collision_sensor = LaunchConfiguration("use_collision_sensor").perform(context)
    use_lidar_gpu = LaunchConfiguration("use_lidar_gpu").perform(context)

    # Currently, the only supported robot is Jackal
    # if more robots are added in the future, a conditional statement can be added
    # to check the robot_name and spawn the corresponding robot with its controllers
    # For example:
    robot_name = LaunchConfiguration("robot_name").perform(context)
    actions = []

    entity_description_arguments = (
        [
            "-database",
            robot_name,
        ]
        if robot_name == "ghost_robot"
        else ["-topic", "robot_description"]
    )

    spawn_robot_node = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name="spawn_robot",
        arguments=[
            "-entity",
            robot_name,
        ]
        + entity_description_arguments
        + [
            "-x",
            LaunchConfiguration("x"),
            "-y",
            LaunchConfiguration("y"),
            "-z",
            LaunchConfiguration("z"),
            "-Y",
            LaunchConfiguration("yaw"),
        ],
        output="screen",
    )
    actions.append(spawn_robot_node)

    if robot_name == "jackal":
        config_jackal_velocity_controller = PathJoinSubstitution(
            [FindPackageShare("jackal_gazebo"), "config", "control.yaml"]
        )
        config_jackal_localization = PathJoinSubstitution(
            [FindPackageShare("jackal_gazebo"), "config", "localization.yaml"]
        )
        config_twist_mux = PathJoinSubstitution(
            [FindPackageShare("jackal_gazebo"), "config", "twist_mux.yaml"]
        )

        robot_description_command = [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare("jackal_description"), "urdf", "jackal.urdf.xacro"]
            ),
            " ",
            "use_gazebo_controllers:=",
            use_gazebo_controllers,
            " ",
            "use_collision_sensor:=",
            use_collision_sensor,
            " ",
            "use_lidar_gpu:=",
            use_lidar_gpu,
            " ",
            "gazebo_sim:=True",
            " ",
            "gazebo_controllers:=",
            config_jackal_velocity_controller,
        ]

        description_launch = include_scoped_launch_py_description(
            pkg_name="jackal_description",
            paths=["launch", "description.launch.py"],
            launch_arguments={"robot_description_command": robot_description_command},
        )
        actions.append(description_launch)

        jackal_control = include_scoped_launch_py_description(
            pkg_name="jackal_control",
            paths=["launch", "control.launch.py"],
            launch_arguments={
                "robot_description_command": robot_description_command,
                "gazebo_sim": "True",
                "config_jackal_velocity": config_jackal_velocity_controller,
                "config_jackal_localization": config_jackal_localization,
            },
            condition=IfCondition(use_gazebo_controllers),
        )
        actions.append(jackal_control)
        teleop_base = include_scoped_launch_py_description(
            pkg_name="jackal_control",
            paths=["launch", "teleop_base.launch.py"],
            launch_arguments={
                "config_twist_mux": config_twist_mux,
            },
        )
        actions.append(teleop_base)

        controller_spawners = GroupAction(
            actions=[
                Node(
                    package="controller_manager",
                    executable="spawner",
                    arguments=[
                        "jackal_velocity_controller",
                        "-c",
                        "/controller_manager",
                    ],
                    output="screen",
                    condition=IfCondition(use_gazebo_controllers),
                ),
                Node(
                    package="controller_manager",
                    executable="spawner",
                    arguments=[
                        "joint_state_broadcaster",
                        "-c",
                        "/controller_manager",
                    ],
                    output="screen",
                    condition=IfCondition(use_gazebo_controllers),
                ),
            ]
        )

        controller_spawners_callback = RegisterEventHandler(
            OnProcessExit(
                target_action=spawn_robot_node,
                on_exit=[
                    LogInfo(msg="Robot spawned, now spawning controllers..."),
                    controller_spawners,
                ],
            ),
            condition=IfCondition(use_gazebo_controllers),
        )
        stop_jackal_controllers_on_exit = RegisterEventHandler(
            OnProcessExit(
                target_action=spawn_robot_node,
                on_exit=[
                    ExecuteProcess(
                        cmd=[
                            "ros2 topic pub /stop/cmd_vel geometry_msgs/msg/Twist",
                            '"linear: {}"',
                        ],
                        output="log",
                        shell=True,
                        on_exit=Shutdown(),
                        condition=UnlessCondition(use_gazebo_controllers),
                    ),
                ],
            )
        )
        actions.append(stop_jackal_controllers_on_exit)
        actions.append(controller_spawners_callback)

    return actions
