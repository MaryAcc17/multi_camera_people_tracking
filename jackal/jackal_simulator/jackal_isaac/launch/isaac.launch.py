from launch import LaunchDescription
from launch.actions import LogInfo, DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, SetEnvironmentVariable, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, FindExecutable, LaunchConfiguration, PathJoinSubstitution, Command

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

from pathlib import Path

from ament_index_python.packages import get_package_share_directory

ARGUMENTS = [
    DeclareLaunchArgument('world_path', default_value='',
                          description='The world path, by default is empty.world'),
]


def generate_launch_description():

    # Launch args
    prefix = LaunchConfiguration('prefix')

    config_jackal_velocity_controller = PathJoinSubstitution(
        [FindPackageShare('jackal_isaac'), 'config', 'control.yaml']
    )

    config_jackal_localization = PathJoinSubstitution(
        [FindPackageShare('jackal_isaac'), 'config', 'localization.yaml'],
    )

    config_twist_mux = PathJoinSubstitution(
        [FindPackageShare('jackal_isaac'), 'config', 'twist_mux.yaml']
    )

    # Get URDF via xacro
    robot_description_command = [
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ',
        PathJoinSubstitution(
            [FindPackageShare('jackal_description'),
             'urdf', 'jackal.urdf.xacro']
        ),
        ' ',
        'isaac_sim:=True',
    ]

    launch_jackal_description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('jackal_description'),
                 'launch',
                 'description.launch.py']
            )
        ),
        launch_arguments=[('robot_description_command',
                           robot_description_command)]
    )

    

    launch_jackal_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('jackal_control'),
                 'launch',
                 'control.launch.py']
            )
        ),
        launch_arguments=[('robot_description_command', robot_description_command),
                          ('isaac_sim', 'True'),
                          ('config_jackal_velocity', config_jackal_velocity_controller),
                          ('config_jackal_localization', config_jackal_localization),
                        ]
    )

    # Launch jackal_control/teleop_base.launch.py which is various ways to tele-op
    # the robot but does not include the joystick. Also, has a twist mux.
    launch_jackal_teleop_base = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('jackal_control'), 'launch', 'teleop_base.launch.py'])),
        launch_arguments=[('config_twist_mux', config_twist_mux),]
    )

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(launch_jackal_description)
    ld.add_action(launch_jackal_control)
    ld.add_action(launch_jackal_teleop_base)

    return ld