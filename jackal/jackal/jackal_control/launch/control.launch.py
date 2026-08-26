from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import UnlessCondition
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    config_jackal_ekf = LaunchConfiguration('config_jackal_localization')
    config_jackal_velocity_controller = LaunchConfiguration('config_jackal_velocity')

    # Configs
    config_jackal_ekf_arg = DeclareLaunchArgument(
        'config_jackal_localization',
        default_value= PathJoinSubstitution(
            [FindPackageShare('jackal_control'),
             'config',
             'localization.yaml'],
        )
    )

    config_imu_filter = PathJoinSubstitution(
        [FindPackageShare('jackal_control'),
         'config',
         'imu_filter.yaml'],
    )

    config_jackal_velocity_controller_arg = DeclareLaunchArgument(
        'config_jackal_velocity',
        default_value= PathJoinSubstitution(
            [FindPackageShare('jackal_control'),
             'config',
             'control.yaml'],
        )
    )

    # Launch Arguments

    robot_description_command_arg = DeclareLaunchArgument(
        'robot_description_command',
        default_value=[
            PathJoinSubstitution([FindExecutable(name='xacro')]),
            ' ',
            PathJoinSubstitution(
                [FindPackageShare('jackal_description'),
                 'urdf', 'jackal.urdf.xacro']
            )
        ]
    )

    gazebo_sim = LaunchConfiguration('gazebo_sim', default=False)

    gazebo_sim_arg = DeclareLaunchArgument(
        'gazebo_sim',
        default_value=gazebo_sim)
    
    isaac_sim = LaunchConfiguration('isaac_sim', default=False)

    isaac_sim_arg = DeclareLaunchArgument(
        'isaac_sim',
        default_value=isaac_sim)
    
    robot_description_content = ParameterValue(
        Command(LaunchConfiguration('robot_description_command')),
        value_type=str
    )

    # Localization
    localization_group_action = GroupAction([
        # Extended Kalman Filter
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_node',
            output='screen',
            parameters=[config_jackal_ekf]
        ),

        # Madgwick Filter
        Node(
            package='imu_filter_madgwick',
            executable='imu_filter_madgwick_node',
            name='imu_filter_node',
            output='screen',
            parameters=[config_imu_filter]
        )
    ])

    # ROS2 Controllers
    control_group_action = GroupAction([
        # ROS2 Control
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            parameters=[{'robot_description': robot_description_content},
                        config_jackal_velocity_controller],
            output={
                'stdout': 'screen',
                'stderr': 'screen',
            },
            condition=UnlessCondition(gazebo_sim)
        ),

        # Joint State Broadcaster
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster'],
            output='screen',
            condition=UnlessCondition(gazebo_sim)
        ),

        # Velocity Controller
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['jackal_velocity_controller'],
            output='screen',
            condition=UnlessCondition(gazebo_sim)
        )
    ])

    ld = LaunchDescription()
    ld.add_action(robot_description_command_arg)
    ld.add_action(gazebo_sim_arg)
    ld.add_action(isaac_sim_arg)
    ld.add_action(localization_group_action)
    ld.add_action(control_group_action)
    ld.add_action(config_jackal_ekf_arg)
    ld.add_action(config_jackal_velocity_controller_arg)
    return ld
