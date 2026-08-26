from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

import re


def launch_robot_description(context, robot_description_parameter: ParameterValue):
    robot_description = robot_description_parameter.evaluate(context)
    print(type(robot_description))
    pattern = r"<!--(.*?)-->"
    robot_description_content = re.sub(pattern, "", robot_description, flags=re.DOTALL)

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description_content}],
        # When launched under a namespace, publish tf to /<ns>/tf (matching the
        # namespaced Nav2 stack) instead of the global /tf.
        remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
    )

    return [rsp]


def generate_launch_description():

    ld = LaunchDescription()

    robot_description_command_arg = DeclareLaunchArgument(
        "robot_description_command",
        default_value=[
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("jackal_description"), "urdf", "jackal.urdf.xacro"]),
        ],
    )

    robot_description_parameter = ParameterValue(
        Command(LaunchConfiguration("robot_description_command")), value_type=str
    )

    # robot_description_content = ParameterValue(
    #     Command(LaunchConfiguration('robot_description_command')),
    #     value_type=str
    # )

    ld.add_action(robot_description_command_arg)
    ld.add_action(OpaqueFunction(function=launch_robot_description, args=[robot_description_parameter]))
    return ld
