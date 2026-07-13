from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    this_pkg_dir = get_package_share_directory("gazebo_test")
    rviz_config_file = LaunchConfiguration("rviz_config_file")

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],
        parameters=[
            {
                "use_sim_time": True,
            },
        ],
    )

    # Declare the launch arguments
    declare_rviz_config_file = DeclareLaunchArgument(
        "rviz_config_file",
        default_value=os.path.join(this_pkg_dir, "rviz", "jackal.rviz"),
        description="Full path to the RViz config file to use",
    )

    ld = LaunchDescription()
    ld.add_action(declare_rviz_config_file)
    ld.add_action(rviz_node)
    return ld
