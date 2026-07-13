import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    experiment_pkg_dir = get_package_share_directory("gazebo_experiments")
    this_pkg_dir = get_package_share_directory("rviz_goal_pose_annotator")
    default_map_yaml = os.path.join(
        experiment_pkg_dir, "maps", "social_env_test_map.yaml"
    )
    start_map_server = LaunchConfiguration("start_map_server")
    map_yaml_path = LaunchConfiguration("map_yaml_path")
    rviz_config_file = LaunchConfiguration("rviz_config_file")

    declare_start_map_server = DeclareLaunchArgument(
        "start_map_server",
        default_value="true",
        description="Whether to launch nav2_map_server.",
    )
    declare_map_yaml_path = DeclareLaunchArgument(
        "map_yaml_path",
        default_value=default_map_yaml,
        description="Full path to map YAML for map_server.",
    )
    declare_rviz_config_file = DeclareLaunchArgument(
        "rviz_config_file",
        default_value=os.path.join(this_pkg_dir, "rviz", "rviz_annotator.rviz"),
        description="Full path to RViz config file.",
    )

    map_server_node = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "yaml_filename": map_yaml_path,
            },
        ],
        condition=IfCondition(start_map_server),
    )

    map_server_lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_map_server",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "autostart": True,
                "node_names": ["map_server"],
            },
        ],
        condition=IfCondition(start_map_server),
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],
        parameters=[{"use_sim_time": True}],
    )
    return LaunchDescription(
        [
            declare_start_map_server,
            declare_map_yaml_path,
            declare_rviz_config_file,
            map_server_node,
            map_server_lifecycle_manager,
            rviz_node,
        ]
    )
