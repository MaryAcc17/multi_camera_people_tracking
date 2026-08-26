from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration

from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    world_file = PathJoinSubstitution(
        ['scenario1.world']
    )

    gazebo_launch = PathJoinSubstitution(
        [FindPackageShare('jackal_gazebo'),
         'launch',
         'gazebo.launch.py'],
    )

    world_arg = DeclareLaunchArgument(
        'world', default_value=world_file, description='Gazebo world file')
    
    gazebo_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([gazebo_launch]),
        launch_arguments={'world_path': LaunchConfiguration('world')}.items()
    )
    
    ld = LaunchDescription()
    ld.add_action(world_arg)
    ld.add_action(gazebo_sim)

    return ld
