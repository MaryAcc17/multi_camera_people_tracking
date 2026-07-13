from launch_ros.substitutions import FindPackageShare
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, EmitEvent,
                            LogInfo, RegisterEventHandler, TimerAction, OpaqueFunction,
                            ExecuteProcess)
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (PathJoinSubstitution, LaunchConfiguration, 
                                  LocalSubstitution, PythonExpression)
from launch.event_handlers import (OnProcessExit, OnShutdown)
from launch.conditions import IfCondition
from launch.events import Shutdown


def generate_launch_description():
    node_params = LaunchConfiguration('node_params')
    node_params_launch_arg = DeclareLaunchArgument(
        'node_params',
        default_value = os.path.join(
            get_package_share_directory('pic4people_tracking'), 
            'params',
            'params.yaml'
        )
    )

    rosbag_filename = LaunchConfiguration('rosbag_filename')
    rosbag_filename_launch_arg = DeclareLaunchArgument(
        'rosbag_filename',
        default_value=''
    )

    record_results = LaunchConfiguration('record_results')
    record_results_launch_arg = DeclareLaunchArgument(
        'record_results',
        default_value='false'
    )

    export_to_csv = LaunchConfiguration('to_csv')
    export_to_csv_launch_arg = DeclareLaunchArgument(
        'to_csv',
        default_value='false'
    )

    # include node, which will run only if arg export_to_csv is set to True
    to_csv_node = Node(
        package='pic4people_tracking',
        executable='to_csv_node',
        name='to_csv_node',
        parameters=[
            node_params,
            {
            'tracker': 'strongsort_pose',
            'filepath': rosbag_filename,
            }
        ],
        condition = IfCondition(export_to_csv)
    )

    # include main node
    tracker_node = Node(
        package='pic4people_tracking',
        executable='tracker',
        name='tracker',
        parameters=[node_params]
    )

    # include realsense launch, which is launched only if the bag args is empty or invalid
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([         
                FindPackageShare('pic4people_tracking'),
                'launch',
                'realsense_launch.py'
            ])
        ]),
        launch_arguments={
            'rosbag_file': rosbag_filename,
            'depth_module.depth_profile': '640x480x30', 
            'rgb_camera.color_profile': '640x480x30', 
            'spatial_filter.filter_magnitude': '5',
            'spatial_filter.holes_fill': '5', 
            'decimation_filter.filter_magnitude': '4',  
            'spatial_filter.enable': 'true',  
            'decimation_filter.enable': 'true'
        }.items()
    )

    # include bag handlers
    db3_bag_launch = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', rosbag_filename],
        output='screen'        
    )

    bag_event_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=db3_bag_launch,
            on_exit=[
                EmitEvent(event=Shutdown(reason='Bag playback finished'))
            ]
        )
    )

    # include functions handling bags
    def get_stream_launch(context):
        bag_name = rosbag_filename.perform(context)
        if bag_name.endswith(".db3"):
            return [TimerAction(period=5.0, actions=[db3_bag_launch])]
        # elif bag_name.endswith(".bag"):
        #     return [TimerAction(period=3.0, actions=[realsense_launch])]
        # else:
        #     return [realsense_launch] 

    def get_record_results(context):
        if record_results.perform(context) == 'true':
            return [ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'record', '-o',
                    PathJoinSubstitution([os.path.dirname(rosbag_filename.perform(context)), 'results_bag']),
                    'vicon/people', 'tracked_people'],
                output='screen'
            )]
        return []
    
    return LaunchDescription([
        node_params_launch_arg,
        rosbag_filename_launch_arg,
        record_results_launch_arg,
        export_to_csv_launch_arg,
        tracker_node,
        to_csv_node,
        OpaqueFunction(function=get_stream_launch),
        OpaqueFunction(function=get_record_results),
        # bag_event_handler,
        RegisterEventHandler(
            OnProcessExit(
                target_action=tracker_node,
                on_exit=[
                    LogInfo(msg=('realsense sort tracker closed')),
                    EmitEvent(event=Shutdown(
                        reason='Window closed'))
                ]
            )
        ),  
        RegisterEventHandler(
            OnShutdown(
                on_shutdown=[LogInfo(
                    msg=['Launch was asked to shutdown: ',
                         LocalSubstitution('event.reason')]
                    )
                ]
            )
        ),
    ])