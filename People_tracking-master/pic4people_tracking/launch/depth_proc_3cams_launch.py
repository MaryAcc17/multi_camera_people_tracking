from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    cams = [
        "front_center",
        "front_left",
        "front_right"
    ]

    nodes = []

    for cam in cams:

        ns = f"/camera_{cam}_color"

        node = Node(
            package='depth_image_proc',
            executable='register_node',
            name=f'depth_register_{cam}',
            parameters=[{
                'use_sim_time': True,
                'queue_size': 10
            }],
            remappings=[
                ('depth/image_rect', f'{ns}/depth/image_raw'),
                ('rgb/image_rect_color', f'{ns}/image_raw'),
                ('rgb/camera_info', f'{ns}/camera_info'),
                ('depth_registered/image_rect', f'{ns}/aligned_depth_to_color/image_raw'),
            ],
            output='screen'
        )

        nodes.append(node)

    return LaunchDescription(nodes)