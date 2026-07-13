import rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

def callback(msg):
    bridge = CvBridge()
    depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    print("Depth image dtype:", depth_image.dtype)
    print("Depth min/max:", np.min(depth_image), np.max(depth_image))

rclpy.init()
node = rclpy.create_node('depth_check')
sub = node.create_subscription(Image, '/camera_front_center_color/depth/image_raw', callback, 10)
rclpy.spin(node)