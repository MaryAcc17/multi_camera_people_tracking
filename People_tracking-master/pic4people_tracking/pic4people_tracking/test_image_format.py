import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import os
from openvino.runtime import Core

# ROS2 helper per trovare la cartella share del pacchetto
from ament_index_python.packages import get_package_share_directory

class ImageTester(Node):
    def __init__(self):
        super().__init__('image_tester')
        self.subscription = self.create_subscription(
            Image,
            '/camera_front_center_color/image_raw',
            self.image_callback,
            10)
        self.get_logger().info("Subscribed to camera topic")

        # Path assoluto al modello usando share del pacchetto
        package_share = get_package_share_directory('pic4people_tracking')
        model_path = os.path.join(package_share, 'models', 'yolov8s-pose_openvino_int8_model', 'yolov8s-pose.xml')

        # OpenVINO inference
        core = Core()
        self.yolo_model = core.compile_model(model_path, device_name="CPU")
        self.get_logger().info(f"YOLO model loaded from {model_path}")
        print("YOLO model path:", model_path)

    def image_callback(self, msg: Image):
        self.get_logger().info(f"Got image: {msg.width}x{msg.height}, encoding={msg.encoding}")
        # convert to numpy array
        img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        self.get_logger().info(f"Numpy shape: {img_np.shape}, dtype: {img_np.dtype}")

def main(args=None):
    rclpy.init(args=args)
    node = ImageTester()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()