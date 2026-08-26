import os

import cv2
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


OUTPUT_DIR = "/tmp/camera_frames"

TOPICS = {
    "camera_0": "/jackal/sensors/camera_0/color/image_raw",
    "camera_1": "/jackal/sensors/camera_1/color/image_raw",
    "camera_2": "/jackal/sensors/camera_2/color/image_raw",
}


class FrameExtractor(Node):

    def __init__(self):
        super().__init__("frame_extractor")

        self.bridge = CvBridge()

        self.saved = {
            "camera_0": False,
            "camera_1": False,
            "camera_2": False,
        }

        # Keep references to subscriptions
        self.image_subscriptions = []

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Remove old images if present
        for camera_name in TOPICS:
            image_path = os.path.join(
                OUTPUT_DIR,
                f"{camera_name}.png",
            )

            if os.path.exists(image_path):
                os.remove(image_path)

        # Subscribe to the three RGB topics
        for camera_name, topic in TOPICS.items():

            subscription = self.create_subscription(
                Image,
                topic,
                lambda msg, cam=camera_name: self.image_callback(msg, cam),
                qos_profile_sensor_data,
            )

            self.image_subscriptions.append(subscription)

            self.get_logger().info(
                f"Subscribed to {topic}"
            )

        self.get_logger().info(
            "Waiting for one RGB frame from each camera..."
        )

    def image_callback(self, msg, camera_name):

        if self.saved[camera_name]:
            return

        try:
            image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                f"{camera_name}.png",
            )

            success = cv2.imwrite(
                output_path,
                image,
            )

            if not success:
                self.get_logger().error(
                    f"Unable to save image for {camera_name}"
                )
                return

            self.saved[camera_name] = True

            self.get_logger().info(
                f"Saved {camera_name} -> {output_path}"
            )

            if all(self.saved.values()):

                self.get_logger().info(
                    "All three camera frames saved."
                )

                self.get_logger().info(
                    f"Images available in: {OUTPUT_DIR}"
                )

                rclpy.shutdown()

        except Exception as exc:
            self.get_logger().error(
                f"Error processing {camera_name}: {exc}"
            )


def main(args=None):

    rclpy.init(args=args)

    node = FrameExtractor()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()