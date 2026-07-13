import os
import rclpy
from rclpy.node import Node
from people_msgs.msg import People, Person
import csv

import rclpy.time

class DataRecorder(Node):
    def __init__(self):
        super().__init__('export_to_csv_node')

        self.declare_parameter('filepath', '') # filepath = rosbag_filename
        self.declare_parameter('tracker', 'sort')
        self.declare_parameter('tracked_people_topic', '/tracked_people')
        self.declare_parameter('vicon_people_topic', '/vicon_people')

        filepath = self.get_parameter('filepath').value
        tracker = self.get_parameter('tracker').value
        tracked_people_topic = self.get_parameter('tracked_people_topic').value
        vicon_people_topic = self.get_parameter('vicon_people_topic').value

        results_path = os.path.join(os.path.dirname(filepath), 'results')
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        
        self.csv_file = open(os.path.join(os.path.dirname(filepath),f'results/results_{tracker}.csv'), 'w')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['stamp_sec','stamp_nanosec', 'source', 'id', 'x', 'y', 'yaw', 'vx', 'vy', 'vz', 'reliability'])

        self.get_logger().info(f"\nRecording csv file at location: {self.csv_file}\n")

        self.subscriber_vicon = self.create_subscription(People, vicon_people_topic, self.callback_vicon, 10)
        self.subscriber_tracker = self.create_subscription(People, tracked_people_topic, self.callback_tracker, 10)

    def callback_vicon(self, msg: People):
        stamp_sec = msg.header.stamp.sec
        stamp_nanosec = msg.header.stamp.nanosec
        source = 'vicon'
        for person in msg.people:
                id = person.tags[person.tagnames.index('id')]
                self.csv_writer.writerow([stamp_sec, stamp_nanosec, source, id, 
                                          person.position.x, person.position.y, person.position.z, 
                                          person.velocity.x, person.velocity.y, person.velocity.z,
                                          person.reliability])

    def callback_tracker(self, msg: People):
        stamp_sec = msg.header.stamp.sec
        stamp_nanosec = msg.header.stamp.nanosec
        source = 'tracker'
        for person in msg.people:
                id = int(person.name)
                self.csv_writer.writerow([stamp_sec, stamp_nanosec, source, id, 
                                          person.position.x, person.position.y, person.position.z, 
                                          person.velocity.x, person.velocity.y, person.velocity.z,
                                          person.reliability])
        
    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    recorder = DataRecorder()
    
    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
         recorder.get_logger().info("Shutting down")
    except Exception as e:
         print(f"Exception caught: {e}")
    finally:
        recorder.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
