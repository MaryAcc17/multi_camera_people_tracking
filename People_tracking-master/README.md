> Project: "People tracking project"

> Owner: "PIC4SeR, Andrea Eirale, Pietro Vignini" 

> Date: "05/2025" 

---

# People Tracking

## Description of the project
Ros 2 package exploiting Yolo-pose and StrongSORT for tracking and re-identification of people. The output provides information about people position and velocities.

## Installation procedure
Import the package in your workspace and build with colcon.

## User Guide
The package requires the models stored on PIC4SeRNAS/Common/people_4Dpose_track. 
The main node is pic4people_tracking/tracker.py, the main launch is pic4people_tracking/launch/tracker_launch.py.
The main configuration file is pic4people_tracking/params/params.yaml, where you can change topics, reference frames, image metadata, and models.

To launch the tracking with camera use:

```
ros2 launch pic4people_tracking tracker_launch.py
```

otherwise, you can use a bag with:


```
ros2 launch pic4people_tracking tracker_launch.py rosbag_filename:='/path/to/bag.db3'
```
