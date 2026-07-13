## Gazebo Simulation Launch workflow

The Gazebo bring-up and the robot spawn/controllers are now split into two launch files:

1. Launch the HuNav world generation and Gazebo server/client (robot is spawned automatically by default so that the HuNav Gazebo plugin can find it right away):

   ```bash
   ros2 launch gazebo_sim simulation.launch.py
   ```

   Pass `spawn_robot:=false` only if you really want to start the simulation without the robot; in that case, expect HuNav plugin warnings until you run the spawn launch.

2. Spawn the robot and bring up its controllers once the simulation is running:

   ```bash
   ros2 launch gazebo_sim spawn_robot.launch.py
   ```

   Use `robot_name`, `x`, `y`, `yaw`, `use_gazebo_controllers`, etc. to customize the spawn without relaunching the simulation.
