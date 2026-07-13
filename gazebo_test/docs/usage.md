# Usage Instructions

This document provides instructions on how to use the `gazebo_test` package for testing Gazebo simulations in ROS2.

## Running Tests (Basic Usage)

To run the tests, follow these steps:

1. **Launch the Gazebo Simulation**: Use the provided launch files to start the Gazebo simulation environment. For example:

   ```bash
   ros2 launch gazebo_sim simulation.launch.py
   ```

2. **Launch the Navigation Stack**: Start the navigation stack that you want to test. This can be done using a separate launch file:

   ```bash
   ros2 launch gazebo_test bringup.launch.py
   ```

3. **Run the Tests**: Execute the test cases using experiment manager. The test cases are designed to measure the performance of the navigation stack in various scenarios. You can run the tests using the following command:

   ```bash
   ros2 run gazebo_test experiment_manager 
   ```

4. **Rviz Visualization**: If you want to visualize the results in RViz, you can run the following command:

TODO: Add the command to run RViz with the appropriate configuration file.

   ```bash
   ros2 launch gazebo_test rviz.launch.py
   ```

## Running Tests (Full Modality)

To run the full test modality, you need to have `tmux` installed. The full test modality allows you to all the previous scripts in a tmux session. To run the full test modality run the following command:

```bash
ros2 run gazebo_test full_experiment
```

This will run all the previous scripts in a tmux session.

If you want to see the running scripts in the tmux session, you can attach to the tmux session using the following command:

```bash
tmux attach -t gazebo_test
```
