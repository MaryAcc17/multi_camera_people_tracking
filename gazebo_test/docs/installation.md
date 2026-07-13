# Installation Instructions

Follow the steps below to install and set up the `gazebo_test` package:

## Prerequisites

Ensure you have the following installed on your system:

- [ROS 2](https://docs.ros.org/en/rolling/Installation.html) (Humble or later)
- [Gazebo](https://gazebosim.org/) (Fortress or later)
- `colcon` build tool
- Python 3.8 or later

## Clone the Repository

```bash
cd ~/workspaces/hunavsim_ws/src
git clone https://github.com/your-repo/gazebo_test.git
```

## Download required repositories

```bash
cd ~/<your_workspace>/src
vcs import < src/gazebo_test/test.repos .
```
This command will download all the required repositories specified in the `test.repos` file.
Make sure to replace `<your_workspace>` with the path to your workspace.

## Install Dependencies

Navigate to the workspace root and install dependencies:

```bash
cd ~/workspaces/hunavsim_ws
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

## Set Up Environment Variables

Set up the environment variables for your workspace:

```bash
export JACKAL_LASER=1
export JACKAL_WHEEL_SLIP=0
```

## Change the publisher of the odom tf

In the `src/jackal/jackal_simulator/jackal_gazebo/config/localization.yaml` file, change the ```publish_tf: true``` to ```publish_tf: false```. This is necessary to avoid conflicts with the EKF node.

```yaml
ekf_node:
  ros__parameters:
    use_sim_time: true
    odom_frame: odom
    base_link_frame: base_link
    world_frame: odom
    publish_tf: false <-- Change this line
    print_diagnostics: true
    two_d_mode: true
...

```

## Build the Package

Build the workspace using `colcon`:

```bash
colcon build --symlink-install --packages-select gazebo_test
```

## Source the Workspace

Source the setup file to overlay the workspace:

```bash
source ~/workspaces/hunavsim_ws/install/setup.bash
```

## Full test modality installation

If you want to install the full test modality, you need to install the `tmux` package and `libtmux` . This package is used to run the tests in a tmux session. To install it, run the following command:

```bash
sudo apt-get install tmux
pip install --upgrade libtmux
```

## Mouse support for tmux

If you want to use the mouse support for tmux, you need to add the following line to your `~/.tmux.conf` file:

```bash
set -g mouse on
```

This will enable the mouse support for tmux. You can also use the `Ctrl + b` and `m` key combination to toggle the mouse support on and off.
The keybinding `Ctrl + b` is the default prefix key for tmux. You can change it to any other key combination you want.
This may not be working in some systems for example in VSCode.
