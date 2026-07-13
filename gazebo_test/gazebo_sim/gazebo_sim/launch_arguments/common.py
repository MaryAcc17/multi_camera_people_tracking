from launch.actions import DeclareLaunchArgument
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class GazeboCommonArgs:
    """This class contains a collection of frequently used LaunchArguments."""

    use_sim_time: DeclareLaunchArgument = DeclareLaunchArgument(
        name="use_sim_time",
        default_value="True",
        choices=["True", "False", "true", "false"],
        description="Use simulation time.",
    )
    namespace: DeclareLaunchArgument = DeclareLaunchArgument(
        name="namespace", default_value="", description="Define namespace of the robot."
    )
    headless: DeclareLaunchArgument = DeclareLaunchArgument(
        name="headless",
        default_value="false",
        choices=["true", "false", "True", "False"],
        description="Run Gazebo in headless mode.",
    )
    robot_name: DeclareLaunchArgument = DeclareLaunchArgument(
        "robot_name",
        default_value="jackal",
        description="Name of the robot. ",
        choices=[
            "pmb2",
            "jackal",
            "turtlebot3",
            "turtlebot3_waffle",
            "ghost_robot",
        ],
    )
    world_name: DeclareLaunchArgument = DeclareLaunchArgument(
        name="world_name",
        default_value="social_nav.world",
        description="Specify world name, will be converted to full path.",
    )
    x: DeclareLaunchArgument = DeclareLaunchArgument(
        name="x", description="X pose of the robot", default_value="0.0"
    )
    y: DeclareLaunchArgument = DeclareLaunchArgument(
        name="y", description="Y pose of the robot", default_value="0.0"
    )
    z: DeclareLaunchArgument = DeclareLaunchArgument(
        name="z", description="Z pose of the robot", default_value="0.0"
    )
    roll: DeclareLaunchArgument = DeclareLaunchArgument(
        name="roll", description="Roll pose of the robot", default_value="0.0"
    )
    pitch: DeclareLaunchArgument = DeclareLaunchArgument(
        name="pitch", description="Pitch pose of the robot", default_value="0.0"
    )
    yaw: DeclareLaunchArgument = DeclareLaunchArgument(
        name="yaw", description="Yaw pose of the robot", default_value="0.0"
    )
    goal_x: DeclareLaunchArgument = DeclareLaunchArgument(
        name="goal_x",
        description="X pose of the goal",
        default_value="0.0",
    )
    goal_y: DeclareLaunchArgument = DeclareLaunchArgument(
        name="goal_y",
        description="Y pose of the goal",
        default_value="0.0",
    )

    start_hunav: DeclareLaunchArgument = DeclareLaunchArgument(
        name="start_hunav",
        default_value="true",
        choices=["true", "false", "True", "False"],
        description="Whether to start the HuNav plugin at Gazebo launch or not.",
    )
