import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory("line_follower"), "config", "params.yaml")
    track = LaunchConfiguration("track")
    ctrl = LaunchConfiguration("controller")
    return LaunchDescription([
        DeclareLaunchArgument("track", default_value="technical"),
        DeclareLaunchArgument("controller", default_value="stanley_ff"),
        Node(package="line_follower", executable="camera_sim", parameters=[params, {"track": track}]),
        Node(package="line_follower", executable="line_perception", parameters=[params]),
        Node(package="line_follower", executable="line_control", parameters=[params, {"controller": ctrl}]),
    ])
