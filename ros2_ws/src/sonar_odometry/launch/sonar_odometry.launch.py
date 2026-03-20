"""
Launch file for the Sonar Inertial Odometry node.

Usage:
    ros2 launch sonar_odometry sonar_odometry.launch.py
    ros2 launch sonar_odometry sonar_odometry.launch.py \
        config:=/path/to/my_params.yaml \
        initial_heading_deg:=45.0
"""

from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("sonar_odometry")
    default_config = PathJoinSubstitution([pkg_share, "config", "sonar_odometry.yaml"])

    return LaunchDescription([
        DeclareLaunchArgument(
            "config",
            default_value=default_config,
            description="Path to YAML parameter file",
        ),
        DeclareLaunchArgument(
            "initial_heading_deg",
            default_value="96.15",
            description="Initial vehicle heading in degrees (0=North, positive=East/CW)",
        ),
        DeclareLaunchArgument(
            "imu_topic",
            default_value="/ouster/imu",
            description="IMU topic (sensor_msgs/Imu, gravity-compensated preferred)",
        ),
        DeclareLaunchArgument(
            "sonar_topic",

            default_value="/oculus/sonar_image",
            description="Sonar image topic (sensor_msgs/Image)",
        ),
        DeclareLaunchArgument(
            "publish_tf",
            default_value="false",
            description="Broadcast odom→base_link TF",
        ),

        Node(
            package="sonar_odometry",
            executable="sonar_odometry_node",
            name="sonar_odometry_node",
            output="screen",
            parameters=[
                LaunchConfiguration("config"),
                {
                    "initial_heading_deg": LaunchConfiguration("initial_heading_deg"),
                    "imu_topic":           LaunchConfiguration("imu_topic"),
                    "sonar_topic":         LaunchConfiguration("sonar_topic"),
                    "publish_tf":          LaunchConfiguration("publish_tf"),
                },
            ],
        ),
    ])
