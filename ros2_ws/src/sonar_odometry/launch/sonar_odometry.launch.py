"""
Launch file for the Sonar Inertial Odometry node.

Usage:
    ros2 launch sonar_odometry sonar_odometry.launch.py
    ros2 launch sonar_odometry sonar_odometry.launch.py \
        config:=/path/to/my_params.yaml \
        initial_heading_deg:=45.0

Topic flow:
    raw_imu_topic (/ouster/imu)
        → imu_filter_madgwick_node
        → imu_topic (/imu/data)
        → sonar_odometry_node
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("sonar_odometry")
    default_config = PathJoinSubstitution([pkg_share, "config", "sonar_odometry.yaml"])

    return LaunchDescription([
        # ── Config file ────────────────────────────────────────────────────
        DeclareLaunchArgument(
            "config",
            default_value=default_config,
            description="Path to YAML parameter file",
        ),

        # ── Topic overrides ────────────────────────────────────────────────
        DeclareLaunchArgument(
            "raw_imu_topic",
            default_value="/ouster/imu",
            description="Raw IMU topic from the sensor (input to Madgwick filter)",
        ),
        DeclareLaunchArgument(
            "imu_topic",
            default_value="/imu/data",
            description="Filtered IMU topic (Madgwick output → sonar_odometry input)",
        ),
        DeclareLaunchArgument(
            "sonar_topic",
            default_value="/oculus/sonar_image",
            description="Sonar image topic (marine_acoustic_msgs/ProjectedSonarImage)",
        ),

        # ── Odometry ───────────────────────────────────────────────────────
        DeclareLaunchArgument(
            "initial_heading_deg",
            default_value="96.15",
            description="Initial vehicle heading in degrees (0=North, positive=East/CW)",
        ),
        DeclareLaunchArgument(
            "publish_tf",
            default_value="false",
            description="Broadcast odom→base_link TF",
        ),

        # ── Madgwick filter parameters ────────────────────────────────────
        DeclareLaunchArgument(
            "madgwick_gain",
            default_value="0.1",
            description="Madgwick beta gain — higher converges faster but is noisier (default: 0.1)",
        ),
        DeclareLaunchArgument(
            "madgwick_zeta",
            default_value="0.0",
            description="Madgwick gyro drift compensation gain — 0 disables it (default: 0.0)",
        ),
        DeclareLaunchArgument(
            "madgwick_world_frame",
            default_value="ned",
            description="World frame for Madgwick output: 'ned', 'enu', or 'nwu' (default: ned)",
        ),
        DeclareLaunchArgument(
            "madgwick_use_mag",
            default_value="false",
            description="Use magnetometer data in Madgwick filter (default: false)",
        ),

        # ── Nodes ──────────────────────────────────────────────────────────
        # NOTE: static transforms (imu → base_link, sonar → base_link) are
        # published by static_tf_mari_rov.launch.py — run that separately.
        Node(
            package="imu_filter_madgwick",
            executable="imu_filter_madgwick_node",
            name="imu_filter_madgwick_node",
            output="screen",
            parameters=[
                LaunchConfiguration("config"),
                {
                    "gain":        LaunchConfiguration("madgwick_gain"),
                    "zeta":        LaunchConfiguration("madgwick_zeta"),
                    "world_frame": LaunchConfiguration("madgwick_world_frame"),
                    "use_mag":     LaunchConfiguration("madgwick_use_mag"),
                },
            ],
            remappings=[
                ("imu/data_raw", LaunchConfiguration("raw_imu_topic")),
                ("imu/data",     LaunchConfiguration("imu_topic")),
            ],
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

        Node(
            package="sonar_odometry",
            executable="gps_path_node",
            name="gps_path_node",
            output="screen",
        ),
    ])
