#!/usr/bin/env python3
"""
Launch static_tf_publisher with a configurable transforms file for the MaRI ROV.

Usage examples:
  # Use the default transforms file provided in the workspace
  ros2 launch <package> static_tf_mari_rov.launch.py

  # Override the transforms file at launch time
  ros2 launch <package> static_tf_mari_rov.launch.py transforms_path:=/path/to/transforms.yaml

Replace <package> with the package name that contains/installs the `static_tf_publisher_node`
(for example, `static_tf_publisher_ros2` if that's the package name in your workspace).
"""

import os

from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Default transforms file (relative to repo root)
    repo_root = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
    default_transforms = os.path.join(
        repo_root,
        "config",
        "transforms.yaml",
    )

    transforms_path = LaunchConfiguration("transforms_path")

    declare_transforms_arg = DeclareLaunchArgument(
        "transforms_path",
        default_value=default_transforms,
        description="Path to transforms yaml file to load",
    )

    node = Node(
        package="tf_static_publisher",
        executable="tf_static_publisher",
        name="tf_static_publisher",
        output="screen",
        # Provide both a parameter and a YAML file path so the node can accept either style
        parameters=[{"transforms_path": transforms_path}],
    )

    return LaunchDescription(
        [
            declare_transforms_arg,
            node,
        ]
    )
