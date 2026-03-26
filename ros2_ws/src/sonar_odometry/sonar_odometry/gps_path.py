"""
ROS2 node: GPS Path Publisher.

Subscribes to NavSatFix messages, converts lat/lon to a local NED/ENU frame,
and publishes a nav_msgs/Path for visualisation.

The first valid fix is used as the origin.  Subsequent fixes are expressed as
(east, north) offsets using a flat-Earth approximation (adequate for short
baselines, < ~10 km).

Subscriptions
-------------
  /fix   sensor_msgs/NavSatFix   – GPS fixes

Publications
------------
  /gps/path   nav_msgs/Path   – accumulated 2-D ground-truth path

Parameters
----------
  fix_topic      : string  (default: "/fix")
  path_topic     : string  (default: "/gps/path")
  frame_id       : string  (default: "odom")
"""

from __future__ import annotations

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class GpsPathNode(Node):

    _EARTH_R = 6_371_000.0

    def __init__(self):
        super().__init__("gps_path_node")

        self.declare_parameter("fix_topic",  "/fix")
        self.declare_parameter("path_topic", "/gps/path")
        self.declare_parameter("frame_id",   "odom")

        self._fix_topic  = self.get_parameter("fix_topic").value
        self._path_topic = self.get_parameter("path_topic").value
        self._frame_id   = self.get_parameter("frame_id").value

        self._gps_origin: tuple[float, float] | None = None

        self._path_msg = Path()
        self._path_msg.header.frame_id = self._frame_id

        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._fix_sub = self.create_subscription(
            NavSatFix, self._fix_topic, self._fix_callback, best_effort_qos
        )
        self._path_pub = self.create_publisher(Path, self._path_topic, reliable_qos)

        self.get_logger().info(
            f"GPS Path Node started — listening on '{self._fix_topic}', "
            f"publishing to '{self._path_topic}'."
        )

    def _fix_callback(self, msg: NavSatFix) -> None:
        if msg.status.status < 0:
            return

        lat, lon = msg.latitude, msg.longitude

        if self._gps_origin is None:
            self._gps_origin = (lat, lon)
            self.get_logger().info(f"GPS origin set: lat={lat:.7f}  lon={lon:.7f}")

        lat0, lon0 = self._gps_origin
        north = np.deg2rad(lat - lat0) * self._EARTH_R
        east  = np.deg2rad(lon - lon0) * self._EARTH_R * np.cos(np.deg2rad(lat0))

        pose = PoseStamped()
        pose.header.stamp    = msg.header.stamp
        pose.header.frame_id = self._frame_id
        pose.pose.position.x = east   # ENU: x = East  (matches EKF px)
        pose.pose.position.y = north  # ENU: y = North (matches EKF py)
        pose.pose.orientation.w = 1.0

        self._path_msg.header.stamp = msg.header.stamp
        self._path_msg.poses.append(pose)
        self._path_pub.publish(self._path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GpsPathNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
