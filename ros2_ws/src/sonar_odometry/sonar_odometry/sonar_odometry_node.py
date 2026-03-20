"""
ROS2 node: Sonar Inertial Odometry with EKF.

Subscriptions
-------------
  /imu/data          sensor_msgs/Imu   – IMU at ~100 Hz (gravity-compensated preferred)
  /oculus/sonar_image sensor_msgs/Image – sonar polar image

Publications
------------
  /odometry          nav_msgs/Odometry – fused position, heading, velocity

Frame conventions (matching the notebook):
  - NED (North-East-Down) navigation frame
  - FRD (Forward-Right-Down) body frame
  - nav_msgs/Odometry is published with:
      frame_id       = "odom"   (NED)
      child_frame_id = "base_link"  (FRD body)
      pose.position.x = North (m)
      pose.position.y = East  (m)
      pose.position.z = 0
      orientation     = quaternion from heading (yaw only)

IMU frame assumption:
  - linear_acceleration.x = forward  (gravity-compensated)
  - linear_acceleration.y = left     (gravity-compensated, ROS REP-103 convention)
  - angular_velocity.z    = yaw rate (up-positive, REP-103 convention)
  These are converted to FRD internally (ay = -msg.ay, gz = -msg.gz).

Sonar measurement:
  [d_forward, d_right, d_theta]  — body frame displacement between consecutive frames

Parameters (all settable via ROS params or a YAML config file)
----------
  imu_topic            : string  (default: "/imu/data")
  sonar_topic          : string  (default: "/oculus/sonar_image")
  odom_topic           : string  (default: "/odometry")
  initial_heading_deg  : float   (default: 0.0)
  accel_bias_x         : float   (default: 0.0)
  accel_bias_y         : float   (default: 0.0)
  gyro_bias_z          : float   (default: 0.002)
  sonar_scale_forward  : float   (default: 1.3)   – gamma in notebook
  sonar_scale_right    : float   (default: 1.3)
  sonar_scale_heading  : float   (default: 1.27)  – beta in notebook
  nis_threshold        : float   (default: 30.0)
  min_inliers          : int     (default: 6)
  min_inlier_ratio     : float   (default: 0.3)
  publish_tf           : bool    (default: false)
"""

from __future__ import annotations

import threading
import time
import numpy as np
import cv2
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TransformStamped, PoseStamped
from builtin_interfaces.msg import Time
from marine_acoustic_msgs.msg import ProjectedSonarImage

try:
    from tf2_ros import TransformBroadcaster
    TF2_AVAILABLE = True
except ImportError:
    TF2_AVAILABLE = False

from .ekf import ESEKF
from .image_processing import SonarImageProcessor
from .feature_matching import SonarFeatureMatcher


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def heading_to_quaternion(theta: float) -> tuple[float, float, float, float]:
    """
    Convert a yaw angle (heading, radians) to a quaternion (x, y, z, w).
    Rotation is around the Z-down axis so the vehicle nose points in the
    NED North direction when theta = 0.
    """
    half = theta / 2.0
    return 0.0, 0.0, np.sin(half), np.cos(half)


def ros_time_to_sec(stamp: Time) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def projected_sonar_to_cv2(msg: ProjectedSonarImage) -> np.ndarray | None:
    """Convert marine_acoustic_msgs/ProjectedSonarImage to a uint8 grayscale OpenCV image."""
    n_ranges = len(msg.ranges)
    n_beams  = msg.image.beam_count
    if n_ranges == 0 or n_beams == 0:
        return None

    dtype_map = {
        0: np.uint8,  1: np.int8,
        2: np.uint16, 3: np.int16,
        4: np.uint32, 5: np.int32,
        6: np.uint64, 7: np.int64,
        8: np.float32, 9: np.float64,
    }
    dtype = dtype_map.get(msg.image.dtype, np.uint8)
    data = np.frombuffer(bytes(msg.image.data), dtype=dtype).reshape(n_ranges, n_beams)

    # Normalise to uint8
    if dtype != np.uint8:
        d_min, d_max = data.min(), data.max()
        if d_max > d_min:
            data = ((data - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            data = np.zeros((n_ranges, n_beams), dtype=np.uint8)

    return data


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

class SonarOdometryNode(Node):

    def __init__(self):
        super().__init__("sonar_odometry_node")
        self._declare_parameters()
        self._load_parameters()
        self._init_ekf()
        self._init_processing()
        self._create_interfaces()
        self.get_logger().info("Sonar Odometry Node started.")

    # ---------------------------------------------------------------- params
    def _declare_parameters(self):
        self.declare_parameter("imu_topic",            "/imu/data")
        self.declare_parameter("sonar_topic",          "/oculus/sonar_image")
        self.declare_parameter("odom_topic",           "/odometry")
        self.declare_parameter("odom_frame_id",        "odom")
        self.declare_parameter("base_frame_id",        "base_link")
        self.declare_parameter("initial_heading_deg",  0.0)
        self.declare_parameter("accel_bias_x",         0.0)
        self.declare_parameter("accel_bias_y",         0.0)
        self.declare_parameter("gyro_bias_z",          0.002)
        self.declare_parameter("sonar_scale_forward",  1.3)
        self.declare_parameter("sonar_scale_right",    1.3)
        self.declare_parameter("sonar_scale_heading",  1.27)
        self.declare_parameter("nis_threshold",        30.0)
        self.declare_parameter("min_inliers",          6)
        self.declare_parameter("min_inlier_ratio",     0.3)
        self.declare_parameter("publish_tf",           False)
        # IMU noise (from IAM-20680HT datasheet defaults)
        self.declare_parameter("n_accel_ug_sqrthz",   135.0)
        self.declare_parameter("n_gyro_dps_sqrthz",   0.005)
        self.declare_parameter("sigma_sonar_pos",      0.023464)
        self.declare_parameter("sigma_sonar_heading",  0.023464)

    def _load_parameters(self):
        gp = self.get_parameter
        self.imu_topic       = gp("imu_topic").value
        self.sonar_topic     = gp("sonar_topic").value
        self.odom_topic      = gp("odom_topic").value
        self.odom_frame_id   = gp("odom_frame_id").value
        self.base_frame_id   = gp("base_frame_id").value
        self.initial_heading = np.deg2rad(gp("initial_heading_deg").value)
        self.accel_bias      = (gp("accel_bias_x").value, gp("accel_bias_y").value, 0.0)
        self.gyro_bias_z     = gp("gyro_bias_z").value
        self.scale_fwd       = gp("sonar_scale_forward").value
        self.scale_right     = gp("sonar_scale_right").value
        self.scale_heading   = gp("sonar_scale_heading").value
        self.nis_threshold   = gp("nis_threshold").value
        self.min_inliers     = gp("min_inliers").value
        self.min_inlier_ratio= gp("min_inlier_ratio").value
        self.publish_tf_flag = gp("publish_tf").value
        self.n_accel         = gp("n_accel_ug_sqrthz").value
        self.n_gyro          = gp("n_gyro_dps_sqrthz").value
        self.sigma_sonar_pos = gp("sigma_sonar_pos").value
        self.sigma_sonar_hdg = gp("sigma_sonar_heading").value

    # ----------------------------------------------------------------- init
    def _init_ekf(self):
        self.ekf = ESEKF(
            initial_heading_rad=self.initial_heading,
            accel_bias=self.accel_bias,
            gyro_bias_z=self.gyro_bias_z,
            n_accel_ug_sqrthz=self.n_accel,
            n_gyro_dps_sqrthz=self.n_gyro,
            sigma_sonar_pos=self.sigma_sonar_pos,
            sigma_sonar_heading=self.sigma_sonar_hdg,
        )
        self._lock = threading.Lock()
        self._prev_imu_time: float | None = None
        self._is_initialized: bool = False       # becomes True after first IMU
        self._sonar_count: int = 0

    def _init_processing(self):
        self._processor = SonarImageProcessor()
        self._matcher   = SonarFeatureMatcher()
        self._prev_sonar_img: np.ndarray | None = None
        self._prev_sonar_stamp: float | None = None

    def _create_interfaces(self):
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

        self._imu_sub = self.create_subscription(
            Imu, self.imu_topic, self._imu_callback, best_effort_qos
        )
        self._sonar_sub = self.create_subscription(
            ProjectedSonarImage, self.sonar_topic, self._sonar_callback, reliable_qos
        )
        self._odom_pub = self.create_publisher(Odometry, self.odom_topic, reliable_qos)

        self._path_pub = self.create_publisher(Path, self.odom_topic + "/path", reliable_qos)
        self._path_msg = Path()
        self._path_msg.header.frame_id = self.odom_frame_id

        self._gps_path_pub = self.create_publisher(Path, "/gps/path", reliable_qos)
        self._gps_path_msg = Path()
        self._gps_path_msg.header.frame_id = self.odom_frame_id
        self._gps_origin: tuple | None = None
        self._gps_sub = self.create_subscription(
            NavSatFix, "/fix", self._gps_callback, best_effort_qos
        )

        if self.publish_tf_flag and TF2_AVAILABLE:
            self._tf_broadcaster = TransformBroadcaster(self)
        else:
            self._tf_broadcaster = None

    # ---------------------------------------------------------- IMU callback
    def _imu_callback(self, msg: Imu) -> None:
        stamp = ros_time_to_sec(msg.header.stamp)

        with self._lock:
            if self._prev_imu_time is None:
                self._prev_imu_time = stamp
                self._is_initialized = True
                self.get_logger().info("IMU: first message received — EKF initialized.")
                return

            dt = stamp - self._prev_imu_time
            if dt <= 0.0:
                self.get_logger().warn(
                    f"IMU: non-positive dt={dt:.4f}s, skipping.", throttle_duration_sec=5.0
                )
                self._prev_imu_time = stamp
                return

            ax_frd =  msg.linear_acceleration.x
            ay_frd = -msg.linear_acceleration.y
            gz_frd = -msg.angular_velocity.z

            u = np.array([ax_frd, ay_frd, gz_frd])
            self.ekf.prediction(u, dt)
            self._prev_imu_time = stamp

        self._publish_odometry(msg.header.stamp, update_path=False)

    # -------------------------------------------------------- sonar callback
    def _sonar_callback(self, msg: ProjectedSonarImage) -> None:
        t0 = time.time()

        img = projected_sonar_to_cv2(msg)
        if img is None:
            self.get_logger().warn(
                "Sonar: failed to decode ProjectedSonarImage.", throttle_duration_sec=5.0
            )
            return

        stamp = ros_time_to_sec(msg.header.stamp)

        img_proc = self._processor.process_image(img)
        t_preproc = time.time()

        with self._lock:
            if not self._is_initialized:
                self.get_logger().warn(
                    "Sonar: waiting for first IMU message before processing.", throttle_duration_sec=5.0
                )
                self._prev_sonar_img   = img_proc
                self._prev_sonar_stamp = stamp
                return

            if self._prev_sonar_img is None:
                self._prev_sonar_img   = img_proc
                self._prev_sonar_stamp = stamp
                self.ekf.prev_px    = self.ekf.px
                self.ekf.prev_py    = self.ekf.py
                self.ekf.prev_theta = self.ekf.theta
                self.get_logger().info("Sonar: first frame stored — ready for matching.")
                return

            result, kp1, kp2, matches = self._matcher.process_sonar_image_pair(
                self._prev_sonar_img, img_proc
            )
            t_match = time.time()

            self.get_logger().info(
                f"Sonar timing — preproc: {(t_preproc-t0)*1000:.1f}ms  "
                f"matching: {(t_match-t_preproc)*1000:.1f}ms  "
                f"total: {(t_match-t0)*1000:.1f}ms",
                throttle_duration_sec=2.0
            )
            self.get_logger().info(
                f"Sonar features — kp1={len(kp1)}  kp2={len(kp2)}  "
                f"matches={len(matches)}  inliers={result['num_inliers']}  "
                f"inlier_ratio={result['inlier_ratio']:.2f}",
                throttle_duration_sec=2.0
            )

            self._prev_sonar_img   = img_proc
            self._prev_sonar_stamp = stamp

            z_sonar = self._sonar_result_to_measurement(result)
            if z_sonar is None:
                self.get_logger().warn(
                    f"Sonar: rejected (inliers={result['num_inliers']}, "
                    f"ratio={result['inlier_ratio']:.2f}) — no EKF update.",
                    throttle_duration_sec=2.0
                )
                return

            self.get_logger().info(
                f"Sonar update #{self._sonar_count}: "
                f"d_fwd={z_sonar[0]:.3f}m  d_right={z_sonar[1]:.3f}m  "
                f"dth={np.rad2deg(z_sonar[2]):.2f}°"
            )

            innovation, S = self.ekf.update(z_sonar, nis_threshold=self.nis_threshold)
            self._sonar_count += 1

            if innovation is not None and S is not None:
                nis = float(innovation.T @ np.linalg.inv(S) @ innovation)
                self.get_logger().info(f"EKF NIS={nis:.2f}  pos=({self.ekf.px:.3f}, {self.ekf.py:.3f})m  hdg={np.rad2deg(self.ekf.theta):.1f}°")

        self._publish_odometry(msg.header.stamp, update_path=True)

    # -------------------------------------------------- measurement builder
    def _sonar_result_to_measurement(self, result: dict) -> np.ndarray | None:
        T = result["transformation"]
        if T is None:
            return None
        if result["num_inliers"] < self.min_inliers:
            return None
        if result["inlier_ratio"] < self.min_inlier_ratio:
            return None

        dp = T[0:2, 2].copy()
        dR  = T[0:2, 0:2]
        dtheta = np.arctan2(dR[1, 0], dR[0, 0])

        # Apply empirical scale calibration (matching notebook)
        dp[0] *= self.scale_fwd
        dp[1] *= self.scale_right
        dtheta *= self.scale_heading

        # Notebook convention: z = [forward, -right_raw, dtheta]
        # dp[0] = forward displacement, dp[1] = lateral (right in body frame after inversion)
        z = np.array([dp[0], -dp[1], dtheta])
        return z

    # ----------------------------------------------------------- GPS callback
    _EARTH_R = 6_371_000.0

    def _gps_callback(self, msg: NavSatFix) -> None:
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
        pose.header.frame_id = self.odom_frame_id
        pose.pose.position.x = north
        pose.pose.position.y = east
        pose.pose.orientation.w = 1.0
        self._gps_path_msg.header.stamp = msg.header.stamp
        self._gps_path_msg.poses.append(pose)
        self._gps_path_pub.publish(self._gps_path_msg)

    # ----------------------------------------------------- publish odometry
    def _publish_odometry(self, stamp: Time, update_path: bool = False) -> None:
        with self._lock:
            px, py    = self.ekf.px, self.ekf.py
            vx, vy    = self.ekf.vx, self.ekf.vy
            theta     = self.ekf.theta
            P         = self.ekf.covariance

        msg = Odometry()
        msg.header.stamp    = stamp
        msg.header.frame_id = self.odom_frame_id
        msg.child_frame_id  = self.base_frame_id

        msg.pose.pose.position.x = px
        msg.pose.pose.position.y = py
        msg.pose.pose.position.z = 0.0

        qx, qy, qz, qw = heading_to_quaternion(theta)
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        # 6×6 pose covariance (x, y, z, roll, pitch, yaw)
        cov6 = np.zeros((6, 6))
        cov6[0, 0] = P[0, 0]  # px-px
        cov6[0, 1] = P[0, 1]  # px-py
        cov6[1, 0] = P[1, 0]
        cov6[1, 1] = P[1, 1]  # py-py
        cov6[5, 5] = P[4, 4]  # theta-theta
        cov6[0, 5] = P[0, 4]
        cov6[5, 0] = P[4, 0]
        cov6[1, 5] = P[1, 4]
        cov6[5, 1] = P[4, 1]
        msg.pose.covariance = cov6.flatten().tolist()

        # Twist (body-frame velocities)
        c, s = np.cos(theta), np.sin(theta)
        vx_body =  c * vx + s * vy
        vy_body = -s * vx + c * vy
        msg.twist.twist.linear.x = vx_body
        msg.twist.twist.linear.y = vy_body
        msg.twist.twist.linear.z = 0.0

        cov6t = np.zeros((6, 6))
        cov6t[0, 0] = P[2, 2]
        cov6t[0, 1] = P[2, 3]
        cov6t[1, 0] = P[3, 2]
        cov6t[1, 1] = P[3, 3]
        msg.twist.covariance = cov6t.flatten().tolist()

        self._odom_pub.publish(msg)

        if update_path:
            pose_stamped = PoseStamped()
            pose_stamped.header = msg.header
            pose_stamped.pose = msg.pose.pose
            self._path_msg.header.stamp = stamp
            self._path_msg.poses.append(pose_stamped)
            self._path_pub.publish(self._path_msg)

        if self._tf_broadcaster is not None:
            tf = TransformStamped()
            tf.header.stamp    = stamp
            tf.header.frame_id = self.odom_frame_id
            tf.child_frame_id  = self.base_frame_id
            tf.transform.translation.x = px
            tf.transform.translation.y = py
            tf.transform.translation.z = 0.0
            tf.transform.rotation.x = qx
            tf.transform.rotation.y = qy
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw
            self._tf_broadcaster.sendTransform(tf)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = SonarOdometryNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
