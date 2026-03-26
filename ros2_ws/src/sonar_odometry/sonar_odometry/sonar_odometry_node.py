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
      pose.position.x = East  (m)   (EKF px = East)
      pose.position.y = North (m)   (EKF py = North)
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

import os
import threading

# Default debug output dir: ros2_ws/debug
# ament_index gives us install/<pkg>/share/<pkg>; four levels up is the workspace root.
from ament_index_python.packages import get_package_share_directory as _get_share
_WS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    _get_share("sonar_odometry")
))))
_DEFAULT_DEBUG_DIR = os.path.join(_WS_ROOT, "debug")
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
    from tf2_ros import TransformBroadcaster, Buffer, TransformListener
    TF2_AVAILABLE = True
except ImportError:
    TF2_AVAILABLE = False

from .ekf import ESEKF
from .image_processing import SonarImageProcessor
from .feature_matching import SonarFeatureMatcher


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a quaternion to a 3×3 rotation matrix."""
    n = qx*qx + qy*qy + qz*qz + qw*qw
    if n < 1e-10:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s*qw*qx, s*qw*qy, s*qw*qz
    xx, xy, xz = s*qx*qx, s*qx*qy, s*qx*qz
    yy, yz, zz = s*qy*qy, s*qy*qz, s*qz*qz
    return np.array([
        [1.0-(yy+zz),    xy-wz,       xz+wy    ],
        [   xy+wz,    1.0-(xx+zz),    yz-wx    ],
        [   xz-wy,       yz+wx,    1.0-(xx+yy)],
    ])


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
        self.declare_parameter("lowe_ratio",           0.50)
        self.declare_parameter("min_inliers",          6)
        self.declare_parameter("min_inlier_ratio",     0.3)
        self.declare_parameter("publish_tf",           False)
        self.declare_parameter("imu_frame_id",         "imu_link")
        self.declare_parameter("sonar_frame_id",       "sonar_link")
        self.declare_parameter("debug_image_dir",      _DEFAULT_DEBUG_DIR)
        self.declare_parameter("debug_save_n_images",  20)
        # IMU noise (from IAM-20680HT datasheet defaults)
        self.declare_parameter("n_accel_ug_sqrthz",       135.0)
        self.declare_parameter("n_gyro_dps_sqrthz",       0.005)
        self.declare_parameter("sigma_sonar_pos",          0.023464)
        self.declare_parameter("sigma_sonar_heading",      0.023464)
        self.declare_parameter("bilateral_d",              9)
        self.declare_parameter("bilateral_sigma_color",    75.0)
        self.declare_parameter("bilateral_sigma_space",    75.0)
        self.declare_parameter("clahe_clip_limit",         3.0)
        self.declare_parameter("clahe_tile_grid_size",     8)
        self.declare_parameter("use_gps_initial_heading",  True)
        self.declare_parameter("gps_heading_min_dist_m",   2.0)

    def _load_parameters(self):
        gp = self.get_parameter
        self.imu_topic       = gp("imu_topic").value
        self.sonar_topic     = gp("sonar_topic").value
        self.odom_topic      = gp("odom_topic").value
        self.odom_frame_id   = gp("odom_frame_id").value
        self.base_frame_id   = gp("base_frame_id").value
        # Convert compass heading (CW from North) to ENU yaw (CCW from East)
        self.initial_heading = np.pi / 2.0 - np.deg2rad(gp("initial_heading_deg").value)
        self.accel_bias      = (gp("accel_bias_x").value, gp("accel_bias_y").value, 0.0)
        self.gyro_bias_z     = gp("gyro_bias_z").value
        self.scale_fwd       = gp("sonar_scale_forward").value
        self.scale_right     = gp("sonar_scale_right").value
        self.scale_heading   = gp("sonar_scale_heading").value
        self.nis_threshold   = gp("nis_threshold").value
        self.lowe_ratio      = gp("lowe_ratio").value
        self.min_inliers     = gp("min_inliers").value
        self.min_inlier_ratio= gp("min_inlier_ratio").value
        self.publish_tf_flag = gp("publish_tf").value
        self.imu_frame_id      = gp("imu_frame_id").value
        self.sonar_frame_id    = gp("sonar_frame_id").value
        self.debug_image_dir   = gp("debug_image_dir").value
        self.debug_save_n      = gp("debug_save_n_images").value
        self.n_accel         = gp("n_accel_ug_sqrthz").value
        self.n_gyro          = gp("n_gyro_dps_sqrthz").value
        self.sigma_sonar_pos = gp("sigma_sonar_pos").value
        self.sigma_sonar_hdg = gp("sigma_sonar_heading").value
        self.use_gps_initial_heading  = gp("use_gps_initial_heading").value
        self.gps_heading_min_dist     = gp("gps_heading_min_dist_m").value
        tile = gp("clahe_tile_grid_size").value
        self._img_proc_config = {
            "bilateral_d":          gp("bilateral_d").value,
            "bilateral_sigma_color":gp("bilateral_sigma_color").value,
            "bilateral_sigma_space":gp("bilateral_sigma_space").value,
            "clahe_clip_limit":     gp("clahe_clip_limit").value,
            "clahe_tile_grid":      (tile, tile),
        }

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
        # Cached rotation matrices from TF (looked up once on first use)
        self._R_imu_to_body:   np.ndarray | None = None
        self._R_sonar_to_body: np.ndarray | None = None

    def _init_processing(self):
        self._processor = SonarImageProcessor()
        self._processor.config.update(self._img_proc_config)
        self._matcher   = SonarFeatureMatcher(lowe_ratio=self.lowe_ratio)
        self._prev_sonar_img: np.ndarray | None = None
        self._prev_sonar_stamp: float | None = None
        self._debug_img_count: int = 0

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

        # GPS heading initialisation (runs once then stops updating)
        self._gps_first_fix:   tuple | None = None
        self._gps_heading_set: bool         = False
        if self.use_gps_initial_heading:
            self._gps_heading_sub = self.create_subscription(
                NavSatFix, "/fix", self._gps_heading_callback, best_effort_qos
            )
            self.get_logger().info(
                f"GPS heading init enabled — waiting for {self.gps_heading_min_dist:.1f} m displacement."
            )
        else:
            self._gps_heading_sub = None

        if TF2_AVAILABLE:
            self._tf_buffer   = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
        else:
            self._tf_buffer   = None
            self._tf_listener = None
            self.get_logger().error("tf2_ros not available — frame transforms will not work.")

        if self.publish_tf_flag and TF2_AVAILABLE:
            self._tf_broadcaster = TransformBroadcaster(self)
        else:
            self._tf_broadcaster = None

    # -------------------------------------------------- TF rotation lookup
    def _lookup_rotation(self, source_frame: str, target_frame: str) -> np.ndarray | None:
        """
        Look up the static TF from source_frame to target_frame and return the
        3×3 rotation matrix.  Returns None if the transform is not yet available.
        """
        if self._tf_buffer is None:
            return None
        try:
            t = self._tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time()
            )
            q = t.transform.rotation
            R = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
            self.get_logger().info(
                f"TF: cached rotation {source_frame} → {target_frame}\n{R}"
            )
            return R
        except Exception as e:
            self.get_logger().warn(
                f"TF lookup {source_frame} → {target_frame} failed: {e}",
                throttle_duration_sec=5.0,
            )
            return None

    # ------------------------------------------- GPS heading initialisation
    _EARTH_R = 6_371_000.0

    def _gps_heading_callback(self, msg: NavSatFix) -> None:
        """
        Runs until the vehicle has moved gps_heading_min_dist metres, then
        computes the GPS bearing, resets ekf.theta, and stops listening.

        The bearing is derived from the displacement vector (east, north) so
        it maps directly to an ENU yaw without any intermediate conversion.
        """
        if self._gps_heading_set or msg.status.status < 0:
            return

        lat, lon = msg.latitude, msg.longitude

        if self._gps_first_fix is None:
            self._gps_first_fix = (lat, lon)
            return

        lat0, lon0 = self._gps_first_fix
        north = np.deg2rad(lat - lat0) * self._EARTH_R
        east  = np.deg2rad(lon - lon0) * self._EARTH_R * np.cos(np.deg2rad(lat0))
        dist  = np.hypot(north, east)

        if dist < self.gps_heading_min_dist:
            return  # not enough displacement yet for a reliable bearing

        # atan2(north, east) is the ENU yaw directly (CCW from East)
        heading_enu = np.arctan2(north, east)

        with self._lock:
            self.ekf.theta = heading_enu

        self._gps_heading_set = True
        self.get_logger().info(
            f"GPS heading initialised: {np.rad2deg(heading_enu):.2f}° ENU  "
            f"({90.0 - np.rad2deg(heading_enu):.2f}° compass)  "
            f"from {dist:.2f} m GPS displacement"
        )

    # ---------------------------------------------------------- IMU callback
    def _imu_callback(self, msg: Imu) -> None:
        stamp = ros_time_to_sec(msg.header.stamp)

        with self._lock:
            # Lazy TF lookup — cached after first success
            if self._R_imu_to_body is None:
                self._R_imu_to_body = self._lookup_rotation(
                    self.imu_frame_id, self.base_frame_id
                )
                if self._R_imu_to_body is None:
                    return  # TF not ready yet

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

            # Rotate from IMU sensor frame to FRD body frame via TF
            a_imu = np.array([msg.linear_acceleration.x,
                               msg.linear_acceleration.y,
                               msg.linear_acceleration.z])
            w_imu = np.array([msg.angular_velocity.x,
                               msg.angular_velocity.y,
                               msg.angular_velocity.z])
            a_body = self._R_imu_to_body @ a_imu
            w_body = self._R_imu_to_body @ w_imu

            # EKF expects [ax_frd, ay_frd, gz_frd]:
            #   ax_frd = forward accel  (body X)
            #   ay_frd = rightward accel (body Y)
            #   gz_frd = yaw rate, CW positive (body Z)
            u = np.array([a_body[0], a_body[1], w_body[2]])
            self.ekf.prediction(u, dt)
            self._prev_imu_time = stamp

        self._publish_odometry(msg.header.stamp, update_path=False)

    # -------------------------------------------------------- sonar callback
    def _sonar_callback(self, msg: ProjectedSonarImage) -> None:
        t0 = time.time()

        # Populate sonar geometry from this message (no-op after first call)
        self._matcher.update_sonar_params(msg)

        img = projected_sonar_to_cv2(msg)
        if img is None:
            self.get_logger().warn(
                "Sonar: failed to decode ProjectedSonarImage.", throttle_duration_sec=5.0
            )
            return

        stamp = ros_time_to_sec(msg.header.stamp)

        img_proc = self._processor.process_image(img)
        t_preproc = time.time()

        # Save raw + processed images for the first N frames
        if self._debug_img_count < self.debug_save_n:
            try:
                os.makedirs(self.debug_image_dir, exist_ok=True)
                idx = self._debug_img_count
                cv2.imwrite(os.path.join(self.debug_image_dir, f"sonar_raw_{idx:03d}.png"), img)
                # processed image is float [0,1] → scale to uint8 for saving
                proc_uint8 = (img_proc * 255).astype(np.uint8) if img_proc.dtype != np.uint8 else img_proc
                cv2.imwrite(os.path.join(self.debug_image_dir, f"sonar_proc_{idx:03d}.png"), proc_uint8)
                self.get_logger().info(f"Debug: saved sonar frame {idx} to {self.debug_image_dir}")
            except Exception as e:
                self.get_logger().warn(f"Debug: could not save image {self._debug_img_count}: {e}")
            self._debug_img_count += 1

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

        # Lazy TF lookup — cached after first success
        if self._R_sonar_to_body is None:
            self._R_sonar_to_body = self._lookup_rotation(
                self.sonar_frame_id, self.base_frame_id
            )
            if self._R_sonar_to_body is None:
                return None

        dp = T[0:2, 2].copy()
        dR = T[0:2, 0:2]
        dtheta = np.arctan2(dR[1, 0], dR[0, 0])

        # Apply empirical scale calibration
        dp[0] *= self.scale_fwd
        dp[1] *= self.scale_right
        dtheta *= self.scale_heading

        # Rotate displacement from sonar frame into FRD body frame via TF.
        # The 2×2 top-left of R handles the XY-plane rotation (yaw offset between frames).
        # R[2,2] handles the sign of the heading change (e.g. -1 if sonar is mounted upside-down).
        dp_body     = self._R_sonar_to_body[:2, :2] @ dp
        dtheta_body = float(self._R_sonar_to_body[2, 2]) * dtheta

        return np.array([dp_body[0], dp_body[1], dtheta_body])

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
        # ENU: vx=East, vy=North; Forward=[cos θ, sin θ], Right=[sin θ, -cos θ]
        c, s = np.cos(theta), np.sin(theta)
        vx_body = c * vx + s * vy    # forward
        vy_body = s * vx - c * vy    # right
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
