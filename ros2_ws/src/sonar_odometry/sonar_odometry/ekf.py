"""
Extended Kalman Filter for Sonar Inertial Odometry.

State vector (10D):
  [px, py, vx, vy, theta, b_ax, b_ay, b_gz, s_ax, s_ay]
   px, py    : position in NED frame (meters)
   vx, vy    : velocity in NED frame (m/s)
   theta     : heading (rad), zero = North, positive = East (clockwise)
   b_ax, b_ay: accelerometer biases (m/s^2)
   b_gz      : gyro yaw-rate bias (rad/s)
   s_ax, s_ay: accelerometer scale factors (dimensionless)

IMU input convention (FRD body frame):
  u = [ax_frd, ay_frd, gz_frd]
  ax_frd: forward acceleration (m/s^2, gravity-compensated)
  ay_frd: rightward acceleration (m/s^2, gravity-compensated)
  gz_frd: yaw-rate positive nose-right / clockwise (rad/s)

Sonar measurement (body frame):
  z = [d_forward, d_right, d_theta]
  d_forward: displacement forward   (meters)
  d_right  : displacement rightward (meters)
  d_theta  : heading change         (rad)
"""

from __future__ import annotations

import numpy as np


class ESEKF:
    # ------------------------------------------------------------------ init
    def __init__(
        self,
        initial_heading_rad: float = 0.0,
        accel_bias: tuple[float, float, float] = (0.0, 0.0, 0.0),
        gyro_bias_z: float = 0.002,
        imu_sample_rate_hz: float = 100.0,
        # IMU noise spectral densities (from datasheet)
        n_accel_ug_sqrthz: float = 135.0,   # µg/√Hz
        n_gyro_dps_sqrthz: float = 0.005,   # dps/√Hz
        # Process noise for random-walk terms
        sigma_b_accel: float = 0.5,
        sigma_b_gyro: float = 0.5,
        sigma_scale: float = 0.001,
        # Measurement noise (sonar, in meters / rad)
        sigma_sonar_pos: float = 0.023464,   # sqrt(0.00055)
        sigma_sonar_heading: float = 0.023464,
    ):
        # ---- Nominal state ----
        self.px: float = 0.0
        self.py: float = 0.0
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.theta: float = initial_heading_rad
        self.b_ax: float = float(accel_bias[0])
        self.b_ay: float = float(accel_bias[1])
        self.b_gz: float = gyro_bias_z
        self.s_ax: float = 1.0
        self.s_ay: float = 1.0

        # ---- Covariance ----
        self.P: np.ndarray = np.eye(10) * 10.0

        # ---- Process noise ----
        n_accel_si = n_accel_ug_sqrthz * 1e-6 * 9.81          # m/s^2/√Hz
        n_gyro_si  = n_gyro_dps_sqrthz * (np.pi / 180.0)      # rad/s/√Hz
        fs = imu_sample_rate_hz
        self.sigma_eta1_sq = (n_accel_si * np.sqrt(fs)) ** 2
        self.sigma_eta2_sq = (n_accel_si * np.sqrt(fs)) ** 2
        self.sigma_eta3_sq = (n_gyro_si  * np.sqrt(fs)) ** 2
        self.sigma_b_ax_sq = sigma_b_accel ** 2
        self.sigma_b_ay_sq = sigma_b_accel ** 2
        self.sigma_b_gz_sq = sigma_b_gyro  ** 2
        self.ss_ax = sigma_scale ** 2
        self.ss_ay = sigma_scale ** 2

        # ---- Measurement noise ----
        self.sigma_w4_sq = sigma_sonar_pos     ** 2
        self.sigma_w5_sq = sigma_sonar_pos     ** 2
        self.sigma_w6_sq = sigma_sonar_heading ** 2

        # ---- Bookkeeping (state at last sonar update) ----
        self.prev_px:    float | None = None
        self.prev_py:    float | None = None
        self.prev_theta: float | None = None

    # ----------------------------------------------------------------- helpers
    def _as_vector(self) -> np.ndarray:
        return np.array([
            self.px, self.py, self.vx, self.vy, self.theta,
            self.b_ax, self.b_ay, self.b_gz, self.s_ax, self.s_ay
        ])

    def _set_from_vector(self, x: np.ndarray) -> None:
        (self.px, self.py, self.vx, self.vy, self.theta,
         self.b_ax, self.b_ay, self.b_gz, self.s_ax, self.s_ay) = x

    @staticmethod
    def _wrap_angle(a: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return float(np.arctan2(np.sin(a), np.cos(a)))

    @staticmethod
    def _wrap_to_2pi(a: float) -> float:
        """Wrap angle to [0, 2pi]."""
        a = np.arctan2(np.sin(a), np.cos(a))
        if a < 0:
            a += 2 * np.pi
        return float(a)

    def _build_R(self) -> np.ndarray:
        return np.diag([self.sigma_w4_sq, self.sigma_w5_sq, self.sigma_w6_sq])

    # --------------------------------------------------------------- prediction
    def prediction(self, u: np.ndarray, dt: float) -> None:
        """
        EKF prediction step driven by IMU measurements.

        Parameters
        ----------
        u  : array [ax_frd, ay_frd, gz_frd]  (FRD body frame, gravity-compensated)
        dt : time step in seconds
        """
        if dt <= 0.0:
            return

        ax_m, ay_m, gz_m = float(u[0]), float(u[1]), float(u[2])

        # Cache previous state
        theta = self.theta
        vx, vy = self.vx, self.vy
        b_ax, b_ay, b_gz = self.b_ax, self.b_ay, self.b_gz
        s_ax, s_ay = self.s_ax, self.s_ay

        # Corrected body-frame accelerations and yaw rate
        ax_c = (ax_m - b_ax) / s_ax
        ay_c = (ay_m - b_ay) / s_ay
        gz_c = gz_m - b_gz

        # Rotation: body → NED
        c, s = np.cos(theta), np.sin(theta)
        ax_ned = c * ax_c - s * ay_c
        ay_ned = s * ax_c + c * ay_c

        # Centripetal correction: ω × v  (2-D)
        ax_total = ax_ned + (-gz_c * vy)
        ay_total = ay_ned + ( gz_c * vx)

        # ------ Linearised continuous-time Jacobian (10×10) ------
        Fc = np.zeros((10, 10))

        Fc[0, 2] = 1.0
        Fc[1, 3] = 1.0

        Fc[2, 3] = -gz_c
        Fc[3, 2] =  gz_c

        fvx_th = -s * ax_c - c * ay_c
        fvy_th =  c * ax_c - s * ay_c
        Fc[0, 4] = 0.5 * dt * fvx_th
        Fc[1, 4] = 0.5 * dt * fvy_th
        Fc[2, 4] = fvx_th
        Fc[3, 4] = fvy_th

        Fc[2, 5] = -c / s_ax
        Fc[2, 6] =  s / s_ay
        Fc[3, 5] = -s / s_ax
        Fc[3, 6] = -c / s_ay

        Fc[2, 7] =  vy
        Fc[3, 7] = -vx

        dax_dsax = -(ax_m - b_ax) / (s_ax ** 2)
        day_dsay = -(ay_m - b_ay) / (s_ay ** 2)
        Fc[2, 8] =  c * dax_dsax
        Fc[2, 9] = -s * day_dsay
        Fc[3, 8] =  s * dax_dsax
        Fc[3, 9] =  c * day_dsay

        Fc[4, 7] = -1.0

        F = np.eye(10) + dt * Fc

        # ------ Process noise mapping (10×8) ------
        Gd = np.zeros((10, 8))

        Gd[0, 0] = 0.5 * dt**2 * c / s_ax
        Gd[0, 1] = -0.5 * dt**2 * s / s_ay
        Gd[1, 0] = 0.5 * dt**2 * s / s_ax
        Gd[1, 1] = 0.5 * dt**2 * c / s_ay

        Gd[2, 0] = dt * c / s_ax
        Gd[2, 1] = -dt * s / s_ay
        Gd[3, 0] = dt * s / s_ax
        Gd[3, 1] = dt * c / s_ay

        Gd[2, 2] = -dt * vy
        Gd[3, 2] =  dt * vx

        Gd[4, 2] = dt

        Gd[5, 3] = dt
        Gd[6, 4] = dt
        Gd[7, 5] = dt
        Gd[8, 6] = dt
        Gd[9, 7] = dt

        Qn = np.diag([
            self.sigma_eta1_sq,
            self.sigma_eta2_sq,
            self.sigma_eta3_sq,
            self.sigma_b_ax_sq,
            self.sigma_b_ay_sq,
            self.sigma_b_gz_sq,
            self.ss_ax,
            self.ss_ay,
        ])
        Qd = Gd @ Qn @ Gd.T

        # ------ Propagate nominal state ------
        self.px    += vx * dt + 0.5 * ax_total * dt**2
        self.py    += vy * dt + 0.5 * ay_total * dt**2
        self.vx    += ax_total * dt
        self.vy    += ay_total * dt
        self.theta += gz_c * dt
        self.theta  = self._wrap_to_2pi(self.theta)

        # ------ Propagate covariance ------
        self.P = F @ self.P @ F.T + Qd

    # ------------------------------------------------------------------ update
    def update(
        self,
        z_sonar: np.ndarray,
        nis_threshold: float = 30.0,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        EKF update with sonar displacement measurement.

        Parameters
        ----------
        z_sonar       : array [d_forward, d_right, d_theta]  (body frame, meters / rad)
        nis_threshold : reject outliers above this NIS value

        Returns
        -------
        innovation, innovation_covariance  (None, None on skip / first call)
        """
        if self.prev_px is None:
            # First sonar measurement — initialise position anchor and return
            self.prev_px    = self.px
            self.prev_py    = self.py
            self.prev_theta = self.theta
            return None, None

        # Global displacement since last sonar update
        dp_n = self.px - self.prev_px
        dp_e = self.py - self.prev_py

        # Predicted body-frame displacement
        c = np.cos(self.prev_theta)
        s = np.sin(self.prev_theta)
        z_pred = np.array([
            c * dp_n + s * dp_e,                             # forward
           -s * dp_n + c * dp_e,                             # right
            self._wrap_angle(self.theta - self.prev_theta),  # dtheta
        ])

        # Innovation
        y = z_sonar - z_pred
        y[2] = self._wrap_angle(y[2])

        # Measurement Jacobian (3×10)
        H = np.zeros((3, 10))
        H[0, 0] =  c;  H[0, 1] = s
        H[1, 0] = -s;  H[1, 1] = c
        H[2, 4] =  1.0

        R = self._build_R()
        S = H @ self.P @ H.T + R

        # NIS gating
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return None, None

        nis = float(y.T @ S_inv @ y)
        if nis > nis_threshold:
            return y, S   # return for logging but do NOT update

        # Kalman gain & state correction
        K  = self.P @ H.T @ S_inv
        dx = K @ y
        self._set_from_vector(self._as_vector() + dx)
        self.theta = self._wrap_to_2pi(self.theta)

        # Joseph-form covariance update
        I_KH = np.eye(10) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        # Advance anchor
        self.prev_px    = self.px
        self.prev_py    = self.py
        self.prev_theta = self.theta

        return y, S

    # ---------------------------------------------------------- public getters
    @property
    def position(self) -> np.ndarray:
        return np.array([self.px, self.py])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy])

    @property
    def heading(self) -> float:
        return self.theta

    @property
    def covariance(self) -> np.ndarray:
        return self.P.copy()
