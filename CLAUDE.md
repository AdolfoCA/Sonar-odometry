# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Sonar-inertial odometry system for underwater vehicles. Fuses sonar visual features (AKAZE + RANSAC) with IMU data via an Extended Kalman Filter (EKF) to estimate 2D position, heading, and velocity in the NED frame. Built as a ROS2 Humble Python package, intended to run in Docker.

## Development Environment

The preferred workflow is Docker. Build and enter the container, then use `colcon` inside it.

```bash
# Start the container (mounts ros2_ws/src as a live volume)
docker-compose up -d

# Or build and run directly
docker build -t sonar_odometry .
```

### Building (inside container or local ROS2 Humble environment)

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### Running

```bash
# Launch the full node with config
ros2 launch sonar_odometry sonar_odometry.launch.py

# Or run node directly (after sourcing install/setup.bash)
ros2 run sonar_odometry sonar_odometry_node
```

### Testing individual modules

The Python modules have no test suite; test interactively via Python:

```bash
python3 ros2_ws/src/sonar_odometry/sonar_odometry/ekf.py
python3 ros2_ws/src/sonar_odometry/sonar_odometry/feature_matching.py
```

## Architecture

### Data Flow

```
IMU (~100 Hz)    →  EKF prediction step  →  /odometry
Sonar image      →  ImageProcessor → FeatureMatcher → EKF update step  →  /odometry + /odometry/path
GPS (optional)   →  NED conversion  →  /gps/path (reference only)
```

### Key Modules

| File | Role |
|------|------|
| `sonar_odometry_node.py` | ROS2 node; sensor callbacks; publishes odometry + TF |
| `ekf.py` | `ESEKF` class — 10D state EKF (position, velocity, heading, IMU biases/scales) |
| `feature_matching.py` | `SonarFeatureMatcher` — AKAZE detection, BFMatcher (Hamming + Lowe ratio test), RANSAC affine transform, polar→Cartesian conversion |
| `image_processing.py` | `SonarImageProcessor` — row-wise background subtraction, Gaussian+median filter, fuzzy gamma correction, morphological opening, thresholding |

### EKF State Vector (10D)

```
[px, py, vx, vy, theta, b_ax, b_ay, b_gz, s_ax, s_ay]
```
- `px, py` — NED position (m)
- `vx, vy` — NED velocity (m/s)
- `theta` — heading (rad)
- `b_ax, b_ay` — accelerometer biases (m/s²)
- `b_gz` — gyro bias (rad/s)
- `s_ax, s_ay` — accelerometer scale factors

**Prediction**: driven by IMU (converted from ROS frame → FRD body frame → NED navigation frame). Applies centripetal acceleration correction (`ω × v`).

**Update**: sonar measurement `[d_forward, d_right, d_theta]` in body frame. Uses NIS gating (chi² threshold) to reject outliers.

### Coordinate Frames

- **NED** (North-East-Down) — navigation/world frame
- **FRD** (Forward-Right-Down) — body frame
- ROS uses a different convention; conversions are applied in the IMU callback

### Sonar Sensor Parameters (Oculus)

Fixed in `feature_matching.py`:
- 512 beams, ±65° azimuth, 0.254°/beam
- Range: 0.0079–9.985 m, resolution 0.0158 m/bin

### Topics

| Topic | Type | Direction |
|-------|------|-----------|
| `/imu/data` (configurable) | `sensor_msgs/Imu` | Input |
| `/oculus/sonar_image` (configurable) | `marine_acoustic_msgs/ProjectedSonarImage` | Input |
| `/fix` | `sensor_msgs/NavSatFix` | Input (optional) |
| `/odometry` | `nav_msgs/Odometry` | Output |
| `/odometry/path` | `nav_msgs/Path` | Output |
| `/gps/path` | `nav_msgs/Path` | Output |

## Configuration

`ros2_ws/src/sonar_odometry/config/sonar_odometry.yaml` — all tunable parameters:

- `initial_heading_deg` — must be set to vehicle's heading at start
- `imu_bias_*` — pre-calibrated accelerometer/gyro biases (m/s², rad/s)
- `sonar_scale_*` — empirically tuned scale factors for sonar measurements
- `nis_threshold` — EKF outlier gate (chi² 3-DOF; default 30, relaxed from theoretical 11.3)
- `min_inliers`, `min_inlier_ratio` — RANSAC quality thresholds

Topic remapping is done in the launch file, not the YAML.
