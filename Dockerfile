FROM ros:humble-ros-base

ARG USERNAME=rosdev
ARG UID=1000
ARG GID=$UID

# Install dependencies (union of both Dockerfiles)
RUN apt update -q \
    && apt upgrade -q -y \
    && apt install -y --no-install-recommends \
    software-properties-common \
    python3-pip \
    python3-dev \
    xauth \
    build-essential \
    cmake \
    git \
    sudo \
    libboost-all-dev \
    ros-humble-cv-bridge \
    netcat \
    iputils-ping \
    ros-humble-rosbag2-storage-mcap \
    ros-humble-vision-msgs \
    ros-humble-sensor-msgs-py \
    libogre-1.12-dev \
    ros-humble-rviz2 \
    gfortran \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    libfreetype6-dev \
    libpng-dev \
    libeigen3-dev \
    libpcl-dev \
    ros-humble-pcl-conversions \
    ros-humble-pcl-msgs \
    ros-humble-tf2 \
    ros-humble-tf2-geometry-msgs \
    ros-humble-geometry-msgs \
    ros-humble-nav-msgs \
    ros-humble-std-msgs \
    ros-humble-marine-acoustic-msgs \
    ros-humble-rosidl-default-generators \
    ros-humble-message-filters \
    libyaml-dev \
    libceres-dev \
    ros-humble-foxglove-bridge \
    ros-humble-perception-pcl \
    pybind11-dev \
    && apt clean \
    apt install ros-humble-foxglove-bridge \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Python dependencies (requirements.txt + explicit packages from second Dockerfile)
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt \
    numpy \
    scipy  \
    matplotlib \
    opencv-python \
    open3d

# Create a non-root user and set permissions
RUN groupadd -g $GID $USERNAME && \
    useradd -m -u $UID -g $GID -s /bin/bash $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set the correct permissions for the workspace directory
ADD ros2_ws /home/$USERNAME/ros2_ws
RUN chown -R $USERNAME:$USERNAME /home/$USERNAME/ros2_ws

##############################################################
# Make sure sudo commands have been executed before this line #
# Switch to the non-root user
USER $USERNAME

# Clean previous build artifacts
RUN rm -rf /home/$USERNAME/ros2_ws/build /home/$USERNAME/ros2_ws/log /home/$USERNAME/ros2_ws/install

# Source ROS in bashrc
RUN echo 'source /opt/ros/'$ROS_DISTRO'/setup.bash' >> /home/$USERNAME/.bashrc \
    && echo 'source /home/'$USERNAME'/ros2_ws/install/setup.bash' >> /home/$USERNAME/.bashrc

# Build the ROS 2 workspace
WORKDIR /home/$USERNAME/ros2_ws
RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && colcon build --symlink-install"
RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && source /home/$USERNAME/ros2_ws/install/setup.bash && rosdep update && rosdep install --from-paths src --ignore-src -r -y && colcon build --symlink-install"

# Copy middleware profile
RUN sudo mkdir -p /usr/local/share/middleware_profiles/
COPY rtps_udp_profile.xml /usr/local/share/middleware_profiles/

CMD ["bash"]