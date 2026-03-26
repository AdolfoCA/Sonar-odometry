from setuptools import find_packages, setup
import os
from glob import glob

package_name = "sonar_odometry"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="todo@todo.com",
    description="Sonar Inertial Odometry with EKF",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            f"sonar_odometry_node = {package_name}.sonar_odometry_node:main",
            f"gps_path_node = {package_name}.gps_path:main",
        ],
    },
)
