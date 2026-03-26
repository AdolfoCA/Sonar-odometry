cmake_minimum_required(VERSION 3.8)
project(tf_static_publisher)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(yaml-cpp REQUIRED)

add_executable(tf_static_publisher src/tf_publisher.cpp)
ament_target_dependencies(tf_static_publisher rclcpp tf2_ros tf2_geometry_msgs tf2 yaml-cpp)

target_link_libraries(tf_static_publisher yaml-cpp)

install(TARGETS
  tf_static_publisher
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
