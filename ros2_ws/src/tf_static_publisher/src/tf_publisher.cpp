#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <fstream>
#include <yaml-cpp/yaml.h>

#include "rclcpp/rclcpp.hpp"
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#define DEG2RAD(deg) ((deg) * M_PI / 180.0)
#define NODE_NAME "tf_static_publisher"

using namespace std::chrono_literals;


class StaticTFPublisher : public rclcpp::Node
{
public:
  StaticTFPublisher() : Node(NODE_NAME)
  {
    std::string package_share_directory = ament_index_cpp::get_package_share_directory(NODE_NAME);
    std::string default_path = package_share_directory + "/config/transforms.yaml";
    this->declare_parameter<std::string>("transforms_path", default_path);

    std::string transforms_path;
    this->get_parameter("transforms_path", transforms_path);
    loadTransformsFromYAML(transforms_path);

    tf_static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    publishTransforms(); 
  }

private:
  std::vector<geometry_msgs::msg::TransformStamped> transforms_;

  void loadTransformsFromYAML(const std::string &yaml_file)
  {
    YAML::Node config = YAML::LoadFile(yaml_file);
    for (const auto &transform : config["transforms"])
    {
      geometry_msgs::msg::TransformStamped tf;
      tf.header.frame_id = transform["frame_id"].as<std::string>();
      tf.child_frame_id = transform["child_frame_id"].as<std::string>();
      tf.transform.translation.x = transform["translation"]["x"].as<double>();
      tf.transform.translation.y = transform["translation"]["y"].as<double>();
      tf.transform.translation.z = transform["translation"]["z"].as<double>();

      double roll = DEG2RAD(transform["rotation"]["x"].as<double>());
      double pitch = DEG2RAD(transform["rotation"]["y"].as<double>());
      double yaw = DEG2RAD(transform["rotation"]["z"].as<double>());

      tf2::Quaternion quat;
      quat.setRPY(roll, pitch, yaw);
      tf.transform.rotation = tf2::toMsg(quat);

      transforms_.push_back(tf);
    }
  }

  void publishTransforms()
  {
    for (auto &tf : transforms_)
    {
      tf.header.stamp = this->now();
      tf_static_broadcaster_->sendTransform(tf);
    }
  }

  rclcpp::TimerBase::SharedPtr timer_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_static_broadcaster_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<StaticTFPublisher>();
  RCLCPP_INFO(rclcpp::get_logger("tf_static_publisher"), "Node started publishing static transforms");
  rclcpp::spin(node);
  RCLCPP_INFO(rclcpp::get_logger("tf_static_publisher"), "Node stopped");
  rclcpp::shutdown();
  return 0;
}