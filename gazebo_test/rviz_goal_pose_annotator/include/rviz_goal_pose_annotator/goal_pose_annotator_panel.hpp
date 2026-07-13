#ifndef RVIZ_GOAL_POSE_ANNOTATOR_PANEL_HPP_
#define RVIZ_GOAL_POSE_ANNOTATOR_PANEL_HPP_

#include <filesystem>
#include <optional>
#include <regex>
#include <string>
#include <utility>

#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <nav2_msgs/srv/load_map.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <rviz_common/panel.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <tf2/utils.h>
#include <yaml-cpp/yaml.h>

namespace rviz_goal_pose_annotator
{

struct Pose2D
{
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
};

class GoalPoseAnnotatorPanel : public rviz_common::Panel
{
Q_OBJECT

public:
  explicit GoalPoseAnnotatorPanel(QWidget * parent = nullptr);
  ~GoalPoseAnnotatorPanel() override;

  void onInitialize() override;

private Q_SLOTS:
  void onCaptureStart();
  void onCaptureGoal();
  void onSaveEpisode();
  void onChooseOutputFile();
  void onChooseMapPath();

private:
  void setupUi();
  void setupSubscriptions();
  void updateSuggestedEpisodeName();
  void updateStatus(const QString & text, bool is_error = false);
  void updatePoseStatusLabel(
    QLabel * label,
    const std::string & text,
    const std::optional<Pose2D> & pose);

  void appendEpisodeToYaml();
  std::string nextSuggestedEpisodeName() const;
  YAML::Node getYamlOrDefault();
  void saveYamlFile(const YAML::Node & root) const;
  std::string resolveMapUrl(const QString & map_path) const;
  std::string inferMapPackageFromPath(const std::string & map_path) const;
  std::string findContainingPackageRoot(const std::string & map_path) const;
  std::pair<std::string, std::string> inferMapMetadataForYaml(
    const std::string & map_path) const;
  void updateMapPackageFromSelectedPath(const QString & map_path);
  void startPoseCallback(
    const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
  void goalPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void publishPosePreviewMarkers(
    const Pose2D & pose, const std::string & frame_id, bool is_initial_pose);
  void clearPosePreviewMarkers(bool is_initial_pose);
  visualization_msgs::msg::Marker createArrowMarker(
    const Pose2D & pose, const std::string & frame_id, int id, const std::string & text);
  visualization_msgs::msg::Marker createTextMarker(
    const Pose2D & pose, const std::string & frame_id, int id, const std::string & text);
  Pose2D poseFromPoseWithCovarianceStamped(
    const geometry_msgs::msg::PoseWithCovarianceStamped & msg) const;
  Pose2D poseFromPoseStamped(const geometry_msgs::msg::PoseStamped & msg) const;

  std::shared_ptr<rviz_common::ros_integration::RosNodeAbstractionIface> node_ptr_;
  rclcpp::Client<nav2_msgs::srv::LoadMap>::SharedPtr map_load_client_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Subscription<
    geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr start_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr preview_marker_pub_;

  QLineEdit * robot_name_edit_{nullptr};
  QLineEdit * map_path_edit_{nullptr};
  QLineEdit * map_package_edit_{nullptr};
  QDoubleSpinBox * init_x_edit_{nullptr};
  QDoubleSpinBox * init_y_edit_{nullptr};
  QDoubleSpinBox * init_yaw_edit_{nullptr};

  QLineEdit * output_file_edit_{nullptr};
  QLineEdit * episode_name_edit_{nullptr};

  QLabel * latest_start_label_{nullptr};
  QLabel * latest_goal_label_{nullptr};
  QLabel * captured_start_label_{nullptr};
  QLabel * captured_goal_label_{nullptr};
  QLabel * status_label_{nullptr};

  QPushButton * capture_start_btn_{nullptr};
  QPushButton * capture_goal_btn_{nullptr};
  QPushButton * save_episode_btn_{nullptr};
  QPushButton * choose_file_btn_{nullptr};
  QPushButton * choose_map_file_btn_{nullptr};

  std::optional<Pose2D> latest_start_;
  std::optional<Pose2D> latest_goal_;
  std::optional<Pose2D> captured_start_;
  std::optional<Pose2D> captured_goal_;
  std::string latest_start_frame_id_{"map"};
  std::string latest_goal_frame_id_{"map"};

};

}  // namespace rviz_goal_pose_annotator

#endif  // RVIZ_GOAL_POSE_ANNOTATOR_PANEL_HPP_
