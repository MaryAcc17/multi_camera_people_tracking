#ifndef RVIZ_SEMANTIC_MAP_PANEL_HPP_
#define RVIZ_SEMANTIC_MAP_PANEL_HPP_

#include <filesystem>
#include <optional>
#include <regex>
#include <string>
#include <utility>

#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QListWidget>
#include <QPushButton>
#include <QLabel>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <nav2_msgs/srv/load_map.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <rviz_common/panel.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <yaml-cpp/yaml.h>

namespace rviz_goal_pose_annotator
{

struct Pose2D
{
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
};

class SemanticMapPanel : public rviz_common::Panel
{
Q_OBJECT

public:
  explicit SemanticMapPanel(QWidget * parent = nullptr);

private Q_SLOTS:
  void onChooseMapPath();
  void onChooseOutputFile();
  void onAddLabel();
  void onAddEntry();
  void onOutputFileChanged();
  void onCapturePose();
  void onCaptureExtentCornerOne();
  void onCaptureExtentCornerTwo();

private:
  void setupUi();
  void onInitialize() override;
  void setupSubscriptions();
  void updateStatus(const QString & text, bool is_error = false);
  void loadYamlPreview();
  void refreshLabelsFromYaml(const YAML::Node & root);
  std::string resolveMapPath(const QString & map_path) const;
  std::string inferMapPackageFromPath(const std::string & map_path) const;
  std::string findContainingPackageRoot(const std::string & map_path) const;
  std::pair<std::string, std::string> inferMapMetadataForYaml(const std::string & map_path) const;
  void updateMapPackageFromSelectedPath(const QString & map_path);
  std::string trim(const std::string & value) const;
  YAML::Node getYamlOrDefault();
  void saveYamlFile(const YAML::Node & root) const;
  bool labelExistsInNode(const YAML::Node & labels, const std::string & label) const;
  Pose2D poseFromPoseStamped(const geometry_msgs::msg::PoseStamped & msg) const;
  void updateExtentFromCorners();
  void goalPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void clickedPointCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg);
  void publishExtentPreview();
  void clearExtentPreview();
  void requestLoadMap(const QString & map_path);

  std::string outputPath() const;
  QLabel * status_label_{nullptr};
  QLabel * pose_status_label_{nullptr};
  QLabel * extent_status_label_{nullptr};

  QLineEdit * map_package_edit_{nullptr};
  QLineEdit * map_path_edit_{nullptr};
  QPushButton * choose_map_file_btn_{nullptr};

  QLineEdit * output_file_edit_{nullptr};
  QPushButton * choose_file_btn_{nullptr};

  QListWidget * labels_list_{nullptr};
  QLineEdit * new_label_edit_{nullptr};
  QPushButton * add_label_btn_{nullptr};

  QLineEdit * entry_name_edit_{nullptr};
  QComboBox * entry_type_combo_{nullptr};
  QDoubleSpinBox * x_min_edit_{nullptr};
  QDoubleSpinBox * y_min_edit_{nullptr};
  QDoubleSpinBox * x_max_edit_{nullptr};
  QDoubleSpinBox * y_max_edit_{nullptr};
  QDoubleSpinBox * entry_x_edit_{nullptr};
  QDoubleSpinBox * entry_y_edit_{nullptr};
  QDoubleSpinBox * entry_yaw_edit_{nullptr};
  QPushButton * add_entry_btn_{nullptr};
  QPushButton * capture_pose_btn_{nullptr};
  QPushButton * capture_extent_corner1_btn_{nullptr};
  QPushButton * capture_extent_corner2_btn_{nullptr};

  std::optional<Pose2D> latest_goal_;
  std::optional<Pose2D> latest_clicked_point_;
  std::string clicked_point_frame_id_{"map"};
  std::optional<Pose2D> extent_corner_a_;
  std::optional<Pose2D> extent_corner_b_;
  bool has_captured_pose_{false};
  bool has_captured_extent_{false};
  rclcpp::Client<nav2_msgs::srv::LoadMap>::SharedPtr map_load_client_;
  std::shared_ptr<rviz_common::ros_integration::RosNodeAbstractionIface> node_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr clicked_point_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr extent_preview_pub_;
};

}  // namespace rviz_goal_pose_annotator

#endif  // RVIZ_SEMANTIC_MAP_PANEL_HPP_
