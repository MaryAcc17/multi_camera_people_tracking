#include "rviz_goal_pose_annotator/goal_pose_annotator_panel.hpp"

#include <algorithm>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <QGroupBox>
#include <QMetaObject>

#include <rviz_common/display_context.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <ament_index_cpp/get_packages_with_prefixes.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace rviz_goal_pose_annotator
{

namespace
{

YAML::Node toThreeDecimalNode(double value)
{
  std::ostringstream stream;
  stream.setf(std::ios::fixed);
  stream << std::setprecision(3) << value;
  return YAML::Load(stream.str());
}

}  // namespace

GoalPoseAnnotatorPanel::GoalPoseAnnotatorPanel(QWidget * parent)
: rviz_common::Panel(parent)
{
  setupUi();
}

GoalPoseAnnotatorPanel::~GoalPoseAnnotatorPanel()
{
  clearPosePreviewMarkers(true);
  clearPosePreviewMarkers(false);
}

void GoalPoseAnnotatorPanel::onInitialize()
{
  node_ptr_ = getDisplayContext()->getRosNodeAbstraction().lock();
  const auto raw_node = node_ptr_ ? node_ptr_->get_raw_node() : nullptr;
  if (!raw_node) {
    return;
  }

  setupSubscriptions();
  map_load_client_ = raw_node->create_client<nav2_msgs::srv::LoadMap>("/map_server/load_map");
  preview_marker_pub_ = raw_node->create_publisher<visualization_msgs::msg::MarkerArray>(
    "/goal_pose_annotator/preview_markers", 10);
  updateSuggestedEpisodeName();
}

void GoalPoseAnnotatorPanel::setupUi()
{
  auto * main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(8, 8, 8, 8);

  auto * robot_group = new QGroupBox("Run metadata", this);
  auto * robot_layout = new QVBoxLayout(robot_group);

  robot_name_edit_ = new QLineEdit("jackal", this);
  map_path_edit_ = new QLineEdit("maps/social_env_test_map.yaml", this);
  map_package_edit_ = new QLineEdit(this);
  map_package_edit_->setPlaceholderText("auto-detect from map path");
  choose_map_file_btn_ = new QPushButton("Choose map path", this);
  connect(choose_map_file_btn_, &QPushButton::clicked, this,
    &GoalPoseAnnotatorPanel::onChooseMapPath);
  connect(map_path_edit_, &QLineEdit::editingFinished, this,
    [this]() {
      updateMapPackageFromSelectedPath(map_path_edit_->text());
    });
  init_x_edit_ = new QDoubleSpinBox(this);
  init_y_edit_ = new QDoubleSpinBox(this);
  init_yaw_edit_ = new QDoubleSpinBox(this);
  for (auto spin : {init_x_edit_, init_y_edit_, init_yaw_edit_}) {
    spin->setRange(-99999.0, 99999.0);
    spin->setDecimals(3);
    spin->setSingleStep(0.05);
  }
  init_x_edit_->setValue(0.6);
  init_y_edit_->setValue(4.25);
  init_yaw_edit_->setValue(-3.14);

  robot_layout->addWidget(new QLabel("robot_name", this));
  robot_layout->addWidget(robot_name_edit_);
  robot_layout->addWidget(new QLabel("map_package", this));
  robot_layout->addWidget(map_package_edit_);
  robot_layout->addWidget(new QLabel("map_path", this));
  auto * map_row_layout = new QHBoxLayout();
  map_row_layout->addWidget(map_path_edit_);
  map_row_layout->addWidget(choose_map_file_btn_);
  robot_layout->addLayout(map_row_layout);
  robot_layout->addWidget(new QLabel("initial_pose [x, y, yaw]", this));
  robot_layout->addWidget(init_x_edit_);
  robot_layout->addWidget(init_y_edit_);
  robot_layout->addWidget(init_yaw_edit_);
  main_layout->addWidget(robot_group);

  auto * output_group = new QGroupBox("YAML output", this);
  auto * output_layout = new QVBoxLayout(output_group);
  output_file_edit_ = new QLineEdit(
    std::filesystem::path(std::filesystem::current_path() / "goals_and_poses_output.yaml")
    .string().c_str(),
    this);
  choose_file_btn_ = new QPushButton("Choose output file", this);
  connect(choose_file_btn_, &QPushButton::clicked, this,
    &GoalPoseAnnotatorPanel::onChooseOutputFile);
  auto * output_file_row = new QHBoxLayout();
  output_file_row->addWidget(output_file_edit_);
  output_file_row->addWidget(choose_file_btn_);
  output_layout->addLayout(output_file_row);
  episode_name_edit_ = new QLineEdit("episode_1", this);
  auto * episode_row = new QVBoxLayout();
  episode_row->addWidget(new QLabel("episode name (editable, auto-suggested)", this));
  episode_row->addWidget(episode_name_edit_);
  output_layout->addLayout(episode_row);
  main_layout->addWidget(output_group);

  auto * poses_group = new QGroupBox("Collection", this);
  auto * poses_layout = new QVBoxLayout(poses_group);
  latest_start_label_ = new QLabel("Latest /initialpose: (none)", this);
  latest_goal_label_ = new QLabel("Latest /goal_pose: (none)", this);
  captured_start_label_ = new QLabel("Captured start: (none)", this);
  captured_goal_label_ = new QLabel("Captured goal: (none)", this);
  capture_start_btn_ = new QPushButton("Capture start from latest /initialpose", this);
  capture_goal_btn_ = new QPushButton("Capture goal from latest /goal_pose", this);
  save_episode_btn_ = new QPushButton("Append episode to YAML", this);

  connect(capture_start_btn_, &QPushButton::clicked, this,
    &GoalPoseAnnotatorPanel::onCaptureStart);
  connect(capture_goal_btn_, &QPushButton::clicked, this,
    &GoalPoseAnnotatorPanel::onCaptureGoal);
  connect(save_episode_btn_, &QPushButton::clicked, this,
    &GoalPoseAnnotatorPanel::onSaveEpisode);

  poses_layout->addWidget(latest_start_label_);
  poses_layout->addWidget(capture_start_btn_);
  poses_layout->addWidget(captured_start_label_);
  poses_layout->addWidget(latest_goal_label_);
  poses_layout->addWidget(capture_goal_btn_);
  poses_layout->addWidget(captured_goal_label_);
  poses_layout->addWidget(save_episode_btn_);
  status_label_ = new QLabel("Waiting for topic messages...", this);
  status_label_->setStyleSheet("color: gray;");
  poses_layout->addWidget(status_label_);
  main_layout->addWidget(poses_group);

  main_layout->addStretch();
}

void GoalPoseAnnotatorPanel::setupSubscriptions()
{
  if (!node_ptr_) {
    return;
  }
  auto raw_node = node_ptr_->get_raw_node();
  if (!raw_node) {
    return;
  }
  start_sub_ = raw_node->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
    "/initialpose", rclcpp::QoS(10),
    std::bind(
      &GoalPoseAnnotatorPanel::startPoseCallback, this, std::placeholders::_1));
  goal_sub_ = raw_node->create_subscription<geometry_msgs::msg::PoseStamped>(
    "/goal_pose", rclcpp::QoS(10),
    std::bind(
      &GoalPoseAnnotatorPanel::goalPoseCallback, this, std::placeholders::_1));
}

Pose2D GoalPoseAnnotatorPanel::poseFromPoseWithCovarianceStamped(
  const geometry_msgs::msg::PoseWithCovarianceStamped & msg) const
{
  Pose2D pose;
  pose.x = msg.pose.pose.position.x;
  pose.y = msg.pose.pose.position.y;
  pose.yaw = tf2::getYaw(msg.pose.pose.orientation);
  return pose;
}

Pose2D GoalPoseAnnotatorPanel::poseFromPoseStamped(
  const geometry_msgs::msg::PoseStamped & msg) const
{
  Pose2D pose;
  pose.x = msg.pose.position.x;
  pose.y = msg.pose.position.y;
  pose.yaw = tf2::getYaw(msg.pose.orientation);
  return pose;
}

void GoalPoseAnnotatorPanel::startPoseCallback(
  const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
  latest_start_ = poseFromPoseWithCovarianceStamped(*msg);
  updatePoseStatusLabel(latest_start_label_, "Latest /initialpose: ", latest_start_);
  const std::string frame_id = msg->header.frame_id.empty() ? "map" : msg->header.frame_id;
  latest_start_frame_id_ = frame_id;
  if (!captured_start_.has_value()) {
    publishPosePreviewMarkers(*latest_start_, frame_id, true);
  }
}

void GoalPoseAnnotatorPanel::goalPoseCallback(
  const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  latest_goal_ = poseFromPoseStamped(*msg);
  updatePoseStatusLabel(latest_goal_label_, "Latest /goal_pose: ", latest_goal_);
  const std::string frame_id = msg->header.frame_id.empty() ? "map" : msg->header.frame_id;
  latest_goal_frame_id_ = frame_id;
  if (!captured_goal_.has_value()) {
    publishPosePreviewMarkers(*latest_goal_, frame_id, false);
  }
}

void GoalPoseAnnotatorPanel::onCaptureStart()
{
  if (!latest_start_.has_value()) {
    updateStatus("No /initialpose received yet", true);
    return;
  }
  captured_start_ = latest_start_;
  clearPosePreviewMarkers(true);
  updatePoseStatusLabel(captured_start_label_, "Captured /initialpose: ", captured_start_);
  updateStatus("Start pose captured.");
}

void GoalPoseAnnotatorPanel::onCaptureGoal()
{
  if (!latest_goal_.has_value()) {
    updateStatus("No /goal_pose received yet", true);
    return;
  }
  captured_goal_ = latest_goal_;
  clearPosePreviewMarkers(false);
  updatePoseStatusLabel(captured_goal_label_, "Captured /goal_pose: ", captured_goal_);
  updateStatus("Goal pose captured.");
}

visualization_msgs::msg::Marker GoalPoseAnnotatorPanel::createArrowMarker(
  const Pose2D & pose, const std::string & frame_id, int id, const std::string & text)
{
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = frame_id;
  marker.ns = "goal_pose_annotator";
  marker.id = id;
  marker.type = visualization_msgs::msg::Marker::ARROW;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.position.x = pose.x;
  marker.pose.position.y = pose.y;
  marker.pose.position.z = 0.02;
  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, pose.yaw);
  marker.pose.orientation = tf2::toMsg(q);
  marker.scale.x = 0.8;
  marker.scale.y = 0.15;
  marker.scale.z = 0.15;
  if (text == "initial pose") {
    marker.color.r = 0.2;
    marker.color.g = 0.8;
    marker.color.b = 0.2;
  } else {
    marker.color.r = 0.9;
    marker.color.g = 0.2;
    marker.color.b = 0.2;
  }
  marker.color.a = 0.95;
  return marker;
}

visualization_msgs::msg::Marker GoalPoseAnnotatorPanel::createTextMarker(
  const Pose2D & pose, const std::string & frame_id, int id, const std::string & text)
{
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = frame_id;
  marker.ns = "goal_pose_annotator";
  marker.id = id;
  marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.position.x = pose.x;
  marker.pose.position.y = pose.y;
  marker.pose.position.z = 0.35;
  marker.pose.orientation.w = 1.0;
  marker.scale.z = 0.35;
  marker.text = text;
  marker.color.r = 1.0;
  marker.color.g = 1.0;
  marker.color.b = 1.0;
  marker.color.a = 0.9;
  return marker;
}

void GoalPoseAnnotatorPanel::publishPosePreviewMarkers(
  const Pose2D & pose, const std::string & frame_id, bool is_initial_pose)
{
  if (!preview_marker_pub_) {
    return;
  }
  auto raw_node = node_ptr_ ? node_ptr_->get_raw_node() : nullptr;
  if (!raw_node) {
    return;
  }
  const std::string label = is_initial_pose ? "initial pose" : "goal pose";
  const int arrow_id = is_initial_pose ? 0 : 2;
  const int text_id = is_initial_pose ? 1 : 3;
  auto arrow_marker = createArrowMarker(pose, frame_id, arrow_id, label);
  auto text_marker = createTextMarker(pose, frame_id, text_id, label);
  arrow_marker.header.stamp = raw_node->get_clock()->now();
  text_marker.header.stamp = raw_node->get_clock()->now();

  visualization_msgs::msg::MarkerArray marker_array;
  marker_array.markers.push_back(arrow_marker);
  marker_array.markers.push_back(text_marker);
  preview_marker_pub_->publish(marker_array);
}

void GoalPoseAnnotatorPanel::clearPosePreviewMarkers(bool is_initial_pose)
{
  if (!preview_marker_pub_) {
    return;
  }
  auto raw_node = node_ptr_ ? node_ptr_->get_raw_node() : nullptr;
  if (!raw_node) {
    return;
  }
  const int arrow_id = is_initial_pose ? 0 : 2;
  const int text_id = is_initial_pose ? 1 : 3;
  visualization_msgs::msg::Marker delete_arrow;
  delete_arrow.header.frame_id = "map";
  delete_arrow.ns = "goal_pose_annotator";
  delete_arrow.id = arrow_id;
  delete_arrow.type = visualization_msgs::msg::Marker::ARROW;
  delete_arrow.action = visualization_msgs::msg::Marker::DELETE;
  delete_arrow.header.stamp = raw_node->get_clock()->now();
  visualization_msgs::msg::Marker delete_text;
  delete_text.header.frame_id = "map";
  delete_text.ns = "goal_pose_annotator";
  delete_text.id = text_id;
  delete_text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  delete_text.action = visualization_msgs::msg::Marker::DELETE;
  delete_text.header.stamp = raw_node->get_clock()->now();

  visualization_msgs::msg::MarkerArray delete_array;
  delete_array.markers.push_back(delete_arrow);
  delete_array.markers.push_back(delete_text);
  preview_marker_pub_->publish(delete_array);
}

void GoalPoseAnnotatorPanel::onChooseOutputFile()
{
  const QString target = QFileDialog::getSaveFileName(
    this,
    "Select annotation YAML file",
    output_file_edit_->text(),
    "YAML files (*.yaml *.yml)");
  if (!target.isEmpty()) {
    output_file_edit_->setText(target);
    updateSuggestedEpisodeName();
  }
}

void GoalPoseAnnotatorPanel::onSaveEpisode()
{
  if (!captured_start_.has_value() || !captured_goal_.has_value()) {
    updateStatus("Capture both start and goal before saving.", true);
    return;
  }
  updateMapPackageFromSelectedPath(map_path_edit_->text());
  appendEpisodeToYaml();
  captured_start_.reset();
  captured_goal_.reset();
  updatePoseStatusLabel(captured_start_label_, "Captured /initialpose: ", captured_start_);
  updatePoseStatusLabel(captured_goal_label_, "Captured /goal_pose: ", captured_goal_);
  if (latest_start_.has_value()) {
    publishPosePreviewMarkers(*latest_start_, latest_start_frame_id_, true);
  }
  if (latest_goal_.has_value()) {
    publishPosePreviewMarkers(*latest_goal_, latest_goal_frame_id_, false);
  }
}

void GoalPoseAnnotatorPanel::onChooseMapPath()
{
  const QString target = QFileDialog::getOpenFileName(
    this,
    "Select map yaml",
    map_path_edit_->text(),
    "YAML files (*.yaml *.yml)");
  if (!target.isEmpty()) {
    const auto metadata = inferMapMetadataForYaml(target.toStdString());
    const auto & map_package = metadata.first;
    const auto & inferred_map_path = metadata.second;
    if (!map_package.empty()) {
      map_package_edit_->setText(QString::fromStdString(map_package));
    }
    if (!inferred_map_path.empty()) {
      map_path_edit_->setText(QString::fromStdString(inferred_map_path));
    } else {
      map_path_edit_->setText(target);
    }
    const std::string map_url = resolveMapUrl(map_path_edit_->text());
    if (map_url.empty()) {
      updateStatus("Could not resolve selected map path.");
      return;
    }
    if (!map_load_client_) {
      updateStatus("Map load client is unavailable.");
      return;
    }
    if (!map_load_client_->service_is_ready()) {
      if (!map_load_client_->wait_for_service(std::chrono::seconds(2))) {
        updateStatus("Could not reach /map_server/load_map", true);
        return;
      }
    }
    auto request = std::make_shared<nav2_msgs::srv::LoadMap::Request>();
    request->map_url = map_url;
    map_load_client_->async_send_request(
      request, [this](rclcpp::Client<nav2_msgs::srv::LoadMap>::SharedFuture future) {
        const auto response = future.get();
        if (!response) {
          QMetaObject::invokeMethod(
            this, [this]() { updateStatus("No response from map server", true); },
            Qt::QueuedConnection);
          return;
        }
        if (response->result == nav2_msgs::srv::LoadMap::Response::RESULT_SUCCESS) {
          QMetaObject::invokeMethod(
            this,
            [this]() {
              updateStatus(QString("Map loaded: %1").arg(map_path_edit_->text()));
            },
            Qt::QueuedConnection);
        } else {
          const auto result = response->result;
          QMetaObject::invokeMethod(
            this,
            [this, result]() {
              updateStatus(
                QString("Failed to load map (code: %1)").arg(
                  static_cast<int>(result)), true);
            },
            Qt::QueuedConnection);
        }
      });
    updateStatus(QString("Map load requested: %1").arg(map_path_edit_->text()));
  }
}

std::string GoalPoseAnnotatorPanel::resolveMapUrl(const QString & map_path) const
{
  const std::string selected_path = map_path.toStdString();
  if (selected_path.empty()) {
    return "";
  }
  if (selected_path.rfind("file://", 0) == 0) {
    if (selected_path.size() > 7) {
      return selected_path.substr(7);
    }
    return "";
  }
  if (selected_path.rfind("package://", 0) == 0) {
    // Keep package:// support for users typing it directly.
    // map_server usually expects local filesystem path.
    const std::string package_prefix = "package://";
    const auto slash_pos = selected_path.find('/', package_prefix.size());
    if (slash_pos == std::string::npos) {
      return selected_path;
    }
    const std::string package_name = selected_path.substr(package_prefix.size(), slash_pos - package_prefix.size());
    const std::string relative_path = selected_path.substr(slash_pos + 1);
    try {
      const std::string package_share = ament_index_cpp::get_package_share_directory(package_name);
      return (std::filesystem::path(package_share) / relative_path).string();
    } catch (...) {
      return selected_path;
    }
  }
  if (selected_path.rfind("/", 0) == 0) {
    return selected_path;
  }
  const std::string package_name = map_package_edit_->text().toStdString();
  if (!package_name.empty()) {
    try {
      const std::string package_share = ament_index_cpp::get_package_share_directory(package_name);
      const std::filesystem::path candidate = std::filesystem::path(package_share) / selected_path;
      if (std::filesystem::exists(candidate)) {
        return candidate.string();
      }
    } catch (...) {
      // Keep falling back to absolute path.
    }
  }
  return std::filesystem::absolute(selected_path).string();
}

std::string GoalPoseAnnotatorPanel::inferMapPackageFromPath(const std::string & map_path) const
{
  std::filesystem::path selected_map_path;
  try {
    selected_map_path = std::filesystem::weakly_canonical(map_path);
  } catch (...) {
    selected_map_path = std::filesystem::absolute(map_path);
  }
  if (!std::filesystem::exists(selected_map_path)) {
    return "";
  }
  const std::string package_root = findContainingPackageRoot(map_path);
  if (!package_root.empty()) {
    const std::filesystem::path package_xml = std::filesystem::path(package_root) / "package.xml";
    try {
      std::ifstream package_file(package_xml);
      if (package_file) {
      const std::string content((std::istreambuf_iterator<char>(package_file)),
        std::istreambuf_iterator<char>());
      std::regex package_regex("<name>\\s*([^<]+)\\s*</name>");
      std::smatch match;
      if (std::regex_search(content, match, package_regex)) {
        return match[1].str();
      }
      }
      return std::filesystem::path(package_root).filename().string();
    } catch (...) {
      // Fall back to ament index.
    }
  }
  try {
    const auto package_prefixes = ament_index_cpp::get_packages_with_prefixes();
    for (const auto & item : package_prefixes) {
      const auto & pkg = item.first;
      const std::filesystem::path pkg_share = std::filesystem::path(item.second) / "share" / pkg;
      auto canonical_share = std::filesystem::weakly_canonical(pkg_share);
      if (selected_map_path.string().find(canonical_share.string()) == 0) {
        return pkg;
      }
    }
  } catch (...) {
    return "";
  }
  return "";
}

std::string GoalPoseAnnotatorPanel::findContainingPackageRoot(const std::string & map_path) const
{
  std::filesystem::path current_path;
  try {
    current_path = std::filesystem::weakly_canonical(map_path);
  } catch (...) {
    current_path = std::filesystem::absolute(map_path);
  }
  if (!std::filesystem::exists(current_path)) {
    return "";
  }
  if (std::filesystem::is_regular_file(current_path)) {
    current_path = current_path.parent_path();
  }
  while (true) {
    if (std::filesystem::exists(current_path / "package.xml")) {
      return current_path.string();
    }
    const auto parent = current_path.parent_path();
    if (parent == current_path) {
      break;
    }
    current_path = parent;
  }
  return "";
}

void GoalPoseAnnotatorPanel::updateMapPackageFromSelectedPath(const QString & map_path)
{
  const auto [package_name, map_relative_path] = inferMapMetadataForYaml(map_path.toStdString());
  if (!package_name.empty()) {
    map_package_edit_->setText(QString::fromStdString(package_name));
  }
  if (!map_relative_path.empty()) {
    map_path_edit_->setText(QString::fromStdString(map_relative_path));
  }
}

std::pair<std::string, std::string> GoalPoseAnnotatorPanel::inferMapMetadataForYaml(
  const std::string & map_path) const
{
  const std::string raw_map_path = map_path;
  const QString resolved_map_path_qs = QString::fromStdString(resolveMapUrl(QString::fromStdString(map_path)));
  if (resolved_map_path_qs.isEmpty()) {
    return {"", raw_map_path};
  }
  const std::string resolved_map_path = resolved_map_path_qs.toStdString();
  const std::string inferred_package = inferMapPackageFromPath(resolved_map_path);
  const std::string explicit_package = map_package_edit_->text().toStdString();

  const std::string package_name =
    inferred_package.empty() ? explicit_package : inferred_package;
  if (package_name.empty()) {
    return {"", raw_map_path};
  }

  try {
    std::string normalized_map_path = resolved_map_path;
    try {
      if (std::filesystem::exists(resolved_map_path)) {
        normalized_map_path = std::filesystem::weakly_canonical(resolved_map_path).string();
      } else {
        normalized_map_path = std::filesystem::absolute(resolved_map_path).string();
      }
    } catch (...) {
      normalized_map_path = std::filesystem::absolute(resolved_map_path).string();
    }

    const std::string package_root = findContainingPackageRoot(resolved_map_path);
    if (!package_root.empty()) {
      std::string normalized_package_root = package_root;
      try {
        normalized_package_root = std::filesystem::weakly_canonical(package_root).string();
      } catch (...) {
        // Ignore canonicalization failure, use raw package root.
      }

      const std::filesystem::path relative_map_path = std::filesystem::relative(
        normalized_map_path,
        normalized_package_root);
      const std::string relative_path_str = relative_map_path.string();
      if (!relative_path_str.empty() &&
        relative_path_str.rfind("..", 0) != 0 &&
        relative_path_str != ".")
      {
        return {package_name, relative_path_str};
      }
    }

    const std::string package_share = ament_index_cpp::get_package_share_directory(package_name);
    const std::filesystem::path resolved_map = std::filesystem::absolute(normalized_map_path);
    const std::filesystem::path package_share_path = std::filesystem::weakly_canonical(package_share);
    const std::filesystem::path relative_map_path = std::filesystem::relative(resolved_map, package_share_path);
    const std::string relative_path_str = relative_map_path.string();
    if (!relative_path_str.empty() &&
      relative_path_str.rfind("..", 0) != 0 &&
      relative_path_str != ".")
    {
      return {package_name, relative_path_str};
    }
  } catch (...) {
    // Fallback to user-provided values.
  }

  return {package_name, raw_map_path};
}

void GoalPoseAnnotatorPanel::appendEpisodeToYaml()
{
  std::string episode_name = episode_name_edit_->text().toStdString();
  if (episode_name.empty()) {
    episode_name = nextSuggestedEpisodeName();
  }

  if (!std::regex_match(episode_name, std::regex("^episode_[A-Za-z0-9_]+$"))) {
    updateStatus("Episode name should be non-empty and look like episode_1 or episode_custom", true);
    return;
  }
  const auto [map_package, map_path] = inferMapMetadataForYaml(map_path_edit_->text().toStdString());

  YAML::Node root = getYamlOrDefault();

  YAML::Node episodes = root["episodes"];
  if (!episodes.IsSequence()) {
    episodes = YAML::Node(YAML::NodeType::Sequence);
  }
  for (const auto & item : episodes) {
    if (item.as<std::string>() == episode_name) {
      updateStatus(QString("Episode already exists: %1").arg(QString::fromStdString(episode_name)), true);
      return;
    }
  }
  episodes.push_back(episode_name);
  root["episodes"] = episodes;

  if (!root["goals"].IsMap()) {
    root["goals"] = YAML::Node(YAML::NodeType::Map);
  }
  if (!root["poses"].IsMap()) {
    root["poses"] = YAML::Node(YAML::NodeType::Map);
  }

  root["robot_name"] = robot_name_edit_->text().toStdString();
  root["map_path"] = map_path;
  if (!map_package.empty()) {
    root["map_package"] = map_package;
  } else if (!map_package_edit_->text().isEmpty()) {
    root["map_package"] = map_package_edit_->text().toStdString();
  }
  YAML::Node initial_pose(YAML::NodeType::Sequence);
  initial_pose.push_back(toThreeDecimalNode(init_x_edit_->value()));
  initial_pose.push_back(toThreeDecimalNode(init_y_edit_->value()));
  initial_pose.push_back(toThreeDecimalNode(init_yaw_edit_->value()));
  initial_pose.SetStyle(YAML::EmitterStyle::Flow);
  root["initial_pose"] = initial_pose;

  YAML::Node goal_pose(YAML::NodeType::Sequence);
  goal_pose.push_back(toThreeDecimalNode(captured_goal_->x));
  goal_pose.push_back(toThreeDecimalNode(captured_goal_->y));
  goal_pose.push_back(toThreeDecimalNode(captured_goal_->yaw));
  goal_pose.SetStyle(YAML::EmitterStyle::Flow);
  root["goals"][episode_name] = goal_pose;

  YAML::Node start_pose(YAML::NodeType::Sequence);
  start_pose.push_back(toThreeDecimalNode(captured_start_->x));
  start_pose.push_back(toThreeDecimalNode(captured_start_->y));
  start_pose.push_back(toThreeDecimalNode(captured_start_->yaw));
  start_pose.SetStyle(YAML::EmitterStyle::Flow);
  root["poses"][episode_name] = start_pose;

  saveYamlFile(root);

  const QString saved_msg = QString("Saved %1 to %2").arg(
    QString::fromStdString(episode_name), output_file_edit_->text());
  updateStatus(saved_msg);
  updateSuggestedEpisodeName();
}

YAML::Node GoalPoseAnnotatorPanel::getYamlOrDefault()
{
  const std::string output_path = output_file_edit_->text().toStdString();
  if (!output_path.empty() && std::filesystem::exists(output_path)) {
    try {
      return YAML::LoadFile(output_path);
    } catch (...) {
      updateStatus(QString("Could not read existing YAML, creating new file: %1")
          .arg(QString::fromStdString(output_path)), true);
    }
  }

  updateMapPackageFromSelectedPath(map_path_edit_->text());

  YAML::Node root;
  root["robot_name"] = robot_name_edit_->text().toStdString();
  const auto [map_package, map_path] = inferMapMetadataForYaml(map_path_edit_->text().toStdString());
  root["initial_pose"] = YAML::Node();
  root["map_path"] = map_path;
  if (!map_package.empty()) {
    root["map_package"] = map_package;
  } else if (!map_package_edit_->text().isEmpty()) {
    root["map_package"] = map_package_edit_->text().toStdString();
  }
  root["episodes"] = YAML::Node(YAML::NodeType::Sequence);
  root["goals"] = YAML::Node(YAML::NodeType::Map);
  root["poses"] = YAML::Node(YAML::NodeType::Map);
  return root;
}

void GoalPoseAnnotatorPanel::saveYamlFile(const YAML::Node & root) const
{
  const std::string output_path = output_file_edit_->text().toStdString();
  std::filesystem::path output_file(output_path);
  if (!output_file.parent_path().empty()) {
    std::filesystem::create_directories(output_file.parent_path());
  }
  std::ofstream out(output_path);
  out << root;
}

void GoalPoseAnnotatorPanel::updatePoseStatusLabel(
  QLabel * label, const std::string & text, const std::optional<Pose2D> & pose)
{
  if (!pose.has_value()) {
    label->setText("n/a");
    return;
  }
  const Pose2D & p = *pose;
  label->setText(QString("%1x=%2 y=%3 yaw=%4")
    .arg(QString::fromStdString(text))
    .arg(QString::number(p.x, 'f', 3))
    .arg(QString::number(p.y, 'f', 3))
    .arg(QString::number(p.yaw, 'f', 3)));
}

void GoalPoseAnnotatorPanel::updateStatus(const QString & text, bool is_error)
{
  status_label_->setText(text);
  status_label_->setStyleSheet(is_error ? "color: #d32f2f;" : "color: #2e7d32;");
}

std::string GoalPoseAnnotatorPanel::nextSuggestedEpisodeName() const
{
  const std::string output_path = output_file_edit_->text().toStdString();
  int max_idx = 0;
  if (output_path.empty() || !std::filesystem::exists(output_path)) {
    return "episode_1";
  }

  try {
    const YAML::Node root = YAML::LoadFile(output_path);
    if (root["episodes"] && root["episodes"].IsSequence()) {
      for (const auto & episode_node : root["episodes"]) {
        if (!episode_node.IsScalar()) {
          continue;
        }
        const std::string episode = episode_node.as<std::string>();
        std::smatch match;
        std::regex regex("^episode_(\\d+)$");
        if (std::regex_match(episode, match, regex)) {
          max_idx = std::max(max_idx, std::stoi(match[1].str()));
        }
      }
    }
  } catch (...) {
    return "episode_1";
  }

  return "episode_" + std::to_string(max_idx + 1);
}

void GoalPoseAnnotatorPanel::updateSuggestedEpisodeName()
{
  episode_name_edit_->setText(QString::fromStdString(nextSuggestedEpisodeName()));
}

}  // namespace rviz_goal_pose_annotator

PLUGINLIB_EXPORT_CLASS(
  rviz_goal_pose_annotator::GoalPoseAnnotatorPanel,
  rviz_common::Panel
)
