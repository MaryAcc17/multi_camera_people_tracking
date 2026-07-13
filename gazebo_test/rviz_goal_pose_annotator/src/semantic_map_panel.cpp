#include "rviz_goal_pose_annotator/semantic_map_panel.hpp"

#include <algorithm>
#include <fstream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <QAbstractSpinBox>
#include <QFileDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QMetaObject>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <ament_index_cpp/get_packages_with_prefixes.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rviz_common/display_context.hpp>
#include <tf2/utils.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace rviz_goal_pose_annotator
{

namespace
{

std::string toLowerCopy(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), ::tolower);
  return value;
}

YAML::Node toThreeDecimalNode(double value)
{
  std::ostringstream stream;
  stream.setf(std::ios::fixed);
  stream << std::setprecision(3) << value;
  return YAML::Load(stream.str());
}

}  // namespace

SemanticMapPanel::SemanticMapPanel(QWidget * parent)
: rviz_common::Panel(parent)
{
  setupUi();
}

void SemanticMapPanel::onInitialize()
{
  node_ = getDisplayContext()->getRosNodeAbstraction().lock();
  if (!node_) {
    return;
  }
  setupSubscriptions();
  const auto raw_node = node_->get_raw_node();
  if (!raw_node) {
    return;
  }
  map_load_client_ = raw_node->create_client<nav2_msgs::srv::LoadMap>("/map_server/load_map");
  extent_preview_pub_ = raw_node->create_publisher<visualization_msgs::msg::MarkerArray>(
    "/semantic_map_annotator/extent_preview",
    rclcpp::QoS(10));
}

void SemanticMapPanel::setupSubscriptions()
{
  if (!node_) {
    return;
  }
  const auto raw_node = node_->get_raw_node();
  if (!raw_node) {
    return;
  }
  goal_sub_ = raw_node->create_subscription<geometry_msgs::msg::PoseStamped>(
    "/goal_pose", rclcpp::QoS(10),
    std::bind(&SemanticMapPanel::goalPoseCallback, this, std::placeholders::_1));
  clicked_point_sub_ = raw_node->create_subscription<geometry_msgs::msg::PointStamped>(
    "/clicked_point", rclcpp::QoS(10),
    std::bind(&SemanticMapPanel::clickedPointCallback, this, std::placeholders::_1));
}

Pose2D SemanticMapPanel::poseFromPoseStamped(const geometry_msgs::msg::PoseStamped & msg) const
{
  Pose2D pose;
  pose.x = msg.pose.position.x;
  pose.y = msg.pose.position.y;
  pose.yaw = tf2::getYaw(msg.pose.orientation);
  return pose;
}

void SemanticMapPanel::goalPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  if (!msg) {
    return;
  }
  latest_goal_ = poseFromPoseStamped(*msg);
}

void SemanticMapPanel::clickedPointCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg)
{
  if (!msg) {
    return;
  }
  Pose2D clicked;
  clicked.x = msg->point.x;
  clicked.y = msg->point.y;
  clicked.yaw = 0.0;
  latest_clicked_point_ = clicked;
  if (!msg->header.frame_id.empty()) {
    clicked_point_frame_id_ = msg->header.frame_id;
  }
}

void SemanticMapPanel::setupUi()
{
  auto * main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(8, 8, 8, 8);

  auto * metadata_group = new QGroupBox("Map metadata", this);
  auto * metadata_layout = new QVBoxLayout(metadata_group);

  map_package_edit_ = new QLineEdit(this);
  map_package_edit_->setPlaceholderText("auto-detected from map path");
  map_path_edit_ = new QLineEdit("maps/social_indoor_map.yaml", this);
  choose_map_file_btn_ = new QPushButton("Choose map path", this);
  connect(choose_map_file_btn_, &QPushButton::clicked, this, &SemanticMapPanel::onChooseMapPath);
  connect(map_path_edit_, &QLineEdit::editingFinished, this,
    [this]() {
      updateMapPackageFromSelectedPath(map_path_edit_->text());
      requestLoadMap(map_path_edit_->text());
      loadYamlPreview();
    });

  auto * map_path_row = new QHBoxLayout();
  map_path_row->addWidget(map_path_edit_);
  map_path_row->addWidget(choose_map_file_btn_);

  metadata_layout->addWidget(new QLabel("map_package", this));
  metadata_layout->addWidget(map_package_edit_);
  metadata_layout->addWidget(new QLabel("map_path", this));
  metadata_layout->addLayout(map_path_row);
  main_layout->addWidget(metadata_group);

  auto * output_group = new QGroupBox("YAML output", this);
  auto * output_layout = new QVBoxLayout(output_group);
  output_file_edit_ = new QLineEdit(
    std::filesystem::path(std::filesystem::current_path() / "semantic_map_output.yaml")
    .string().c_str(),
    this);
  choose_file_btn_ = new QPushButton("Choose output file", this);
  connect(choose_file_btn_, &QPushButton::clicked, this, &SemanticMapPanel::onChooseOutputFile);
  connect(output_file_edit_, &QLineEdit::editingFinished, this,
    &SemanticMapPanel::onOutputFileChanged);
  auto * output_file_row = new QHBoxLayout();
  output_file_row->addWidget(output_file_edit_);
  output_file_row->addWidget(choose_file_btn_);
  output_layout->addLayout(output_file_row);
  main_layout->addWidget(output_group);

  auto * label_group = new QGroupBox("Labels", this);
  auto * label_layout = new QVBoxLayout(label_group);
  labels_list_ = new QListWidget(this);
  new_label_edit_ = new QLineEdit(this);
  new_label_edit_->setPlaceholderText("new label (e.g. kitchen)");
  add_label_btn_ = new QPushButton("Add label", this);
  connect(add_label_btn_, &QPushButton::clicked, this, &SemanticMapPanel::onAddLabel);
  auto * label_input_row = new QHBoxLayout();
  label_input_row->addWidget(new_label_edit_);
  label_input_row->addWidget(add_label_btn_);

  label_layout->addWidget(labels_list_);
  label_layout->addLayout(label_input_row);
  main_layout->addWidget(label_group);

  auto * entry_group = new QGroupBox("Add semantic element", this);
  auto * entry_layout = new QVBoxLayout(entry_group);
  entry_name_edit_ = new QLineEdit("room_1", this);
  entry_type_combo_ = new QComboBox(this);
  entry_type_combo_->addItems({"room", "object", "corridor"});
  x_min_edit_ = new QDoubleSpinBox(this);
  y_min_edit_ = new QDoubleSpinBox(this);
  x_max_edit_ = new QDoubleSpinBox(this);
  y_max_edit_ = new QDoubleSpinBox(this);
  entry_x_edit_ = new QDoubleSpinBox(this);
  entry_y_edit_ = new QDoubleSpinBox(this);
  entry_yaw_edit_ = new QDoubleSpinBox(this);
  for (auto spin : {x_min_edit_, y_min_edit_, x_max_edit_, y_max_edit_,
                   entry_x_edit_, entry_y_edit_, entry_yaw_edit_}) {
    spin->setRange(-99999.0, 99999.0);
    spin->setDecimals(3);
    spin->setSingleStep(0.05);
    spin->setReadOnly(true);
    spin->setButtonSymbols(QAbstractSpinBox::NoButtons);
  }

  add_entry_btn_ = new QPushButton("Add / append map entry", this);
  connect(add_entry_btn_, &QPushButton::clicked, this, &SemanticMapPanel::onAddEntry);
  capture_pose_btn_ = new QPushButton("Capture pose from /goal_pose", this);
  capture_extent_corner1_btn_ = new QPushButton("Set extent corner 1 (from /clicked_point)", this);
  capture_extent_corner2_btn_ = new QPushButton("Set extent corner 2 (from /clicked_point)", this);
  connect(capture_pose_btn_, &QPushButton::clicked, this, &SemanticMapPanel::onCapturePose);
  connect(capture_extent_corner1_btn_, &QPushButton::clicked, this, &SemanticMapPanel::onCaptureExtentCornerOne);
  connect(capture_extent_corner2_btn_, &QPushButton::clicked, this, &SemanticMapPanel::onCaptureExtentCornerTwo);
  pose_status_label_ = new QLabel("Pose: not captured", this);
  extent_status_label_ = new QLabel("Extent: not defined", this);

  entry_layout->addWidget(new QLabel("name", this));
  entry_layout->addWidget(entry_name_edit_);
  entry_layout->addWidget(new QLabel("type", this));
  entry_layout->addWidget(entry_type_combo_);

  auto * extent_row_1 = new QHBoxLayout();
  auto * extent_row_2 = new QHBoxLayout();
  extent_row_1->addWidget(capture_extent_corner1_btn_);
  extent_row_2->addWidget(capture_extent_corner2_btn_);
  extent_row_1->addWidget(pose_status_label_);
  extent_row_2->addWidget(extent_status_label_);
  extent_row_1->addWidget(capture_pose_btn_);
  extent_row_1->addWidget(new QLabel("extent x_min", this));
  extent_row_1->addWidget(x_min_edit_);
  extent_row_2->addWidget(new QLabel("extent y_min", this));
  extent_row_2->addWidget(y_min_edit_);
  extent_row_1->addWidget(new QLabel("extent x_max", this));
  extent_row_1->addWidget(x_max_edit_);
  extent_row_2->addWidget(new QLabel("extent y_max", this));
  extent_row_2->addWidget(y_max_edit_);
  entry_layout->addLayout(extent_row_1);
  entry_layout->addLayout(extent_row_2);

  auto * pose_row_1 = new QHBoxLayout();
  auto * pose_row_2 = new QHBoxLayout();
  pose_row_1->addWidget(new QLabel("pose x", this));
  pose_row_1->addWidget(entry_x_edit_);
  pose_row_1->addWidget(new QLabel("pose y", this));
  pose_row_1->addWidget(entry_y_edit_);
  pose_row_2->addWidget(new QLabel("pose yaw", this));
  pose_row_2->addWidget(entry_yaw_edit_);
  entry_layout->addLayout(pose_row_1);
  entry_layout->addLayout(pose_row_2);
  entry_layout->addWidget(add_entry_btn_);
  main_layout->addWidget(entry_group);

  status_label_ = new QLabel("Ready", this);
  status_label_->setStyleSheet("color: #2e7d32;");
  main_layout->addWidget(status_label_);
  main_layout->addStretch();

  loadYamlPreview();
}

void SemanticMapPanel::onOutputFileChanged()
{
  loadYamlPreview();
}

void SemanticMapPanel::loadYamlPreview()
{
  const std::string output_path = outputPath();
  if (output_path.empty() || !std::filesystem::exists(output_path)) {
    labels_list_->clear();
    return;
  }
  try {
    YAML::Node root = YAML::LoadFile(output_path);
    if (root["labels"] && root["labels"].IsSequence()) {
      refreshLabelsFromYaml(root);
      const std::string package = root["map_package"] ? root["map_package"].as<std::string>() : "";
      const std::string map_path = root["map_path"] ? root["map_path"].as<std::string>() : "";
      if (!package.empty()) {
        map_package_edit_->setText(QString::fromStdString(package));
      }
      if (!map_path.empty()) {
        map_path_edit_->setText(QString::fromStdString(map_path));
      }
    }
  } catch (...) {
    labels_list_->clear();
    updateStatus("Could not read existing YAML file", true);
  }
}

void SemanticMapPanel::onCapturePose()
{
  if (!latest_goal_.has_value()) {
    updateStatus("No /goal_pose received yet", true);
    return;
  }
  const Pose2D & pose = *latest_goal_;
  has_captured_pose_ = true;
  entry_x_edit_->setValue(pose.x);
  entry_y_edit_->setValue(pose.y);
  entry_yaw_edit_->setValue(pose.yaw);
  pose_status_label_->setText(
    QString("Pose: %1, %2, %3")
      .arg(QString::number(pose.x, 'f', 3))
      .arg(QString::number(pose.y, 'f', 3))
      .arg(QString::number(pose.yaw, 'f', 3)));
  updateStatus("Pose captured from /goal_pose");
}

void SemanticMapPanel::onCaptureExtentCornerOne()
{
  if (!latest_clicked_point_.has_value()) {
    updateStatus("No /clicked_point received yet", true);
    return;
  }
  extent_corner_a_ = latest_clicked_point_;
  extent_corner_b_.reset();
  has_captured_extent_ = false;
  extent_status_label_->setText("Extent: corner 1 captured");
  clearExtentPreview();
  updateStatus("Extent corner 1 captured from /clicked_point");
}

void SemanticMapPanel::onCaptureExtentCornerTwo()
{
  if (!latest_clicked_point_.has_value()) {
    updateStatus("No /clicked_point received yet", true);
    return;
  }
  if (!extent_corner_a_.has_value()) {
    updateStatus("Capture extent corner 1 first (from /clicked_point)", true);
    return;
  }
  extent_corner_b_ = latest_clicked_point_;
  updateExtentFromCorners();
  has_captured_extent_ = true;
  updateStatus("Extent captured from /clicked_point corners");
}

void SemanticMapPanel::updateExtentFromCorners()
{
  if (!extent_corner_a_.has_value() || !extent_corner_b_.has_value()) {
    return;
  }
  const Pose2D & a = *extent_corner_a_;
  const Pose2D & b = *extent_corner_b_;
  const double x_min = std::min(a.x, b.x);
  const double y_min = std::min(a.y, b.y);
  const double x_max = std::max(a.x, b.x);
  const double y_max = std::max(a.y, b.y);
  x_min_edit_->setValue(x_min);
  y_min_edit_->setValue(y_min);
  x_max_edit_->setValue(x_max);
  y_max_edit_->setValue(y_max);
  extent_status_label_->setText(
    QString("Extent: [%1, %2, %3, %4], area=%5")
      .arg(QString::number(x_min, 'f', 3))
      .arg(QString::number(y_min, 'f', 3))
      .arg(QString::number(x_max, 'f', 3))
      .arg(QString::number(y_max, 'f', 3))
      .arg(QString::number(std::abs((x_max - x_min) * (y_max - y_min)), 'f', 3)));
  publishExtentPreview();
}

void SemanticMapPanel::publishExtentPreview()
{
  if (!extent_preview_pub_ || !extent_corner_a_.has_value() || !extent_corner_b_.has_value()) {
    return;
  }
  const auto raw_node = node_ ? node_->get_raw_node() : nullptr;
  if (!raw_node) {
    return;
  }

  const Pose2D & a = *extent_corner_a_;
  const Pose2D & b = *extent_corner_b_;
  const double x_min = std::min(a.x, b.x);
  const double y_min = std::min(a.y, b.y);
  const double x_max = std::max(a.x, b.x);
  const double y_max = std::max(a.y, b.y);
  const double area = std::abs((x_max - x_min) * (y_max - y_min));
  const std::string frame_id = clicked_point_frame_id_.empty() ? "map" : clicked_point_frame_id_;

  visualization_msgs::msg::Marker outline;
  outline.header.frame_id = frame_id;
  outline.header.stamp = raw_node->get_clock()->now();
  outline.ns = "semantic_map_extent";
  outline.id = 0;
  outline.type = visualization_msgs::msg::Marker::LINE_STRIP;
  outline.action = visualization_msgs::msg::Marker::ADD;
  outline.pose.orientation.w = 1.0;
  outline.scale.x = 0.05;
  outline.color.g = 1.0;
  outline.color.a = 0.9;

  geometry_msgs::msg::Point p1;
  p1.z = 0.02;
  p1.x = x_min;
  p1.y = y_min;
  outline.points.push_back(p1);
  p1.x = x_min;
  p1.y = y_max;
  outline.points.push_back(p1);
  p1.x = x_max;
  p1.y = y_max;
  outline.points.push_back(p1);
  p1.x = x_max;
  p1.y = y_min;
  outline.points.push_back(p1);
  outline.points.push_back(outline.points.front());

  visualization_msgs::msg::Marker area_label;
  area_label.header.frame_id = frame_id;
  area_label.header.stamp = raw_node->get_clock()->now();
  area_label.ns = "semantic_map_extent";
  area_label.id = 1;
  area_label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  area_label.action = visualization_msgs::msg::Marker::ADD;
  area_label.pose.position.x = (x_min + x_max) * 0.5;
  area_label.pose.position.y = (y_min + y_max) * 0.5;
  area_label.pose.position.z = 0.25;
  area_label.pose.orientation.w = 1.0;
  area_label.scale.z = 0.30;
  area_label.color.r = 1.0;
  area_label.color.g = 1.0;
  area_label.color.b = 1.0;
  area_label.color.a = 1.0;
  std::ostringstream area_text;
  area_text.setf(std::ios::fixed);
  area_text << std::setprecision(3) << area;
  area_label.text = "Area: " + area_text.str() + " m^2";

  visualization_msgs::msg::MarkerArray markers;
  markers.markers.push_back(outline);
  markers.markers.push_back(area_label);
  extent_preview_pub_->publish(markers);
}

void SemanticMapPanel::clearExtentPreview()
{
  if (!extent_preview_pub_) {
    return;
  }
  const auto raw_node = node_ ? node_->get_raw_node() : nullptr;
  if (!raw_node) {
    return;
  }

  visualization_msgs::msg::MarkerArray markers;
  for (int id = 0; id < 2; ++id) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = clicked_point_frame_id_.empty() ? "map" : clicked_point_frame_id_;
    marker.header.stamp = raw_node->get_clock()->now();
    marker.ns = "semantic_map_extent";
    marker.id = id;
    marker.action = visualization_msgs::msg::Marker::DELETE;
    markers.markers.push_back(marker);
  }
  extent_preview_pub_->publish(markers);
}

void SemanticMapPanel::refreshLabelsFromYaml(const YAML::Node & root)
{
  labels_list_->clear();
  if (!root["labels"] || !root["labels"].IsSequence()) {
    return;
  }
  for (const auto & label : root["labels"]) {
    if (!label.IsScalar()) {
      continue;
    }
    labels_list_->addItem(QString::fromStdString(label.as<std::string>()));
  }
}

void SemanticMapPanel::onChooseMapPath()
{
  const QString target = QFileDialog::getOpenFileName(
    this,
    "Select map yaml",
    map_path_edit_->text(),
    "YAML files (*.yaml *.yml)");
  if (target.isEmpty()) {
    return;
  }

  map_path_edit_->setText(target);
  updateMapPackageFromSelectedPath(target);
  requestLoadMap(target);
  loadYamlPreview();
}

void SemanticMapPanel::requestLoadMap(const QString & map_path)
{
  if (!map_load_client_) {
    updateStatus("Map load client is unavailable.");
    return;
  }
  const std::string map_url = resolveMapPath(map_path);
  if (map_url.empty()) {
    updateStatus("Could not resolve selected map path.", true);
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
          this, [this]() { updateStatus("Map loaded via /map_server/load_map"); },
          Qt::QueuedConnection);
      } else {
        const auto result = response->result;
        QMetaObject::invokeMethod(
          this,
          [this, result]() {
            updateStatus(QString("Failed to load map (code: %1)").arg(static_cast<int>(result)), true);
          },
          Qt::QueuedConnection);
      }
    });
  updateStatus(QString("Map load requested: %1").arg(QString::fromStdString(map_url)));
}

void SemanticMapPanel::onChooseOutputFile()
{
  const QString target = QFileDialog::getSaveFileName(
    this,
    "Select semantic map YAML file",
    output_file_edit_->text(),
    "YAML files (*.yaml *.yml)");
  if (!target.isEmpty()) {
    output_file_edit_->setText(target);
    loadYamlPreview();
  }
}

void SemanticMapPanel::onAddLabel()
{
  const std::string label = trim(new_label_edit_->text().toStdString());
  if (label.empty()) {
    updateStatus("Enter a valid label", true);
    return;
  }
  auto root = getYamlOrDefault();
  if (!root["labels"].IsSequence()) {
    root["labels"] = YAML::Node(YAML::NodeType::Sequence);
  }
  if (labelExistsInNode(root["labels"], label)) {
    updateStatus(QString("Label already exists: %1").arg(QString::fromStdString(label)), true);
    return;
  }
  root["labels"].push_back(label);
  saveYamlFile(root);
  new_label_edit_->clear();
  refreshLabelsFromYaml(root);
  updateStatus(QString("Added label: %1").arg(QString::fromStdString(label)));
}

void SemanticMapPanel::onAddEntry()
{
  const std::string name = trim(entry_name_edit_->text().toStdString());
  if (name.empty()) {
    updateStatus("Entry name cannot be empty", true);
    return;
  }
  if (!has_captured_pose_) {
    updateStatus("Capture pose from /goal_pose before adding entry", true);
    return;
  }

  const std::string type = toLowerCopy(trim(entry_type_combo_->currentText().toStdString()));
  if (type != "object" && !has_captured_extent_) {
    updateStatus("Capture both extent corners from /clicked_point for room/corridor", true);
    return;
  }

  auto root = getYamlOrDefault();
  if (!root["map"].IsSequence()) {
    root["map"] = YAML::Node(YAML::NodeType::Sequence);
  }

  YAML::Node new_entry(YAML::NodeType::Map);
  new_entry["name"] = name;
  new_entry["type"] = type;

  if (type != "object") {
    YAML::Node extent(YAML::NodeType::Sequence);
    extent.push_back(toThreeDecimalNode(x_min_edit_->value()));
    extent.push_back(toThreeDecimalNode(y_min_edit_->value()));
    extent.push_back(toThreeDecimalNode(x_max_edit_->value()));
    extent.push_back(toThreeDecimalNode(y_max_edit_->value()));
    extent.SetStyle(YAML::EmitterStyle::Flow);
    new_entry["extent"] = extent;
  }

  YAML::Node pose(YAML::NodeType::Sequence);
  pose.push_back(toThreeDecimalNode(entry_x_edit_->value()));
  pose.push_back(toThreeDecimalNode(entry_y_edit_->value()));
  pose.push_back(toThreeDecimalNode(entry_yaw_edit_->value()));
  pose.SetStyle(YAML::EmitterStyle::Flow);
  new_entry["pose"] = pose;

  root["map"].push_back(new_entry);
  saveYamlFile(root);

  const std::string filename = outputPath();
  updateStatus(QString("Added %1 '%2' to %3")
      .arg(QString::fromStdString(type))
      .arg(QString::fromStdString(name))
      .arg(QString::fromStdString(filename)));

  has_captured_pose_ = false;
  has_captured_extent_ = false;
  extent_corner_a_.reset();
  extent_corner_b_.reset();
  clearExtentPreview();
  pose_status_label_->setText("Pose: not captured");
  extent_status_label_->setText("Extent: not defined");
}

void SemanticMapPanel::updateStatus(const QString & text, bool is_error)
{
  status_label_->setText(text);
  status_label_->setStyleSheet(is_error ? "color: #d32f2f;" : "color: #2e7d32;");
}

std::string SemanticMapPanel::trim(const std::string & value) const
{
  const auto begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return "";
  }
  const auto end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

std::string SemanticMapPanel::outputPath() const
{
  return output_file_edit_->text().toStdString();
}

std::string SemanticMapPanel::resolveMapPath(const QString & map_path) const
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
  if (selected_path.rfind("/", 0) == 0) {
    return selected_path;
  }
  if (selected_path.rfind("package://", 0) == 0) {
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
  const std::string package_name = map_package_edit_->text().toStdString();
  if (!package_name.empty()) {
    try {
      const std::string package_share = ament_index_cpp::get_package_share_directory(package_name);
      const std::filesystem::path candidate = std::filesystem::path(package_share) / selected_path;
      if (std::filesystem::exists(candidate)) {
        return candidate.string();
      }
    } catch (...) {
      // Keep absolute fallback.
    }
  }
  return std::filesystem::absolute(selected_path).string();
}

std::string SemanticMapPanel::findContainingPackageRoot(const std::string & map_path) const
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

std::string SemanticMapPanel::inferMapPackageFromPath(const std::string & map_path) const
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
      // Fallback to ament index.
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

std::pair<std::string, std::string> SemanticMapPanel::inferMapMetadataForYaml(
  const std::string & map_path) const
{
  const std::string raw_map_path = map_path;
  const QString resolved_map_path_qs = QString::fromStdString(resolveMapPath(QString::fromStdString(map_path)));
  if (resolved_map_path_qs.isEmpty()) {
    return {"", raw_map_path};
  }
  const std::string resolved_map_path = resolved_map_path_qs.toStdString();
  const std::string inferred_package = inferMapPackageFromPath(resolved_map_path);
  const std::string explicit_package = map_package_edit_->text().toStdString();

  const std::string package_name = inferred_package.empty() ? explicit_package : inferred_package;
  if (package_name.empty()) {
    return {"", raw_map_path};
  }

  try {
    std::string normalized_map_path = resolved_map_path;
    if (std::filesystem::exists(resolved_map_path)) {
      normalized_map_path = std::filesystem::weakly_canonical(resolved_map_path).string();
    } else {
      normalized_map_path = std::filesystem::absolute(resolved_map_path).string();
    }
    const std::string package_root = findContainingPackageRoot(resolved_map_path);
    if (!package_root.empty()) {
      const std::filesystem::path relative_map_path = std::filesystem::relative(
        std::filesystem::weakly_canonical(std::filesystem::path(normalized_map_path)),
        std::filesystem::weakly_canonical(std::filesystem::path(package_root)));
      const std::string relative_path_str = relative_map_path.string();
      if (!relative_path_str.empty() &&
        relative_path_str.rfind("..", 0) != 0 &&
        relative_path_str != ".")
      {
        return {package_name, relative_path_str};
      }
    }

    const std::string package_share = ament_index_cpp::get_package_share_directory(package_name);
    const std::filesystem::path resolved_map = std::filesystem::weakly_canonical(normalized_map_path);
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
    // Ignore and keep fallback.
  }
  return {package_name, raw_map_path};
}

void SemanticMapPanel::updateMapPackageFromSelectedPath(const QString & map_path)
{
  const auto & [package_name, inferred_map_path] = inferMapMetadataForYaml(map_path.toStdString());
  if (!package_name.empty()) {
    map_package_edit_->setText(QString::fromStdString(package_name));
  }
  if (!inferred_map_path.empty()) {
    map_path_edit_->setText(QString::fromStdString(inferred_map_path));
  }
}

void SemanticMapPanel::saveYamlFile(const YAML::Node & root) const
{
  const std::string output_path = outputPath();
  if (output_path.empty()) {
    throw std::runtime_error("output path is empty");
  }
  std::filesystem::path output_file(output_path);
  if (!output_file.parent_path().empty()) {
    std::filesystem::create_directories(output_file.parent_path());
  }
  std::ofstream out(output_path);
  out << root;
}

bool SemanticMapPanel::labelExistsInNode(const YAML::Node & labels, const std::string & label) const
{
  if (!labels.IsSequence()) {
    return false;
  }
  const std::string target = toLowerCopy(label);
  for (const auto & current : labels) {
    if (!current.IsScalar()) {
      continue;
    }
    if (toLowerCopy(current.as<std::string>()) == target) {
      return true;
    }
  }
  return false;
}

YAML::Node SemanticMapPanel::getYamlOrDefault()
{
  const std::string output_path = outputPath();
  YAML::Node root;
  if (!output_path.empty() && std::filesystem::exists(output_path)) {
    try {
      root = YAML::LoadFile(output_path);
    } catch (...) {
      updateStatus("Could not read file; creating a new one", true);
    }
  }

  if (!root || !root.IsMap()) {
    const auto metadata = inferMapMetadataForYaml(map_path_edit_->text().toStdString());
    root["map_package"] = metadata.first;
    root["map_path"] = metadata.second;
    root["labels"] = YAML::Node(YAML::NodeType::Sequence);
    root["map"] = YAML::Node(YAML::NodeType::Sequence);
  } else {
    const auto metadata = inferMapMetadataForYaml(map_path_edit_->text().toStdString());
    if (metadata.first.empty()) {
      root["map_package"] = root["map_package"];
    } else {
      root["map_package"] = metadata.first;
    }
    if (!metadata.second.empty()) {
      root["map_path"] = metadata.second;
    }
    if (!root["labels"].IsSequence()) {
      root["labels"] = YAML::Node(YAML::NodeType::Sequence);
    }
    if (!root["map"].IsSequence()) {
      root["map"] = YAML::Node(YAML::NodeType::Sequence);
    }
  }
  refreshLabelsFromYaml(root);
  return root;
}

PLUGINLIB_EXPORT_CLASS(
  rviz_goal_pose_annotator::SemanticMapPanel,
  rviz_common::Panel
)

}  // namespace rviz_goal_pose_annotator
