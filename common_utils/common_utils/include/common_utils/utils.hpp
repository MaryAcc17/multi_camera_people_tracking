// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
// Copyright (c) 2023 Open Navigation LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef COMMON_UTILS__UTILS_HPP_
#define COMMON_UTILS__UTILS_HPP_

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "angles/angles.h"
#include "builtin_interfaces/msg/time.hpp"
#include "geometry_msgs/msg/pose2_d.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav2_core/goal_checker.hpp"
#include "nav2_util/node_utils.hpp"
#include "nav_msgs/msg/path.hpp"
#include "people_msgs/msg/people.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "tf2/utils.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include <opencv2/opencv.hpp>
#include <geometry_msgs/msg/polygon.hpp>
#include <geometry_msgs/msg/point32.hpp>

namespace common_utils::utils
{
/**
 * @brief Convert data into pose
 * @param x X position
 * @param y Y position
 * @param z Z position
 * @return Pose object
 */
inline geometry_msgs::msg::Pose createPose(double x, double y, double z)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;
  pose.orientation.w = 1;
  pose.orientation.x = 0;
  pose.orientation.y = 0;
  pose.orientation.z = 0;
  return pose;
}

/**
 * @brief Convert data into scale
 * @param x X scale
 * @param y Y scale
 * @param z Z scale
 * @return Scale object
 */
inline geometry_msgs::msg::Vector3 createScale(double x, double y, double z)
{
  geometry_msgs::msg::Vector3 scale;
  scale.x = x;
  scale.y = y;
  scale.z = z;
  return scale;
}

/**
 * @brief Convert data into color
 * @param r Red component
 * @param g Green component
 * @param b Blue component
 * @param a Alpha component (transparency)
 * @return Color object
 */
inline std_msgs::msg::ColorRGBA createColor(float r, float g, float b, float a)
{
  std_msgs::msg::ColorRGBA color;
  color.r = r;
  color.g = g;
  color.b = b;
  color.a = a;
  return color;
}

/**
 * @brief Convert data into a Maarker
 * @param id Marker ID
 * @param pose Marker pose
 * @param scale Marker scale
 * @param color Marker color
 * @param frame Reference frame to use
 * @return Visualization Marker
 */
inline visualization_msgs::msg::Marker createMarker(
  int id, const geometry_msgs::msg::Pose & pose,
  const geometry_msgs::msg::Vector3 & scale,
  const std_msgs::msg::ColorRGBA & color, const std::string & frame_id,
  const std::string & ns)
{
  using visualization_msgs::msg::Marker;
  Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = rclcpp::Time(0, 0);
  marker.ns = ns;
  marker.id = id;
  marker.type = Marker::SPHERE;
  marker.action = Marker::ADD;

  marker.pose = pose;
  marker.scale = scale;
  marker.color = color;
  return marker;
}

inline visualization_msgs::msg::Marker removeAllMarkers(const std::string & frame_id)
{
  using visualization_msgs::msg::Marker;
  Marker marker;
  marker.header.frame_id = frame_id;
  marker.action = Marker::DELETEALL;

  return marker;
}

/**
 * @brief Convert data into a Marker
 * @param id Marker ID
 * @param points Marker points
 * @param scale Marker scale
 * @param color Marker color
 * @param frame Reference frame to use
 * @return Visualization Marker
 */
inline visualization_msgs::msg::Marker createMarker(
  int id, const std::vector<geometry_msgs::msg::Point> & points,
  const geometry_msgs::msg::Vector3 & scale,
  const std_msgs::msg::ColorRGBA & color, const std::string & frame_id,
  const std::string & ns)
{
  using visualization_msgs::msg::Marker;
  Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = rclcpp::Time(0, 0);
  marker.ns = ns;
  marker.id = id;
  marker.type = Marker::POINTS;
  marker.action = Marker::ADD;

  marker.points = points;
  marker.scale = scale;
  marker.color = color;
  return marker;
}

/**
 * @brief Convert data into TwistStamped
 * @param vx X velocity
 * @param wz Angular velocity
 * @param stamp Timestamp
 * @param frame Reference frame to use
 */
inline geometry_msgs::msg::TwistStamped toTwistStamped(
  float vx, float wz, const builtin_interfaces::msg::Time & stamp,
  const std::string & frame)
{
  geometry_msgs::msg::TwistStamped twist;
  twist.header.frame_id = frame;
  twist.header.stamp = stamp;
  twist.twist.linear.x = vx;
  twist.twist.angular.z = wz;

  return twist;
}

/**
 * @brief Convert data into TwistStamped
 * @param vx X velocity
 * @param vy Y velocity
 * @param wz Angular velocity
 * @param stamp Timestamp
 * @param frame Reference frame to use
 */
inline geometry_msgs::msg::TwistStamped toTwistStamped(
  float vx, float vy, float wz,
  const builtin_interfaces::msg::Time & stamp,
  const std::string & frame)
{
  auto twist = toTwistStamped(vx, wz, stamp, frame);
  twist.twist.linear.y = vy;

  return twist;
}

/**
 * @brief Check if the robot pose is within the Goal Checker's tolerances to goal
 * @param global_checker Pointer to the goal checker
 * @param robot Pose of robot
 * @param goal Goal pose
 * @return bool If robot is within goal checker tolerances to the goal
 */
inline bool withinPositionGoalTolerance(
  nav2_core::GoalChecker * goal_checker, const geometry_msgs::msg::Pose & robot,
  const geometry_msgs::msg::Pose & goal)
{
  if (goal_checker) {
    geometry_msgs::msg::Pose pose_tolerance;
    geometry_msgs::msg::Twist velocity_tolerance;
    goal_checker->getTolerances(pose_tolerance, velocity_tolerance);

    const auto pose_tolerance_sq = pose_tolerance.position.x * pose_tolerance.position.x;

    auto dx = robot.position.x - goal.position.x;
    auto dy = robot.position.y - goal.position.y;

    auto dist_sq = dx * dx + dy * dy;

    if (dist_sq < pose_tolerance_sq) {
      return true;
    }
  }

  return false;
}

/**
 * @brief Check if the robot pose is within tolerance to the goal
 * @param pose_tolerance Pose tolerance to use
 * @param robot Pose of robot
 * @param goal Goal pose
 * @return bool If robot is within tolerance to the goal
 */
inline bool withinPositionGoalTolerance(
  float pose_tolerance, const geometry_msgs::msg::Pose & robot,
  const geometry_msgs::msg::Pose & goal)
{
  const double & dist_sq =
    std::pow(
    goal.position.x - robot.position.x,
    2) + std::pow(goal.position.y - robot.position.y, 2);

  const float pose_tolerance_sq = pose_tolerance * pose_tolerance;

  if (dist_sq < pose_tolerance_sq) {
    return true;
  }

  return false;
}

/**
 * @brief evaluate angle from pose (have angle) to point (no angle)
 * @param pose pose
 * @param point_x Point to find angle relative to X axis
 * @param point_y Point to find angle relative to Y axis
 * @param forward_preference If reversing direction is valid
 * @return Angle between two points
 */
inline float posePointAngle(
  const geometry_msgs::msg::Pose & pose, double point_x, double point_y,
  bool forward_preference)
{
  float pose_x = pose.position.x;
  float pose_y = pose.position.y;
  float pose_yaw = tf2::getYaw(pose.orientation);

  float yaw = atan2f(point_y - pose_y, point_x - pose_x);

  // If no preference for forward, return smallest angle either in heading or 180 of heading
  if (!forward_preference) {
    return std::min(
      fabs(angles::shortest_angular_distance(yaw, pose_yaw)),
      fabs(angles::shortest_angular_distance(yaw, angles::normalize_angle(pose_yaw + M_PI))));
  }

  return fabs(angles::shortest_angular_distance(yaw, pose_yaw));
}

/**
 * @brief Find the iterator of the first pose at which there is an inversion on the path,
 * @param path to check for inversion
 * @return the first point after the inversion found in the path
 */
inline unsigned int findFirstPathInversion(nav_msgs::msg::Path & path)
{
  // At least 3 poses for a possible inversion
  if (path.poses.size() < 3) {
    return path.poses.size();
  }

  // Iterating through the path to determine the position of the path inversion
  for (unsigned int idx = 1; idx < path.poses.size() - 1; ++idx) {
    // We have two vectors for the dot product OA and AB. Determining the vectors.
    float oa_x = path.poses[idx].pose.position.x - path.poses[idx - 1].pose.position.x;
    float oa_y = path.poses[idx].pose.position.y - path.poses[idx - 1].pose.position.y;
    float ab_x = path.poses[idx + 1].pose.position.x - path.poses[idx].pose.position.x;
    float ab_y = path.poses[idx + 1].pose.position.y - path.poses[idx].pose.position.y;

    // Checking for the existance of cusp, in the path, using the dot product.
    float dot_product = (oa_x * ab_x) + (oa_y * ab_y);
    if (dot_product < 0.0) {
      return idx + 1;
    }
  }

  return path.poses.size();
}

/**
 * @brief Find and remove poses after the first inversion in the path
 * @param path to check for inversion
 * @return The location of the inversion, return 0 if none exist
 */
inline unsigned int removePosesAfterFirstInversion(nav_msgs::msg::Path & path)
{
  nav_msgs::msg::Path cropped_path = path;
  const unsigned int first_after_inversion = findFirstPathInversion(cropped_path);
  if (first_after_inversion == path.poses.size()) {
    return 0u;
  }

  cropped_path.poses.erase(
    cropped_path.poses.begin() + first_after_inversion, cropped_path.poses.end());
  path = cropped_path;
  return first_after_inversion;
}

inline void createPooledCostmap(
  const nav2_costmap_2d::Costmap2D * source_costmap,
  std::unique_ptr<nav2_costmap_2d::Costmap2D> & pool_costmap, double new_resolution)
{
  // check if the source costmap is not null
  if (source_costmap == nullptr) {
    RCLCPP_ERROR(rclcpp::get_logger("common_utils"), "Source costmap is null");
    return;
  }
  // if pool_size is 1, no need to pool
  if (new_resolution == source_costmap->getResolution()) {
    return;
  }

  double old_resolution = source_costmap->getResolution();
  unsigned int old_size_x = source_costmap->getSizeInCellsX();
  unsigned int old_size_y = source_costmap->getSizeInCellsY();
  double origin_x = source_costmap->getOriginX();
  double origin_y = source_costmap->getOriginY();

  unsigned int new_size_x = std::ceil(old_size_x * old_resolution / new_resolution);
  unsigned int new_size_y = std::ceil(old_size_y * old_resolution / new_resolution);

  // check if the execution has reached this point
  RCLCPP_DEBUG(
    rclcpp::get_logger(
      "common_utils"), "Old size (%d, %d)\n Pooled costmap size: (%d, %d)", old_size_x,
    old_size_y, new_size_x, new_size_y);

  // if the pointer is not null resize the costmap
  if (pool_costmap != nullptr) {
    pool_costmap->resizeMap(new_size_x, new_size_y, new_resolution, origin_x, origin_y);
  } else {
    pool_costmap = std::make_unique<nav2_costmap_2d::Costmap2D>(
      new_size_x, new_size_y, new_resolution, origin_x, origin_y,
      const_cast<nav2_costmap_2d::Costmap2D *>(source_costmap)->getDefaultValue());
  }
}

inline void fastResampleCostmap(
  const nav2_costmap_2d::Costmap2D * source_costmap,
  std::unique_ptr<nav2_costmap_2d::Costmap2D> & target_costmap)
{
  double src_res = source_costmap->getResolution();
  double tgt_res = target_costmap->getResolution();
  if (tgt_res < src_res) {
    throw std::runtime_error("Target resolution must be >= source resolution for downsampling.");
  }

  const unsigned int pool_size = static_cast<unsigned int>(std::round(tgt_res / src_res));
  if (pool_size < 1) {
    throw std::runtime_error("Invalid pooling scale computed.");
  }

  unsigned int src_size_x = source_costmap->getSizeInCellsX();
  unsigned int src_size_y = source_costmap->getSizeInCellsY();
  const unsigned char * src_data = source_costmap->getCharMap();

  unsigned int tgt_size_x = target_costmap->getSizeInCellsX();
  unsigned int tgt_size_y = target_costmap->getSizeInCellsY();
  unsigned char * tgt_data = target_costmap->getCharMap();

  // update the target origin
  target_costmap->updateOrigin(source_costmap->getOriginX(), source_costmap->getOriginY());

  for (unsigned int j = 0; j < tgt_size_y; ++j) {
    // Calculate the source y range for this target row
    const unsigned int y0 = static_cast<unsigned int>(j * pool_size);
    const unsigned int y1 = std::min(y0 + pool_size, src_size_y);

    for (unsigned int i = 0; i < tgt_size_x; ++i) {
      // Calculate the source x range for this target column
      const unsigned int x0 = static_cast<unsigned int>(i * pool_size);
      const unsigned int x1 = std::min(x0 + pool_size, src_size_x);
      unsigned char max_cost = 0;

      for (unsigned int dy = y0; dy < y1; ++dy) {
        const unsigned int row_base = dy * src_size_x;
        for (unsigned int dx = x0; dx < x1; ++dx) {
          max_cost = std::max(max_cost, src_data[row_base + dx]);
        }
      }
      tgt_data[j * tgt_size_x + i] = max_cost;
    }
  }
}

/**
 * @brief Visualize the markers
 * @param agents The vector of social agents
 * @param obstacles The vector of obstacles
 * @return Visualization Marker array
 */
inline visualization_msgs::msg::MarkerArray visualizeMarkers(
  const std::vector<geometry_msgs::msg::Point> & agents,
  const std::vector<geometry_msgs::msg::Point> & obstacles,
  const std::string & frame_id = "map")
{
  visualization_msgs::msg::MarkerArray markers;
  // Remove all previous markers if present
  markers.markers.push_back(removeAllMarkers(frame_id));

  // Obstacles are visualized as red spheres
  auto scale = createScale(0.05, 0.05, 0.05);
  auto color = createColor(1.0, 0.0, 0.0, 1.0);
  markers.markers.push_back(createMarker(0, obstacles, scale, color, frame_id, "obstacles"));
  // Create a marker for each agent
  for (unsigned int i = 0; i < agents.size(); i++) {
    auto pose = createPose(agents[i].x, agents[i].y, 0.0);
    auto scale = createScale(0.1, 0.1, 0.1);
    auto color = createColor(0.0, 1.0, 0.0, 1.0);
    markers.markers.push_back(createMarker(i + 1, pose, scale, color, frame_id, "agents"));
  }
  return markers;
}

/**
 * @brief get the obstacles from the costmap
 * @param pool_costmap The costmap to use
 * @param agent_positions The vector of agent positions
 * @param people_radius_cell The radius of the people in cells
 * @param obstacles The vector of obstacles
 */
inline void getObstacles(
  std::unique_ptr<nav2_costmap_2d::Costmap2D> & pool_costmap,
  const std::vector<geometry_msgs::msg::Point> & agent_positions, double person_radius,
  std::vector<geometry_msgs::msg::Point> & obstacles)
{
  const unsigned int size_x = pool_costmap->getSizeInCellsX();
  const unsigned int size_y = pool_costmap->getSizeInCellsY();
  const double resolution = pool_costmap->getResolution();
  const unsigned char * map = pool_costmap->getCharMap();
  const double origin_x = pool_costmap->getOriginX();
  const double origin_y = pool_costmap->getOriginY();

  int people_radius_cell = person_radius / resolution + 1;
  // check if the agent is in the local costmap and update the obstacles
  std::vector<std::pair<unsigned int, unsigned int>> agent_cells;
  agent_cells.reserve(agent_positions.size());
  for (const auto & pos : agent_positions) {
    unsigned int ax, ay;
    if (pool_costmap->worldToMap(pos.x, pos.y, ax, ay)) {
      agent_cells.emplace_back(ax, ay);
    }
  }

  for (unsigned int j = 0; j < size_y; ++j) {
    for (unsigned int i = 0; i < size_x; ++i) {
      const unsigned int index = j * size_x + i;
      if (map[index] != nav2_costmap_2d::LETHAL_OBSTACLE) {
        continue;
      }

      // Check if this cell overlaps with any person
      bool is_person = false;
      for (const auto & [px, py] : agent_cells) {
        if (std::abs((int)i - (int)px) > people_radius_cell ||
          std::abs((int)j - (int)py) > people_radius_cell)
        {
          continue;
        }

        const float dx = static_cast<float>(i) - px;
        const float dy = static_cast<float>(j) - py;
        if (dx * dx + dy * dy < people_radius_cell * people_radius_cell) {
          is_person = true;
          break;
        }
      }

      if (is_person) {
        continue;
      }

      // Only now compute world coords (expensive)
      double wx = origin_x + (i + 0.5) * resolution;
      double wy = origin_y + (j + 0.5) * resolution;

      geometry_msgs::msg::Point pt;
      pt.x = wx;
      pt.y = wy;
      obstacles.push_back(pt);
      // obs_points.emplace_back(wx, wy);
    }
  }
}

/**
 * @brief Transform a pose and velocity to a new frame
 * @param pose_agent The pose to transform
 * @param velocity The velocity to transform
 * @param out_frame_id The frame id to transform to
 * @param t The timestamp to use for the transformation
 * @param tf_buffer The tf buffer to use for transformation
 */
inline bool transformPoseAndVelocity(
  geometry_msgs::msg::PoseStamped & pose_agent, geometry_msgs::msg::Vector3 & velocity,
  const std::string & out_frame_id, const builtin_interfaces::msg::Time & t,
  std::shared_ptr<tf2_ros::Buffer> & tf_buffer)
{
  try {
    geometry_msgs::msg::PoseStamped pose_local = tf_buffer->transform(pose_agent, out_frame_id);

    geometry_msgs::msg::Vector3Stamped velocity_stamped, localV;
    velocity_stamped.vector = velocity;
    velocity_stamped.vector.z = 0.0;
    velocity_stamped.header.frame_id = pose_agent.header.frame_id;
    velocity_stamped.header.stamp = t;

    localV = tf_buffer->transform(velocity_stamped, out_frame_id);

    pose_agent = pose_local;
    velocity = localV.vector;
  } catch (tf2::TransformException & ex) {
    RCLCPP_ERROR(rclcpp::get_logger("common_utils"), "Transform failed: %s", ex.what());
    return false;
  }
  return true;
}

struct GridCoord
{
  int x;  // X coordinate in the grid
  int y;  // Y coordinate in the grid

  bool operator==(const GridCoord & other) const
  {
    return x == other.x && y == other.y;
  }
};

inline bool isValid(int x, int y, int width, int height)
{
  return x >= 0 && x < width && y >= 0 && y < height;
}

std::vector<std::vector<GridCoord>> extractClustersFromCostmap(
  const nav2_costmap_2d::Costmap2D * costmap)
{
  int width = costmap->getSizeInCellsX();
  int height = costmap->getSizeInCellsY();
  std::vector<std::vector<bool>> visited(height, std::vector<bool>(width, false));  // Visited grid cells
  std::vector<std::vector<GridCoord>> clusters;                                     // Resulting clusters

  const int dx[8] = {-1, -1, -1, 0, 1, 1, 1, 0};
  const int dy[8] = {-1, 0, 1, 1, 1, 0, -1, -1};

  for (int y = 0; y < height; ++y) { // Iterate over each cell in the costmap
    for (int x = 0; x < width; ++x) { // Check if the cell is not visited and is an obstacle
      if (visited[y][x] ||
        costmap->getCost(x, y) < nav2_costmap_2d::LETHAL_OBSTACLE)    // LETHAL_OBSTACLE is the threshold for obstacles
      {
        continue;
      }

      std::vector<GridCoord> cluster;
      std::queue<GridCoord> q;
      q.push({x, y});
      visited[y][x] = true;

      while (!q.empty()) {
        GridCoord p = q.front();
        q.pop();               // Process the current cell
        cluster.push_back(p);  // Add the current cell to the cluster

        for (int i = 0; i < 8; ++i) { // Check all 8 neighbors of the current cell
          int nx = p.x + dx[i], ny = p.y + dy[i];
          if (!isValid(nx, ny, width, height)) {
            continue;  // Skip if the neighbor is out of bounds
          }
          if (!visited[ny][nx] && costmap->getCost(nx, ny) >= nav2_costmap_2d::LETHAL_OBSTACLE) {
            visited[ny][nx] = true;  // Mark the neighbor as visited
            q.push({nx, ny});        // Add the neighbor to the queue for processing
          }
        }
      }

      clusters.push_back(cluster);  // Add the found cluster to the result
    }
  }

  return clusters;
}

geometry_msgs::msg::Polygon toPolygonMsg(const cv::RotatedRect & rect)
{
  geometry_msgs::msg::Polygon poly;
  poly.points.reserve(4);

  cv::Point2f corners[4];
  rect.points(corners);
  for (int i = 0; i < 4; ++i) {
    geometry_msgs::msg::Point32 pt;
    pt.x = corners[i].x;
    pt.y = corners[i].y;
    pt.z = 0;
    poly.points.push_back(pt);
  }
  return poly;
}

geometry_msgs::msg::Polygon computeMinAreaBox(
  const std::vector<GridCoord> & cluster,
  const nav2_costmap_2d::Costmap2D * costmap)
{
  if (cluster.empty()) {
    return geometry_msgs::msg::Polygon();
  }
  std::vector<cv::Point2f> world_points;
  world_points.reserve(cluster.size());

  for (const auto & cell : cluster) {
    double wx, wy;
    costmap->mapToWorld(cell.x, cell.y, wx, wy);
    world_points.emplace_back(wx, wy);
  }

  cv::RotatedRect rect = cv::minAreaRect(world_points);
  return toPolygonMsg(rect);
}

visualization_msgs::msg::MarkerArray createPolygonMarkers(
  const std::vector<geometry_msgs::msg::Polygon> & polygons, const std::string & frame_id,
  const std::string & ns = "obstacle_boxes", const std::array<float, 4> & color = {0.0f, 1.0f, 0.0f,
    1.0f},                                                                                                     // RGBA
  double line_width = 0.05)
{
  visualization_msgs::msg::MarkerArray marker_array;

  for (size_t i = 0; i < polygons.size(); ++i) {
    const auto & poly = polygons[i];

    if (poly.points.size() < 3) {
      continue;
    }

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = rclcpp::Clock().now();
    marker.ns = ns;
    marker.id = static_cast<int>(i);
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;

    marker.scale.x = line_width;

    marker.color.r = color[0];
    marker.color.g = color[1];
    marker.color.b = color[2];
    marker.color.a = color[3];

    for (const auto & pt : poly.points) {
      geometry_msgs::msg::Point p;
      p.x = pt.x;
      p.y = pt.y;
      p.z = pt.z;
      marker.points.push_back(p);
    }

    // Close the loop
    geometry_msgs::msg::Point first = marker.points.front();
    marker.points.push_back(first);

    marker_array.markers.push_back(marker);
  }

  return marker_array;
}

geometry_msgs::msg::Point closestPointOnSegment(
  const geometry_msgs::msg::Point & p, const geometry_msgs::msg::Point & a,
  const geometry_msgs::msg::Point & b)
{
  double dx = b.x - a.x;
  double dy = b.y - a.y;

  if (dx == 0 && dy == 0) {
    return a;  // segment is a point

  }
  double t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy);
  t = std::max(0.0, std::min(1.0, t));

  geometry_msgs::msg::Point proj;
  proj.x = a.x + t * dx;
  proj.y = a.y + t * dy;
  proj.z = 0.0;
  return proj;
}

bool isPointInsidePolygon(
  const geometry_msgs::msg::Point & p,
  const geometry_msgs::msg::Polygon & poly)
{
  bool inside = false;
  size_t n = poly.points.size();
  for (size_t i = 0, j = n - 1; i < n; j = i++) {
    const auto & pi = poly.points[i];
    const auto & pj = poly.points[j];

    if (((pi.y > p.y) != (pj.y > p.y)) &&
      (p.x < (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y + 1e-12) + pi.x))
    {
      inside = !inside;
    }
  }
  return inside;
}

geometry_msgs::msg::Point findClosestPointToPolygon(
  const geometry_msgs::msg::Point & p,
  const geometry_msgs::msg::Polygon & poly)
{
  geometry_msgs::msg::Point closest;
  double min_dist_sq = std::numeric_limits<double>::max();
  size_t n = poly.points.size();

  for (size_t i = 0; i < n; ++i) {
    geometry_msgs::msg::Point a, b;
    a.x = poly.points[i].x;
    a.y = poly.points[i].y;
    b.x = poly.points[(i + 1) % n].x;
    b.y = poly.points[(i + 1) % n].y;

    geometry_msgs::msg::Point proj = closestPointOnSegment(p, a, b);
    double dx = proj.x - p.x;
    double dy = proj.y - p.y;
    double dist_sq = dx * dx + dy * dy;

    if (dist_sq < min_dist_sq) {
      min_dist_sq = dist_sq;
      closest = proj;
    }
  }

  return closest;
}

std::pair<bool, geometry_msgs::msg::Point> isPointNearPolygon(
  const geometry_msgs::msg::Point & p,
  const geometry_msgs::msg::Polygon & poly, double threshold)
{
  if (isPointInsidePolygon(p, poly)) {
    return {true, p};
  }                      // Point is inside the polygon

  geometry_msgs::msg::Point closest = findClosestPointToPolygon(p, poly);
  double dx = closest.x - p.x;
  double dy = closest.y - p.y;
  double dist_sq = dx * dx + dy * dy;

  return {dist_sq <= threshold * threshold, closest};
}

}  // namespace common_utils::utils

#endif  // COMMON_UTILS__UTILS_HPP_
