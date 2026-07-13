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

#ifndef COMMON_UTILS__HUNAV_UTILS_HPP_
#define COMMON_UTILS__HUNAV_UTILS_HPP_

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "angles/angles.h"
#include "builtin_interfaces/msg/time.hpp"
#include "common_utils/utils.hpp"
#include "geometry_msgs/msg/pose2_d.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "lightsfm/angle.hpp"
#include "lightsfm/sfm.hpp"
#include "lightsfm/vector2d.hpp"
#include "nav2_core/goal_checker.hpp"
#include "nav2_util/node_utils.hpp"
#include "nav_msgs/msg/path.hpp"
#include "people_msgs/msg/people.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "tf2/utils.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

typedef utils::Vector2d Vector2d;
typedef utils::Angle Angle;

namespace common_utils::hunav_utils
{
/**
 * @brief get the agents in the fov of the robot
 * @param costmap The costmap to use
 * @param source_agents The vector of source agents
 * @param pose The pose of the robot
 * @param fov_angle The field of view angle
 * @param agent_positions The vector of agent positions
 * @param output_agents The vector of output agents
 */
inline void getAgentsInFov(
  const nav2_costmap_2d::Costmap2D * costmap, const std::vector<sfm::Agent> & source_agents,
  const geometry_msgs::msg::Pose2D & pose, double fov_angle,
  std::vector<geometry_msgs::msg::Point> & agent_positions,
  std::vector<sfm::Agent> & output_agents)
{
  for (auto & agent : source_agents) {
    // skip the robot
    if (agent.id == 0) {
      continue;
    }
    unsigned int mx, my;
    if (!costmap->worldToMap(agent.position.getX(), agent.position.getY(), mx, my)) {
      continue;
    }
    // Check if the agent is in the fov of the robot
    double dx = agent.position.getX() - pose.x;
    double dy = agent.position.getY() - pose.y;
    double angle = atan2(dy, dx);
    double angle_diff = angle - pose.theta;
    angle_diff = atan2(sin(angle_diff), cos(angle_diff));

    geometry_msgs::msg::Point pt;
    pt.x = agent.position.getX();
    pt.y = agent.position.getY();
    agent_positions.push_back(pt);
    if (fabs(angle_diff) > fov_angle) {
      continue;
    }
    output_agents.push_back(agent);
  }
}

/**
 * @brief get the agents in the fov of the robot
 * @param costmap The costmap to use
 * @param source_agents The vector of source agents
 * @param pose The pose of the robot
 * @param fov_angle The field of view angle
 * @param agent_positions The vector of agent positions
 * @param output_agents The vector of output agents
 */
inline void getAgentsInFov(
  const nav2_costmap_2d::Costmap2D * costmap, const std::vector<sfm::Agent> & source_agents,
  const geometry_msgs::msg::Pose & pose, double fov_angle,
  std::vector<geometry_msgs::msg::Point> & agent_positions,
  std::vector<sfm::Agent> & output_agents)
{
  // convert the pose to 2d
  geometry_msgs::msg::Pose2D pose_2d;
  pose_2d.x = pose.position.x;
  pose_2d.y = pose.position.y;
  pose_2d.theta = tf2::getYaw(pose.orientation);
  getAgentsInFov(costmap, source_agents, pose_2d, fov_angle, agent_positions, output_agents);
}

/**
 * @brief convert a geometry_msgs::msg::Point to a Vector2d
 * @param point The point to convert
 * @return The Vector2d
 */
inline Vector2d convertToVector2d(const geometry_msgs::msg::Point & point)
{
  Vector2d pt;
  pt.set(point.x, point.y);
  return pt;
}
/**
 * @brief convert a Vector2d to a geometry_msgs::msg::Point
 * @param point The Vector2d to convert
 * @return The geometry_msgs::msg::Point
 */
inline geometry_msgs::msg::Point convertToPoint(const Vector2d & point)
{
  geometry_msgs::msg::Point pt;
  pt.x = point.getX();
  pt.y = point.getY();
  return pt;
}

/**
 * @brief convert a std::vector<sfm::Agent> to a std::vector<geometry_msgs::msg::Point>
 * @param agents The vector of sfm::Agent to convert
 * @return The vector of geometry_msgs::msg::Point
 */
inline std::vector<geometry_msgs::msg::Point> convertToPoint(const std::vector<sfm::Agent> & agents)
{
  std::vector<geometry_msgs::msg::Point> points_out;
  points_out.reserve(agents.size());
  for (const auto & agent : agents) {
    geometry_msgs::msg::Point pt = convertToPoint(agent.position);
    points_out.push_back(pt);
  }
  return points_out;
}

/**
 * @brief convert a std::vector<Vector2d> to a std::vector<geometry_msgs::msg::Point>
 * @param points The vector of Vector2d to convert
 * @return The vector of geometry_msgs::msg::Point
 */
inline std::vector<geometry_msgs::msg::Point> convertToPoints(const std::vector<Vector2d> & points)
{
  std::vector<geometry_msgs::msg::Point> points_out;
  points_out.reserve(points.size());
  for (const auto & point : points) {
    geometry_msgs::msg::Point pt = convertToPoint(point);
    points_out.push_back(pt);
  }
  return points_out;
}
/**
 * @brief convert a std::vector<geometry_msgs::msg::Point> to a std::vector<Vector2d>
 * @param points The vector of geometry_msgs::msg::Point to convert
 * @return The vector of Vector2d
 */
inline std::vector<Vector2d> convertToVector2d(
  const std::vector<geometry_msgs::msg::Point> & points)
{
  std::vector<Vector2d> points_out;
  points_out.reserve(points.size());
  for (const auto & point : points) {
    Vector2d pt = convertToVector2d(point);
    points_out.push_back(pt);
  }
  return points_out;
}

/**
 * @brief Create an agent from a person
 * @param person The person to convert
 * @param agent The agent to fill
 * @param in_frame_id The frame id of the person
 * @param out_frame_id The frame id of the agent
 * @param tf_buffer The tf buffer to use for transformation
 * @param t The timestamp to use for the transformation
 * @param person_radius The radius of the person
 * @param person_velocity The velocity of the person
 * @param naive_goal_time The time to reach the goal
 */
inline bool createAgentFromPerson(
  const people_msgs::msg::Person & person, sfm::Agent & agent,
  const std::string & in_frame_id, const std::string & out_frame_id,
  std::shared_ptr<tf2_ros::Buffer> & tf_buffer, const builtin_interfaces::msg::Time & t,
  const float person_radius, const float person_velocity, const float naive_goal_time,
  const sfm::Parameters & params = sfm::Parameters())
{
  try {
    if (person.tags.size() > 0) {
      RCLCPP_DEBUG(
        rclcpp::get_logger(
          "common_utils::hunav_utils"), "Person ID tag: %s", person.tags[0].c_str());
      agent.id = std::stoi(person.tags[0]);
    } else {
      RCLCPP_WARN_ONCE(
        rclcpp::get_logger(
          "common_utils::hunav_utils"), "Person has no ID tag, setting to -1");
      agent.id = -1;
    }
  } catch (const std::exception & e) {
    if (person.tags.size() > 0) {
      RCLCPP_DEBUG(
        rclcpp::get_logger(
          "common_utils::hunav_utils"), "Person ID tag: %s", person.tags[0].c_str());
      RCLCPP_WARN_ONCE(
        rclcpp::get_logger("common_utils::hunav_utils"),
        "Person ID tag is not a valid integer: %s", e.what());
    }
  }
  try {
    if (person.tags.size() > 1) {
      RCLCPP_DEBUG(
        rclcpp::get_logger(
          "common_utils::hunav_utils"), "Person group ID tag: %s", person.tags[1].c_str());
      agent.groupId = std::stoi(person.tags[1]);
    } else {
      RCLCPP_WARN_ONCE(
        rclcpp::get_logger(
          "common_utils::hunav_utils"), "Person has no group ID tag, setting to -1");
      agent.groupId = -1;
    }
  } catch (const std::exception & e) {
    RCLCPP_WARN_ONCE(
      rclcpp::get_logger("common_utils::hunav_utils"),
      "Person group ID tag is not a valid integer: %s", e.what());
  }
  geometry_msgs::msg::PoseStamped pose_agent;
  pose_agent.header.frame_id = in_frame_id;
  pose_agent.header.stamp = t;
  pose_agent.pose.position = person.position;
  pose_agent.pose.position.z = 0.0;

  tf2::Quaternion quat;
  quat.setRPY(0, 0, person.position.z);
  pose_agent.pose.orientation = tf2::toMsg(quat);

  geometry_msgs::msg::Vector3 velocity;
  velocity.x = person.velocity.x;
  velocity.y = person.velocity.y;
  velocity.z = 0.0;

  if (in_frame_id != out_frame_id) {
    if (!common_utils::utils::transformPoseAndVelocity(
        pose_agent, velocity, out_frame_id, t,
        tf_buffer))
    {
      return false;
    }
  }

  agent.position.set(pose_agent.pose.position.x, pose_agent.pose.position.y);
  agent.velocity.set(velocity.x, velocity.y);
  agent.linearVelocity = agent.velocity.norm();

  agent.yaw = (fabs(agent.linearVelocity) < 0.09) ?
    Angle::fromRadian(tf2::getYaw(pose_agent.pose.orientation)) :
    Angle::fromRadian(atan2(agent.velocity.getY(), agent.velocity.getX()));

  agent.angularVelocity = person.velocity.z;
  agent.radius = person_radius;
  agent.teleoperated = false;
  agent.desiredVelocity = person_velocity;

  sfm::Goal naiveGoal;
  Vector2d v = agent.position + naive_goal_time * agent.velocity.normalized() * person_velocity;
  RCLCPP_DEBUG(
    rclcpp::get_logger(
      "common_utils::hunav_utils"),
    "Agent %d: Naive goal at (%f, %f) initial position (%f, %f) with velocity (%f, %f)", agent.id,
    v.getX(),
    v.getY(), agent.position.getX(), agent.position.getY(),
    agent.velocity.getX(), agent.velocity.getY());
  naiveGoal.center.set(v.getX(), v.getY());
  naiveGoal.radius = person_radius;
  agent.goals.clear();
  agent.goals.push_back(naiveGoal);
  agent.params = params;

  return true;
}

inline visualization_msgs::msg::Marker createForceMarker(
  const visualization_msgs::msg::Marker & base_marker,
  const Vector2d & force_vector,
  const std_msgs::msg::ColorRGBA & color,
  const uint32_t marker_id_offset = 1)
{
  using visualization_msgs::msg::Marker;
  Marker force_marker = base_marker;  // Start with the base marker properties (position, orientation, etc.)
  force_marker.id = base_marker.id + marker_id_offset;  // Increment the ID for the force marker
  force_marker.action =
    force_vector.norm() >
    1e-4 ? visualization_msgs::msg::Marker::ADD : visualization_msgs::msg::Marker::DELETE;
  force_marker.color = color;
  force_marker.scale.x = std::max(1e-4, force_vector.norm());  // Set the length of the arrow based on the force vector
  force_marker.scale.y = 0.1;                                  // Set a small width for the arrow
  force_marker.scale.z = 0.1;                                  // Set a small height for the arrow
  tf2::Quaternion quat;
  quat.setRPY(
    0, 0,
    force_vector.angle().toRadian());            // Set the orientation of the arrow based on the force vector angle
  force_marker.pose.orientation = tf2::toMsg(quat);
  return force_marker;
}

/**
 * @brief Create a marker for the agent forces
 * @param agent The agent to create the marker for
 *
 */
inline visualization_msgs::msg::MarkerArray createAgentForcesMarker(
  const sfm::Agent & agent,
  const std_msgs::msg::Header & header)
{
  using visualization_msgs::msg::Marker;
  using visualization_msgs::msg::MarkerArray;
  MarkerArray marker_array;
  Marker marker, goal_marker;

  const sfm::Forces & forces = agent.forces;
  const Vector2d & velocity = agent.velocity;

  marker.header = header;
  marker.ns = "agent" + std::to_string(agent.id) + "/robot_forces";
  marker.id = agent.id;
  marker.type = visualization_msgs::msg::Marker::ARROW;

  marker.pose.position = convertToPoint(agent.position);
  marker.lifetime = rclcpp::Duration(1, 0);

  marker_array.markers.push_back(
    createForceMarker(
      marker, forces.obstacleForce,
      common_utils::utils::createColor(1.0, 0.0, 0.0, 0.5),
      1));                                               // #F00 Red
  marker_array.markers.push_back(
    createForceMarker(
      marker, forces.socialForce,
      common_utils::utils::createColor(0.0, 0.0, 1.0, 0.5),
      2));                                               // #00F Blue
  marker_array.markers.push_back(
    createForceMarker(
      marker, forces.desiredForce,
      common_utils::utils::createColor(0.0, 1.0, 0.0, 0.5),
      3));                                               // #0F0 Green
  marker_array.markers.push_back(
    createForceMarker(
      marker, forces.globalForce,
      common_utils::utils::createColor(1.0, 1.0, 1.0, 1.0),
      4));                                               // #FFF White
  marker_array.markers.push_back(
    createForceMarker(
      marker, velocity,
      common_utils::utils::createColor(1.0, 1.0, 0.0, 0.5),
      0));                                               // #FF0 Yellow
                                                         // Add the goal marker
  goal_marker = marker;                                  // Start with the base marker properties
  goal_marker.ns = "agent" + std::to_string(agent.id) + "/robot_goal";
  goal_marker.id = 0;  // Use a different ID for the goal marker
  goal_marker.type = visualization_msgs::msg::Marker::SPHERE;
  goal_marker.pose.position = convertToPoint(agent.goals.front().center);    // Use the first goal's center
  goal_marker.scale.x = agent.goals.front().radius * 2.0;                    // Diameter of the sphere
  goal_marker.scale.y = agent.goals.front().radius * 2.0;                    // Diameter of the sphere
  goal_marker.scale.z = agent.goals.front().radius * 2.0;                    // Diameter of the sphere
  goal_marker.color = common_utils::utils::createColor(1.0, 0.5, 0.0, 0.5);  // Orange color
  marker_array.markers.push_back(goal_marker);
  return marker_array;
}

/**
 * @brief Create agents marker array from a vector of agents
 * @param agents The vector of agents to create markers from
 * @param frame_id The frame ID to use for the markers
 * @return A marker array representing the agents
 */
inline visualization_msgs::msg::MarkerArray createAgentsMarkerArray(
  const std::vector<sfm::Agent> & agents,
  const std_msgs::msg::Header & header)
{
  using visualization_msgs::msg::MarkerArray;
  MarkerArray marker_array;
  marker_array.markers.reserve(agents.size() * 5 + 1);  // Reserve space for the markers

  // Remove all previous markers if present
  marker_array.markers.push_back(common_utils::utils::removeAllMarkers(header.frame_id));

  for (const auto & agent : agents) {
    // marker_array.markers.push_back(createAgentForcesMarker(agent, header));  // Create a marker for each agent's
    // forces
    MarkerArray forces_marker_array =
      createAgentForcesMarker(agent, header);    // Create a marker for each agent's forces
    marker_array.markers.insert(
      marker_array.markers.end(), forces_marker_array.markers.begin(),
      forces_marker_array.markers.end());                            // Insert the forces markers into the marker array
  }
  return marker_array;
}

}  // namespace common_utils::hunav_utils

#endif  // COMMON_UTILS__HUNAV_UTILS_HPP_
