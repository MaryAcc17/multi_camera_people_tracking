#include "dwb_additional_critics/social_force.hpp"

#include <math.h>

#include <algorithm>
#include <cmath>

#include "dwb_core/exceptions.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_util/node_utils.hpp"
#include "nav_2d_utils/path_ops.hpp"
#include "nav_2d_utils/tf_help.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

PLUGINLIB_EXPORT_CLASS(dwb_critics::SocialForceCritic, dwb_core::TrajectoryCritic)

using nav2_util::declare_parameter_if_not_declared;

namespace dwb_critics
{

SocialForceCritic::~SocialForceCritic()
{
  if (marker_pub_) {
    if (marker_pub_->is_activated()) {
      marker_pub_->on_deactivate();
    }
    marker_pub_.reset();
  }

  people_sub_.reset();

  if (tf_listener_) {
    tf_listener_.reset();
  }

  if (tf_buffer_) {
    tf_buffer_->clear();
    tf_buffer_.reset();
  }
}

void SocialForceCritic::onInit()
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error{"Failed to lock node"};
  }
  RCLCPP_INFO(logger_, "Initializing SocialForceCritic");

  declare_parameter_if_not_declared(
    node, dwb_plugin_name_ + "." + name_ + ".person_radius", rclcpp::ParameterValue(0.4));
  declare_parameter_if_not_declared(
    node, dwb_plugin_name_ + "." + name_ + ".robot_radius", rclcpp::ParameterValue(0.4));
  declare_parameter_if_not_declared(
    node, dwb_plugin_name_ + "." + name_ + ".person_velocity", rclcpp::ParameterValue(0.5));
  declare_parameter_if_not_declared(
    node, dwb_plugin_name_ + "." + name_ + ".naive_goal_time", rclcpp::ParameterValue(1.0));
  declare_parameter_if_not_declared(
    node, dwb_plugin_name_ + "." + name_ + ".fov_angle", rclcpp::ParameterValue(M_PI / 2));
  declare_parameter_if_not_declared(
    node, dwb_plugin_name_ + "." + name_ + ".new_resolution", rclcpp::ParameterValue(0.2));
  declare_parameter_if_not_declared(
    node, dwb_plugin_name_ + "." + ".max_vel_x", rclcpp::ParameterValue(0.6));

  node->get_parameter(dwb_plugin_name_ + "." + name_ + ".person_radius", person_radius_);
  node->get_parameter(dwb_plugin_name_ + "." + name_ + ".robot_radius", robot_radius_);
  node->get_parameter(dwb_plugin_name_ + "." + name_ + ".person_velocity", person_velocity_);
  node->get_parameter(dwb_plugin_name_ + "." + name_ + ".naive_goal_time", naive_goal_time_);
  node->get_parameter(dwb_plugin_name_ + "." + name_ + ".fov_angle", fov_angle_);
  node->get_parameter(dwb_plugin_name_ + "." + name_ + ".new_resolution", new_resolution_);
  node->get_parameter(dwb_plugin_name_ + "." + ".max_vel_x_", robot_linear_velocity_);
  costmap_ = costmap_ros_->getCostmap();
  std::unique_lock<nav2_costmap_2d::Costmap2D::mutex_t> costmap_lock(*(costmap_->getMutex()));

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  agents_ = std::make_shared<std::vector<sfm::Agent>>();
  auto robot = sfm::Agent();
  initializeRobot(robot);
  agents_->push_back(robot);
  RCLCPP_INFO(logger_, "Robot initialized: (%f, %f)", robot.position.getX(), robot.position.getY());

  common_utils::utils::createPooledCostmap(costmap_, pool_costmap_, new_resolution_);

  costmap_lock.unlock();

  RCLCPP_INFO(logger_, "Pooled costmap created");
  if (pool_costmap_ == nullptr) {
    throw std::runtime_error{"Failed to create pooled costmap"};
  }

  RCLCPP_INFO(
    logger_, "Pooled costmap initialized: (%d, %d)", pool_costmap_->getSizeInCellsX(),
    pool_costmap_->getSizeInCellsY());
  people_sub_ = node->create_subscription<people_msgs::msg::People>(
    "/people", 10, std::bind(&SocialForceCritic::peopleCallback, this, std::placeholders::_1));

  marker_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>("obstacles", 10);
  marker_pub_->on_activate();
}

bool SocialForceCritic::prepare(
  const geometry_msgs::msg::Pose2D & pose, const nav_2d_msgs::msg::Twist2D & vel,
  const geometry_msgs::msg::Pose2D &, const nav_2d_msgs::msg::Path2D & global_plan)
{
  // Get the last pose on the costmap (local goal)
  if (global_plan.poses.empty()) {
    RCLCPP_ERROR(logger_, "Global plan is empty");
    return false;
  }

  geometry_msgs::msg::Pose2D local_goal = global_plan.poses.back();

  // Set the local goal
  sfm::Goal g;
  g.center.set(local_goal.x, local_goal.y);
  g.radius = 0.20;
  goal_ = std::make_shared<sfm::Goal>(g);

  RCLCPP_DEBUG(logger_, "Local goal: (%f, %f)", local_goal.x, local_goal.y);

  std::vector<sfm::Agent> agents = std::vector<sfm::Agent>();
  std::vector<geometry_msgs::msg::Point> agent_positions;
  // check if the agent is in the local costmap and update the obstacles
  // get the agents in the fov of the robot
  common_utils::hunav_utils::getAgentsInFov(
    costmap_, getAgents(), pose, fov_angle_, agent_positions, agents);

  // remove people from the obstacles
  std::vector<geometry_msgs::msg::Point> obstacles;
  // Select a coarser resolution for the local costmap and retrieve the obstacles
  // select a resolution that is a multiple of the global costmap resolution
  common_utils::utils::fastResampleCostmap(costmap_, pool_costmap_);

  // Check if the agent is in the local costmap and update the obstacles
  common_utils::utils::getObstacles(pool_costmap_, agent_positions, person_radius_ * 2, obstacles);

  const std::vector<utils::Vector2d> obs_points =
    common_utils::hunav_utils::convertToVector2d(obstacles);

  RCLCPP_DEBUG(logger_, "Number of obstacles: %lu", obs_points.size());

  marker_pub_->publish(common_utils::utils::visualizeMarkers(agent_positions, obstacles));

  std::lock_guard<std::mutex> lock(agents_mutex_);
  agents_->resize(1);
  auto & robot = (*agents_)[0];
  robot.position.set(pose.x, pose.y);
  robot.yaw = utils::Angle::fromRadian(pose.theta);
  robot.linearVelocity = hypotf(vel.x, vel.y);
  robot.angularVelocity = vel.theta;
  robot.velocity.set(vel.x, vel.y);

  robot.goals.clear();
  robot.goals.push_back(getGoal());
  robot.obstacles1.clear();
  robot.obstacles1 = obs_points;
  robot.teleoperated = true;

  agents_->resize(agents.size() + 1);
  for (unsigned int i = 1; i < agents.size(); i++) {
    (*agents_)[i] = agents[i - 1];
    auto & agent = (*agents_)[i];
    RCLCPP_DEBUG(
      logger_, "Agent %d: (%f, %f)", agent.id, agent.position.getX(), agent.position.getY());
    agent.obstacles1.clear();
    agent.obstacles1 = obs_points;
  }
  return true;
}

double SocialForceCritic::scoreTrajectory(const dwb_msgs::msg::Trajectory2D & traj)
{
  std::vector<sfm::Agent> myagents = getAgents();
  if (myagents.size() == 0) {
    return 0.0;
  }
  RCLCPP_DEBUG(logger_, "Number of agents: %lu", myagents.size());
  auto & robot = myagents[0];
  RCLCPP_DEBUG(logger_, "Robot: (%f, %f)", robot.position.getX(), robot.position.getY());
  RCLCPP_DEBUG(logger_, "Robot velocity: %f", robot.linearVelocity);

  // TODO: use only a subset of the time offsets
  auto previoustime = rclcpp::Duration(0, 0);
  double social_work = 0.0;
  const auto pose_count = traj.poses.size();
  const auto offset_count = traj.time_offsets.size();
  if (pose_count == 0 || offset_count == 0) {
    RCLCPP_WARN(
      logger_, "Trajectory missing poses (%zu) or time offsets (%zu)", pose_count,
      offset_count);
    return social_work;
  }

  const auto samples = std::min(pose_count, offset_count);

  if (pose_count != offset_count) {
    RCLCPP_WARN(
      logger_, "Trajectory poses (%zu) and time offsets (%zu) sizes differ; using the minimum",
      pose_count, offset_count);
  }

  RCLCPP_DEBUG(logger_, "Trajectory size: %zu", samples);
  for (size_t i = 0; i < samples; i += 10) {
    if (i >= pose_count || i >= offset_count) {
      break;
    }
    auto current_time_offset = rclcpp::Duration(traj.time_offsets[i]);
    rclcpp::Duration dt = current_time_offset - previoustime;
    previoustime = current_time_offset;
    // Compute Social Forces
    sfm::SFM.computeForces(myagents);

    // update agents
    sfm::SFM.updatePosition(myagents, dt.seconds());

    // Update agent robot that does not move using SFM
    updateAgentRobot(robot, traj, i);

    RCLCPP_DEBUG(
      logger_, "Agent robot: (%f, %f) obstacle size: %lu", robot.position.getX(),
      robot.position.getY(), robot.obstacles1.size());
    RCLCPP_DEBUG(logger_, "Agent robot velocity: %f", robot.linearVelocity);
    // Compute the social work
    RCLCPP_DEBUG(logger_, "Social work computed");

    // Check if a collision with a dynamic obstacle is possible
    for (unsigned int j = 1; j < myagents.size(); j++) {
      double dx = robot.position.getX() - myagents[j].position.getX();
      double dy = robot.position.getY() - myagents[j].position.getY();
      double d = dx * dx + dy * dy;  // hypotf(dx, dy);
      if (d <= (robot_radius_ * robot_radius_ + person_radius_ * person_radius_)) {
        throw dwb_core::IllegalTrajectoryException(name_, "Trajectory Hits another Actor.");
      }
    }

    social_work += computeSocialWork(myagents);
  }
  RCLCPP_DEBUG(logger_, "Social work: %f", social_work);

  return social_work;
}

void SocialForceCritic::peopleCallback(const std::shared_ptr<people_msgs::msg::People> people)
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error{"Failed to lock node"};
  }
  builtin_interfaces::msg::Time t = node->get_clock()->now();

  std::vector<sfm::Agent> agents;

  geometry_msgs::msg::PoseStamped pose_agent;
  RCLCPP_DEBUG(logger_, "People frame: %s", people->header.frame_id.c_str());

  if (people->people.size() == 0) {
    RCLCPP_DEBUG(logger_, "No people detected");
    return;
  }

  for (const auto & person : people->people) {
    // Create the agent
    sfm::Agent agent;
    if (
      !common_utils::hunav_utils::createAgentFromPerson(
        person, agent, people->header.frame_id, costmap_ros_->getGlobalFrameID(), tf_buffer_, t,
        person_radius_, person_velocity_, naive_goal_time_))
    {
      return;  // Transform failed
    }
    agents.push_back(agent);
  }

  updateAgents(agents);
  RCLCPP_DEBUG(logger_, "Number of agents: %lu", agents.size());
  for (unsigned int i = 0; i < agents.size(); i++) {
    RCLCPP_DEBUG(
      logger_, "Agent %d: (%f, %f)", agents[i].id, agents[i].position.getX(),
      agents[i].position.getY());
  }
}

double SocialForceCritic::computeSocialWork(const std::vector<sfm::Agent> & agents)
{
  // The social work of the robot
  if (agents.size() == 0) {
    return 0.0;
  }
  auto & robot = agents[0];
  double wr = robot.forces.socialForce.norm() + robot.forces.obstacleForce.norm();

  // Compute the social work provoked by the robot in the other agents
  std::vector<sfm::Agent> agent_robot;

  agent_robot.push_back(robot);
  double wp = 0.0;
  for (unsigned int i = 1; i < agents.size(); i++) {
    sfm::Agent agent = agents[i];
    // Compute the forces between the robot and the agent
    sfm::SFM.computeForces(agent, agent_robot);
    // Sum the social force norm
    wp += agent.forces.socialForce.norm();
  }

  // Sum Wr and Wp
  return wr + wp;
}

std::vector<sfm::Agent> SocialForceCritic::getAgents()
{
  if (agents_ == nullptr) {
    throw std::runtime_error{"Agents are not set"};
  }
  std::lock_guard<std::mutex> lock(agents_mutex_);
  return *agents_;
}

void SocialForceCritic::initializeRobot(sfm::Agent & agent)
{
  agent.id = 0;
  agent.groupId = -1;
  agent.position.set(0.0, 0.0);
  agent.velocity.set(0.0, 0.0);
  agent.linearVelocity = 0.0;
  agent.yaw = utils::Angle::fromRadian(0.0);
  agent.angularVelocity = 0.0;
  agent.radius = robot_radius_;
  agent.teleoperated = true;
  agent.desiredVelocity = robot_linear_velocity_;
}

void SocialForceCritic::updateAgentRobot(
  sfm::Agent & robot, const dwb_msgs::msg::Trajectory2D & traj, size_t index)
{
  robot.position.set(traj.poses[index].x, traj.poses[index].y);
  robot.yaw = utils::Angle::fromRadian(traj.poses[index].theta);
  robot.linearVelocity = hypotf(traj.velocity.x, traj.velocity.y);
  robot.angularVelocity = traj.velocity.theta;
  robot.velocity.set(traj.velocity.x, traj.velocity.y);

  robot.goals.clear();
  robot.goals.push_back(getGoal());
}

void SocialForceCritic::updateAgents(const std::vector<sfm::Agent> & agents)
{
  if (agents_ == nullptr || agents_->empty()) {
    throw std::runtime_error{"Robot agent not initialized"};
  }

  std::lock_guard<std::mutex> lock(agents_mutex_);
  auto robot = (*agents_)[0];  // save robot
  agents_->resize(agents.size() + 1);
  (*agents_)[0] = robot;
  for (size_t i = 0; i < agents.size(); ++i) {
    (*agents_)[i + 1] = agents[i];
  }
}

sfm::Goal SocialForceCritic::getGoal()
{
  if (goal_ == nullptr) {
    throw std::runtime_error{"Goal is not set"};
  }
  return *goal_;
}

}  // namespace dwb_critics
