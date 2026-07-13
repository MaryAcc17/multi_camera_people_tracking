#ifndef DWB_ADDITIONAL_CRITICS__SOCIAL_FORCE_HPP_
#define DWB_ADDITIONAL_CRITICS__SOCIAL_FORCE_HPP_

#include <string>

#include "dwb_critics/map_grid.hpp"
// Social Force Model
#include <lightsfm/angle.hpp>
#include <lightsfm/sfm.hpp>
#include <lightsfm/vector2d.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "common_utils/hunav_utils.hpp"
#include "common_utils/utils.hpp"
#include "geometry_msgs/msg/pose2_d.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "people_msgs/msg/people.hpp"

namespace dwb_critics
{

/**
 * @class SocialForceCritic
 * @brief Penalize trajectories based on the social force model
 *
 * This critic is based on the social force model, which is a model of pedestrian behavior.
 */
class SocialForceCritic : public dwb_core::TrajectoryCritic
{
public:
  SocialForceCritic() {}
  ~SocialForceCritic() override;

  /**
   * @brief Initialize the critic, including reading parameters
   */
  void onInit() override;

  /**
    * @brief prepare the critic for scoring
    * @param pose The current robot pose
    * @param vel The current robot velocity
    * @param goal The ultimate robot goal
    * @param global_plan The plan that the robot is attempting to follow
    */
  bool prepare(
    const geometry_msgs::msg::Pose2D & pose, const nav_2d_msgs::msg::Twist2D & vel,
    const geometry_msgs::msg::Pose2D & goal, const nav_2d_msgs::msg::Path2D & global_plan) override;

  /**
   * @brief Score a trajectory based on the social force model
   */
  double scoreTrajectory(const dwb_msgs::msg::Trajectory2D & traj) override;

  /**
   * @brief  compute the social work cost
   * @param agents The vector of social agents
   */
  double computeSocialWork(const std::vector<sfm::Agent> & agents);

  /**
   * @brief  get the people callback
   */

  void peopleCallback(const std::shared_ptr<people_msgs::msg::People> msg);

  /**
   * @brief  get the agents
   */
  std::vector<sfm::Agent> getAgents();

  /**
   * @brief  initialize the robot agent
   * @param agent The robot agent
   */
  void initializeRobot(sfm::Agent & agent);

  /**
   * @brief  update the robot agent
   * @param robot The robot agent
   * @param traj The robot trajectory
   * @param i The index of the trajectory
   */
  void updateAgentRobot(
    sfm::Agent & robot, const dwb_msgs::msg::Trajectory2D & traj, size_t index);

  /**
   * @brief  update the agents
   * @param agents The vector of social agents
   */
  void updateAgents(const std::vector<sfm::Agent> & agents);

  /**
   * @brief  get local goal
   */
  sfm::Goal getGoal();

private:
  std::shared_ptr<std::vector<sfm::Agent>> agents_, agents_calc_;
  std::mutex agents_mutex_;
  std::shared_ptr<sfm::Goal> goal_;
  nav2_costmap_2d::Costmap2D * costmap_;
  std::unique_ptr<nav2_costmap_2d::Costmap2D> pool_costmap_;
  std::vector<utils::Vector2d> obstacles_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
  double person_radius_, robot_radius_, person_velocity_, naive_goal_time_, robot_linear_velocity_,
    fov_angle_, new_resolution_;
  rclcpp::Subscription<people_msgs::msg::People>::SharedPtr people_sub_;
  rclcpp_lifecycle::LifecyclePublisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Logger logger_{rclcpp::get_logger("SocialForceCritic")};
};

}  // namespace dwb_critics
#endif  // DWB_ADDITIONAL_CRITICS__SOCIAL_FORCE_HPP_
