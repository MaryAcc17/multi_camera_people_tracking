#ifndef MPPI_SOCIAL_CRITICS__SOCIAL_CRITIC_HPP_
#define MPPI_SOCIAL_CRITICS__SOCIAL_CRITIC_HPP_

#include "nav2_mppi_controller/critic_function.hpp"
#include "nav2_mppi_controller/models/state.hpp"
#include <sensor_msgs/msg/laser_scan.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include "lightsfm/angle.hpp"
#include "lightsfm/sfm.hpp"
#include "lightsfm/vector2d.hpp"
#include <people_msgs/msg/people.hpp>
#include <people_msgs/msg/person.hpp>
#include <memory>
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "rclcpp/rclcpp.hpp"
#include <chrono>
#include <functional>
#include <math.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "nav2_util/node_utils.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <nav_msgs/msg/odometry.hpp>


#include "std_msgs/msg/string.hpp"

#include "people_msgs/msg/people.hpp"
#include "people_msgs/msg/person.hpp"

#include "nav2_costmap_2d/costmap_2d_publisher.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "tf2_ros/create_timer_ros.h"
#include "tf2_ros/message_filter.h"
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <boost/thread.hpp>
#include "nav2_costmap_2d/layer.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"


namespace mppi::critics
{

/**
 * @class mppi::critics::SocialCritic
 * @brief Critic objective function for avoiding employing a social behavior, allowing it to deviate off
 * the planned path considering the repulsive forces applied by the human agents.
 */
class SocialCritic : public CriticFunction
{
public:
  /**
    * @brief Initialize critic
    */
  void initialize() override;

  /**
    * @brief Scoring function
    * @param data data to be used for scoring function
    */
  void score(CriticData & data) override;

  /**
   * @brief function which obtains information about people and transforms them into agents
   * @param people information about agents on scene
   */
  void peopleCallback(const people_msgs::msg::People::SharedPtr people);

  /**
   * @brief function which calls the marker publisher used for obstacles' points
   * @param points points transformed in map frame
   */
  void publish_obstacle_points(const std::vector<utils_lightsfm::Vector2d> & points);
  /**
   * @brief function which calls the marker publisher to highlight the used agents
   * @param agents to be used in the scene
   */

  void publish_agents_points(const std::vector<sfm::Agent> & agents);
  /**
   * @brief function which computes the total social work at a certain time instant
   * @param agents information about agents (position, velocity, local goal)
   */
  double computeSocialWork(const std::vector<sfm::Agent> & agents);
  /**
   * @brief function which locks information about agents to be used in scoring function
   */
  std::vector<sfm::Agent> getAgents();
  /**
   * @brief function connected to timer to perform costmap elaboration and detect obstacles
   */
  void costmapElaboration();
  /**
   * @brief function which performs max pooling on the costmap
   * @param costmap costmap to be used for max pooling
   * @param pooling_size integer value to be used for max pooling
   */
  nav2_costmap_2d::Costmap2D maxPooling(
    nav2_costmap_2d::Costmap2D * costmap,
    unsigned int pooling_size);

protected:
  //parameters to be set in config file
  std::string global_frame_;
  int step_granularity;
  int pooling_size_;
  float person_radius_;
  double FOV;
  unsigned int laser_grouping;
  float laser_cutoff;
  float max_robo_agent_x;
  float max_robo_agent_y;
  int time_steps_;
  float model_dt_;
  //useful stuff for the people callback
  people_msgs::msg::People people_;
  std::mutex people_mutex_;
  people_msgs::msg::People people_list_;
  builtin_interfaces::msg::Time laser_time_, people_time_;
  //the subscriber to be used
  rclcpp::Subscription<people_msgs::msg::People>::SharedPtr people_sub_;
  //the timer connected to the costmap elaboration
  rclcpp::TimerBase::SharedPtr pooling_timer;
  //two publishers for markers associated to obstacles and agents
  std::shared_ptr<
    rclcpp_lifecycle::LifecyclePublisher<visualization_msgs::msg::Marker>>
  points_pub_;
  std::shared_ptr<
    rclcpp_lifecycle::LifecyclePublisher<visualization_msgs::msg::Marker>>
  agents_points_pub_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

  std::vector<sfm::Agent> agents_; // 0: robot, 1..: Others
  std::vector<sfm::Agent> agents_calc;
  std::mutex ppl_message_mutex_;
  std::mutex agents_mutex_;

  std::vector<utils_lightsfm::Vector2d> obstacles_;
  std::mutex obs_mutex_;
  std::mutex odom_mutex_;
  // rclcpp::Time last_laser_;
  //nav_msgs::msg::Odometry base_odom_;
  //cost function parameters

  unsigned int power_{0};
  float social_weight_{0};

};

}  // namespace mppi::critics

#endif  // MPPI_SOCIAL_CRITICS__SOCIAL_CRITIC_HPP_
