#include <cmath>
#include "mppi_social_critics/social_critic.hpp"
#include <math.h>
#include <limits>
namespace mppi::critics

{

void SocialCritic::initialize()
{   

  using std::placeholders::_1;
  auto node = parent_.lock();
  people_sub_ = node->create_subscription<people_msgs::msg::People>(
    "people", rclcpp::SensorDataQoS(), std::bind(&SocialCritic::peopleCallback, this, _1));
  pooling_timer =
    node->create_wall_timer(
    std::chrono::milliseconds(50),
    std::bind(&SocialCritic::costmapElaboration, this));

  points_pub_ = node->create_publisher<visualization_msgs::msg::Marker>(
    "sfm/markers/obstacle_points",0);
  points_pub_ ->on_activate();
  agents_points_pub_ = node->create_publisher<visualization_msgs::msg::Marker>(
    "sfm/markers/agents_points",0);
  agents_points_pub_ ->on_activate();
  
  std::chrono::duration<int> buffer_timeout(1);
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(pooling_size_, "pooling_size", 5);
  getParam(person_radius_, "person_radius", 0.5);
  getParam(power_, "cost_power", 1);
  getParam(social_weight_, "social_weight", 3000.0);
  getParam(step_granularity, "step_grouping", 12);
  getParam(FOV, "field_of_view", 90.0);
  getParam(laser_grouping, "laser_filter", 4);
  getParam(max_robo_agent_x, "max_distance_robo_agent_x", 3.5);
  getParam(max_robo_agent_y, "max_distance_robo_agent_y", 3.5);
  getParam(laser_cutoff, "laser_distance_cut_off", 3.5);
  getParam(global_frame_, "global_frame", std::string("map"));
  RCLCPP_INFO_ONCE(
    logger_, "SocialCritic instantiated with %d power,%f weight,%d steps skipped,%f degrees FOV",
    power_,
    social_weight_, step_granularity, FOV);

  auto getParentParam = parameters_handler_->getParamGetter(parent_name_);
  getParentParam(time_steps_, "time_steps", 60);
  getParentParam(model_dt_, "model_dt", 0.05);
  // Initialize agents' vector, starting with the robot
  agents_.resize(1);
  agents_[0].desiredVelocity = 0.5f;
  agents_[0].radius = 0.35f;
  agents_[0].cyclicGoals = false;
  agents_[0].teleoperated = true;
  agents_[0].groupId = -1;
}

void SocialCritic::publish_obstacle_points(
  const std::vector<utils_lightsfm::Vector2d> & points)
{
  // function to publish the points to be used for the obstacles for the agents
  visualization_msgs::msg::Marker m;
  m.header.frame_id = global_frame_; // robot_frame_
  m.header.stamp = laser_time_;
  m.ns = "sfm_obstacle_points";
  m.type = visualization_msgs::msg::Marker::POINTS;
  m.action = visualization_msgs::msg::Marker::ADD;
  m.pose.orientation.x = 0.0;
  m.pose.orientation.y = 0.0;
  m.pose.orientation.z = 0.0;
  m.pose.orientation.w = 1.0;
  m.scale.x = 0.15;
  m.scale.y = 0.15;
  m.scale.z = 0.15;
  m.color.r = 1.0;
  m.color.g = 0.0;
  m.color.b = 0.0;
  m.color.a = 1.0;
  m.id = 1000;
  m.lifetime = rclcpp::Duration::from_seconds(0.3);
  for (utils_lightsfm::Vector2d p : points) {
    geometry_msgs::msg::Point pt;
    pt.x = p.getX();
    pt.y = p.getY();
    pt.z = 0.2;
    m.points.push_back(pt);
  }
  points_pub_->publish(m);
}

void SocialCritic::publish_agents_points(const std::vector<sfm::Agent> &agents){
  //function to publish the points of the agents used in computations
  visualization_msgs::msg::Marker a;
  a.header.frame_id = global_frame_; // robot_frame_
  a.header.stamp = people_time_;
  a.ns = "used_agents_points";
  a.type = visualization_msgs::msg::Marker::POINTS;
  a.action = visualization_msgs::msg::Marker::ADD;
  a.pose.orientation.x = 0.0;
  a.pose.orientation.y = 0.0;
  a.pose.orientation.z = 0.0;
  a.pose.orientation.w = 1.0;
  a.scale.x = 0.25;
  a.scale.y = 0.25;
  a.scale.z = 0.25;
  a.color.r = 0.0;
  a.color.g = 0.0;
  a.color.b = 1.0;
  a.color.a = 1.0;
  a.id = 1000;
  a.lifetime = rclcpp::Duration::from_seconds(0.3);
  for (size_t i = 0;i < agents.size();++i) {
    geometry_msgs::msg::Point pt;
    pt.x = agents[i].position.getX();
    pt.y = agents[i].position.getY();
    pt.z = 0.2;
    a.points.push_back(pt);
  }
  agents_points_pub_->publish(a);
}

void SocialCritic::peopleCallback(const people_msgs::msg::People::SharedPtr people) {

  RCLCPP_INFO_ONCE(logger_, "People received");
  people_mutex_.lock();
  people_ = *people;
  people_mutex_.unlock();
  
  RCLCPP_INFO_ONCE(logger_,"Time acquired");
  std::vector<sfm::Agent> agents;
  // check if people are not in odom frame
  geometry_msgs::msg::PoseStamped ps;
  for (unsigned i = 0; i < people->people.size(); i++) {
    sfm::Agent ag;
    try {
      if (people->people[i].tags.size()<1) {
        ag.id = -1;
        ag.groupId = -1;
      } else if (people->people[i].tags.size()<2) {
        ag.id = std::stoi(people->people[i].tags[0]);
        ag.groupId = -1;
      } else {
        ag.id = std::stoi(people->people[i].tags[0]);
        ag.groupId = std::stoi(people->people[i].tags[1]);
      }
    } catch (const std::exception &e) {
      RCLCPP_WARN(logger_, "PeopleCallback: bad tags for person %u: %s", i, e.what());
      ag.id = -1;
      ag.groupId = -1;
    }

    ps.header.frame_id = people->header.frame_id;
    // builtin_interfaces::msg::Time t = time_;
    people_time_ = people->header.stamp;
    ps.header.stamp = people_time_;
    ps.pose.position = people->people[i].position;
    ps.pose.position.z = 0.0;
    tf2::Quaternion quat;
    quat.setRPY(0, 0, people->people[i].position.z);
    ps.pose.orientation = tf2::toMsg(quat);
    ag.position.set(ps.pose.position.x, ps.pose.position.y);

    geometry_msgs::msg::Vector3 velocity;
    velocity.x = people->people[i].velocity.x;
    velocity.y = people->people[i].velocity.y;
    velocity.z = 0.0;

    ag.velocity.set(velocity.x, velocity.y);
    ag.linearVelocity = ag.velocity.norm();
    if (fabs(ag.linearVelocity) < 0.09) {
      ag.yaw.setRadian(tf2::getYaw(ps.pose.orientation));
    } else {
      ag.yaw = utils_lightsfm::Angle::fromRadian(
        atan2(ag.velocity.getY(), ag.velocity.getX()));
    }
    ag.angularVelocity = people->people[i].velocity.z;
    ag.radius = 0.3f;
    ag.teleoperated = false;

    // The SFM requires a local goal for each agent. We will assume that the
    // goal for people depends on its current velocity
    ag.goals.clear();
    sfm::Goal naiveGoal;
    utils_lightsfm::Vector2d v =
      ag.position + time_steps_ * model_dt_ * ag.velocity;
    naiveGoal.center.set(v.getX(), v.getY());
    naiveGoal.radius = 0.3f;
    ag.goals.push_back(naiveGoal);
    ag.desiredVelocity = 0.4f;
    agents.push_back(ag);
    RCLCPP_INFO_ONCE(logger_, "agents updated");
  }

  // Fill the obstacles of the agents
  obs_mutex_.lock();
  std::vector<utils_lightsfm::Vector2d> obs_points = obstacles_;
  obs_mutex_.unlock();
  for (unsigned int i = 0; i < agents.size(); i++) {
    agents[i].obstacles1.clear();
    agents[i].obstacles1 = obs_points;
  }

  agents_mutex_.lock();
  agents_.resize(people->people.size() + 1);
  //agents_[0].obstacles1 = obs_points;
  for (unsigned int i = 1; i < agents_.size(); i++) {
    agents_[i] = agents[i - 1];
  }
  agents_mutex_.unlock();
}


std::vector<sfm::Agent> SocialCritic::getAgents()
{
  // obtaining informations about agents to be used in scoring function
  agents_mutex_.lock();
  std::vector<sfm::Agent> agents_calc = agents_;
  agents_mutex_.unlock();
  return agents_calc;
}

double SocialCritic::computeSocialWork(const std::vector<sfm::Agent> & agents)
{
  // The social work of the robot
  double wr = agents[0].forces.socialForce.norm();

  // Compute the social work provoked by the robot in the other agents
  std::vector<sfm::Agent> agent_robot;
  agent_robot.push_back(agents[0]);
  double wp = 0.0;
  for (unsigned int i = 1; i < agents.size(); i++) {
    sfm::Agent agent = agents[i];
    sfm::SFM.computeForces(agent, agent_robot);
    wp += agent.forces.socialForce.norm();
  }
  return wr + wp;
}

void SocialCritic::costmapElaboration(){
  std::vector<utils_lightsfm::Vector2d> obs_points;
  RCLCPP_DEBUG(logger_, "Costmap elaboration");
  // check if the agent is in the local costmap and update the obstacles
  std::vector<geometry_msgs::msg::Point> obstacles;
  // Select a coarser resolution for the local costmap and retrieve the obstacles
  // select a resolution that is a multiple of the global costmap resolution
  nav2_costmap_2d::Costmap2D costmap = maxPooling(costmap_, pooling_size_);
  double resolution = costmap.getResolution();
  // selec n points in the local costmap and check if they are obstacles
  RCLCPP_DEBUG(
    logger_, "Costmap resolution: %f", resolution);
  for (unsigned int i = 0; i < costmap.getSizeInCellsX(); i++) {
    for (unsigned int j = 0; j < costmap.getSizeInCellsY(); j++) {
      unsigned int mx, my;
      RCLCPP_DEBUG(
        logger_, "Cell: (%d, %d)", i, j);
      auto index = costmap.getIndex(i, j);
      costmap.indexToCells(index, mx, my);
      if (costmap.getCost(mx, my) == nav2_costmap_2d::LETHAL_OBSTACLE) {
        double wx, wy;
        // check if the obstacle is not a person
        costmap.mapToWorld(mx, my, wx, wy);
        obstacles.push_back(geometry_msgs::msg::Point());
        obstacles.back().x = wx;
        obstacles.back().y = wy;
        obs_points.push_back(utils_lightsfm::Vector2d(wx, wy));
      }
    }
  }

  RCLCPP_DEBUG(
    logger_, "Number of obstacles: %lu", obs_points.size());
  obs_mutex_.lock();
  obstacles_ = obs_points;
  obs_mutex_.unlock();
}


void SocialCritic::score(CriticData & data)
{
  //using xt::evaluation_strategy::immediate;
  if (!enabled_) {
    RCLCPP_INFO_ONCE(logger_, "agent critic disabled!");
    return;

  }
  std::vector<sfm::Agent> myagents = getAgents();
  if (myagents.empty()) {
    RCLCPP_WARN(logger_, "No agents available for scoring");
    return;
  }
  agents_mutex_.lock();
  sfm::Agent agent = myagents[0];
  agents_mutex_.unlock();

  agent.position.set(data.state.pose.pose.position.x, data.state.pose.pose.position.y);
  agent.yaw =
    utils_lightsfm::Angle::fromRadian(tf2::getYaw(data.state.pose.pose.orientation));

  agent.linearVelocity =
    std::sqrt(
    data.state.speed.linear.x * data.state.speed.linear.x +
    data.state.speed.linear.y * data.state.speed.linear.y);
  agent.angularVelocity = data.state.speed.angular.z;

  // The velocity in the odom messages is in the robot local frame!!!
  geometry_msgs::msg::Vector3 velocity;
  velocity.x = data.state.speed.linear.x;
  velocity.y = data.state.speed.linear.y;

  agent.velocity.set(velocity.x, velocity.y);
  agents_mutex_.lock();
  myagents[0] = agent;
  //obstacle_agents = myagents;
  agents_mutex_.unlock();
  // select the agents to be used in the computation
  std::vector<sfm::Agent> used_agents;
  if (myagents.empty()) {
    RCLCPP_WARN(logger_, "no agent acquired!");
  }

  //auto && social_cost = xt::xtensor<float, 1>::from_shape({data.costs.shape(0)});
  //social_cost.fill(0.0f);
  Eigen::ArrayXf social_cost = Eigen::ArrayXf::Zero(data.costs.size());
  float vx_i = 0.0f;
  float vy_i = 0.0f;
  const size_t num_traj = data.trajectories.x.rows();
  const size_t traj_len = data.trajectories.x.cols() / step_granularity;
  float diff_x;
  float diff_y;
  float dist_x;
  float dist_y;
  utils_lightsfm::Vector2d distance_vector;
  utils_lightsfm::Vector2d robot_vector;
  utils_lightsfm::Angle robo_agent_angle;
  double FOV_radians = FOV / 360 * M_PI;
  double value_ang;

  // bool variable not to use the FOV filter
  bool use_fov_filter = FOV > 0.0;

  robot_vector.set(myagents[0].yaw.cos(), myagents[0].yaw.sin());
  //float dist_;
  short int counter_ = 1;
  used_agents.push_back(myagents[0]);

  for (size_t z = 1; z < myagents.size(); ++z) {
    diff_x = myagents[z].position.getX() - myagents[0].position.getX();
    diff_y = myagents[z].position.getY() - myagents[0].position.getY();
    distance_vector.set(diff_x, diff_y);
    robo_agent_angle = distance_vector.angleTo(robot_vector);
    value_ang = std::abs(robo_agent_angle.toRadian());

    bool inside_fov =!use_fov_filter || value_ang < FOV_radians;
    
    //square filter area
    dist_x = std::abs(diff_x);
    dist_y = std::abs(diff_y);
    if (dist_x < max_robo_agent_x && dist_y < max_robo_agent_y && inside_fov) {
      ++counter_;
      used_agents.push_back(myagents[z]);
    }
  }
  publish_agents_points(used_agents);
  std::vector<utils_lightsfm::Vector2d> new_obstacles;
  for (auto & point : obstacles_) {
    bool agent_point = false;
    for (auto & agent : used_agents) {
      if (hypotf(
          point.getX() - agent.position.getX(),
          point.getY() - agent.position.getY()) < person_radius_)
      {
        RCLCPP_INFO(logger_, "obstacle detected");
        agent_point = true;
        break;
      }
    }
    if (!agent_point) {
      new_obstacles.push_back(point);
    }

  }
  for (unsigned int i = 0; i < used_agents.size(); i++) {
    used_agents[i].obstacles1.clear();
    used_agents[i].obstacles1 = new_obstacles;
  }
  publish_obstacle_points(new_obstacles);

  for (size_t i = 0; i < num_traj; ++i) {
    const auto & traj = data.trajectories;
    double social_work = 0.0;
    double social_step = 0.0;
    sfm::Goal g;
    g.center.set(traj.x(i, traj_len), traj.y(i, traj_len));
    g.radius = 0.20;
    used_agents[0].goals.clear();
    used_agents[0].goals.push_back(g);
    std::vector<sfm::Agent> updating_agents = used_agents;
    for (size_t j = 0; j < traj_len; ++j) {
      //compute forces for this step
      sfm::SFM.computeForces(updating_agents);
      // update agents position for this step
      sfm::SFM.updatePosition(updating_agents, 0.05f * step_granularity);
      // update robot position for this time step
      updating_agents[0].position.set(
        traj.x(i, step_granularity * j),
        traj.y(i, step_granularity * j));
      updating_agents[0].yaw = utils_lightsfm::Angle::fromRadian(traj.yaws(i, step_granularity * j));
      // update robot velocity for this time step
      vx_i =
        (-traj.x(
          i,
          step_granularity * j) +
        traj.x(i, step_granularity * (j + 1) - 1)) / (0.05f * step_granularity);
      vy_i =
        (-traj.y(
          i,
          step_granularity * j) +
        traj.y(i, step_granularity * (j + 1) - 1)) / (0.05f * step_granularity);
      updating_agents[0].velocity.set(vx_i, vy_i);
      updating_agents[0].angularVelocity =
        (-traj.yaws(
          i,
          step_granularity * j) +
        traj.yaws(i, step_granularity * (j + 1) - 1)) / (0.05f * step_granularity);
      updating_agents[0].linearVelocity = hypotf(vx_i, vy_i);
      //compute social work for this step, add it to overall work
      social_step = computeSocialWork(updating_agents);
      social_work += social_step;
    }


    social_cost[i] = social_work;
    //RCLCPP_INFO(logger_,"traj has social cost %f",social_cost[i]);
  }


  data.costs += (social_cost * social_weight_ / traj_len).pow(power_);
}
nav2_costmap_2d::Costmap2D SocialCritic::maxPooling(
  nav2_costmap_2d::Costmap2D * costmap,
  unsigned int pooling_size)
{
  RCLCPP_DEBUG(
    logger_, "Max pooling");
  // Create a new costmap with a size reduced by pooling_size
  unsigned int size_x = costmap->getSizeInCellsX() % pooling_size == 0 ?
    costmap->getSizeInCellsX() / pooling_size :
    costmap->getSizeInCellsX() / pooling_size + 1;
  unsigned int size_y = costmap->getSizeInCellsY() % pooling_size == 0 ?
    costmap->getSizeInCellsY() / pooling_size :
    costmap->getSizeInCellsY() / pooling_size + 1;

  nav2_costmap_2d::Costmap2D new_costmap = nav2_costmap_2d::Costmap2D(
    size_x,
    size_y,
    costmap->getResolution() * pooling_size,
    costmap->getOriginX(), costmap->getOriginY(), costmap->getDefaultValue());
  // Initialize the new costmap
  new_costmap.setDefaultValue(new_costmap.getDefaultValue());
  new_costmap.resetMap(0, 0, new_costmap.getSizeInCellsX(), new_costmap.getSizeInCellsY());

  // Apply the max pooling
  for (unsigned int i = 0; i < costmap->getSizeInCellsX(); i++) {
    for (unsigned int j = 0; j < costmap->getSizeInCellsY(); j++) {
      unsigned int mx, my;
      auto index = costmap->getIndex(i, j);
      costmap->indexToCells(index, mx, my);
      unsigned int new_mx = mx / pooling_size;
      unsigned int new_my = my / pooling_size;

      if (costmap->getCost(mx, my) > new_costmap.getCost(new_mx, new_my)) {
        new_costmap.setCost(new_mx, new_my, costmap->getCost(mx, my));
      }
    }
  }


  RCLCPP_DEBUG(
    logger_, "Max pooling done");
  return new_costmap;
}


}// namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(
  mppi::critics::SocialCritic,
  mppi::critics::CriticFunction)
