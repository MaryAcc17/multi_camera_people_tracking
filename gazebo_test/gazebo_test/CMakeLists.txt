cmake_minimum_required(VERSION 3.8)
project(gazebo_test)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(hunav_sim QUIET)
find_package(std_srvs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(gazebo_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(people_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

#display a warning if hunav_sim is not found

if(NOT hunav_sim_FOUND)
  message(WARNING "hunav_sim not found. this is fine for testing worlds and experiments, but without human simulation")
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(DIRECTORY
  launch
  behavior_trees
  rviz
  cmake
  DESTINATION share/${PROJECT_NAME}/
)


# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python scripts
install(PROGRAMS
  scripts/experiment_manager
  scripts/full_experiment
  scripts/environment_camera
  scripts/annotate_snapshot
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(
  rclpy
  std_srvs
  cv_bridge
  gazebo_msgs
  geometry_msgs
  sensor_msgs
  std_msgs
  people_msgs
  visualization_msgs
)

ament_package(CONFIG_EXTRAS gazebo_test-extras.cmake.in)
