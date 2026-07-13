cmake_minimum_required(VERSION 3.8)
project(gazebo_navigators)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(gazebo_sim REQUIRED)
find_package(gazebo_test REQUIRED)

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


# Register controllers from YAML configuration
register_controller_templates_from_yaml(
  YAML_FILE templates/controller_templates.yaml
)

register_navigator_templates_from_yaml(
  YAML_FILE templates/navigator_templates.yaml
)

register_planners_templates_from_yaml(
  YAML_FILE templates/planner_templates.yaml
)

register_costmap_layer_templates_from_yaml(
  YAML_FILE templates/costmap_layer_templates.yaml
)

register_navigator_configs(
  ${PROJECT_NAME}
  NAVIGATOR_FILE navigators/navigators.yaml
)

ament_package()
  