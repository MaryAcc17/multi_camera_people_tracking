cmake_minimum_required(VERSION 3.8)
project(rviz_goal_pose_annotator)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rviz_common REQUIRED)
find_package(rviz_rendering REQUIRED)
find_package(rviz_ogre_vendor REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)

set(CMAKE_AUTOMOC ON)
qt5_wrap_cpp(MOC_FILES
  include/rviz_goal_pose_annotator/goal_pose_annotator_panel.hpp
  include/rviz_goal_pose_annotator/semantic_map_panel.hpp
)

add_library(goal_pose_annotator_panel SHARED
  src/goal_pose_annotator_panel.cpp
  src/semantic_map_panel.cpp
  ${MOC_FILES}
)

target_include_directories(goal_pose_annotator_panel PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(goal_pose_annotator_panel
  pluginlib
  rclcpp
  rviz_common
  rviz_rendering
  rviz_ogre_vendor
  geometry_msgs
  nav2_msgs
  ament_index_cpp
  visualization_msgs
  tf2
  tf2_geometry_msgs
)

target_link_libraries(goal_pose_annotator_panel
  Qt5::Widgets
  yaml-cpp
)

install(TARGETS goal_pose_annotator_panel
  EXPORT export_rviz_goal_pose_annotator
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(FILES rviz_common_plugins.xml
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY launch rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_targets(export_rviz_goal_pose_annotator)
pluginlib_export_plugin_description_file(rviz_common rviz_common_plugins.xml)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
