cmake_minimum_required(VERSION 3.8)
project(common_utils)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(people_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(nav_2d_utils REQUIRED)
find_package(nav2_core REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  include
  /usr/local/include #to find lightsfm just in case
)

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

set(libraries path_handler parameter_handler)

foreach(library ${libraries})
  add_library(${library} SHARED 
    src/${library}.cpp)
  ament_target_dependencies(${library}
    rclcpp
    rclcpp_lifecycle
    nav2_util
    nav2_costmap_2d
    geometry_msgs
    people_msgs
    nav2_msgs
    tf2_ros
    nav_2d_utils
    nav2_core
    OpenCV
  )
  target_include_directories(${library}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
      ${OpenCV_INCLUDE_DIRS}
  )
endforeach()

# install libraries
install(TARGETS
  ${libraries}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

# install headers
install(DIRECTORY include/
  DESTINATION include
 )
ament_export_include_directories(include
  ${OpenCV_INCLUDE_DIRS}
)

ament_export_libraries(${libraries})
ament_export_dependencies(${dependencies})
ament_package()
