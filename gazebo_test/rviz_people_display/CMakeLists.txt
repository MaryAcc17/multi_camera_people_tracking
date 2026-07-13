cmake_minimum_required(VERSION 3.8)
project(rviz_people_display)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---- dependencies ----
find_package(ament_cmake REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rviz_common REQUIRED)
find_package(rviz_rendering REQUIRED)
find_package(rviz_ogre_vendor REQUIRED)
find_package(people_msgs REQUIRED)

# ---- Qt MOC generation ----
set(CMAKE_AUTOMOC ON)
qt5_wrap_cpp(MOC_FILES
  include/rviz_people_display/people_display.hpp
)

# ---- library ----
add_library(people_display SHARED
  src/people_display.cpp
  ${MOC_FILES}
)
target_include_directories(people_display PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(people_display
  pluginlib
  rviz_common
  rviz_rendering
  rviz_ogre_vendor
  people_msgs
)

# ---- install ----
install(TARGETS people_display
  EXPORT export_rviz_people_display
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

install(DIRECTORY icons/
  DESTINATION share/${PROJECT_NAME}/icons
)

# ---- exports ----
ament_export_include_directories(include)
ament_export_targets(export_rviz_people_display)
pluginlib_export_plugin_description_file(rviz_common rviz_common_plugins.xml)

# ---- tests ----
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
