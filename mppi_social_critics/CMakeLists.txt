cmake_minimum_required(VERSION 3.8)
project(mppi_social_critics)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


#add_definitions(-DXTENSOR_ENABLE_XSIMD)
#add_definitions(-DXTENSOR_USE_XSIMD)
#
#set(XTENSOR_USE_TBB 0)
#set(XTENSOR_USE_OPENMP 0)
#set(XTENSOR_USE_XSIMD 1)


find_package(ament_cmake REQUIRED)
#find_package(xtensor REQUIRED)
#find_package(xsimd REQUIRED)
find_package(people_msgs REQUIRED)
find_package(nav2_mppi_controller REQUIRED)

include_directories(
  /workspaces/hunavsim_devcontainer/src/navigation2/nav2_mppi_controller/include
  /usr/local/include
  include
)

set(dependencies_pkgs
  rclcpp
  nav2_common
  pluginlib
  tf2
  geometry_msgs
  visualization_msgs
  people_msgs
  nav_msgs
  nav2_core
  nav2_costmap_2d
  nav2_util
  tf2_geometry_msgs
  tf2_eigen
  tf2_ros
  #xsimd
  common_utils
)

foreach(pkg IN LISTS dependencies_pkgs)
  find_package(${pkg} REQUIRED)
endforeach()

include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-mno-avx512f" COMPILER_SUPPORTS_AVX512)
check_cxx_compiler_flag("-msse4.2" COMPILER_SUPPORTS_SSE4)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)


if(COMPILER_SUPPORTS_AVX512)
  add_compile_options(-mno-avx512f)
endif()

if(COMPILER_SUPPORTS_SSE4)
  add_compile_options(-msse4.2)
endif()

if(COMPILER_SUPPORTS_AVX2)
  add_compile_options(-mavx2)
endif()

if(COMPILER_SUPPORTS_FMA)
  add_compile_options(-mfma)
endif()

# If building one the same hardware to be deployed on, try `-march=native`!
add_compile_options(-O2 -finline-limit=10000000 -ffp-contract=fast  -mtune=generic)

add_library(${PROJECT_NAME} SHARED
  src/social_critic.cpp
)

target_compile_options(${PROJECT_NAME} PUBLIC -fconcepts)
#target_include_directories(${PROJECT_NAME} PUBLIC ${xsimd_INCLUDE_DIRS})  #  ${OpenMP_INCLUDE_DIRS}
#target_link_libraries(${PROJECT_NAME} xtensor xtensor::optimize xtensor::use_xsimd)
ament_target_dependencies(${PROJECT_NAME} ${dependencies_pkgs})

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include/
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


ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(${dependencies_pkgs})
ament_export_include_directories(include)
install(FILES social_critics.xml
  DESTINATION share/${PROJECT_NAME})

pluginlib_export_plugin_description_file(nav2_mppi_controller social_critics.xml)

ament_package()
