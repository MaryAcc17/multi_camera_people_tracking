# Navigator registration macro
# Register navigator configurations with the ament resource index and installs them
#
# usage: register_navigator_configs(
#       <list_name> EXP_FILE <file> RESOURCE_INDEX <resource_index>)
#
#   :param list_name: the navigator list name
#   :type name: string
#   :param file: navigator list config file to register
#   :type file: string
#   :param resource_index: the resource index to register the configs under
#   :type resource_index: string

macro(register_navigator_configs list_name)

  cmake_parse_arguments(ARGS "" "NAVIGATOR_FILE;RESOURCE_INDEX" "" ${ARGN})
  if(ARGS_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "register_navigator_configs():called with unused "
      "arguments: ${ARGS_UNPARSED_ARGUMENTS}")
  endif()

  if("${ARGS_NAVIGATOR_FILE}" STREQUAL "")
    message(FATAL_ERROR "register_navigator_configs(): NAVIGATOR_FILE is required for navigator registration of ${list_name}")
  endif()

  # Set default resource index to "gazebo_test_navigators" if not specified otherwise
  set(resource_index "gazebo_test_navigators")
  if(NOT "${ARGS_RESOURCE_INDEX}" STREQUAL "")
    set(resource_index "${ARGS_RESOURCE_INDEX}")
    message(STATUS "register_navigator_configs(): Using RESOURCE_INDEX ${resource_index} for navigator registration of ${list_name}")
  endif()

  set(nav_file "${ARGS_NAVIGATOR_FILE}")
  # Make CMake re-run configure if the navigator YAML file changes
  set(nav_path "${CMAKE_CURRENT_SOURCE_DIR}/${nav_file}")
  if(EXISTS "${nav_path}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${nav_path}")
  endif()
  _experiment_register_package_hook()

  # Append the file to the list of navigator lists for this resource index
  set(_EXPERIMENTS_${resource_index}__EXP_LISTS 
    "${_EXPERIMENTS_${resource_index}__EXP_LISTS}${list_name};${nav_file}\n")
  list(APPEND _EXPERIMENTS_PACKAGE_RESOURCE_INDICES ${resource_index})

  message(STATUS "register_navigator_configs(): Registered navigator config ${nav_file} for ${list_name}
   under ${resource_index} and installed it at share/${PROJECT_NAME}/navigators")

  install(
    FILES ${nav_file}
    DESTINATION share/${PROJECT_NAME}/navigators)
endmacro()
