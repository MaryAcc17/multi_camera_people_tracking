# Experiment registration macro
# Register experiment configurations with the ament resource index and installs them
#
# usage: experiment_register_configs(
#       <list_name> EXP_FILE <file> RESOURCE_INDEX <resource_index>)
#
#   :param list_name: the experiment list name
#   :type name: string
#   :param file: experiment list config file to register
#   :type file: string
#   :param resource_index: the resource index to register the configs under
#   :type resource_index: string

macro(experiment_register_configs list_name)

  cmake_parse_arguments(ARGS "" "EXPERIMENT_FILE;RESOURCE_INDEX" "" ${ARGN})
  if(ARGS_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "experiment_register_configs():called with unused "
      "arguments: ${ARGS_UNPARSED_ARGUMENTS}")
  endif()

  if("${ARGS_EXPERIMENT_FILE}" STREQUAL "")
    message(FATAL_ERROR "experiment_register_configs(): EXPERIMENT_FILE is required for experiment registration of ${list_name}")
  endif()

  # Set default resource index to "gazebo_test_experiments" if not specified otherwise
  set(resource_index "gazebo_test_experiments")
  if(NOT "${ARGS_RESOURCE_INDEX}" STREQUAL "")
    set(resource_index "${ARGS_RESOURCE_INDEX}")
    message(STATUS "experiment_register_configs(): Using RESOURCE_INDEX ${resource_index} for experiment registration of ${list_name}")
  endif()

  set(exp_file "${ARGS_EXPERIMENT_FILE}")
  _experiment_register_package_hook()

  # Append the file to the list of experiment lists for this resource index
  set(_EXPERIMENTS_${resource_index}__EXP_LISTS 
    "${_EXPERIMENTS_${resource_index}__EXP_LISTS}${list_name};${exp_file}\n")
  list(APPEND _EXPERIMENTS_PACKAGE_RESOURCE_INDICES ${resource_index})

  message(STATUS "experiment_register_configs(): Registered experiment config ${exp_file} for ${list_name}
   under ${resource_index} and installed it at share/${PROJECT_NAME}/experiments")

  install(
    FILES ${exp_file}
    DESTINATION share/${PROJECT_NAME}/experiments)
endmacro()
