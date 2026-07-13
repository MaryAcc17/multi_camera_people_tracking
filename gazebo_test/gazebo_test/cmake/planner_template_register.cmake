# Planner template registration macro
# Register planner configurations with the ament resource index and installs them
#
# usage: register_planner_template(
#       <planner_template_name> PLANNER_TEMPLATE_FILE <file> RESOURCE_INDEX <resource_index>)
#
#   :param planner_template_name: the planner name
#   :type planner_template_name: string
#   :param file: planner list config file to register
#   :type file: string
#   :param resource_index: the resource index to register the configs under
#   :type resource_index: string

macro(register_planner_template planner_template_name)

  cmake_parse_arguments(ARGS "" "PLANNER_TEMPLATE_FILE;RESOURCE_INDEX" "" ${ARGN})
  if(ARGS_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "register_planner_template():called with unused "
      "arguments: ${ARGS_UNPARSED_ARGUMENTS}")
  endif()

  if("${ARGS_PLANNER_TEMPLATE_FILE}" STREQUAL "")
    message(FATAL_ERROR "register_planner_template(): PLANNER_TEMPLATE_FILE is required for planner registration of ${planner_template_name}")
  endif()

  # Set default resource index to "gazebo_test_planner_templates" if not specified otherwise
  set(resource_index "gazebo_test_planner_templates")
  if(NOT "${ARGS_RESOURCE_INDEX}" STREQUAL "")
    set(resource_index "${ARGS_RESOURCE_INDEX}")
    message(STATUS "register_planner_template(): Using RESOURCE_INDEX ${resource_index} for planner registration of ${planner_template_name}")
  endif()

  set(planner_template_file "${ARGS_PLANNER_TEMPLATE_FILE}")
  _experiment_register_package_hook()

  # Append the file to the list of planner lists for this resource index
  set(_PLANNERS_${resource_index}__PLANNER_LISTS 
    "${_PLANNERS_${resource_index}__PLANNER_LISTS}${planner_template_name};${planner_template_file}\n")
  list(APPEND _PLANNERS_PACKAGE_RESOURCE_INDICES ${resource_index})

  message(STATUS "register_planner_template(): Registered planner config ${planner_template_file} for ${planner_template_name} under ${resource_index} and installed it at share/${PROJECT_NAME}/templates/planners")

  install(
    FILES ${planner_template_file}
    DESTINATION share/${PROJECT_NAME}/templates/planners)
endmacro()


# # Register multiple planners from a YAML configuration file
# This macro reads a YAML file containing a list of planners and their associated
# template files, then registers each planner template using the
# `register_planner_template` macro.
# # usage: register_planners_templates_from_yaml(
#       YAML_FILE <yaml_file> RESOURCE_INDEX <resource_index> TEMPLATES_BASE_DIR <base_dir>)
# #
#   :param yaml_file: path to the YAML file listing planners and their templates
#   :type yaml_file: string
#   :param resource_index: the resource index to register the configs under
#   :type resource_index: string
#   :param templates_base_dir: base directory to prepend to template file paths
#   :type templates_base_dir: string

macro(register_planners_templates_from_yaml)
  cmake_parse_arguments(ARGS "" "YAML_FILE;RESOURCE_INDEX;TEMPLATES_BASE_DIR" "" ${ARGN})
  
  if("${ARGS_YAML_FILE}" STREQUAL "")
    message(FATAL_ERROR "register_planners_templates_from_yaml(): YAML_FILE is required")
  endif()

  set(yaml_file "${ARGS_YAML_FILE}")
  # Make CMake re-run configure if the YAML file changes
  set(yaml_path "${CMAKE_CURRENT_SOURCE_DIR}/${yaml_file}")
  if(EXISTS "${yaml_path}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${yaml_path}")
  endif()
  set(resource_index "gazebo_test_planner_templates")
  set(templates_base_dir "templates/")
  
  if(NOT "${ARGS_RESOURCE_INDEX}" STREQUAL "")
    set(resource_index "${ARGS_RESOURCE_INDEX}")
  endif()
  
  if(NOT "${ARGS_TEMPLATES_BASE_DIR}" STREQUAL "")
    set(templates_base_dir "${ARGS_TEMPLATES_BASE_DIR}")
  endif()

  # Read the YAML file
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${yaml_file}" yaml_content)
  
  # Extract planner names from planners_templates list
  string(REGEX MATCHALL "- ([a-zA-Z0-9_]+)" planner_matches "${yaml_content}")
  
  set(registered_count 0)
  foreach(match ${planner_matches})
    string(REGEX REPLACE "- " "" planner_template_name "${match}")
    
    # Look for the planner's template configuration
    string(REGEX MATCH "${planner_template_name}:[\n\r ]*template: *'([^']+)'" template_match "${yaml_content}")
    
    if(template_match)
      set(template_path "${CMAKE_MATCH_1}")
      
      # Ensure template path includes base directory
      if(NOT template_path MATCHES "^${templates_base_dir}")
        set(template_path "${templates_base_dir}${template_path}")
      endif()
      
      if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${template_path}")
        message(STATUS "register_planners_templates_from_yaml(): Registering planner ${planner_template_name} with template ${template_path} under ${resource_index}")
        register_planner_template(${planner_template_name}
          PLANNER_TEMPLATE_FILE ${template_path}
          RESOURCE_INDEX ${resource_index}
        )
        math(EXPR registered_count "${registered_count} + 1")
      else()
        message(WARNING "Template file not found for ${planner_template_name}: ${CMAKE_CURRENT_SOURCE_DIR}/${template_path}")
      endif()
    else()
      message(WARNING "No template configuration found for planner: ${planner_template_name}")
    endif()
  endforeach()

  message(STATUS "register_planners_templates_from_yaml(): Successfully registered ${registered_count} planners from ${yaml_file}")
endmacro()
