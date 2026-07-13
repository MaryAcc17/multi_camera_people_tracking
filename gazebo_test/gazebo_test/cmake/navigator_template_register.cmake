# Controller template registration macro
# Register navigator configurations with the ament resource index and installs them
#
# usage: register_navigator_template(
#       <navigator_template_name> CONTROLLER_TEMPLATE_FILE <file> RESOURCE_INDEX <resource_index>)
#
#   :param navigator_template_name: the navigator name
#   :type navigator_template_name: string
#   :param file: navigator list config file to register
#   :type file: string
#   :param resource_index: the resource index to register the configs under
#   :type resource_index: string

macro(register_navigator_template navigator_template_name)

  cmake_parse_arguments(ARGS "" "CONTROLLER_TEMPLATE_FILE;RESOURCE_INDEX" "" ${ARGN})
  if(ARGS_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "register_navigator_template():called with unused "
      "arguments: ${ARGS_UNPARSED_ARGUMENTS}")
  endif()

  if("${ARGS_CONTROLLER_TEMPLATE_FILE}" STREQUAL "")
    message(FATAL_ERROR "register_navigator_template(): CONTROLLER_TEMPLATE_FILE is required for navigator registration of ${navigator_template_name}")
  endif()

  # Set default resource index to "gazebo_test_navigator_templates" if not specified otherwise
  set(resource_index "gazebo_test_navigator_templates")
  if(NOT "${ARGS_RESOURCE_INDEX}" STREQUAL "")
    set(resource_index "${ARGS_RESOURCE_INDEX}")
    message(STATUS "register_navigator_template(): Using RESOURCE_INDEX ${resource_index} for navigator registration of ${navigator_template_name}")
  endif()

  set(navigator_template_file "${ARGS_CONTROLLER_TEMPLATE_FILE}")
  _experiment_register_package_hook()

  # Append the file to the list of navigator lists for this resource index
  set(_CONTROLLERS_${resource_index}__CONTROLLER_LISTS 
    "${_CONTROLLERS_${resource_index}__CONTROLLER_LISTS}${navigator_template_name};${navigator_template_file}\n")
  list(APPEND _CONTROLLERS_PACKAGE_RESOURCE_INDICES ${resource_index})

  message(STATUS "register_navigator_template(): Registered navigator config ${navigator_template_file} for ${navigator_template_name} under ${resource_index} and installed it at share/${PROJECT_NAME}/templates/navigators")

  install(
    FILES ${navigator_template_file}
    DESTINATION share/${PROJECT_NAME}/templates/nav_templates)
endmacro()


# # Register multiple navigators from a YAML configuration file
# This macro reads a YAML file containing a list of navigators and their associated
# template files, then registers each navigator template using the
# `register_navigator_template` macro.
# # usage: register_navigator_templates_from_yaml(
#       YAML_FILE <yaml_file> RESOURCE_INDEX <resource_index> TEMPLATES_BASE_DIR <base_dir>)
# #
#   :param yaml_file: path to the YAML file listing navigators and their templates
#   :type yaml_file: string
#   :param resource_index: the resource index to register the configs under
#   :type resource_index: string
#   :param templates_base_dir: base directory to prepend to template file paths
#   :type templates_base_dir: string

macro(register_navigator_templates_from_yaml)
  cmake_parse_arguments(ARGS "" "YAML_FILE;RESOURCE_INDEX;TEMPLATES_BASE_DIR" "" ${ARGN})
  
  if("${ARGS_YAML_FILE}" STREQUAL "")
    message(FATAL_ERROR "register_navigator_templates_from_yaml(): YAML_FILE is required")
  endif()

  set(yaml_file "${ARGS_YAML_FILE}")
  # Make CMake re-run configure if the YAML file changes
  set(yaml_path "${CMAKE_CURRENT_SOURCE_DIR}/${yaml_file}")
  if(EXISTS "${yaml_path}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${yaml_path}")
  endif()
  set(resource_index "gazebo_test_navigator_templates")
  set(templates_base_dir "templates/")
  
  if(NOT "${ARGS_RESOURCE_INDEX}" STREQUAL "")
    set(resource_index "${ARGS_RESOURCE_INDEX}")
  endif()
  
  if(NOT "${ARGS_TEMPLATES_BASE_DIR}" STREQUAL "")
    set(templates_base_dir "${ARGS_TEMPLATES_BASE_DIR}")
  endif()

  # Read the YAML file
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${yaml_file}" yaml_content)
  
  # Extract navigator names from navigators_templates list
  string(REGEX MATCHALL "- ([a-zA-Z0-9_]+)" navigator_matches "${yaml_content}")
  
  set(registered_count 0)
  foreach(match ${navigator_matches})
    string(REGEX REPLACE "- " "" navigator_template_name "${match}")
    
    # Look for the navigator's template configuration
    string(REGEX MATCH "${navigator_template_name}:[\n\r ]*template: *'([^']+)'" template_match "${yaml_content}")
    
    if(template_match)
      set(template_path "${CMAKE_MATCH_1}")
      
      # Ensure template path includes base directory
      if(NOT template_path MATCHES "^${templates_base_dir}")
        set(template_path "${templates_base_dir}${template_path}")
      endif()
      
      if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${template_path}")
        message(STATUS "register_navigator_templates_from_yaml(): Registering navigator ${navigator_template_name} with template ${template_path} under ${resource_index}")
        register_navigator_template(${navigator_template_name}
          CONTROLLER_TEMPLATE_FILE ${template_path}
          RESOURCE_INDEX ${resource_index}
        )
        math(EXPR registered_count "${registered_count} + 1")
      else()
        message(WARNING "Template file not found for ${navigator_template_name}: ${CMAKE_CURRENT_SOURCE_DIR}/${template_path}")
      endif()
    else()
      message(WARNING "No template configuration found for navigator: ${navigator_template_name}")
    endif()
  endforeach()

  message(STATUS "register_navigator_templates_from_yaml(): Successfully registered ${registered_count} navigators from ${yaml_file}")
endmacro()
