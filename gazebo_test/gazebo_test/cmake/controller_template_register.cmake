# Controller template registration macro
# Register controller configurations with the ament resource index and installs them
#
# usage: register_controller_template(
#       <controller_template_name> CONTROLLER_TEMPLATE_FILE <file> RESOURCE_INDEX <resource_index>)
#
#   :param controller_template_name: the controller name
#   :type controller_template_name: string
#   :param file: controller list config file to register
#   :type file: string
#   :param resource_index: the resource index to register the configs under
#   :type resource_index: string

macro(register_controller_template controller_template_name)

  cmake_parse_arguments(ARGS "" "CONTROLLER_TEMPLATE_FILE;RESOURCE_INDEX" "" ${ARGN})
  if(ARGS_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "register_controller_template():called with unused "
      "arguments: ${ARGS_UNPARSED_ARGUMENTS}")
  endif()

  if("${ARGS_CONTROLLER_TEMPLATE_FILE}" STREQUAL "")
    message(FATAL_ERROR "register_controller_template(): CONTROLLER_TEMPLATE_FILE is required for controller registration of ${controller_template_name}")
  endif()

  # Set default resource index to "gazebo_test_controller_templates" if not specified otherwise
  set(resource_index "gazebo_test_controller_templates")
  if(NOT "${ARGS_RESOURCE_INDEX}" STREQUAL "")
    set(resource_index "${ARGS_RESOURCE_INDEX}")
    message(STATUS "register_controller_template(): Using RESOURCE_INDEX ${resource_index} for controller registration of ${controller_template_name}")
  endif()

  set(controller_template_file "${ARGS_CONTROLLER_TEMPLATE_FILE}")
  _experiment_register_package_hook()

  # Append the file to the list of controller lists for this resource index
  set(_CONTROLLERS_${resource_index}__CONTROLLER_LISTS 
    "${_CONTROLLERS_${resource_index}__CONTROLLER_LISTS}${controller_template_name};${controller_template_file}\n")
  list(APPEND _CONTROLLERS_PACKAGE_RESOURCE_INDICES ${resource_index})

  message(STATUS "register_controller_template(): Registered controller config ${controller_template_file} for ${controller_template_name} under ${resource_index} and installed it at share/${PROJECT_NAME}/templates/controllers")

  install(
    FILES ${controller_template_file}
    DESTINATION share/${PROJECT_NAME}/templates/controllers)
endmacro()


# # Register multiple controller templates from a YAML configuration file
# This macro reads a YAML file containing a list of controllers and their associated
# template files, then registers each controller template using the
# `register_controller_template` macro.
# # usage: register_controller_templates_from_yaml(
#       YAML_FILE <yaml_file> RESOURCE_INDEX <resource_index> TEMPLATES_BASE_DIR <base_dir>)
# #
#   :param yaml_file: path to the YAML file listing controllers and their templates
#   :type yaml_file: string
#   :param resource_index: the resource index to register the configs under
#   :type resource_index: string
#   :param templates_base_dir: base directory to prepend to template file paths
#   :type templates_base_dir: string

macro(register_controller_templates_from_yaml)
  cmake_parse_arguments(ARGS "" "YAML_FILE;RESOURCE_INDEX;TEMPLATES_BASE_DIR" "" ${ARGN})
  
  if("${ARGS_YAML_FILE}" STREQUAL "")
    message(FATAL_ERROR "register_controller_templates_from_yaml(): YAML_FILE is required")
  endif()

  set(yaml_file "${ARGS_YAML_FILE}")
  # Make CMake re-run configure if the YAML file changes
  set(yaml_path "${CMAKE_CURRENT_SOURCE_DIR}/${yaml_file}")
  if(EXISTS "${yaml_path}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${yaml_path}")
  endif()
  set(resource_index "gazebo_test_controller_templates")
  set(templates_base_dir "templates/")
  
  if(NOT "${ARGS_RESOURCE_INDEX}" STREQUAL "")
    set(resource_index "${ARGS_RESOURCE_INDEX}")
  endif()
  
  if(NOT "${ARGS_TEMPLATES_BASE_DIR}" STREQUAL "")
    set(templates_base_dir "${ARGS_TEMPLATES_BASE_DIR}")
  endif()

  # Read the YAML file
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${yaml_file}" yaml_content)
  
  # Extract controller names from controller_templates list
  string(REGEX MATCHALL "- ([a-zA-Z0-9_]+)" controller_matches "${yaml_content}")
  
  set(registered_count 0)
  foreach(match ${controller_matches})
    string(REGEX REPLACE "- " "" controller_template_name "${match}")
    
    # Look for the controller's template configuration
    string(REGEX MATCH "${controller_template_name}:[\n\r ]*template: *'([^']+)'" template_match "${yaml_content}")
    
    if(template_match)
      set(template_path "${CMAKE_MATCH_1}")
      
      # Ensure template path includes base directory
      if(NOT template_path MATCHES "^${templates_base_dir}")
        set(template_path "${templates_base_dir}${template_path}")
      endif()
      
      if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${template_path}")
        message(STATUS "register_controller_templates_from_yaml(): Registering controller ${controller_template_name} with template ${template_path} under ${resource_index}")
        register_controller_template(${controller_template_name}
          CONTROLLER_TEMPLATE_FILE ${template_path}
          RESOURCE_INDEX ${resource_index}
        )
        math(EXPR registered_count "${registered_count} + 1")
      else()
        message(WARNING "Template file not found for ${controller_template_name}: ${CMAKE_CURRENT_SOURCE_DIR}/${template_path}")
      endif()
    else()
      message(WARNING "No template configuration found for controller: ${controller_template_name}")
    endif()
  endforeach()

  message(STATUS "register_controller_templates_from_yaml(): Successfully registered ${registered_count} controllers from ${yaml_file}")
endmacro()
