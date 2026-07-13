# register experiment resources
list(REMOVE_DUPLICATES _EXPERIMENTS_PACKAGE_RESOURCE_INDICES)
foreach(resource_index ${_EXPERIMENTS_PACKAGE_RESOURCE_INDICES})
  ament_index_register_resource(
    ${resource_index} CONTENT "${_EXPERIMENTS_${resource_index}__EXP_LISTS}")
endforeach()

# register controller resources
if(DEFINED _CONTROLLERS_PACKAGE_RESOURCE_INDICES)
  list(REMOVE_DUPLICATES _CONTROLLERS_PACKAGE_RESOURCE_INDICES)
  foreach(resource_index ${_CONTROLLERS_PACKAGE_RESOURCE_INDICES})
    ament_index_register_resource(
      ${resource_index} CONTENT "${_CONTROLLERS_${resource_index}__CONTROLLER_LISTS}")
  endforeach()
endif()

# register planner resources
if(DEFINED _PLANNERS_PACKAGE_RESOURCE_INDICES)
  list(REMOVE_DUPLICATES _PLANNERS_PACKAGE_RESOURCE_INDICES)
  foreach(resource_index ${_PLANNERS_PACKAGE_RESOURCE_INDICES})
    ament_index_register_resource(
      ${resource_index} CONTENT "${_PLANNERS_${resource_index}__PLANNER_LISTS}")
  endforeach()
endif()

# register navigator resources
if(DEFINED _NAVIGATORS_PACKAGE_RESOURCE_INDICES)
  list(REMOVE_DUPLICATES _NAVIGATORS_PACKAGE_RESOURCE_INDICES)
  foreach(resource_index ${_NAVIGATORS_PACKAGE_RESOURCE_INDICES})
    ament_index_register_resource(
      ${resource_index} CONTENT "${_NAVIGATORS_${resource_index}__NAVIGATOR_LISTS}")
  endforeach()
endif()

# register costmap_layer resources
if(DEFINED _COSTMAP_LAYERS_PACKAGE_RESOURCE_INDICES)
  list(REMOVE_DUPLICATES _COSTMAP_LAYERS_PACKAGE_RESOURCE_INDICES)
  foreach(resource_index ${_COSTMAP_LAYERS_PACKAGE_RESOURCE_INDICES})
    ament_index_register_resource(
      ${resource_index} CONTENT "${_COSTMAP_LAYERS_${resource_index}__COSTMAP_LAYER_LISTS}")
  endforeach()
endif()