// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef COMMON_UTILS__PARAMETERS_HANDLER_HPP_
#define COMMON_UTILS__PARAMETERS_HANDLER_HPP_

#include <functional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "nav2_util/node_utils.hpp"
#include "rclcpp/parameter_value.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"

namespace common_utils
{

/**
 * @class common_utils::ParametersHandler
 * @brief Handles getting parameters and dynamic parmaeter changes
 */
class ParametersHandler
{
public:
  /**
    * @brief Constructor for common_utils::ParametersHandler
    */
  ParametersHandler() = default;

  /**
    * @brief Constructor for common_utils::ParametersHandler
    * @param parent Weak ptr to node
    */
  explicit ParametersHandler(const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent);

  /**
    * @brief Get an object to retreive parameters
    * @param ns Namespace to get parameters within
    * @return Parameter getter object
    */
  inline auto getParamGetter(const std::string & ns);

protected:
  /**
    * @brief Gets parameter
    * @param setting Return Parameter type
    * @param name Parameter name
    * @param default_value Default parameter value
    */
  template<typename SettingT, typename ParamT>
  void getParam(SettingT & setting, const std::string & name, ParamT default_value);

  /**
    * @brief Set a parameter
    * @param setting Return Parameter type
    * @param name Parameter name
    * @param node Node to set parameter via
    */
  template<typename ParamT, typename SettingT, typename NodeT>
  void setParam(SettingT & setting, const std::string & name, NodeT node) const;

  rclcpp::Logger logger_{rclcpp::get_logger("common_utils::ParametersHandler")};
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr on_set_param_handler_;
  rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
  std::string node_name_;

  bool verbose_{false};
};

inline auto ParametersHandler::getParamGetter(const std::string & ns)
{
  return [this, ns](auto & setting, const std::string & name, auto default_value) {
           getParam(setting, ns.empty() ? name : ns + "." + name, std::move(default_value));
         };
}

template<typename SettingT, typename ParamT>
void ParametersHandler::getParam(SettingT & setting, const std::string & name, ParamT default_value)
{
  auto node = node_.lock();

  nav2_util::declare_parameter_if_not_declared(node, name, rclcpp::ParameterValue(default_value));

  setParam<ParamT>(setting, name, node);
}

template<typename ParamT, typename SettingT, typename NodeT>
void ParametersHandler::setParam(SettingT & setting, const std::string & name, NodeT node) const
{
  ParamT param_in{};
  node->get_parameter(name, param_in);
  setting = static_cast<SettingT>(param_in);
}

}  // namespace common_utils

#endif  // COMMON_UTILS__PARAMETER_HANDLER_HPP_
