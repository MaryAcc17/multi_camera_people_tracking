// Copyright 2024 Andrea Ostuni
// SPDX-License-Identifier: MIT

#ifndef RVIZ_PEOPLE_DISPLAY__PEOPLE_DISPLAY_HPP_
#define RVIZ_PEOPLE_DISPLAY__PEOPLE_DISPLAY_HPP_

#include <memory>
#include <string>
#include <vector>

#include <rviz_common/message_filter_display.hpp>
#include <rviz_common/properties/bool_property.hpp>
#include <rviz_common/properties/color_property.hpp>
#include <rviz_common/properties/float_property.hpp>

#include <rviz_rendering/objects/arrow.hpp>
#include <rviz_rendering/objects/movable_text.hpp>

#include <people_msgs/msg/people.hpp>

namespace rviz_people_display
{

/// Holds Ogre objects for a single person visualisation.
struct PersonVisual
{
  std::unique_ptr<rviz_rendering::Arrow> heading_arrow;
  std::unique_ptr<rviz_rendering::Arrow> velocity_arrow;
  Ogre::SceneNode * text_node{nullptr};
  rviz_rendering::MovableText * text{nullptr};
};

class PeopleDisplay
  : public rviz_common::MessageFilterDisplay<people_msgs::msg::People>
{
  Q_OBJECT

public:
  PeopleDisplay();
  ~PeopleDisplay() override;

protected:
  void onInitialize() override;
  void reset() override;
  void processMessage(
    const people_msgs::msg::People::ConstSharedPtr msg) override;

private Q_SLOTS:
  void updateStyle();

private:
  void clearVisuals();

  // --- Properties ---
  // Heading arrow
  rviz_common::properties::ColorProperty * heading_color_property_;
  rviz_common::properties::FloatProperty * heading_alpha_property_;
  rviz_common::properties::FloatProperty * arrow_length_property_;
  rviz_common::properties::FloatProperty * arrow_shaft_diameter_property_;
  rviz_common::properties::FloatProperty * arrow_head_diameter_property_;
  rviz_common::properties::FloatProperty * arrow_head_length_property_;
  rviz_common::properties::FloatProperty * arrow_z_offset_property_;

  // Velocity arrow
  rviz_common::properties::BoolProperty * show_velocity_property_;
  rviz_common::properties::ColorProperty * velocity_color_property_;
  rviz_common::properties::FloatProperty * velocity_alpha_property_;
  rviz_common::properties::FloatProperty * velocity_scale_property_;

  // Text labels
  rviz_common::properties::BoolProperty * show_text_property_;
  rviz_common::properties::FloatProperty * text_z_offset_property_;
  rviz_common::properties::FloatProperty * text_size_property_;
  rviz_common::properties::ColorProperty * text_color_property_;

  // --- Visuals ---
  std::vector<PersonVisual> visuals_;
};

}  // namespace rviz_people_display

#endif  // RVIZ_PEOPLE_DISPLAY__PEOPLE_DISPLAY_HPP_
