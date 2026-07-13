// Copyright 2024 Andrea Ostuni
// SPDX-License-Identifier: MIT

#include "rviz_people_display/people_display.hpp"

#include <cmath>
#include <string>

#include <OgreSceneManager.h>
#include <OgreSceneNode.h>
#include <OgreQuaternion.h>
#include <OgreVector3.h>

#include <rviz_common/logging.hpp>
#include <rviz_common/properties/parse_color.hpp>
#include <rviz_common/properties/status_property.hpp>

namespace rviz_people_display
{

// Arrow default orientation is -Z, so we need a base rotation to align with +X
static Ogre::Quaternion ARROW_BASE_ROTATION(
  Ogre::Degree(-90.0f), Ogre::Vector3::UNIT_Y);

static Ogre::Quaternion yawToQuaternion(float yaw)
{
  return Ogre::Quaternion(Ogre::Radian(yaw), Ogre::Vector3::UNIT_Z);
}

// ============================================================
// Construction / Destruction
// ============================================================

PeopleDisplay::PeopleDisplay()
{
  // --- Heading arrow properties ---
  heading_color_property_ = new rviz_common::properties::ColorProperty(
    "Heading Color", QColor(50, 180, 50),
    "Color of the heading arrow.", this, SLOT(updateStyle()));

  heading_alpha_property_ = new rviz_common::properties::FloatProperty(
    "Heading Alpha", 1.0f,
    "Transparency of the heading arrow (0 = invisible, 1 = opaque).",
    this, SLOT(updateStyle()));
  heading_alpha_property_->setMin(0.0f);
  heading_alpha_property_->setMax(1.0f);

  arrow_length_property_ = new rviz_common::properties::FloatProperty(
    "Arrow Length", 0.6f,
    "Length of the heading arrow shaft.", this, SLOT(updateStyle()));
  arrow_length_property_->setMin(0.01f);

  arrow_shaft_diameter_property_ = new rviz_common::properties::FloatProperty(
    "Arrow Shaft Diameter", 0.08f,
    "Diameter of the heading arrow shaft.", this, SLOT(updateStyle()));
  arrow_shaft_diameter_property_->setMin(0.001f);

  arrow_head_diameter_property_ = new rviz_common::properties::FloatProperty(
    "Arrow Head Diameter", 0.15f,
    "Diameter of the heading arrow head.", this, SLOT(updateStyle()));
  arrow_head_diameter_property_->setMin(0.001f);

  arrow_head_length_property_ = new rviz_common::properties::FloatProperty(
    "Arrow Head Length", 0.12f,
    "Length of the heading arrow head.", this, SLOT(updateStyle()));
  arrow_head_length_property_->setMin(0.001f);

  arrow_z_offset_property_ = new rviz_common::properties::FloatProperty(
    "Z Offset", 0.1f,
    "Height of arrows above the ground plane.", this, SLOT(updateStyle()));

  // --- Velocity arrow properties ---
  show_velocity_property_ = new rviz_common::properties::BoolProperty(
    "Show Velocity", true,
    "Draw a velocity arrow for each person.", this, SLOT(updateStyle()));

  velocity_color_property_ = new rviz_common::properties::ColorProperty(
    "Velocity Color", QColor(230, 50, 50),
    "Color of the velocity arrow.", show_velocity_property_, SLOT(updateStyle()), this);

  velocity_alpha_property_ = new rviz_common::properties::FloatProperty(
    "Velocity Alpha", 0.8f,
    "Transparency of the velocity arrow.",
    show_velocity_property_, SLOT(updateStyle()), this);
  velocity_alpha_property_->setMin(0.0f);
  velocity_alpha_property_->setMax(1.0f);

  velocity_scale_property_ = new rviz_common::properties::FloatProperty(
    "Velocity Scale", 1.0f,
    "Scale factor applied to velocity magnitude for arrow length.",
    show_velocity_property_, SLOT(updateStyle()), this);
  velocity_scale_property_->setMin(0.01f);

  // --- Text label properties ---
  show_text_property_ = new rviz_common::properties::BoolProperty(
    "Show Names", true,
    "Show person name labels.", this, SLOT(updateStyle()));

  text_z_offset_property_ = new rviz_common::properties::FloatProperty(
    "Text Z Offset", 0.8f,
    "Height of the text label above ground.",
    show_text_property_, SLOT(updateStyle()), this);

  text_size_property_ = new rviz_common::properties::FloatProperty(
    "Text Size", 0.3f,
    "Character height of the name label.",
    show_text_property_, SLOT(updateStyle()), this);
  text_size_property_->setMin(0.01f);

  text_color_property_ = new rviz_common::properties::ColorProperty(
    "Text Color", QColor(255, 255, 255),
    "Color of the name label.",
    show_text_property_, SLOT(updateStyle()), this);
}

PeopleDisplay::~PeopleDisplay()
{
  clearVisuals();
}

// ============================================================
// Lifecycle
// ============================================================

void PeopleDisplay::onInitialize()
{
  MFDClass::onInitialize();
}

void PeopleDisplay::reset()
{
  MFDClass::reset();
  clearVisuals();
}

// ============================================================
// Helpers
// ============================================================

void PeopleDisplay::clearVisuals()
{
  for (auto & v : visuals_) {
    if (v.text_node) {
      v.text_node->detachAllObjects();
      scene_manager_->destroySceneNode(v.text_node);
    }
    if (v.text) {
      delete v.text;
    }
  }
  visuals_.clear();
}

void PeopleDisplay::updateStyle()
{
  // Force re-processing of the last message by simply invalidating the visuals.
  // The next processMessage call will pick up new property values.
  // (Properties are read directly in processMessage.)
}

// ============================================================
// Main processing
// ============================================================

void PeopleDisplay::processMessage(
  const people_msgs::msg::People::ConstSharedPtr msg)
{
  // --- Get the TF transform for the message frame ---
  Ogre::Vector3 frame_position;
  Ogre::Quaternion frame_orientation;
  if (!context_->getFrameManager()->getTransform(
      msg->header, frame_position, frame_orientation))
  {
    setStatus(
      rviz_common::properties::StatusProperty::Error, "Transform",
      QString("Error transforming from frame '%1' to frame '%2'")
      .arg(msg->header.frame_id.c_str())
      .arg(qPrintable(fixed_frame_)));
    return;
  }
  setStatus(rviz_common::properties::StatusProperty::Ok, "Transform", "OK");

  // --- Read current property values ---
  const Ogre::ColourValue heading_color =
    rviz_common::properties::qtToOgre(heading_color_property_->getColor());
  const float heading_alpha = heading_alpha_property_->getFloat();
  const float shaft_length = arrow_length_property_->getFloat();
  const float shaft_diam = arrow_shaft_diameter_property_->getFloat();
  const float head_diam = arrow_head_diameter_property_->getFloat();
  const float head_length = arrow_head_length_property_->getFloat();
  const float z_offset = arrow_z_offset_property_->getFloat();
  const bool show_velocity = show_velocity_property_->getBool();
  const float vel_scale = velocity_scale_property_->getFloat();
  const Ogre::ColourValue vel_color =
    rviz_common::properties::qtToOgre(velocity_color_property_->getColor());
  const float vel_alpha = velocity_alpha_property_->getFloat();
  const bool show_text = show_text_property_->getBool();
  const float text_z = text_z_offset_property_->getFloat();
  const float text_size = text_size_property_->getFloat();
  const Ogre::ColourValue text_color =
    rviz_common::properties::qtToOgre(text_color_property_->getColor());

  // --- Rebuild visuals ---
  clearVisuals();
  visuals_.resize(msg->people.size());

  scene_node_->setPosition(frame_position);
  scene_node_->setOrientation(frame_orientation);

  for (size_t i = 0; i < msg->people.size(); ++i) {
    const auto & person = msg->people[i];
    auto & vis = visuals_[i];

    const float px = static_cast<float>(person.position.x);
    const float py = static_cast<float>(person.position.y);
    const float yaw = static_cast<float>(person.position.z);

    // ---- Heading arrow ----
    vis.heading_arrow = std::make_unique<rviz_rendering::Arrow>(
      scene_manager_, scene_node_,
      shaft_length, shaft_diam, head_length, head_diam);

    Ogre::Vector3 pos(px, py, z_offset);
    vis.heading_arrow->setPosition(pos);
    vis.heading_arrow->setOrientation(yawToQuaternion(yaw) * ARROW_BASE_ROTATION);
    vis.heading_arrow->setColor(
      heading_color.r, heading_color.g, heading_color.b, heading_alpha);

    // ---- Velocity arrow ----
    if (show_velocity) {
      const float vx = static_cast<float>(person.velocity.x);
      const float vy = static_cast<float>(person.velocity.y);
      const float speed = std::hypot(vx, vy);

      if (speed > 0.01f) {
        const float vel_length = speed * vel_scale;
        vis.velocity_arrow = std::make_unique<rviz_rendering::Arrow>(
          scene_manager_, scene_node_,
          vel_length * 0.8f, shaft_diam * 0.7f,
          vel_length * 0.2f, head_diam * 0.7f);

        Ogre::Vector3 vel_pos(px, py, z_offset + 0.05f);
        vis.velocity_arrow->setPosition(vel_pos);
        const float vel_yaw = std::atan2(vy, vx);
        vis.velocity_arrow->setOrientation(
          yawToQuaternion(vel_yaw) * ARROW_BASE_ROTATION);
        vis.velocity_arrow->setColor(
          vel_color.r, vel_color.g, vel_color.b, vel_alpha);
      }
    }

    // ---- Text label ----
    if (show_text && !person.name.empty()) {
      vis.text_node = scene_node_->createChildSceneNode();
      vis.text_node->setPosition(Ogre::Vector3(px, py, text_z));

      vis.text = new rviz_rendering::MovableText(person.name);
      vis.text->setCharacterHeight(text_size);
      vis.text->setColor(Ogre::ColourValue(
        text_color.r, text_color.g, text_color.b, 1.0f));
      vis.text->setTextAlignment(
        rviz_rendering::MovableText::H_CENTER,
        rviz_rendering::MovableText::V_CENTER);

      vis.text_node->attachObject(vis.text);
    }
  }

  // --- Status ---
  setStatus(
    rviz_common::properties::StatusProperty::Ok, "People",
    QString("%1 people displayed").arg(msg->people.size()));
}

}  // namespace rviz_people_display

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(rviz_people_display::PeopleDisplay, rviz_common::Display)
