#!/usr/bin/env python3
"""
Dynamic URDF generator for inspection tables of various shapes and dimensions.
Supports rectangle, circular, and square tables.
"""

def generate_table_urdf(
    shape: str = "rectangle",
    height: float = 0.40,
    thickness: float = 0.04,
    depth: float = 0.50,
    width: float = 0.80,
    size: float = 0.60,
    radius: float = 0.30,
    material: str = "Gazebo/WoodFloor"
) -> str:
    """
    Generate a URDF XML string for a static inspection table.
    - height: total height from ground to top surface.
    - thickness: tabletop slab thickness.
    """
    height = float(height)
    thickness = float(thickness)
    pedestal_h = max(0.01, height - thickness)
    top_z = height - (thickness / 2.0)
    ped_z = pedestal_h / 2.0

    shape_lower = str(shape).strip().lower()

    if shape_lower == "circular":
        r = float(radius)
        pillar_r = max(0.04, r * 0.35)
        base_r = max(0.06, r * 0.65)
        base_h = min(0.02, pedestal_h * 0.2)
        mid_h = pedestal_h - base_h
        mid_z = base_h + (mid_h / 2.0)

        urdf = f"""<?xml version="1.0"?>
<robot name="inspection_table">
  <link name="table_link">
    <inertial>
      <mass value="40.0"/>
      <origin xyz="0 0 {ped_z}" rpy="0 0 0"/>
      <inertia ixx="2.0" ixy="0" ixz="0" iyy="2.0" iyz="0" izz="3.0"/>
    </inertial>
    <!-- Tabletop -->
    <visual>
      <origin xyz="0 0 {top_z}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{r}" length="{thickness}"/>
      </geometry>
      <material name="wood_color">
        <color rgba="0.8 0.6 0.4 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 {top_z}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{r}" length="{thickness}"/>
      </geometry>
    </collision>
    <!-- Central Pillar -->
    <visual>
      <origin xyz="0 0 {mid_z}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{pillar_r}" length="{mid_h}"/>
      </geometry>
      <material name="wood_color"/>
    </visual>
    <collision>
      <origin xyz="0 0 {mid_z}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{pillar_r}" length="{mid_h}"/>
      </geometry>
    </collision>
    <!-- Base Plate -->
    <visual>
      <origin xyz="0 0 {base_h / 2.0}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{base_r}" length="{base_h}"/>
      </geometry>
      <material name="wood_color"/>
    </visual>
    <collision>
      <origin xyz="0 0 {base_h / 2.0}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{base_r}" length="{base_h}"/>
      </geometry>
    </collision>
  </link>
  <gazebo reference="table_link">
    <material>{material}</material>
  </gazebo>
  <gazebo>
    <static>true</static>
  </gazebo>
</robot>
"""
    else:
        # Rectangle or Square
        if shape_lower == "square":
            d = float(size)
            w = float(size)
        else:  # rectangle
            d = float(depth)
            w = float(width)

        ped_d = max(0.05, d * 0.75)
        ped_w = max(0.05, w * 0.75)

        urdf = f"""<?xml version="1.0"?>
<robot name="inspection_table">
  <link name="table_link">
    <inertial>
      <mass value="40.0"/>
      <origin xyz="0 0 {ped_z}" rpy="0 0 0"/>
      <inertia ixx="2.0" ixy="0" ixz="0" iyy="2.0" iyz="0" izz="3.0"/>
    </inertial>
    <!-- Tabletop -->
    <visual>
      <origin xyz="0 0 {top_z}" rpy="0 0 0"/>
      <geometry>
        <box size="{d} {w} {thickness}"/>
      </geometry>
      <material name="wood_color">
        <color rgba="0.8 0.6 0.4 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 {top_z}" rpy="0 0 0"/>
      <geometry>
        <box size="{d} {w} {thickness}"/>
      </geometry>
    </collision>
    <!-- Pedestal Base -->
    <visual>
      <origin xyz="0 0 {ped_z}" rpy="0 0 0"/>
      <geometry>
        <box size="{ped_d} {ped_w} {pedestal_h}"/>
      </geometry>
      <material name="wood_color"/>
    </visual>
    <collision>
      <origin xyz="0 0 {ped_z}" rpy="0 0 0"/>
      <geometry>
        <box size="{ped_d} {ped_w} {pedestal_h}"/>
      </geometry>
    </collision>
  </link>
  <gazebo reference="table_link">
    <material>{material}</material>
  </gazebo>
  <gazebo>
    <static>true</static>
  </gazebo>
</robot>
"""
    return urdf
