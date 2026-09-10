"""Assemble steve.urdf combining MPO-700 base, cabinet, UR5 arm, Robotiq-85 gripper, and cameras."""

import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UR5_ROBOTIQ_URDF = os.path.join(
    REPO_ROOT, "third_party", "shelf_gym_repo", "shelf_gym", "meshes", "urdf", "ur5_robotiq_85.urdf"
)
OUTPUT_URDF = os.path.join(os.path.dirname(__file__), "models", "steve.urdf")

with open(UR5_ROBOTIQ_URDF, "r") as f:
    ur5_content = f.read()

# Prefix base_link to ur5_base_link to avoid conflict with MPO-700 chassis base_link
ur5_content = re.sub(r'<link name="base_link">', '<link name="ur5_base_link">', ur5_content)
ur5_content = re.sub(r'<parent link="base_link"/>', '<parent link="ur5_base_link"/>', ur5_content)

# Remove legacy world link and world_arm_joint completely
ur5_content = re.sub(r'<link name="world"\s*/>', '', ur5_content)
ur5_content = re.sub(r'<joint name="world_arm_joint"[\s\S]*?</joint>', '', ur5_content)

# Fix non-positive-definite finger pad inertia tensors
ur5_content = ur5_content.replace(
    '<inertia ixx="1E-10" ixy="1E-10" ixz="1E-10" iyy="1E-10" iyz="1E-10" izz="1E-10"/>',
    '<inertia ixx="1E-4" ixy="0" ixz="0" iyy="1E-4" iyz="0" izz="1E-4"/>'
)

# Equip dummy_camera_link visual with actual RealSense D435 camera mesh
ur5_content = ur5_content.replace(
    '<link name="dummy_camera_link">\n'
    '    <contact>\n'
    '      <lateral_friction value="0.5"/>\n'
    '      <rolling_friction value="0.0001"/>\n'
    '      <inertia_scaling value="3.0"/>\n'
    '    </contact>\n'
    '    <inertial>\n'
    '      <origin rpy="0 0 0" xyz="0 0 0"/>\n'
    '       <mass value=".1"/>\n'
    '       <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>\n'
    '    </inertial>\n'
    '    <visual>\n'
    '      <origin rpy="0 0 0" xyz="0 0 0"/>\n'
    '      <geometry>\n'
    '         <box size="0.042 0.042 0.023"/>\n'
    '      </geometry>',
    '<link name="dummy_camera_link">\n'
    '    <contact>\n'
    '      <lateral_friction value="0.5"/>\n'
    '      <rolling_friction value="0.0001"/>\n'
    '      <inertia_scaling value="3.0"/>\n'
    '    </contact>\n'
    '    <inertial>\n'
    '      <origin rpy="0 0 0" xyz="0 0 0"/>\n'
    '       <mass value=".1"/>\n'
    '       <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>\n'
    '    </inertial>\n'
    '    <visual>\n'
    '      <origin rpy="0 0 0" xyz="0 0 0"/>\n'
    '      <geometry>\n'
    '         <mesh filename="meshes/d435.dae"/>\n'
    '      </geometry>'
)

# Fix relative mesh paths from ../meshes/... to meshes/...
ur5_content = ur5_content.replace('../meshes/', 'meshes/')

# Extract elements inside <robot ...> ... </robot>
start_idx = ur5_content.find('<robot')
start_idx = ur5_content.find('>', start_idx) + 1
end_idx = ur5_content.rfind('</robot>')
arm_elements = ur5_content[start_idx:end_idx].strip()

steve_base_xml = """
  <!-- Steve MPO-700 Omnidirectional Mobile Base Footprint -->
  <link name="base_footprint">
    <inertial>
      <mass value="0.001"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="base_footprint_joint" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="base_footprint"/>
    <child link="base_link"/>
  </joint>

  <!-- Steve MPO-700 Omnidirectional Mobile Base Chassis -->
  <link name="base_link">
    <inertial>
      <mass value="140.0"/>
      <origin xyz="0 0 0.18" rpy="0 0 0"/>
      <inertia ixx="7.8" ixy="0" ixz="0" iyy="7.8" iyz="0" izz="7.8"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 -1.57079632679"/>
      <geometry>
        <mesh filename="meshes/MPO-700-BODY.dae" scale="0.001 0.001 0.001"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0.18" rpy="0 0 0"/>
      <geometry>
        <box size="0.77 0.55 0.36"/>
      </geometry>
    </collision>
  </link>

  <!-- Front Left Caster -->
  <joint name="caster_front_left_joint" type="fixed">
    <origin xyz="0.24 0.18 0.22" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="caster_front_left_link"/>
  </joint>
  <link name="caster_front_left_link">
    <inertial>
      <mass value="12.7"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.27258" ixy="0" ixz="0" iyy="0.27258" iyz="0" izz="0.27258"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="meshes/MPO-700-HEAD.dae" scale="0.001 0.001 0.001"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="meshes/MPO-700-HEAD.dae" scale="0.001 0.001 0.001"/>
      </geometry>
    </collision>
  </link>

  <!-- Front Right Caster -->
  <joint name="caster_front_right_joint" type="fixed">
    <origin xyz="0.24 -0.18 0.22" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="caster_front_right_link"/>
  </joint>
  <link name="caster_front_right_link">
    <inertial>
      <mass value="12.7"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.27258" ixy="0" ixz="0" iyy="0.27258" iyz="0" izz="0.27258"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 3.14159265"/>
      <geometry>
        <mesh filename="meshes/MPO-700-HEAD.dae" scale="0.001 0.001 0.001"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 3.14159265"/>
      <geometry>
        <mesh filename="meshes/MPO-700-HEAD.dae" scale="0.001 0.001 0.001"/>
      </geometry>
    </collision>
  </link>

  <!-- Back Left Caster -->
  <joint name="caster_back_left_joint" type="fixed">
    <origin xyz="-0.24 0.18 0.22" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="caster_back_left_link"/>
  </joint>
  <link name="caster_back_left_link">
    <inertial>
      <mass value="12.7"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.27258" ixy="0" ixz="0" iyy="0.27258" iyz="0" izz="0.27258"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="meshes/MPO-700-HEAD.dae" scale="0.001 0.001 0.001"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="meshes/MPO-700-HEAD.dae" scale="0.001 0.001 0.001"/>
      </geometry>
    </collision>
  </link>

  <!-- Back Right Caster -->
  <joint name="caster_back_right_joint" type="fixed">
    <origin xyz="-0.24 -0.18 0.22" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="caster_back_right_link"/>
  </joint>
  <link name="caster_back_right_link">
    <inertial>
      <mass value="12.7"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.27258" ixy="0" ixz="0" iyy="0.27258" iyz="0" izz="0.27258"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 3.14159265"/>
      <geometry>
        <mesh filename="meshes/MPO-700-HEAD.dae" scale="0.001 0.001 0.001"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 3.14159265"/>
      <geometry>
        <mesh filename="meshes/MPO-700-HEAD.dae" scale="0.001 0.001 0.001"/>
      </geometry>
    </collision>
  </link>

  <!-- Front Left Wheel -->
  <joint name="wheel_front_left_joint" type="fixed">
    <origin xyz="0.0 0.045 -0.12" rpy="0 0 0"/>
    <parent link="caster_front_left_link"/>
    <child link="wheel_front_left_link"/>
  </joint>
  <link name="wheel_front_left_link">
    <inertial>
      <mass value="1.2"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.11042" ixy="0" ixz="0" iyy="0.11042" iyz="0" izz="0.11042"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 3.14159265"/>
      <geometry>
        <mesh filename="meshes/MPO-700-WHEEL.dae" scale="0.001 0.001 0.001"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.09"/>
      </geometry>
    </collision>
  </link>

  <!-- Front Right Wheel -->
  <joint name="wheel_front_right_joint" type="fixed">
    <origin xyz="0.0 -0.045 -0.12" rpy="0 0 3.14159265"/>
    <parent link="caster_front_right_link"/>
    <child link="wheel_front_right_link"/>
  </joint>
  <link name="wheel_front_right_link">
    <inertial>
      <mass value="1.2"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.11042" ixy="0" ixz="0" iyy="0.11042" iyz="0" izz="0.11042"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 3.14159265"/>
      <geometry>
        <mesh filename="meshes/MPO-700-WHEEL.dae" scale="0.001 0.001 0.001"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.09"/>
      </geometry>
    </collision>
  </link>

  <!-- Back Left Wheel -->
  <joint name="wheel_back_left_joint" type="fixed">
    <origin xyz="0.0 0.045 -0.12" rpy="0 0 0"/>
    <parent link="caster_back_left_link"/>
    <child link="wheel_back_left_link"/>
  </joint>
  <link name="wheel_back_left_link">
    <inertial>
      <mass value="1.2"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.11042" ixy="0" ixz="0" iyy="0.11042" iyz="0" izz="0.11042"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 3.14159265"/>
      <geometry>
        <mesh filename="meshes/MPO-700-WHEEL.dae" scale="0.001 0.001 0.001"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.09"/>
      </geometry>
    </collision>
  </link>

  <!-- Back Right Wheel -->
  <joint name="wheel_back_right_joint" type="fixed">
    <origin xyz="0.0 -0.045 -0.12" rpy="0 0 3.14159265"/>
    <parent link="caster_back_right_link"/>
    <child link="wheel_back_right_link"/>
  </joint>
  <link name="wheel_back_right_link">
    <inertial>
      <mass value="1.2"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.11042" ixy="0" ixz="0" iyy="0.11042" iyz="0" izz="0.11042"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 3.14159265"/>
      <geometry>
        <mesh filename="meshes/MPO-700-WHEEL.dae" scale="0.001 0.001 0.001"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.09"/>
      </geometry>
    </collision>
  </link>

  <!-- Front Laser Scanner SICK S300 -->
  <link name="lidar_1_link">
    <inertial>
      <mass value="1.2"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.11042" ixy="0" ixz="0" iyy="0.11042" iyz="0" izz="0.11042"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.12" rpy="-1.57079632679 0 3.14159265"/>
      <geometry>
        <mesh filename="meshes/SICK-S300.dae" scale="0.001 0.001 0.001"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 -0.12" rpy="-1.57079632679 0 3.14159265"/>
      <geometry>
        <mesh filename="meshes/SICK-S300.dae" scale="0.001 0.001 0.001"/>
      </geometry>
    </collision>
  </link>
  <joint name="lidar_1_joint" type="fixed">
    <origin xyz="0.338 0.288 0.223" rpy="3.14159265 0 0.79"/>
    <parent link="base_link"/>
    <child link="lidar_1_link"/>
  </joint>

  <!-- Rear Laser Scanner SICK S300 -->
  <link name="lidar_2_link">
    <inertial>
      <mass value="1.2"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.11042" ixy="0" ixz="0" iyy="0.11042" iyz="0" izz="0.11042"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.12" rpy="-1.57079632679 0 3.14159265"/>
      <geometry>
        <mesh filename="meshes/SICK-S300.dae" scale="0.001 0.001 0.001"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 -0.12" rpy="-1.57079632679 0 3.14159265"/>
      <geometry>
        <mesh filename="meshes/SICK-S300.dae" scale="0.001 0.001 0.001"/>
      </geometry>
    </collision>
  </link>
  <joint name="lidar_2_joint" type="fixed">
    <origin xyz="-0.338 -0.288 0.223" rpy="3.14159265 0 3.93"/>
    <parent link="base_link"/>
    <child link="lidar_2_link"/>
  </joint>

  <!-- Steve Equipment Cabinet -->
  <link name="cabinet_link">
    <inertial>
      <mass value="15.0"/>
      <origin xyz="0 0 0.208" rpy="0 0 0"/>
      <inertia ixx="0.5" ixy="0" ixz="0" iyy="0.5" iyz="0" izz="0.5"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="meshes/cabin.dae" scale="0.001 0.001 0.001"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0.208" rpy="0 0 0"/>
      <geometry>
        <box size="0.45 0.45 0.416"/>
      </geometry>
    </collision>
  </link>
  <joint name="cabinet_joint" type="fixed">
    <origin xyz="0.025 0.0 0.35" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="cabinet_link"/>
  </joint>

  <!-- Chassis Intel RealSense D435 Camera -->
  <link name="chassis_camera_link">
    <inertial>
      <mass value="0.072"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
    <visual>
      <geometry>
        <mesh filename="meshes/d435.dae"/>
      </geometry>
    </visual>
  </link>
  <joint name="chassis_camera_joint" type="fixed">
    <origin xyz="0.292 0.0 0.716" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="chassis_camera_link"/>
  </joint>

  <!-- Mount UR5 Arm onto Cabinet Top -->
  <joint name="arm_base_joint" type="fixed">
    <origin xyz="0.133 0.0 0.416" rpy="0 0 -1.57079632679"/>
    <parent link="cabinet_link"/>
    <child link="ur5_base_link"/>
  </joint>
"""

full_urdf = f"""<?xml version="1.0" ?>
<robot name="steve" xmlns:xacro="http://ros.org/wiki/xacro">
{steve_base_xml}
{arm_elements}
</robot>
"""

with open(OUTPUT_URDF, "w") as f:
    f.write(full_urdf)

print(f"Generated clean {OUTPUT_URDF} ({os.path.getsize(OUTPUT_URDF):,} bytes)")
