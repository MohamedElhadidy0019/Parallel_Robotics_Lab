# Steve ROS2 Docker Container Setup & Fixes

Quick developer reference to build and run the Steve robot stack inside the `steve_ros2_ws_student` Docker container on ROS 2 Humble without errors.

---

## 1. Single APT Dependency Installation

Run inside the container:

```bash
sudo apt update && sudo apt install -y \
  ros-humble-ur-robot-driver \
  ros-humble-ur-controllers \
  ros-humble-robotiq-description \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-gazebo-ros2-control \
  libspnav-dev \
  libbluetooth-dev \
  libcwiid-dev
```

### Why:
* `ros-humble-ur-robot-driver` & `ur-controllers`: Required for Universal Robots arm trajectory control (`scaled_joint_trajectory_controller`).
* `ros-humble-robotiq-description`: URDF and meshes for Robotiq 2F-85 gripper.
* `ros-humble-gazebo-ros-pkgs` & `ros-humble-gazebo-ros2-control`: Gazebo Classic ROS 2 plugins (`gazebo_ros`, entity spawning, ros2_control simulation interfaces).
* `libspnav-dev`, `libbluetooth-dev`, `libcwiid-dev`: Headers required to build joystick drivers (`spacenav`, `wiimote`) in `src/joystick_drivers`.

---

## 2. Clone Simulation Environment

Clone Neobotix simulation bringup into the workspace:

```bash
cd /home/ws/src
git clone -b humble https://github.com/neobotix/neo_simulation2.git
```

### Why:
* Provides the Gazebo simulation worlds (`neo_workshop`, `neo_track1`) and robot spawner launch scripts for MPO-700.

---

## 3. Patch Robotiq Gripper Xacro (with Wrist RealSense D435)

In `/home/ws/src/neo_simulation2/components/arm/robotiq_gripper.urdf.xacro`, replace the file contents with:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="robotiq_gripper">
    <!-- UR to Robotiq adapter -->
    <xacro:include filename="$(find robotiq_description)/urdf/ur_to_robotiq_adapter.urdf.xacro"/>
    <xacro:ur_to_robotiq 
        prefix="" 
        connected_to="$(arg tf_prefix)tool0" 
        rotation="${pi/2}"
    />

    <!-- Robotiq 2F-85 Gripper -->
    <xacro:include filename="$(find robotiq_description)/urdf/robotiq_2f_85_macro.urdf.xacro"/>
    <xacro:robotiq_gripper name="robotiq_gripper" prefix="" parent="gripper_mount_link" include_ros2_control="false">
        <origin xyz="0 0 0" rpy="0 0 -${pi / 2}"/>
    </xacro:robotiq_gripper>

    <!-- Wrist-mounted RealSense D435 Depth Camera -->
    <xacro:include filename="$(find realsense2_description)/urdf/_d435.urdf.xacro"/>
    <xacro:sensor_d435 parent="$(arg tf_prefix)tool0" name="camera" use_nominal_extrinsics="true">
        <origin xyz="0.02 0.0 -0.06" rpy="1.5707963 3.14159265 1.5707963"/>
    </xacro:sensor_d435>

    <gazebo reference="camera_link">
        <sensor name="camera_depth_sensor" type="depth">
            <always_on>true</always_on>
            <update_rate>15.0</update_rate>
            <camera>
                <horizontal_fov>1.01229</horizontal_fov>
                <image>
                    <width>640</width>
                    <height>480</height>
                    <format>R8G8B8</format>
                </image>
                <clip>
                    <near>0.1</near>
                    <far>2.5</far>
                </clip>
            </camera>
            <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
                <ros>
                    <remapping>camera/image_raw:=camera/color/image_raw</remapping>
                    <remapping>camera/camera_info:=camera/color/camera_info</remapping>
                    <remapping>camera/depth/image_raw:=camera/depth/image_rect_raw</remapping>
                    <remapping>camera/depth/camera_info:=camera/depth/camera_info</remapping>
                    <remapping>camera/points:=camera/depth/color/points</remapping>
                </ros>
                <camera_name>camera</camera_name>
                <frame_name>camera_color_optical_frame</frame_name>
                <min_depth>0.1</min_depth>
                <max_depth>2.5</max_depth>
            </plugin>
        </sensor>
    </gazebo>
</robot>
```

### Why:
* Fixes `[ERROR] [launch]: Caught exception in launch: Invalid parameter "parent"` when launching the UR arm.
* Upstream ROS 2 Humble `robotiq_description` uses `connected_to` (not `parent`) for `ur_to_robotiq`, outputs the child frame `gripper_mount_link`, and provides the `robotiq_2f_85_macro` macro.
* Surgically attaches the RealSense D435 camera and Gazebo depth plugin to the UR5 wrist (`ur5tool0`), enabling live PointCloud2 on `/camera/depth/color/points` that rigidly tracks arm movement.

---

## 4. Build Workspace

Compile all 20 packages (ignoring unused Nav2 2D base navigation plugins):

```bash
source /opt/ros/humble/setup.bash
cd /home/ws
colcon build --symlink-install --packages-ignore neo_local_planner2 neo_localization2
```

### Why:
* `neo_local_planner2` and `neo_localization2` require uninstalled Nav2 internal development headers and are not used for arm/camera inspection.

---

## 5. Verification & Launch Commands

Source the workspace:

```bash
source /home/ws/install/setup.bash
```

### Launch Gazebo Simulation (MPO-700 + UR5)
```bash
ros2 launch neo_simulation2 simulation.launch.py my_robot:=mpo_700 world:=neo_workshop arm_type:=ur5
```

### Launch Real Robot Bringup
```bash
ros2 launch neo_mpo_700-2 bringup.launch.py arm_type:=ur5 gripper_type:=2f_85 d435_enable:=True robot_ip:=<UR5_IP>
```

### Teleoperate Base via Keyboard
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```
