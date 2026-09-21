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

    <!-- Define ee_link matching steve.urdf kinematics -->
    <link name="$(arg tf_prefix)ee_link"/>
    <joint name="$(arg tf_prefix)ee_fixed_joint" type="fixed">
        <parent link="$(arg tf_prefix)wrist_3_link"/>
        <child link="$(arg tf_prefix)ee_link"/>
        <origin xyz="0.0 0.0823 0.0" rpy="0.0 0.0 1.57079632679"/>
    </joint>

    <!-- Wrist-mounted RealSense D435 Depth Camera attached to ee_link -->
    <xacro:include filename="$(find realsense2_description)/urdf/_d435.urdf.xacro"/>
    <xacro:sensor_d435 parent="$(arg tf_prefix)ee_link" name="camera" use_nominal_extrinsics="true">
        <origin xyz="0.02 0.0 -0.06" rpy="0 0 0"/>
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

### 3.1 Set Default Arm Type & Custom Spawn Coordinates in Simulation Bringup

In `/home/ws/src/neo_simulation2/launch/simulation.launch.py`:

1. Set `arm_type` default to `'ur5'` (spawns UR5 arm, cabinet, gripper, and wrist camera automatically):
```python
declare_arm_type_cmd = DeclareLaunchArgument(
    'arm_type', default_value='ur5',
    description='Available arms: "ur5", "ur10", "ur5e", "ur10e"'
)
```

2. Add optional initial robot spawn coordinate arguments (`spawn_x`, `spawn_y`, `spawn_z`, `spawn_yaw`):
```python
# In generate_launch_description:
declare_spawn_x_cmd = DeclareLaunchArgument('spawn_x', default_value='0.0', description='Robot spawn X')
declare_spawn_y_cmd = DeclareLaunchArgument('spawn_y', default_value='0.0', description='Robot spawn Y')
declare_spawn_z_cmd = DeclareLaunchArgument('spawn_z', default_value='0.0', description='Robot spawn Z')
declare_spawn_yaw_cmd = DeclareLaunchArgument('spawn_yaw', default_value='0.0', description='Robot spawn Yaw (rad)')

# Pass to spawn_entity node:
arguments=['-entity', my_neo_robot, '-topic', '/robot_description',
           '-x', spawn_x, '-y', spawn_y, '-z', spawn_z, '-Y', spawn_yaw]
```

#### Why:
* Neobotix upstream defaults `arm_type` to empty (`''`), spawning only the bare mobile platform if omitted.
* Enables setting the robot's initial spawn location anywhere in the Gazebo world at launch time without breaking physics or `/odom` odometry.

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
ros2 launch neo_simulation2 simulation.launch.py spawn_x:=-1.5 spawn_y:=-0.5 spawn_yaw:=0.0
```

### Launch Real Robot Bringup
```bash
ros2 launch neo_mpo_700-2 bringup.launch.py arm_type:=ur5 gripper_type:=2f_85 d435_enable:=True robot_ip:=<UR5_IP>
```

### Teleoperate Base via Keyboard
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

---

## 6. GPU Accelerated Inspection Stack (CUDA 12.1, PyTorch & cuRobo)

Run inside the container to equip Steve with full GPU motion planning, cuRobo kinematics, and custom CUDA ray scoring:

```bash
# 1. NVIDIA CUDA 12.1 NVCC Compiler (no host kernel driver conflict in Docker)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y --no-install-recommends cuda-nvcc-12-1 cuda-cudart-dev-12-1
sudo ln -sf /usr/local/cuda-12.1 /usr/local/cuda
sudo ln -sf /usr/local/cuda-12.1/bin/nvcc /usr/local/bin/nvcc

# Environment variables
echo 'export CUDA_HOME=/usr/local/cuda' | sudo tee /etc/profile.d/cuda.sh
echo 'export PATH=/usr/local/cuda/bin:$PATH' | sudo tee -a /etc/profile.d/cuda.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' | sudo tee -a /etc/profile.d/cuda.sh
source /etc/profile.d/cuda.sh

# 2. PyTorch, Warp, & 3D Geometry Libraries
pip install --upgrade pip
pip install torch==2.4.1+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "numpy>=2" ninja warp-lang trimesh open3d rerun-sdk

# 3. Build & Install cuRobo with Native CUDA Kernels
git clone https://github.com/NVlabs/curobo.git /tmp/curobo
cd /tmp/curobo
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"
pip install --no-build-isolation .   # not -e: cuRobo has no PEP 660 build_editable hook

# 4. Verify Stack
python3 -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python3 -c "import curobo; print('cuRobo:', curobo.__version__)"
```

---

## 7. SAM2 Segmentation (scan / both modes)

`--mode scan` and `--mode both` need SAM2. The container ships neither the package nor the
checkpoint, and `scripts/sync_ros_workspace.sh` only rsyncs `nbv_planner/`, so `checkpoints/`
never arrives.

```bash
# 1. Runtime dependencies SAM2 imports but does not vendor
pip install --no-cache-dir opencv-python hydra-core omegaconf iopath

# 2. SAM2 itself, WITHOUT its dependency resolution
pip install --no-cache-dir --no-deps --no-build-isolation \
  "git+https://github.com/facebookresearch/sam2.git"

# 3. Verify torch was not replaced
python3 -c "import torch; print(torch.__version__)"                      # must stay 2.4.1+cu121
python3 -c "from curobo.curobolib import kinematics_fused_cu; print('curobo ok')"
python3 -c "import sam2; print(sam2.__file__)"
```

Then copy the checkpoint in from the host:

```bash
docker cp <repo>/checkpoints <container>:/home/ws/src/parallel_robotics_lab/checkpoints
```

### Why `--no-deps` is mandatory

SAM2 declares an unpinned `torch` requirement. Installing it normally pulls the newest wheel
(torch 2.14.0+cu130), which silently replaces the pinned 2.4.1+cu121 from section 6. cuRobo's
prebuilt kernels were compiled against 2.4.1 and then fail with:

```
ImportError: kinematics_fused_cu...so: undefined symbol:
  _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
```

cuRobo falls back to JIT compiling, which also fails (`check_cuda.h: No such file or directory`)
because its headers are not on the include path. The whole inspection node dies at
`_verify_camera_kinematics`, long before any segmentation runs.

Recovery if torch was already clobbered:

```bash
pip install --no-cache-dir --force-reinstall \
  torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

### Notes

- The package installs under the name `SAM-2`, so `pip install "sam2 @ git+..."` fails with
  `inconsistent name: filename has 'sam2', but metadata has 'sam-2'`. Use the bare git URL.
- `pip install --user` writes to `/home/djyjyh/.local`, which is bind-mounted from the host.
  Breakage there follows you across container rebuilds.
- `inspect.launch.py` does not forward `segmenter` or `scan_views` to the node, so those stay at
  their node defaults (`sam`, 8). To override, run the node directly:

```bash
ros2 run nbv_planner_ros inspection_node --ros-args \
  -p mode:=scan -p segmenter:=depth -p object_name:=mustard_bottle
```

### numpy

Do not pin `numpy<2.0`. torch 2.4.1+cu121, cuRobo, open3d and rerun-sdk 0.36.1 all run on
numpy 2.x, and rerun-sdk and opencv-python both *require* `numpy>=2`. Downgrading only creates
conflicts.

Working combination, verified in the container and on the host:

```
torch       2.4.1+cu121
torchvision 0.19.1+cu121
numpy       2.2.6
rerun-sdk   0.36.1
SAM-2       1.0
```

The `sam-2 1.0 requires torch>=2.5.1` pip warning is cosmetic. It is a declared floor SAM2 does
not actually exercise; the same pairing runs on the host.
