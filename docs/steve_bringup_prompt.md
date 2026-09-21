# Steve container bringup

Bring up the Steve simulation in the ROS 2 container and leave Gazebo and the scene
running with the GUI visible. Stop after step 5 and report.

## Environment

- ROS 2 Humble, Gazebo Classic, inside a devcontainer. Workspace is `/home/ws`, bind
  mounted from the host at `~/dev/steve_ros2_ws_student`.
- Robot: Neobotix MPO-700 base with a UR5 arm, Robotiq 2F-85 gripper, RealSense D435
  on the wrist. Base frame `base_link`, camera frame `camera_link`.
- `ROS_DOMAIN_ID=42`, `ROS_LOCALHOST_ONLY=1`. Every new shell needs
  `source /home/ws/install/setup.bash`.
- The planner lives in a separate repo, synced into the workspace by
  `~/dev/Parallel_Robotics_Lab/scripts/sync_ros_workspace.sh` (run on the host, not in
  the container).

## Steps

**1. apt dependencies.** Skip any already installed.

```bash
sudo apt update && sudo apt install -y \
  ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control ros-humble-gazebo-dev \
  ros-humble-robotiq-description ros-humble-ur-robot-driver ros-humble-ur-controllers \
  ros-humble-ur-client-library ros-humble-ur-dashboard-msgs ros-humble-ur-msgs \
  ros-humble-ur-description libspnav-dev libspnav0
pip install --no-cache-dir "setuptools==58.2.0"
```

**2. Build only what the inspection stack needs.** Building the whole workspace fails on
the nav2 packages and none of them are used here.

```bash
cd /home/ws
colcon build --symlink-install --packages-up-to \
  neo_simulation2 steve_sim_prep nbv_planner_ros realsense2_description
source install/setup.bash
```

**3. Clear any stale Gazebo.** A previous crash leaves gzserver holding port 11345, and
the next launch then fails with `Header is empty` and `/spawn_entity unavailable`.

```bash
pkill -9 -f 'gzserver|gzclient'; sleep 2
```

**4. Gazebo, terminal 1.**

```bash
source /home/ws/install/setup.bash
ros2 launch neo_simulation2 simulation.launch.py \
  spawn_x:=-1.5 spawn_y:=-0.5 spawn_yaw:=0.0 gui:=true
```

Wait for `Successfully spawned entity [mpo_700]` and both controller spawners to report
`Configured and activated`. If the spawners loop on
`waiting for service /controller_manager/list_controllers`, gzserver did not come up:
go back to step 3, then retry with `gui:=false` and attach `gzclient` separately.

**5. Scene, terminal 2.**

```bash
source /home/ws/install/setup.bash
ros2 launch steve_sim_prep prepare_sim.launch.py \
  object_name:=mustard_bottle table_y:=-1.1
```

`table_y:=-1.1` places the table clear of the robot's base. Without it the table lands
where the arm cannot reach past its own cabinet.

## Verify before reporting success

```bash
ros2 node list          # expect gazebo, controller_manager, robot_state_publisher,
                        # joint_state_broadcaster, joint_trajectory_controller
python3 - <<'PY'
import rclpy, time
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
rclpy.init(); n=Node("check"); b=Buffer(); TransformListener(b,n)
end=time.monotonic()+6
while time.monotonic()<end: rclpy.spin_once(n,timeout_sec=0.1)
for f in ("table_frame","object_frame","camera_link","cabinet_link"):
    try:
        t=b.lookup_transform("base_link",f,rclpy.time.Time()).transform.translation
        print("%-14s [%.3f, %.3f, %.3f]"%(f,t.x,t.y,t.z))
    except Exception:
        print("%-14s MISSING"%f)
PY
```

All four frames must resolve. `object_frame` missing means step 5 did not run or failed.

## Known failures

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Could NOT find SPNAV` | `libspnav-dev` missing | step 1 |
| `nav2_ros_common` not found | image has a newer nav2 than this source expects | use the `--packages-up-to` list in step 2, do not build everything |
| `package 'gazebo_ros' not found` | gazebo apt packages missing | step 1 |
| `option --uninstall not recognized` | setuptools too new for colcon `--symlink-install` | `pip install setuptools==58.2.0` |
| `Header is empty`, `/spawn_entity unavailable` | stale gzserver | step 3 |
| Gazebo dies after a few minutes | GPU VRAM exhausted | `gui:=false`, attach `gzclient` only when looking |

## Do not

- Run `colcon build` without `--packages-up-to`. It fails on nav2 and wastes minutes.
- Install anything unpinned that depends on torch. An unpinned install once pulled
  torch 2.14 over the pinned 2.4.1+cu121 and broke cuRobo's prebuilt kernels.
- Launch the inspection node. That is a separate task, after this one reports success.
