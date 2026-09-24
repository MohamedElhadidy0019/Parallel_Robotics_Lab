# Running on the real Steve

Laptop connected straight to the robot by Ethernet, everything running in the
`steve_ros2_ws_student` container on the laptop. The robot's onboard PC runs the
arm, base and TF. The laptop runs the D415 camera driver and the planner.

Sim instructions are in `steve_run_inspection.md`. The same packages run on both;
`target:=real` switches config, clock and safety gate.

---

## What is different on the robot

| | sim | real |
| --- | --- | --- |
| arm | UR5 + Robotiq, prefix `ur5` | UR5e, no gripper, prefix `ur5e` |
| controller | `joint_trajectory_controller` | `scaled_joint_trajectory_controller`, active only while External Control runs on the pendant |
| camera | Gazebo D435 | wrist D415 on the laptop's USB, pose from `handeye_result.json` |
| table, object | published by `steve_sim_prep` | table published by the launch file from measured numbers; no object frame |
| modes | `cad`, `scan`, `both` | `scan` only (the real box has no CAD model) |
| clock | `/clock` | wall time |
| ROS | domain 42, localhost only | domain 74, Fast DDS, 192.168.60.0/24 |

On `target:=real`, before anything moves, the nodes check:

1. The live joints are prefixed `ur5e` (catches pointing a real run at the sim, and the reverse).
2. `robot_mode` is RUNNING and `safety_mode` is NORMAL.
3. `scaled_joint_trajectory_controller` is `active`.
4. `confirm:=go` was given. Without it the run is a **dry run**: it plans the
   first motion, draws it in Rerun, prints its duration and peak joint speed, and
   stops. Nothing moves.

---

## One time: laptop network

On the **host**, not in the container. `eno1` is this laptop's Ethernet port.
`.51` avoids the lab PC, which uses `.50`.

```bash
nmcli con add type ethernet ifname eno1 con-name steve \
  ipv4.method manual ipv4.addresses 192.168.60.51/24 ipv6.method disabled
nmcli con up steve
ping -c3 192.168.60.90
```

If ping works but no topics show up later, check the firewall is not dropping DDS:

```bash
sudo ufw status
sudo ufw allow from 192.168.60.0/24   # only if ufw is active
```

## One time: calibration file

The hand-eye result lives in the robot repo. The config expects it at
`/home/ws/nbv_scratch/handeye_result.json`, which is this path on the host:

```bash
mkdir -p ~/dev/steve_ros2_ws_student/nbv_scratch
cp <steve-real-robot-nbv>/nbv_scratch/handeye_result.json ~/dev/steve_ros2_ws_student/nbv_scratch/
```

It is only valid while the D415 stays in the mount it was calibrated in.

---

## Every session

### 1. At the robot

1. Power the robot. The onboard bringup starts on its own (`screen -r bringup` on
   the onboard PC to watch it; `Ctrl+A D` to leave, never `Ctrl+C`).
2. On the pendant: power the arm, release brakes, load and **Play** the External
   Control program, leave it running.
3. Pendant speed slider at 100%. The planner's own `speed_scale` (0.1) does the
   slowing down.
4. Someone stands at the e-stop.

### 2. Laptop

Plug in the Ethernet cable, and the D415 USB cable **before** starting the
container: the container does not see USB devices plugged in after it starts.

Sync and build if the code changed (host, then container):

```bash
~/dev/Parallel_Robotics_Lab/scripts/sync_ros_workspace.sh
```

```bash
cd /home/ws && source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select nbv_planner_ros nbv_first_view_test
```

**Every** container terminal starts with:

```bash
source /home/ws/src/parallel_robotics_lab/ros/nbv_planner_ros/scripts/steve_real_env.sh
```

It should say the onboard PC is reachable and list about 40 topics. Discovery
is flaky on a cold start; run it again once before concluding anything.

Check the arm state:

```bash
ros2 topic echo --once --qos-durability transient_local /io_and_status_controller/robot_mode      # mode: 7
ros2 topic echo --once --qos-durability transient_local /io_and_status_controller/safety_mode     # mode: 1
ros2 control list_controllers | grep scaled                                                        # active
ros2 topic echo --once /speed_scaling_state_broadcaster/speed_scaling                              # data: 100.0
```

### 3. Camera (terminal 1)

```bash
ros2 launch realsense2_camera rs_launch.py \
  camera_name:=d415_calib camera_namespace:=d415_calib align_depth.enable:=true
```

`align_depth.enable:=true` is required. Without it the depth topic the planner
reads does not exist. Check:

```bash
ros2 topic hz /d415_calib/d415_calib/aligned_depth_to_color/image_raw
```

### 4. Measure the table (first session, and whenever the table moves)

The numbers in `inspection_config_real.yaml` marked MEASURE are Gazebo values.
The quickest way to measure is with the arm itself: freedrive the flange onto a
point, then read it in the robot base frame:

```bash
ros2 run tf2_ros tf2_echo base_link ur5etool0
```

- **Table:** touch the centre of the top. `x y` go into `table_xyz:="x y 0"`,
  `z` into `table_height`. Set `table_depth` and `table_width` from a tape
  measure, and `table_yaw` so `table_width` runs along the long side (1.5708
  puts it along the robot's x).
- **Object:** touch the centre of its top. `start_look_at_base` is that point
  minus half the object's height in z.
- **Start camera position:** about 0.34 m back towards the robot and 0.34 m up
  from the look-at point. That is the sim's 45 degree, 0.48 m view.

Reference: the arm's base sits at `(0.205, 0, 0.766)` in `base_link`, yawed -90
degrees. The last box pose the robot repo estimated was at about
`(0.42, -0.14, 0.83)` in `base_link`.

Edit the config, then sync and build as in step 2.

### 5. Dry run the first view (terminal 2)

Start a Rerun viewer first, the same way as in sim.

```bash
ros2 launch nbv_first_view_test first_view.launch.py target:=real \
  table_xyz:="0.0 -0.60 0.0" table_yaw:=1.5708
```

Expect, in order: `Graph matches target real`, `Injected nbv_camera_optical_frame`,
`Publishing ur5etool0 -> nbv_camera_optical_frame`, `Camera kinematics: <1 mm`,
then `DRY RUN OK, the arm did not move: planned ... peak joint speed N deg/s`.

In Rerun, before going further:

1. `world/planned/collision_spheres`: a row of red balls should cover the D415
   on the wrist. If they sit beside the camera instead, stop and fix
   `collision_sphere_overrides`.
2. `world/planned/camera_path`: the orange line must stay clear of the table
   box and anything the arm could hit.
3. The table box should be where the real table is.

A dry run also reports, without stopping, any safety check that would block a
real run (`Would refuse to move: ...`).

### 6. First real motion

```bash
ros2 launch nbv_first_view_test first_view.launch.py target:=real confirm:=go \
  table_xyz:="0.0 -0.60 0.0" table_yaw:=1.5708
```

Arm moves to the start pose at `speed_scale` 0.1, captures one frame, prints
`FIRST VIEW OK`. Check the RGB and depth in Rerun show the table and object.

### 7. Inspection, scan mode

Dry run, then real:

```bash
ros2 launch nbv_planner_ros inspect.launch.py target:=real mode:=scan \
  max_views:=3 scan_views:=5 segmenter:=depth table_xyz:="0.0 -0.60 0.0" table_yaw:=1.5708
```

```bash
ros2 launch nbv_planner_ros inspect.launch.py target:=real mode:=scan confirm:=go \
  max_views:=3 scan_views:=5 segmenter:=depth table_xyz:="0.0 -0.60 0.0" table_yaw:=1.5708
```

`segmenter:=depth` (plane removal plus clustering) for the first runs:
cuRobo and SAM2 together are tight on a 4 GB GPU. Switch to `segmenter:=sam`
once it works. The dry run stops at the first motion, which is the start pose;
every later motion only happens with `confirm:=go`.

---

## Failures

| message | cause | fix |
| --- | --- | --- |
| `onboard PC ... NOT reachable` | cable, or no 192.168.60.x address on the host | one-time network step; `ip -br addr show eno1` |
| reachable but 0 topics | discovery is flaky, or firewall | rerun the env script; `sudo ufw status` |
| `expects joints prefixed 'ur5e' but ... 'ur5'` | env script not sourced, so you see your own sim | source it in this terminal |
| `No ['mode', 'safety'] ... after 15 s` | arm bringup down or wrong domain | `screen -r bringup` on the onboard PC |
| `robot_mode is 3` / `safety_mode is 5` | arm powered off, or a safeguard stop | pendant; a SICK scanner stop needs the TA, there is no software bypass |
| `scaled_joint_trajectory_controller is inactive` | External Control not playing | press Play on the pendant |
| `Robot inputs missing ... ['camera_info', 'depth']` | camera node not running, or no aligned depth | step 3, with `align_depth.enable:=true` |
| `No RealSense devices were found` | D415 plugged in after the container started | replug, restart the container |
| `handeye_result.json` not found | calibration not copied | one-time calibration step |
| `cuRobo will not plan from the arm's current pose: ...WORLD_COLLISION` | arm inside the table box or the keep-clear box at the look-at point | freedrive it clear, or fix the table numbers |
| `...SELF_COLLISION` | camera spheres overlap the arm at this pose | move the arm; if it persists, check the spheres in Rerun |
| `Trajectory execution timed out ... speed scaling 30%` | slider low, or the arm paused | slider to 100% |
| `mode:=cad cannot run on target:=real` | no CAD model or object frame on the robot | `mode:=scan` |
| `WORLD_COLLISION` right after changing the table numbers | an old `nbv_table_frame` publisher from a killed run still publishes the old table | `ros2 node list \| grep nbv_table_frame`, kill it |

## Not checked on the robot yet

These were tested offline against the real URDF (`steve_hardware_bringup`'s
`mmo_700_real.urdf`), with the real hand-eye file and four recorded real arm
poses, and in Gazebo. None of them has run on the physical arm.

1. The live `/robot_description` matches that URDF. The file in the repo is two
   documents pasted together, so the first part was used.
2. The D415 spheres cover the real camera (step 5, item 1).
3. `speed_scaling` is published in percent (step 2 shows `data: 100.0`). If it
   shows `1.0`, the timeouts are ten times too long, which is safe but slow.
4. The table and start pose numbers, until step 4 is done.
