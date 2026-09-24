# First view test

Stage 1 of the inspection pipeline on its own: drive the arm to the start pose,
capture one RGB-D frame, and show it in Rerun. No CAD mesh, no surface sampling,
no NBV loop.

It imports `RosRobot`, `_reach_start_pose` and `NBVVisualizer` rather than
reimplementing them, so a pass here means the path the inspection node actually
takes works up to the first image. That is the difference from
`nbv_start_pose_test`, which plans and executes its own motion and captures no
camera frames.

## Sim

Run the simulation and `steve_sim_prep` first, then:

```bash
ros2 launch nbv_first_view_test first_view.launch.py target:=sim
```

## Real robot

Full procedure, network setup and failures: `docs/steve_run_real_robot.md`. Needs
the hardware bringup, an External Control program running on the pendant,
`steve_real_env.sh` sourced, and the D415 node. The launch file publishes
`table_frame` itself, since nothing on the real robot does.

```bash
ros2 launch nbv_first_view_test first_view.launch.py target:=real            # dry run
ros2 launch nbv_first_view_test first_view.launch.py target:=real confirm:=go
```

`target` is declared, not detected. Detection failing towards `sim` only wastes
a run; detection failing towards `real` would drive physical hardware with the
safety gate skipped. The node checks the declaration against the live URDF's
joint prefix (`ur5` for Gazebo and the mock, `ur5e` for the arm) and refuses on
a mismatch, so a wrong `target` stops before anything is built.

`target:=real` also:

- uses wall time, since only Gazebo publishes `/clock`
- loads `inspection_config_real.yaml`
- requires `robot_mode` 7 (RUNNING) and `safety_mode` 1 (NORMAL), read with
  Transient Local QoS because both are latched and a Volatile subscription
  receives nothing at all, which would read as a powered-down arm
- requires `scaled_joint_trajectory_controller` to be active
- without `confirm:=go`, plans the start pose, draws the plan and the collision
  spheres in Rerun, prints its peak joint speed, and stops before sending it
- injects the calibrated camera into the URDF handed to cuRobo, from
  `handeye_file` in the config, and publishes the same transform on TF. cuRobo
  reads kinematics from the URDF alone, so a camera published only as a static
  TF is invisible to it. The robot's own URDF is not touched.

The safety gate lives in `nbv_planner_ros.targets` and is shared with the
inspection node.

Build:

```bash
cd /home/ws && source /opt/ros/humble/setup.bash && source install/setup.bash
colcon build --packages-select nbv_first_view_test --symlink-install
source install/setup.bash
```

## Arguments

| argument | default | notes |
| --- | --- | --- |
| `config_file` | packaged `inspection_config.yaml` | must carry the start pose keys |
| `object_name` | `mustard_bottle` | names the Rerun recording only |
| `viz` | true | false runs the motion with no viewer |
| `keep_alive` | true | false exits once the frame is logged |

## Output

```
[1/2] Moving arm to start pose with cuRobo...
      Reached start pose: camera at (...) facing (...) | 0.48 m, 45 deg elevation
      | error 1.3 mm / 0.0 deg | plan 15419 ms, drive 2193 ms
[2/2] Capturing and logging the first frame...
      First frame: 640x480, 61.2% valid depth, range 0.31 to 2.50 m
      Camera at (...) looking at (...)
FIRST VIEW OK
```

Rerun opens with three panels: the world and robot in 3D on the left, the
captured RGB frame top right, the depth frame below it.

The package sends its own blueprint. `NBVVisualizer` lays out Reconstruction,
Coverage and Status HUD panels that stay empty here, and its 3D view carries
`- $origin/start_pose/pinhole/rgb`, so the captured frame is logged and then
hidden. Reusing that layout shows the motion and no picture.

Exits nonzero on failure. `keep_alive:=true` holds the viewer open after
`FIRST VIEW OK`; Ctrl-C to quit.

## What a failure here tells you

| symptom | meaning |
| --- | --- |
| `Start pose N/60` lines | the configured pose did not plan and it is working through the fallback ring |
| `Camera kinematics ... does not match` | the cuRobo model disagrees with the live robot, checked after the arm settles |
| `config_file not found` | bad path, or the packaged config was not installed |
| low or zero valid depth | the camera is publishing but sees nothing at the start pose |
