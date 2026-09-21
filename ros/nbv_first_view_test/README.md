# First view test

Stage 1 of the inspection pipeline on its own: drive the arm to the start pose,
capture one RGB-D frame, and show it in Rerun. No CAD mesh, no surface sampling,
no NBV loop.

It imports `RosRobot`, `_reach_start_pose` and `NBVVisualizer` rather than
reimplementing them, so a pass here means the path the inspection node actually
takes works up to the first image. That is the difference from
`nbv_start_pose_test`, which plans and executes its own motion and captures no
camera frames.

Run the simulation and `steve_sim_prep` first, then:

```bash
ros2 launch nbv_first_view_test first_view.launch.py
```

Or directly, which needs `NBV_YCB_ROOT` because only the launch file sets it:

```bash
ros2 run nbv_first_view_test first_view_node --ros-args -p viz:=true
```

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

Rerun shows the table, the ground, the posed robot from its live URDF, the
camera frustum at the start pose with the RGB image on its image plane, and the
look-at point.

Exits nonzero on failure. `keep_alive:=true` holds the viewer open after
`FIRST VIEW OK`; Ctrl-C to quit.

## What a failure here tells you

| symptom | meaning |
| --- | --- |
| `Start pose N/60` lines | the configured pose did not plan and it is working through the fallback ring |
| `Camera kinematics ... does not match` | the cuRobo model disagrees with the live robot, checked after the arm settles |
| `config_file not found` | bad path, or the packaged config was not installed |
| low or zero valid depth | the camera is publishing but sees nothing at the start pose |
