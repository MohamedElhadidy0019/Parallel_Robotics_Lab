# Running the inspection stack

Picks up where `steve_bringup_prompt.md` stops. Everything here runs inside the
`steve_ros2_ws_student` container. Three terminals, in order.

Sync first if you changed anything under `nbv_planner/` or `ros/`. This runs on the
host, not in the container:

```bash
~/dev/Parallel_Robotics_Lab/scripts/sync_ros_workspace.sh
```

Then rebuild in the container:

```bash
cd /home/ws && source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select nbv_planner_ros
```

---

## Terminal 1: Gazebo

```bash
source /home/ws/install/setup.bash && cd /home/ws && \
pkill -9 -f 'gzserver|gzclient'; sleep 2; \
ros2 launch neo_simulation2 simulation.launch.py \
  spawn_x:=-1.5 spawn_y:=-0.5 spawn_yaw:=0.0 gui:=false
```

Wait for `Successfully spawned entity [mpo_700]` and both spawners reporting
`Configured and activated`.

`cd /home/ws` is not optional. Launching from anywhere else dies on
`Permission denied: 'robot_name.txt'`, because the launch file writes that file
into the working directory.

`gui:=false` by default. The GPU is a 4 GB GTX 1650 and gzclient competes with
cuRobo for it. Attach `gzclient` separately when you actually want to look.

## Terminal 2: Scene

```bash
source /home/ws/install/setup.bash && cd /home/ws && \
ros2 launch steve_sim_prep prepare_sim.launch.py \
  object_name:=mustard_bottle table_y:=-1.1
```

Wait for `=== Scene Preparation Complete! ===`.

`table_y:=-1.1` is required, not cosmetic. It places the table at base frame
`(0, -0.60, 0.01)`, which is what `start_look_at_base` in `inspection_config.yaml`
aims at. Omit it and the preparer's auto-align path puts the table on the robot's
`+y` side while the planner still aims at `-y`.

## Terminal 3: Inspection

```bash
source /home/ws/install/setup.bash && cd /home/ws && \
export NBV_YCB_ROOT=/home/ws/install/steve_sim_prep/share/steve_sim_prep/models/ycb_objects && \
ros2 run nbv_planner_ros inspection_node --ros-args \
  -p viz:=true -p mode:=cad -p max_views:=2
```

---

## Modes

| mode | what it scores against | stages | needs CAD |
| --- | --- | --- | --- |
| `cad` | the ground truth CAD mesh | 5 | yes |
| `scan` | a mesh built from segmented views | 6 | no |
| `both` | CAD, after checking it against the scan | 6 | yes |

### cad

```bash
ros2 run nbv_planner_ros inspection_node --ros-args \
  -p viz:=true -p mode:=cad -p max_views:=8 -p target_coverage:=0.95
```

Skips discovery entirely and loads the CAD mesh at the pose `object_frame`
publishes. Fastest path, and the one to use when testing motion or planning
changes.

### scan

```bash
ros2 run nbv_planner_ros inspection_node --ros-args \
  -p viz:=true -p mode:=scan -p scan_views:=8 -p max_views:=8 -p segmenter:=sam
```

Adds a discovery stage: segment the start frame, visit a ring of `scan_views`
azimuths merging each detection, then build the mesh from those points. Roughly
two to four minutes on top of cad mode.

`-p segmenter:=depth` skips SAM2 and uses RANSAC plane plus DBSCAN only. Use it
if you hit a CUDA OOM, since cuRobo and SAM2 are resident at the same time.

Coverage in scan mode is measured against what was scanned, not against the CAD,
so it is not comparable to a cad mode number.

### both

```bash
ros2 run nbv_planner_ros inspection_node --ros-args \
  -p viz:=true -p mode:=both -p scan_views:=8 -p max_views:=8
```

Scans, compares the CAD against the scan, then plans NBV against the CAD. Raises
and stops if they disagree by more than 15 mm of centre error or 10 mm at the
95th percentile. The strictest check of the three.

---

## Parameters

| parameter | default | notes |
| --- | --- | --- |
| `mode` | `cad` | `cad`, `scan`, `both` |
| `max_views` | 8 | NBV iterations. Use 2 while iterating |
| `target_coverage` | 0.95 | stops early once reached |
| `scan_views` | 8 | discovery ring slots, scan and both only |
| `segmenter` | `sam` | `sam` or `depth`, scan and both only |
| `object_name` | `mustard_bottle` | must match what terminal 2 spawned |
| `viz` | true | Rerun viewer |
| `config_file` | packaged YAML | must exist, and must carry the start pose keys |

### Why ros2 run rather than ros2 launch

`inspect.launch.py` forwards only `config_file`, `object_name`, `mode`, `viz`,
`max_views`, `target_coverage` and `use_sim_time`. It does not forward
`scan_views` or `segmenter`, so scan mode tuning needs `ros2 run`.

The launch file does set `NBV_YCB_ROOT` for you, which is why `ros2 run` needs
the explicit export:

```bash
ros2 launch nbv_planner_ros inspect.launch.py mode:=cad max_views:=2 viz:=false
```

---

## Test packages

Both run after terminals 1 and 2, in place of terminal 3.

### nbv_first_view_test

Start pose plus the first RGB-D frame in Rerun. Nothing else. Use this first
when something is broken: it runs the same `RosRobot`, `_reach_start_pose` and
`NBVVisualizer` the inspection node uses, so a pass means the production path
works up to the first image.

```bash
ros2 launch nbv_first_view_test first_view.launch.py
```

Rerun shows the world and robot in 3D, the captured RGB frame, and depth beside
it. Reports `FIRST VIEW OK` and exits nonzero on failure.

```bash
ros2 launch nbv_first_view_test first_view.launch.py viz:=false keep_alive:=false
```

### nbv_start_pose_test

Start pose only, with its own motion code rather than the pipeline's. Useful for
isolating whether a problem is in the pipeline or in the robot and controller.
Captures no camera frames.

```bash
ros2 launch nbv_start_pose_test start_pose.launch.py keep_alive:=false viz:=false
```

Plan without sending any motion:

```bash
ros2 launch nbv_start_pose_test start_pose.launch.py plan_only:=true keep_alive:=false
```

Reports `START POSE REACHED` on success and exits nonzero on failure.

Build either one:

```bash
cd /home/ws && source /opt/ros/humble/setup.bash && source install/setup.bash
colcon build --symlink-install --packages-select nbv_first_view_test
source install/setup.bash
```

---

## What a good run looks like

```
[1/5] Moving arm to start pose with cuRobo...
      Reached start pose: camera at (-1.500, -0.760, 1.130) facing (-1.500, -1.100, 0.790)
      | 0.48 m, 45 deg elevation | error 1.3 mm / 0.0 deg | plan 15419 ms, drive 2193 ms
```

The start pose should be reached on the first candidate. Any `Start pose N/60`
line means it failed and is working through the fallback ring.

Roughly 15 s of one time cuRobo warmup before the first plan, then about 90 s for
cad mode at `max_views:=2`.

Outputs land in `captures/`:

- `scan_<object>.ply`, `coverage_<object>.ply` in every mode
- `discovery_<object>.ply`, `object_mesh_<object>.ply` in scan and both

---

## Warnings that are fine

| message | why |
| --- | --- |
| `Converting continuous joint to revolute` | the Robotiq mimic joints have no limits and cuRobo needs one on every movable joint |
| `camera.py RuntimeWarning: invalid value encountered in subtract` | Gazebo depth carries NaN for out of range pixels. Those pixels are correctly excluded already, only the warning is noise |
| `view N: skipped, no reachable collision-free pose` | the far side of the object is genuinely beyond the UR5's 0.85 m reach from a parked base. Slots 2, 3 and 4 have zero reachable candidates in the standard scene |
| `Warning: could not return to the start pose` | the arm finishes a view with the forearm inside the inflated object obstacle, so replanning from there is refused. Costs nothing, discovery is already complete |
| `[cuRobo batch opt error] IndexError` | cuRobo's retry merge indexes a scalar tensor. Caught and treated as a blocked plan rather than losing the run |

---

## Known failures

| symptom | cause | fix |
| --- | --- | --- |
| `Permission denied: 'robot_name.txt'` | terminal 1 not launched from `/home/ws` | `cd /home/ws` first |
| `config_file not found` | bad `-p config_file:=` path | omit it to use the packaged default |
| `No CAD mesh for [...]` | `NBV_YCB_ROOT` unset under `ros2 run` | export it, or use `ros2 launch` |
| `Camera kinematics check: N mm ... does not match` | the arm was still moving when the check ran | wait for it to settle and rerun |
| `Header is empty`, `/spawn_entity unavailable` | stale gzserver | `pkill -9 -f 'gzserver|gzclient'` |
| arm reaches nothing, every goal returns `-4` | arm jammed against geometry from an earlier bad run | restart terminals 1 and 2 |
| Gazebo dies after a few minutes | 4 GB VRAM exhausted | `gui:=false`, attach gzclient only when looking |

---

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  /home/ws/src/parallel_robotics_lab/ros/nbv_start_pose_test/test -q
```

Expect 7 passed, 1 skipped. Set `NBV_TEST_LIVE=1` to include the read only live
URDF and Rerun test, which opens no GUI and sends no controller goals.

---

## Open issues that affect what you see

1. Single pose moves go through `plan_batch`, which silently drops graph
   fallback and can raise `IndexError` on its retry merge. Routing them through
   `plan_single` fixes both.
2. `TrajectoryClient.execute` waits in wall time for a trajectory the controller
   runs in sim time. Harmless while the real time factor is near 1, not
   otherwise.
3. The discovery ring spans a full 360 degrees, but a base parked on one side of
   the table can only reach about five of eight slots.
4. `_verify_camera_kinematics` compares joints and TF sampled at different
   instants, with no wait for the arm to settle.
