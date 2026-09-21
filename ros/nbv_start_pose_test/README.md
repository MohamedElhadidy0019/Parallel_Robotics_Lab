# Steve start-pose test

Connects to the already-running Gazebo simulation and sim preparer. Reuses
`nbv_planner_ros.RosRobot`, the live-URDF adapter/converter, and the shared cuRobo
model/world/start-pose modules. It does not launch Gazebo, scan objects, or run NBV.
Rerun displays only the live robot; joint updates continue after motion completes.

The default config is resolved from the installed `nbv_planner_ros` package:
`config/inspection_config.yaml`. Its current Gazebo target is camera position
`[0, -0.26, 1.14]`, looking at `[0, -0.60, 0.80]`, in the robot base frame.
This deliberately loads the package config rather than the inspection executable's
positive-Y fallback when no config file is supplied. The table and object safety
box are included in planning but not displayed. No camera images are required.

Build in the container:

```bash
cd /home/ws
source /opt/ros/humble/setup.bash
source install/setup.bash
colcon build --packages-select nbv_start_pose_test --symlink-install
source install/setup.bash
```

After starting the simulation and `steve_sim_prep`, run:

```bash
ros2 launch nbv_start_pose_test start_pose.launch.py
```

Or directly (also defaults to simulation time and the packaged config):

```bash
ros2 run nbv_start_pose_test start_pose_node
```

Success is reported only after both the controller result and camera TF meet the
target: `START POSE REACHED`. Default tolerances are 20 mm and 0.05 rad. Ctrl-C
exits the viewer publisher and cancels an active goal. Failures exit nonzero.
Do not run the inspection planner or another arm motion node at the same time.

Useful launch arguments:

```bash
# Plan from live state without sending any motion; exit after validation.
ros2 launch nbv_start_pose_test start_pose.launch.py plan_only:=true keep_alive:=false

# Send the robot visualization to an already-running Rerun viewer.
ros2 launch nbv_start_pose_test start_pose.launch.py rerun_url:=rerun+http://127.0.0.1:9876/proxy

# Disable visualization, or use an alternate inspection-format YAML.
ros2 launch nbv_start_pose_test start_pose.launch.py viz:=false
ros2 launch nbv_start_pose_test start_pose.launch.py config_file:=/absolute/path/inspection_config.yaml
```

Motion uses `MotionGen.plan_single` with graph fallback and the returned
interpolated positions, velocities, accelerations, and sample interval. Default
`speed_scale:=0.25` uniformly stretches time by four and scales derivatives
consistently. The client uses simulation-time execution deadlines, a wall-time
clock/feedback-stall watchdog, and controller error text plus joint-error feedback.
It never sends an unplanned home recovery, disables collision checking, or treats
stale joint messages as proof the arm has stopped.

Collision geometry remains the existing NBV adapter's sphere approximation of
the live robot. A successful plan is not a physical execution test. If the start
state is colliding or Gazebo cannot track the motion, the node reports the failure
instead of claiming success or automatically driving through an unchecked path.

Tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest /home/ws/src/parallel_robotics_lab/ros/nbv_start_pose_test/test -q
```

Set `NBV_TEST_LIVE=1` as well to include the read-only live URDF/Rerun recording
test; it opens no GUI and sends no controller goals. `NBV_TEST_COLLISIONS=1`
also prints the current model's overlapping sphere pairs.

Validation in the running container: package build succeeded and all eight tests
passed, including live URDF conversion and Rerun recording. The planning-only
run correctly rejected its then-current state with
`INVALID_START_STATE_SELF_COLLISION`. A later read-only diagnostic, after the
existing inspection config enabled cabinet refitting, showed a 26.9 mm overlap
between `cabinet_link` and `ur5upper_arm_link` collision spheres. These are model
overlaps, not proof of physical mesh penetration. No arm motion was sent during
validation; successful physical arrival has not been verified. The simulator
must start in a state accepted by its collision model before this node can plan.
