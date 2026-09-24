#!/usr/bin/env bash
# Point this container shell at Steve's ROS graph. Source it, do not run it:
#
#   source /home/ws/src/parallel_robotics_lab/ros/nbv_planner_ros/scripts/steve_real_env.sh
#
# The devcontainer defaults to ROS_DOMAIN_ID=42 and ROS_LOCALHOST_ONLY=1, which hides the robot.
# The onboard PC runs domain 74, Fast DDS, on 192.168.60.0/24 (values from the robot repo's
# steve_env.sh). The laptop's own address on that subnet is set on the host, see
# docs/steve_run_real_robot.md, since the container shares the host network.

STEVE_ONBOARD="${STEVE_ONBOARD:-192.168.60.90}"

export ROS_DOMAIN_ID=74
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source /opt/ros/humble/setup.bash
source /home/ws/install/setup.bash
ros2 daemon stop >/dev/null 2>&1

echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY RMW=$RMW_IMPLEMENTATION"
if ping -c1 -W1 "$STEVE_ONBOARD" >/dev/null 2>&1; then
  echo "onboard PC $STEVE_ONBOARD reachable"
  echo "topics seen: $(timeout 10 ros2 topic list --no-daemon 2>/dev/null | wc -l) (expect about 40; retry once if 0)"
else
  echo "onboard PC $STEVE_ONBOARD NOT reachable: check the cable and the host's 192.168.60.x address"
fi
