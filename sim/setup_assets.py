"""Asset setup script to isolate Steve robot meshes and assets."""

import os
import shutil
import urllib.request

DEST_MESHES = os.path.join(os.path.dirname(__file__), "models", "meshes")
os.makedirs(DEST_MESHES, exist_ok=True)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_NEO_WS = os.environ.get("STEVE_WS", os.path.join(os.path.dirname(REPO_ROOT), "steve_ros2_ws_student"))
SHELF_GYM_MESHES = os.path.join(REPO_ROOT, "third_party", "shelf_gym_repo", "shelf_gym", "meshes")

# 1. Local files in steve_ros2_ws_student
mesh_files_to_copy = [
    ("src/neo_mpo_700-2/robot_model/meshes/MPO-700-BODY.dae", "MPO-700-BODY.dae"),
    ("src/neo_mpo_700-2/robot_model/meshes/MPO-700-HEAD.dae", "MPO-700-HEAD.dae"),
    ("src/neo_mpo_700-2/robot_model/meshes/MPO-700-WHEEL.dae", "MPO-700-WHEEL.dae"),
    ("src/neo_mpo_700-2/robot_model/meshes/MPO-700-caster.dae", "MPO-700-caster.dae"),
    ("src/neo_mpo_700-2/robot_model/meshes/MPO-700-charge-port.dae", "MPO-700-charge-port.dae"),
    ("src/neo_mpo_700-2/robot_model/meshes/SICK-S300.dae", "SICK-S300.dae"),
    ("src/realsense-ros/realsense2_description/meshes/d435.dae", "d435.dae"),
]

for rel_src, filename in mesh_files_to_copy:
    src_path = os.path.join(SRC_NEO_WS, rel_src)
    dst_path = os.path.join(DEST_MESHES, filename)
    if os.path.exists(src_path) and not os.path.exists(dst_path):
        shutil.copy2(src_path, dst_path)

# 2. Cabinet mesh
cabin_dst = os.path.join(DEST_MESHES, "cabin.dae")
if not os.path.exists(cabin_dst):
    cabin_url = "https://raw.githubusercontent.com/neobotix/neo_simulation2/humble/robots/mpo_700/meshes/cabin.dae"
    urllib.request.urlretrieve(cabin_url, cabin_dst)

# 3. UR5 and Robotiq-85 meshes from shelf_gym
ur5_dst = os.path.join(DEST_MESHES, "ur5")
if not os.path.exists(ur5_dst) and os.path.exists(os.path.join(SHELF_GYM_MESHES, "ur5")):
    shutil.copytree(os.path.join(SHELF_GYM_MESHES, "ur5"), ur5_dst)
    print(f"Copied UR5 meshes to {ur5_dst}")

robotiq_dst = os.path.join(DEST_MESHES, "robotiq_85")
if not os.path.exists(robotiq_dst) and os.path.exists(os.path.join(SHELF_GYM_MESHES, "robotiq_85")):
    shutil.copytree(os.path.join(SHELF_GYM_MESHES, "robotiq_85"), robotiq_dst)
    print(f"Copied Robotiq-85 meshes to {robotiq_dst}")

print("All robot assets verified and present.")
