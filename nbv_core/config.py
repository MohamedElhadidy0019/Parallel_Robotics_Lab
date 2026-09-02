"""Project constants: one place for every number and path the whole codebase shares."""

import os
import numpy as np

# -- Paths --------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_PATH  = os.path.join(PROJECT_ROOT, "third_party/shelf_gym_repo/shelf_gym/meshes/urdf")
URDF_PATH   = os.path.join(ASSET_PATH, "ur5_robotiq_85.urdf")
YCB_ROOT    = os.path.join(ASSET_PATH, "ycb_objects")
CACHE_DIR   = os.path.join(PROJECT_ROOT, "reachability")

CUROBO_CONFIGS_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curobo_configs")
ROBOT_CONFIG_PATH        = os.path.join(CUROBO_CONFIGS_DIR, "ur5_robotiq_85_camera.yml")
GRADIENT_TRAJOPT_FILE    = os.path.join(CUROBO_CONFIGS_DIR, "gradient_trajopt.yml")
FINETUNE_TRAJOPT_FILE    = os.path.join(CUROBO_CONFIGS_DIR, "finetune_trajopt.yml")

DEFAULT_YCB_OBJECT = "YcbMustardBottle"

# -- Robot link names ---------------------------------------------------------

BASE_LINK = "base_link"
EE_LINK   = "dummy_camera_link"  # the camera, not the gripper; that's what we're placing

# -- Sampling grid ------------------------------------------------------------

N_RADIUS     = 2
N_AZIMUTH    = 36
ELEVATION_DEG = (20.0, 55.0, 3)  # min, max, count

# -- Camera look-at -----------------------------------------------------------

WORLD_UP_Z           = np.array([0.0, 0.0, 1.0])
WORLD_UP_Y_FALLBACK  = np.array([0.0, 1.0, 0.0])
NEAR_VERTICAL_COSINE = 0.99  # cos(~8 deg)

# -- CuRobo IK ----------------------------------------------------------------

IK_POSITION_THRESHOLD_M = 0.005
IK_ROTATION_THRESHOLD_RAD = 0.05
IK_NUM_SEEDS = 20

# -- PyBullet-scipy xyzw → CuRobo wxyz -----------------------------------------

XYZW_TO_WXYZ = (3, 0, 1, 2)

# -- Scene geometry -----------------------------------------------------------

PLACEMENT_DIRECTION = (0.82, 0.57)  # world (x, y) from the base; clears the arm at rest
ORBIT_DEPTH_FRACTION = 0.45         # share of leftover reach spent on shell depth

# -- Motion planning ----------------------------------------------------------

MAX_POSE_ERROR_M         = 0.015
TABLE_CLEARANCE_M        = 0.12
TABLE_COLLISION_HALF_HEIGHT = 0.15

# -- Coverage target & base exclusion -----------------------------------------

BASE_EXCLUSION_HEIGHT_M       = 0.015  # 15mm: bottom of object resting on table excluded from coverage denominator
TABLE_CLEARANCE_MARGIN_M      = 0.003  # 3mm: drop depth points below/on table surface
WORKSPACE_RADIUS_M            = 0.25   # 25cm: horizontal radius enclosing object inspection zone
ROBOT_SELF_FILTER_MIN_DEPTH_M = 0.12   # 12cm: drop robot hand / gripper self-capture points

DEFAULT_CAMERA_WIDTH  = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FOV    = 60.0
DEFAULT_CAMERA_NEAR   = 0.05
DEFAULT_CAMERA_FAR    = 2.0

# OpenGL camera frame (X right, Y up, Z back) -> optical (X right, Y down, Z forward)
T_OPENGL_OPTICAL = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)