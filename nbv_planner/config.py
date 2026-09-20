"""Project constants: one place for every number and path the whole codebase shares."""

import os
import numpy as np

# -- Paths --------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_PATH  = os.path.join(PROJECT_ROOT, "third_party/shelf_gym_repo/shelf_gym/meshes/urdf")
URDF_PATH   = os.environ.get("NBV_URDF_PATH", os.path.join(PROJECT_ROOT, "sim/models/steve.urdf"))
YCB_ROOT    = os.environ.get("NBV_YCB_ROOT", os.path.join(ASSET_PATH, "ycb_objects"))
CACHE_DIR   = os.path.join(PROJECT_ROOT, "reachability")

CUROBO_CONFIGS_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curobo_configs")
ROBOT_CONFIG_PATH        = os.path.join(CUROBO_CONFIGS_DIR, "steve_ur5.yml")
GRADIENT_TRAJOPT_FILE    = os.path.join(PROJECT_ROOT, "nbv_planner/curobo_configs/gradient_trajopt.yml")
FINETUNE_TRAJOPT_FILE    = os.path.join(PROJECT_ROOT, "nbv_planner/curobo_configs/finetune_trajopt.yml")

DEFAULT_YCB_OBJECT = "YcbMustardBottle"


def ycb_urdf(name: str) -> str:
    """Path to a vendored YCB object's URDF."""
    return os.path.join(YCB_ROOT, name, "model.urdf")


def ycb_names() -> list[str]:
    """Every vendored object that ships a URDF."""
    if not os.path.isdir(YCB_ROOT):
        return [DEFAULT_YCB_OBJECT]
    return sorted(d for d in os.listdir(YCB_ROOT) if os.path.isfile(ycb_urdf(d)))

# -- Robot link names ---------------------------------------------------------

BASE_LINK = "base_link"
EE_LINK   = "dummy_camera_link"  # the camera, not the gripper; that's what we're placing

# -- Sampling grid ------------------------------------------------------------

N_RADIUS     = 2
N_AZIMUTH    = 36
ELEVATION_DEG = (20.0, 55.0, 3)  # min, max, count

# -- Camera look-at -----------------------------------------------------------

WORLD_UP_Z           = np.array([0.0, 0.0, 1.0])
WORLD_UP_Y_FALLBACK  = np.array([0.0, -1.0, 0.0])
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
COLLISION_SPHERE_BUFFER_M = 0.015  # 15 mm: CuRobo robot link collision sphere inflation margin
MOTION_PLAN_MAX_ATTEMPTS  = 4      # Trajectory optimization attempts per candidate

# -- Start pose -----------------------------------------------------------------
# Reached with cuRobo before the planner starts. Robot base frame (sim: base at world origin).
# Default: camera ~0.4 m from the table placement spot at ~45 deg, so the object and table are in view.

START_CAMERA_POSITION_BASE = (0.50, 0.0, 1.15)
START_LOOK_AT_BASE         = (0.785, 0.0, 0.85)
START_SAFETY_RADIUS_M      = 0.15   # box kept clear around the look-at point while the object is unknown
START_ROTATION_TOL_RAD     = 0.05
START_PLAN_ATTEMPTS        = 3      # trajectory optimisation is seeded randomly, so retry a blocked start pose

# -- Scan vs CAD alignment (mode "both") -----------------------------------------

ALIGN_MAX_CENTER_ERROR_M   = 0.015  # scanned box center vs CAD box center
ALIGN_MAX_POINT_DISTANCE_M = 0.010  # 95th percentile distance from scanned points to the CAD surface

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