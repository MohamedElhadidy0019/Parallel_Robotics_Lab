# Autonomous Next-Best-View (NBV) 3D Scanning Pipeline

Autonomous Next-Best-View (NBV) 3D scanning system on a simulated UR5 robot arm with Robotiq-85 gripper and wrist-mounted RGB-D camera in PyBullet.

---

## 1. System Architecture Diagram

```
 +-------------------------------------------------------------------------+
 |                              OFFLINE ASSETS                             |
 |  - YCB Object Visual CAD Mesh (.obj)                                    |
 |  - URDF Model (Inertial COM + Visual Frame Offsets)                     |
 +------------------------------------+------------------------------------+
                                      |
                                      v
 +-------------------------------------------------------------------------+
 |                      STAGE 1: SCENE & TARGET SETUP                      |
 |  - PyBullet settles object on table                                     |
 |  - Exact Mesh Transform: T_world_vis = T_world_ine * T_ine^-1 * T_vis   |
 |  - Sample target surface points (full CAD mesh surface)                  |
 |  - Initialize CPU CoverageTracker (scipy.spatial.cKDTree)               |
 +------------------------------------+------------------------------------+
                                      |
                                      v
 +-------------------------------------------------------------------------+
 |                 STAGE 2: VIEWPOINT CANDIDATE GENERATION                 |
 |  - Sample spherical orbit shell around object (radius, azimuth, elev)   |
 |  - Compute Look-At quaternions aimed at object center                   |
 |  - cuRobo Batched IK Filter: Prune kinematically unreachable poses      |
 +------------------------------------+------------------------------------+
                                      |
                                      v
 +-------------------------------------------------------------------------+
 |             STAGE 3: CUSTOM CUDA RAY SCORING (PYTORCH EXT)              |
 |  - Parallel GPU threads: M candidate views x N unseen target points     |
 |  - Surface normal backface culling                                      |
 |  - Moller-Trumbore ray-triangle intersection with early-exit occlusion  |
 |  - Return expected information gain per viewpoint in < 250ms            |
 +------------------------------------+------------------------------------+
                                      |
                                      v
 +-------------------------------------------------------------------------+
 |             STAGE 4: COLLISION-AWARE MOTION PLANNING (CUROBO)           |
 |  - Sort candidate viewpoints by expected information gain               |
 |  - cuRobo MotionGen optimizes collision-free trajectory past table      |
 |  - Rank Fallback: If top candidate is blocked, try rank #2, #3, etc.    |
 +------------------------------------+------------------------------------+
                                      |
                                      v
 +-------------------------------------------------------------------------+
 |                 STAGE 5: CAPTURE, BACKPROJECT & UPDATE                  |
 |  - Render RGB-D depth buffer at achieved camera link pose               |
 |  - Depth linearization + Edge discontinuity filtering                   |
 |  - 2D Depth -> 3D World frame point cloud backprojection                |
 |  - CPU KDTree confirms observed target surface points                   |
 |  - Check stopping criteria: Target coverage reached (e.g. 95%)          |
 +------------------------------------+------------------------------------+
                                      |
                                      v
 +-------------------------------------------------------------------------+
 |                    STAGE 6: EXPORT & VISUALIZATION                      |
 |  - Save dense reconstructed point cloud (captures/scan_<object>.ply)    |
 |  - Save color-coded coverage mesh (captures/coverage_<object>.ply)      |
 |  - Interactive 3D visualization in Rerun or PyBullet GUI                |
 +-------------------------------------------------------------------------+
```

---

## 2. Core Modules Breakdown

### 1. Scene & Physics (`nbv_core/sim_env.py`)
* **Robot Setup:** UR5 6-DOF manipulator with Robotiq-85 gripper and wrist-mounted RealSense D435 camera.
* **Objects:** 12 YCB benchmark objects placed on a tabletop workspace.
* **Settling Physics:** Physics-steps object until linear and angular velocity fall below tolerance.
* **Inertial vs Visual Frame Math:**
  PyBullet `getBasePositionAndOrientation()` reports the Center-of-Mass (Inertial) frame. The visual CAD mesh is transformed into world coordinates via:
  $$T_{\text{world\_vis}} = T_{\text{world\_inertial}} \cdot T_{\text{link\_inertial}}^{-1} \cdot T_{\text{link\_visual}}$$

### 2. Candidate Generation & Kinematics (`nbv_core/reachability.py`)
* **Orbit Shell:** Generates camera viewpoints across spherical shell radii, azimuths, and elevations above the tabletop plane.
* **Camera Look-At:** Computes camera orientation targeting object centroid with positive Z camera optical axis.
* **cuRobo Batched IK:** Parallel inverse kinematics solver filters candidate pool down to reachable poses before trajectory planning.

### 3. Custom CUDA Ray Scoring Kernel (`nbv_core/csrc/` & `nbv_core/ray_scoring.py`)
* **PyTorch C++/CUDA Extension:** JIT-compiled native kernel (`score_candidate_views_cuda`).
* **Moller-Trumbore Algorithm:** Evaluates ray-triangle intersections against the object CAD mesh.
* **Early-Exit Occlusion:** Thread terminates immediately upon the first blocking triangle hit.
* **Backface Culling:** Ignores surface normals angled $> 90^\circ$ away from the camera optical axis.
* **Performance:** Evaluates 200 viewpoints against 3,500 target points and 15,000 mesh triangles in $\sim 200 - 250\text{ms}$ on GTX 1650.

### 4. Collision-Aware Motion Planning (`nbv_core/motion_planning.py`)
* **cuRobo MotionGen:** GPU gradient trajectory optimizer with collision spheres avoiding the table slab and object bounding boxes.
* **Rank-Fallback Execution:** If candidate #1 fails trajectory planning due to table collision constraints, candidate #2, #3, etc. are attempted automatically.

### 5. Sensing & Coverage Tracking (`nbv_core/camera.py` & `nbv_core/coverage.py`)
* **Depth Linearization:** Converts non-linear OpenGL depth buffer values to metric distance.
* **Edge Masking:** Drops pixel boundary steps $> 2\text{cm}$ to avoid flying edge artifacts.
* **Coverage Tracker:** CPU-based `scipy.spatial.cKDTree` matches reconstructed points to surface targets within 8mm radius and normal alignment $> 45^\circ$.
* **Target Surface:** Evaluates complete CAD mesh surface; points on the bottom contact area remain naturally occluded by the tabletop.

---

## 3. Quickstart Guide

### Activate Environment

```bash
source env.sh
```

### Run Autonomous NBV Scan

```bash
# Headless run on Mustard Bottle (default 8 views)
python main.py YcbMustardBottle --views 8

# Run with PyBullet GUI to watch robot move live
python main.py YcbMustardBottle --views 6 --gui

# Run and automatically launch interactive Rerun 3D viewer
python main.py YcbMustardBottle --views 6 --viz

# Run across other YCB objects
python main.py YcbChipsCan --views 8 --viz
python main.py YcbGelatinBox --views 6 --viz
python main.py YcbCrackerBox --views 8 --viz
```

---

## 4. Viewing Scan Results

### Interactive 3D Viewer (`view_scan.py`)

```bash
# View reconstructed point cloud in Rerun
python view_scan.py captures/scan_YcbMustardBottle.ply

# View coverage-colored mesh (Green = Seen, Gray = Unseen)
python view_scan.py captures/coverage_YcbMustardBottle.ply

# View in PyBullet OpenGL window instead of Rerun
python view_scan.py captures/scan_YcbMustardBottle.ply --pb
```

---

## 5. Test Suite

Run full test suite (39 unit and integration tests):

```bash
source env.sh
pytest tests/ -v
```
