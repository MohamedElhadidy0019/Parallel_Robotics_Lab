# NBV Planner (`nbv_planner`)

Autonomous Next-Best-View (NBV) planning and inspection package.

```
                  +----------------------------------------------+
                  |               Target CAD Mesh                |
                  +----------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
| coverage.py                                                                     |
| Surface point and normal sampling with tabletop base contact exclusion          |
+---------------------------------------------------------------------------------+
     |                                                                |
     | Target points (N, 3) & normals (N, 3)                          | Target triangles (T, 3, 3)
     v                                                                v
+------------------------------------+             +------------------------------+
| reachability.py                    |             | ray_scoring.py + csrc/       |
| Spherical shell candidate poses    |             | GPU Moller-Trumbore ray cast |
| GPU batch IK reachability filter   |             | Backface culling & occlusion |
+------------------------------------+             +------------------------------+
     |                                                                |
     | Reachable camera poses (M, 7)                                  | Gain score per pose (M,)
     +-------------------------------+--------------------------------+
                                     |
                                     v
                  +-------------------------------------+
                  | api.py (NBVPlanner)                 |
                  | Rank candidates by visibility gain  |
                  +-------------------------------------+
                                     |
                                     v
                  +-------------------------------------+
                  | motion_planning.py                  |
                  | cuRobo MotionGen trajectory solve   |
                  | Fallback to next ranked candidate   |
                  +-------------------------------------+
                                     |
                                     v
                  +-------------------------------------+
                  | TrajectoryPlan                      |
                  | Joint waypoints (K, 6) & timings    |
                  +-------------------------------------+
```

---

## 1. Submodule Specifications

### `api.py`

High-level coordinator class (`NBVPlanner`). Orchestrates surface sampling, viewpoint generation, reachability filtering, and ray scoring into a unified interface.

#### Interface Contract
* **Inputs**:
  * `mesh_world`: `trimesh.Trimesh` target object mesh in world coordinates.
  * `n_surface_samples`: `int` (default `4000`), count of surface evaluation points.
  * `base_exclusion_z`: `float | None`, world Z height below which target points are excluded.
  * `urdf_path`: `str`, path to robot URDF.
  * `base_link`: `str` (default `"ur5_base_link"`), reference frame link name.
  * `ee_link`: `str` (default `"camera_color_optical_frame"`), end-effector/sensor link name.
* **Outputs**:
  * `sample_candidates(...)` -> `(t_cand: np.ndarray, q_cand: np.ndarray)`
  * `filter_reachability(...)` -> `(reachable_mask: np.ndarray, ik_solutions: np.ndarray)`
  * `score_unvisited_views(...)` -> `(gains: np.ndarray, visibility_matrix: np.ndarray)`
  * `update_coverage(...)` -> `(newly_covered_count: int, coverage_fraction: float)`

---

### `camera.py`

Pure geometric pinhole projection and point backprojection. Contains zero simulation or GUI dependencies.

#### Interface Contract
* **`CameraIntrinsics`**:
  * Inputs: `width: int`, `height: int`, `fov: float` (vertical FOV in degrees), `near: float`, `far: float`.
  * Properties:
    $$\text{fy} = \frac{\text{height}}{2 \tan(\text{fov} / 2)}, \quad \text{fx} = \text{fy}, \quad \text{cx} = \frac{\text{width}}{2}, \quad \text{cy} = \frac{\text{height}}{2}$$
* **`edge_discontinuity_mask(depth_m, threshold_m=0.02)`**:
  * Inputs: `depth_m: np.ndarray` shape `(H, W)`.
  * Outputs: `mask: np.ndarray` shape `(H, W)` boolean.
  * Logic: Rejects boundary step discontinuities where 4-neighbor depth jump exceeds threshold.
* **`backproject_depth(depth_m, intrinsics, rgb=None, drop_edges=True)`**:
  * Inputs:
    * `depth_m`: `(H, W)` metric depth array.
    * `intrinsics`: `CameraIntrinsics` specification.
    * `rgb`: `Optional[np.ndarray]` shape `(H, W, 3)`.
  * Outputs:
    * `points_cam`: `(P, 3)` float32 points in optical frame (+X right, +Y down, +Z forward).
    * `colors`: `(P, 3)` uint8/float colors if provided.
  * Math:
    $$z = \text{depth}(v, u), \quad x = \frac{(u - c_x) \cdot z}{f_x}, \quad y = \frac{(v - c_y) \cdot z}{f_y}$$
* **`transform_points(points, transform_4x4)`**:
  * Inputs: `points: (P, 3)`, `transform_4x4: (4, 4)`.
  * Outputs: `transformed: (P, 3)`.
  * Math:
    $$p_{\text{world}} = R \cdot p_{\text{cam}} + t$$

---

### `reachability.py`

Spherical shell viewpoint sampling and parallel GPU inverse kinematics filtering.

#### Viewpoint Generation Math
Camera positions $t \in \mathbb{R}^3$ are sampled across spherical coordinate intervals:
$$r \in [r_{\min}, r_{\max}], \quad \theta \in [0, 2\pi), \quad \phi \in [\phi_{\min}, \phi_{\max}]$$
$$t = t_{\text{target}} + \begin{bmatrix} r \cos \phi \cos \theta \\ r \cos \phi \sin \theta \\ r \sin \phi \end{bmatrix}$$

Camera look-at orientation ($+Z$ pointing at object center $t_{\text{target}}$):
$$z_{\text{cam}} = \frac{t_{\text{target}} - t}{\|t_{\text{target}} - t\|}$$
$$x_{\text{cam}} = \frac{u_{\text{world}} \times z_{\text{cam}}}{\|u_{\text{world}} \times z_{\text{cam}}\|}, \quad y_{\text{cam}} = z_{\text{cam}} \times x_{\text{cam}}$$
Rotation matrix $R = [x_{\text{cam}} \mid y_{\text{cam}} \mid z_{\text{cam}}]$ converted to quaternion $(x, y, z, w)$.

#### Interface Contract
* **`sample_candidate_camera_poses(...)`**:
  * Inputs: `t_obj_world: (3,)`, `radius: (min, max, count)`, `elevation_deg: (min, max, count)`, `n_azimuth: int`, `z_min_world: float | None`.
  * Outputs: `(t_candidates: (M, 3), q_candidates: (M, 4))`.
* **`ik_filter(urdf_path, base_link, ee_link, t_cand, q_cand, t_base_world, q_base_world_xyzw)`**:
  * Inputs: Candidate poses and robot base frame in world coordinates.
  * Outputs:
    * `reachable: (M,)` boolean array.
    * `q_solutions: (M, DOF)` float32 joint angles satisfying IK tolerance.

---

### `ray_scoring.py` & Native CUDA Kernel (`csrc/`)

Hardware-accelerated viewpoint visibility scoring using custom PyTorch C++/CUDA extension.

```
Ray Origin: Camera Position c_cam
Ray Direction: v = (p_unseen - c_cam) / ||p_unseen - c_cam||

                   +------------------+
                   |  Backface Test   |  cos(alpha) = - (v . n_surface)
                   +------------------+  If cos(alpha) < margin -> REJECT
                            |
                            v [Front-Facing]
                   +------------------+
                   | Moller-Trumbore  |  Intersect ray against mesh triangles
                   | CUDA Kernel      |  t in [eps, dist - eps]
                   +------------------+
                            |
           +----------------+----------------+
           | Hit Blocking Triangle           | No Intersection
           v                                 v
     OCCLUDED (Gain = 0)              VISIBLE (Gain += 1)
```

#### Algorithm Details
1. **Backface Culling**:
   $$\cos \alpha = - (v \cdot n_{\text{target}})$$
   If $\cos \alpha < \tau_{\text{backface}}$ (default `0.35`), target surface normal faces away from camera; marked invisible without ray tracing.
2. **Moller-Trumbore Ray-Triangle Intersection**:
   Evaluates ray $R(t) = c_{\text{cam}} + t v$ against mesh triangle vertices $(v_0, v_1, v_2)$. Computes barycentric coordinates $(u, v)$ and ray distance $t$.
3. **Early-Exit Occlusion**:
   CUDA thread terminates immediately upon finding the first triangle satisfying $t \in [\epsilon, L - \epsilon]$.

#### Interface Contract
* **`score_candidate_views(candidate_positions, unseen_points, unseen_normals, triangles, ...)`**:
  * Inputs:
    * `candidate_positions`: `(M, 3)` float32.
    * `unseen_points`: `(N, 3)` float32.
    * `unseen_normals`: `(N, 3)` float32.
    * `triangles`: `(T, 3, 3)` float32.
  * Outputs:
    * `scores`: `(M,)` int32 count of visible unseen points per candidate.
    * `visibility_matrix`: `(M, N)` uint8 visibility bitmask.

---

### `motion_planning.py`

Collision-aware robot trajectory optimization using cuRobo `MotionGen`.

#### Interface Contract
* **`TrajectoryPlan` Data Structure**:
  ```python
  @dataclass
  class TrajectoryPlan:
      success: bool
      positions: np.ndarray        # (K, DOF) joint angles
      velocities: np.ndarray       # (K, DOF) joint velocities
      accelerations: np.ndarray    # (K, DOF) joint accelerations
      duration: float              # Total trajectory execution time in seconds
      status: str                  # Status string or failure reason
  ```
* **`plan_arm_trajectory(target_pose_world, current_joints, ...)`**:
  * Inputs:
    * `target_pose_world`: `(3,)` translation and `(4,)` quaternion `[w, x, y, z]`.
    * `current_joints`: `(DOF,)` current robot joint values.
    * `obstacle_spheres` / `table_obstacle`: Collision geometry primitives in world frame.
  * Outputs: `TrajectoryPlan`.
* **Fallback Rank Search**:
  If candidate 1 trajectory generation fails due to collision constraints with the table slab or joint limits, candidate 2, 3, etc. are sequentially evaluated.

---

### `coverage.py`

Tracks surface point inspection progress and generates coverage-colored visualization meshes.

#### Interface Contract
* **`CoverageTracker`**:
  * Inputs: `target_points: (N, 3)`, `target_normals: (N, 3)`, `radius_m: float` (default `0.008`), `min_cosine: float` (default `0.707`).
  * Methods:
    * `update(reconstructed_pts: np.ndarray) -> int`: Queries `scipy.spatial.cKDTree` for nearest neighbors within `radius_m`. Verifies normal alignment. Marks points as covered.
    * `coverage_fraction -> float`: Ratio of covered points to total valid targets.
    * `get_unseen_points() -> (U, 3)`: Remaining uninspected surface coordinates.
* **`sample_surface_points_and_normals(mesh, n_samples, base_exclusion_z)`**:
  * Rejects triangle centers below `base_exclusion_z` to avoid counting tabletop contact surfaces as inspectable targets.
* **`build_coverage_colored_mesh(mesh, tracker)`**:
  * Outputs: `trimesh.Trimesh` with vertex colors: Green for inspected areas, Gray for uninspected areas.

---

### `viz.py`

Streaming 3D visualization using Rerun SDK (`rerun-sdk`).

#### Logged Entities
| Entity Path | Type | Description |
| :--- | :--- | :--- |
| `world/target_mesh` | `rr.Mesh3D` | Target CAD model geometry |
| `world/surface_points` | `rr.Points3D` | Surface points color-coded by coverage status |
| `world/candidates` | `rr.Arrows3D` / `Points3D` | Sampled candidate camera viewpoints and optical axes |
| `world/selected_view` | `rr.Arrows3D` | Highest scoring NBV candidate selected for execution |
| `world/trajectory` | `rr.LineStrips3D` | Planned robot end-effector Cartesian path |
| `world/reconstructed_pcd` | `rr.Points3D` | Accumulated live point cloud from backprojected depth frames |

---

### `config.py`

Central configuration repository for the planner:
* Sensor parameters: resolution `(640, 480)`, vertical FOV `58.0°`, near/far planes.
* Sampling parameters: azimuth steps `36`, radius steps `2`, elevation limits `[15°, 75°]`.
* Numerical margins: backface cosine `0.35`, occlusion ray offset `3 mm`, surface match distance `8 mm`.
* Kinematics links: base link `ur5_base_link`, camera link `camera_color_optical_frame`.
