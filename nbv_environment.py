import os
from typing import Callable

import numpy as np
import pybullet_data
from scipy.spatial.transform import Rotation
from shelf_gym.environments.ur5_environment import RobotEnv
from shelf_gym.utils.camera_utils import Camera

from nbv_core.camera_geometry import (
    CapturedFrame, capture_rgb_and_depth, compute_intrinsics, backproject_depth, edge_discontinuity_mask,
)
from nbv_core.voxel_grid import OccupancyVoxelGrid


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ASSET_PATH = os.path.join(PROJECT_ROOT, "third_party/shelf_gym_repo/shelf_gym/meshes/urdf")
YCB_OBJECT = "YcbMustardBottle"

TABLE_TOP_Z = 0.90  # matches robot base z; table half-height=0.45, center at z=0.45


class NBVEnv2(RobotEnv):
    # Attributes set somewhere in the RobotEnv/BasePybulletEnv chain (vendored
    # shelf_gym code, not ours to edit) that we rely on directly. Declared
    # here purely for static analysis/IDE tooling - they're otherwise set via
    # an AttrDict deep in the parent classes, which Pylance can't see through,
    # so without this `self.camera_link` etc. show up as untyped/unresolved.
    robot_id: int
    camera_link: int
    eef_id: int
    tool_tip_id: int
    client_id: int
    init_pos: list[float]

    # Attributes this class sets itself.
    plane_id: int
    table_id: int
    obj_id: int
    obj_pos: np.ndarray
    camera: Camera

    def __init__(self, render: bool = True) -> None:
        super().__init__(render=render, show_vis=False)
        self._build_scene()
        self._place_object()
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)
        self.camera = Camera(width=640, height=480)

    def _build_scene(self) -> None:
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = self._p.loadURDF("plane.urdf")

        # Thin tabletop slab — top at z=0.90, only 4 cm thick.
        # A solid block would fill z=0..0.90 and its side walls would be
        # directly in the camera's line of sight from the orbit viewpoints.
        # A thin slab lets the camera look over the edge and see the object.
        col = self._p.createCollisionShape(
            self._p.GEOM_BOX, halfExtents=[0.50, 0.65, 0.02]
        )
        vis = self._p.createVisualShape(
            self._p.GEOM_BOX, halfExtents=[0.50, 0.65, 0.02],
            rgbaColor=[0.65, 0.45, 0.25, 1]
        )
        self.table_id = self._p.createMultiBody(0, col, vis, [0.0, 0.15, 0.88])

        # Table legs (visual only)
        leg_he = [0.03, 0.03, 0.44]
        leg_positions = [
            [ 0.45,  0.75, 0.44],
            [-0.45,  0.75, 0.44],
            [ 0.45, -0.45, 0.44],
            [-0.45, -0.45, 0.44],
        ]
        for lp in leg_positions:
            lv = self._p.createVisualShape(
                self._p.GEOM_BOX, halfExtents=leg_he, rgbaColor=[0.50, 0.33, 0.18, 1]
            )
            self._p.createMultiBody(0, -1, lv, lp)

    def _place_object(self) -> None:
        # init_pos is the tool_tip world position at the robot's rest pose (set by super().__init__)
        obj_x = self.init_pos[0]
        obj_y = self.init_pos[1] + 0.10  # 10 cm in front of the gripper tip
        obj_z = TABLE_TOP_Z + 0.15       # drop from slightly above the table surface

        obj_path = os.path.join(ASSET_PATH, f"ycb_objects/{YCB_OBJECT}/model_textureless.urdf")
        self.obj_id = self._p.loadURDF(
            obj_path, [obj_x, obj_y, obj_z],
            self._p.getQuaternionFromEuler([0, 0, 0])
        )
        for _ in range(240):
            self._p.stepSimulation()

        pos, _ = self._p.getBasePositionAndOrientation(self.obj_id)
        self.obj_pos = np.array(pos)
        print(f"Gripper tip at  {np.round(self.init_pos, 3)}")
        print(f"Object settled at {np.round(self.obj_pos, 3)}")

    def capture_frame(self, min_depth_m: float = 0.15, max_depth_m: float | None = None) -> CapturedFrame:
        """
        Capture RGB-D from the wrist camera (nbv_core.camera_geometry does
        the actual rendering + depth-buffer conversion - see
        capture_rgb_and_depth) and back-project depth into our own
        world-frame point cloud. Fully our own code end to end: we never
        call shelf_gym's Camera.get_cam_in_hand()/get_pointcloud() here.

        Drops points closer than min_depth_m (default 15cm, e.g. gripper
        self-hits), points at/near the camera's far plane (background, no
        real surface hit), and pixels at depth discontinuities (silhouette
        edges where the rasterizer can emit an interpolated "flying pixel"
        depth that back-projects into a spurious streak - see
        nbv_core.camera_geometry.edge_discontinuity_mask).
        """
        rgb, transformed_depth, cam_pos, cam_quat, body_ids = capture_rgb_and_depth(
            self._p, self.robot_id, self.camera_link,
            self.camera.width, self.camera.height, self.camera.fov,
            self.camera.near, self.camera.far,
        )
        fx, fy, cx, cy = compute_intrinsics(self.camera.width, self.camera.height, self.camera.fov)
        points_world = backproject_depth(
            transformed_depth, cam_pos, cam_quat, fx, fy, cx, cy
        )

        if max_depth_m is None:
            max_depth_m = self.camera.far * 0.98
        depth_m = transformed_depth.astype(np.float64) / 1000.0
        range_valid = (depth_m > min_depth_m) & (depth_m < max_depth_m)
        edge_valid = edge_discontinuity_mask(depth_m)
        valid = range_valid & edge_valid

        return {
            'rgb': rgb,
            'transformed_depth': transformed_depth,
            'points_world': points_world[valid],       # (N, 3), gripper/background/edge dropped
            'points_world_grid': points_world,          # (H, W, 3), unfiltered, for debugging
            'valid_mask': valid,
            'range_valid_mask': range_valid,             # in [min_depth_m, max_depth_m)
            'edge_valid_mask': edge_valid,               # not a silhouette-edge "flying pixel"
            'body_ids': body_ids,                        # ground-truth PyBullet body ID per pixel
            'cam_pos': np.array(cam_pos),
            'cam_quat': np.array(cam_quat),
        }

    def _compute_safe_orbit_radius(self, safety_margin: float = 0.12) -> float:
        """
        Minimum radius so the gripper tip never touches the object:
          radius > obj_bounding_radius + gripper_extension + margin
        """
        aabb_min, aabb_max = self._p.getAABB(self.obj_id)
        obj_bounding_radius = np.linalg.norm(
            (np.array(aabb_max) - np.array(aabb_min)) / 2
        )
        eef_state = self._p.getLinkState(self.robot_id, self.eef_id)
        tip_state = self._p.getLinkState(self.robot_id, self.tool_tip_id)
        gripper_ext = np.linalg.norm(
            np.array(tip_state[0]) - np.array(eef_state[0])
        )
        radius = obj_bounding_radius + gripper_ext + safety_margin
        print(f"obj_radius={obj_bounding_radius:.3f}  gripper_ext={gripper_ext:.3f}  "
              f"orbit_radius={radius:.3f}")
        return radius

    def _lookat_quaternion(self, eye: np.ndarray, target: np.ndarray) -> list[float]:
        x_axis = target - eye
        x_axis /= np.linalg.norm(x_axis)
        world_up = np.array([0, 0, 1])
        if abs(np.dot(x_axis, world_up)) > 0.99:
            world_up = np.array([0, 1, 0])
        z_axis = np.cross(x_axis, world_up)
        z_axis /= np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis)
        R = np.column_stack([x_axis, y_axis, z_axis])
        return Rotation.from_matrix(R).as_quat().tolist()

    def _camera_lookat_quaternion(self, eye: np.ndarray, target: np.ndarray) -> list[float]:
        """
        Camera-frame lookat: local +Z = forward (toward target), local +Y =
        up - matches the axis convention camera_geometry.capture_rgb_and_depth
        actually renders with (forward_direction = R @ [0,0,1], up_direction
        = R @ [0,1,0]). Deliberately separate from _lookat_quaternion (local
        +X = forward), which orbit_and_capture keeps using paired with its
        eef_id IK target - the two conventions happen to roughly cancel out
        there, so don't mix this Z-forward one with an eef_id IK target or
        vice versa (see project_ik_motion_convergence_bug memory - swapping
        just one half without the other measured ~108deg off).
        """
        z_axis = target - eye
        z_axis = z_axis / np.linalg.norm(z_axis)
        world_up = np.array([0, 0, 1])
        if abs(np.dot(z_axis, world_up)) > 0.99:
            world_up = np.array([0, 1, 0])
        x_axis = np.cross(world_up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        R = np.column_stack([x_axis, y_axis, z_axis])
        return Rotation.from_matrix(R).as_quat().tolist()

    def _get_camera_ik_joints(self, position: list[float], orientation: list[float]) -> list[float]:
        """
        Solve IK for camera_link directly (not eef_id - see
        _camera_lookat_quaternion), seeded from self.initial_parameters
        instead of the arm's live joint state.

        calculateInverseKinematics is a local solver seeded from wherever
        the arm currently is. When move_through_orbit's absolute=False glide
        has drifted the arm into an awkward configuration, IK can converge
        to a different branch for the exact same Cartesian target - one that
        can violate a real joint limit (e.g. shoulder_lift_joint's +-pi
        range) and then physically stall there with no error surfaced to the
        caller. Seeding from a fixed reference config every time makes the
        chosen IK branch deterministic and independent of the arrival path,
        while the arm itself still moves smoothly from wherever it actually
        is (this only changes what config IK is *computed* from, not what
        the robot physically does).
        """
        n_joints = self._p.getNumJoints(self.robot_id)
        live_joint_pos = [self._p.getJointState(self.robot_id, j)[0] for j in range(n_joints)]

        for i, idx in enumerate(self.arm_joint_indices):
            self._p.resetJointState(self.robot_id, idx, self.initial_parameters[i])

        joints = self._p.calculateInverseKinematics(
            self.robot_id, self.camera_link, position, orientation,
            solver=self._p.IK_DLS, maxNumIterations=1000, residualThreshold=1e-5,
            physicsClientId=self.client_id,
        )

        for j in range(n_joints):
            self._p.resetJointState(self.robot_id, j, live_joint_pos[j])

        return list(joints)[:6]

    def _wait_for_arm_at_rest(self, max_extra_steps: int = 50, velocity_threshold: float = 0.005) -> None:
        """
        shelf_gym's execute_joint_states/simulate_until_motion_done only checks
        joint POSITION against a coarse 1e-2 rad tolerance, not velocity - so a
        capture can fire while the arm still has residual velocity carried over
        from the glide leading up to it. That residual grows over a long glide
        and resets right after a fresh direct jump, producing a per-view
        systematic camera-settle bias (confirmed via signed point-to-mesh
        distance - see project_pointcloud_layering_bug memory). Drains that
        residual by stepping a few more times until every arm joint's velocity
        drops below threshold, or a step budget runs out (near-singular/
        boundary poses may never fully zero out).
        """
        for _ in range(max_extra_steps):
            velocities = [self._p.getJointState(self.robot_id, idx)[1] for idx in self.arm_joint_indices]
            if max(abs(v) for v in velocities) < velocity_threshold:
                break
            self.step_simulation(self.per_step_iterations)

    def _snap_to_joint_targets(self, joint_targets: list[float]) -> None:
        """
        Teleports the arm's joints to the exact IK-computed targets via
        resetJointState, eliminating the position controller's steady-state
        droop for the capture instant. Confirmed by direct experiment this
        droop is NOT a convergence-time issue: even 300 extra physics steps
        with a target tolerance 100x tighter than shelf_gym's default
        (1e-4 rad vs. simulate_until_motion_done's 1e-2 rad) left the same
        per-view point-to-mesh bias essentially unchanged (view 6: 3.80mm vs
        3.77mm) - proportional-only position control (positionGain=0.005 in
        the vendored control_robot) has a genuine nonzero equilibrium error
        fighting gravity/dynamics that waiting longer can never close.

        This only affects the pose used for the upcoming capture (rendering
        + backprojection both read joint state fresh via getLinkState) - it
        does not change how the arm got there or how it moves next; the
        real physics-based glide/settle still happens first, this just
        removes the last few mm of unavoidable controller offset right
        before the sensor reads the scene.
        """
        for idx, target in zip(self.arm_joint_indices, joint_targets):
            self._p.resetJointState(self.robot_id, idx, target)

    def move_through_orbit(
        self,
        n_views: int = 8,
        height: float = 1.2,
        on_stop: Callable[[int], None] | None = None,
        max_pose_error_m: float = 0.015,
    ) -> None:
        """
        Drive the arm through the same fixed half-orbit around the object
        used by orbit_and_capture (theta from pi to 2pi: left -> in front ->
        right, staying on the robot-facing side). This method only moves the
        arm - it doesn't capture or save anything itself. At each of the
        n_views stops, it calls on_stop(view_index) so the caller decides
        what to do there (save an image, capture depth, fuse into a voxel
        grid, ...). This is the shared movement logic behind
        scan_fixed_views_and_integrate() and scan_and_save.py, factored out
        so a third caller doesn't need its own copy of it.

        Near the two ends of this fixed orbit (roughly theta=170-190deg and
        265-280deg at the default height/radius) the requested camera pose
        can be beyond the arm's reach - IK settles into a fully-extended,
        near-singular configuration (elbow_joint pinned at 0) and simply
        can't get closer, regardless of iterations or seeding (see
        project_ik_motion_convergence_bug memory). Rather than silently
        saving/fusing a badly-aimed capture there, each stop's actual camera
        pose is checked against the intended viewpoint after settling; if
        the residual exceeds max_pose_error_m, on_stop is skipped for that
        view. This is a small, static preview of the reachability filter
        nbv_planner.py will eventually apply to dynamically generated
        candidates instead of a fixed orbit.
        """
        radius = self._compute_safe_orbit_radius()
        angles = np.linspace(np.pi, 2 * np.pi, n_views, endpoint=True)

        def move_to(theta: float, absolute: bool = True) -> tuple[np.ndarray, list[float]]:
            viewpoint = np.array([
                self.obj_pos[0] + radius * np.cos(theta),
                self.obj_pos[1] + radius * np.sin(theta),
                height
            ])
            orientation = self._camera_lookat_quaternion(viewpoint, self.obj_pos)
            joint_targets = self._get_camera_ik_joints(viewpoint.tolist(), orientation)
            self.execute_joint_states(joint_targets, absolute=absolute)
            return viewpoint, joint_targets

        # theta=3pi/2 is directly in front of the object - closest to the arm's
        # natural forward pose, so the first IK call lands in the right config.
        MID = 3 * np.pi / 2
        move_to(MID, absolute=True)

        # Glide from MID to the start of the arc with fine steps so the arm
        # flows continuously instead of stalling between big jumps.
        for t in np.linspace(MID, angles[0], num=16)[1:]:
            move_to(t, absolute=False)

        for view_index, theta in enumerate(angles):
            if view_index > 0:
                # Fine arc steps between consecutive viewpoints keep the IK
                # in the same configuration branch without needing a seed.
                for t in np.linspace(angles[view_index - 1], theta, num=10)[1:-1]:
                    move_to(t, absolute=False)
            viewpoint, joint_targets = move_to(theta, absolute=True)
            self._wait_for_arm_at_rest()
            self._snap_to_joint_targets(joint_targets)

            cam_pos = np.array(self._p.getLinkState(self.robot_id, self.camera_link)[0])
            pose_error_m = float(np.linalg.norm(cam_pos - viewpoint))
            if pose_error_m > max_pose_error_m:
                print(f"View {view_index} (theta={np.degrees(theta):.1f}deg): skipped - "
                      f"camera settled {pose_error_m * 1000:.1f}mm from the intended "
                      f"viewpoint (> {max_pose_error_m * 1000:.0f}mm tolerance), likely "
                      f"beyond the arm's reach here")
                continue

            if on_stop is not None:
                on_stop(view_index)

    def orbit_and_capture(self, n_views: int = 8, height: float = 1.2, save_dir: str = "captures2") -> None:
        from PIL import Image
        os.makedirs(save_dir, exist_ok=True)

        radius = self._compute_safe_orbit_radius()

        # Half-orbit on the robot-facing side: theta π → 2π
        # (left → in-front → right), keeping all viewpoints between the robot and object.
        angles = np.linspace(np.pi, 2 * np.pi, n_views, endpoint=True)

        # Mark all viewpoints as red spheres so you can see the orbit layout
        sphere_vis = self._p.createVisualShape(
            self._p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 1]
        )
        for theta in angles:
            vp = [
                self.obj_pos[0] + radius * np.cos(theta),
                self.obj_pos[1] + radius * np.sin(theta),
                height
            ]
            self._p.createMultiBody(0, -1, sphere_vis, vp)

        def _move_to(theta, absolute=True):
            vp = np.array([
                self.obj_pos[0] + radius * np.cos(theta),
                self.obj_pos[1] + radius * np.sin(theta),
                height
            ])
            ori = self._lookat_quaternion(vp, self.obj_pos)
            joints = self.get_ik_joints(vp.tolist(), ori)
            self.execute_joint_states(joints, absolute=absolute)
            return vp

        # theta=3pi/2 is directly in front of the object — closest to the arm's
        # natural forward pose, so the first IK call lands in the right config.
        MID = 3 * np.pi / 2
        _move_to(MID, absolute=True)

        # Glide from MID to the start of the arc (theta=pi, left side) with fine
        # steps. absolute=False lets the arm flow continuously without stalling.
        for t in np.linspace(MID, angles[0], num=16)[1:]:
            _move_to(t, absolute=False)

        for i, theta in enumerate(angles):
            if i > 0:
                # Fine arc steps between consecutive viewpoints — small increments
                # keep the IK in the same configuration branch without seeding.
                for t in np.linspace(angles[i - 1], theta, num=10)[1:-1]:
                    _move_to(t, absolute=False)
            vp = _move_to(theta, absolute=True)

            result = self.camera.get_cam_in_hand(
                self.robot_id, self.camera_link,
                remove_gripper=False,
                client_id=self.client_id,
                no_conversion=True
            )
            rgb = result['rgb']
            img = Image.fromarray(rgb.astype(np.uint8))
            path = os.path.join(save_dir, f"view_{i:02d}.png")
            img.save(path)
            print(f"View {i}: pos={np.round(vp, 2)} -> saved {path}")

        print("Done.")

    def scan_fixed_views_and_integrate(
        self,
        n_views: int = 8,
        height: float = 1.2,
        voxel_size: float = 0.005,
        margin: float = 0.15,
        save_dir: str = "captures/voxel_fusion_test",
        on_view: Callable[[int, CapturedFrame, OccupancyVoxelGrid], None] | None = None,
    ) -> OccupancyVoxelGrid:
        """
        Milestone 2 test rig: drives the arm through the same fixed half-orbit
        as orbit_and_capture (via move_through_orbit) - this milestone is only
        about validating multi-view voxel fusion, not planning - but captures
        depth + integrates into an OccupancyVoxelGrid at each stop instead of
        just saving RGB. Exports the occupied-voxel cloud after every view so
        fusion correctness can be inspected progressively.

        on_view(i, frame, grid), if given, is called after each view is
        captured and integrated - used by scripts/gui_scan_debug.py to
        overlay live debug visualization without duplicating this loop.
        """
        os.makedirs(save_dir, exist_ok=True)

        aabb_min, aabb_max = self._p.getAABB(self.obj_id)
        grid = OccupancyVoxelGrid.from_aabb(aabb_min, aabb_max, margin=margin, voxel_size=voxel_size)
        print(f"Voxel grid: size={grid.size} voxel_size={voxel_size} "
              f"total_cells={np.prod(grid.size)}")

        def capture_and_integrate(view_index: int) -> None:
            frame = self.capture_frame()
            grid.integrate_points(frame['points_world'], frame['cam_pos'], max_range=self.camera.far)

            occupied_pts = grid.get_occupied_points()
            np.save(os.path.join(save_dir, f"occupied_after_view_{view_index:02d}.npy"), occupied_pts)
            print(f"View {view_index}: "
                  f"occupied={grid.get_occupied_count()} unknown={grid.get_unknown_count()}")

            if on_view is not None:
                on_view(view_index, frame, grid)

        self.move_through_orbit(n_views=n_views, height=height, on_stop=capture_and_integrate)

        print("Done.")
        return grid


if __name__ == "__main__":
    import time
    env = NBVEnv2(render=True)
    time.sleep(2)
    env.orbit_and_capture(n_views=8, height=1.2, save_dir="captures")
    while True:
        env.step_simulation(env.per_step_iterations)
