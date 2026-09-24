"""Autonomous next best view inspection, driven through one robot interface.

The robot is PyBullet in development and a ROS 2 adapter on the real cell; the stages below are the same.
"""

import os
import time

import numpy as np
import open3d as o3d
import torch

from nbv_planner.config import (
    ALIGN_MAX_CENTER_ERROR_M,
    ALIGN_MAX_POINT_DISTANCE_M,
    BASE_EXCLUSION_HEIGHT_M,
    MAX_POSE_ERROR_M,
    N_AZIMUTH,
    N_RADIUS,
    ORBIT_DEPTH_FRACTION,
    ROBOT_SELF_FILTER_MIN_DEPTH_M,
    START_CAMERA_POSITION_BASE,
    START_LOOK_AT_BASE,
    START_FALLBACK_AZIMUTH_STEPS,
    START_FALLBACK_ELEVATION_STEPS,
    START_FALLBACK_RADIUS_STEPS_M,
    START_PLAN_ATTEMPTS,
    START_ROTATION_TOL_RAD,
    START_SAFETY_RADIUS_M,
    TABLE_CLEARANCE_MARGIN_M,
    WORKSPACE_RADIUS_M,
)
from nbv_planner.camera import backproject_depth, transform_points
from nbv_planner.coverage import CoverageTracker, build_coverage_colored_mesh, sample_surface_points_and_normals
from nbv_planner.motion_planning import (build_world_config, move_camera, plan_motion_batch, robot_spheres_world,
                                         start_state_problem)
from nbv_planner.object_estimate import mesh_alignment
from nbv_planner.object_scan import box_world_config, scan_object
from nbv_planner.ray_scoring import score_candidate_views
from nbv_planner.reachability import ik_filter, sample_candidate_camera_poses
from nbv_planner.start_pose import pose_errors, start_camera_pose_world, start_pose_candidates
from nbv_planner.viz import NBVVisualizer


def _capture_object_cloud(robot, object_center: np.ndarray):
    """Capture RGB-D at the measured camera pose and return filtered world-frame object points + the observation."""
    observation = robot.capture_observation()
    pts_cam, _ = backproject_depth(
        observation.depth_m,
        robot.intrinsics,
        rgb=observation.rgb,
        min_depth=ROBOT_SELF_FILTER_MIN_DEPTH_M,
        max_depth=robot.intrinsics.far * 0.9,
        drop_edges=True,
    )
    if pts_cam.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), observation

    pts_world = transform_points(pts_cam, observation.world_from_camera)

    # Real-world tabletop filter: drop points below table surface, keep workspace XY radius (no height limit)
    is_above_table = pts_world[:, 2] >= (robot.table_surface_z + TABLE_CLEARANCE_MARGIN_M)
    in_workspace_xy = np.linalg.norm(pts_world[:, :2] - object_center[:2], axis=-1) < WORKSPACE_RADIUS_M

    return pts_world[is_above_table & in_workspace_xy], observation


def _reach_start_pose(
    robot,
    start_position_base: np.ndarray,
    look_at_base: np.ndarray,
    visualizer: NBVVisualizer,
):
    """Drive the camera to the configured start pose, or the nearest neighbour that plans.

    Returns (camera position world, camera quaternion world, look-at point world, observation at the start pose).
    """
    t_base, q_base = robot.base_pose()
    lo, hi = robot.table_aabb
    _, _, look_at_world = start_camera_pose_world(start_position_base, look_at_base, t_base, q_base)

    # The object is not known yet: keep a conservative box around the look-at point clear of the arm.
    world_cfg = build_world_config(
        t_base_world=t_base,
        q_base_world_xyzw=q_base,
        table_lo=lo,
        table_hi=hi,
        t_obj_world=look_at_world,
        q_obj_world_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        obj_dims=np.full(3, 2.0 * START_SAFETY_RADIUS_M),
    )

    problem = start_state_problem(robot.current_arm_joints(), robot.arm_joint_names, world_cfg)
    if problem is not None:
        raise RuntimeError(
            f"cuRobo will not plan from the arm's current pose: {problem}. The arm is inside the "
            f"table box or the keep-clear box around the look-at point, or in self collision. "
            f"Move it clear first (current joints, rad: "
            f"{np.round(robot.current_arm_joints(), 3).tolist()}).")

    candidates = start_pose_candidates(
        start_position_base,
        look_at_base,
        START_FALLBACK_RADIUS_STEPS_M,
        START_FALLBACK_ELEVATION_STEPS,
        START_FALLBACK_AZIMUTH_STEPS,
    )
    reason = "no candidate was tried"

    for candidate_idx, position_base in enumerate(candidates):
        t_start, q_start, _ = start_camera_pose_world(position_base, look_at_base, t_base, q_base)
        # Trajectory optimisation is seeded randomly, so the configured pose is worth retrying.
        # Fallbacks get one shot each: there are dozens of them, and re-seeding every one costs
        # far more than simply moving on to the next.
        for attempt in range(START_PLAN_ATTEMPTS if candidate_idx == 0 else 1):
            ok, _, trajectory, opt_ms = plan_motion_batch(
                t_targets_world=t_start[None, :],
                q_targets_world=q_start[None, :],
                current_joints=robot.current_arm_joints(),
                t_base_world=t_base,
                q_base_world_xyzw=q_base,
                world_config=world_cfg,
                arm_joint_names=robot.arm_joint_names,
                enable_graph=attempt > 0,
            )
            if ok and trajectory is not None:
                t_exec_0 = time.perf_counter()
                robot.execute_trajectory(trajectory, visualizer=visualizer)
                exec_ms = (time.perf_counter() - t_exec_0) * 1000.0
                pos_err, rot_err = pose_errors(robot.camera_world_transform(), t_start, q_start)
                if pos_err <= MAX_POSE_ERROR_M and rot_err <= START_ROTATION_TOL_RAD:
                    break
                reason = f"arm stopped {pos_err * 1000:.1f} mm / {np.degrees(rot_err):.1f} deg away"
            else:
                reason = "no collision-free path"
            print(
                f"      Start pose {candidate_idx + 1}/{len(candidates)} "
                f"{np.round(position_base, 3).tolist()} attempt {attempt + 1}/{START_PLAN_ATTEMPTS}: "
                f"{reason}, retrying...",
                flush=True,
            )
            # No recovery motion between attempts. Driving to home meant a straight line in joint
            # space with nothing checking it, which is how the arm ended up inside the table and
            # every later goal came back PATH_TOLERANCE_VIOLATED. The next candidate plans from
            # wherever the arm actually is.
        else:
            continue
        break
    else:
        raise RuntimeError(
            f"Could not reach the start pose {np.round(start_position_base, 3).tolist()} looking at "
            f"{np.round(look_at_base, 3).tolist()} (base frame), nor any of {len(candidates) - 1} "
            f"nearby poses: {reason}. Adjust --start-pos / --look-at."
        )

    observation = robot.capture_observation()
    distance = np.linalg.norm(look_at_world - t_start)
    offset = t_start - look_at_world
    elevation = np.degrees(np.arctan2(offset[2], np.linalg.norm(offset[:2])))
    moved = "" if candidate_idx == 0 else f" (fell back to candidate {candidate_idx + 1}/{len(candidates)})"
    print(
        f"      Reached start pose{moved}: camera at ({t_start[0]:.3f}, {t_start[1]:.3f}, {t_start[2]:.3f}) facing "
        f"({look_at_world[0]:.3f}, {look_at_world[1]:.3f}, {look_at_world[2]:.3f}) | {distance:.2f} m, "
        f"{elevation:.0f} deg elevation | error {pos_err * 1000:.1f} mm / {np.degrees(rot_err):.1f} deg | "
        f"plan {opt_ms:.0f} ms, drive {exec_ms:.0f} ms"
    )
    return t_start, q_start, look_at_world, observation


def _discover_object(robot, segmenter_name: str, scan_views: int, start_position: np.ndarray,
                     start_quaternion: np.ndarray, visualizer: NBVVisualizer):
    """Segment the object from the start pose, refine its box and points over a ring of views, and return to start."""
    visualizer.update_setup_stage("Object Discovery", "RUNNING", "segmenting start frame...")
    if segmenter_name == "sam":
        from nbv_planner.segmentation import Sam2Segmenter
        segmenter = Sam2Segmenter()
    else:
        segmenter = None

    def on_frame(label, observation, detection, estimate):
        box = estimate.box
        shift = "-" if np.isinf(estimate.center_shift) else f"{estimate.center_shift * 1000:.1f} mm"
        print(
            f"      {label:<7}: box center ({box.center[0]:.3f}, {box.center[1]:.3f}, {box.center[2]:.3f}) | "
            f"size {np.round(box.size * 1000).astype(int).tolist()} mm | shift {shift} | {len(estimate.points):,} pts"
        )
        visualizer.log_discovery_frame(label, observation, detection, estimate)
        visualizer.update_setup_stage("Object Discovery", "RUNNING", f"{label}: shift {shift}")

    estimate = scan_object(
        robot,
        segmenter,
        move=lambda position, quaternion, world: move_camera(robot, position, quaternion, world, visualizer),
        n_views=scan_views,
        on_frame=on_frame,
        on_views=visualizer.log_ring_views,
        log=lambda message: print(f"      {message}"),
    )
    del segmenter
    torch.cuda.empty_cache()

    if move_camera(robot, start_position, start_quaternion, box_world_config(robot, estimate.box), visualizer):
        print("      Returned to start pose")
    else:
        print("      Warning: could not return to the start pose")
    visualizer.update_setup_stage(
        "Object Discovery", "DONE",
        f"{len(estimate.boxes)} frames, box {np.round(estimate.box.size * 1000).astype(int).tolist()} mm",
    )
    return estimate


def _orbit_shell(robot, mesh_world) -> tuple[float, float]:
    """Candidate view radii: close enough to frame the object, bounded by what the arm can reach."""
    size = mesh_world.bounds[1] - mesh_world.bounds[0]
    center = mesh_world.bounds.mean(axis=0)
    framing = float(size.max()) / (2.0 * np.tan(np.radians(robot.intrinsics.fov) / 2.0))
    r_min = max(float(np.linalg.norm(size)) / 2.0 + robot.intrinsics.near, framing)
    arm_distance = float(np.linalg.norm(center[:2] - robot.arm_base_pos()[:2]))
    slack = max(0.0, robot.max_reach() - arm_distance - r_min)
    return r_min, r_min + slack * ORBIT_DEPTH_FRACTION


def run_inspection(
    robot,
    visualizer,
    object_name: str,
    mode: str = "cad",
    max_views: int = 8,
    target_coverage: float = 0.95,
    n_surface_samples: int = 4000,
    start_position_base: np.ndarray = START_CAMERA_POSITION_BASE,
    look_at_base: np.ndarray = START_LOOK_AT_BASE,
    segmenter_name: str = "sam",
    scan_views: int = 8,
    capture_dir: str = "captures",
) -> dict:
    """Run NBV against the CAD mesh ("cad"), a mesh built from segmented views ("scan"),
    or the CAD after checking it against the scan ("both")."""
    if mode not in ("cad", "scan", "both"):
        raise ValueError(f"Unknown mode: {mode}")
    print(f"=== Autonomous NBV Scan: {object_name} ({mode} mode) ===")
    total_stages = 5 if mode == "cad" else 6
    stage_numbers = iter(range(1, total_stages + 1))

    def announce(text: str) -> None:
        print(f"[{next(stage_numbers)}/{total_stages}] {text}")

    visualizer.init_scene(robot)

    announce("Moving arm to start pose with cuRobo...")
    visualizer.update_setup_stage("Start Pose", "RUNNING", "cuRobo planning...")
    start_position, start_quaternion, look_at_world, start_observation = _reach_start_pose(
    robot, start_position_base, look_at_base, visualizer
)
    visualizer.log_start_pose(start_observation, look_at_world)
    visualizer.update_setup_stage("Start Pose", "DONE", "reached ({:.2f}, {:.2f}, {:.2f})".format(*start_position))

    estimate, alignment = None, None
    if mode == "cad":
        visualizer.update_setup_stage("Object Discovery", "DONE", "not used in CAD mode")
    else:
        announce(f"Discovering object ({segmenter_name} segmentation, {scan_views} ring views)...")
        estimate = _discover_object(robot, segmenter_name, scan_views, start_position, start_quaternion, visualizer)

    if mode == "cad":
        announce(f"Loading CAD mesh & sampling target surface: {object_name}...")
        mesh_world = robot.object_mesh_world()
    elif mode == "both":
        announce(f"Loading CAD mesh, checking it against the scan & sampling target surface: {object_name}...")
        mesh_world = robot.object_mesh_world()
        alignment = mesh_alignment(estimate, mesh_world)
        aligned = alignment.aligned()
        visualizer.log_cad_alignment(mesh_world, aligned)
        print(
            f"      Scan vs CAD: center error {alignment.center_error * 1000:.1f} mm "
            f"(limit {ALIGN_MAX_CENTER_ERROR_M * 1000:.0f}) | scanned points to CAD surface "
            f"mean {alignment.point_distance_mean * 1000:.1f} mm, p95 {alignment.point_distance_p95 * 1000:.1f} mm "
            f"(limit {ALIGN_MAX_POINT_DISTANCE_M * 1000:.0f}) -> {'aligned' if aligned else 'NOT aligned'}"
        )
        if not aligned:
            raise RuntimeError("CAD mesh does not align with the scanned object; refusing to plan NBV against it")
    else:
        announce("Building object mesh from accumulated points & sampling target surface...")
        mesh_world = estimate.surface_mesh()
        print(
            f"      Object mesh: {len(mesh_world.vertices):,} vertices, {len(mesh_world.faces):,} faces, "
            f"extent {np.round((mesh_world.bounds[1] - mesh_world.bounds[0]) * 1000).astype(int).tolist()} mm"
        )

    triangles_world = np.asarray(mesh_world.vertices[mesh_world.faces], dtype=np.float32)
    object_center = np.array((mesh_world.bounds[0] + mesh_world.bounds[1]) / 2.0, dtype=np.float64)
    base_exclusion_z = float(mesh_world.vertices[:, 2].min()) + BASE_EXCLUSION_HEIGHT_M
    surface_pts, surface_nrm = sample_surface_points_and_normals(
        mesh_world, n_samples=n_surface_samples, base_exclusion_z=base_exclusion_z
    )
    tracker = CoverageTracker(surface_pts, surface_nrm)

    visualizer.log_object_mesh(mesh_world)
    visualizer.update_setup_stage("Scene & Target Surface", "DONE", f"{len(surface_pts):,} samples")
    print(f"      Target surface: {len(surface_pts):,} samples")

    announce("Sampling orbit viewpoints & checking cuRobo reachability...")
    r_min, r_max = _orbit_shell(robot, mesh_world)
    t_cand, q_cand = sample_candidate_camera_poses(
        object_center,
        radius=(r_min, r_max, N_RADIUS),
        n_azimuth=N_AZIMUTH,
        z_min_world=robot.table_surface_z + 0.02,
    )
    t_base, q_base = robot.base_pose()
    visualizer.update_setup_stage("Orbit Viewpoints", "DONE", f"{len(t_cand)} generated")
    visualizer.update_setup_stage("cuRobo IK Reachability", "RUNNING", "checking kinematics...")

    t0_ik = time.perf_counter()
    reachable, ik_joints = ik_filter(
        robot.urdf_path, robot.base_link, robot.ee_link, t_cand, q_cand, t_base, q_base
    )
    ik_ms = (time.perf_counter() - t0_ik) * 1000.0
    reach_idx = np.where(reachable)[0]
    print(f"      Candidates: {len(t_cand)} generated -> {len(reach_idx)} reachable by UR5 ({ik_ms:.0f} ms)")
    visualizer.update_setup_stage("cuRobo IK Reachability", "DONE", f"{len(reach_idx)} reachable ({ik_ms:.0f} ms)")

    if len(reach_idx) == 0:
        print("Error: No reachable candidate viewpoints found.")
        return {"coverage": 0.0, "views": 0, "points": 0}

    # Scoring must know the arm holding the camera can block its own view.
    candidate_spheres = robot_spheres_world(ik_joints[reach_idx], t_base, q_base)
    sphere_row = np.full(len(t_cand), -1)
    sphere_row[reach_idx] = np.arange(len(reach_idx))

    announce(f"Running NBV loop (max {max_views} views, target {target_coverage * 100:.0f}%):")
    accumulated_clouds = []
    visited = np.zeros(len(t_cand), dtype=bool)
    views_executed = 0

    # Construct cuRobo collision world: table slab + axis-aligned box around the CAD mesh
    lo, hi = robot.table_aabb
    world_cfg = build_world_config(
        t_base_world=t_base,
        q_base_world_xyzw=q_base,
        table_lo=lo,
        table_hi=hi,
        t_obj_world=object_center,
        q_obj_world_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        obj_dims=mesh_world.bounds[1] - mesh_world.bounds[0],
    )

    while views_executed < max_views:
        cov_current = tracker.coverage_fraction()
        if cov_current >= target_coverage:
            print(f"      Target coverage {target_coverage * 100:.1f}% reached.")
            break

        unvisited_reach = [i for i in reach_idx if not visited[i]]
        if not unvisited_reach:
            print("      All reachable viewpoints visited.")
            break

        unseen_pts = tracker.get_unseen_points()
        unseen_nrm = tracker.get_unseen_normals()
        if len(unseen_pts) == 0:
            print("      100% surface coverage confirmed.")
            break

        # Parallel GPU ray scoring
        cand_positions = t_cand[unvisited_reach]
        visualizer.start_view(view_idx=views_executed + 1, max_views=max_views)

        t0 = time.perf_counter()
        scores, _ = score_candidate_views(
            cand_positions, unseen_pts, unseen_nrm, triangles_world,
        )
        score_ms = (time.perf_counter() - t0) * 1000.0
        visualizer.update_view_stage("CUDA Ray Scoring", "DONE", score_ms)
        visualizer.update_view_stage("cuRobo Batch Opt", "RUNNING")

        # Take top K non-zero gain candidates for batched planning (cap at 4 for VRAM efficiency)
        sorted_local = np.argsort(scores)[::-1]
        valid_sorted = [l for l in sorted_local if scores[l] > 0][:4]
        if not valid_sorted:
            print("      No remaining candidates with positive gain.")
            break

        top_cand_indices = [unvisited_reach[l] for l in valid_sorted]
        target_t = t_cand[top_cand_indices]
        target_q = q_cand[top_cand_indices]

        # Plan collision-free trajectory using cuRobo on GPU
        ok, best_k, trajectory, opt_ms = plan_motion_batch(
            t_targets_world=target_t,
            q_targets_world=target_q,
            current_joints=robot.current_arm_joints(),
            t_base_world=t_base,
            q_base_world_xyzw=q_base,
            world_config=world_cfg,
            arm_joint_names=robot.arm_joint_names,
            enable_graph=False,
        )
        visualizer.update_view_stage("cuRobo Batch Opt", "DONE", opt_ms)
        visualizer.update_view_stage("Arm Waypoint Drive (120Hz)", "RUNNING")

        if ok and trajectory is not None:
            t_achieved, exec_ms = robot.execute_trajectory(trajectory, visualizer=visualizer)
            err = float(np.linalg.norm(t_achieved - target_t[best_k]))
            reached = err <= MAX_POSE_ERROR_M
        else:
            exec_ms = 0.0
            reached = False

        visualizer.update_view_stage("Arm Waypoint Drive (120Hz)", "DONE", exec_ms)
        visualizer.update_view_stage("RGB-D Camera Capture", "RUNNING")

        # Mark evaluated candidates as visited
        num_to_mark = best_k + 1 if reached else len(top_cand_indices)
        for c_idx in top_cand_indices[:num_to_mark]:
            visited[c_idx] = True

        if not reached:
            print("      Top batch candidates blocked by table collision, trying next candidates...")
            continue

        best_cand_idx = top_cand_indices[best_k]
        cand_score = int(scores[valid_sorted[best_k]])

        t_cap_0 = time.perf_counter()
        cloud, observation = _capture_object_cloud(robot, object_center)
        cap_ms = (time.perf_counter() - t_cap_0) * 1000.0
        visualizer.update_view_stage("RGB-D Camera Capture", "DONE", cap_ms)
        visualizer.update_view_stage("KDTree Coverage Match", "RUNNING")

        if len(cloud) == 0:
            continue

        t_cov_0 = time.perf_counter()
        newly_seen = tracker.update(cloud)
        cov_ms = (time.perf_counter() - t_cov_0) * 1000.0
        visualizer.update_view_stage("KDTree Coverage Match", "DONE", cov_ms)

        accumulated_clouds.append(cloud)
        views_executed += 1
        cov_now = tracker.coverage_fraction()

        visualizer.complete_view(
            cand_idx=best_cand_idx,
            gain=cand_score,
            newly_seen=newly_seen,
            cov_pct=cov_now * 100.0,
        )

        # Live streaming to Rerun (3D views, cameras, point cloud)
        visualizer.log_step(
            step_idx=views_executed,
            cand_idx=best_cand_idx,
            gain=cand_score,
            newly_seen=newly_seen,
            score_ms=score_ms,
            view_matrix=observation.view_matrix,
            intrinsics=robot.intrinsics,
            tracker=tracker,
            new_cloud=cloud,
        )

        print(
            f"  [View {views_executed}/{max_views}] Cand #{best_cand_idx:03d} "
            f"(gain={cand_score:4d}) -> +{newly_seen:4d} seen | "
            f"Coverage: {cov_now * 100:4.1f}% | Scored in {score_ms:4.0f}ms"
        )

        # Flush GPU cache between views to maintain clean VRAM headroom
        torch.cuda.empty_cache()

    os.makedirs(capture_dir, exist_ok=True)
    final_cov = tracker.coverage_fraction()
    total_pts = sum(len(c) for c in accumulated_clouds)

    if accumulated_clouds:
        full_cloud = np.concatenate(accumulated_clouds, axis=0)
        cloud_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(full_cloud.astype(np.float64)))
        o3d.io.write_point_cloud(os.path.join(capture_dir, f"scan_{object_name}.ply"), cloud_o3d)

    saved = [f"{capture_dir}/scan_{object_name}.ply", f"{capture_dir}/coverage_{object_name}.ply"]
    if estimate is not None:
        discovery_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(estimate.points))
        o3d.io.write_point_cloud(os.path.join(capture_dir, f"discovery_{object_name}.ply"), discovery_cloud)
        saved.append(f"{capture_dir}/discovery_{object_name}.ply")
    if mode == "scan":
        mesh_world.export(os.path.join(capture_dir, f"object_mesh_{object_name}.ply"))
        saved.append(f"{capture_dir}/object_mesh_{object_name}.ply")

    cov_mesh_o3d = build_coverage_colored_mesh(mesh_world, tracker, base_exclusion_z=base_exclusion_z)
    cov_mesh_path = os.path.join(capture_dir, f"coverage_{object_name}.ply")
    o3d.io.write_triangle_mesh(cov_mesh_path, cov_mesh_o3d)

    announce(f"Complete: {views_executed} views executed | Reconstructed {total_pts:,} pts | Final Coverage: {final_cov * 100:.1f}%")
    print(f"      Saved: {', '.join(saved)}\n")

    return {
        "coverage": final_cov,
        "views": views_executed,
        "points": total_pts,
        "start_pose": start_position.tolist(),
        "mode": mode,
        **({} if alignment is None else {"scan_cad_center_error_mm": alignment.center_error * 1000,
                                          "scan_cad_point_distance_p95_mm": alignment.point_distance_p95 * 1000}),
    }
