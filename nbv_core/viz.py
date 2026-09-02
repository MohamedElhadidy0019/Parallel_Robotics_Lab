"""Live Rerun multi-tab visualizer with full 3D robot arm, camera FOV, and visual metrics HUD."""

import os
import time
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from nbv_core.camera import CameraIntrinsics
from nbv_core.config import T_OPENGL_OPTICAL
from nbv_core.coverage import CoverageTracker


class NBVVisualizer:
    """Manages 4-tab interactive Rerun visualization including 3D robot arm motion."""

    def __init__(self, obj_name: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self.obj_name = obj_name
        self.executed_cams: list[np.ndarray] = []
        self.accumulated_clouds: list[np.ndarray] = []
        self.metrics_history: list[dict] = []
        self.robot_links_meta: list[tuple[int, tuple, tuple]] = []

        if self.enabled:
            # Single-frame multi-pane layout:
            # Left: Large 3D World (top) + Live Stage Benchmark (bottom)
            # Right: Reconstruction (top), Coverage (mid), Status HUD (bottom)
            blueprint = rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Vertical(
                        rrb.Spatial3DView(name="Live 3D Environment (World & Robot)", contents=["+ /**"]),
                        rrb.TextDocumentView(name="Live Stage Benchmark (CPU / GPU)", contents=["+ benchmark/**", "+ benchmark"]),
                        row_shares=[3.5, 1.0],
                    ),
                    rrb.Vertical(
                        rrb.Spatial3DView(name="Reconstruction", contents=["+ scene/**", "+ reconstruction/**"]),
                        rrb.Spatial3DView(name="Coverage", contents=["+ scene/**", "+ coverage/**"]),
                        rrb.TextDocumentView(name="Status HUD", contents=["+ metrics/hud/**"]),
                        row_shares=[1.0, 1.0, 0.7],
                    ),
                    column_shares=[1.65, 1.0],
                ),
            )
            rr.init(f"nbv_scan_{obj_name}", spawn=True)
            rr.send_blueprint(blueprint)

    def init_scene(self, env, mesh_world) -> None:
        """Log table, object CAD mesh, and robot UR5 link assets."""
        if not self.enabled:
            return

        # 1. Object CAD Mesh
        vertices = np.asarray(mesh_world.vertices, dtype=np.float32)
        faces = np.asarray(mesh_world.faces, dtype=np.uint32)
        rr.log("scene/cad_mesh", rr.Mesh3D(vertex_positions=vertices, triangle_indices=faces))

        # 2. Table Slab
        r = float(env.max_reach())
        bx, by = float(env.base_pose()[0][0]), float(env.base_pose()[0][1])
        table_center = [bx, by, float(env.table_top_z) - 0.02]
        rr.log(
            "world/table",
            rr.Boxes3D(
                centers=[table_center],
                half_sizes=[[r, r, 0.02]],
                colors=[[165, 115, 65]],
            ),
        )

        # 3. Robot Arm Links (UR5 + Camera + Robotiq-85 Gripper)
        shapes = env._p.getVisualShapeData(env.robot_id, physicsClientId=env.client_id)
        robotiq_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "third_party/shelf_gym_repo/shelf_gym/meshes/robotiq_85/visual",
        )
        mesh_map = {
            15: os.path.join(robotiq_dir, "robotiq_arg2f_85_outer_knuckle.obj"),
            16: os.path.join(robotiq_dir, "robotiq_arg2f_85_outer_finger.obj"),
            17: os.path.join(robotiq_dir, "robotiq_arg2f_85_inner_finger.obj"),
            19: os.path.join(robotiq_dir, "robotiq_arg2f_85_inner_knuckle.obj"),
        }

        self.robot_links_meta = []
        for s in shapes:
            link_id = s[1]
            geom = s[2]
            dims = s[3]
            filename = s[4].decode("utf-8") if isinstance(s[4], bytes) else s[4]
            local_pos, local_orn, rgba = s[5], s[6], s[7]

            if filename.endswith(".dae"):
                filename = filename.replace(".dae", ".obj")
            if link_id in mesh_map:
                filename = mesh_map[link_id]

            if filename and os.path.exists(filename):
                rr.log(f"world/robot/link_{link_id}/mesh", rr.Asset3D(path=filename, albedo_factor=rgba))
                self.robot_links_meta.append((link_id, local_pos, local_orn))
            elif geom == 3:  # Box geometry (camera sensor body, gripper pads, ee_link)
                half_sizes = [[d / 2.0 for d in dims]]
                rr.log(f"world/robot/link_{link_id}/box", rr.Boxes3D(half_sizes=half_sizes, colors=[rgba]))
                self.robot_links_meta.append((link_id, local_pos, local_orn))

        self.update_robot_pose(env)
        self._update_metrics_hud(current_cov=0.0, total_samples=len(vertices))

        # Initial benchmark tree before first view executes
        init_tree = [
            "```text",
            "Pipeline Status: Initialized",
            "├── GPU Compute [cuRobo Reachability & CUDA Ray Scoring standby]",
            "└── CPU Host    [PyBullet Sim & CAD Surface Sampling ready]",
            "```",
        ]
        rr.log("benchmark", rr.TextDocument("\n".join(init_tree), media_type="text/markdown"))

    def update_robot_pose(self, env) -> None:
        """Stream current 3D robot arm joint link poses to Rerun."""
        if not self.enabled or not self.robot_links_meta:
            return

        for link_id, local_pos, local_orn in self.robot_links_meta:
            if link_id == -1:
                base_pos, base_orn = env._p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client_id)
                w_pos, w_orn = env._p.multiplyTransforms(base_pos, base_orn, local_pos, local_orn)
            else:
                st = env._p.getLinkState(env.robot_id, link_id, physicsClientId=env.client_id)
                w_pos, w_orn = env._p.multiplyTransforms(st[4], st[5], local_pos, local_orn)
            rr.log(f"world/robot/link_{link_id}", rr.Transform3D(translation=w_pos, quaternion=w_orn))

    def log_step(
        self,
        step_idx: int,
        cand_idx: int,
        gain: int,
        newly_seen: int,
        score_ms: float,
        view_matrix: np.ndarray,
        intrinsics: CameraIntrinsics,
        tracker: CoverageTracker,
        new_cloud: np.ndarray | None = None,
    ) -> None:
        """Log camera pose, optical FOV frustum, coverage, point cloud, and visual metrics HUD."""
        if not self.enabled:
            return

        # 1. Camera Pose & Optical Frame 3D Frustum
        T_world_opengl = np.linalg.inv(view_matrix)
        T_world_cam = T_world_opengl @ T_OPENGL_OPTICAL
        t_cam = T_world_cam[:3, 3]
        R_cam = T_world_cam[:3, :3]

        rr.log("world/camera", rr.Transform3D(translation=t_cam, mat3x3=R_cam))
        rr.log(
            "world/camera/pinhole",
            rr.Pinhole(
                resolution=[intrinsics.width, intrinsics.height],
                focal_length=float(intrinsics.fx),
                principal_point=[float(intrinsics.cx), float(intrinsics.cy)],
                image_plane_distance=0.18,
            ),
        )

        # 2. Visited Camera Waypoints Trail
        self.executed_cams.append(t_cam)
        rr.log(
            "world/camera_path",
            rr.Points3D(positions=np.array(self.executed_cams), colors=[0, 180, 255], radii=0.008),
        )

        # 3. Target Surface Coverage Samples (Green = Seen, Gray = Unseen)
        colors = np.where(tracker.seen[:, None], [50, 220, 70], [160, 160, 160]).astype(np.uint8)
        rr.log(
            "coverage/surface_samples",
            rr.Points3D(positions=tracker.surface_points, colors=colors, radii=0.002),
        )

        # 4. Dense Reconstructed Point Cloud
        if new_cloud is not None and len(new_cloud) > 0:
            self.accumulated_clouds.append(new_cloud)
            full_cloud = np.concatenate(self.accumulated_clouds, axis=0)
            rr.log(
                "reconstruction/point_cloud",
                rr.Points3D(positions=full_cloud, colors=[240, 200, 40], radii=0.0015),
            )

        # 5. Record & Update Visual Metrics HUD & Bar Chart
        cov_pct = tracker.coverage_fraction() * 100.0
        total_pts = sum(len(c) for c in self.accumulated_clouds)
        self.metrics_history.append({
            "step": step_idx,
            "cand": cand_idx,
            "gain": gain,
            "newly_seen": newly_seen,
            "cov_pct": cov_pct,
            "total_pts": total_pts,
            "score_ms": score_ms,
        })
        self._update_metrics_hud(current_cov=cov_pct, total_samples=len(tracker.seen), n_seen=int(np.sum(tracker.seen)))

    def _update_metrics_hud(self, current_cov: float, total_samples: int, n_seen: int = 0) -> None:
        """Render status HUD with progress bar and full per-view execution table."""
        bar_len = 22
        filled = int(round(bar_len * (current_cov / 100.0)))
        filled = min(max(filled, 0), bar_len)
        bar_str = "[" + "=" * filled + ">" + " " * max(0, bar_len - filled - 1) + "]" if filled < bar_len else "[" + "=" * bar_len + "]"

        total_pts = sum(len(c) for c in self.accumulated_clouds)

        hud = [
            f"### Scan: `{self.obj_name}`",
            f"```text\n{bar_str} {current_cov:5.1f}%\n```",
            f"- **Coverage**: {n_seen:,} / {total_samples:,} ({current_cov:.1f}%)",
            f"- **Points**: {total_pts:,} reconstructed",
            "",
            "| View | Target | Gain | New Pts | Cloud | Time |",
            "|:---:|:---:|---:|---:|---:|---:|",
        ]
        for m in self.metrics_history:
            hud.append(
                f"| #{m['step']} | Cand #{m['cand']:03d} | {m['gain']:,} | +{m['newly_seen']:,} | {m['total_pts']:,} | {m['score_ms']:.0f}ms |"
            )

        rr.log("metrics/hud", rr.TextDocument("\n".join(hud), media_type="text/markdown"))

    def log_execution_tree(self, stages: list[dict], title: str = "Inspection Execution Tree") -> None:
        """Log live pipeline execution tree using native Markdown formatting supported by Rerun."""
        if not self.enabled:
            return

        total_ms = sum(s.get("ms", 0.0) for s in stages if s.get("status") == "DONE")
        lines = [
            f"### {title}",
            "",
        ]

        categories = ["GPU Acceleration", "Simulation & Physical Motion", "CPU Host Processing"]
        for cat in categories:
            cat_stages = [s for s in stages if s.get("cat") == cat]
            if not cat_stages:
                continue

            cat_ms = sum(s.get("ms", 0.0) for s in cat_stages if s.get("status") == "DONE")
            cat_pct = (cat_ms / total_ms * 100.0) if total_ms > 0 else 0.0
            header_stat = f" [{cat_ms:6.1f} ms | {cat_pct:4.1f}%]" if total_ms > 0 and cat_ms > 0 else ""
            lines.append(f"**{cat}**{header_stat}")

            for s in cat_stages:
                status = s.get("status", "STANDBY")
                name = s.get("name", "")
                ms = s.get("ms", 0.0)
                pct = (ms / total_ms * 100.0) if total_ms > 0 and status == "DONE" else 0.0

                if status == "RUNNING":
                    box = "- [ ]"
                    tag = "`[RUNNING]`"
                    info = "*in progress...*"
                elif status == "DONE":
                    box = "- [x]"
                    tag = "`[DONE]`"
                    pct_str = f" ({pct:4.1f}%)" if total_ms > 0 else ""
                    info = f"{ms:6.1f} ms{pct_str}"
                else:
                    box = "- [ ]"
                    tag = "`[STANDBY]`"
                    info = "-"

                lines.append(f"{box} {tag} **{name}** : {info}")
            lines.append("")

        if total_ms > 0:
            lines.append(f"**Decision Cycle Total:** `{total_ms:6.1f} ms`")

        rr.log("benchmark", rr.TextDocument("\n".join(lines), media_type="text/markdown"))

    def log_benchmark(self, timings: list[tuple[str, str, float]]) -> None:
        """Compatibility wrapper for simple flat timing lists."""
        stages = []
        for name, dev, ms in timings:
            if "Arm" in name or "Waypoint" in name:
                cat = "Simulation & Physical Motion"
            elif dev == "GPU":
                cat = "GPU Acceleration"
            else:
                cat = "CPU Host Processing"
            stages.append({"name": name, "cat": cat, "status": "DONE", "ms": ms})
        self.log_execution_tree(stages)
