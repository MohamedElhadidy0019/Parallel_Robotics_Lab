"""Live Rerun multi-tab visualizer with full 3D robot arm, camera FOV, and visual metrics HUD."""

import os
import sys
import time
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from nbv_planner.camera import CameraIntrinsics
from nbv_planner.config import T_OPENGL_OPTICAL
from nbv_planner.coverage import CoverageTracker


class NBVVisualizer:
    """Manages 4-tab interactive Rerun visualization including 3D robot arm motion."""

    def __init__(self, obj_name: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self.obj_name = obj_name
        self.executed_cams: list[np.ndarray] = []
        self.accumulated_clouds: list[np.ndarray] = []
        self.metrics_history: list[dict] = []
        self.robot_links_meta: list[tuple[int, tuple, tuple]] = []

        self.setup_stages: list[dict] = [
            {"name": "Start Pose", "status": "STANDBY", "info": ""},
            {"name": "Scene & Target Surface", "status": "STANDBY", "info": ""},
            {"name": "Orbit Viewpoints", "status": "STANDBY", "info": ""},
            {"name": "cuRobo IK Reachability", "status": "STANDBY", "info": ""},
        ]
        self.completed_views: list[dict] = []
        self.active_view: dict | None = None
        self._last_telemetry_time: float = 0.0
        self._cached_telemetry_str: str = ""

        if self.enabled:
            # Multi-tab layout:
            # Tab 0: 3D Inspection (World, Robot, Reconstruction, Coverage, HUD)
            # Tab 1: Pipeline Benchmark Tree (Unfolding execution tree & telemetry)
            blueprint = rrb.Blueprint(
                rrb.Tabs(
                    rrb.Horizontal(
                        rrb.Spatial3DView(name="Live 3D Environment (World & Robot)", contents=["+ /**"]),
                        rrb.Vertical(
                            rrb.Spatial3DView(name="Reconstruction", contents=["+ scene/**", "+ reconstruction/**"]),
                            rrb.Spatial3DView(name="Coverage", contents=["+ scene/**", "+ coverage/**"]),
                            rrb.TextDocumentView(name="Status HUD", contents=["+ metrics/hud/**"]),
                            row_shares=[1.0, 1.0, 0.7],
                        ),
                        column_shares=[1.75, 1.0],
                        name="3D Inspection",
                    ),
                    rrb.TextDocumentView(
                        name="Pipeline Benchmark Tree",
                        contents=["+ benchmark/**", "+ benchmark"],
                    ),
                    active_tab=0,
                ),
            )
            # rr.spawn looks up the `rerun` viewer binary on PATH; it lives next to the interpreter.
            os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
            rr.init(f"nbv_scan_{obj_name}", spawn=True)
            rr.send_blueprint(blueprint)

    def log_cad_mesh(self, mesh_world) -> None:
        """Log the object CAD mesh once the planner has loaded it."""
        if not self.enabled:
            return
        vertices = np.asarray(mesh_world.vertices, dtype=np.float32)
        faces = np.asarray(mesh_world.faces, dtype=np.uint32)
        rr.log("scene/cad_mesh", rr.Mesh3D(vertex_positions=vertices, triangle_indices=faces))

    def log_start_pose(self, observation, look_at_world: np.ndarray) -> None:
        """Log the camera frustum and RGB image captured at the start pose, plus the look-at point."""
        if not self.enabled:
            return
        T = observation.world_from_camera
        intr = observation.intrinsics
        rr.log("world/start_pose", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
        rr.log(
            "world/start_pose/pinhole",
            rr.Pinhole(
                resolution=[intr.width, intr.height],
                focal_length=float(intr.fx),
                principal_point=[float(intr.cx), float(intr.cy)],
                image_plane_distance=0.18,
            ),
        )
        rr.log("world/start_pose/pinhole/rgb", rr.Image(observation.rgb))
        rr.log("world/start_look_at", rr.Points3D(positions=[look_at_world], colors=[255, 120, 30], radii=0.01))

    def init_scene(self, env) -> None:
        """Log table and robot UR5 link assets."""
        if not self.enabled:
            return

        # 1. Table Slab
        if hasattr(env, "table_aabb"):
            lo, hi = env.table_aabb
            table_center = ((lo + hi) / 2.0).tolist()
            half_sizes = [((hi - lo) / 2.0).tolist()]
        else:
            r = float(env.max_reach())
            bx, by = float(env.base_pose()[0][0]), float(env.base_pose()[0][1])
            table_center = [bx, by, float(env.table_top_z) - 0.02]
            half_sizes = [[r, r, 0.02]]
        rr.log(
            "world/table",
            rr.Boxes3D(
                centers=[table_center],
                half_sizes=half_sizes,
                colors=[[165, 115, 65]],
            ),
        )

        # Ground plane at z = 0 (Steve base sits directly on ground)
        rr.log(
            "world/ground",
            rr.Boxes3D(
                centers=[[0.0, 0.0, -0.01]],
                half_sizes=[[2.5, 2.5, 0.01]],
                colors=[[210, 210, 215]],
            ),
        )

        # 2. Robot Links (MPO-700 Chassis + Cabinet + UR5 + Camera + Robotiq-85 Gripper)
        shapes = env._p.getVisualShapeData(env.robot_id, physicsClientId=env.client_id)
        self.robot_links_meta = []
        for s in shapes:
            link_id = s[1]
            geom = s[2]
            dims = s[3]
            filename = s[4].decode("utf-8") if isinstance(s[4], bytes) else s[4]
            local_pos, local_orn, rgba = s[5], s[6], s[7]

            if filename:
                if not os.path.isabs(filename):
                    filename = os.path.abspath(filename)
                if filename.endswith(".dae"):
                    obj_cand = filename[:-4] + ".obj"
                    if os.path.exists(obj_cand):
                        filename = obj_cand
                if os.path.exists(filename):
                    rr.log(f"world/robot/link_{link_id}/mesh", rr.Asset3D(path=filename, albedo_factor=rgba))
                    scale_vec = dims if (dims and len(dims) == 3) else (1.0, 1.0, 1.0)
                    self.robot_links_meta.append((link_id, local_pos, local_orn, scale_vec))
            elif geom == 3:  # Box geometry (camera sensor body, gripper pads, ee_link)
                half_sizes = [[d / 2.0 for d in dims]]
                rr.log(f"world/robot/link_{link_id}/box", rr.Boxes3D(half_sizes=half_sizes, colors=[rgba]))
                self.robot_links_meta.append((link_id, local_pos, local_orn, (1.0, 1.0, 1.0)))

        self.update_robot_pose(env)
        self._update_metrics_hud(current_cov=0.0, total_samples=0)

        # Initial benchmark tree before first view executes
        init_tree = [
            "```text",
            "Pipeline Status: Initialized",
            "├── GPU Compute [cuRobo Reachability & CUDA Ray Scoring standby]",
            "└── CPU Host    [PyBullet Sim ready]",
            "```",
        ]
        rr.log("benchmark", rr.TextDocument("\n".join(init_tree), media_type="text/markdown"))

    def update_robot_pose(self, env) -> None:
        """Stream current 3D robot joint link poses to Rerun."""
        if not self.enabled or not self.robot_links_meta:
            return

        for link_id, local_pos, local_orn, scale_vec in self.robot_links_meta:
            if link_id == -1:
                base_pos, base_orn = env._p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client_id)
                w_pos, w_orn = env._p.multiplyTransforms(base_pos, base_orn, local_pos, local_orn)
            else:
                st = env._p.getLinkState(env.robot_id, link_id, physicsClientId=env.client_id)
                w_pos, w_orn = env._p.multiplyTransforms(st[4], st[5], local_pos, local_orn)
            rr.log(f"world/robot/link_{link_id}", rr.Transform3D(translation=w_pos, quaternion=w_orn, scale=scale_vec))

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


    def update_setup_stage(self, name: str, status: str, info: str = "") -> None:
        """Update scene or kinematics setup stage status in the execution tree."""
        if not self.enabled:
            return
        for s in self.setup_stages:
            if s["name"] == name:
                s["status"] = status
                s["info"] = info
                break
        else:
            self.setup_stages.append({"name": name, "status": status, "info": info})
        self._render_pipeline_tree()

    def start_view(self, view_idx: int, max_views: int) -> None:
        """Unfold a new scanning view in the execution tree."""
        if not self.enabled:
            return
        self.active_view = {
            "view_idx": view_idx,
            "max_views": max_views,
            "stages": [
                {"name": "CUDA Ray Scoring", "status": "RUNNING", "ms": 0.0},
                {"name": "cuRobo Batch Opt", "status": "STANDBY", "ms": 0.0},
                {"name": "Arm Waypoint Drive (120Hz)", "status": "STANDBY", "ms": 0.0},
                {"name": "RGB-D Camera Capture", "status": "STANDBY", "ms": 0.0},
                {"name": "KDTree Coverage Match", "status": "STANDBY", "ms": 0.0},
            ],
            "summary": None,
        }
        self._render_pipeline_tree()

    def update_view_stage(self, stage_name: str, status: str, ms: float = 0.0) -> None:
        """Update an active view sub-stage status and latency."""
        if not self.enabled or not self.active_view:
            return
        for s in self.active_view["stages"]:
            if s["name"] == stage_name:
                s["status"] = status
                s["ms"] = ms
                break
        self._render_pipeline_tree()

    def complete_view(self, cand_idx: int, gain: int, newly_seen: int, cov_pct: float) -> None:
        """Finish active view, record metrics and timings, and persist in the tree."""
        if not self.enabled or not self.active_view:
            return
        total_ms = sum(s["ms"] for s in self.active_view["stages"])
        self.active_view["summary"] = {
            "cand_idx": cand_idx,
            "gain": gain,
            "newly_seen": newly_seen,
            "cov_pct": cov_pct,
            "total_ms": total_ms,
        }
        self.completed_views.append(self.active_view)
        self.active_view = None
        self._render_pipeline_tree()

    def _get_telemetry_str(self) -> str:
        now = time.perf_counter()
        if now - self._last_telemetry_time < 0.5 and self._cached_telemetry_str:
            return self._cached_telemetry_str
        self._last_telemetry_time = now
        telemetry = ""
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            ram_used = (vm.total - vm.available) / (1024**3)
            ram_tot = vm.total / (1024**3)
            telemetry = f"`CPU: {cpu:.0f}%` | `RAM: {ram_used:.1f} / {ram_tot:.1f} GB`"
        except Exception:
            telemetry = "`CPU: -`"

        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                text=True, timeout=0.08
            ).strip().split(",")
            if len(out) >= 3:
                telemetry += f" | `GPU: {out[0].strip()}%` | `VRAM: {out[1].strip()} / {out[2].strip()} MB`"
        except Exception:
            pass

        self._cached_telemetry_str = telemetry
        return telemetry

    def _render_pipeline_tree(self) -> None:
        """Assemble and stream the unfolding execution tree to Rerun."""
        if not self.enabled:
            return

        lines = ["### NBV Inspection Pipeline", ""]

        lines.append("**1. Scene & Kinematics Setup**")
        for s in self.setup_stages:
            status = s.get("status", "STANDBY")
            if status == "DONE":
                box = "- [x]"
                tag = "`[DONE]`"
            elif status == "RUNNING":
                box = "- [ ]"
                tag = "`[RUNNING]`"
            else:
                box = "- [ ]"
                tag = "`[STANDBY]`"
            info = f" : {s['info']}" if s.get("info") else ""
            lines.append(f"{box} {tag} **{s['name']}**{info}")

        lines.append("")
        lines.append("**2. Autonomous Scan Execution Tree**")

        if not self.completed_views and not self.active_view:
            lines.append("- *Awaiting scan loop start...*")

        for cv in self.completed_views:
            sm = cv["summary"]
            lines.append(
                f"- [x] **View {cv['view_idx']}** : `Cand #{sm['cand_idx']:03d}` "
                f"(+{sm['newly_seen']} seen | {sm['cov_pct']:.1f}% cov | {sm['total_ms']:.0f} ms)"
            )
            for st in cv["stages"]:
                lines.append(f"  - `[DONE]` {st['name']} : `{st['ms']:6.1f} ms`")

        if self.active_view:
            lines.append(f"- [ ] **View {self.active_view['view_idx']}** : *in progress...*")
            for st in self.active_view["stages"]:
                status = st.get("status", "STANDBY")
                if status == "RUNNING":
                    lines.append(f"  - [ ] `[RUNNING]` **{st['name']}** : *in progress...*")
                elif status == "DONE":
                    lines.append(f"  - [x] `[DONE]` **{st['name']}** : `{st['ms']:6.1f} ms`")
                else:
                    lines.append(f"  - [ ] `[STANDBY]` **{st['name']}** : -")

        lines.append("")
        lines.append("---")
        # Coverage Progress Bar
        bar_width = 30
        last_cov = self.completed_views[-1]["summary"]["cov_pct"] if self.completed_views else 0.0
        ratio = min(1.0, max(0.0, last_cov / 100.0))
        filled = int(ratio * bar_width)
        bar_str = "█" * filled + "░" * (bar_width - filled)
        lines.append(f"**Coverage:** `[{bar_str}] {last_cov:5.1f}% / 95% target`")
        lines.append(f"**Telemetry:** {self._get_telemetry_str()}")

        rr.log("benchmark", rr.TextDocument("\n".join(lines), media_type="text/markdown"))

    def log_execution_tree(self, stages: list[dict], title: str = "NBV Inspection Pipeline") -> None:
        """Backward-compatible wrapper: forwards stage updates to active view."""
        if not self.enabled:
            return
        if self.active_view:
            for s in stages:
                self.update_view_stage(s.get("name", ""), s.get("status", "STANDBY"), s.get("ms", 0.0))
        else:
            self._render_pipeline_tree()

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
