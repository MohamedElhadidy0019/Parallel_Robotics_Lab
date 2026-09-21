"""Interactive viewer for scan point clouds and coverage meshes."""

import argparse
import os
import time
import numpy as np
import open3d as o3d


def load_ply_points(path: str) -> np.ndarray:
    """Parse ASCII PLY file vertices."""
    points = []
    header = True
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if header:
                if line == "end_header":
                    header = False
                continue
            parts = line.split()
            if len(parts) >= 3:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.asarray(points, dtype=np.float32)


def view_in_rerun(path: str) -> None:
    """View point cloud or mesh in Rerun interactive 3D viewer."""
    import rerun as rr

    rr.init("scan_viewer")
    addr = os.environ.get("RERUN_ADDR", os.environ.get("RERUN_SERVER", "127.0.0.1:9876"))
    try:
        rr.connect(addr)
    except Exception:
        try:
            rr.spawn()
        except Exception:
            pass
    if "coverage" in path:
        mesh = o3d.io.read_triangle_mesh(path)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.triangles, dtype=np.uint32)
        colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8) if len(mesh.vertex_colors) > 0 else None
        rr.log("mesh/coverage_mesh", rr.Mesh3D(vertex_positions=vertices, triangle_indices=faces, vertex_colors=colors))
    else:
        pts = load_ply_points(path)
        rr.log("cloud/reconstructed_points", rr.Points3D(positions=pts, colors=[240, 200, 40], radii=0.0015))
    print(f"[Rerun] Displaying {path}")


def view_in_pybullet(path: str, n_points: int = 15000) -> None:
    """View 3D scan points in an interactive PyBullet GUI window."""
    import pybullet as p
    import pybullet_data

    pts = load_ply_points(path)
    if len(pts) == 0:
        print(f"Error: No points found in {path}")
        return

    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=cid)
    p.loadURDF("plane.urdf", physicsClientId=cid)

    idx = np.random.choice(len(pts), min(n_points, len(pts)), replace=False)
    pts_sub = pts[idx]

    z_min, z_max = pts_sub[:, 2].min(), pts_sub[:, 2].max()
    z_norm = (pts_sub[:, 2] - z_min) / max(z_max - z_min, 1e-4)
    colors = np.stack([z_norm, 1.0 - 0.5 * z_norm, 0.2 * np.ones_like(z_norm)], axis=-1)

    p.addUserDebugPoints(pts_sub.tolist(), colors.tolist(), pointSize=3, physicsClientId=cid)

    center = pts_sub.mean(axis=0)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.4,
        cameraYaw=45,
        cameraPitch=-25,
        cameraTargetPosition=center.tolist(),
        physicsClientId=cid,
    )

    print(f"Showing {len(pts_sub)} / {len(pts)} points in PyBullet.")
    print("Hold Left-Click + Drag: Rotate | Scroll: Zoom | Close window to exit")

    while p.isConnected(cid):
        time.sleep(1 / 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", default="captures/scan_YcbMustardBottle.ply")
    parser.add_argument("--pb", action="store_true", help="Use PyBullet viewer instead of Rerun")
    args = parser.parse_args()

    target = args.file
    if not os.path.exists(target):
        alt = os.path.join("captures", target)
        if os.path.exists(alt):
            target = alt
        else:
            print(f"Error: File not found: {target}")
            return

    if args.pb:
        view_in_pybullet(target)
    else:
        view_in_rerun(target)


if __name__ == "__main__":
    main()
