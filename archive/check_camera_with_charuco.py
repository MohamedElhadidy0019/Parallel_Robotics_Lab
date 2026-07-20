"""
Standalone sanity check: replaces the mustard bottle with a charuco board of
KNOWN, exact geometry, runs the same 20-view orbit scan used elsewhere, and
cross-checks what the sim tells us against what we actually see - two
independent measurements of camera pose, neither trusting the other:

  1. t_cam_world/q_cam_world from PyBullet's getLinkState - what the rest of
     this project's pipeline currently trusts and feeds into backprojection.
  2. A pose solved purely from the rendered RGB image via cv2.aruco charuco
     detection + solvePnP - computed with zero dependency on PyBullet's
     internal state reporting, using only 2D pixel detections + the board's
     known physical geometry + camera intrinsics.

If (1) and (2) agree, getLinkState is telling the truth and any remaining
scan imprecision is a real mechanical/IK limit, not a software pose-reading
bug. If they disagree - especially if the disagreement follows the same
per-view pattern seen scanning the mustard bottle (small near arc edges,
peaking mid-arc) - that would point at a genuine, previously-invisible bug
in how camera pose is read, independent of the coarse VHACD mesh that made
the bottle comparison ambiguous.

Also directly checks reconstruction accuracy against an unambiguous ground
truth: for each detected corner, compares our own depth-based backprojected
3D position (same nbv_core.camera_geometry math the real pipeline uses)
against the corner's exactly-known world position (the board is static,
placed at a fixed pose we choose - no coarse-mesh approximation involved).

Usage:
  python check_camera_with_charuco.py           # headless (default)
  python check_camera_with_charuco.py --gui      # show the PyBullet GUI/robot
"""
import argparse
import os

import cv2
import numpy as np

from nbv_environment import NBVEnv2, TABLE_TOP_Z
from nbv_core.camera_geometry import capture_rgb_and_depth, compute_intrinsics, backproject_depth

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures", "check_camera_with_charuco")
NUM_VIEWS = 20
ORBIT_HEIGHT_M = 1.2

# Fewer, bigger squares than a typical printed board - at this camera's
# resolution (640x480) and orbit distance (~0.4m), the board only occupies
# ~140x140px of the frame. A finer 6x8 grid gave ~23px per marker cell,
# too small for cv2.aruco to detect at all (confirmed: 0 corners found,
# even though the exact same board image detects perfectly at full
# resolution). 4x5 keeps a similar physical footprint but with far larger
# cells.
SQUARES_X, SQUARES_Y = 4, 5
SQUARE_LENGTH_M = 0.035
MARKER_LENGTH_M = 0.026
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
MIN_CORNERS_FOR_POSE = 6


def build_charuco_board() -> tuple[cv2.aruco.CharucoBoard, cv2.aruco.CharucoDetector]:
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH_M, MARKER_LENGTH_M, ARUCO_DICT)
    detector = cv2.aruco.CharucoDetector(board)
    return board, detector


def _write_textured_quad_obj(obj_path: str, texture_path: str, board_w: float, board_h: float) -> None:
    """
    A GEOM_BOX visual shape with a loaded texture silently renders as a flat
    untextured surface in this headless/EGL setup (confirmed directly - even
    a plain 4-color test pattern didn't show up). A UV-mapped GEOM_MESH quad
    does render correctly, so the board is a flat two-triangle mesh instead
    of a box. Quad spans local board (0,0,0) to (board_w, board_h, 0) - the
    same origin convention as cv2's board.getChessboardCorners(), so vertex
    positions and detected-corner lookups agree without any extra offset.
    UV v is flipped (v = local_y/board_h, not 1-minus) - fixes marker
    bit-decoding, which originally failed 100% under a pure mirror (ArUco
    markers still form valid-looking square candidates under a reflection -
    same border shape - but never decode, since a reflection isn't a valid
    re-encoding of a marker, only rotations are - the giveaway this was a
    mirror, not a resolution problem).

    With v fixed, markers decode correctly, but cv2's own internal notion
    of "board-local frame" (as baked into board.getChessboardCorners()/
    matchImagePoints()) still doesn't coincide with the simple local-origin-
    at-one-corner frame assumed here - verified by solving for it directly
    against a manually-chosen, exactly-known camera pose. Rather than
    guessing further UV/winding fixes blind, BOARD_LOCAL_TO_RENDER_LOCAL
    below is that solved correction, applied wherever cv2's board-local
    coordinates need to become world coordinates (both here for rendering
    and in solve_camera_pose_from_charuco/check_view for the ground-truth
    corner math) - see BOARD_LOCAL_TO_RENDER_LOCAL's own docstring.
    """
    mtl_path = obj_path.replace(".obj", ".mtl")
    with open(mtl_path, "w") as f:
        f.write(f"newmtl mat0\nmap_Kd {os.path.basename(texture_path)}\n")
    with open(obj_path, "w") as f:
        f.write(f"""mtllib {os.path.basename(mtl_path)}
v 0 0 0
v {board_w} 0 0
v {board_w} {board_h} 0
v 0 {board_h} 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
usemtl mat0
f 1/1 2/2 3/3
f 1/1 3/3 4/4
""")


def place_charuco_board(env: NBVEnv2, board: cv2.aruco.CharucoBoard) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Removes the mustard bottle and places the board lying flat (pattern face
    up) on the table at the same x/y the bottle would have occupied, so the
    existing orbit geometry (radius/height tuned around that footprint)
    stays representative. Static (mass=0) - its world pose is then known
    exactly, with no drop/settle physics to introduce uncertainty.

    Returns: (board_body_id, R_world_board, t_world_board) that map cv2's
    OWN internal board-local frame (as used by board.getChessboardCorners()/
    matchImagePoints()) into world coordinates - NOT the simpler frame used
    to place the physical mesh below. Solved directly against a manually-
    chosen, exactly-known camera pose (see module debugging history): cv2's
    convention has Z pointing INTO the board (away from a viewer reading it
    - standard OpenCV convention, not "outward surface normal" as naively
    assumed at first) and its origin at the opposite corner along local X.
    Both are a fixed local-frame reinterpretation, expressed here as
    R_render_board -> R_render_board @ diag(-1,1,-1) plus an origin shift -
    generalizes to any physical placement, not just this flat/no-tilt one.
    """
    env._p.removeBody(env.obj_id)

    board_w = SQUARES_X * SQUARE_LENGTH_M
    board_h = SQUARES_Y * SQUARE_LENGTH_M
    thickness = 0.005

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    board_image = board.generateImage(
        (int(board_w * 4000), int(board_h * 4000)), marginSize=0, borderBits=1
    )
    texture_path = os.path.join(OUTPUT_DIR, "charuco_board_texture.png")
    cv2.imwrite(texture_path, cv2.cvtColor(board_image, cv2.COLOR_GRAY2BGR))
    obj_path = os.path.join(OUTPUT_DIR, "charuco_board_quad.obj")
    _write_textured_quad_obj(obj_path, texture_path, board_w, board_h)

    # World placement: board's local origin (one outer corner) sits at
    # (t_x, t_y) with local +X/+Y axes aligned to world +X/+Y (no tilt) -
    # offset by -board_w/2,-board_h/2 so the board is CENTERED at the same
    # point the mustard bottle used to occupy (keeps the existing orbit
    # geometry, tuned around that footprint, representative).
    t_x = env.init_pos[0] - board_w / 2
    t_y = env.init_pos[1] + 0.10 - board_h / 2
    t_z = TABLE_TOP_Z + thickness
    t_render_board = np.array([t_x, t_y, t_z])
    q_world_board = env._p.getQuaternionFromEuler([0, 0, 0])  # flat, no tilt
    R_render_board = np.array(env._p.getMatrixFromQuaternion(q_world_board)).reshape(3, 3)

    half_extents = [board_w / 2, board_h / 2, thickness / 2]
    col = env._p.createCollisionShape(
        env._p.GEOM_BOX, halfExtents=half_extents,
        collisionFramePosition=[board_w / 2, board_h / 2, 0],
    )
    vis = env._p.createVisualShape(env._p.GEOM_MESH, fileName=obj_path)
    board_id = env._p.createMultiBody(0, col, vis, t_render_board.tolist(), q_world_board)

    # cv2-local-frame correction (see docstring): Z into the board, origin
    # at the opposite local-X corner.
    R_world_board = R_render_board @ np.diag([-1.0, 1.0, -1.0])
    t_world_board = t_render_board + R_render_board @ np.array([board_w, 0, 0])

    print(f"Board placed at t_render_board={np.round(t_render_board, 4)}, "
          f"size={board_w:.3f}x{board_h:.3f}m, {len(board.getChessboardCorners())} internal corners")
    return board_id, R_world_board, t_world_board


def solve_camera_pose_from_charuco(
    rgb: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    detector: cv2.aruco.CharucoDetector,
    camera_matrix: np.ndarray,
    R_world_board: np.ndarray,
    t_world_board: np.ndarray,
    approx_viewpoint_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Detects the charuco board in `rgb` and solves for the camera's pose in
    WORLD frame using only the 2D detections + board geometry + intrinsics -
    independent of anything PyBullet reports internally.

    `approx_viewpoint_world` is the orbit's own COMMANDED viewpoint for this
    view (radius/height/board-center geometry only, NOT anything PyBullet
    reports about where the arm actually ended up) - needed to break a
    genuine planar-PnP ambiguity, see below.

    Returns: (t_cam_world_pnp, R_cam_world_pnp, charuco_corners_px, charuco_ids) or None if not enough
             corners were detected to solve a pose.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_ids is None or len(charuco_ids) < MIN_CORNERS_FOR_POSE:
        return None

    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    # SOLVEPNP_IPPE (not the default iterative solver) - board points are
    # coplanar (Z=0), and both give TWO mathematically valid solutions for a
    # planar target, mirrored through the board's plane. Confirmed directly
    # this is a genuine ambiguity, not just a fixed convention bug: a
    # near-fronto-parallel test view landed on the correct side by default,
    # but oblique orbit views consistently landed on the mirrored side
    # (~784mm/106deg error, constant regardless of orbit angle - the
    # signature of a systematically-wrong branch pick). Also confirmed
    # reprojection error can't disambiguate - for one oblique view the WRONG
    # (mirrored, 772mm off) candidate had LOWER reprojection error (0.14)
    # than the correct one (65mm off, error 1.24), so picking min-error is
    # actively unreliable here. A table-height check (camera must be above
    # the table) doesn't work either - both candidates land above table
    # height for these viewing angles. What DOES work: pick whichever
    # candidate is closer to the orbit's own commanded viewpoint - we know
    # that only approximately (real settle error is a few mm-cm, exactly
    # what the rest of this project's diagnostics have been chasing), but
    # it's ~1000x smaller than the ~700-800mm gap between the two mirrored
    # candidates, so it cleanly and robustly picks the right branch without
    # leaning on getLinkState's actual reported pose (which is what this
    # whole check exists to independently verify).
    n_solutions, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        obj_points, img_points, camera_matrix, None, flags=cv2.SOLVEPNP_IPPE
    )
    if n_solutions == 0:
        return None

    T_board_world = np.eye(4)
    T_board_world[:3, :3] = R_world_board
    T_board_world[:3, 3] = t_world_board

    candidates = []
    for i in range(n_solutions):
        R_cam_board, _ = cv2.Rodrigues(rvecs[i])  # X_cam = R_cam_board @ X_board + t_cam_board
        t_cam_board = tvecs[i].reshape(3)

        T_board_cam = np.eye(4)
        T_board_cam[:3, :3] = R_cam_board
        T_board_cam[:3, 3] = t_cam_board

        T_cam_world = T_board_world @ np.linalg.inv(T_board_cam)
        candidates.append((T_cam_world[:3, 3], T_cam_world[:3, :3]))

    best_t_cam_world, best_R_cam_world = min(
        candidates, key=lambda c: np.linalg.norm(c[0] - approx_viewpoint_world)
    )
    return best_t_cam_world, best_R_cam_world, charuco_corners, charuco_ids


def check_view(
    view_index: int,
    env: NBVEnv2,
    board: cv2.aruco.CharucoBoard,
    detector: cv2.aruco.CharucoDetector,
    R_world_board: np.ndarray,
    t_world_board: np.ndarray,
    approx_viewpoint_world: np.ndarray,
) -> None:
    rgb, depth_mm, t_cam_world, q_cam_world, _ = capture_rgb_and_depth(
        env._p, env.robot_id, env.camera_link,
        env.camera.width, env.camera.height, env.camera.fov,
        env.camera.near, env.camera.far,
    )
    fx, fy, cx, cy = compute_intrinsics(env.camera.width, env.camera.height, env.camera.fov)
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    result = solve_camera_pose_from_charuco(
        rgb, board, detector, camera_matrix, R_world_board, t_world_board, approx_viewpoint_world
    )
    if result is None:
        print(f"View {view_index}: board not detected (too few corners visible) - skipping")
        return
    t_cam_world_pnp, R_cam_world_pnp, charuco_corners, charuco_ids = result

    R_cam_world_sim = np.array(env._p.getMatrixFromQuaternion(q_cam_world)).reshape(3, 3)

    pos_error_mm = np.linalg.norm(t_cam_world_pnp - np.asarray(t_cam_world)) * 1000
    forward_sim = R_cam_world_sim[:, 2]
    forward_pnp = R_cam_world_pnp[:, 2]
    forward_angle_deg = np.degrees(np.arccos(np.clip(np.dot(forward_sim, forward_pnp), -1, 1)))

    # Direct reconstruction-accuracy check: our own depth backprojection at
    # each detected corner's pixel vs that corner's exactly-known world
    # position (board is static at a known pose - unambiguous ground truth).
    points_world_grid = backproject_depth(depth_mm, t_cam_world, q_cam_world, fx, fy, cx, cy)
    board_corners_local = board.getChessboardCorners()
    recon_errors_mm = []
    for corner_px, corner_id in zip(charuco_corners.reshape(-1, 2), charuco_ids.reshape(-1)):
        u, v = int(round(corner_px[0])), int(round(corner_px[1]))
        if not (0 <= v < depth_mm.shape[0] and 0 <= u < depth_mm.shape[1]):
            continue
        known_world = R_world_board @ board_corners_local[corner_id] + t_world_board
        recon_world = points_world_grid[v, u]
        recon_errors_mm.append(np.linalg.norm(recon_world - known_world) * 1000)
    recon_errors_mm = np.array(recon_errors_mm)

    print(f"View {view_index}: {len(charuco_ids):>2} corners detected | "
          f"getLinkState-vs-PnP camera position diff={pos_error_mm:6.2f}mm, "
          f"forward-direction diff={forward_angle_deg:5.2f}deg | "
          f"depth-recon-vs-known-corner error: mean={recon_errors_mm.mean():5.2f}mm "
          f"median={np.median(recon_errors_mm):5.2f}mm max={recon_errors_mm.max():5.2f}mm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true", help="Show the PyBullet GUI (see the robot move).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    board, detector = build_charuco_board()

    env = NBVEnv2(render=args.gui)
    board_id, R_world_board, t_world_board = place_charuco_board(env, board)
    env.obj_id = board_id
    board_w = SQUARES_X * SQUARE_LENGTH_M
    board_h = SQUARES_Y * SQUARE_LENGTH_M
    # move_through_orbit orbits around/looks at env.obj_pos - needs the
    # board's CENTER, not t_world_board (which is the corner-origin used for
    # the ground-truth corner math, matching getChessboardCorners()).
    env.obj_pos = t_world_board + R_world_board @ np.array([board_w / 2, board_h / 2, 0])

    # Recomputed independently here (same formula move_through_orbit uses
    # internally) purely to give solve_camera_pose_from_charuco a rough,
    # PyBullet-state-independent prior for disambiguating the planar-PnP
    # mirror ambiguity - see that function's docstring.
    radius = env._compute_safe_orbit_radius()
    angles = np.linspace(np.pi, 2 * np.pi, NUM_VIEWS, endpoint=True)

    def approx_viewpoint(view_index: int) -> np.ndarray:
        theta = angles[view_index]
        return np.array([
            env.obj_pos[0] + radius * np.cos(theta),
            env.obj_pos[1] + radius * np.sin(theta),
            ORBIT_HEIGHT_M,
        ])

    env.move_through_orbit(
        n_views=NUM_VIEWS,
        height=ORBIT_HEIGHT_M,
        on_stop=lambda view_index: check_view(
            view_index, env, board, detector, R_world_board, t_world_board, approx_viewpoint(view_index)
        ),
    )


if __name__ == "__main__":
    main()
