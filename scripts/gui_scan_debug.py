"""
Interactive GUI debugging tool for the scan/reconstruct pipeline.

WHAT THIS SCRIPT DOES, IN ORDER:
  1. Opens the real PyBullet GUI (same window you'd see running the sim
     normally - the arm, table, and object are all visible and moving).
  2. Calls env.scan_fixed_views_and_integrate(), which drives the arm through
     its fixed 8-view sweep. That method already does the real work (capture
     depth, back-project to 3D, filter bad points, fuse into the voxel grid)
     - this script does NOT reimplement any of that. It only adds a callback,
     `on_view`, which scan_fixed_views_and_integrate calls after each of the
     8 stops.
  3. Inside that callback, we draw 3 sets of colored points directly into the
     PyBullet window (so you can see them overlaid on the real robot/object),
     and save a same-view RGB+depth image to disk:
       GREEN  = points from this view that passed all filters
       RED    = points from this view that were in a plausible depth range
                but got thrown out by the edge-discontinuity filter (i.e.
                "possible flying-pixel artifact caught here")
       BLUE   = the full reconstruction accumulated so far, across every
                view up to and including this one
  4. Pauses after each view so you can rotate/inspect before continuing.

Usage:
  conda activate rob_env
  python scripts/gui_scan_debug.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nbv_environment import NBVEnv2
from nbv_core.io_utils import save_rgb_depth_composite

# Rendering thousands of debug points every view would be slow and cluttered,
# so anything larger gets randomly thinned down to this many points first.
MAX_POINTS_TO_DRAW = 3000

# Where per-view point clouds and RGB/depth images get saved.
SAVE_DIR = "captures/voxel_fusion_test"

# If there's no real terminal attached (e.g. running via `conda run` instead
# of `conda activate`), input() can't wait for a keypress - we pause for this
# many seconds instead so the script still runs instead of crashing.
FALLBACK_PAUSE_SECONDS = 3.0

GREEN = [0, 1, 0]
RED = [1, 0, 0]
BLUE = [0.15, 0.4, 1.0]


class LiveDebugPoints:
    """
    A point cloud drawn into the PyBullet GUI that REPLACES itself every time
    you call .update(), instead of piling up a new drawing on top of the old
    one. PyBullet does this via addUserDebugPoints' `replaceItemUniqueId`
    argument: pass -1 the first time ("nothing to replace yet"), then pass
    back whatever ID the previous call returned. This class just remembers
    that ID for you so the calling code doesn't have to.
    """

    def __init__(self, sim):
        self._sim = sim
        self._debug_item_id = -1

    def update(self, points, color, point_size):
        if len(points) == 0:
            return
        colors = [color] * len(points)
        self._debug_item_id = self._sim.addUserDebugPoints(
            points.tolist(), colors, pointSize=point_size,
            replaceItemUniqueId=self._debug_item_id,
        )


def subsample(points, max_points):
    """Randomly thin a point array down to at most max_points rows."""
    if len(points) <= max_points:
        return points
    keep = np.random.choice(len(points), max_points, replace=False)
    return points[keep]


def split_accepted_and_edge_rejected(frame):
    """
    frame comes from NBVEnv2.capture_frame(). It already tells us, per pixel,
    whether a point was kept ('valid_mask'), whether it was in a sane depth
    range ('range_valid_mask'), and whether it passed the edge-discontinuity
    check ('edge_valid_mask'). Here we pull out two point sets for display:
      - accepted:      what actually got integrated into the voxel grid
      - edge_rejected: points that WOULD have been in range, but specifically
                        failed the edge check - i.e. exactly the points the
                        edge filter is responsible for removing. We isolate
                        these (rather than showing all rejected points) so
                        the huge, uninteresting mass of background pixels
                        doesn't drown out the thing we actually want to see.
    """
    accepted = frame['points_world']

    edge_rejected_mask = frame['range_valid_mask'] & ~frame['edge_valid_mask']
    edge_rejected = frame['points_world_grid'][edge_rejected_mask]

    return accepted, edge_rejected


def print_view_summary(view_index, frame, edge_rejected, grid):
    print(f"\n=== View {view_index} ===")
    print(f"  accepted (green):            {len(frame['points_world'])}")
    print(f"  edge-rejected (red):         {len(edge_rejected)}")
    print(f"  accumulated occupied (blue): {grid.get_occupied_count()}")


def save_view_snapshot(env, frame, view_index):
    """Save this view's real camera image next to its colorized depth map."""
    path = os.path.join(SAVE_DIR, f"rgb_depth_view_{view_index:02d}.png")
    save_rgb_depth_composite(
        frame['rgb'], frame['transformed_depth'], path,
        near_m=env.camera.near, far_m=env.camera.far,
    )
    print(f"  saved RGB|depth image to {path}")


def wait_for_user():
    """
    Block until the user presses Enter, so they have time to look at the
    GUI before the arm moves to the next view. Falls back to a fixed sleep
    if stdin isn't a real terminal (see FALLBACK_PAUSE_SECONDS above).
    """
    try:
        input("  Press Enter to continue to the next view (Ctrl+C to stop)...")
    except EOFError:
        print(f"  (no interactive terminal - pausing {FALLBACK_PAUSE_SECONDS}s instead)")
        time.sleep(FALLBACK_PAUSE_SECONDS)


def handle_view(view_index, frame, grid, env, drawers):
    """
    Everything that happens after ONE view has been captured and fused into
    the grid: draw it, save it, report it, and pause. This is the function
    passed to scan_fixed_views_and_integrate as `on_view` (via the small
    wrapper in main() below) - it does not move the arm or touch the grid
    itself, that's already done by the time this runs.
    """
    accepted, edge_rejected = split_accepted_and_edge_rejected(frame)

    drawers.accepted.update(subsample(accepted, MAX_POINTS_TO_DRAW), GREEN, point_size=5)
    drawers.edge_rejected.update(subsample(edge_rejected, MAX_POINTS_TO_DRAW), RED, point_size=5)

    reconstruction = subsample(grid.get_occupied_points(), MAX_POINTS_TO_DRAW)
    drawers.reconstruction.update(reconstruction, BLUE, point_size=3)

    print_view_summary(view_index, frame, edge_rejected, grid)
    save_view_snapshot(env, frame, view_index)
    wait_for_user()


class Drawers:
    """
    Three independent LiveDebugPoints, one per color. Kept as separate
    objects (rather than one shared drawing) so updating one color each view
    doesn't erase the other two - each has its own remembered debug-item ID.
    """

    def __init__(self, sim):
        self.accepted = LiveDebugPoints(sim)
        self.edge_rejected = LiveDebugPoints(sim)
        self.reconstruction = LiveDebugPoints(sim)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    env = NBVEnv2(render=True)
    drawers = Drawers(env._p)

    def on_view(view_index, frame, grid):
        handle_view(view_index, frame, grid, env, drawers)

    env.scan_fixed_views_and_integrate(
        n_views=8, height=1.2, save_dir=SAVE_DIR, on_view=on_view
    )

    print("\nScan complete. GUI window stays open - press Enter here to exit.")
    wait_for_user()


if __name__ == "__main__":
    main()
