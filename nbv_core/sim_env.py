"""UR5 + table + target object. Sizes itself from whatever is loaded."""

import os

import numpy as np
import pybullet_data
from shelf_gym.environments.ur5_environment import RobotEnv

from nbv_core.camera import CameraIntrinsics
from nbv_core.config import (
    ASSET_PATH,
    DEFAULT_YCB_OBJECT,
    ORBIT_DEPTH_FRACTION,
    PLACEMENT_DIRECTION,
    PROJECT_ROOT,
    YCB_ROOT,
)


def ycb_urdf(name: str) -> str:
    """Path to a vendored YCB object's urdf."""
    # model.urdf renders the detailed scanned mesh; model_textureless renders the coarse
    # collision hull instead, which the depth camera would then see.
    return os.path.join(YCB_ROOT, name, "model.urdf")


def ycb_names() -> list:
    """Every vendored object that ships a urdf; some dirs are loose meshes."""
    return sorted(d for d in os.listdir(YCB_ROOT) if os.path.isfile(ycb_urdf(d)))


class SimEnv(RobotEnv):
    # Set by the shelf_gym parent chain; declared for tooling only.
    robot_id: int
    camera_link: int
    client_id: int
    init_pos: list
    joints: dict
    arm_joint_names: list
    arm_joint_indices: list

    plane_id: int
    table_id: int
    obj_id: int
    obj_pos: np.ndarray
    table_top_z: float

    def __init__(
        self,
        render: bool = True,
        intrinsics: CameraIntrinsics | None = None,
        ycb_object: str = DEFAULT_YCB_OBJECT,
    ) -> None:
        super().__init__(render=render, show_vis=False)
        self.intrinsics = intrinsics or CameraIntrinsics()
        self.ycb_object = ycb_object
        self._max_reach = None
        self.table_top_z = float(self.base_pose()[0][2])  # arm mount height == work surface

        self._build_scene()
        self._place_object()
        # shelf_gym turns rendering off to load and never turns it back on.
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)

    # --- measured ------------------------------------------------------------

    def base_pose(self):
        """(position, xyzw quaternion) of the arm's base in world."""
        t, q = self._p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        return np.array(t), np.array(q)

    def max_reach(self, samples: int = 6000, seed: int = 0) -> float:
        """How far the camera link can get from the base, horizontally."""
        if self._max_reach is not None:
            return self._max_reach

        recs = [self.joints[n] for n in self.arm_joint_names]
        ids = [r.id for r in recs]
        lo = np.array([r.lowerLimit for r in recs])
        hi = np.array([r.upperLimit for r in recs])
        start = [self._p.getJointState(self.robot_id, i, physicsClientId=self.client_id)[0] for i in ids]

        # No reach figure exists in the urdf, so try random arm poses and keep the furthest.
        rng = np.random.default_rng(seed)
        base_xy = self.base_pose()[0][:2]
        best = 0.0
        for q in rng.uniform(lo, hi, size=(samples, len(ids))):
            for i, qi in zip(ids, q):
                self._p.resetJointState(self.robot_id, i, float(qi), physicsClientId=self.client_id)
            xy = self._p.getLinkState(self.robot_id, self.camera_link, physicsClientId=self.client_id)[4][:2]
            best = max(best, float(np.linalg.norm(np.array(xy) - base_xy)))

        for i, qi in zip(ids, start):
            self._p.resetJointState(self.robot_id, i, qi, physicsClientId=self.client_id)
        self._max_reach = best  # never changes, and callers ask repeatedly
        return best

    def object_size(self) -> np.ndarray:
        """Object bounding box extents (x, y, z)."""
        lo, hi = np.array(self._p.getAABB(self.obj_id, physicsClientId=self.client_id))
        return hi - lo

    def object_radius(self) -> float:
        """Radius of a circle enclosing the object's footprint."""
        return float(np.linalg.norm(self.object_size()[:2]) / 2)

    def framing_distance(self) -> float:
        """Distance at which the object just fills the camera's view."""
        return float(self.object_size().max()) / (2 * np.tan(np.radians(self.intrinsics.fov) / 2))

    def safe_orbit_radius(self) -> float:
        """Closest the camera can sit and still get a usable image."""
        # Too close and the object either clips through the near plane or overflows frame.
        return max(self.object_radius() + self.intrinsics.near, self.framing_distance())

    def orbit_shell(self) -> tuple[float, float]:
        """Inner and outer radius for camera viewpoints around the object."""
        r_min = self.safe_orbit_radius()
        d = float(np.linalg.norm(self.obj_pos[:2] - self.base_pose()[0][:2]))
        slack = max(0.0, self.max_reach() - d - r_min)  # what is left before the arm can't reach
        return r_min, r_min + slack * ORBIT_DEPTH_FRACTION

    # --- scene ---------------------------------------------------------------

    def _build_scene(self) -> None:
        """Load the ground plane and a tabletop covering the arm's workspace."""
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = self._p.loadURDF("plane.urdf")

        # Thin slab, not a block: a full-height table's side walls would sit in the camera's
        # line of sight from low viewpoints.
        r = self.max_reach()
        bx, by = self.base_pose()[0][:2]
        half = [r, r, 0.02]
        col = self._p.createCollisionShape(self._p.GEOM_BOX, halfExtents=half)
        vis = self._p.createVisualShape(self._p.GEOM_BOX, halfExtents=half, rgbaColor=[0.65, 0.45, 0.25, 1])
        self.table_id = self._p.createMultiBody(0, col, vis, [bx, by, self.table_top_z - 0.02])

    def _placement_xy(self) -> np.ndarray:
        """Where on the table to stand the object."""
        base = self.base_pose()[0]
        direction = np.asarray(PLACEMENT_DIRECTION, dtype=float)
        direction /= np.linalg.norm(direction)

        # Distance that leaves the whole orbit shell inside the arm's reach:
        #   d + r_min + (reach - d - r_min) * FRACTION <= reach
        r_min = self.safe_orbit_radius()
        d = (self.max_reach() - r_min) / (1.0 + ORBIT_DEPTH_FRACTION)
        return base[:2] + d * direction

    def _settle(self, steps: int = 2400, tol: float = 1e-3) -> None:
        """Run physics until the object stops moving."""
        for i in range(steps):
            self._p.stepSimulation()
            v, w = self._p.getBaseVelocity(self.obj_id, physicsClientId=self.client_id)
            if i > 100 and max(np.abs(v).max(), np.abs(w).max()) < tol:
                break

    def _place_object(self) -> None:
        """Drop the object, stand it at its computed spot, and record where it ended up."""
        # Dropped somewhere provisional first: _placement_xy needs the settled footprint.
        bx, by = self.base_pose()[0][:2]
        self.obj_id = self._p.loadURDF(
            ycb_urdf(self.ycb_object), [bx, by + 0.3, self.table_top_z + 0.15]
        )
        self._settle()

        x, y = self._placement_xy()
        pos, orn = self._p.getBasePositionAndOrientation(self.obj_id, physicsClientId=self.client_id)
        lo, _ = self._p.getAABB(self.obj_id, physicsClientId=self.client_id)
        self._p.resetBasePositionAndOrientation(
            self.obj_id, [x, y, pos[2] + (self.table_top_z - lo[2])], orn, physicsClientId=self.client_id
        )
        self._settle()
        self.obj_pos = np.array(
            self._p.getBasePositionAndOrientation(self.obj_id, physicsClientId=self.client_id)[0]
        )

    # --- arm motion -----------------------------------------------------------

    def _wait_for_arm_at_rest(self, max_extra_steps: int = 50, velocity_threshold: float = 0.005) -> None:
        """Step until every arm joint's velocity settles, or the budget runs out."""
        for _ in range(max_extra_steps):
            v = [self._p.getJointState(self.robot_id, i, physicsClientId=self.client_id)[1]
                 for i in self.arm_joint_indices]
            if max(abs(x) for x in v) < velocity_threshold:
                break
            self.step_simulation(self.per_step_iterations)

    def _snap_to_joint_targets(self, joint_targets: list[float]) -> None:
        """Teleport to the plan endpoint -- position control has steady-state droop."""
        for i, target in zip(self.arm_joint_indices, joint_targets):
            self._p.resetJointState(self.robot_id, i, float(target), physicsClientId=self.client_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("object", nargs="?", default=DEFAULT_YCB_OBJECT, choices=ycb_names())
    args = parser.parse_args()

    env = SimEnv(render=False, ycb_object=args.object)
    base = env.base_pose()[0]
    r_min, r_max = env.orbit_shell()
    d = np.linalg.norm(env.obj_pos[:2] - base[:2])
    print(f"base          {np.round(base, 3)}")
    print(f"table top     {env.table_top_z:.3f}")
    print(f"max reach     {env.max_reach():.3f}  (measured)")
    print(f"object        {np.round(env.obj_pos, 3)}   {d:.3f} from base")
    print(f"object radius {env.object_radius():.3f}")
    print(f"orbit shell   {r_min:.3f} .. {r_max:.3f}")
    print(f"far side      {d + r_max:.3f}  vs reach {env.max_reach():.3f}")
    env.close()
