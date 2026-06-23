import os
import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation
from shelf_gym.environments.ur5_environment import RobotEnv
from shelf_gym.utils.camera_utils import Camera


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ASSET_PATH = os.path.join(PROJECT_ROOT, "third_party/shelf_gym_repo/shelf_gym/meshes/urdf")
YCB_OBJECT = "YcbMustardBottle"

TABLE_TOP_Z = 0.90  # matches robot base z; table half-height=0.45, center at z=0.45


class NBVEnv2(RobotEnv):
    def __init__(self, render=True):
        super().__init__(render=render, show_vis=False)
        self._build_scene()
        self._place_object()
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)
        self.camera = Camera(width=640, height=480)

    def _build_scene(self):
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

    def _place_object(self):
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

    def _compute_safe_orbit_radius(self, safety_margin=0.12):
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

    def _lookat_quaternion(self, eye, target):
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

    def orbit_and_capture(self, n_views=8, height=1.2, save_dir="captures2"):
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


if __name__ == "__main__":
    import time
    env = NBVEnv2(render=True)
    time.sleep(2)
    env.orbit_and_capture(n_views=8, height=1.2, save_dir="captures")
    while True:
        env.step_simulation(env.per_step_iterations)
