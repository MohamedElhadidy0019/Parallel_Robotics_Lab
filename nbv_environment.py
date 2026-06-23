import os
import pybullet as p
import pybullet_data
from shelf_gym.environments.ur5_environment import RobotEnv


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ASSET_PATH = os.path.join(PROJECT_ROOT, "third_party/shelf_gym_repo/shelf_gym/meshes/urdf")
YCB_OBJECT = "YcbMustardBottle"


class NBVEnv(RobotEnv):
    def __init__(self, render=True):
        super().__init__(render=render, show_vis=False)
        self._build_scene()
        self._place_object()
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)

    def _build_scene(self):
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = self._p.loadURDF("plane.urdf")
        self.table_id = self._p.loadURDF(
            os.path.join(ASSET_PATH, "environment/table.urdf"),
            [0.0, -0.15, 0.45],
            self._p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )

    def _place_object(self):
        # pedestal
        col = self._p.createCollisionShape(self._p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.47])
        vis = self._p.createVisualShape(self._p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.47], rgbaColor=[0.6, 0.6, 0.6, 1])
        self._p.createMultiBody(0, col, vis, [0.0, 0.7, 0.47])

        # object on top of pedestal
        obj_path = os.path.join(ASSET_PATH, f"ycb_objects/{YCB_OBJECT}/model_textureless.urdf")
        self.obj_id = self._p.loadURDF(obj_path, [0.0, 0.7, 1.1], self._p.getQuaternionFromEuler([0, 0, 0]))
        for _ in range(240):
            self._p.stepSimulation()


if __name__ == "__main__":
    env = NBVEnv(render=True)
    while True:
        env.step_simulation(env.per_step_iterations)
