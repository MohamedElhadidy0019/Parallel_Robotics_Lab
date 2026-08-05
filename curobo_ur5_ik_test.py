"""One-off spike: confirm CuRobo v0.7.8 can load our actual ur5_robotiq_85 URDF and
solve IK against dummy_camera_link - the pose this project actually cares about, since
grasping is out of scope (per the supervisor's scope update) and the gripper/tool links
are irrelevant to camera-viewpoint reachability. Mirrors curobo_legacy/examples/ik_example.py's
demo_basic_ik(), pointed at our real robot instead of the stock ur10e example.

Run with: conda run -n curobo_test python curobo_ur5_ik_test.py
"""
import time

import torch
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

URDF_PATH = (
    "/home/mohamed/repos/Parallel_Robotics_Lab/third_party/shelf_gym_repo/"
    "shelf_gym/meshes/urdf/ur5_robotiq_85.urdf"
)
BASE_LINK = "base_link"
EE_LINK = "dummy_camera_link"


def main():
    tensor_args = TensorDeviceType()

    robot_cfg = RobotConfig.from_basic(URDF_PATH, BASE_LINK, EE_LINK, tensor_args)

    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        None,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=20,
        self_collision_check=False,
        self_collision_opt=False,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)

    for _ in range(10):
        q_sample = ik_solver.sample_configs(100)
        kin_state = ik_solver.fk(q_sample)
        goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)

        st_time = time.time()
        result = ik_solver.solve_batch(goal)
        torch.cuda.synchronize()
        print(
            "Success, Solve Time(s), hz ",
            torch.count_nonzero(result.success).item() / len(q_sample),
            result.solve_time,
            q_sample.shape[0] / (time.time() - st_time),
            torch.mean(result.position_error),
            torch.mean(result.rotation_error),
        )


if __name__ == "__main__":
    main()
