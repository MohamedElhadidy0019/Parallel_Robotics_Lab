"""Only the converted live URDF and measured joint poses are logged."""

import os
import sys

import rerun as rr
import rerun.blueprint as rrb


class RobotViewer:
    def __init__(self, robot, url):
        self.robot = robot
        os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
        rr.init("steve_start_pose", spawn=not bool(url))
        if url:
            rr.connect_grpc(url)
        rr.send_blueprint(rrb.Spatial3DView(name="Live robot", origin="world"))
        self.tree = rr.urdf.UrdfTree.from_file_path(robot.viewer_urdf_path, entity_path_prefix="world")
        self.tree.log_urdf_to_recording()
        self.update()

    def update(self):
        root = self.tree.root_link().name
        position, quaternion = self.robot.link_pose(root)
        rr.log("world/robot_pose", rr.Transform3D(
            translation=position, quaternion=quaternion,
            parent_frame="tf#/world", child_frame=root))
        values = self.robot.joint_values()
        for joint in self.tree.joints():
            if joint.name in values:
                rr.log(f"world/robot/joints/{joint.name}", joint.compute_transform(float(values[joint.name])))
