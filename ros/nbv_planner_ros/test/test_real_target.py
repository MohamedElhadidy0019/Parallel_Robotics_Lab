"""Real-robot plumbing that can be checked without a robot: config adaptation and the target rules."""

import json

import pytest

from nbv_planner_ros.robot_model import adapt_robot_config, camera_link_from_handeye, urdf_names
from nbv_planner_ros.targets import controller_name, validate_mode, validate_target

URDF = """<robot name="steve">
  <link name="base_link"/><link name="ur5ebase_link"/><link name="ur5etool0"/><link name="camera_link"/>
  <joint name="ur5eshoulder_pan_joint" type="revolute"><parent link="base_link"/><child link="ur5ebase_link"/>
    <limit lower="-6.28" upper="6.28" effort="1" velocity="1"/></joint>
  <joint name="ur5etool_joint" type="fixed"><parent link="ur5ebase_link"/><child link="ur5etool0"/></joint>
  <joint name="pan_joint" type="fixed"><parent link="base_link"/><child link="camera_link"/></joint>
</robot>"""

CAMERA = "nbv_camera_optical_frame"


def packaged():
    return {"robot_cfg": {"kinematics": {
        "base_link": "base_link",
        "collision_spheres": {"tool0": [{"center": [0, 0, 0], "radius": 0.02}],
                              "dummy_camera_link": [{"center": [0, 0, 0], "radius": 0.035}]},
        "self_collision_ignore": {"dummy_camera_link": ["tool0"]},
        "self_collision_buffer": {"dummy_camera_link": 0.01},
    }}}


@pytest.fixture
def urdf(tmp_path):
    path = tmp_path / "handeye.json"
    path.write_text(json.dumps({"translation_m": [0.035, -0.004, 0.066], "quaternion_xyzw": [0, 0, 1, 0]}))
    return camera_link_from_handeye(URDF, str(path), "ur5etool0", CAMERA)


def test_handeye_injects_camera_link(urdf):
    assert CAMERA in urdf_names(urdf)[0]


def test_default_alias_puts_camera_spheres_on_the_pan_tilt_camera(urdf):
    config, *_ = adapt_robot_config(packaged(), urdf, CAMERA)
    spheres = config["robot_cfg"]["kinematics"]["collision_spheres"]
    assert "camera_link" in spheres and CAMERA not in spheres


def test_real_alias_and_overrides_model_the_wrist_camera(urdf):
    override = [{"center": [0.07, 0, -0.012], "radius": 0.03}, {"center": [-0.07, 0, -0.012], "radius": 0.03}]
    config, *_ = adapt_robot_config(packaged(), urdf, CAMERA,
                                    link_aliases={"dummy_camera_link": CAMERA},
                                    sphere_overrides={"dummy_camera_link": override})
    kinematics = config["robot_cfg"]["kinematics"]
    assert kinematics["collision_spheres"][CAMERA] == override
    assert "camera_link" not in kinematics["collision_spheres"]
    assert kinematics["self_collision_ignore"][CAMERA] == ["ur5etool0"]


def test_only_scan_mode_runs_on_the_real_robot():
    validate_mode("real", "scan")
    for mode in ("cad", "both"):
        with pytest.raises(RuntimeError):
            validate_mode("real", mode)
    validate_mode("sim", "cad")


def test_unknown_target_is_refused():
    with pytest.raises(RuntimeError):
        validate_target("mock")


def test_controller_name_from_action():
    assert controller_name("/scaled_joint_trajectory_controller/follow_joint_trajectory") == \
        "scaled_joint_trajectory_controller"
