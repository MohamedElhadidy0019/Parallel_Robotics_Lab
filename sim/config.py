"""Configuration parameters for the Steve mobile manipulator simulation environment."""

from dataclasses import dataclass, field
import os
from typing import Sequence

from nbv_planner.camera import CameraIntrinsics


@dataclass
class TableConfig:
    """Explicit parameters defining the physical inspection table."""
    center_xy: tuple[float, float] = (0.785, 0.0)
    surface_elevation: float = 0.75
    length_x: float = 0.60
    width_y: float = 0.70
    thickness_z: float = 0.04
    leg_radius: float = 0.025
    color: tuple[float, float, float, float] = (0.72, 0.62, 0.52, 1.0)
    leg_color: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 1.0)


@dataclass
class SteveRobotConfig:
    """Placement and link definitions for Steve mobile manipulator."""
    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_orn_euler: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Arm joint names matching UR5 standard
    arm_joint_names: tuple[str, ...] = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )

    # Home joint configuration (retracted safely above base, facing forward)
    initial_arm_joints: tuple[float, ...] = (
        0.0,
        -1.570796,
        1.570796,
        -1.570796,
        -1.570796,
        0.0,
    )

    # Link identifiers
    camera_link_name: str = "dummy_camera_link"
    tool_tip_link_name: str = "tool0"

    # Collision buffer added to robot link spheres for obstacle avoidance in cuRobo
    collision_sphere_buffer: float = 0.02


@dataclass
class SimConfig:
    """Root simulation configuration."""
    table: TableConfig = field(default_factory=TableConfig)
    robot: SteveRobotConfig = field(default_factory=SteveRobotConfig)
    intrinsics: CameraIntrinsics = field(default_factory=CameraIntrinsics)
    hz: int = 240
    gravity: float = -9.81
    settle_steps: int = 400
    settle_velocity_tol: float = 1e-3
    placement_offset_from_center: tuple[float, float] = (0.0, 0.0)
