"""Explicit physical table entity for simulation world construction."""

from typing import Any
import numpy as np

from sim.config import TableConfig


class Table:
    """Explicit inspection workstation/table in the PyBullet simulation."""

    def __init__(self, p_module: Any, config: TableConfig | None = None) -> None:
        self._p = p_module
        self.config = config or TableConfig()
        self.table_id = self._build()

    def _build(self) -> int:
        """Construct the physical table with tabletop slab and legs."""
        cfg = self.config
        top_half = [cfg.length_x / 2.0, cfg.width_y / 2.0, cfg.thickness_z / 2.0]
        top_center_z = cfg.surface_elevation - top_half[2]

        # Top slab
        top_col = self._p.createCollisionShape(self._p.GEOM_BOX, halfExtents=top_half)
        top_vis = self._p.createVisualShape(
            self._p.GEOM_BOX, halfExtents=top_half, rgbaColor=list(cfg.color)
        )

        table_id = self._p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=top_col,
            baseVisualShapeIndex=top_vis,
            basePosition=[cfg.center_xy[0], cfg.center_xy[1], top_center_z],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
        )
        return table_id

    @property
    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Exact 3D bounding box (min_bounds, max_bounds) queried from physics."""
        lo, hi = self._p.getAABB(self.table_id)
        return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)

    @property
    def surface_z(self) -> float:
        """Exact physical top surface elevation of the table."""
        _, hi = self.aabb
        return float(hi[2])

    @property
    def center_xy(self) -> np.ndarray:
        """Geometric (x, y) center of the table surface."""
        lo, hi = self.aabb
        return (lo[:2] + hi[:2]) / 2.0

    @property
    def half_extents(self) -> np.ndarray:
        """3D half-extents of the table obstacle bounding box."""
        lo, hi = self.aabb
        return (hi - lo) / 2.0
