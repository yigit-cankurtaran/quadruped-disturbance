from __future__ import annotations

import mujoco
import numpy as np


def draw_push_arrow(
    user_scene: mujoco.MjvScene | None,
    target_position: np.ndarray,
    push_force: np.ndarray,
    force_scale: float = 0.006,
) -> bool:
    """Draw a red arrow into the robot at the active push location."""

    if user_scene is None:
        return False

    user_scene.ngeom = 0
    force = np.asarray(push_force, dtype=np.float64)
    force_norm = float(np.linalg.norm(force))
    if force_norm < 1e-6:
        return False

    if user_scene.ngeom >= user_scene.maxgeom:
        return False

    direction = force / force_norm
    arrow_length = float(np.clip(force_norm * force_scale, 0.12, 0.45))
    end = np.asarray(target_position, dtype=np.float64).copy()
    start = end - direction * arrow_length

    geom = user_scene.geoms[user_scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.array([1.0, 0.0, 0.0, 0.9], dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.035,
        start,
        end,
    )
    user_scene.ngeom += 1
    return True
