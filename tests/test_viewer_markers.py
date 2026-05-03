from __future__ import annotations

import mujoco
import numpy as np

from quadruped_demo.rl_env import Go1TrotRLEnv
from quadruped_demo.viewer_markers import draw_push_arrow


def test_draw_push_arrow_adds_one_red_user_geom() -> None:
    env = Go1TrotRLEnv()
    scene = mujoco.MjvScene(env.model, 4)

    drawn = draw_push_arrow(
        scene,
        target_position=np.array([0.0, 0.0, 0.3], dtype=np.float64),
        push_force=np.array([10.0, 0.0, 0.0], dtype=np.float64),
    )

    assert drawn
    assert scene.ngeom == 1
    assert scene.geoms[0].type == mujoco.mjtGeom.mjGEOM_ARROW
    assert np.allclose(scene.geoms[0].rgba, np.array([1.0, 0.0, 0.0, 0.9]))


def test_draw_push_arrow_clears_scene_when_push_is_inactive() -> None:
    env = Go1TrotRLEnv()
    scene = mujoco.MjvScene(env.model, 4)
    draw_push_arrow(
        scene,
        target_position=np.array([0.0, 0.0, 0.3], dtype=np.float64),
        push_force=np.array([10.0, 0.0, 0.0], dtype=np.float64),
    )

    drawn = draw_push_arrow(
        scene,
        target_position=np.array([0.0, 0.0, 0.3], dtype=np.float64),
        push_force=np.zeros(3, dtype=np.float64),
    )

    assert not drawn
    assert scene.ngeom == 0
