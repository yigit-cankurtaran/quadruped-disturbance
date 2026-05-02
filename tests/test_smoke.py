from __future__ import annotations

import numpy as np

from quadruped_demo.controllers import StandingController, TrotController
from quadruped_demo.env import Go1Env
from quadruped_demo.gait import N_ACTUATORS
from quadruped_demo.metrics import is_upright


def test_model_loads_and_home_stands() -> None:
    env = Go1Env()
    controller = StandingController()
    for _ in range(50):
        obs = env.observation()
        ctrl = controller(float(env.data.time), obs)
        assert ctrl.shape == (N_ACTUATORS,)
        env.step(ctrl)
    assert is_upright(env.data.qpos)


def test_trot_controller_outputs_finite_clipped_targets() -> None:
    env = Go1Env()
    ctrl = TrotController()(0.25, env.observation())
    assert ctrl.shape == (N_ACTUATORS,)
    assert np.all(np.isfinite(ctrl))
    assert np.all(ctrl >= env.model.actuator_ctrlrange[:, 0])
    assert np.all(ctrl <= env.model.actuator_ctrlrange[:, 1])
