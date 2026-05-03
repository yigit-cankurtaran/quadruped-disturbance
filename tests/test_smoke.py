from __future__ import annotations

import numpy as np

from quadruped_demo.controllers import StandingController, TrotController
from quadruped_demo.env import Go1Env
from quadruped_demo.gait import N_ACTUATORS
from quadruped_demo.metrics import base_roll_pitch, base_yaw, is_upright, wrap_angle


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


def test_upright_metric_rejects_low_or_tilted_base() -> None:
    env = Go1Env()
    qpos = env.data.qpos.copy()

    assert is_upright(qpos)

    low_qpos = qpos.copy()
    low_qpos[2] = 0.05
    assert not is_upright(low_qpos)

    tilted_qpos = qpos.copy()
    tilted_qpos[3:7] = np.array([0.0, 1.0, 0.0, 0.0])
    assert not is_upright(tilted_qpos)


def test_base_roll_pitch_from_identity_quaternion() -> None:
    env = Go1Env()
    roll, pitch = base_roll_pitch(env.data.qpos)
    assert np.isclose(roll, 0.0)
    assert np.isclose(pitch, 0.0)


def test_base_yaw_from_identity_and_yaw_quaternion() -> None:
    env = Go1Env()
    qpos = env.data.qpos.copy()

    assert np.isclose(base_yaw(qpos), 0.0)

    half_yaw = np.pi / 4.0
    qpos[3:7] = np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)])
    assert np.isclose(base_yaw(qpos), np.pi / 2.0)


def test_wrap_angle_maps_to_signed_pi_range() -> None:
    assert np.isclose(wrap_angle(0.0), 0.0)
    assert np.isclose(wrap_angle(3.0 * np.pi / 2.0), -np.pi / 2.0)
    assert np.isclose(wrap_angle(-3.0 * np.pi / 2.0), np.pi / 2.0)
