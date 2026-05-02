from __future__ import annotations

import mujoco
import numpy as np

from quadruped_demo.gait import N_ACTUATORS
from quadruped_demo.rl_env import Go1RLEnvConfig, Go1TrotRLEnv, RandomPushConfig


def test_rl_env_reset_observation_is_in_space() -> None:
    env = Go1TrotRLEnv()
    obs, info = env.reset(seed=123)

    assert env.observation_space.contains(obs)
    assert obs.shape == env.observation_space.shape
    assert info["time"] == 0.0
    assert np.isfinite(obs).all()


def test_rl_env_step_returns_gymnasium_tuple() -> None:
    env = Go1TrotRLEnv(Go1RLEnvConfig(max_episode_time_s=0.2))
    env.reset(seed=123)

    result = env.step(np.zeros(N_ACTUATORS, dtype=np.float32))

    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "reward_terms" in info
    assert np.isfinite(reward)


def test_rl_env_random_actions_short_rollout_do_not_crash() -> None:
    env = Go1TrotRLEnv(Go1RLEnvConfig(max_episode_time_s=0.25))
    env.reset(seed=123)
    rng = np.random.default_rng(123)

    for _ in range(20):
        action = rng.uniform(-1.0, 1.0, size=N_ACTUATORS).astype(np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        if terminated or truncated:
            break


def test_rl_env_terminates_when_base_height_is_invalid() -> None:
    env = Go1TrotRLEnv()
    env.reset(seed=123)
    env.data.qpos[2] = 0.05
    mujoco.mj_forward(env.model, env.data)

    _, reward, terminated, truncated, _ = env.step(np.zeros(N_ACTUATORS, dtype=np.float32))

    assert terminated
    assert not truncated
    assert np.isfinite(reward)


def test_rl_env_random_pushes_apply_external_force() -> None:
    env = Go1TrotRLEnv(
        Go1RLEnvConfig(
            random_pushes=RandomPushConfig(
                enabled=True,
                min_interval_s=0.0,
                max_interval_s=0.0,
                min_duration_s=0.02,
                max_duration_s=0.02,
                min_force_n=5.0,
                max_force_n=5.0,
            )
        )
    )
    env.reset(seed=123)

    _, _, _, _, info = env.step(np.zeros(N_ACTUATORS, dtype=np.float32))

    assert np.isclose(np.linalg.norm(info["push_force"]), 5.0)
