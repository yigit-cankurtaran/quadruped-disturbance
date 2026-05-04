from __future__ import annotations

import mujoco
import numpy as np

from quadruped_demo.gait import N_ACTUATORS
from quadruped_demo.rl_env import (
    Go1RLEnvConfig,
    Go1TrotRLEnv,
    RandomPushConfig,
    ResetRandomizationConfig,
)


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
    assert "x_position" in info
    assert "straight_line" in info["reward_terms"]
    assert "lateral_velocity_penalty" in info["reward_terms"]
    assert "heading" in info["reward_terms"]
    assert "yaw_rate_penalty" in info["reward_terms"]
    assert "lateral_displacement" in info
    assert "heading_error" in info
    assert "base_yaw" in info
    assert "yaw_rate" in info


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


def test_rl_env_straight_line_reward_penalizes_sideways_drift() -> None:
    env = Go1TrotRLEnv(Go1RLEnvConfig(decimation=1))
    action = np.zeros(N_ACTUATORS, dtype=np.float32)

    env.reset(seed=123)
    _, _, _, _, centered_info = env.step(action)

    env.reset(seed=123)
    env.data.qpos[1] += 0.4
    env.data.qvel[1] = 1.0
    mujoco.mj_forward(env.model, env.data)
    _, _, _, _, drifted_info = env.step(action)

    assert drifted_info["reward_terms"]["straight_line"] < centered_info["reward_terms"][
        "straight_line"
    ]
    assert drifted_info["reward_terms"]["lateral_velocity_penalty"] < centered_info[
        "reward_terms"
    ]["lateral_velocity_penalty"]


def test_rl_env_heading_reward_penalizes_yaw_error_and_yaw_rate() -> None:
    env = Go1TrotRLEnv(Go1RLEnvConfig(decimation=1))
    action = np.zeros(N_ACTUATORS, dtype=np.float32)

    env.reset(seed=123)
    _, _, _, _, centered_info = env.step(action)

    env.reset(seed=123)
    half_yaw = np.pi / 4.0
    env.data.qpos[3:7] = np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)])
    env.data.qvel[5] = 1.0
    mujoco.mj_forward(env.model, env.data)
    _, _, _, _, yawed_info = env.step(action)

    assert yawed_info["reward_terms"]["heading"] < centered_info["reward_terms"]["heading"]
    assert yawed_info["reward_terms"]["yaw_rate_penalty"] < centered_info["reward_terms"][
        "yaw_rate_penalty"
    ]


def test_rl_env_reset_randomization_is_seeded_and_in_bounds() -> None:
    config = Go1RLEnvConfig(
        command_velocity_range=(0.2, 0.4),
        randomize_command_velocity=True,
        reset_randomization=ResetRandomizationConfig(enabled=True),
    )
    env = Go1TrotRLEnv(config)

    obs_a, info_a = env.reset(seed=123)
    qpos_a = env.data.qpos.copy()
    qvel_a = env.data.qvel.copy()
    obs_b, info_b = env.reset(seed=123)

    assert np.allclose(obs_a, obs_b)
    assert np.allclose(qpos_a, env.data.qpos)
    assert np.allclose(qvel_a, env.data.qvel)
    assert 0.2 <= info_a["command_velocity_x"] <= 0.4
    assert info_a["command_velocity_x"] == info_b["command_velocity_x"]
    assert env.observation_space.contains(obs_a)


def test_rl_env_rejects_invalid_training_ranges() -> None:
    bad_command = Go1RLEnvConfig(command_velocity_range=(1.0, -1.0))
    bad_push = Go1RLEnvConfig(
        random_pushes=RandomPushConfig(enabled=True, min_force_n=10.0, max_force_n=5.0)
    )
    bad_reset = Go1RLEnvConfig(
        reset_randomization=ResetRandomizationConfig(enabled=True, joint_velocity_range=-0.1)
    )

    for config in (bad_command, bad_push, bad_reset):
        try:
            Go1TrotRLEnv(config)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid config should raise ValueError")
