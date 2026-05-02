from __future__ import annotations

import numpy as np
from stable_baselines3.common.env_checker import check_env

from quadruped_demo.rl_env import Go1RLEnvConfig, Go1TrotRLEnv
from quadruped_demo.training import (
    PPOTrainingConfig,
    default_training_env_config,
    load_eval_vec_env,
    train_ppo,
)


def test_sb3_env_checker_accepts_go1_env() -> None:
    env = Go1TrotRLEnv(default_training_env_config(pushes=False))
    try:
        check_env(env, warn=True, skip_render_check=True)
    finally:
        env.close()


def test_tiny_ppo_training_saves_model_and_normalizer(tmp_path) -> None:
    env_config = Go1RLEnvConfig(max_episode_time_s=0.2)
    result = train_ppo(
        PPOTrainingConfig(
            total_timesteps=8,
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            verbose=0,
            run_name="ppo_smoke",
        ),
        env_config=env_config,
        output_dir=tmp_path,
    )

    assert result.model_path.exists()
    assert result.vecnormalize_path is not None
    assert result.vecnormalize_path.exists()

    eval_env = load_eval_vec_env(env_config, seed=0, vecnormalize_path=result.vecnormalize_path)
    try:
        obs = eval_env.reset()
        assert obs.shape == (1, eval_env.observation_space.shape[0])
        assert np.isfinite(obs).all()
    finally:
        eval_env.close()
