from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize

from quadruped_demo.paths import RESULTS_DIR
from quadruped_demo.rl_env import (
    Go1RLEnvConfig,
    Go1TrotRLEnv,
    RandomPushConfig,
    ResetRandomizationConfig,
)


@dataclass(frozen=True)
class PPOTrainingConfig:
    total_timesteps: int = 200_000
    seed: int = 0
    n_envs: int = 1
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    device: str = "auto"
    normalize: bool = True
    verbose: int = 1
    run_name: str = "go1_ppo"


@dataclass(frozen=True)
class PPOTrainingResult:
    run_dir: Path
    model_path: Path
    vecnormalize_path: Path | None


def default_training_env_config(pushes: bool = False) -> Go1RLEnvConfig:
    return Go1RLEnvConfig(
        max_episode_time_s=8.0,
        command_velocity_x=0.6,
        command_velocity_range=(0.2, 1.0),
        randomize_command_velocity=True,
        action_scale=0.18,
        random_pushes=RandomPushConfig(
            enabled=pushes,
            min_interval_s=1.5,
            max_interval_s=4.0,
            min_duration_s=0.04,
            max_duration_s=0.12,
            min_force_n=5.0,
            max_force_n=30.0,
        ),
        reset_randomization=ResetRandomizationConfig(
            enabled=True,
            base_xy_range_m=0.02,
            base_yaw_range_rad=0.08,
            root_velocity_range=0.05,
            joint_position_range_rad=0.04,
            joint_velocity_range=0.25,
            randomize_gait_phase=True,
        ),
    )


def make_env(
    env_config: Go1RLEnvConfig,
    seed: int,
    rank: int = 0,
    monitor_dir: Path | None = None,
) -> Callable[[], Monitor]:
    def _init() -> Monitor:
        env = Go1TrotRLEnv(env_config)
        env_seed = seed + rank
        env.reset(seed=env_seed)
        env.action_space.seed(env_seed)
        monitor_path = None if monitor_dir is None else str(monitor_dir / f"env_{rank}")
        return Monitor(env, filename=monitor_path)

    return _init


def build_vec_env(
    env_config: Go1RLEnvConfig,
    seed: int,
    n_envs: int,
    monitor_dir: Path | None = None,
    normalize: bool = True,
) -> VecEnv:
    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")

    if monitor_dir is not None:
        monitor_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [
        make_env(env_config, seed=seed, rank=rank, monitor_dir=monitor_dir)
        for rank in range(n_envs)
    ]
    vec_env: VecEnv = DummyVecEnv(env_fns)
    if normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return vec_env


def load_eval_vec_env(
    env_config: Go1RLEnvConfig,
    seed: int,
    vecnormalize_path: Path | None,
) -> VecEnv:
    vec_env = build_vec_env(
        env_config,
        seed=seed,
        n_envs=1,
        monitor_dir=None,
        normalize=False,
    )
    if vecnormalize_path is not None:
        vec_env = VecNormalize.load(str(vecnormalize_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    return vec_env


def create_ppo_model(vec_env: VecEnv, config: PPOTrainingConfig) -> PPO:
    return PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        seed=config.seed,
        device=config.device,
        verbose=config.verbose,
    )


def train_ppo(
    config: PPOTrainingConfig,
    env_config: Go1RLEnvConfig | None = None,
    output_dir: Path | None = None,
) -> PPOTrainingResult:
    if config.total_timesteps < 1:
        raise ValueError("total_timesteps must be >= 1")
    if config.n_steps < 2:
        raise ValueError("n_steps must be >= 2")
    if config.batch_size < 2:
        raise ValueError("batch_size must be >= 2")

    run_dir = (output_dir or RESULTS_DIR / "ppo") / config.run_name
    monitor_dir = run_dir / "monitor"
    run_dir.mkdir(parents=True, exist_ok=True)

    vec_env = build_vec_env(
        env_config or default_training_env_config(),
        seed=config.seed,
        n_envs=config.n_envs,
        monitor_dir=monitor_dir,
        normalize=config.normalize,
    )
    try:
        model = create_ppo_model(vec_env, config)
        model.learn(total_timesteps=config.total_timesteps)

        model_path = run_dir / "model.zip"
        model.save(str(model_path))

        vecnormalize_path = None
        if isinstance(vec_env, VecNormalize):
            vecnormalize_path = run_dir / "vecnormalize.pkl"
            vec_env.save(str(vecnormalize_path))

        return PPOTrainingResult(
            run_dir=run_dir,
            model_path=model_path,
            vecnormalize_path=vecnormalize_path,
        )
    finally:
        vec_env.close()
