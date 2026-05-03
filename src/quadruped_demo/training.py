from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
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
    n_envs: int = 4
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
    checkpoint_freq: int = 50_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 5


@dataclass(frozen=True)
class PPOTrainingResult:
    run_dir: Path
    model_path: Path
    vecnormalize_path: Path | None
    best_model_path: Path | None
    best_vecnormalize_path: Path | None
    checkpoint_dir: Path


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


def default_eval_env_config(pushes: bool = False) -> Go1RLEnvConfig:
    return Go1RLEnvConfig(
        max_episode_time_s=8.0,
        command_velocity_x=0.6,
        command_velocity_range=(0.6, 0.6),
        randomize_command_velocity=False,
        action_scale=0.18,
        random_pushes=RandomPushConfig(
            enabled=pushes,
            min_interval_s=2.5,
            max_interval_s=2.5,
            min_duration_s=0.08,
            max_duration_s=0.08,
            min_force_n=20.0,
            max_force_n=20.0,
        ),
        reset_randomization=ResetRandomizationConfig(enabled=False),
    )


class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_path: Path, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is None:
            return True
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        vec_normalize.save(str(self.save_path))
        return True


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


def save_model_artifacts(
    model: PPO,
    model_path: Path,
    vecnormalize_path: Path | None,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    vec_normalize = model.get_vec_normalize_env()
    if vecnormalize_path is not None and vec_normalize is not None:
        vecnormalize_path.parent.mkdir(parents=True, exist_ok=True)
        vec_normalize.save(str(vecnormalize_path))


def _callback_freq(freq_timesteps: int, n_envs: int) -> int:
    return max(freq_timesteps // max(n_envs, 1), 1)


def build_training_callbacks(
    config: PPOTrainingConfig,
    env_config: Go1RLEnvConfig,
    run_dir: Path,
    eval_env_config: Go1RLEnvConfig | None = None,
) -> tuple[CallbackList | None, VecEnv | None, Path | None, Path | None, Path]:
    checkpoint_dir = run_dir / "checkpoints"
    callbacks: list[BaseCallback] = []

    if config.checkpoint_freq > 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=_callback_freq(config.checkpoint_freq, config.n_envs),
                save_path=str(checkpoint_dir),
                name_prefix="ppo_go1",
                save_vecnormalize=config.normalize,
                verbose=max(config.verbose - 1, 0),
            )
        )

    eval_env = None
    best_model_path = None
    best_vecnormalize_path = None
    if config.eval_freq > 0 and config.n_eval_episodes > 0:
        best_dir = run_dir / "best"
        best_model_path = best_dir / "best_model.zip"
        best_vecnormalize_path = best_dir / "best_vecnormalize.pkl" if config.normalize else None
        eval_env = build_vec_env(
            eval_env_config or env_config,
            seed=config.seed + 10_000,
            n_envs=1,
            monitor_dir=run_dir / "eval_monitor",
            normalize=config.normalize,
        )
        callbacks.append(
            EvalCallback(
                eval_env=eval_env,
                best_model_save_path=str(best_dir),
                log_path=str(run_dir / "eval"),
                eval_freq=_callback_freq(config.eval_freq, config.n_envs),
                n_eval_episodes=config.n_eval_episodes,
                deterministic=True,
                callback_on_new_best=SaveVecNormalizeCallback(best_vecnormalize_path)
                if best_vecnormalize_path is not None
                else None,
                verbose=max(config.verbose - 1, 0),
            )
        )

    if not callbacks:
        return None, eval_env, best_model_path, best_vecnormalize_path, checkpoint_dir
    return (
        CallbackList(callbacks),
        eval_env,
        best_model_path,
        best_vecnormalize_path,
        checkpoint_dir,
    )


def train_ppo(
    config: PPOTrainingConfig,
    env_config: Go1RLEnvConfig | None = None,
    eval_env_config: Go1RLEnvConfig | None = None,
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
    resolved_env_config = env_config or default_training_env_config()

    vec_env = build_vec_env(
        resolved_env_config,
        seed=config.seed,
        n_envs=config.n_envs,
        monitor_dir=monitor_dir,
        normalize=config.normalize,
    )
    callbacks, eval_env, best_model_path, best_vecnormalize_path, checkpoint_dir = (
        build_training_callbacks(
            config=config,
            env_config=resolved_env_config,
            eval_env_config=eval_env_config,
            run_dir=run_dir,
        )
    )
    model: PPO | None = None
    try:
        model = create_ppo_model(vec_env, config)
        model.learn(total_timesteps=config.total_timesteps, callback=callbacks)

        model_path = run_dir / "model.zip"
        vecnormalize_path = (
            run_dir / "vecnormalize.pkl" if isinstance(vec_env, VecNormalize) else None
        )
        save_model_artifacts(model, model_path, vecnormalize_path)

        return PPOTrainingResult(
            run_dir=run_dir,
            model_path=model_path,
            vecnormalize_path=vecnormalize_path,
            best_model_path=best_model_path
            if best_model_path and best_model_path.exists()
            else None,
            best_vecnormalize_path=best_vecnormalize_path
            if best_vecnormalize_path and best_vecnormalize_path.exists()
            else None,
            checkpoint_dir=checkpoint_dir,
        )
    except KeyboardInterrupt:
        if model is not None:
            model_path = run_dir / "interrupted_model.zip"
            vecnormalize_path = (
                run_dir / "interrupted_vecnormalize.pkl"
                if isinstance(vec_env, VecNormalize)
                else None
            )
            save_model_artifacts(model, model_path, vecnormalize_path)
        raise
    finally:
        if eval_env is not None:
            eval_env.close()
        vec_env.close()
