from __future__ import annotations

import argparse
from pathlib import Path

from quadruped_demo.paths import RESULTS_DIR
from quadruped_demo.training import (
    PPOTrainingConfig,
    default_eval_env_config,
    default_training_env_config,
    train_ppo,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO residual control for Unitree Go1.")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default="go1_ppo")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "ppo")
    parser.add_argument(
        "--vec-env",
        choices=("auto", "dummy", "subproc"),
        default="auto",
        help="Vector env backend. auto uses SubprocVecEnv when --n-envs > 1.",
    )
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument(
        "--pushes",
        action="store_true",
        help="Enable randomized push disturbances.",
    )
    args = parser.parse_args()

    result = train_ppo(
        PPOTrainingConfig(
            total_timesteps=args.total_timesteps,
            seed=args.seed,
            n_envs=args.n_envs,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            normalize=not args.no_normalize,
            run_name=args.run_name,
            vec_env_type=args.vec_env,
            checkpoint_freq=args.checkpoint_freq,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
        ),
        env_config=default_training_env_config(pushes=args.pushes),
        eval_env_config=default_eval_env_config(pushes=args.pushes),
        output_dir=args.output_dir,
    )

    print(f"saved_model={result.model_path}")
    if result.vecnormalize_path is not None:
        print(f"saved_vecnormalize={result.vecnormalize_path}")
    if result.best_model_path is not None:
        print(f"saved_best_model={result.best_model_path}")
    if result.best_vecnormalize_path is not None:
        print(f"saved_best_vecnormalize={result.best_vecnormalize_path}")
    print(f"checkpoint_dir={result.checkpoint_dir}")


if __name__ == "__main__":
    main()
