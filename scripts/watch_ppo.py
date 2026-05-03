from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize

from quadruped_demo.paths import RESULTS_DIR
from quadruped_demo.training import default_training_env_config, load_eval_vec_env
from quadruped_demo.viewer_markers import draw_push_arrow


def _go1_env_from_vec_env(vec_env: VecEnv):
    base_env = vec_env.venv if isinstance(vec_env, VecNormalize) else vec_env
    if not isinstance(base_env, DummyVecEnv):
        raise TypeError("watch script expects a single DummyVecEnv")
    monitor_env = base_env.envs[0]
    if not isinstance(monitor_env, Monitor):
        raise TypeError("watch script expects Monitor-wrapped env")
    return monitor_env.unwrapped


def default_policy_paths(run_dir: Path) -> tuple[Path, Path | None]:
    best_model = run_dir / "best" / "best_model.zip"
    best_vecnormalize = run_dir / "best" / "best_vecnormalize.pkl"
    if best_model.exists():
        return best_model, best_vecnormalize
    return run_dir / "model.zip", run_dir / "vecnormalize.pkl"


def companion_vecnormalize_path(model_path: Path) -> Path | None:
    if model_path.name == "best_model.zip":
        return model_path.parent / "best_vecnormalize.pkl"
    if model_path.name == "interrupted_model.zip":
        return model_path.parent / "interrupted_vecnormalize.pkl"
    if model_path.name == "model.zip":
        return model_path.parent / "vecnormalize.pkl"

    stem = model_path.stem
    if stem.endswith("_steps"):
        before_steps = stem.removesuffix("_steps")
        try:
            prefix, step = before_steps.rsplit("_", 1)
        except ValueError:
            return None
        return model_path.parent / f"{prefix}_vecnormalize_{step}_steps.pkl"

    return None


def main() -> None:
    default_run_dir = RESULTS_DIR / "ppo" / "go1_ppo"
    default_model_path, _ = default_policy_paths(default_run_dir)

    parser = argparse.ArgumentParser(description="Watch a trained PPO Go1 residual policy.")
    parser.add_argument("--model", type=Path, default=default_model_path)
    parser.add_argument("--vecnormalize", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-viewer", action="store_true", help="Run headlessly for smoke tests.")
    parser.add_argument(
        "--pushes",
        action="store_true",
        help="Enable randomized push disturbances.",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(
            f"Missing trained model at {args.model}. "
            "Run `uv run python scripts/train_ppo.py` first."
        )
    vecnormalize_path = args.vecnormalize or companion_vecnormalize_path(args.model)
    if vecnormalize_path is not None and not vecnormalize_path.exists():
        vecnormalize_path = None

    vec_env = load_eval_vec_env(
        default_training_env_config(pushes=args.pushes),
        seed=args.seed,
        vecnormalize_path=vecnormalize_path,
    )
    model = PPO.load(str(args.model), env=vec_env, device=args.device)
    obs = vec_env.reset()
    go1_env = _go1_env_from_vec_env(vec_env)

    viewer = None
    if not args.no_viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(go1_env.model, go1_env.data)

    total_reward = 0.0
    total_time = 0.0
    episodes = 1
    start = time.time()
    try:
        while total_time < args.duration:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            total_reward += float(rewards[0])
            total_time += go1_env.dt

            if viewer is not None:
                with viewer.lock():
                    draw_push_arrow(
                        viewer.user_scn,
                        go1_env.data.xipos[go1_env.physics.trunk_id],
                        infos[0]["push_force"],
                    )
                viewer.sync()
                elapsed = time.time() - start
                if total_time > elapsed:
                    time.sleep(total_time - elapsed)

            if bool(dones[0]) and total_time < args.duration:
                episodes += 1
    finally:
        if viewer is not None:
            viewer.close()
        vec_env.close()

    info = infos[0]
    print(
        f"final_time={total_time:.2f}s episodes={episodes} "
        f"x={go1_env.data.qpos[0]:.3f}m height={info['base_height']:.3f}m "
        f"total_reward={total_reward:.3f}"
    )


if __name__ == "__main__":
    main()
