from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize

from quadruped_demo.training import default_training_env_config, load_eval_vec_env


def _go1_env_from_vec_env(vec_env: VecEnv):
    base_env = vec_env.venv if isinstance(vec_env, VecNormalize) else vec_env
    if not isinstance(base_env, DummyVecEnv):
        raise TypeError("viewer evaluation expects a single DummyVecEnv")
    monitor_env = base_env.envs[0]
    if not isinstance(monitor_env, Monitor):
        raise TypeError("viewer evaluation expects Monitor-wrapped env")
    return monitor_env.unwrapped


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO Go1 residual policy.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vecnormalize", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument(
        "--pushes",
        action="store_true",
        help="Enable randomized push disturbances.",
    )
    args = parser.parse_args()

    vec_env = load_eval_vec_env(
        default_training_env_config(pushes=args.pushes),
        seed=args.seed,
        vecnormalize_path=args.vecnormalize,
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
