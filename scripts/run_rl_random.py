from __future__ import annotations

import argparse
import time

from quadruped_demo.rl_env import Go1RLEnvConfig, Go1TrotRLEnv, RandomPushConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Go1 residual-RL environment with random actions."
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--pushes", action="store_true", help="Apply randomized push disturbances.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = Go1TrotRLEnv(
        Go1RLEnvConfig(
            max_episode_time_s=args.duration,
            random_pushes=RandomPushConfig(enabled=args.pushes),
        )
    )
    env.action_space.seed(args.seed)
    _, info = env.reset(seed=args.seed)

    viewer = None
    if not args.no_viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    total_reward = 0.0
    last_reward = 0.0
    total_time = 0.0
    episodes = 1
    start = time.time()
    try:
        while total_time < args.duration:
            action = env.action_space.sample()
            _, reward, terminated, truncated, info = env.step(action)
            last_reward = reward
            total_reward += reward
            total_time += env.dt

            if viewer is not None:
                viewer.sync()
                elapsed = time.time() - start
                if total_time > elapsed:
                    time.sleep(total_time - elapsed)

            if (terminated or truncated) and total_time < args.duration:
                episodes += 1
                _, info = env.reset(seed=args.seed + episodes)
    finally:
        if viewer is not None:
            viewer.close()
        env.close()

    print(
        f"final_time={total_time:.2f}s episodes={episodes} "
        f"x={env.data.qpos[0]:.3f}m height={info['base_height']:.3f}m "
        f"last_reward={last_reward:.3f} total_reward={total_reward:.3f}"
    )


if __name__ == "__main__":
    main()
