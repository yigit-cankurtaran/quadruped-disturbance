from __future__ import annotations

import argparse

from quadruped_demo.controllers import TrotController
from quadruped_demo.env import PushConfig, run_controller
from quadruped_demo.metrics import body_height, forward_distance


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple open-loop Unitree Go1 trot.")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--pushes", action="store_true", help="Apply alternating lateral pushes.")
    args = parser.parse_args()

    env = run_controller(
        TrotController(),
        duration_s=args.duration,
        push=PushConfig(force_n=25.0) if args.pushes else None,
        render=not args.no_viewer,
    )
    print(
        f"final_time={env.data.time:.2f}s "
        f"x={forward_distance(env.data.qpos):.3f}m height={body_height(env.data.qpos):.3f}m"
    )


if __name__ == "__main__":
    main()
