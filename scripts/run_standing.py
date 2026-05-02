from __future__ import annotations

import argparse

from quadruped_demo.controllers import StandingController
from quadruped_demo.env import PushConfig, run_controller
from quadruped_demo.metrics import body_height


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Unitree Go1 standing controller.")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--pushes", action="store_true", help="Apply alternating lateral pushes.")
    args = parser.parse_args()

    env = run_controller(
        StandingController(),
        duration_s=args.duration,
        push=PushConfig() if args.pushes else None,
        render=not args.no_viewer,
    )
    print(f"final_time={env.data.time:.2f}s height={body_height(env.data.qpos):.3f}m")


if __name__ == "__main__":
    main()
