from __future__ import annotations

import importlib.util
from pathlib import Path

import mujoco

from quadruped_demo.env import FOLLOW_CAMERA_NAME
from quadruped_demo.rl_env import Go1TrotRLEnv

WATCH_PPO_PATH = Path(__file__).resolve().parents[1] / "scripts" / "watch_ppo.py"
WATCH_PPO_SPEC = importlib.util.spec_from_file_location("watch_ppo", WATCH_PPO_PATH)
assert WATCH_PPO_SPEC is not None
assert WATCH_PPO_SPEC.loader is not None
watch_ppo = importlib.util.module_from_spec(WATCH_PPO_SPEC)
WATCH_PPO_SPEC.loader.exec_module(watch_ppo)


def test_watch_defaults_prefer_best_model_when_present(tmp_path) -> None:
    run_dir = tmp_path / "go1_ppo"
    run_dir.mkdir()
    final_model = run_dir / "model.zip"
    final_vecnormalize = run_dir / "vecnormalize.pkl"
    final_model.touch()
    final_vecnormalize.touch()

    model_path, vecnormalize_path = watch_ppo.default_policy_paths(run_dir)

    assert model_path == final_model
    assert vecnormalize_path == final_vecnormalize

    best_dir = run_dir / "best"
    best_dir.mkdir()
    best_model = best_dir / "best_model.zip"
    best_vecnormalize = best_dir / "best_vecnormalize.pkl"
    best_model.touch()
    best_vecnormalize.touch()

    model_path, vecnormalize_path = watch_ppo.default_policy_paths(run_dir)

    assert model_path == best_model
    assert vecnormalize_path == best_vecnormalize


def test_watch_defaults_use_latest_timestamped_run(tmp_path) -> None:
    run_root = tmp_path / "go1_ppo"
    old_run = run_root / "20260503-120000"
    latest_run = run_root / "20260504-120000"
    old_run.mkdir(parents=True)
    (old_run / "model.zip").touch()
    (old_run / "vecnormalize.pkl").touch()

    latest_best = latest_run / "best"
    latest_best.mkdir(parents=True)
    latest_model = latest_best / "best_model.zip"
    latest_vecnormalize = latest_best / "best_vecnormalize.pkl"
    latest_model.touch()
    latest_vecnormalize.touch()

    model_path, vecnormalize_path = watch_ppo.default_policy_paths(run_root)

    assert model_path == latest_model
    assert vecnormalize_path == latest_vecnormalize


def test_watch_defaults_fall_back_to_flat_run_dir(tmp_path) -> None:
    run_root = tmp_path / "go1_ppo"
    run_root.mkdir()
    final_model = run_root / "model.zip"
    final_vecnormalize = run_root / "vecnormalize.pkl"
    final_model.touch()
    final_vecnormalize.touch()

    model_path, vecnormalize_path = watch_ppo.default_policy_paths(run_root)

    assert model_path == final_model
    assert vecnormalize_path == final_vecnormalize


def test_watch_companion_vecnormalize_paths() -> None:
    assert watch_ppo.companion_vecnormalize_path(
        Path("run/best/best_model.zip")
    ) == Path("run/best/best_vecnormalize.pkl")
    assert watch_ppo.companion_vecnormalize_path(Path("run/model.zip")) == Path(
        "run/vecnormalize.pkl"
    )
    assert watch_ppo.companion_vecnormalize_path(
        Path("run/interrupted_model.zip")
    ) == Path("run/interrupted_vecnormalize.pkl")
    assert watch_ppo.companion_vecnormalize_path(
        Path("run/checkpoints/ppo_go1_50000_steps.zip")
    ) == Path("run/checkpoints/ppo_go1_vecnormalize_50000_steps.pkl")


def test_watch_configures_follow_camera_by_default() -> None:
    env = Go1TrotRLEnv()
    viewer_camera = mujoco.MjvCamera()

    try:
        camera_id = watch_ppo.configure_viewer_camera(
            viewer_camera,
            env,
            watch_ppo.FOLLOW_CAMERA_NAME,
        )
    finally:
        env.close()

    assert watch_ppo.FOLLOW_CAMERA_NAME == FOLLOW_CAMERA_NAME
    assert camera_id is None
    assert viewer_camera.type == int(mujoco.mjtCamera.mjCAMERA_TRACKING)
    assert viewer_camera.trackbodyid == env.physics.trunk_id


def test_watch_free_camera_leaves_viewer_camera_interactive() -> None:
    env = Go1TrotRLEnv()

    try:
        assert watch_ppo.viewer_camera_id(env.model, watch_ppo.FREE_CAMERA_NAME) is None
    finally:
        env.close()


def test_watch_can_select_named_fixed_model_camera() -> None:
    env = Go1TrotRLEnv()
    viewer_camera = mujoco.MjvCamera()
    fixed_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "tracking")

    try:
        camera_id = watch_ppo.configure_viewer_camera(
            viewer_camera,
            env,
            "tracking",
            fixed_id,
        )
    finally:
        env.close()

    assert camera_id == fixed_id
    assert viewer_camera.type == int(mujoco.mjtCamera.mjCAMERA_FIXED)
    assert viewer_camera.fixedcamid == fixed_id
