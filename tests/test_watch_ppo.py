from __future__ import annotations

import importlib.util
from pathlib import Path

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
