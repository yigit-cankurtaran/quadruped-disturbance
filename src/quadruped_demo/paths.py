from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PACKAGE_ROOT / "models" / "unitree_go1"
MODEL_XML = MODEL_DIR / "scene.xml"
RESULTS_DIR = PACKAGE_ROOT / "results"


def require_model() -> Path:
    """Return the local Menagerie Go1 scene path, or fail with a useful message."""
    if not MODEL_XML.exists():
        raise FileNotFoundError(
            f"Missing MuJoCo model at {MODEL_XML}. Run `uv run python scripts/download_model.py`."
        )
    return MODEL_XML
