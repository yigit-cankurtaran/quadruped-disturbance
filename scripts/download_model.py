from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

# link for the repo to get the Unitree Go1 model we'll use
REPO = "https://github.com/google-deepmind/mujoco_menagerie"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEST = PROJECT_ROOT / "models" / "unitree_go1"


def replace_model(source: Path) -> None:
    """copy the downloaded model into a staging folder before replacing the current one.
    keeps the existing model intact if the copy fails, and restores it if the
    final swap cannot complete."""
    with tempfile.TemporaryDirectory(dir=DEST.parent) as tmp:
        staged = Path(tmp) / DEST.name
        backup = Path(tmp) / f"{DEST.name}.old"

        shutil.copytree(source, staged)
        if DEST.exists():
            DEST.rename(backup)
        try:
            staged.rename(DEST)
        except Exception:
            if backup.exists() and not DEST.exists():
                backup.rename(DEST)
            raise


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    # download the repo into temp folder, extract the model we need, delete temp folder
    with tempfile.TemporaryDirectory() as tmp:
        repo_dir = Path(tmp) / "mujoco_menagerie"
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, str(repo_dir)],
            check=True,
        )
        subprocess.run(["git", "sparse-checkout", "set", "unitree_go1"], cwd=repo_dir, check=True)
        replace_model(repo_dir / "unitree_go1")
    print(f"Downloaded Unitree Go1 model to {DEST}")


if __name__ == "__main__":
    main()
