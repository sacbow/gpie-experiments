from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    candidate = Path(path).expanduser()

    if candidate.is_absolute():
        return candidate.resolve()

    return (PROJECT_ROOT / candidate).resolve()


def resolve_config_path(path: str | Path) -> Path:
    """Resolve a config path."""
    candidate = Path(path).expanduser()

    if candidate.is_absolute():
        return candidate.resolve()

    project_candidate = PROJECT_ROOT / candidate
    config_candidate = CONFIGS_DIR / candidate

    for item in (project_candidate, config_candidate):
        if item.is_file():
            return item.resolve()

    if candidate.parts and candidate.parts[0] == "configs":
        return project_candidate.resolve()

    return config_candidate.resolve()


def ensure_directory(path: str | Path) -> Path:
    """Create a directory and return its resolved path."""
    directory = resolve_project_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _slugify(value: str) -> str:
    """Convert a name into a filesystem-safe identifier."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    normalized = normalized.strip("._-")

    if not normalized:
        raise ValueError(
            "Name must contain at least one filesystem-safe character."
        )

    return normalized


def create_run_directory(
    *,
    problem: str,
    experiment_name: str,
    output_root: str | Path = RESULTS_DIR,
    run_id: str | None = None,
) -> Path:
    """Create a unique result directory for one experiment run."""
    root = resolve_project_path(output_root)

    experiment_dir = (
        root
        / _slugify(problem)
        / _slugify(experiment_name)
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    base_run_id = (
        run_id
        or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    )
    base_run_id = _slugify(base_run_id)

    run_dir = experiment_dir / base_run_id

    suffix = 1
    while run_dir.exists():
        run_dir = experiment_dir / f"{base_run_id}-{suffix:02d}"
        suffix += 1

    run_dir.mkdir()
    return run_dir.resolve()