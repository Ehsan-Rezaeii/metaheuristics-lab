"""Small artifact helpers shared by reproducible serial experiment scripts."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np


def output_directory(path: str | Path) -> Path:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metadata(config: dict) -> dict:
    """Record source identity and environment without reading credentials."""
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unavailable", None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "source_revision": revision,
        "source_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            name: version(name)
            for name in ["metaheuristics-lab", "numpy", "matplotlib", "scikit-learn"]
        },
        "timing": "perf_counter wall seconds; serial; excludes plotting/data generation",
    }


def sample_std(values: list[float]) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) > 1 else None
