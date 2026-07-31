from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return a mutable dictionary."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the root of {config_path}")
    return data


def require_keys(mapping: dict[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"Missing configuration keys: {', '.join(missing)}")
