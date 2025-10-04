from __future__ import annotations
import yaml
from typing import Any, Dict


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

