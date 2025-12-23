import datetime
import hashlib
import json
import os
from typing import Any, Dict


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default or {}


def build_run_dir(results_root: str, model: str, mode: str, config: Dict[str, Any] | None = None) -> str:
    ensure_dir(results_root)
    model_dir = os.path.join(results_root, model)
    mode_dir = os.path.join(model_dir, mode)
    ensure_dir(mode_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = ""
    if config:
        cfg_str = json.dumps(config, sort_keys=True)
        suffix = "_" + hashlib.sha1(cfg_str.encode()).hexdigest()[:8]
    run_dir = os.path.join(mode_dir, f"run_{timestamp}{suffix}")
    ensure_dir(run_dir)
    return run_dir


def status_path(run_dir: str) -> str:
    return os.path.join(run_dir, "status.json")

