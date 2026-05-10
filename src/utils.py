from __future__ import annotations

import csv
import json
import platform
import pickle
import random
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import torch

from models import MLP, MLPConfig

TIMEZONE = ZoneInfo("Europe/Vienna")

def get_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_model(model_cfg: MLPConfig, seed: int, dtype: torch.dtype) -> MLP:
    # Important: seed before constructing model to ensure identical init across optimizers
    set_global_seed(seed)
    model = MLP(model_cfg)
    # Ensure model parameters are in desired dtype
    return model.to(dtype=dtype)


def save_experiment_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(result, handle)


def load_experiment_result(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def make_sweep_dir(results_root: Path, problem_name: str, sweep_name: str | None = None) -> Path:
    timestamp = datetime.now(TIMEZONE).strftime("%Y%m%d-%H%M%S")
    suffix = f"_{sweep_name}" if sweep_name else ""
    sweep_dir = results_root / problem_name / f"sweep_{timestamp}{suffix}"
    sweep_dir.mkdir(parents=True, exist_ok=False)
    return sweep_dir


def make_seed_dir(sweep_dir: Path, seed: int) -> Path:
    seed_dir = sweep_dir / f"seed_{seed:07d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    return seed_dir


def make_experiment_dir(sweep_dir: Path, experiment_name: str) -> Path:
    experiment_dir = sweep_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def make_run_dir(experiment_dir: Path, seed: int) -> Path:
    run_dir = experiment_dir / f"seed_{seed:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_run_id(experiment_name: str, seed: int) -> str:
    return f"{experiment_name}_seed_{seed:07d}"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    # Covers foreign dtype objects (for example from JAX/NumPy scalar internals)
    if value.__class__.__name__ == "dtype":
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def json_default(value: Any) -> Any:
    return _json_default(value)


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    return path


def append_jsonl(path: Path, row: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=json_default))
        handle.write("\n")
    return path


def run_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "config": run_dir / "config.json",
        "metrics": run_dir / "metrics.jsonl",
        "summary": run_dir / "summary.json",
        "result": run_dir / "result.pkl",
    }


def get_git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def get_system_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(device)
    return info


def save_manifest(
    sweep_dir: Path,
    *,
    problem_name: str,
    seeds: list[int],
    experiment_names: list[str],
    shared_config: dict,
) -> Path:
    manifest = {
        "problem_name": problem_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seeds": seeds,
        "experiment_names": experiment_names,
        "shared_config": shared_config,
    }
    path = sweep_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=json_default))
    return path


def rebuild_sweep_summary_csv(sweep_dir: Path) -> Path:
    summary_paths = sorted(sweep_dir.glob("*/seed_*/summary.json"))
    rows: list[dict[str, Any]] = []
    base_fields = [
        "run_id",
        "sweep_name",
        "problem_name",
        "experiment_name",
        "seed",
        "status",
        "stop_reason",
        "wall_time_s",
        "completed_steps",
        "num_parameters",
        "peak_gpu_mem_mb",
        "primary_metric",
        "best_step",
    ]
    row_metadata_keys = {
        "run_id",
        "sweep_name",
        "problem_name",
        "experiment_name",
        "seed",
        "step",
        "wall_time_s",
    }
    best_metadata_keys = {
        "selection_metric",
        "best_step",
        "best_wall_time_s",
    }

    for path in summary_paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        final_metrics = summary.get("final_metrics") or {}
        best_metrics = summary.get("best_metrics") or {}
        resources = summary.get("resources", {})
        model = summary.get("model", {})

        row = {
            "run_id": summary.get("run_id"),
            "sweep_name": summary.get("sweep_name"),
            "problem_name": summary.get("problem_name"),
            "experiment_name": summary.get("experiment_name"),
            "seed": summary.get("seed"),
            "status": summary.get("status"),
            "stop_reason": summary.get("stop_reason"),
            "wall_time_s": summary.get("wall_time_s"),
            "completed_steps": summary.get("completed_steps"),
            "num_parameters": model.get("num_parameters"),
            "peak_gpu_mem_mb": resources.get("peak_gpu_mem_mb"),
            "primary_metric": summary.get("primary_metric"),
            "best_step": best_metrics.get("best_step"),
        }
        for key, value in final_metrics.items():
            if key not in row_metadata_keys:
                row[f"final_{key}"] = value
        for key, value in best_metrics.items():
            if key not in best_metadata_keys:
                row[f"best_{key}"] = value
        rows.append(row)

    csv_path = sweep_dir / "summary.csv"
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return csv_path

    fieldnames = list(base_fields)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def experiment_path(seed_dir: Path, experiment_name: str) -> Path:
    return seed_dir / f"{experiment_name}.pkl"
