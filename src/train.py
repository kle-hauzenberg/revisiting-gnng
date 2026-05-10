from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from optimizers import GaussNewton
from utils import (
    TIMEZONE,
    append_jsonl,
    get_git_commit,
    get_system_info,
    make_model,
    make_run_id,
    run_paths,
    save_experiment_result,
    write_json,
)


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adam"
    lr: float = 1e-3
    cosine_annealing: bool = False
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 10000
    log_every: int = 500
    time_constraint_s: Optional[float] = None
    seed: int = 0
    dtype: torch.dtype = torch.float32
    nx_int: int = 40
    ny_int: int = 40
    n_bnd: int = 80
    nx_val: int = 120
    ny_val: int = 120
    resample_train_batch_each_step: bool = False


def make_optimizer(
    model: nn.Module,
    cfg: OptimizerConfig,
    *,
    problem: Optional[Any] = None,
    train_batch: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.optim.Optimizer:
    name = cfg.name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, **cfg.kwargs)

    if name == "lbfgs":
        return torch.optim.LBFGS(model.parameters(), lr=cfg.lr, **cfg.kwargs)

    if name in {"gauss_newton", "gnng"}:
        if problem is None or train_batch is None:
            raise ValueError("GaussNewton requires problem and train_batch.")

        def residual_closure(param_dict=None):
            return problem.residual_vector(model, train_batch, param_dict=param_dict)

        return GaussNewton(
            model=model,
            residual_closure=residual_closure,
            problem=problem,
            train_batch=train_batch,
            lr=cfg.lr,
            **cfg.kwargs,
        )

    raise ValueError("Unknown optimizer. Use 'adam', 'lbfgs', or 'gauss_newton'.")


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def _problem_config_payload(problem: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": getattr(problem, "name", problem.__class__.__name__)}

    domain = getattr(problem, "domain", None)
    if domain is not None:
        payload["domain"] = asdict(domain) if is_dataclass(domain) else str(domain)

    cfg = getattr(problem, "cfg", None)
    if cfg is not None:
        payload["cfg"] = asdict(cfg) if is_dataclass(cfg) else str(cfg)

    seed = getattr(problem, "seed", None)
    if seed is not None:
        payload["seed"] = seed

    return payload


def _build_experiment_result(
    *,
    experiment_name: str,
    trainer: "Trainer",
    history: list[dict[str, Any]],
    problem: Any,
    opt_cfg: OptimizerConfig,
    train_cfg: TrainConfig,
    model_cfg: Any,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "experiment_name": experiment_name,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "history": history,
        "final_metrics": history[-1] if history else None,
        "model_state_dict": _cpu_state_dict(trainer.model),
        "config": {
            "seed": seed,
            "dtype": str(dtype),
            "device": str(device),
            "model": asdict(model_cfg),
            "optimizer": asdict(opt_cfg),
            "train": asdict(train_cfg),
            "problem": _problem_config_payload(problem),
        },
    }


def _build_run_config_payload(
    *,
    run_id: str,
    sweep_name: str,
    experiment_name: str,
    problem: Any,
    opt_cfg: OptimizerConfig,
    train_cfg: TrainConfig,
    model_cfg: Any,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    started_at: str,
    git_commit: Optional[str],
    system_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "sweep_name": sweep_name,
        "problem_name": problem.name,
        "experiment_name": experiment_name,
        "seed": seed,
        "started_at": started_at,
        "git_commit": git_commit,
        "device": str(device),
        "dtype": str(dtype),
        "model": asdict(model_cfg),
        "optimizer": asdict(opt_cfg),
        "train": asdict(train_cfg),
        "problem": _problem_config_payload(problem),
        "system": system_info,
    }


def _primary_metric_key(problem: Any) -> str:
    return str(getattr(problem, "primary_metric", "loss"))


def _metric_keys(problem: Any, eval_logs: Optional[dict[str, Any]] = None) -> tuple[str, ...]:
    keys = getattr(problem, "metric_keys", None)
    if keys is not None:
        return tuple(str(key) for key in keys)
    if eval_logs is not None:
        return tuple(str(key) for key in eval_logs.keys())
    return ()


def _metric_display_name(problem: Any, key: str) -> str:
    display_names = getattr(problem, "metric_display_names", {})
    if isinstance(display_names, dict):
        return str(display_names.get(key, key))
    return key


def _build_metrics_row(
    *,
    run_id: str,
    sweep_name: str,
    problem_name: str,
    experiment_name: str,
    seed: int,
    step: int,
    wall_time_s: float,
    train_logs: dict[str, Any],
    eval_logs: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "run_id": run_id,
        "sweep_name": sweep_name,
        "problem_name": problem_name,
        "experiment_name": experiment_name,
        "seed": seed,
        "step": step,
        "wall_time_s": wall_time_s,
    }
    row.update(train_logs)
    row.update(eval_logs)
    return row


def _build_pretty_log_row(
    *,
    problem: Any,
    experiment_name: str,
    step: int,
    wall_time_s: float,
    row: dict[str, Any],
) -> dict[str, str]:
    pretty = {
        "step": f"{step:06d}",
        "experiment": experiment_name,
        "wall time [s]": f"{wall_time_s:8.1f}",
        "loss": f"{row['loss']:.2e}",
    }
    metric_key = _primary_metric_key(problem)
    metric_value = row.get(metric_key)
    if metric_value is not None:
        pretty[_metric_display_name(problem, metric_key)] = f"{float(metric_value):.2e}"
    if row.get("gn.backend") is not None:
        pretty["backend"] = row["gn.backend"]
    if row.get("gn.damping") is not None and row["gn.damping"] > 0.0:
        pretty["damping"] = f"{row['gn.damping']:.2e}"
    if row.get("gn.cg_iters") is not None:
        pretty["CG iter"] = f"{int(row['gn.cg_iters']):03d}"
    if row.get("gn.step_size") is not None:
        pretty["step size"] = f"{row['gn.step_size']:.2e}"
    if row.get("lr") is not None:
        pretty["lr"] = f"{row['lr']:.2e}"
    if row.get("gpu_peak_mem_mb") is not None:
        pretty["gpu_peak_mem_mb"] = f"{row['gpu_peak_mem_mb']:.1f}"
    return pretty


def _train_summary_payload(
    problem: Any,
    train_cfg: TrainConfig,
    train_batch: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    training_summary = getattr(problem, "training_summary", None)
    if callable(training_summary):
        return training_summary(train_cfg, train_batch)

    payload = {
        "steps_requested": train_cfg.steps,
        "log_every": train_cfg.log_every,
        "time_constraint_s": train_cfg.time_constraint_s,
        "nx_int": train_cfg.nx_int,
        "ny_int": train_cfg.ny_int,
        "n_bnd": train_cfg.n_bnd,
        "nx_val": train_cfg.nx_val,
        "ny_val": train_cfg.ny_val,
        "n_validation_points": train_cfg.nx_val * train_cfg.ny_val,
    }

    if train_batch is None:
        payload["n_interior_points"] = train_cfg.nx_int * train_cfg.ny_int
        return payload

    for key, value in train_batch.items():
        if key.startswith("n_"):
            payload[f"{key}_points"] = int(value)
    return payload


def _best_metrics_payload(
    *,
    problem: Any,
    best_row: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if best_row is None:
        return None

    metric_key = _primary_metric_key(problem)
    payload = {
        "selection_metric": metric_key,
        "best_step": best_row["step"],
        "best_wall_time_s": best_row["wall_time_s"],
    }
    for key in _metric_keys(problem):
        if key in best_row:
            payload[key] = best_row[key]
    return payload


def _build_summary_payload(
    *,
    run_id: str,
    sweep_name: str,
    experiment_name: str,
    problem: Any,
    opt_cfg: OptimizerConfig,
    train_cfg: TrainConfig,
    model_cfg: Any,
    model: nn.Module,
    seed: int,
    dtype: torch.dtype,
    started_at: str,
    finished_at: str,
    stop_reason: str,
    system_info: dict[str, Any],
    git_commit: Optional[str],
    history: list[dict[str, Any]],
    train_batch: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    final_metrics = history[-1] if history else None
    metric_key = _primary_metric_key(problem)
    metric_history = [row for row in history if row.get(metric_key) is not None]
    best_row = min(metric_history, key=lambda row: row[metric_key]) if metric_history else None
    peak_gpu_mem_mb = max(
        (row["gpu_peak_mem_mb"] for row in history if row.get("gpu_peak_mem_mb") is not None),
        default=None,
    )
    return {
        "run_id": run_id,
        "sweep_name": sweep_name,
        "problem_name": problem.name,
        "experiment_name": experiment_name,
        "seed": seed,
        "status": "finished",
        "stop_reason": stop_reason,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_time_s": final_metrics["wall_time_s"] if final_metrics is not None else 0.0,
        "completed_steps": final_metrics["step"] if final_metrics is not None else 0,
        "system": {
            **system_info,
            "git_commit": git_commit,
            "dtype": str(dtype),
        },
        "model": {
            "architecture": "MLP",
            "layer_dims": list(model_cfg.layer_dims),
            "activation": model_cfg.activation,
            "init": model_cfg.init,
            "num_parameters": sum(p.numel() for p in model.parameters()),
        },
        "problem": _problem_config_payload(problem),
        "train": _train_summary_payload(problem, train_cfg, train_batch),
        "optimizer": {
            "name": opt_cfg.name,
            "lr": opt_cfg.lr,
            "backend": opt_cfg.kwargs.get("backend"),
            "kwargs": dict(opt_cfg.kwargs),
        },
        "resources": {
            "peak_gpu_mem_mb": peak_gpu_mem_mb,
        },
        "metric_keys": list(_metric_keys(problem)),
        "primary_metric": metric_key,
        "final_metrics": final_metrics,
        "best_metrics": _best_metrics_payload(problem=problem, best_row=best_row),
    }


def run_experiment(
    experiment_name: str,
    problem: Any,
    opt_cfg: OptimizerConfig,
    train_cfg: TrainConfig,
    model_cfg: Any,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    sweep_name: str,
    repo_root: Optional[Path] = None,
    run_dir: Optional[Path] = None,
    verbose: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any], Optional[Path]]:
    model = make_model(model_cfg=model_cfg, seed=seed, dtype=dtype)

    trainer = Trainer(
        problem=problem,
        model=model,
        train_cfg=train_cfg,
        opt_cfg=opt_cfg,
        device=device,
    )

    run_id = make_run_id(experiment_name, seed)
    paths = run_paths(run_dir) if run_dir is not None else None
    started_at = datetime.now(TIMEZONE).isoformat(timespec="seconds")
    git_commit = get_git_commit(repo_root) if repo_root is not None else None
    system_info = get_system_info(device)

    config_payload = _build_run_config_payload(
        run_id=run_id,
        sweep_name=sweep_name,
        experiment_name=experiment_name,
        problem=problem,
        opt_cfg=opt_cfg,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        seed=seed,
        dtype=dtype,
        device=device,
        started_at=started_at,
        git_commit=git_commit,
        system_info=system_info,
    )
    if paths is not None:
        write_json(paths["config"], config_payload)

    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    stop_reason = "completed"

    for step in range(1, train_cfg.steps + 1):
        train_logs = trainer.step()

        if step % train_cfg.log_every == 0 or step == 1 or step == train_cfg.steps:
            eval_logs = trainer.evaluate()
            t_elapsed = time.perf_counter() - t0
            row = _build_metrics_row(
                run_id=run_id,
                sweep_name=sweep_name,
                problem_name=problem.name,
                experiment_name=experiment_name,
                seed=seed,
                step=step,
                wall_time_s=t_elapsed,
                train_logs=train_logs,
                eval_logs=eval_logs,
            )
            history.append(row)

            if paths is not None:
                append_jsonl(paths["metrics"], row)

            if verbose is True:
                print(_build_pretty_log_row(
                    problem=problem,
                    experiment_name=experiment_name,
                    step=step,
                    wall_time_s=t_elapsed,
                    row=row,
                ))
            elif verbose is False and step == 1:
                print(f"Started experiment '{experiment_name}'")

            if train_cfg.time_constraint_s is not None and t_elapsed > train_cfg.time_constraint_s:
                stop_reason = "time_budget"
                metric_key = _primary_metric_key(problem)
                metric_label = _metric_display_name(problem, metric_key)
                metric_value = row.get(metric_key)
                metric_text = (
                    f"{metric_label} = {float(metric_value):.2e}"
                    if metric_value is not None
                    else f"{metric_label} unavailable"
                )
                print(
                    f"Hit time constraint ({train_cfg.time_constraint_s:.1f} s) after {step:06d} steps; "
                    f"{metric_text}"
                )
                break

    finished_at = datetime.now(TIMEZONE).isoformat(timespec="seconds")
    summary_payload = _build_summary_payload(
        run_id=run_id,
        sweep_name=sweep_name,
        experiment_name=experiment_name,
        problem=problem,
        opt_cfg=opt_cfg,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        model=model,
        seed=seed,
        dtype=dtype,
        started_at=started_at,
        finished_at=finished_at,
        stop_reason=stop_reason,
        system_info=system_info,
        git_commit=git_commit,
        history=history,
        train_batch=trainer.train_batch,
    )

    save_path = None
    if paths is not None:
        write_json(paths["summary"], summary_payload)
        result = _build_experiment_result(
            experiment_name=experiment_name,
            trainer=trainer,
            history=history,
            problem=problem,
            opt_cfg=opt_cfg,
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            seed=seed,
            dtype=dtype,
            device=device,
        )
        save_path = paths["result"]
        save_experiment_result(save_path, result)
        print(f"saved: {save_path}")

    return history, summary_payload, save_path


class Trainer:
    def __init__(
        self,
        problem: Any,
        model: nn.Module,
        train_cfg: TrainConfig = TrainConfig(),
        opt_cfg: OptimizerConfig = OptimizerConfig(),
        device: Optional[torch.device] = None,
    ):
        self.problem = problem
        self.model = model
        self.cfg = train_cfg
        self.opt_cfg = opt_cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)

        self.model.to(device=self.device, dtype=self.cfg.dtype)
        self.train_batch = self.problem.sample_train_batch(self.cfg, self.device, self.cfg.dtype)
        self.val_data = self.problem.sample_validation_grid(
            nx=self.cfg.nx_val,
            ny=self.cfg.ny_val,
            device=self.device,
            dtype=self.cfg.dtype,
        )
        self._set_requires_grad_for_pde_points(self.train_batch)
        self.opt = make_optimizer(
            self.model,
            self.opt_cfg,
            problem=self.problem,
            train_batch=self.train_batch,
        )
        self.scheduler = self._make_scheduler()
        self._track_cuda_peak_memory = self.device.type == "cuda" and torch.cuda.is_available()
        if self._track_cuda_peak_memory:
            torch.cuda.reset_peak_memory_stats(self.device)

    def _refresh_train_batch(self) -> None:
        new_batch = self.problem.sample_train_batch(self.cfg, self.device, self.cfg.dtype)
        if self.train_batch is new_batch:
            self._set_requires_grad_for_pde_points(self.train_batch)
            return
        self.train_batch.clear()
        self.train_batch.update(new_batch)
        self._set_requires_grad_for_pde_points(self.train_batch)

    def _set_requires_grad_for_pde_points(self, batch: Dict[str, torch.Tensor]) -> None:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point() and key.endswith("_int"):
                value.requires_grad_(True)

    def _make_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        if self.opt_cfg.name.lower() != "adam":
            return None
        if not self.opt_cfg.cosine_annealing:
            return None
        eta_min = 0.001 * self.opt_cfg.lr
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt,
            T_max=self.cfg.steps,
            eta_min=eta_min,
        )

    def _maybe_add_gpu_peak_memory(self, logs: Dict[str, float]) -> None:
        if not self._track_cuda_peak_memory:
            return
        logs["gpu_peak_mem_mb"] = float(torch.cuda.max_memory_allocated(self.device) / (1024**2))

    def step(self) -> Dict[str, float]:
        if self.cfg.resample_train_batch_each_step:
            self._refresh_train_batch()
        self.model.train()
        name = self.opt_cfg.name.lower()

        if name == "lbfgs":
            logs_holder: Dict[str, float] = {}

            def closure() -> torch.Tensor:
                self.opt.zero_grad(set_to_none=True)
                self._set_requires_grad_for_pde_points(self.train_batch)
                loss, logs = self.problem.total_loss(self.model, self.train_batch)
                loss.backward()
                logs_holder.clear()
                logs_holder.update(logs)
                return loss

            self.opt.step(closure)
            logs = dict(logs_holder)
            logs["lr"] = float(self.opt.param_groups[0]["lr"])
            self._maybe_add_gpu_peak_memory(logs)
            return logs

        if name in {"gauss_newton", "gnng"}:
            self._set_requires_grad_for_pde_points(self.train_batch)
            logs = dict(self.opt.step())
            logs["lr"] = float(self.opt.param_groups[0]["lr"])
            self._maybe_add_gpu_peak_memory(logs)
            return logs

        self.opt.zero_grad(set_to_none=True)
        loss, logs = self.problem.total_loss(self.model, self.train_batch)
        loss.backward()
        self.opt.step()
        if self.scheduler is not None:
            self.scheduler.step()
        logs["lr"] = float(self.opt.param_groups[0]["lr"])
        self._maybe_add_gpu_peak_memory(logs)
        return logs

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        if isinstance(self.val_data, (tuple, list)):
            return self.problem.metrics(self.model, *self.val_data)
        return self.problem.metrics(self.model, self.val_data)


__all__ = ["OptimizerConfig", "TrainConfig", "Trainer", "make_optimizer", "run_experiment"]
