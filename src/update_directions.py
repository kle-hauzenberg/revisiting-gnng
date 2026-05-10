from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.func import functional_call, jvp

from models import MLPConfig
from optimizers import GaussNewton
from problems.kovasznay import Domain2D, KovasznayConfig, KovasznayProblem
from train import OptimizerConfig, TrainConfig, Trainer
from utils import get_device, make_model


@dataclass(frozen=True)
class UpdateDirectionConfig:
    seed: int
    device_preference: str
    model_cfg: MLPConfig
    train_cfg: TrainConfig
    gnng_dense_opt_cfg: OptimizerConfig
    problem_cfg: KovasznayConfig = KovasznayConfig()
    domain: Domain2D = Domain2D()
    plot_nx: int = 170
    plot_ny: int = 153


def make_problem(
    *,
    cfg: KovasznayConfig = KovasznayConfig(),
    domain: Domain2D = Domain2D(),
    seed: int = 0,
) -> KovasznayProblem:
    return KovasznayProblem(domain=domain, cfg=cfg, seed=seed)


def _clone_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _load_model(
    *,
    model_cfg: MLPConfig,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> torch.nn.Module:
    model = make_model(model_cfg=model_cfg, seed=seed, dtype=dtype)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.to(device=device, dtype=dtype)
    return model


def _theta_to_param_dict(model: torch.nn.Module, theta: torch.Tensor) -> dict[str, torch.Tensor]:
    params: dict[str, torch.Tensor] = {}
    offset = 0
    for name, param in model.named_parameters():
        numel = param.numel()
        params[name] = theta[offset : offset + numel].view_as(param)
        offset += numel
    return params


def _linearized_u_update(
    *,
    model: torch.nn.Module,
    theta: torch.Tensor,
    delta_theta: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    xy = torch.cat([x, y], dim=1)

    def u_of_theta(theta_flat: torch.Tensor) -> torch.Tensor:
        params = _theta_to_param_dict(model, theta_flat)
        return functional_call(model, params, (xy,))[:, 0:1]

    _, delta_u = jvp(u_of_theta, (theta,), (delta_theta,))
    return delta_u


@torch.no_grad()
def evaluate_u(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model(torch.cat([x, y], dim=1))[:, 0:1]


def _reshape_field(values: torch.Tensor, *, nx: int, ny: int):
    return values.detach().cpu().reshape(ny, nx).numpy()


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-16) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    denom = torch.linalg.norm(a_flat) * torch.linalg.norm(b_flat)
    if denom.item() <= eps:
        return float("nan")
    return float(torch.dot(a_flat, b_flat).item() / denom.item())


def _relative_field_error(reference: torch.Tensor, candidate: torch.Tensor, eps: float = 1e-16) -> float:
    ref_norm = torch.linalg.norm(reference)
    if ref_norm.item() <= eps:
        return float("nan")
    return float((torch.linalg.norm(candidate - reference) / ref_norm).item())


def _make_trainer(
    cfg: UpdateDirectionConfig,
    opt_cfg: OptimizerConfig,
    device: torch.device,
) -> Trainer:
    problem = make_problem(cfg=cfg.problem_cfg, domain=cfg.domain, seed=cfg.seed)
    model = _load_model(
        model_cfg=cfg.model_cfg,
        seed=cfg.seed,
        dtype=cfg.train_cfg.dtype,
        device=device,
    )
    return Trainer(
        problem=problem,
        model=model,
        train_cfg=cfg.train_cfg,
        opt_cfg=opt_cfg,
        device=device,
    )

def replay_checkpoints(
    steps: Sequence[int],
    *,
    opt_cfg: OptimizerConfig | None = None,
    cfg: UpdateDirectionConfig,
) -> dict[int, dict[str, torch.Tensor]]:
    selected_steps = sorted({int(step) for step in steps})
    if not selected_steps:
        raise ValueError("Need at least one step.")
    if min(selected_steps) < 1:
        raise ValueError("Steps must be >= 1.")

    device = get_device(cfg.device_preference)
    trainer = _make_trainer(cfg, opt_cfg or cfg.gnng_dense_opt_cfg, device)

    checkpoints: dict[int, dict[str, torch.Tensor]] = {}
    for step in range(1, max(selected_steps) + 1):
        trainer.step()
        if step in selected_steps:
            checkpoints[step] = _clone_model_state(trainer.model)
    return checkpoints


def _dense_direction_info_from_state(
    *,
    cfg: UpdateDirectionConfig,
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    trainer = _make_trainer(cfg, cfg.gnng_dense_opt_cfg, device)
    trainer.model.load_state_dict(state_dict)

    if not isinstance(trainer.opt, GaussNewton):
        raise TypeError("Expected a GaussNewton optimizer.")

    info = trainer.opt.dense_direction_info()
    info["model"] = trainer.model
    return info


def compare_update_directions(
    steps: Sequence[int],
    *,
    cfg: UpdateDirectionConfig,
) -> dict[str, Any]:
    device = get_device(cfg.device_preference)
    checkpoints = replay_checkpoints(steps, cfg=cfg)
    problem = make_problem(cfg=cfg.problem_cfg, domain=cfg.domain, seed=cfg.seed)

    x_plot, y_plot = problem.sample_validation_grid(
        nx=cfg.plot_nx,
        ny=cfg.plot_ny,
        device=device,
        dtype=cfg.train_cfg.dtype,
    )
    u_exact, _, _ = problem.analytic(x_plot, y_plot)

    results: dict[int, dict[str, Any]] = {}
    for step, state_dict in sorted(checkpoints.items()):
        model = _load_model(
            model_cfg=cfg.model_cfg,
            seed=cfg.seed,
            dtype=cfg.train_cfg.dtype,
            device=device,
            state_dict=state_dict,
        )
        u_current = evaluate_u(model, x_plot, y_plot).detach().clone()
        ideal_delta_u = (u_exact - u_current).detach().clone()

        dense_info = _dense_direction_info_from_state(cfg=cfg, state_dict=state_dict, device=device)
        theta = dense_info["theta"]
        dense_model = dense_info["model"]
        rhs = dense_info["rhs"]
        dense_direction = dense_info["direction"]

        gradient_delta_u = _linearized_u_update(
            model=dense_model,
            theta=theta,
            delta_theta=-rhs,
            x=x_plot,
            y=y_plot,
        ).detach().clone()
        gnng_delta_u = _linearized_u_update(
            model=dense_model,
            theta=theta,
            delta_theta=-dense_direction,
            x=x_plot,
            y=y_plot,
        ).detach().clone()

        step_result = {
            "step": step,
            "x_grid": _reshape_field(x_plot, nx=cfg.plot_nx, ny=cfg.plot_ny),
            "y_grid": _reshape_field(y_plot, nx=cfg.plot_nx, ny=cfg.plot_ny),
            "u_current": _reshape_field(u_current, nx=cfg.plot_nx, ny=cfg.plot_ny),
            "u_exact": _reshape_field(u_exact, nx=cfg.plot_nx, ny=cfg.plot_ny),
            "ideal_delta_u": _reshape_field(ideal_delta_u, nx=cfg.plot_nx, ny=cfg.plot_ny),
            "gnng_delta_u": _reshape_field(gnng_delta_u, nx=cfg.plot_nx, ny=cfg.plot_ny),
            "gradient_delta_u": _reshape_field(gradient_delta_u, nx=cfg.plot_nx, ny=cfg.plot_ny),
            "metrics": {
                "ideal_delta_u_norm": float(torch.linalg.norm(ideal_delta_u).item()),
                "gnng_delta_theta_norm": float(torch.linalg.norm(dense_direction).item()),
                "gradient_delta_theta_norm": float(torch.linalg.norm(rhs).item()),
                "gnng_delta_u_norm": float(torch.linalg.norm(gnng_delta_u).item()),
                "gradient_delta_u_norm": float(torch.linalg.norm(gradient_delta_u).item()),
                "gnng_cosine": _cosine_similarity(ideal_delta_u, gnng_delta_u),
                "gradient_cosine": _cosine_similarity(ideal_delta_u, gradient_delta_u),
                "gnng_relative_error": _relative_field_error(ideal_delta_u, gnng_delta_u),
                "gradient_relative_error": _relative_field_error(ideal_delta_u, gradient_delta_u),
            },
            "logs": {
                "dense": {
                    "loss": float(dense_info["loss"].item()),
                    "damping": float(dense_info["damping"]),
                    **dense_info["solve_logs"],
                }
            },
        }
        results[step] = step_result

    return {
        "seed": cfg.seed,
        "device": str(device),
        "plot_nx": cfg.plot_nx,
        "plot_ny": cfg.plot_ny,
        "steps": results,
    }


__all__ = [
    "UpdateDirectionConfig",
    "compare_update_directions",
    "evaluate_u",
    "make_problem",
    "replay_checkpoints",
]
