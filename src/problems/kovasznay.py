from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

from diffops import hessian, jacobian
from structured_derivatives import make_structured_derivative_propagator


@dataclass(frozen=True)
class Domain2D:
    x_min: float = -0.5
    x_max: float = 1.0
    y_min: float = -0.5
    y_max: float = 1.5


@dataclass(frozen=True)
class KovasznayConfig:
    Re: float = 40.0
    lambda_bc: float = 1.0


class KovasznayProblem:
    name = "kovasznay"
    primary_metric = "rel_l2_mean"
    metric_keys = (
        "rel_l2_u",
        "rel_l2_v",
        "rel_l2_p",
        "rel_l2_grad_p",
        "rel_l2_mean",
        "rel_l2_mean_gradp",
    )
    metric_display_names = {
        "rel_l2_mean": "rel L2 (mean)",
        "rel_l2_mean_gradp": "rel L2 (mean grad p)",
    }
    
    def __init__(
        self,
        domain: Domain2D = Domain2D(),
        cfg: KovasznayConfig = KovasznayConfig(),
        seed: int = 0,
    ):
        self.domain = domain
        self.cfg = cfg
        self.seed = seed
        self._structured_model = None
        self._structured_propagator = None

    def _assemble_residual_vector(
        self,
        blocks: Dict[str, torch.Tensor],
        *,
        n_int: int,
        n_b: int,
    ) -> torch.Tensor:
        # We normalize the contributions of interior and boundary terms in a problem-aware way.
        # This implicitly balances the rows of J before J^T J is ever formed.
        n_int = max(n_int, 1)
        n_b = max(n_b, 1)
        s_int = (1.0 / n_int) ** 0.5
        s_bc = (1.0 * float(self.cfg.lambda_bc) / n_b) ** 0.5
        return torch.cat([
            s_int * blocks["mom_x"],
            s_int * blocks["mom_y"],
            s_int * blocks["cont"],
            s_bc * blocks["bc_u"],
            s_bc * blocks["bc_v"],
        ])

    def analytic(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Re = float(self.cfg.Re)
        lam = 0.5 * (Re - np.sqrt(Re**2 + 16.0 * np.pi**2))
        lam = torch.tensor(lam, dtype=x.dtype, device=x.device)

        exp_lx = torch.exp(lam * x)
        u = 1.0 - exp_lx * torch.cos(2.0 * np.pi * y)
        v = (lam / (2.0 * np.pi)) * exp_lx * torch.sin(2.0 * np.pi * y)
        p = 0.5 * (1.0 - exp_lx**2)
        return u, v, p

    def analytic_pressure_gradient(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        Re = float(self.cfg.Re)
        lam = 0.5 * (Re - np.sqrt(Re**2 + 16.0 * np.pi**2))
        lam = torch.tensor(lam, dtype=x.dtype, device=x.device)
        p_x = -lam * torch.exp(2.0 * lam * x)
        p_y = torch.zeros_like(y)
        return p_x, p_y

    def sample_interior_grid(
        self,
        nx: int,
        ny: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d = self.domain
        xs = np.linspace(d.x_min, d.x_max, nx)
        ys = np.linspace(d.y_min, d.y_max, ny)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        return (
            torch.tensor(X.reshape(-1, 1), device=device, dtype=dtype),
            torch.tensor(Y.reshape(-1, 1), device=device, dtype=dtype),
        )

    def sample_boundary_grid(
        self,
        n_per_side: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d = self.domain
        xs = np.linspace(d.x_min, d.x_max, n_per_side)
        ys = np.linspace(d.y_min, d.y_max, n_per_side)

        xb = np.vstack([
            np.full((n_per_side, 1), d.x_min),
            np.full((n_per_side, 1), d.x_max),
            xs.reshape(-1, 1),
            xs.reshape(-1, 1),
        ])
        yb = np.vstack([
            ys.reshape(-1, 1),
            ys.reshape(-1, 1),
            np.full((n_per_side, 1), d.y_min),
            np.full((n_per_side, 1), d.y_max),
        ])
        return (
            torch.tensor(xb, device=device, dtype=dtype),
            torch.tensor(yb, device=device, dtype=dtype),
        )

    def sample_validation_grid(
        self,
        nx: int,
        ny: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sample_interior_grid(nx=nx, ny=ny, device=device, dtype=dtype)

    def sample_train_batch(
        self,
        train_cfg: Any,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, torch.Tensor]:
        nx = int(train_cfg.nx_int)
        ny = int(train_cfg.ny_int)
        x_int, y_int = self.sample_interior_grid(
            nx=nx, ny=ny, device=device, dtype=dtype,
        )
        n_per_side = int(train_cfg.n_bnd)
        x_b, y_b = self.sample_boundary_grid(
            n_per_side=n_per_side, device=device, dtype=dtype,
        )
        return {
            "x_int": x_int, "y_int": y_int, "x_b": x_b, "y_b": y_b,
            "n_int": nx * ny, "n_b": 4 * n_per_side
        }

    def training_summary(
        self,
        train_cfg: Any,
        train_batch: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, Any]:
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
            payload["n_boundary_points"] = 4 * train_cfg.n_bnd
            return payload

        payload["n_interior_points"] = int(train_batch.get("n_int", train_cfg.nx_int * train_cfg.ny_int))
        payload["n_boundary_points"] = int(train_batch.get("n_b", 4 * train_cfg.n_bnd))
        return payload
    
    def iter_residual_chunks(
        self,
        batch: Dict[str, torch.Tensor],
        chunk_size: int,
    ):
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        n_int = int(batch["n_int"])
        n_b = int(batch["n_b"])

        if n_int == 0 and n_b == 0:
            yield {"kind": "interior", "x_int": batch["x_int"][:0], "y_int": batch["y_int"][:0]}
            return

        for start in range(0, n_int, chunk_size):
            stop = start + chunk_size
            yield {
                "kind": "interior",
                "x_int": batch["x_int"][start:stop],
                "y_int": batch["y_int"][start:stop],
            }

        for start in range(0, n_b, chunk_size):
            stop = start + chunk_size
            yield {
                "kind": "boundary",
                "x_b": batch["x_b"][start:stop],
                "y_b": batch["y_b"][start:stop],
            }

    def residual_blocks_chunk(
        self,
        model,
        chunk: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        *,
        param_dict=None,
        linear_params=None,
    ) -> Dict[str, torch.Tensor]:
        kind = chunk["kind"]

        ref = batch["x_int"] if batch["x_int"].numel() > 0 else batch["x_b"]
        device = ref.device
        dtype = ref.dtype

        empty = torch.empty(0, device=device, dtype=dtype)
        s_int = (1.0 / max(int(batch["n_int"]), 1)) ** 0.5
        s_bc = (1.0 * float(self.cfg.lambda_bc) / max(int(batch["n_b"]), 1)) ** 0.5

        if kind == "interior":
            res = self._pde_residuals_structured(
                model,
                chunk["x_int"],
                chunk["y_int"],
                param_dict=param_dict,
                linear_params=linear_params,
            )
            return {
                "mom_x": s_int * res["mom_x"].reshape(-1),
                "mom_y": s_int * res["mom_y"].reshape(-1),
                "cont": s_int * res["cont"].reshape(-1),
                "bc_u": empty,
                "bc_v": empty,
            }

        if kind == "boundary":
            Xb = torch.cat([chunk["x_b"], chunk["y_b"]], dim=1)
            pred = self._model_eval(model, Xb, param_dict=param_dict)
            u_true, v_true, _ = self.analytic(chunk["x_b"], chunk["y_b"])
            return {
                "mom_x": empty,
                "mom_y": empty,
                "cont": empty,
                "bc_u": s_bc * (pred[:, 0:1] - u_true).reshape(-1),
                "bc_v": s_bc * (pred[:, 1:2] - v_true).reshape(-1),
            }

        raise ValueError(f"Unknown chunk kind: {kind}")

    def residual_vector_chunk(
        self,
        model,
        chunk: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        *,
        param_dict=None,
        linear_params=None,
    ) -> torch.Tensor:
        blocks = self.residual_blocks_chunk(
            model,
            chunk,
            batch,
            param_dict=param_dict,
            linear_params=linear_params,
        )
        return torch.cat([
            blocks["mom_x"],
            blocks["mom_y"],
            blocks["cont"],
            blocks["bc_u"],
            blocks["bc_v"],
        ])

    def _get_structured_propagator(self, model):
        if self._structured_model is not model:
            self._structured_model = model
            self._structured_propagator = make_structured_derivative_propagator(model)
        return self._structured_propagator

    def _model_eval(
        self,
        model,
        X: torch.Tensor,
        *,
        param_dict=None,
    ) -> torch.Tensor:
        return model(X) if param_dict is None else functional_call(model, param_dict, (X,))

    def _pde_residuals_autograd(
        self,
        model,
        x: torch.Tensor,
        y: torch.Tensor,
        param_dict=None,
    ) -> Dict[str, torch.Tensor]:
        Re = float(self.cfg.Re)
        nu = 1.0 / Re

        X = torch.cat([x, y], dim=1)
        out = self._model_eval(model, X, param_dict=param_dict)
        
        def model_single(x_single: torch.Tensor) -> torch.Tensor:
            return self._model_eval(
                model,
                x_single.unsqueeze(0),
                param_dict=param_dict,
            ).squeeze(0)

        J = jacobian(model_single, X)
        H = hessian(model_single, X)

        u = out[:, 0:1]
        v = out[:, 1:2]

        u_x = J[:, 0, 0:1]
        u_y = J[:, 0, 1:2]
        v_x = J[:, 1, 0:1]
        v_y = J[:, 1, 1:2]
        p_x = J[:, 2, 0:1]
        p_y = J[:, 2, 1:2]
        lap_u = H[:, 0, 0, 0:1] + H[:, 0, 1, 1:2]
        lap_v = H[:, 1, 0, 0:1] + H[:, 1, 1, 1:2]

        return {
            "mom_x": u * u_x + v * u_y + p_x - nu * lap_u,
            "mom_y": u * v_x + v * v_y + p_y - nu * lap_v,
            "cont": u_x + v_y,
        }

    def _pde_residuals_structured(
        self,
        model,
        x: torch.Tensor,
        y: torch.Tensor,
        param_dict=None,
        *,
        propagate=None,
        linear_params=None,
    ) -> Dict[str, torch.Tensor]:
        Re = float(self.cfg.Re)
        nu = 1.0 / Re

        X = torch.cat([x, y], dim=1)
        if propagate is None:
            if param_dict is None and linear_params is None:
                propagate = self._get_structured_propagator(model)
            else:
                propagate = make_structured_derivative_propagator(
                    model,
                    param_dict=param_dict,
                    linear_params=linear_params,
                )
        out, d_out, d2_out = propagate(X)

        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]

        u_x = d_out[:, 0, 0:1]
        u_y = d_out[:, 0, 1:2]
        v_x = d_out[:, 1, 0:1]
        v_y = d_out[:, 1, 1:2]
        p_x = d_out[:, 2, 0:1]
        p_y = d_out[:, 2, 1:2]

        lap_u = d2_out[:, 0, 0, 0:1] + d2_out[:, 0, 1, 1:2]
        lap_v = d2_out[:, 1, 0, 0:1] + d2_out[:, 1, 1, 1:2]

        return {
            "mom_x": u * u_x + v * u_y + p_x - nu * lap_u,
            "mom_y": u * v_x + v * v_y + p_y - nu * lap_v,
            "cont": u_x + v_y,
        }

    def _raw_residual_blocks_autograd(
        self,
        model,
        batch: Dict[str, torch.Tensor],
        param_dict=None,
    ) -> Dict[str, torch.Tensor]:
        res = self._pde_residuals_autograd(
            model,
            batch["x_int"],
            batch["y_int"],
            param_dict=param_dict,
        )
        Xb = torch.cat([batch["x_b"], batch["y_b"]], dim=1)
        pred = self._model_eval(model, Xb, param_dict=param_dict)
        u_true, v_true, _ = self.analytic(batch["x_b"], batch["y_b"])

        return {
            "mom_x": res["mom_x"].reshape(-1),
            "mom_y": res["mom_y"].reshape(-1),
            "cont": res["cont"].reshape(-1),
            "bc_u": (pred[:, 0:1] - u_true).reshape(-1),
            "bc_v": (pred[:, 1:2] - v_true).reshape(-1),
        }

    def _raw_residual_blocks_structured(
        self,
        model,
        batch: Dict[str, torch.Tensor],
        param_dict=None,
        *,
        propagate=None,
        linear_params=None,
    ) -> Dict[str, torch.Tensor]:
        res = self._pde_residuals_structured(
            model,
            batch["x_int"],
            batch["y_int"],
            param_dict=param_dict,
            propagate=propagate,
            linear_params=linear_params,
        )
        Xb = torch.cat([batch["x_b"], batch["y_b"]], dim=1)
        pred = self._model_eval(model, Xb, param_dict=param_dict)
        u_true, v_true, _ = self.analytic(batch["x_b"], batch["y_b"])

        return {
            "mom_x": res["mom_x"].reshape(-1),
            "mom_y": res["mom_y"].reshape(-1),
            "cont": res["cont"].reshape(-1),
            "bc_u": (pred[:, 0:1] - u_true).reshape(-1),
            "bc_v": (pred[:, 1:2] - v_true).reshape(-1),
        }

    def residual_vector(
        self,
        model,
        batch: Dict[str, torch.Tensor],
        param_dict=None,
    ) -> torch.Tensor:
        # The GN optimizer uses the structured residual path for efficiency.
        # We keep the residual ordering/scaling identical to the training loss,
        # but compute PDE derivatives via structured propagation instead of autograd.
        blocks = self._raw_residual_blocks_structured(model, batch, param_dict=param_dict)
        return self._assemble_residual_vector(
            blocks,
            n_int=blocks["mom_x"].numel(),
            n_b=blocks["bc_u"].numel(),
        )

    def pde_loss(self, model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        blocks = self._raw_residual_blocks_autograd(
            model,
            {
                "x_int": x,
                "y_int": y,
                "x_b": x[:0],
                "y_b": y[:0],
            },
        )
        return 0.5 * (
            blocks["mom_x"].square().mean()
            + blocks["mom_y"].square().mean()
            + blocks["cont"].square().mean()
        )

    def boundary_loss(self, model, xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        if xb.numel() == 0:
            return torch.zeros((), device=xb.device, dtype=xb.dtype)
        Xb = torch.cat([xb, yb], dim=1)
        pred = model(Xb)
        u_true, v_true, _ = self.analytic(xb, yb)
        return 0.5 * (F.mse_loss(pred[:, 0:1], u_true) + F.mse_loss(pred[:, 1:2], v_true))

    def total_loss(
        self,
        model,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Off-the-shelf optimizers intentionally stay on the standard autograd
        # loss path so the usual PyTorch training workflow remains unchanged.
        # The structured derivative backend is reserved for GN residual assembly.
        pde = self.pde_loss(model, batch["x_int"], batch["y_int"])
        bc = self.boundary_loss(model, batch["x_b"], batch["y_b"])
        loss = pde + float(self.cfg.lambda_bc) * bc
        return loss, {
            "loss": float(loss.item()),
            "pde": float(pde.item()),
            "bc": float(bc.item()),
        }

    def residual_logs(
        self,
        model,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        _, logs = self.total_loss(model, batch)
        return logs

    @torch.no_grad()
    def relative_l2_components(
        self,
        model,
        xv: torch.Tensor,
        yv: torch.Tensor,
    ) -> Dict[str, float]:
        Xv = torch.cat([xv, yv], dim=1)
        propagate = self._get_structured_propagator(model)
        pred, d_pred, _ = propagate(Xv)
        u_true, v_true, p_true = self.analytic(xv, yv)
        u_pred, v_pred, p_pred = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
        grad_p_pred = d_pred[:, 2, 0:2]
        p_x_true, p_y_true = self.analytic_pressure_gradient(xv, yv)
        grad_p_true = torch.cat([p_x_true, p_y_true], dim=1)

        eps = 1e-16
        rel_u = torch.linalg.norm(u_pred - u_true) / (torch.linalg.norm(u_true) + eps)
        rel_v = torch.linalg.norm(v_pred - v_true) / (torch.linalg.norm(v_true) + eps)
        p0 = p_true - p_true.mean()
        dp = (p_pred - p_true)
        dp = dp - dp.mean()
        rel_p = torch.linalg.norm(dp) / (torch.linalg.norm(p0) + eps)
        rel_grad_p = torch.linalg.norm(grad_p_pred - grad_p_true) / (
            torch.linalg.norm(grad_p_true) + eps
        )
        rel_mean = (rel_u + rel_v + rel_p) / 3.0
        rel_mean_gradp = (rel_u + rel_v + rel_grad_p) / 3.0
        return {
            "rel_l2_u": float(rel_u.item()),
            "rel_l2_v": float(rel_v.item()),
            "rel_l2_p": float(rel_p.item()),
            "rel_l2_grad_p": float(rel_grad_p.item()),
            "rel_l2_mean": float(rel_mean.item()),
            "rel_l2_mean_gradp": float(rel_mean_gradp.item()),
        }

    @torch.no_grad()
    def metrics(
        self,
        model,
        xv: torch.Tensor,
        yv: torch.Tensor,
    ) -> Dict[str, float]:
        return self.relative_l2_components(model, xv, yv)
