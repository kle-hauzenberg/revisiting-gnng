from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from torch.func import functional_call

from diffops import hessian, jacobian
from structured_derivatives import make_structured_derivative_propagator


@dataclass(frozen=True)
class Domain3DTime:
    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0
    z_min: float = -1.0
    z_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0


@dataclass(frozen=True)
class BeltramiConfig:
    Re: float = 1.0
    a: float = 1.0
    d: float = 1.0
    lambda_bc: float = 1.0
    lambda_ic: float = 1.0
    include_final_time_bc: bool = False
    lambda_final: float = 1.0


class BeltramiProblem:
    name = "beltrami"
    primary_metric = "rel_l2_mean"
    metric_keys = (
        "rel_l2_u",
        "rel_l2_v",
        "rel_l2_w",
        "rel_l2_p",
        "rel_l2_grad_p",
        "rel_l2_mean",
        "rel_l2_mean_gradp",
        "rel_l2_u_t1",
        "rel_l2_v_t1",
        "rel_l2_w_t1",
        "rel_l2_p_t1",
        "rel_l2_mean_t1",
    )
    metric_display_names = {
        "rel_l2_mean": "rel L2 (mean)",
        "rel_l2_mean_gradp": "rel L2 (mean grad p)",
        "rel_l2_mean_t1": "rel L2 t=1 (mean)",
    }

    def __init__(
        self,
        domain: Domain3DTime = Domain3DTime(),
        cfg: BeltramiConfig = BeltramiConfig(),
        seed: int = 0,
    ):
        self.domain = domain
        self.cfg = cfg
        self.seed = seed
        self._structured_model = None
        self._structured_propagator = None

    def _loss_chunk_size(
        self,
        ref: torch.Tensor,
        *,
        for_autograd_hessian: bool = False,
    ) -> int:
        if ref.device.type != "cuda":
            return 2048 if not for_autograd_hessian else 256
        if for_autograd_hessian:
            return 8 if ref.dtype == torch.float64 else 16
        return 128 if ref.dtype == torch.float64 else 256

    def _chunk_slices(
        self,
        count: int,
        chunk_size: int,
    ):
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        for start in range(0, count, chunk_size):
            yield slice(start, min(start + chunk_size, count))

    def _assemble_residual_vector(
        self,
        blocks: Dict[str, torch.Tensor],
        *,
        n_int: int,
        n_b: int,
        n_i: int,
        n_f: int,
    ) -> torch.Tensor:
        # We normalize each residual family separately so the GN residual keeps
        # a balanced least-squares scaling as optional faces are enabled.
        n_int = max(n_int, 1)
        n_b = max(n_b, 1)
        n_i = max(n_i, 1)
        n_f = max(n_f, 1)
        s_int = (1.0 / n_int) ** 0.5
        s_bc = (float(self.cfg.lambda_bc) / n_b) ** 0.5
        s_ic = (float(self.cfg.lambda_ic) / n_i) ** 0.5
        s_final = (float(self.cfg.lambda_final) / n_f) ** 0.5
        pieces = [
            s_int * blocks["mom_x"],
            s_int * blocks["mom_y"],
            s_int * blocks["mom_z"],
            s_int * blocks["cont"],
            s_bc * blocks["bc_u"],
            s_bc * blocks["bc_v"],
            s_bc * blocks["bc_w"],
            s_bc * blocks["bc_p"],
            s_ic * blocks["ic_u"],
            s_ic * blocks["ic_v"],
            s_ic * blocks["ic_w"],
            s_ic * blocks["ic_p"],
            s_final * blocks["final_u"],
            s_final * blocks["final_v"],
            s_final * blocks["final_w"],
            s_final * blocks["final_p"],
        ]
        return torch.cat(pieces)

    def analytic(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a = float(self.cfg.a)
        d = float(self.cfg.d)

        ax = a * x
        ay = a * y
        az = a * z
        dx = d * x
        dy = d * y
        dz = d * z

        exp_ax = torch.exp(ax)
        exp_ay = torch.exp(ay)
        exp_az = torch.exp(az)
        decay = torch.exp(-(d**2) * t)
        decay_sq = torch.exp(-2.0 * (d**2) * t)

        phase_xy = ax + dy
        phase_yz = ay + dz
        phase_zx = az + dx

        u = -a * (exp_ax * torch.sin(phase_yz) + exp_az * torch.cos(phase_xy)) * decay
        v = -a * (exp_ay * torch.sin(phase_zx) + exp_ax * torch.cos(phase_yz)) * decay
        w = -a * (exp_az * torch.sin(phase_xy) + exp_ay * torch.cos(phase_zx)) * decay

        pressure_terms = (
            torch.exp(2.0 * ax)
            + torch.exp(2.0 * ay)
            + torch.exp(2.0 * az)
            + 2.0 * torch.sin(phase_xy) * torch.cos(phase_zx) * torch.exp(ay + az)
            + 2.0 * torch.sin(phase_yz) * torch.cos(phase_xy) * torch.exp(az + ax)
            + 2.0 * torch.sin(phase_zx) * torch.cos(phase_yz) * torch.exp(ax + ay)
        )
        p = -0.5 * (a**2) * pressure_terms * decay_sq
        return u, v, w, p

    def _sobol_points(
        self,
        count: int,
        *,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
        seed_offset: int,
    ) -> torch.Tensor:
        if count < 0:
            raise ValueError("count must be >= 0")
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if count == 0:
            return torch.empty((0, dim), device=device, dtype=dtype)

        engine = torch.quasirandom.SobolEngine(
            dimension=dim,
            scramble=True,
            seed=int(self.seed) + int(seed_offset),
        )
        return engine.draw(count).to(device=device, dtype=dtype)

    def _scale_unit_column(
        self,
        values: torch.Tensor,
        *,
        low: float,
        high: float,
    ) -> torch.Tensor:
        return float(low) + (float(high) - float(low)) * values

    def sample_interior_grid(
        self,
        nx: int,
        ny: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        rng: object | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_points = int(nx) * int(ny)
        d = self.domain
        points = self._sobol_points(
            n_points,
            dim=4,
            device=device,
            dtype=dtype,
            seed_offset=10,
        )
        return (
            self._scale_unit_column(points[:, 0:1], low=d.x_min, high=d.x_max),
            self._scale_unit_column(points[:, 1:2], low=d.y_min, high=d.y_max),
            self._scale_unit_column(points[:, 2:3], low=d.z_min, high=d.z_max),
            self._scale_unit_column(points[:, 3:4], low=d.t_min, high=d.t_max),
        )

    def sample_boundary_grid(
        self,
        n_per_face: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        rng: object | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.domain
        n_per_face = int(n_per_face)
        if n_per_face < 0:
            raise ValueError("n_per_face must be >= 0")

        def face_samples(axis: str, value: float, seed_base: int):
            points = self._sobol_points(
                n_per_face,
                dim=3,
                device=device,
                dtype=dtype,
                seed_offset=seed_base,
            )
            fixed = torch.full((n_per_face, 1), value, device=device, dtype=dtype)
            t = self._scale_unit_column(points[:, 2:3], low=d.t_min, high=d.t_max)
            if axis == "x":
                x = fixed
                y = self._scale_unit_column(points[:, 0:1], low=d.y_min, high=d.y_max)
                z = self._scale_unit_column(points[:, 1:2], low=d.z_min, high=d.z_max)
            elif axis == "y":
                x = self._scale_unit_column(points[:, 0:1], low=d.x_min, high=d.x_max)
                y = fixed
                z = self._scale_unit_column(points[:, 1:2], low=d.z_min, high=d.z_max)
            elif axis == "z":
                x = self._scale_unit_column(points[:, 0:1], low=d.x_min, high=d.x_max)
                y = self._scale_unit_column(points[:, 1:2], low=d.y_min, high=d.y_max)
                z = fixed
            else:
                raise ValueError(f"Unknown axis '{axis}'")
            return x, y, z, t

        faces = [
            face_samples("x", d.x_min, 100),
            face_samples("x", d.x_max, 110),
            face_samples("y", d.y_min, 120),
            face_samples("y", d.y_max, 130),
            face_samples("z", d.z_min, 140),
            face_samples("z", d.z_max, 150),
        ]

        x_b = torch.cat([face[0] for face in faces], dim=0)
        y_b = torch.cat([face[1] for face in faces], dim=0)
        z_b = torch.cat([face[2] for face in faces], dim=0)
        t_b = torch.cat([face[3] for face in faces], dim=0)
        return x_b, y_b, z_b, t_b

    def sample_initial_grid(
        self,
        n_points: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        rng: object | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.domain
        n_points = int(n_points)
        points = self._sobol_points(
            n_points,
            dim=3,
            device=device,
            dtype=dtype,
            seed_offset=200,
        )
        x_i = self._scale_unit_column(points[:, 0:1], low=d.x_min, high=d.x_max)
        y_i = self._scale_unit_column(points[:, 1:2], low=d.y_min, high=d.y_max)
        z_i = self._scale_unit_column(points[:, 2:3], low=d.z_min, high=d.z_max)
        t_i = torch.full((n_points, 1), d.t_min, device=device, dtype=dtype)
        return x_i, y_i, z_i, t_i

    def sample_validation_grid(
        self,
        nx: int,
        ny: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_points = int(nx) * int(ny)
        d = self.domain
        points = self._sobol_points(
            n_points,
            dim=4,
            device=device,
            dtype=dtype,
            seed_offset=300,
        )
        x_v = self._scale_unit_column(points[:, 0:1], low=d.x_min, high=d.x_max)
        y_v = self._scale_unit_column(points[:, 1:2], low=d.y_min, high=d.y_max)
        z_v = self._scale_unit_column(points[:, 2:3], low=d.z_min, high=d.z_max)
        t_v = self._scale_unit_column(points[:, 3:4], low=d.t_min, high=d.t_max)
        return x_v, y_v, z_v, t_v

    def sample_final_time_validation_grid(
        self,
        n_points: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.domain
        points = self._sobol_points(
            int(n_points),
            dim=3,
            device=device,
            dtype=dtype,
            seed_offset=310,
        )
        x_v = self._scale_unit_column(points[:, 0:1], low=d.x_min, high=d.x_max)
        y_v = self._scale_unit_column(points[:, 1:2], low=d.y_min, high=d.y_max)
        z_v = self._scale_unit_column(points[:, 2:3], low=d.z_min, high=d.z_max)
        t_v = torch.full((int(n_points), 1), d.t_max, device=device, dtype=dtype)
        return x_v, y_v, z_v, t_v

    def sample_final_time_grid(
        self,
        n_points: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sample_final_time_validation_grid(
            n_points=n_points,
            device=device,
            dtype=dtype,
        )

    def sample_train_batch(
        self,
        train_cfg: Any,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, torch.Tensor]:
        nx = int(train_cfg.nx_int)
        ny = int(train_cfg.ny_int)
        x_int, y_int, z_int, t_int = self.sample_interior_grid(
            nx=nx,
            ny=ny,
            device=device,
            dtype=dtype,
        )
        n_per_face = int(train_cfg.n_bnd)
        x_b, y_b, z_b, t_b = self.sample_boundary_grid(
            n_per_face=n_per_face,
            device=device,
            dtype=dtype,
        )
        x_i, y_i, z_i, t_i = self.sample_initial_grid(
            n_points=n_per_face,
            device=device,
            dtype=dtype,
        )
        n_final = n_per_face if bool(self.cfg.include_final_time_bc) else 0
        x_f, y_f, z_f, t_f = self.sample_final_time_grid(
            n_points=n_final,
            device=device,
            dtype=dtype,
        )
        return {
            "x_int": x_int,
            "y_int": y_int,
            "z_int": z_int,
            "t_int": t_int,
            "x_b": x_b,
            "y_b": y_b,
            "z_b": z_b,
            "t_b": t_b,
            "x_i": x_i,
            "y_i": y_i,
            "z_i": z_i,
            "t_i": t_i,
            "x_f": x_f,
            "y_f": y_f,
            "z_f": z_f,
            "t_f": t_f,
            "n_int": nx * ny,
            "n_b": 6 * n_per_face,
            "n_i": n_per_face,
            "n_f": n_final,
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
            payload["n_boundary_points"] = 6 * train_cfg.n_bnd
            payload["n_initial_points"] = train_cfg.n_bnd
            payload["n_final_points"] = train_cfg.n_bnd if bool(self.cfg.include_final_time_bc) else 0
            return payload

        payload["n_interior_points"] = int(train_batch.get("n_int", train_cfg.nx_int * train_cfg.ny_int))
        payload["n_boundary_points"] = int(train_batch.get("n_b", 6 * train_cfg.n_bnd))
        payload["n_initial_points"] = int(train_batch.get("n_i", train_cfg.n_bnd))
        payload["n_final_points"] = int(train_batch.get("n_f", 0))
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
        n_i = int(batch["n_i"])
        n_f = int(batch.get("n_f", 0))

        if n_int == 0 and n_b == 0 and n_i == 0 and n_f == 0:
            yield {
                "kind": "interior",
                "x_int": batch["x_int"][:0],
                "y_int": batch["y_int"][:0],
                "z_int": batch["z_int"][:0],
                "t_int": batch["t_int"][:0],
            }
            return

        for start in range(0, n_int, chunk_size):
            stop = start + chunk_size
            yield {
                "kind": "interior",
                "x_int": batch["x_int"][start:stop],
                "y_int": batch["y_int"][start:stop],
                "z_int": batch["z_int"][start:stop],
                "t_int": batch["t_int"][start:stop],
            }

        for start in range(0, n_b, chunk_size):
            stop = start + chunk_size
            yield {
                "kind": "boundary",
                "x_b": batch["x_b"][start:stop],
                "y_b": batch["y_b"][start:stop],
                "z_b": batch["z_b"][start:stop],
                "t_b": batch["t_b"][start:stop],
            }

        for start in range(0, n_i, chunk_size):
            stop = start + chunk_size
            yield {
                "kind": "initial",
                "x_i": batch["x_i"][start:stop],
                "y_i": batch["y_i"][start:stop],
                "z_i": batch["z_i"][start:stop],
                "t_i": batch["t_i"][start:stop],
            }

        for start in range(0, n_f, chunk_size):
            stop = start + chunk_size
            yield {
                "kind": "final",
                "x_f": batch["x_f"][start:stop],
                "y_f": batch["y_f"][start:stop],
                "z_f": batch["z_f"][start:stop],
                "t_f": batch["t_f"][start:stop],
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

        ref = batch["x_int"]
        device = ref.device
        dtype = ref.dtype

        empty = torch.empty(0, device=device, dtype=dtype)
        s_int = (1.0 / max(int(batch["n_int"]), 1)) ** 0.5
        s_bc = (float(self.cfg.lambda_bc) / max(int(batch["n_b"]), 1)) ** 0.5
        s_ic = (float(self.cfg.lambda_ic) / max(int(batch["n_i"]), 1)) ** 0.5
        s_final = (float(self.cfg.lambda_final) / max(int(batch.get("n_f", 0)), 1)) ** 0.5

        if kind == "interior":
            res = self._pde_residuals_structured(
                model,
                chunk["x_int"],
                chunk["y_int"],
                chunk["z_int"],
                chunk["t_int"],
                param_dict=param_dict,
                linear_params=linear_params,
            )
            return {
                "mom_x": s_int * res["mom_x"].reshape(-1),
                "mom_y": s_int * res["mom_y"].reshape(-1),
                "mom_z": s_int * res["mom_z"].reshape(-1),
                "cont": s_int * res["cont"].reshape(-1),
                "bc_u": empty,
                "bc_v": empty,
                "bc_w": empty,
                "ic_u": empty,
                "ic_v": empty,
                "ic_w": empty,
                "bc_p": empty,
                "ic_p": empty,
                "final_u": empty,
                "final_v": empty,
                "final_w": empty,
                "final_p": empty,
            }

        if kind == "boundary":
            Xb = torch.cat([chunk["x_b"], chunk["y_b"], chunk["z_b"], chunk["t_b"]], dim=1)
            pred = self._model_eval(model, Xb, param_dict=param_dict)
            u_true, v_true, w_true, p_true = self.analytic(
                chunk["x_b"],
                chunk["y_b"],
                chunk["z_b"],
                chunk["t_b"],
            )
            return {
                "mom_x": empty,
                "mom_y": empty,
                "mom_z": empty,
                "cont": empty,
                "bc_u": s_bc * (pred[:, 0:1] - u_true).reshape(-1),
                "bc_v": s_bc * (pred[:, 1:2] - v_true).reshape(-1),
                "bc_w": s_bc * (pred[:, 2:3] - w_true).reshape(-1),
                "ic_u": empty,
                "ic_v": empty,
                "ic_w": empty,
                "bc_p": s_bc * (pred[:, 3:4] - p_true).reshape(-1),
                "ic_p": empty,
                "final_u": empty,
                "final_v": empty,
                "final_w": empty,
                "final_p": empty,
            }

        if kind == "initial":
            Xi = torch.cat([chunk["x_i"], chunk["y_i"], chunk["z_i"], chunk["t_i"]], dim=1)
            pred = self._model_eval(model, Xi, param_dict=param_dict)
            u_true, v_true, w_true, p_true = self.analytic(
                chunk["x_i"],
                chunk["y_i"],
                chunk["z_i"],
                chunk["t_i"],
            )
            return {
                "mom_x": empty,
                "mom_y": empty,
                "mom_z": empty,
                "cont": empty,
                "bc_u": empty,
                "bc_v": empty,
                "bc_w": empty,
                "ic_u": s_ic * (pred[:, 0:1] - u_true).reshape(-1),
                "ic_v": s_ic * (pred[:, 1:2] - v_true).reshape(-1),
                "ic_w": s_ic * (pred[:, 2:3] - w_true).reshape(-1),
                "bc_p": empty,
                "ic_p": s_ic * (pred[:, 3:4] - p_true).reshape(-1),
                "final_u": empty,
                "final_v": empty,
                "final_w": empty,
                "final_p": empty,
            }

        if kind == "final":
            Xf = torch.cat([chunk["x_f"], chunk["y_f"], chunk["z_f"], chunk["t_f"]], dim=1)
            pred = self._model_eval(model, Xf, param_dict=param_dict)
            u_true, v_true, w_true, p_true = self.analytic(
                chunk["x_f"],
                chunk["y_f"],
                chunk["z_f"],
                chunk["t_f"],
            )
            return {
                "mom_x": empty,
                "mom_y": empty,
                "mom_z": empty,
                "cont": empty,
                "bc_u": empty,
                "bc_v": empty,
                "bc_w": empty,
                "ic_u": empty,
                "ic_v": empty,
                "ic_w": empty,
                "bc_p": empty,
                "ic_p": empty,
                "final_u": s_final * (pred[:, 0:1] - u_true).reshape(-1),
                "final_v": s_final * (pred[:, 1:2] - v_true).reshape(-1),
                "final_w": s_final * (pred[:, 2:3] - w_true).reshape(-1),
                "final_p": s_final * (pred[:, 3:4] - p_true).reshape(-1),
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
            blocks["mom_z"],
            blocks["cont"],
            blocks["bc_u"],
            blocks["bc_v"],
            blocks["bc_w"],
            blocks["bc_p"],
            blocks["ic_u"],
            blocks["ic_v"],
            blocks["ic_w"],
            blocks["ic_p"],
            blocks["final_u"],
            blocks["final_v"],
            blocks["final_w"],
            blocks["final_p"],
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

    def _centered_pressure_error(
        self,
        p_pred: torch.Tensor,
        p_true: torch.Tensor,
    ) -> torch.Tensor:
        if p_pred.numel() == 0:
            return p_pred - p_true
        return (p_pred - p_pred.mean()) - (p_true - p_true.mean())

    def _analytic_pressure_gradient(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            x_req = x.detach().clone().requires_grad_(True)
            y_req = y.detach().clone().requires_grad_(True)
            z_req = z.detach().clone().requires_grad_(True)
            t_req = t.detach().clone().requires_grad_(True)
            _, _, _, p_true = self.analytic(x_req, y_req, z_req, t_req)
            p_x, p_y, p_z = torch.autograd.grad(
                p_true.sum(),
                (x_req, y_req, z_req),
                create_graph=False,
            )
        return p_x, p_y, p_z

    def _pde_residuals_autograd(
        self,
        model,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        param_dict=None,
    ) -> Dict[str, torch.Tensor]:
        Re = float(self.cfg.Re)
        nu = 1.0 / Re

        X = torch.cat([x, y, z, t], dim=1)
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
        w = out[:, 2:3]

        u_x = J[:, 0, 0:1]
        u_y = J[:, 0, 1:2]
        u_z = J[:, 0, 2:3]
        u_t = J[:, 0, 3:4]
        v_x = J[:, 1, 0:1]
        v_y = J[:, 1, 1:2]
        v_z = J[:, 1, 2:3]
        v_t = J[:, 1, 3:4]
        w_x = J[:, 2, 0:1]
        w_y = J[:, 2, 1:2]
        w_z = J[:, 2, 2:3]
        w_t = J[:, 2, 3:4]
        p_x = J[:, 3, 0:1]
        p_y = J[:, 3, 1:2]
        p_z = J[:, 3, 2:3]

        lap_u = H[:, 0, 0, 0:1] + H[:, 0, 1, 1:2] + H[:, 0, 2, 2:3]
        lap_v = H[:, 1, 0, 0:1] + H[:, 1, 1, 1:2] + H[:, 1, 2, 2:3]
        lap_w = H[:, 2, 0, 0:1] + H[:, 2, 1, 1:2] + H[:, 2, 2, 2:3]

        return {
            "mom_x": u_t + u * u_x + v * u_y + w * u_z + p_x - nu * lap_u,
            "mom_y": v_t + u * v_x + v * v_y + w * v_z + p_y - nu * lap_v,
            "mom_z": w_t + u * w_x + v * w_y + w * w_z + p_z - nu * lap_w,
            "cont": u_x + v_y + w_z,
        }

    def _pde_residuals_structured(
        self,
        model,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        param_dict=None,
        *,
        propagate=None,
        linear_params=None,
    ) -> Dict[str, torch.Tensor]:
        Re = float(self.cfg.Re)
        nu = 1.0 / Re

        X = torch.cat([x, y, z, t], dim=1)
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
        w = out[:, 2:3]

        u_x = d_out[:, 0, 0:1]
        u_y = d_out[:, 0, 1:2]
        u_z = d_out[:, 0, 2:3]
        u_t = d_out[:, 0, 3:4]
        v_x = d_out[:, 1, 0:1]
        v_y = d_out[:, 1, 1:2]
        v_z = d_out[:, 1, 2:3]
        v_t = d_out[:, 1, 3:4]
        w_x = d_out[:, 2, 0:1]
        w_y = d_out[:, 2, 1:2]
        w_z = d_out[:, 2, 2:3]
        w_t = d_out[:, 2, 3:4]
        p_x = d_out[:, 3, 0:1]
        p_y = d_out[:, 3, 1:2]
        p_z = d_out[:, 3, 2:3]

        lap_u = d2_out[:, 0, 0, 0:1] + d2_out[:, 0, 1, 1:2] + d2_out[:, 0, 2, 2:3]
        lap_v = d2_out[:, 1, 0, 0:1] + d2_out[:, 1, 1, 1:2] + d2_out[:, 1, 2, 2:3]
        lap_w = d2_out[:, 2, 0, 0:1] + d2_out[:, 2, 1, 1:2] + d2_out[:, 2, 2, 2:3]

        return {
            "mom_x": u_t + u * u_x + v * u_y + w * u_z + p_x - nu * lap_u,
            "mom_y": v_t + u * v_x + v * v_y + w * v_z + p_y - nu * lap_v,
            "mom_z": w_t + u * w_x + v * w_y + w * w_z + p_z - nu * lap_w,
            "cont": u_x + v_y + w_z,
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
            batch["z_int"],
            batch["t_int"],
            param_dict=param_dict,
        )

        Xb = torch.cat([batch["x_b"], batch["y_b"], batch["z_b"], batch["t_b"]], dim=1)
        pred_b = self._model_eval(model, Xb, param_dict=param_dict)
        u_b, v_b, w_b, p_b = self.analytic(
            batch["x_b"],
            batch["y_b"],
            batch["z_b"],
            batch["t_b"],
        )

        Xi = torch.cat([batch["x_i"], batch["y_i"], batch["z_i"], batch["t_i"]], dim=1)
        pred_i = self._model_eval(model, Xi, param_dict=param_dict)
        u_i, v_i, w_i, p_i = self.analytic(
            batch["x_i"],
            batch["y_i"],
            batch["z_i"],
            batch["t_i"],
        )

        if int(batch.get("n_f", 0)) > 0:
            Xf = torch.cat([batch["x_f"], batch["y_f"], batch["z_f"], batch["t_f"]], dim=1)
            pred_f = self._model_eval(model, Xf, param_dict=param_dict)
            u_f, v_f, w_f, p_f = self.analytic(
                batch["x_f"],
                batch["y_f"],
                batch["z_f"],
                batch["t_f"],
            )
            final_u = (pred_f[:, 0:1] - u_f).reshape(-1)
            final_v = (pred_f[:, 1:2] - v_f).reshape(-1)
            final_w = (pred_f[:, 2:3] - w_f).reshape(-1)
            final_p = (pred_f[:, 3:4] - p_f).reshape(-1)
        else:
            empty = batch["x_int"].new_empty(0)
            final_u = empty
            final_v = empty
            final_w = empty
            final_p = empty

        return {
            "mom_x": res["mom_x"].reshape(-1),
            "mom_y": res["mom_y"].reshape(-1),
            "mom_z": res["mom_z"].reshape(-1),
            "cont": res["cont"].reshape(-1),
            "bc_u": (pred_b[:, 0:1] - u_b).reshape(-1),
            "bc_v": (pred_b[:, 1:2] - v_b).reshape(-1),
            "bc_w": (pred_b[:, 2:3] - w_b).reshape(-1),
            "ic_u": (pred_i[:, 0:1] - u_i).reshape(-1),
            "ic_v": (pred_i[:, 1:2] - v_i).reshape(-1),
            "ic_w": (pred_i[:, 2:3] - w_i).reshape(-1),
            "bc_p": (pred_b[:, 3:4] - p_b).reshape(-1),
            "ic_p": (pred_i[:, 3:4] - p_i).reshape(-1),
            "final_u": final_u,
            "final_v": final_v,
            "final_w": final_w,
            "final_p": final_p,
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
            batch["z_int"],
            batch["t_int"],
            param_dict=param_dict,
            propagate=propagate,
            linear_params=linear_params,
        )

        Xb = torch.cat([batch["x_b"], batch["y_b"], batch["z_b"], batch["t_b"]], dim=1)
        pred_b = self._model_eval(model, Xb, param_dict=param_dict)
        u_b, v_b, w_b, p_b = self.analytic(
            batch["x_b"],
            batch["y_b"],
            batch["z_b"],
            batch["t_b"],
        )

        Xi = torch.cat([batch["x_i"], batch["y_i"], batch["z_i"], batch["t_i"]], dim=1)
        pred_i = self._model_eval(model, Xi, param_dict=param_dict)
        u_i, v_i, w_i, p_i = self.analytic(
            batch["x_i"],
            batch["y_i"],
            batch["z_i"],
            batch["t_i"],
        )

        if int(batch.get("n_f", 0)) > 0:
            Xf = torch.cat([batch["x_f"], batch["y_f"], batch["z_f"], batch["t_f"]], dim=1)
            pred_f = self._model_eval(model, Xf, param_dict=param_dict)
            u_f, v_f, w_f, p_f = self.analytic(
                batch["x_f"],
                batch["y_f"],
                batch["z_f"],
                batch["t_f"],
            )
            final_u = (pred_f[:, 0:1] - u_f).reshape(-1)
            final_v = (pred_f[:, 1:2] - v_f).reshape(-1)
            final_w = (pred_f[:, 2:3] - w_f).reshape(-1)
            final_p = (pred_f[:, 3:4] - p_f).reshape(-1)
        else:
            empty = batch["x_int"].new_empty(0)
            final_u = empty
            final_v = empty
            final_w = empty
            final_p = empty

        return {
            "mom_x": res["mom_x"].reshape(-1),
            "mom_y": res["mom_y"].reshape(-1),
            "mom_z": res["mom_z"].reshape(-1),
            "cont": res["cont"].reshape(-1),
            "bc_u": (pred_b[:, 0:1] - u_b).reshape(-1),
            "bc_v": (pred_b[:, 1:2] - v_b).reshape(-1),
            "bc_w": (pred_b[:, 2:3] - w_b).reshape(-1),
            "ic_u": (pred_i[:, 0:1] - u_i).reshape(-1),
            "ic_v": (pred_i[:, 1:2] - v_i).reshape(-1),
            "ic_w": (pred_i[:, 2:3] - w_i).reshape(-1),
            "bc_p": (pred_b[:, 3:4] - p_b).reshape(-1),
            "ic_p": (pred_i[:, 3:4] - p_i).reshape(-1),
            "final_u": final_u,
            "final_v": final_v,
            "final_w": final_w,
            "final_p": final_p,
        }

    def residual_vector(
        self,
        model,
        batch: Dict[str, torch.Tensor],
        param_dict=None,
    ) -> torch.Tensor:
        blocks = self._raw_residual_blocks_structured(model, batch, param_dict=param_dict)
        return self._assemble_residual_vector(
            blocks,
            n_int=blocks["mom_x"].numel(),
            n_b=blocks["bc_u"].numel(),
            n_i=blocks["ic_u"].numel(),
            n_f=blocks["final_u"].numel(),
        )

    def pde_loss(
        self,
        model,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if x.numel() == 0:
            return torch.zeros((), device=x.device, dtype=x.dtype)

        try:
            chunk_size = self._loss_chunk_size(x)
            propagate = self._get_structured_propagator(model)
            sumsq_mx = torch.zeros((), device=x.device, dtype=x.dtype)
            sumsq_my = torch.zeros((), device=x.device, dtype=x.dtype)
            sumsq_mz = torch.zeros((), device=x.device, dtype=x.dtype)
            sumsq_cont = torch.zeros((), device=x.device, dtype=x.dtype)

            for chunk in self._chunk_slices(x.shape[0], chunk_size):
                res = self._pde_residuals_structured(
                    model,
                    x[chunk],
                    y[chunk],
                    z[chunk],
                    t[chunk],
                    propagate=propagate,
                )
                sumsq_mx = sumsq_mx + res["mom_x"].square().sum()
                sumsq_my = sumsq_my + res["mom_y"].square().sum()
                sumsq_mz = sumsq_mz + res["mom_z"].square().sum()
                sumsq_cont = sumsq_cont + res["cont"].square().sum()

            n = max(int(x.shape[0]), 1)
            return 0.5 * (
                sumsq_mx / n
                + sumsq_my / n
                + sumsq_mz / n
                + sumsq_cont / n
            )
        except (NotImplementedError, TypeError):
            chunk_size = self._loss_chunk_size(x, for_autograd_hessian=True)
            sumsq_mx = torch.zeros((), device=x.device, dtype=x.dtype)
            sumsq_my = torch.zeros((), device=x.device, dtype=x.dtype)
            sumsq_mz = torch.zeros((), device=x.device, dtype=x.dtype)
            sumsq_cont = torch.zeros((), device=x.device, dtype=x.dtype)

            for chunk in self._chunk_slices(x.shape[0], chunk_size):
                res = self._pde_residuals_autograd(
                    model,
                    x[chunk],
                    y[chunk],
                    z[chunk],
                    t[chunk],
                )
                sumsq_mx = sumsq_mx + res["mom_x"].square().sum()
                sumsq_my = sumsq_my + res["mom_y"].square().sum()
                sumsq_mz = sumsq_mz + res["mom_z"].square().sum()
                sumsq_cont = sumsq_cont + res["cont"].square().sum()

            n = max(int(x.shape[0]), 1)
            return 0.5 * (
                sumsq_mx / n
                + sumsq_my / n
                + sumsq_mz / n
                + sumsq_cont / n
            )

    def boundary_loss(
        self,
        model,
        xb: torch.Tensor,
        yb: torch.Tensor,
        zb: torch.Tensor,
        tb: torch.Tensor,
    ) -> torch.Tensor:
        if xb.numel() == 0:
            return torch.zeros((), device=xb.device, dtype=xb.dtype)
        chunk_size = self._loss_chunk_size(xb)
        sumsq_u = torch.zeros((), device=xb.device, dtype=xb.dtype)
        sumsq_v = torch.zeros((), device=xb.device, dtype=xb.dtype)
        sumsq_w = torch.zeros((), device=xb.device, dtype=xb.dtype)
        sumsq_p = torch.zeros((), device=xb.device, dtype=xb.dtype)

        for chunk in self._chunk_slices(xb.shape[0], chunk_size):
            Xb = torch.cat([xb[chunk], yb[chunk], zb[chunk], tb[chunk]], dim=1)
            pred = model(Xb)
            u_true, v_true, w_true, p_true = self.analytic(
                xb[chunk],
                yb[chunk],
                zb[chunk],
                tb[chunk],
            )
            sumsq_u = sumsq_u + (pred[:, 0:1] - u_true).square().sum()
            sumsq_v = sumsq_v + (pred[:, 1:2] - v_true).square().sum()
            sumsq_w = sumsq_w + (pred[:, 2:3] - w_true).square().sum()
            sumsq_p = sumsq_p + (pred[:, 3:4] - p_true).square().sum()

        n = max(int(xb.shape[0]), 1)
        return ((sumsq_u / n) + (sumsq_v / n) + (sumsq_w / n) + (sumsq_p / n)) / 2.0

    def initial_loss(
        self,
        model,
        xi: torch.Tensor,
        yi: torch.Tensor,
        zi: torch.Tensor,
        ti: torch.Tensor,
    ) -> torch.Tensor:
        if xi.numel() == 0:
            return torch.zeros((), device=xi.device, dtype=xi.dtype)
        chunk_size = self._loss_chunk_size(xi)
        sumsq_u = torch.zeros((), device=xi.device, dtype=xi.dtype)
        sumsq_v = torch.zeros((), device=xi.device, dtype=xi.dtype)
        sumsq_w = torch.zeros((), device=xi.device, dtype=xi.dtype)
        sumsq_p = torch.zeros((), device=xi.device, dtype=xi.dtype)

        for chunk in self._chunk_slices(xi.shape[0], chunk_size):
            Xi = torch.cat([xi[chunk], yi[chunk], zi[chunk], ti[chunk]], dim=1)
            pred = model(Xi)
            u_true, v_true, w_true, p_true = self.analytic(
                xi[chunk],
                yi[chunk],
                zi[chunk],
                ti[chunk],
            )
            sumsq_u = sumsq_u + (pred[:, 0:1] - u_true).square().sum()
            sumsq_v = sumsq_v + (pred[:, 1:2] - v_true).square().sum()
            sumsq_w = sumsq_w + (pred[:, 2:3] - w_true).square().sum()
            sumsq_p = sumsq_p + (pred[:, 3:4] - p_true).square().sum()

        n = max(int(xi.shape[0]), 1)
        return ((sumsq_u / n) + (sumsq_v / n) + (sumsq_w / n) + (sumsq_p / n)) / 2.0

    def final_time_loss(
        self,
        model,
        xf: torch.Tensor,
        yf: torch.Tensor,
        zf: torch.Tensor,
        tf: torch.Tensor,
    ) -> torch.Tensor:
        return self.initial_loss(model, xf, yf, zf, tf)

    def total_loss(
        self,
        model,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pde = self.pde_loss(
            model,
            batch["x_int"],
            batch["y_int"],
            batch["z_int"],
            batch["t_int"],
        )
        bc = self.boundary_loss(
            model,
            batch["x_b"],
            batch["y_b"],
            batch["z_b"],
            batch["t_b"],
        )
        ic = self.initial_loss(
            model,
            batch["x_i"],
            batch["y_i"],
            batch["z_i"],
            batch["t_i"],
        )
        if "x_f" in batch:
            final = self.final_time_loss(
                model,
                batch["x_f"],
                batch["y_f"],
                batch["z_f"],
                batch["t_f"],
            )
        else:
            final = torch.zeros((), device=batch["x_int"].device, dtype=batch["x_int"].dtype)
        loss = (
            pde
            + float(self.cfg.lambda_bc) * bc
            + float(self.cfg.lambda_ic) * ic
            + float(self.cfg.lambda_final) * final
        )
        return loss, {
            "loss": float(loss.item()),
            "pde": float(pde.item()),
            "bc": float(bc.item()),
            "ic": float(ic.item()),
            "final": float(final.item()),
        }

    def residual_logs(
        self,
        model,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        _, logs = self.total_loss(model, batch)
        return logs

    def _relative_l2_solution_components(
        self,
        pred: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        *,
        suffix: str = "",
    ) -> Dict[str, float]:
        u_true, v_true, w_true, p_true = self.analytic(x, y, z, t)
        u_pred = pred[:, 0:1]
        v_pred = pred[:, 1:2]
        w_pred = pred[:, 2:3]
        p_pred = pred[:, 3:4]

        eps = 1e-16
        rel_u = torch.linalg.norm(u_pred - u_true) / (torch.linalg.norm(u_true) + eps)
        rel_v = torch.linalg.norm(v_pred - v_true) / (torch.linalg.norm(v_true) + eps)
        rel_w = torch.linalg.norm(w_pred - w_true) / (torch.linalg.norm(w_true) + eps)
        p0 = p_true - p_true.mean()
        p_err = self._centered_pressure_error(p_pred, p_true)
        rel_p = torch.linalg.norm(p_err) / (torch.linalg.norm(p0) + eps)
        rel_mean = (rel_u + rel_v + rel_w + rel_p) / 4.0

        return {
            f"rel_l2_u{suffix}": float(rel_u.item()),
            f"rel_l2_v{suffix}": float(rel_v.item()),
            f"rel_l2_w{suffix}": float(rel_w.item()),
            f"rel_l2_p{suffix}": float(rel_p.item()),
            f"rel_l2_mean{suffix}": float(rel_mean.item()),
        }

    @torch.no_grad()
    def relative_l2_components(
        self,
        model,
        xv: torch.Tensor,
        yv: torch.Tensor,
        zv: torch.Tensor,
        tv: torch.Tensor,
    ) -> Dict[str, float]:
        Xv = torch.cat([xv, yv, zv, tv], dim=1)
        propagate = self._get_structured_propagator(model)
        pred, d_pred, _ = propagate(Xv)
        grad_p_pred = d_pred[:, 3, 0:3]
        p_x_true, p_y_true, p_z_true = self._analytic_pressure_gradient(xv, yv, zv, tv)
        grad_p_true = torch.cat([p_x_true, p_y_true, p_z_true], dim=1)

        eps = 1e-16
        rel_grad_p = torch.linalg.norm(grad_p_pred - grad_p_true) / (
            torch.linalg.norm(grad_p_true) + eps
        )
        metrics = self._relative_l2_solution_components(pred, xv, yv, zv, tv)
        rel_u = torch.tensor(metrics["rel_l2_u"], device=xv.device, dtype=xv.dtype)
        rel_v = torch.tensor(metrics["rel_l2_v"], device=xv.device, dtype=xv.dtype)
        rel_w = torch.tensor(metrics["rel_l2_w"], device=xv.device, dtype=xv.dtype)
        rel_mean_gradp = (rel_u + rel_v + rel_w + rel_grad_p) / 4.0
        metrics["rel_l2_grad_p"] = float(rel_grad_p.item())
        metrics["rel_l2_mean_gradp"] = float(rel_mean_gradp.item())
        return metrics

    @torch.no_grad()
    def metrics(
        self,
        model,
        xv: torch.Tensor,
        yv: torch.Tensor,
        zv: torch.Tensor,
        tv: torch.Tensor,
    ) -> Dict[str, float]:
        metrics = self.relative_l2_components(model, xv, yv, zv, tv)
        x_t1, y_t1, z_t1, t_t1 = self.sample_final_time_validation_grid(
            n_points=xv.shape[0],
            device=xv.device,
            dtype=xv.dtype,
        )
        X_t1 = torch.cat([x_t1, y_t1, z_t1, t_t1], dim=1)
        pred_t1 = self._model_eval(model, X_t1)
        metrics.update(
            self._relative_l2_solution_components(
                pred_t1,
                x_t1,
                y_t1,
                z_t1,
                t_t1,
                suffix="_t1",
            )
        )
        return metrics


__all__ = ["BeltramiConfig", "BeltramiProblem", "Domain3DTime"]
