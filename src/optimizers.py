from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.func import jvp, vjp

from structured_derivatives import structured_linear_layer_params

# I use a residual closure so the optimizer stays independent of the specific
# problem implementation. The closure provides "residual vector at current or
# trial parameters", which the optimizer needs for GN solves and line search.
# If one were to couple the optimizer directly to the model/residual terms,
# the same role is handled implicitly without a separate closure.

ClosureOut = Union[
    Tensor,
    Tuple[Tensor, Dict[str, float]],
    Dict[str, object],
]
ParamDict = Dict[str, Tensor]
ResidualClosure = Callable[[Optional[ParamDict]], ClosureOut]
LinearOp = Callable[[Tensor], Tensor]
PrecondFn = Callable[[Tensor], Tensor]


def _flatten_residual(r: Tensor) -> Tensor:
    if r.dim() == 0:
        return r.reshape(1)
    return r.reshape(-1)


def loss_from_residual(r: Tensor) -> Tensor:
    return 0.5 * torch.dot(r, r)


def _gauss_newton_matvec_autograd_functional(
    residual_fn: Callable[[Tensor], Tensor],
    theta: Tensor,
    v: Tensor,
    damping: float,
) -> Tensor:
    _, jv = torch.autograd.functional.jvp(residual_fn, (theta,), (v,), create_graph=False)
    _, jt_jv = torch.autograd.functional.vjp(residual_fn, theta, v=jv, create_graph=False)
    return jt_jv + float(damping) * v


def _gauss_newton_matvec_torch_func(
    residual_fn: Callable[[Tensor], Tensor],
    theta: Tensor,
    v: Tensor,
    damping: float,
) -> Tensor:
    _, jv = jvp(residual_fn, (theta,), (v,))
    _, vjp_fn = vjp(residual_fn, theta)
    jt_jv = vjp_fn(jv)[0]
    return jt_jv + float(damping) * v


def gauss_newton_matvec(
    residual_fn: Callable[[Tensor], Tensor],
    theta: Tensor,
    v: Tensor,
    damping: float,
    *,
    backend: str = "torch_func",
) -> Tensor:
    if backend == "torch_func":
        return _gauss_newton_matvec_torch_func(residual_fn, theta, v, damping)
    if backend == "autograd_functional":
        return _gauss_newton_matvec_autograd_functional(residual_fn, theta, v, damping)
    raise ValueError(f"Unknown gauss_newton_matvec backend: {backend}")


@dataclass
class CGResult:
    x: Tensor
    iters: int
    converged: bool
    r_norm: float


def cg_solve(
    matvec: LinearOp,
    g: Tensor,
    x0: Optional[Tensor] = None,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    maxiter: int = 500,
    M_inv: Optional[PrecondFn] = None,
) -> CGResult:
    if g.numel() == 0:
        return CGResult(x=b.clone(), iters=0, converged=True, r_norm=0.0)

    x = torch.zeros_like(g) if x0 is None else x0.clone()
    r = g - matvec(x)
    z = M_inv(r) if M_inv is not None else r.clone()
    p = z.clone()

    g_norm = max(torch.linalg.norm(g).item(), 1e-30)
    stopping_crit = float(atol) + float(rtol) * g_norm
    rz_old = torch.dot(r, z)

    converged = False
    it = 0
    for it in range(1, maxiter + 1):
        Ap = matvec(p)
        denom = torch.dot(p, Ap)
        if denom.abs().item() < 1e-30:
            r_norm = torch.linalg.norm(r).item()
            break

        alpha = rz_old / denom
        x = x + alpha * p
        r = r - alpha * Ap

        r_norm = torch.linalg.norm(r).item()
        if r_norm <= stopping_crit:
            converged = True
            break

        z = M_inv(r) if M_inv is not None else r
        rz_new = torch.dot(r, z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    return CGResult(x=x, iters=it, converged=converged, r_norm=r_norm)


class GaussNewton(torch.optim.Optimizer):
    """
    Gauss-Newton / natural-gradient optimizer family with two backends:
      - dense:     solve (J^T J + mu I) d = J^T r
      - matfree:   solve (J^T J + mu I) d = J^T r via CG
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        residual_closure: Optional[ResidualClosure] = None,
        problem: Optional[Any] = None,
        train_batch: Optional[Dict[str, Tensor]] = None,
        lr: float = 1.0,
        chunk_size: int = 128,
        damping_cap: float = 1e-5,
        damping_floor: float = 1e-6,
        damping_floor_late: Optional[float] = None,
        damping_floor_switch_loss: Optional[float] = None,
        damping_scale: float = 1.0,
        backend: str = "matfree",
        cg_rtol: float = 1e-8,
        cg_atol: float = 0.0,
        cg_maxiter: int = 200,
        use_cg_precond: bool = False,
        precond_nx_int: Optional[int] = None,
        precond_ny_int: Optional[int] = None,
        precond_n_bnd: Optional[int] = None,
        do_line_search: bool = True,
        line_search_steps: int = 31,
        line_search_min_step: float = 1e-4,
    ):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if damping_cap < 0:
            raise ValueError("need damping_cap >= 0")
        if damping_scale <= 0:
            raise ValueError("need damping_scale > 0")
        if damping_floor < 0:
            raise ValueError("need damping_floor >= 0")
        if damping_floor > damping_cap:
            raise ValueError("damping_floor must be <= damping_cap")
        if (damping_floor_late is None) != (damping_floor_switch_loss is None):
            raise ValueError(
                "damping_floor_late and damping_floor_switch_loss must be set together"
            )
        if damping_floor_late is not None:
            if damping_floor_late < 0:
                raise ValueError("need damping_floor_late >= 0")
            if damping_floor_late > damping_cap:
                raise ValueError("damping_floor_late must be <= damping_cap")
            if damping_floor_switch_loss is None or damping_floor_switch_loss <= 0:
                raise ValueError("damping_floor_switch_loss must be > 0")
        if backend not in {"dense", "matfree"}:
            raise ValueError("backend must be 'dense' or 'matfree'")
        if line_search_steps < 2:
            raise ValueError("line_search_steps must be >= 2")
        if precond_nx_int is not None and precond_nx_int < 1:
            raise ValueError("precond_nx_int must be >= 1")
        if precond_ny_int is not None and precond_ny_int < 1:
            raise ValueError("precond_ny_int must be >= 1")
        if precond_n_bnd is not None and precond_n_bnd < 1:
            raise ValueError("precond_n_bnd must be >= 1")
        defaults = dict(
            lr=lr,
            chunk_size=chunk_size,
            damping_cap=damping_cap,
            damping_floor=damping_floor,
            damping_floor_late=damping_floor_late,
            damping_floor_switch_loss=damping_floor_switch_loss,
            damping_scale=damping_scale,
            backend=backend,
            cg_rtol=cg_rtol,
            cg_atol=cg_atol,
            cg_maxiter=cg_maxiter,
            use_cg_precond=use_cg_precond,
            precond_nx_int=precond_nx_int,
            precond_ny_int=precond_ny_int,
            precond_n_bnd=precond_n_bnd,
            do_line_search=do_line_search,
            line_search_steps=line_search_steps,
            line_search_min_step=line_search_min_step,
        )
        # Call torch.optim.Optimizer.__init__ and make defaults part of self.param_groups
        super().__init__(model.parameters(), defaults)

        self.model = model
        self.problem = problem
        self.train_batch = train_batch
        self._default_residual_closure = residual_closure
        self._last_step_logs: Dict[str, float] = {}

        self._param_names = [name for name, _ in self.model.named_parameters()]
        plist = self._params_list()
        if len(self._param_names) != len(plist):
            raise ValueError("model.named_parameters() must match the optimizer parameter list")
        self._param_shapes = [p.shape for p in plist]
        self._param_numels = [p.numel() for p in plist]

    def _effective_damping_floor(
        self,
        *,
        loss_value: float,
        damping_floor: float,
        damping_floor_late: Optional[float],
        damping_floor_switch_loss: Optional[float],
    ) -> Tuple[float, bool]:
        if damping_floor_late is None or damping_floor_switch_loss is None:
            return damping_floor, False
        if loss_value <= float(damping_floor_switch_loss):
            return float(damping_floor_late), True
        return damping_floor, False

    def _params_list(self) -> List[Tensor]:
        ps: List[Tensor] = []
        for group in self.param_groups:
            ps.extend(group["params"])
        return ps

    def _gather_flat(self) -> Tensor:
        return parameters_to_vector(self._params_list())

    @torch.no_grad()
    def _set_flat(self, theta: Tensor) -> None:
        vector_to_parameters(theta, self._params_list())

    def _theta_to_param_dict(self, theta: Tensor) -> ParamDict:
        out: ParamDict = {}
        i = 0
        for name, shape, n in zip(self._param_names, self._param_shapes, self._param_numels):
            out[name] = theta[i : i + n].view(shape)
            i += n
        return out

    def _theta_to_param_dict_and_linear_params(
        self,
        theta: Tensor,
    ) -> Tuple[ParamDict, List[Tuple[Tensor, Optional[Tensor]]]]:
        param_dict = self._theta_to_param_dict(theta)
        linear_params = structured_linear_layer_params(self.model, param_dict)
        return param_dict, linear_params

    def _parse_closure_output(self, out: ClosureOut) -> Tuple[Tensor, Dict[str, float]]:
        if torch.is_tensor(out):
            return _flatten_residual(out), {}
        if isinstance(out, tuple):
            r, logs = out
            return _flatten_residual(r), dict(logs)
        if isinstance(out, dict):
            if "residual" not in out:
                raise ValueError("closure dict output must contain key 'residual'")
            r = out["residual"]
            logs = out.get("logs", {})
            if not torch.is_tensor(r):
                raise TypeError("'residual' must be a Tensor")
            return _flatten_residual(r), dict(logs)
        raise TypeError("Unsupported closure output type")

    def _resolve_closure(
        self,
        closure: Optional[ResidualClosure],
    ) -> ResidualClosure:
        c = closure if closure is not None else self._default_residual_closure
        if c is None:
            raise ValueError("GaussNewton requires a residual closure")
        return c

    def _residual_at_theta(self, theta: Tensor, closure: ResidualClosure) -> Tensor:
        param_dict = self._theta_to_param_dict(theta)
        with torch.enable_grad():
            r, _ = self._parse_closure_output(closure(param_dict))
        return r

    def _loss_at_theta(self, theta: Tensor, closure: ResidualClosure) -> Tensor:
        return loss_from_residual(self._residual_at_theta(theta, closure))

    def _chunk_residual_at_theta(
        self,
        theta: Tensor,
        chunk: Dict[str, Tensor],
        batch: Dict[str, Tensor],
    ) -> Tensor:
        if self.problem is None or self.model is None:
            raise ValueError("Chunked GN requires problem and model.")

        param_dict, linear_params = self._theta_to_param_dict_and_linear_params(theta)
        return self.problem.residual_vector_chunk(
            self.model,
            chunk,
            batch,
            param_dict=param_dict,
            linear_params=linear_params,
        )
    
    def _chunk_dense_jacobian(
        self,
        theta: Tensor,
        chunk: Dict[str, Tensor],
        batch: Dict[str, Tensor],
    ) -> Tensor:
        def residual_fn(th: Tensor) -> Tensor:
            return self._chunk_residual_at_theta(th, chunk, batch)

        with torch.enable_grad():
            th = theta.detach().clone().requires_grad_(True)
            J_chunk = torch.autograd.functional.jacobian(
                residual_fn,
                th,
                create_graph=False,
                vectorize=True,  # works well and fast in chunk mode
            )

        return J_chunk.reshape(-1, theta.numel()).detach()

    def _chunked_rhs(self, theta: Tensor, *, chunk_size: int) -> Tensor:
        if self.problem is None or self.train_batch is None:
            raise ValueError("Chunked GN requires problem and train_batch.")

        rhs = torch.zeros_like(theta)

        for chunk in self.problem.iter_residual_chunks(self.train_batch, chunk_size):
            r_chunk = self._chunk_residual_at_theta(theta, chunk, self.train_batch).detach()
            J_chunk = self._chunk_dense_jacobian(theta, chunk, self.train_batch)
            rhs = rhs + J_chunk.T @ r_chunk

        return rhs
    
    def _chunked_gramian(self, theta: Tensor, *, chunk_size: int) -> Tensor:
        if self.problem is None or self.train_batch is None:
            raise ValueError("Chunked GN requires problem and train_batch.")

        n_params = theta.numel()
        gram = torch.zeros(
            (n_params, n_params),
            device=theta.device,
            dtype=theta.dtype,
        )

        for chunk in self.problem.iter_residual_chunks(self.train_batch, chunk_size):
            J_chunk = self._chunk_dense_jacobian(theta, chunk, self.train_batch)
            gram = gram + J_chunk.T @ J_chunk

        return gram

    def _full_rhs(self, theta: Tensor, closure: ResidualClosure) -> Tensor:
        r = self._residual_at_theta(theta, closure).detach()
        J = self._dense_jacobian(theta, closure)
        return J.T @ r
    
    def _full_gramian(self, theta: Tensor, closure: ResidualClosure) -> Tensor:
        J = self._dense_jacobian(theta, closure)
        return J.T @ J
    
    def _chunked_rhs_and_gramian(
        self,
        theta: Tensor,
        batch: Dict[str, Tensor],
        *,
        chunk_size: int,
    ) -> Tuple[Tensor, Tensor]:
        if self.problem is None or self.train_batch is None:
            raise ValueError("Chunked GN requires a problem object.")

        n_params = theta.numel()
        rhs = torch.zeros(n_params, device=theta.device, dtype=theta.dtype)
        gram = torch.zeros((n_params, n_params), device=theta.device, dtype=theta.dtype)

        for chunk in self.problem.iter_residual_chunks(batch, chunk_size):
            r_chunk = self._chunk_residual_at_theta(theta, chunk, batch).detach()
            J_chunk = self._chunk_dense_jacobian(theta, chunk, batch)
            rhs = rhs + J_chunk.T @ r_chunk
            gram = gram + J_chunk.T @ J_chunk

        return rhs, gram
    
    def _sample_precond_batch(self) -> Dict[str, Tensor]:
        if self.problem is None or self.train_batch is None:
            raise ValueError("Preconditioner requires problem object and train_batch.")

        group = self.param_groups[0]

        precond_nx_int = group["precond_nx_int"]
        precond_ny_int = group["precond_ny_int"]
        precond_n_bnd = group["precond_n_bnd"]

        nx_int = precond_nx_int if precond_nx_int is not None else max(1, round(self.train_batch["n_int"] ** 0.5 / 3))
        ny_int = precond_ny_int if precond_ny_int is not None else nx_int
        n_bnd = precond_n_bnd if precond_n_bnd is not None else max(1, self.train_batch["n_b"] // (4 * 4))

        class _PrecondCfg:
            pass

        cfg = _PrecondCfg()
        cfg.nx_int = nx_int
        cfg.ny_int = ny_int
        cfg.n_bnd = n_bnd

        ref_tensor = self.train_batch["x_int"]  # tensor anchor for device/dtype
        return self.problem.sample_train_batch(
            cfg,
            device=ref_tensor.device,
            dtype=ref_tensor.dtype,
        )
    
    def _build_reduced_gramian_preconditioner(
        self,
        theta: Tensor,
        *,
        damping: float,
        chunk_size: int,
    ) -> PrecondFn:
        reduced_batch = self._sample_precond_batch()
        _, G_red = self._chunked_rhs_and_gramian(
            theta,
            reduced_batch,
            chunk_size=chunk_size,
        )

        if damping != 0.0:
            G_red.diagonal().add_(damping)

        try:
            L = torch.linalg.cholesky(G_red)

            def M_inv(v: Tensor) -> Tensor:
                return torch.cholesky_solve(v.unsqueeze(1), L).squeeze(1)

        except RuntimeError:
            A = G_red

            def M_inv(v: Tensor) -> Tensor:
                return torch.linalg.solve(A, v)

        return M_inv

    def _dense_jacobian(
        self,
        theta: Tensor,
        closure: ResidualClosure,
    ) -> Tensor:
        def residual_fn(th: Tensor) -> Tensor:
            return self._residual_at_theta(th, closure)

        with torch.enable_grad():
            th = theta.detach().clone().requires_grad_(True)
            J = torch.autograd.functional.jacobian(
                residual_fn,
                th,
                create_graph=False,
                vectorize=False,
            )
        return J.reshape(-1, theta.numel()).detach()

    def _solve_direction(
        self,
        theta: Tensor,
        closure: ResidualClosure,
        *,
        backend: str,
        chunk_size: int,
        damping: float,
        cg_rtol: float,
        cg_atol: float,
        cg_maxiter: int,
        g: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        if backend == "dense":
            if self.train_batch is None:
                raise ValueError("Dense chunked GN requires train_batch.")

            rhs, gram = self._chunked_rhs_and_gramian(
                theta,
                self.train_batch,
                chunk_size=chunk_size,
            )

            if damping != 0.0:
                gram.diagonal().add_(damping)

            try:
                chol = torch.linalg.cholesky(gram, upper=False)
                direction = torch.cholesky_solve(rhs.unsqueeze(1), chol, upper=False).squeeze(1)
                solve_method = "cholesky"
            except RuntimeError:
                eigvals, eigvecs = torch.linalg.eigh(gram, UPLO="L")
                direction = eigvecs @ ((eigvecs.T @ rhs) / eigvals)
                solve_method = "eigh_fallback"

            return direction, {
                "gn.backend": backend,
                "gn.solve_method": solve_method,
                "gn.solve_residual": float(torch.linalg.norm(rhs - (gram @ direction)).item()),
            }

        def residual_fn(th: Tensor) -> Tensor:
            return self._residual_at_theta(th, closure)

        def mv(v: Tensor) -> Tensor:
            th = theta.detach().clone().requires_grad_(True)
            return gauss_newton_matvec(
                residual_fn,
                th,
                v.detach(),
                damping=damping,
            ).detach()

        if g is None:
            raise ValueError("matfree backend requires gradient rhs g = J^T r")

        M_inv = None
        use_cg_precond = self.param_groups[0]["use_cg_precond"]
        if use_cg_precond:
            M_inv = self._build_reduced_gramian_preconditioner(
                theta,
                damping=damping,
                chunk_size=chunk_size,
            )

        cg = cg_solve(mv, g, atol=cg_atol, rtol=cg_rtol, maxiter=cg_maxiter, M_inv=M_inv)
        g_norm = max(torch.linalg.norm(g).item(), 1e-30)
        cg_stop_threshold = float(cg_atol) + float(cg_rtol) * g_norm

        return cg.x, {
            "gn.backend": backend,
            "gn.precond": bool(use_cg_precond),
            "gn.cg_iters": int(cg.iters),
            "gn.cg_converged": bool(cg.converged),
            "gn.solve_residual": float(cg.r_norm),
            "gn.solve_residual_over_tol": float(cg.r_norm / max(cg_stop_threshold, 1e-30)),
            "gn.relative_solve_residual": float(cg.r_norm / g_norm),
        }

    def dense_direction_info(
        self,
        closure: Optional[ResidualClosure] = None,
    ) -> Dict[str, Tensor | float | Dict[str, float]]:
        closure_fn = self._resolve_closure(closure)
        group = self.param_groups[0]
        backend = str(group["backend"])
        if backend != "dense":
            raise ValueError("dense_direction_info is only available for backend='dense'.")

        chunk_size = int(group["chunk_size"])
        damping_cap = float(group["damping_cap"])
        damping_floor = float(group["damping_floor"])
        damping_floor_late = group["damping_floor_late"]
        damping_floor_switch_loss = group["damping_floor_switch_loss"]
        damping_scale = float(group["damping_scale"])

        theta = self._gather_flat().detach()
        theta_req = theta.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            r = self._residual_at_theta(theta_req, closure_fn)
            loss = loss_from_residual(r)

        loss_value = float(loss.item())
        effective_damping_floor, _ = self._effective_damping_floor(
            loss_value=loss_value,
            damping_floor=damping_floor,
            damping_floor_late=damping_floor_late,
            damping_floor_switch_loss=damping_floor_switch_loss,
        )

        n_residual = 1.0 #max(r.numel(), 1)
        if damping_cap > 0.0:
            damping = min(damping_scale * loss_value / n_residual, damping_cap)
        else:
            damping = damping_cap
        if effective_damping_floor <= damping_cap:
            damping = max(damping, effective_damping_floor)

        rhs, gram = self._chunked_rhs_and_gramian(
            theta,
            self.train_batch,
            chunk_size=chunk_size,
        )

        if damping != 0.0:
            gram.diagonal().add_(damping)

        try:
            chol = torch.linalg.cholesky(gram, upper=False)
            direction = torch.cholesky_solve(rhs.unsqueeze(1), chol, upper=False).squeeze(1)
            solve_method = "cholesky"
        except RuntimeError:
            eigvals, eigvecs = torch.linalg.eigh(gram, UPLO="L")
            direction = eigvecs @ ((eigvecs.T @ rhs) / eigvals)
            solve_method = "eigh_fallback"

        return direction, {
            "gn.backend": backend,
            "gn.solve_method": solve_method,
            "gn.solve_residual": float(torch.linalg.norm(rhs - (gram @ direction)).item()),
        }

    def _line_search(
        self,
        theta: Tensor,
        d: Tensor,
        closure: ResidualClosure,
        lr: float,
        steps: int,
        min_step: float,
    ) -> Tuple[float, float]:
        best_step_size = 0.0
        best_loss = self._loss_at_theta(theta, closure).item()
        line_search_grid = torch.logspace(
            0, math.log10(min_step), steps=steps, dtype=theta.dtype, device="cpu"
        )
        for grid_point in line_search_grid:
            step_size = float(lr * grid_point.item())
            val = float(self._loss_at_theta(theta - step_size * d, closure).item())
            if math.isfinite(val) and val < best_loss:
                best_loss = val
                best_step_size = step_size
        return float(best_step_size), float(best_loss)

    @torch.no_grad()
    def step(
        self,
        closure: Optional[ResidualClosure] = None,
    ) -> Dict[str, float]:
        closure_fn = self._resolve_closure(closure)
        group = self.param_groups[0]
        lr = float(group["lr"])
        chunk_size = int(group["chunk_size"])
        damping_cap = float(group["damping_cap"])
        damping_floor = float(group["damping_floor"])
        damping_floor_late = group["damping_floor_late"]
        damping_floor_switch_loss = group["damping_floor_switch_loss"]
        damping_scale = float(group["damping_scale"])
        backend = str(group["backend"])
        cg_rtol = float(group["cg_rtol"])
        cg_atol = float(group["cg_atol"])
        cg_maxiter = int(group["cg_maxiter"])
        do_line_search = bool(group["do_line_search"])
        line_search_steps = int(group["line_search_steps"])
        line_search_min_step = float(group["line_search_min_step"])

        theta = self._gather_flat().detach()
        theta_req = theta.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            r = self._residual_at_theta(theta_req, closure_fn)
            loss = loss_from_residual(r)

        g = None
        if backend != "dense":
            g = torch.autograd.grad(loss, theta_req, create_graph=False)[0].detach()

        loss_before = float(loss.item())
        effective_damping_floor, damping_floor_late_active = self._effective_damping_floor(
            loss_value=loss_before,
            damping_floor=damping_floor,
            damping_floor_late=damping_floor_late,
            damping_floor_switch_loss=damping_floor_switch_loss,
        )

        n_residual = 1.0 #max(r.numel(), 1)
        if damping_cap > 0.0:
            damping = min(damping_scale * loss_before / n_residual, damping_cap)
        else:
            damping = damping_cap

        if effective_damping_floor <= damping_cap:
            damping = max(damping, effective_damping_floor)

        d, solve_logs = self._solve_direction(
            theta=theta,
            closure=closure_fn,
            backend=backend,
            chunk_size=chunk_size,
            damping=damping,
            cg_rtol=cg_rtol,
            cg_atol=cg_atol,
            cg_maxiter=cg_maxiter,
            g=g,
        )

        if do_line_search:
            eta, loss_after = self._line_search(
                theta, d, closure_fn, lr, line_search_steps, line_search_min_step
            )
        else:
            eta = lr
            loss_after = float(self._loss_at_theta(theta - eta * d, closure_fn).item())

        theta_next = theta - eta * d
        self._set_flat(theta_next)

        direction_norm = float(torch.linalg.norm(d).item())
        logs = {
            "loss": float(loss_after),
            "loss_before": float(loss_before),
            "gn.damping": float(damping),
            "gn.damping_floor": float(effective_damping_floor),
            "gn.damping_floor_late_active": bool(damping_floor_late_active),
            "gn.direction_norm": direction_norm,
            "gn.update_norm": float(eta * direction_norm),
            "gn.step_size": float(eta),
            "gn.relative_loss_decrease": float((loss_before - loss_after) / max(loss_before, 1e-30)),
        }
        if g is not None:
            logs["gn.grad_norm"] = float(torch.linalg.norm(g).item())
        logs.update(solve_logs)
        self._last_step_logs = logs
        return logs


__all__ = ["CGResult", "GaussNewton", "cg_solve", "gauss_newton_matvec", "loss_from_residual"]
