from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor
from torch.func import hessian as func_hessian
from torch.func import jacrev, vmap

SingleInputFn = Callable[[Tensor], Tensor]


def _ensure_batched_inputs(X: Tensor) -> None:
    if X.ndim != 2:
        raise ValueError(f"Expected X with shape (N, d), got {tuple(X.shape)}")


def _as_vector_output(y: Tensor) -> tuple[Tensor, bool]:
    if y.ndim == 0:
        return y.unsqueeze(0), True
    if y.ndim == 1:
        return y, y.shape[0] == 1
    raise ValueError(f"Expected scalar or 1D output, got shape {tuple(y.shape)}")

def jacobian(func: SingleInputFn, X: Tensor) -> Tensor:
    _ensure_batched_inputs(X)
    if X.shape[0] == 0:
        sample = func(X.new_zeros((X.shape[1],)))
        sample_vec, is_scalar = _as_vector_output(sample)
        if is_scalar:
            return X.new_zeros((0, X.shape[1]))
        return X.new_zeros((0, sample_vec.shape[0], X.shape[1]))

    _, is_scalar = _as_vector_output(func(X[0]))

    def vector_func(x: Tensor) -> Tensor:
        y_vec, _ = _as_vector_output(func(x))
        return y_vec

    J = vmap(jacrev(vector_func))(X)
    return J[:, 0, :] if is_scalar else J


def hessian(func: SingleInputFn, X: Tensor) -> Tensor:
    _ensure_batched_inputs(X)
    if X.shape[0] == 0:
        sample = func(X.new_zeros((X.shape[1],)))
        sample_vec, is_scalar = _as_vector_output(sample)
        if is_scalar:
            return X.new_zeros((0, X.shape[1], X.shape[1]))
        return X.new_zeros((0, sample_vec.shape[0], X.shape[1], X.shape[1]))

    _, is_scalar = _as_vector_output(func(X[0]))

    def vector_func(x: Tensor) -> Tensor:
        y_vec, _ = _as_vector_output(func(x))
        return y_vec

    H = vmap(func_hessian(vector_func))(X)
    return H[:, 0, :, :] if is_scalar else H


def grad(func: SingleInputFn, X: Tensor) -> Tensor:
    return jacobian(func, X)


def laplacian(func: SingleInputFn, X: Tensor) -> Tensor:
    _ensure_batched_inputs(X)
    H = hessian(func, X)
    if H.ndim != 3:
        raise ValueError("laplacian expects a scalar-valued function.")
    return H.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)


__all__ = ["grad", "hessian", "jacobian", "laplacian"]
