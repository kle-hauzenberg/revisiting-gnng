from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

from models import MLP


def _tanh_with_derivatives(a: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    sigma = torch.tanh(a)
    sigma_prime = 1.0 - sigma.square()
    sigma_double_prime = -2.0 * sigma * sigma_prime
    return sigma, sigma_prime, sigma_double_prime


def _activation_rule(name: str) -> Callable[[Tensor], tuple[Tensor, Tensor, Tensor]]:
    key = name.lower()
    if key == "tanh":
        return _tanh_with_derivatives
    raise NotImplementedError(
        f"Structured derivative propagation is not implemented for activation '{name}'."
    )


def _linear_layers(model: MLP) -> list[nn.Linear]:
    layers = [module for module in model.net if isinstance(module, nn.Linear)]
    if not layers:
        raise ValueError("Expected MLP to contain at least one nn.Linear layer.")
    return layers


def _linear_layer_params(
    model: MLP,
    param_dict: dict[str, Tensor] | None,
) -> list[tuple[Tensor, Tensor | None]]:
    params = []
    for idx, module in enumerate(model.net):
        if not isinstance(module, nn.Linear):
            continue

        if param_dict is None:
            params.append((module.weight, module.bias))
            continue

        prefix = f"net.{idx}"
        weight = param_dict[f"{prefix}.weight"]
        bias = param_dict.get(f"{prefix}.bias")
        params.append((weight, bias))

    if not params:
        raise ValueError("Expected MLP to contain at least one nn.Linear layer.")
    return params


def structured_linear_layer_params(
    model: MLP,
    param_dict: dict[str, Tensor] | None = None,
) -> list[tuple[Tensor, Tensor | None]]:
    return _linear_layer_params(model, param_dict)


def make_structured_derivative_propagator(
    model: MLP,
    param_dict: dict[str, Tensor] | None = None,
    *,
    linear_params: list[tuple[Tensor, Tensor | None]] | None = None,
) -> Callable[[Tensor], tuple[Tensor, Tensor, Tensor]]:
    """
    Build a callable that propagates outputs, first input derivatives, and
    second input derivatives through an MLP layer by layer.

    Requires 'tanh' activation.

    The returned callable expects batched inputs of shape (N, d) and returns:
    - z: shape (N, m)
    - dz_dx: shape (N, m, d)
    - d2z_dxx: shape (N, m, d, d)
    """
    if not isinstance(model, MLP):
        raise TypeError(
            "Structured derivative propagation currently supports only models.MLP."
    )

    if model.cfg.activation.lower() != "tanh":
        raise NotImplementedError(
            "Structured derivatives currently support only activation='tanh'."
        )

    activation_rule = _activation_rule(model.cfg.activation)
    _linear_layers(model)
    if linear_params is None:
        linear_params = _linear_layer_params(model, param_dict)

    def propagate(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 2:
            raise ValueError(f"Expected x with shape (N, d), got {tuple(x.shape)}")

        batch_size, input_dim = x.shape
        z = x
        dz_dx = torch.eye(input_dim, device=x.device, dtype=x.dtype)
        dz_dx = dz_dx.unsqueeze(0).expand(batch_size, input_dim, input_dim).clone()
        d2z_dxx = torch.zeros(
            batch_size,
            input_dim,
            input_dim,
            input_dim,
            device=x.device,
            dtype=x.dtype,
        )

        for W, b in linear_params[:-1]:

            a = z @ W.T
            if b is not None:
                a = a + b

            da_dx = torch.einsum("oi,nid->nod", W, dz_dx)
            d2a_dxx = torch.einsum("oi,nijk->nojk", W, d2z_dxx)

            sigma, sigma_prime, sigma_double_prime = activation_rule(a)

            term1 = sigma_double_prime[:, :, None, None] * torch.einsum(
                "noj,nok->nojk", da_dx, da_dx
            )
            term2 = sigma_prime[:, :, None, None] * d2a_dxx

            z = sigma
            dz_dx = sigma_prime[:, :, None] * da_dx
            d2z_dxx = term1 + term2

        W, b = linear_params[-1]

        z = z @ W.T
        if b is not None:
            z = z + b

        dz_dx = torch.einsum("oi,nid->nod", W, dz_dx)
        d2z_dxx = torch.einsum("oi,nijk->nojk", W, d2z_dxx)

        return z, dz_dx, d2z_dxx

    return propagate


__all__ = ["make_structured_derivative_propagator", "structured_linear_layer_params"]
