"""Weight init helpers for the shared video encoder."""

from __future__ import annotations

import torch.nn as nn


def init_linear_default(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_linear_kaiming(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_linear_orthogonal(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_layernorm_identity_bias(m: nn.Module) -> None:
    if isinstance(m, nn.LayerNorm) and m.elementwise_affine and m.weight is not None:
        nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class InitStrategy:
    """Maps string tags to module-wise init callables."""

    _registry: dict[str, callable] = {
        "xavier": init_linear_default,
        "kaiming": init_linear_kaiming,
        "orthogonal": init_linear_orthogonal,
    }

    @classmethod
    def apply(cls, module: nn.Module, name: str = "xavier") -> None:
        fn = cls._registry.get(name, init_linear_default)
        module.apply(fn)
        module.apply(init_layernorm_identity_bias)
