"""LoRA utilities for encoder sequence-classification models."""

from __future__ import annotations

from typing import List, Tuple

from ..utils.errors import ConfigError


def apply_encoder_lora(model, rank: int, targets: List[str]) -> List[str]:
    """Apply LoRA adapters to matching linear modules.

    Returns:
        List of module paths that were replaced with LoRA layers.
    """
    try:
        import mlx.nn as nn
    except ImportError as e:
        raise ConfigError("mlx is required for encoder LoRA") from e

    if not hasattr(nn, "LoRALinear"):
        raise ConfigError("Installed mlx version does not include nn.LoRALinear")

    replaced: List[str] = []

    for module_path, parent, attr_name, linear in _iter_target_linear_modules(
        model,
        targets=targets,
        linear_type=nn.Linear,
    ):
        lora_layer = _build_lora_layer(nn, linear, rank)
        setattr(parent, attr_name, lora_layer)
        replaced.append(module_path)

    return replaced


def _iter_target_linear_modules(
    model,
    targets: List[str],
    linear_type,
) -> List[Tuple[str, object, str, object]]:
    """Find `(path, parent, attr_name, module)` for matching linear layers."""
    matches: List[Tuple[str, object, str, object]] = []

    def walk(parent, path_prefix: str) -> None:
        if isinstance(parent, list):
            for idx, value in enumerate(parent):
                path = f"{path_prefix}.{idx}" if path_prefix else str(idx)
                if _looks_like_module(value) or isinstance(value, list):
                    walk(value, path)
            return

        if isinstance(parent, tuple):
            for idx, value in enumerate(parent):
                path = f"{path_prefix}.{idx}" if path_prefix else str(idx)
                if _looks_like_module(value) or isinstance(value, (list, tuple)):
                    walk(value, path)
            return

        if isinstance(parent, dict):
            for key, value in parent.items():
                path = f"{path_prefix}.{key}" if path_prefix else str(key)
                if _looks_like_module(value) or isinstance(value, (list, tuple, dict)):
                    walk(value, path)
            return

        for attr_name, value in vars(parent).items():
            if attr_name.startswith("_"):
                continue

            path = f"{path_prefix}.{attr_name}" if path_prefix else attr_name
            if isinstance(value, linear_type) and any(t in path for t in targets):
                matches.append((path, parent, attr_name, value))
                continue

            if _looks_like_module(value) or isinstance(value, (list, tuple, dict)):
                walk(value, path)

    walk(model, "")
    return matches


def _build_lora_layer(nn, linear, rank: int):
    """Construct a LoRA layer from an existing linear layer across MLX versions."""
    if hasattr(nn.LoRALinear, "from_linear"):
        from_linear = nn.LoRALinear.from_linear
        try:
            return from_linear(linear, rank=rank)
        except TypeError:
            return from_linear(linear, r=rank)

    # Fallback for constructor-style APIs.
    in_features = getattr(linear, "in_features", None)
    out_features = getattr(linear, "out_features", None)
    if in_features is None or out_features is None:
        raise ConfigError("Unable to derive in/out features for LoRA replacement")

    try:
        return nn.LoRALinear(in_features, out_features, rank=rank)
    except TypeError:
        return nn.LoRALinear(in_features, out_features, r=rank)


def _looks_like_module(value) -> bool:
    """Heuristic: MLX modules expose parameters() and are usually objects."""
    return hasattr(value, "parameters") and hasattr(value, "__dict__")
