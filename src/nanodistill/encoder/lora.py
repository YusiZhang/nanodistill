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

    try:
        from mlx_lm.tuner.lora import LoRALinear
    except ImportError as e:
        raise ConfigError(
            "mlx-lm is required for encoder LoRA (pip install mlx-lm)"
        ) from e

    replaced: List[str] = []

    for module_path, parent, attr_name, linear in _iter_target_linear_modules(
        model,
        targets=targets,
        linear_type=nn.Linear,
    ):
        lora_layer = LoRALinear.from_base(linear, r=rank)
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

        # MLX Module inherits from dict, so check for .children()
        # *before* the plain-dict branch to iterate modules properly.
        if hasattr(parent, "children") and callable(parent.children):
            items: dict = {}
            items.update(parent.children())
            for k, v in vars(parent).items():
                if not k.startswith("_") and k not in items:
                    items[k] = v

            for attr_name, value in items.items():
                path = f"{path_prefix}.{attr_name}" if path_prefix else attr_name
                if isinstance(value, linear_type) and any(t in path for t in targets):
                    matches.append((path, parent, attr_name, value))
                    continue

                if _looks_like_module(value) or isinstance(value, (list, tuple, dict)):
                    walk(value, path)
            return

        if isinstance(parent, dict):
            for key, value in parent.items():
                path = f"{path_prefix}.{key}" if path_prefix else str(key)
                if _looks_like_module(value) or isinstance(value, (list, tuple, dict)):
                    walk(value, path)
            return

    walk(model, "")
    return matches



def _looks_like_module(value) -> bool:
    """Heuristic: MLX modules expose parameters() and are usually objects."""
    return hasattr(value, "parameters") and hasattr(value, "__dict__")
