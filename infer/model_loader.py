"""Utilities for loading Unify-Agent/BAGEL checkpoints and constructing inferencer objects."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig


_REQUIRED_INFERENCER_APIS = (
    "init_gen_context",
    "update_context_text",
    "update_context_image",
    "gen_text",
    "gen_image",
    "interleave_inference",
)


def _build_quant_config(mode: int) -> BitsAndBytesConfig | None:
    """Build bitsandbytes quantization config by loading mode."""
    if mode == 1:
        return None
    if mode == 2:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    if mode == 3:
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unsupported mode={mode}, expected one of [1, 2, 3].")


def _resolve_inferencer(model: Any) -> Any:
    """Return an inferencer-like object from a loaded model instance."""
    # Preferred: remote code may expose `inferencer` directly.
    candidate = getattr(model, "inferencer", None)
    if candidate is not None:
        return candidate

    # Fallback: some repos expose `build_inferencer` or equivalent builder.
    for builder_name in ("build_inferencer", "get_inferencer", "create_inferencer"):
        builder = getattr(model, builder_name, None)
        if callable(builder):
            return builder()

    # Last resort: model itself already implements inference APIs.
    return model


def _validate_inferencer_apis(inferencer: Any) -> None:
    missing = [name for name in _REQUIRED_INFERENCER_APIS if not hasattr(inferencer, name)]
    if missing:
        raise RuntimeError(
            "Loaded object does not expose required inferencer APIs. "
            f"Missing: {missing}.\n"
            "Please ensure `model_path` points to the official Unify-Agent/BAGEL checkpoint "
            "with `trust_remote_code=True` inference helpers."
        )


def load_model_and_inferencer(
    model_path: str,
    mode: int = 1,
    device_map: str = "auto",
    base_model_path: str | None = None,
    ema_path: str | None = None,
    cast_ema_to_bfloat16: bool = False,
    ema_bf16_cache_path: str | None = None,
) -> Any:
    """Load model and return inferencer used by inference/eval scripts.

    Args:
        model_path: Local path or HF repo id.
        mode: 1=full precision, 2=NF4, 3=INT8.
        device_map: Device placement strategy passed to Transformers.
    """
    quant_config = _build_quant_config(mode)
    base_path = base_model_path or model_path

    _ = AutoConfig.from_pretrained(base_path, trust_remote_code=True)
    _ = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)

    common_kwargs = dict(
        trust_remote_code=True,
        device_map=device_map,
    )
    optional_model_kwargs = {}
    if base_model_path is not None:
        optional_model_kwargs["base_model_path"] = base_model_path
    if ema_path is not None:
        optional_model_kwargs["ema_path"] = ema_path
    if cast_ema_to_bfloat16:
        optional_model_kwargs["cast_ema_to_bfloat16"] = cast_ema_to_bfloat16
    if ema_bf16_cache_path is not None:
        optional_model_kwargs["ema_bf16_cache_path"] = ema_bf16_cache_path

    if quant_config is not None:
        model = AutoModel.from_pretrained(
            model_path,
            quantization_config=quant_config,
            **optional_model_kwargs,
            **common_kwargs,
        )
    else:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            **optional_model_kwargs,
            **common_kwargs,
        )

    model.eval()
    inferencer = _resolve_inferencer(model)
    _validate_inferencer_apis(inferencer)
    return inferencer
