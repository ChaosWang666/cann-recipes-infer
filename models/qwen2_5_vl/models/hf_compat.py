# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Lightweight compatibility helpers that mirror a subset of the HuggingFace APIs.

The Qwen2.5-VL Ascend implementation reuses parts of the open-source architecture
but needs to avoid importing the `transformers` package inside the modeling file.
This module provides small shims for the pieces of the API that the model relies on
at runtime (mask creation, cache containers, model outputs, etc.).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


ALL_LAYERNORM_LAYERS: List[Type[nn.Module]] = []


def auto_docstring(*args, **kwargs):
    """No-op decorator used to keep the code structure close to HF."""

    def decorator(func):
        return func

    if args and callable(args[0]) and not kwargs:
        return decorator(args[0])
    return decorator


def can_return_tuple(func):
    """Decorator shim that returns the function unchanged."""

    return func


def is_torchdynamo_compiling() -> bool:
    """Return False in environments where torchdynamo is not available."""

    return False


class GenerationMixin:
    """Minimal mixin that exposes `prepare_inputs_for_generation` used by the runner."""

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        model_inputs: Dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
        if use_cache is not None:
            model_inputs["use_cache"] = use_cache
        if kwargs:
            model_inputs.update(kwargs)
        return model_inputs


@dataclass
class ModelOutput:
    """Simplified model output base container."""

    def to_tuple(self) -> Tuple[Any, ...]:
        return tuple(getattr(self, field) for field in self.__dataclass_fields__ if getattr(self, field) is not None)


@dataclass
class BaseModelOutputWithPast(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class Cache:
    """Cache container that mimics the behaviour of HF's dynamic cache."""

    def __init__(self) -> None:
        self.key_cache: List[Optional[torch.Tensor]] = []
        self.value_cache: List[Optional[torch.Tensor]] = []

    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_position = None
        if cache_kwargs is not None:
            cache_position = cache_kwargs.get("cache_position")
        self._ensure_layer(layer_idx)

        current_key = self.key_cache[layer_idx]
        current_value = self.value_cache[layer_idx]

        if current_key is None:
            if cache_position is None:
                current_key = key_states
                current_value = value_states
            else:
                max_index = int(cache_position.max().item()) + 1
                new_shape = key_states.shape[:-2] + (max_index, key_states.shape[-1])
                current_key = key_states.new_zeros(new_shape)
                current_value = value_states.new_zeros(new_shape)
                current_key.index_copy_(2, cache_position, key_states)
                current_value.index_copy_(2, cache_position, value_states)
        else:
            if cache_position is None:
                current_key = torch.cat([current_key, key_states], dim=2)
                current_value = torch.cat([current_value, value_states], dim=2)
            else:
                max_index = int(cache_position.max().item()) + 1
                if current_key.shape[2] < max_index:
                    pad = max_index - current_key.shape[2]
                    pad_shape = current_key.shape[:-2] + (pad, current_key.shape[-1])
                    current_key = torch.cat([current_key, current_key.new_zeros(pad_shape)], dim=2)
                    current_value = torch.cat([current_value, current_value.new_zeros(pad_shape)], dim=2)
                current_key = current_key.clone()
                current_value = current_value.clone()
                current_key.index_copy_(2, cache_position, key_states)
                current_value.index_copy_(2, cache_position, value_states)

        self.key_cache[layer_idx] = current_key
        self.value_cache[layer_idx] = current_value
        return current_key, current_value

    def get_seq_length(self) -> int:
        if not self.key_cache or self.key_cache[0] is None:
            return 0
        return self.key_cache[0].shape[2]


class DynamicCache(Cache):
    """Alias that matches the dynamic cache used by HF."""

    pass


class GradientCheckpointingLayer(nn.Module):
    """Simple mixin enabling gradient checkpointing toggling."""

    def __init__(self) -> None:
        super().__init__()
        self.gradient_checkpointing = False

    def set_gradient_checkpointing(self, enable: bool = True) -> None:
        self.gradient_checkpointing = enable


def _make_base_causal_mask(
    batch_size: int,
    query_len: int,
    total_len: int,
    device: torch.device,
    dtype: torch.dtype,
    window: Optional[int] = None,
) -> torch.Tensor:
    q_positions = torch.arange(total_len - query_len, total_len, device=device)
    k_positions = torch.arange(total_len, device=device)
    diff = q_positions[:, None] - k_positions[None, :]
    mask = torch.full((query_len, total_len), float("-inf"), device=device)
    mask = torch.where(diff >= 0, torch.zeros_like(mask), mask)
    if window is not None and window > 0:
        mask = torch.where(diff > window, float("-inf"), mask)
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
    return mask.to(dtype)


def _apply_attention_mask_bias(
    mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if attention_mask is None:
        return mask
    if attention_mask.dim() == 2:
        attention_mask = attention_mask[:, None, None, :]
    attention_mask = attention_mask.to(mask.dtype)
    mask = mask + (1.0 - attention_mask) * float("-inf")
    return mask


def create_causal_mask(
    config: Any,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    position_ids: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    batch, query_len = input_embeds.shape[:2]
    past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
    total_len = past_len + query_len
    mask = _make_base_causal_mask(batch, query_len, total_len, input_embeds.device, torch.float32)
    mask = _apply_attention_mask_bias(mask, attention_mask)
    return mask.to(input_embeds.dtype)


def create_sliding_window_causal_mask(
    config: Any,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    position_ids: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    window = getattr(config, "sliding_window", None)
    batch, query_len = input_embeds.shape[:2]
    past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
    total_len = past_len + query_len
    mask = _make_base_causal_mask(batch, query_len, total_len, input_embeds.device, torch.float32, window=window)
    mask = _apply_attention_mask_bias(mask, attention_mask)
    return mask.to(input_embeds.dtype)


def dynamic_rope_update(func: Callable) -> Callable:
    """Decorator placeholder for compatibility with HF dynamic RoPE APIs."""

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def _default_rope_init(config: Any, device: Optional[torch.device]) -> Tuple[torch.Tensor, float]:
    dim = getattr(config, "head_dim", None) or getattr(config, "hidden_size") // getattr(config, "num_attention_heads")
    base = getattr(config, "rope_theta", 10000)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    return inv_freq, 1.0


ROPE_INIT_FUNCTIONS: Dict[str, Callable[[Any, Optional[torch.device]], Tuple[torch.Tensor, float]]] = {
    "default": _default_rope_init,
    "linear": _default_rope_init,
    "llama3": _default_rope_init,
    "mrope": _default_rope_init,
}


def register_rope_init(name: str, fn: Callable[[Any, Optional[torch.device]], Tuple[torch.Tensor, float]]) -> None:
    ROPE_INIT_FUNCTIONS[name] = fn


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


ACT2FN: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": F.relu,
    "silu": F.silu,
    "swish": F.silu,
    "gelu": F.gelu,
    "gelu_new": gelu_new,
    "gelu_fast": F.gelu,
    "gelu_pytorch_tanh": gelu_new,
    "identity": lambda x: x,
    "linear": lambda x: x,
    "tanh": torch.tanh,
}


ALL_ATTENTION_FUNCTIONS: Dict[str, Callable[..., Tuple[torch.Tensor, Optional[torch.Tensor]]]] = {}


def register_attention_backend(name: str, fn: Callable[..., Tuple[torch.Tensor, Optional[torch.Tensor]]]) -> None:
    ALL_ATTENTION_FUNCTIONS[name] = fn

