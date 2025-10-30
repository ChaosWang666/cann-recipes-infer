# coding=utf-8
"""Configuration definitions for the Qwen2.5-VL Ascend recipe."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Qwen2_5_VLTextConfig(PretrainedConfig):
    model_type = "qwen2_5_vl_text"

    def __init__(
        self,
        vocab_size: int = 152064,
        hidden_size: int = 8192,
        intermediate_size: int = 28672,
        num_hidden_layers: int = 80,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        use_sliding_window: bool = False,
        sliding_window: int = 4096,
        layer_types: Optional[List[str]] = None,
        attn_implementation: str = "sdpa",
        pad_token_id: int = 151643,
        bos_token_id: int = 151643,
        eos_token_id: int = 151645,
        tie_word_embeddings: bool = False,
        use_cache: bool = True,
        use_return_dict: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            is_decoder=True,
            use_cache=use_cache,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        if self.rope_scaling is not None:
            rope_config_validation(self)
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.layer_types = layer_types or ["attention"] * num_hidden_layers
        self._attn_implementation = attn_implementation
        self.use_cache = use_cache
        self.use_return_dict = use_return_dict
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        # Additional attributes that some checkpoints rely on.
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.tie_word_embeddings = tie_word_embeddings


class Qwen2_5_VLVisionConfig(PretrainedConfig):
    model_type = "qwen2_5_vl_vision"

    def __init__(
        self,
        hidden_size: int = 1664,
        out_hidden_size: int = 2048,
        num_heads: int = 26,
        depth: int = 32,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        temporal_patch_stride: int = 1,
        in_channels: int = 3,
        image_size: int = 560,
        spatial_merge_size: int = 2,
        window_size: int = 112,
        num_register_tokens: int = 8,
        tokens_per_second: int = 25,
        fullatt_block_indexes: Optional[List[int]] = None,
        final_layernorm_eps: float = 1e-6,
        attn_dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(is_encoder=True, use_cache=use_cache, **kwargs)
        self.hidden_size = hidden_size
        self.out_hidden_size = out_hidden_size
        self.num_heads = num_heads
        self.depth = depth
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.temporal_patch_stride = temporal_patch_stride
        self.in_channels = in_channels
        self.image_size = image_size
        self.spatial_merge_size = spatial_merge_size
        self.window_size = window_size
        self.num_register_tokens = num_register_tokens
        self.tokens_per_second = tokens_per_second
        self.fullatt_block_indexes = fullatt_block_indexes or []
        self.final_layernorm_eps = final_layernorm_eps
        self.attn_dropout = attn_dropout
        self.mlp_ratio = mlp_ratio
        self.use_cache = use_cache


class Qwen2_5_VLConfig(PretrainedConfig):
    model_type = "qwen2_5_vl"

    def __init__(
        self,
        text_config: Optional[Union[Qwen2_5_VLTextConfig, Dict[str, Any]]] = None,
        vision_config: Optional[Union[Qwen2_5_VLVisionConfig, Dict[str, Any]]] = None,
        image_token_id: int = 151652,
        video_token_id: int = 151653,
        vision_start_token_id: int = 151651,
        image_token_index: int = -200,
        video_token_index: int = -201,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        pad_token_id: int = 151643,
        bos_token_id: int = 151643,
        eos_token_id: int = 151645,
        projection_dim: int = 8192,
        max_position_embeddings: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            is_decoder=True,
            use_cache=use_cache,
            **kwargs,
        )

        if isinstance(text_config, dict) or text_config is None:
            text_config = text_config or {}
            self.text_config = Qwen2_5_VLTextConfig(**text_config)
        elif isinstance(text_config, Qwen2_5_VLTextConfig):
            self.text_config = text_config
        else:
            raise TypeError("text_config must be a dict or Qwen2_5_VLTextConfig instance")

        if isinstance(vision_config, dict) or vision_config is None:
            vision_config = vision_config or {}
            self.vision_config = Qwen2_5_VLVisionConfig(**vision_config)
        elif isinstance(vision_config, Qwen2_5_VLVisionConfig):
            self.vision_config = vision_config
        else:
            raise TypeError("vision_config must be a dict or Qwen2_5_VLVisionConfig instance")

        self.tie_word_embeddings = tie_word_embeddings
        self.use_cache = use_cache
        self.vocab_size = self.text_config.vocab_size
        self.hidden_size = self.text_config.hidden_size
        self.max_position_embeddings = max_position_embeddings or self.text_config.max_position_embeddings
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.image_token_index = image_token_index
        self.video_token_index = video_token_index
        self.projection_dim = projection_dim
        self.output_attentions = getattr(self.text_config, "output_attentions", False)
        self.output_hidden_states = getattr(self.text_config, "output_hidden_states", False)
        self.use_return_dict = getattr(self.text_config, "use_return_dict", True)
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.layer_types = self.text_config.layer_types

        if self.text_config.pad_token_id != pad_token_id:
            logger.warning(
                "Overriding text_config.pad_token_id (%s) with top-level pad_token_id (%s)",
                self.text_config.pad_token_id,
                pad_token_id,
            )
            self.text_config.pad_token_id = pad_token_id

        if self.text_config.bos_token_id != bos_token_id:
            self.text_config.bos_token_id = bos_token_id
        if self.text_config.eos_token_id != eos_token_id:
            self.text_config.eos_token_id = eos_token_id


__all__ = [
    "Qwen2_5_VLConfig",
    "Qwen2_5_VLTextConfig",
    "Qwen2_5_VLVisionConfig",
]
