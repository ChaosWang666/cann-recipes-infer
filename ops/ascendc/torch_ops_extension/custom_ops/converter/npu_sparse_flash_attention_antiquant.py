# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
import torchair
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, \
    BOOL, Support
from torchair.ge import attr


# 为自定义算子注册converter，用于torch.compile 场景成图
# 注意： meta_outputs形参名为固定写法，若写错会影响ge节点的输出dtype与shape推导
@register_fx_node_ge_converter(torch.ops.custom.npu_sparse_flash_attention_antiquant.default)
def convert_npu_sparse_flash_attention_antiquant(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    sparse_indices: Tensor,
    scale_value: float,
    sparse_block_size: int,
    key_quant_mode: int,
    value_quant_mode: int,
    *,
    key_dequant_scale: Optional[Tensor] = None,
    value_dequant_scale: Optional[Tensor] = None,
    block_table: Optional[Tensor] = None,
    actual_seq_lengths_query: Optional[Tensor] = None,
    actual_seq_lengths_kv: Optional[Tensor] = None,
    layout_query: str = "BSND",
    layout_kv: str = "BSND",
    sparse_mode: int = 3,
    attention_mode: int = 0,
    quant_scale_repo_mode: int = 0,
    tile_size: int = 0,
    rope_head_dim: int = 0,
    meta_outputs: TensorSpec = None,
):
    return torchair.ge.custom_op(
        "SparseFlashAttentionAntiquant",
        inputs={"query": query, 
                "key": key,
                "value": value,
                "sparse_indices": sparse_indices,
                "key_dequant_scale": key_dequant_scale,
                "value_dequant_scale": value_dequant_scale,
                "block_table": block_table,
                "actual_seq_lengths_query": actual_seq_lengths_query,
                "actual_seq_lengths_kv": actual_seq_lengths_kv,
               },
        attrs={"scale_value": attr.Float(scale_value),
               "sparse_block_size": attr.Int(sparse_block_size),
               "key_quant_mode": attr.Int(key_quant_mode),
               "value_quant_mode": attr.Int(value_quant_mode),
               "layout_query": attr.Str(layout_query),
               "layout_kv": attr.Str(layout_kv),
               "sparse_mode": attr.Int(sparse_mode),
               "attention_mode": attr.Int(attention_mode),
               "quant_scale_repo_mode": attr.Int(quant_scale_repo_mode),
               "tile_size": attr.Int(tile_size),
               "rope_head_dim": attr.Int(rope_head_dim),
               },
        outputs=['attention_out']
    )
