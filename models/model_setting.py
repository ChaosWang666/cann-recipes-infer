# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Helper that dispatches model specific settings updates."""

from importlib import import_module
from typing import Any, Dict


_MODEL_SETTING_MODULES = {
    "qwen3-moe": "models.qwen3_moe.models.model_setting",
    "qwen3_moe": "models.qwen3_moe.models.model_setting",
    "qwen2.5-vl": "models.qwen2_5_vl.models.model_setting",
    "qwen2_5_vl": "models.qwen2_5_vl.models.model_setting",
}


def _resolve_module(runner_settings: Dict[str, Any]):
    model_name = runner_settings.get("model_name", "").lower()
    module_path = _MODEL_SETTING_MODULES.get(model_name)
    if module_path is None:
        raise ValueError(f"Unsupported model_name `{runner_settings.get('model_name')}` for model settings")
    return import_module(module_path)


def update_vars(world_size: int, runner_settings: Dict[str, Any]):
    module = _resolve_module(runner_settings)
    return module.update_vars(world_size, runner_settings)


def check_vars(world_size: int, runner_settings: Dict[str, Any]):
    module = _resolve_module(runner_settings)
    return module.check_vars(world_size, runner_settings)
