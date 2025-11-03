# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import logging
from executor.utils.common_utils import update_common_vars, check_common_parallel_settings
from executor.utils.common_utils import update_settings


def update_vars(world_size, runner_settings):
    update_common_vars(world_size, runner_settings)
    embed_tp_size = runner_settings.get("parallel_config").get("embed_tp_size", 1)
    batch_size = runner_settings.get("data_config").get("batch_size", 1)
    batch_size_per_rank = max(1, batch_size // max(1, world_size // embed_tp_size))
    runner_settings = update_settings(runner_settings, "data_config", "batch_size_per_rank", batch_size_per_rank)
    if runner_settings.get("exe_mode") == "acl_graph" and os.getenv("TASK_QUEUE_ENABLE", "2") != "1":
        os.environ["TASK_QUEUE_ENABLE"] = "1"
    else:
        os.environ["TASK_QUEUE_ENABLE"] = "2"


def check_model_settings(world_size, runner_settings):
    exe_mode = runner_settings.get("exe_mode")
    enable_cache_compile = runner_settings.get("model_config").get("enable_cache_compile", False)

    if exe_mode not in ["ge_graph", "eager", "acl_graph"]:
        raise ValueError(f"{exe_mode=} does not supported! Only the eager, ge_graph and acl_graph mode are supported!")

    if exe_mode == "eager" and enable_cache_compile:
        logging.info("Cache compile is not required when running in eager mode, disable enable_cache_compile automatically.")
        runner_settings.get("model_config")["enable_cache_compile"] = False


def check_parallel_settings(world_size, runner_settings):
    check_common_parallel_settings(world_size, runner_settings)
    attn_tp_size = runner_settings.get("parallel_config").get("attn_tp_size", 1)
    embed_tp_size = runner_settings.get("parallel_config").get("embed_tp_size", 1)
    lmhead_tp_size = runner_settings.get("parallel_config").get("lmhead_tp_size", 1)

    if world_size % attn_tp_size != 0:
        raise ValueError(f"world_size({world_size}) must be divisible by attn_tp_size({attn_tp_size})")
    if world_size % embed_tp_size != 0:
        raise ValueError(f"world_size({world_size}) must be divisible by embed_tp_size({embed_tp_size})")
    if world_size % lmhead_tp_size != 0:
        raise ValueError(f"world_size({world_size}) must be divisible by lmhead_tp_size({lmhead_tp_size})")


def check_vars(world_size, runner_settings):
    check_parallel_settings(world_size, runner_settings)
    check_model_settings(world_size, runner_settings)
