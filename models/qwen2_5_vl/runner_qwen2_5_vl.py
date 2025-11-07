# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging
import os
import time
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor

from models.qwen2_5_vl.models.configuration_qwen2_5_vl import Qwen2_5_VLConfig

from executor.model_runner import ModelRunner
from models.qwen2_5_vl.models.modeling_qwen2_5_vl import Qwen2_5_VLForCausalLM

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)

torch.manual_seed(42)
torch.npu.manual_seed_all(42)


def _format_text_content(text: str) -> List[Dict[str, Any]]:
    return [{"type": "text", "text": text}]


def _ensure_conversation(prompt: Any) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """Convert the prompt to Qwen2.5-VL chat format."""
    if isinstance(prompt, dict) and "messages" in prompt:
        messages = prompt["messages"]
        images = prompt.get("images", [])
    elif isinstance(prompt, dict) and "conversations" in prompt:
        messages = prompt["conversations"]
        images = prompt.get("images", [])
    elif isinstance(prompt, str):
        messages = [{"role": "user", "content": _format_text_content(prompt)}]
        images = []
    else:
        raise ValueError("Unsupported prompt type for Qwen2.5-VL inference")

    # Normalize message content to list of dicts with type/text entries.
    normalized_messages = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content")
        if isinstance(content, str):
            content = _format_text_content(content)
        elif isinstance(content, list):
            new_content = []
            for item in content:
                if isinstance(item, str):
                    new_content.append({"type": "text", "text": item})
                elif isinstance(item, dict):
                    new_content.append(item)
            content = new_content
        else:
            content = _format_text_content(str(content))
        normalized_messages.append({"role": role, "content": content})

    return normalized_messages, images


def _load_image(image_item: Any) -> Any:
    if image_item is None:
        return None
    if isinstance(image_item, str):
        if os.path.exists(image_item):
            return Image.open(image_item).convert("RGB")
        logging.warning("Image path %s not found, skip it.", image_item)
        return None
    return image_item


class Qwen2_5_VLRunner(ModelRunner):
    def __init__(self, runner_settings):
        super().__init__(runner_settings)
        self.processor = None

    def init_model(self):
        super().init_model(Qwen2_5_VLForCausalLM, Qwen2_5_VLConfig)
        self.model.eval()
        self.init_processor()

    def init_processor(self):
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        if hasattr(self.processor, "tokenizer"):
            self.tokenizer = self.processor.tokenizer
        else:
            super().init_tokenizer()

    def compile_model(self):
        logging.info("The final model structure is: \n %s", self.model)
        if "graph" in self.execute_mode:
            logging.info("Compile decode path for graph execution.")
            self.graph_compile()

    def graph_compile(self):
        import torchair as tng
        from torchair.configs.compiler_config import CompilerConfig

        compiler_config = CompilerConfig()
        compiler_config.experimental_config.frozen_parameter = True
        compiler_config.experimental_config.tiling_schedule_optimize = True
        npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
        self.model.decode = torch.compile(self.model.decode, dynamic=True, fullgraph=True, backend=npu_backend)

    def _prepare_processor_inputs(self, prompts: List[Any]):
        conversations = []
        images_batch = []
        for prompt in prompts:
            conversation, images = _ensure_conversation(prompt)
            conversations.append(conversation)
            images_batch.append([img for img in ([_load_image(item) for item in images]) if img is not None])
        text_batch = [
            self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in conversations
        ]
        inputs = self.processor(
            text=text_batch,
            images=images_batch,
            padding="longest",
            truncation=True,
            max_length=self.input_max_len,
            return_tensors="pt"
        )
        for key, value in list(inputs.items()):
            if torch.is_tensor(value):
                inputs[key] = value.to(self.device)
        return inputs

    def _run_prefill(self, inputs: Dict[str, Any]):
        torch.npu.synchronize()
        start_time = time.time()
        outputs = self.model.prefill(**inputs)
        torch.npu.synchronize()
        end_time = time.time()
        return outputs, end_time - start_time

    def _run_decode(self, inputs: Dict[str, Any]):
        torch.npu.synchronize()
        start_time = time.time()
        outputs = self.model.decode(**inputs)
        torch.npu.synchronize()
        end_time = time.time()
        return outputs, end_time - start_time

    def model_generate(self, prompts, warm_up=False):
        inputs = self._prepare_processor_inputs(prompts)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        pixel_values = inputs.get("pixel_values")
        pixel_attention_mask = inputs.get("pixel_attention_mask")

        prefill_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
            "use_cache": True,
        }
        prefill_inputs = {k: v for k, v in prefill_inputs.items() if v is not None}
        prefill_outputs, prefill_time = self._run_prefill(prefill_inputs)
        logits = prefill_outputs.logits
        past_key_values = prefill_outputs.past_key_values
        next_tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids = torch.cat([input_ids, next_tokens], dim=-1)
        current_attention_mask = attention_mask
        if current_attention_mask is not None:
            current_attention_mask = torch.cat([current_attention_mask, torch.ones_like(next_tokens)], dim=-1)

        decode_times: List[float] = []
        max_decode_steps = self.max_new_tokens
        if warm_up:
            max_decode_steps = min(self.max_new_tokens, 2)
        for step in range(max_decode_steps - 1):
            model_inputs = self.model.prepare_inputs_for_generation(
                next_tokens,
                past_key_values=past_key_values,
                attention_mask=current_attention_mask,
                use_cache=True
            )
            if pixel_values is not None and "pixel_values" not in model_inputs:
                model_inputs["pixel_values"] = pixel_values
            if pixel_attention_mask is not None and "pixel_attention_mask" not in model_inputs:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask
            model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
            decode_outputs, decode_time = self._run_decode(model_inputs)
            decode_times.append(decode_time)
            logits = decode_outputs.logits
            past_key_values = decode_outputs.past_key_values
            next_tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)
            if current_attention_mask is not None:
                current_attention_mask = torch.cat([current_attention_mask, torch.ones_like(next_tokens)], dim=-1)

        if not warm_up:
            avg_decode = sum(decode_times) / max(1, len(decode_times)) if decode_times else 0.0
            logging.info("Prefill latency: %.2f ms", prefill_time * 1000)
            logging.info("Average decode latency: %.2f ms", avg_decode * 1000)

        outputs = generated_ids[:, input_ids.size(1):]
        outputs = outputs.clip(0, self.model.config.vocab_size - 1)
        result = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if isinstance(result, list):
            logging.info("Inference decode result for batch 0: \n%s", result[0])
        else:
            logging.info("Inference decode result: \n%s", result)
        return result
