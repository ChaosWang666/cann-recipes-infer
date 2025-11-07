# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import json
import os
import logging
from datasets import load_dataset  # requires version == 3.6.0
from typing import Any, Dict, Iterable, List, Optional

try:
    from PIL import Image as PILImage
except ImportError:  # pragma: no cover - PIL is available in deployment images
    PILImage = None


def load_infinitebench_dataset(data_path):
    prompts = []
    datasets = ["longbook_qa_eng.jsonl"]
    data = load_dataset(data_path, data_files=datasets, split="train", trust_remote_code=True)
    for d in data:
        prompts.append(d['context'])
    return prompts


def load_longbench_dataset(data_path):
    prompts = []
    datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    datasets_e = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", "trec", \
                  "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    datasets_e = [item + "_e" for item in datasets_e]

    for dataset in datasets + datasets_e:
        data = load_dataset(data_path, dataset, split='test', trust_remote_code=True)
        for d in data:
            prompts.append(d['context'])
    return prompts


def _normalize_vision_prompt(entry: Dict[str, Any], image_root: Optional[str] = None) -> Dict[str, Any]:
    if "messages" in entry:
        messages = entry["messages"]
    elif "conversations" in entry:
        messages = entry["conversations"]
    else:
        question = entry.get("question") or entry.get("prompt") or entry.get("text") or ""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question,
                    }
                ],
            }
        ]
    images_field = entry.get("images") or entry.get("image") or []

    def _collect_image_references(value: Any) -> List[Any]:
        if value is None:
            return []

        # Already a PIL image instance – return directly.
        if PILImage is not None and isinstance(value, PILImage.Image):
            return [value]

        if isinstance(value, (str, os.PathLike)):
            image_path = os.fspath(value)
            if image_root and not os.path.isabs(image_path):
                image_path = os.path.join(image_root, image_path)
            return [image_path]

        if hasattr(value, "filename") and getattr(value, "filename"):
            return _collect_image_references(getattr(value, "filename"))
        if hasattr(value, "path") and getattr(value, "path"):
            return _collect_image_references(getattr(value, "path"))

        if isinstance(value, dict):
            collected: List[Any] = []
            for key in ("path", "image", "value", "url", "file", "bytes", "data"):
                if key in value:
                    collected.extend(_collect_image_references(value[key]))
            if collected:
                return collected
            return []

        if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
            collected: List[Any] = []
            for item in value:
                collected.extend(_collect_image_references(item))
            return collected

        # Fallback: keep original object (e.g. NumPy arrays, raw bytes).
        return [value]

    normalized_images: List[Any] = _collect_image_references(images_field)
    return {"messages": messages, "images": normalized_images}


def load_vision_arena_bench(dataset_path: str) -> List[Dict[str, Any]]:
    abs_dataset_path = os.path.abspath(dataset_path)
    bench_dirs = []
    for candidate in (
        abs_dataset_path,
        os.path.join(abs_dataset_path, "vision-arena-bench-v0.1"),
    ):
        if candidate not in bench_dirs:
            bench_dirs.append(candidate)

    def _load_jsonl(jsonl_path: str, image_root: str) -> List[Dict[str, Any]]:
        prompts: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    prompts.append(_normalize_vision_prompt(json.loads(line), image_root))
        return prompts

    def _load_json(json_path: str, image_root: str) -> List[Dict[str, Any]]:
        with open(json_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, list):
            entries = payload
        elif isinstance(payload, dict):
            for key in ("data", "instances", "samples", "records"):
                if isinstance(payload.get(key), list):
                    entries = payload[key]
                    break
            else:
                entries = [payload]
        else:
            raise ValueError(f"Unsupported data format in {json_path}")
        return [_normalize_vision_prompt(entry, image_root) for entry in entries]

    def _load_parquet_files(parquet_files: List[str], image_root: Optional[str]) -> List[Dict[str, Any]]:
        dataset = load_dataset("parquet", data_files={"train": parquet_files}, split="train")
        return [_normalize_vision_prompt(entry, image_root) for entry in dataset]

    for bench_dir in bench_dirs:
        if not os.path.isdir(bench_dir):
            continue

        candidates = [
            os.path.join(bench_dir, "vision_arena_bench.jsonl"),
            os.path.join(bench_dir, "dataset.json"),
            os.path.join(bench_dir, "vision_arena_bench.json"),
        ]
        prompts: List[Dict[str, Any]] = []
        for candidate in candidates:
            if os.path.exists(candidate):
                if candidate.endswith(".jsonl"):
                    prompts = _load_jsonl(candidate, bench_dir)
                else:
                    prompts = _load_json(candidate, bench_dir)
                break

        if not prompts:
            parquet_files: List[str] = []
            for root_dir, _, files in os.walk(bench_dir):
                for filename in files:
                    if filename.lower().endswith(".parquet"):
                        parquet_files.append(os.path.join(root_dir, filename))
            if parquet_files:
                parquet_files.sort()
                prompts = _load_parquet_files(parquet_files, bench_dir)

        if prompts:
            return prompts

    raise FileNotFoundError(
        "vision_arena_bench dataset not found or empty. Please verify the local dataset path."
    )


def generate_default_prompt(dataset_dir):
    json_path = os.path.join(dataset_dir, "default_prompt.json")
    json_path = os.path.abspath(json_path)
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            text = data["text"]
    except FileNotFoundError as e:
        raise FileNotFoundError(f"prompt error: prompt file({json_path}) not find.") from e
    except json.JSONDecodeError as e:
        logging.error(f"prompt error: the json format of prompt file({json_path}) is incorrect.")
        raise e
    except Exception as e:
        raise e
    preset_prompts = [
        text,
    ]
    return preset_prompts


def get_prompts_for_cur_rank(preset_prompts, global_bs, batch_size_per_rank, global_dp_rank):
    preset_prompts = preset_prompts * (global_bs // len(preset_prompts) + 1)
    preset_prompts = preset_prompts[global_dp_rank * batch_size_per_rank: (global_dp_rank + 1) * batch_size_per_rank]
    query_id_list = list(range(global_dp_rank * batch_size_per_rank, (global_dp_rank + 1) * batch_size_per_rank))
    logging.info(f"prompt batch size: {len(preset_prompts)}/{global_bs}, {query_id_list=}")
    return (preset_prompts, query_id_list)


def generate_prompt(runner_settings):
    batch_size = runner_settings.get("data_config").get("batch_size", 1)
    attn_tp_size = runner_settings.get("parallel_config").get("attn_tp_size", 1)
    cp_size = runner_settings.get("parallel_config").get("cp_size", 1)
    global_rank = int(os.getenv("RANK_ID", 0))
    global_dp_rank = global_rank // cp_size // attn_tp_size
    bs_per_cp_group = runner_settings.get("data_config").get("bs_per_cp_group", 1)
    batch_size_per_rank = bs_per_cp_group if cp_size > 1 \
        else runner_settings.get("data_config").get("batch_size_per_rank", 1)
    dataset = runner_settings.get("data_config").get("dataset", "default")

    cur_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(cur_dir, "../../dataset")
    if dataset == "default":
        preset_prompts = generate_default_prompt(dataset_path)
    elif dataset == "LongBench":
        dataset_path = os.path.abspath(os.path.join(dataset_path, f"{dataset}"))
        if os.path.isdir(dataset_path): # use local LongBench dataset first
            dataset = dataset_path
        else:
            dataset = "THUDM/LongBench"
        preset_prompts = load_longbench_dataset(dataset)
    elif dataset == "InfiniteBench":
        dataset_path = os.path.abspath(os.path.join(dataset_path, f"{dataset}"))
        if os.path.isdir(dataset_path): # use local InfiniteBench dataset first
            dataset = dataset_path
        preset_prompts = load_infinitebench_dataset(dataset)
    elif dataset == "vision-arena-bench-v0.1":
        preset_prompts = load_vision_arena_bench(dataset_path)
    else:
        raise Exception(
            f"your dataset {dataset} is not supported, dataset supported: LongBench, InfiniteBench, "
            "vision-arena-bench-v0.1"
        )
    return get_prompts_for_cur_rank(preset_prompts, batch_size, batch_size_per_rank, global_dp_rank)


def build_dataset_input(tokenizer, prompts, input_max_len):
    prompts_inputids = tokenizer(prompts).input_ids
    out_prompts = []
    for prompt_inputids in prompts_inputids:
        prompt = "Please read a part of the book below, and then give me the summary.\n[start of the book]\n" + \
            tokenizer.decode(prompt_inputids[:input_max_len - 70], skip_special_tokens=True) + \
            "\n[end of the book]\n\nNow you have read it. Please summarize it for me. " + \
            "First, tell me the title and the author, and then tell the story in 400 words.\n\n"
        out_prompts.append(prompt)
    return out_prompts
