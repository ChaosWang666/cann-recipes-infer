# coding=utf-8
# Adapted from  
# https://github.com/Tencent-Hunyuan/HunyuanVideo,
# Copyright (c) Huawei Technologies Co., Ltd.
# Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
#
# This code is based on Tencent-Hunyuan's HunyuanVideo library and the HunyuanVideo
# implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to HunyuanVideo used by Tencent-Hunyuan team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
from pathlib import Path
from datetime import datetime

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.distributed as dist
from loguru import logger

import hyvideo.monkey_patch
from hyvideo.utils.file_utils import save_videos_grid, load_json
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.cache.teacache import add_teacache_class

torch_npu.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False


def main():
    args = parse_args()
    logger.add(os.path.join(args.save_path, "logs/hy_{time:YYYY-MM-DD}.log"),
           rotation="00:00",
           retention="7 days",
           compression="zip",
           encoding="utf-8",
           level="DEBUG")
    logger.info(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix == "" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    
    # Get the updated args
    args = hunyuan_video_sampler.args
    
    #TeaCache
    if args.teacache:
        add_teacache_class(hunyuan_video_sampler.pipeline.transformer, args)

    # Start sampling
    # TODO: batch inference check
    if args.prompt_path:
        full_prompts_info = load_json(args.prompt_path)
        prompts = []
        for prompt_dict in full_prompts_info:
            prompts.append(prompt_dict['prompt_en'])
    else:
        prompts = [args.prompt]
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
    for prompt in prompts:
        logger.info('prompt: ', prompt)
        outputs = hunyuan_video_sampler.predict(
            prompt=prompt, 
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            seed=args.seed,
            negative_prompt=args.neg_prompt,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            num_videos_per_prompt=args.num_videos,
            flow_shift=args.flow_shift,
            batch_size=args.batch_size,
            embedded_guidance_scale=args.embedded_cfg_scale
        )
        samples = outputs['samples']

        if args.teacache:
            skip_cnt = hunyuan_video_sampler.pipeline.transformer.teacache_cnt
            logger.info(f'Total skip {skip_cnt}/{args.infer_steps} iter by TeaCache')
        
        # Save samples
        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
            for i, sample in enumerate(samples):
                sample = samples[i].unsqueeze(0)
                cur_save_path = os.path.join(
                            save_path,
                            f"{time_flag}_seed{outputs['seeds'][i]}/{outputs['prompts'][i][:100].replace('/','')}.mp4"
                )
                save_videos_grid(sample, cur_save_path, fps=24)
                logger.info(f'Sample save to: {cur_save_path}')

if __name__ == "__main__":
    main()