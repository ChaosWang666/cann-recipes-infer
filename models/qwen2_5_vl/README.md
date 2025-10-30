# Qwen2.5-VL-72B模型在NPU实现低时延推理

## 概述
Qwen2.5-VL-72B-Instruct是阿里通义千问团队发布的多模态大模型，能够同时处理图像和文本。本样例基于Qwen2.5-VL官方权重，在昇腾NPU环境中完成推理流程的适配与性能优化，整体目录结构与`qwen3_moe`样例保持一致，但针对非MoE的视觉语言模型做了算子调优与图模式适配。

## 支持的产品型号
<term>Atlas A3 系列产品</term>

## 环境准备
1. 安装CANN软件包。

   样例工程依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的软件版本为`CANN 8.3.RC1.alpha002`。请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.3.RC1.alpha002)获取`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Ascend-cann-kernels-${chip_type}_${version}_linux-${arch}.run`，并参考[官方安装指引](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit)完成安装。

2. 安装Ascend Extension for PyTorch（torch_npu）。

   torch_npu是PyTorch框架运行在昇腾NPU上的适配插件，本样例要求Ascend Extension for PyTorch版本`7.2.RC1.alpha002`、PyTorch版本`2.6.0`。请从[源码仓](https://gitee.com/ascend/pytorch/tree/v7.2.RC1.alpha002-pytorch2.6.0)下载对应源码并按照[编译指南](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0005.html)完成安装。

3. 下载项目源码并安装依赖Python库。

   ```bash
   git clone https://gitcode.com/cann/cann-recipes-infer.git
   cd cann-recipes-infer/models/qwen2_5_vl
   pip3 install -r requirements.txt
   ```

4. 配置环境变量。

   修改`executor/scripts/set_env.sh`中的如下字段：
   - `IPs`：所有节点的IP地址，按rank id顺序填写，多个节点使用空格分隔，例如`('192.168.0.1' '192.168.0.2')`。
   - `recipes_path`：当前代码仓根目录，例如`/home/cann-recipes-infer`。
   - `cann_path`：CANN软件包安装路径，例如`/usr/local/Ascend/ascend-toolkit/latest`。
   - `driver_path`：固件驱动安装路径，例如`/usr/local/Ascend/driver`。
   > 说明：HCCL相关环境变量（如`HCCL_SOCKET_IFNAME`、`HCCL_OP_EXPANSION_MODE`）可参考[集合通信文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/maintenref/envvar/envref_07_0001.html#ZH-CN_TOPIC_0000002449945377__section163522499503)在`executor/scripts/function.sh`中自行配置。

## 权重准备
请提前下载Qwen2.5-VL-72B-Instruct原始权重，假设存放在`/data/models/origin/Qwen2.5-VL-72B-Instruct`目录，并在推理配置文件中填写该路径。

## 数据集准备
样例默认读取`dataset/default_prompt.json`中的文本prompt。若需要复现视觉多模态场景，可在`dataset`目录下创建`vision_arena_bench`文件夹，并放置`dataset.jsonl`（或同结构的JSON文件）以及对应的图像文件。推理脚本会自动从该目录读取数据。

## 推理执行
1. 配置YAML文件。

   `models/qwen2_5_vl/config/qwen2_5_vl_72b.yaml`给出了16卡推理的默认配置。请根据实际情况修改如下参数：
   - `model_path`：设置为前述权重所在目录。
   - `data_config.dataset`：可设置为`"default"`使用内置文本prompt，或设置为`"vision-arena-bench-v0.1"`加载本地视觉数据集。

2. 运行推理脚本。

   ```shell
   cd models/qwen2_5_vl
   bash infer.sh ./config/qwen2_5_vl_72b.yaml
   ```

   > 注意：当前样例已启用图模式decode优化，prefill阶段仍采用eager模式；默认支持TP=8的张量并行分片策略，并自动完成通信组初始化。

## 其他说明
有关模型结构、并行切分与优化细节，可参考目录下的`models`子模块。若需自定义推理流程，可基于`runner_qwen2_5_vl.py`扩展逻辑。
