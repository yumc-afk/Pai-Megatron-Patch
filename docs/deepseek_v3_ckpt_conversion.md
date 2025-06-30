# DeepSeek-V3 权重转换指南

本文档介绍如何将官方发布的 DeepSeek-V3 FP8 HuggingFace 权重先转换为 BF16，随后再生成 Megatron-Core 分布式权重。

## 1. FP8 权重转 BF16
在 `toolkits/model_checkpoints_convertor/deepseek` 目录下提供了 `fp8_cast_bf16.py` 脚本，可将 FP8 权重转换为 BF16：

```bash
pip install torch safetensors triton tqdm
python fp8_cast_bf16.py --input-fp8-hf-path /mnt/deepseek-ckpts/DeepSeek-V3 \
                        --output-bf16-hf-path /mnt/deepseek-ckpts/DeepSeek-V3-bf16
```
上述命令会在指定目录生成 BF16 版本的权重。

## 2. BF16 权重转 Megatron 分布式权重
进入 `toolkits/distributed_checkpoints_convertor`，使用 `run_32xH20.sh` 脚本完成转换。示例环境变量与执行命令如下：

```bash
export WORLD_SIZE=32       # 总进程数
export RANK=0              # 当前节点编号
export MASTER_ADDR=host0   # 主节点地址
export MASTER_PORT=6000    # 主节点端口

cd toolkits/distributed_checkpoints_convertor
bash scripts/deepseek_v3/run_32xH20.sh \
    A37B \
    /mnt/deepseek-ckpts/DeepSeek-V3-bf16 \
    /mnt/deepseek-ckpts/DeepSeek-V3-to-mcore \
    false \
    true \
    bf16
```
脚本将使用以上环境变量启动 `torchrun`，在 32 张 GPU（4 节点 × 8 GPU）上生成 Megatron-Core 权重，保存在 `/mnt/deepseek-ckpts/DeepSeek-V3-to-mcore`。

## 3. 转换时间预估
基于阿里云 CPFS 文件系统的 4 机 32 卡环境，`fp8_cast_bf16.py` 脚本处理 DeepSeek-V3-671B 大约需要 10 分钟。随后使用 `run_32xH20.sh` 将 BF16 权重转换为 Megatron-Core 分布式权重约需 5 分钟，具体耗时会随存储与网络性能略有波动。
