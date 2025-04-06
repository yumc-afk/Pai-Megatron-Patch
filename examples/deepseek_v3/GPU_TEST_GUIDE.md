# DeepSeek V3 MLA/MOE GPU测试指南

本指南详细介绍如何在GPU环境下测试DeepSeek V3模型的MLA（Multi-Latent Attention）和MOE（Mixture of Experts）组件实现。这些测试需要NVIDIA GPU硬件和CUDA支持，用于验证模型在实际训练和推理场景中的正确性。

## 目录

1. [环境要求](#环境要求)
2. [环境安装](#环境安装)
3. [测试脚本概述](#测试脚本概述)
4. [运行测试](#运行测试)
5. [测试结果解读](#测试结果解读)
6. [常见问题](#常见问题)
7. [高级配置](#高级配置)

## 环境要求

- NVIDIA GPU（建议至少8GB显存）
- CUDA 11.8+
- Python 3.8+
- PyTorch 2.0+（CUDA版本）
- 至少50GB磁盘空间（用于模型权重和测试数据）

## 环境安装

### 自动安装

我们提供了自动安装脚本，可以一键安装所有必要的依赖：

```bash
# 克隆仓库（如果尚未克隆）
git clone https://github.com/haoweiliang1996/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch

# 运行安装脚本
bash examples/deepseek_v3/install_gpu_test.sh

# 设置环境变量
source examples/deepseek_v3/setup_gpu_env.sh
```

### 手动安装

如果您希望手动安装依赖，请确保安装以下组件：

1. PyTorch（CUDA版本）
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. Transformers库和其他依赖
   ```bash
   pip install transformers huggingface_hub numpy tqdm matplotlib tensorboard
   ```

3. Megatron-LM依赖
   ```bash
   pip install regex nltk sentencepiece
   ```

4. （可选）NVIDIA APEX
   ```bash
   git clone https://github.com/NVIDIA/apex
   cd apex
   pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
   ```

5. （可选）Flash Attention
   ```bash
   pip install flash-attn --no-build-isolation
   ```

6. 设置PYTHONPATH
   ```bash
   export PYTHONPATH=/path/to/Pai-Megatron-Patch:/path/to/Pai-Megatron-Patch/Megatron-LM-250328:$PYTHONPATH
   ```

## 测试脚本概述

我们提供了多个GPU测试脚本，每个脚本专注于测试不同的方面：

1. **run_mini_deepseek_test.sh**：
   - 主要测试脚本，用于在不同并行配置下测试DeepSeek V3模型
   - 支持Tensor Parallelism (TP)和Pipeline Parallelism (PP)配置
   - 使用随机初始化的小型模型进行测试

2. **test_mla_moe_correctness.py**：
   - 验证MLA和MOE实现的正确性
   - 比较Megatron和HuggingFace模型的输出
   - 使用钩子捕获中间激活值进行比较

3. **convert_mini_model.py**：
   - 将HuggingFace模型转换为Megatron格式
   - 支持选择性转换部分层
   - 适用于创建小型测试模型

## 运行测试

### 基本测试

测试TP=2, PP=1配置：

```bash
# 运行TP=2, PP=1测试
bash examples/deepseek_v3/run_mini_deepseek_test.sh dsw 2 1
```

测试TP=1, PP=2配置：

```bash
# 运行TP=1, PP=2测试
bash examples/deepseek_v3/run_mini_deepseek_test.sh dsw 1 2
```

### 运行所有测试

使用提供的脚本运行所有测试：

```bash
# 运行所有测试
bash examples/deepseek_v3/run_mla_moe_tests.sh
```

### 自定义测试

如果您想使用自定义参数运行测试：

```bash
# 自定义测试参数
python examples/deepseek_v3/test_mla_moe_correctness.py \
  --hf-model-path /path/to/hf/model \
  --test-tp 2 \
  --test-pp 1 \
  --test-batch-size 2 \
  --test-seq-length 512 \
  --debug
```

## 测试结果解读

### run_mini_deepseek_test.sh 结果

此脚本会输出类似以下内容：

```
===== 测试 TP=2, PP=1 配置 =====
Creating a test script for validating MLA/MOE implementation...
Testing with configuration:
  - Number of layers: 6
  - Hidden size: 1024
  - Number of attention heads: 16
  - Number of experts: 16
  - Tensor parallelism: 2
  - Pipeline parallelism: 1

Loading HuggingFace model...
Creating Megatron model...
Running validation...

============= Testing Results =============
--- Checking Multi-Latent Attention (MLA) ---
PASS: Layer 0 Attention Output - Max diff: 0.000123, Avg diff: 0.000032

--- Checking Mixture of Experts (MOE) ---
PASS: Layer 1 MOE Output - Max diff: 0.000245, Avg diff: 0.000078
PASS: Layer 1 Router Output - Max diff: 0.000189, Avg diff: 0.000045

--- Checking Final Output ---
PASS: Final Output - Max diff: 0.000312, Avg diff: 0.000097

============= Overall Result =============
PASS
```

关键检查点：
- 每个组件测试是否通过（PASS/FAIL）
- 最大差异和平均差异值是否在可接受范围内（通常<1e-3）
- 最终结果是否为PASS

### test_mla_moe_correctness.py 结果

此脚本会输出更详细的比较结果：

```
Setting up test data and hooks...
Registering hooks for HuggingFace model...
Registering hooks for Megatron model...
Running forward pass for HuggingFace model...
Running forward pass for Megatron model...

============= Testing Results =============
--- Checking Multi-Latent Attention (MLA) ---
PASS: Layer 0 Attention Output - Max diff: 0.000123, Avg diff: 0.000032
...

--- Checking Mixture of Experts (MOE) ---
PASS: Layer 1 MOE Output - Max diff: 0.000245, Avg diff: 0.000078
...

--- Checking Final Output ---
PASS: Final Output - Max diff: 0.000312, Avg diff: 0.000097

============= Overall Result =============
PASS
```

关键检查点：
- 钩子是否正确注册
- 前向传播是否成功完成
- 各层激活值的差异是否在可接受范围内

## 常见问题

### 1. CUDA out of memory

如果遇到显存不足的问题：
- 减小模型大小（隐藏层大小、层数等）
- 减小批次大小和序列长度
- 增加并行度（TP或PP）
- 使用混合精度训练（fp16或bf16）

### 2. 模型加载失败

如果模型加载失败：
- 检查HuggingFace模型路径是否正确
- 确保有足够的磁盘空间存储模型权重
- 检查网络连接是否正常
- 尝试手动下载模型后再加载

### 3. 并行配置错误

如果遇到并行配置错误：
- 确保TP和PP的乘积不超过可用GPU数量
- 检查每个PP阶段的层数是否合理
- 确保MOE配置与并行策略兼容

## 高级配置

### 自定义模型配置

您可以通过修改`run_mini_deepseek_test.sh`中的参数来自定义模型配置：

```bash
# 修改模型大小
NUM_LAYERS=6                # 层数
HIDDEN_SIZE=1024            # 隐藏层大小
NUM_ATTENTION_HEADS=16      # 注意力头数
INTERMEDIATE_SIZE=2816      # 中间层大小
MOE_INTERMEDIATE_SIZE=512   # MOE中间层大小
NUM_EXPERTS=16              # 专家数量
ROUTER_TOPK=4               # 路由器TopK
```

### 自定义并行策略

您可以尝试不同的并行策略组合：

```bash
# 并行配置
TP=2    # 张量并行度
PP=1    # 流水线并行度
EP=1    # 专家并行度
ETP=1   # 专家张量并行度
```

### 添加自定义测试

如果您想添加新的测试用例，可以参考现有测试脚本的结构，并添加您自己的测试函数。例如：

```python
def test_custom_component():
    # 初始化组件
    # 运行测试
    # 验证结果
    pass

if __name__ == "__main__":
    # 添加您的新测试
    test_custom_component()
```

## 结论

通过这些GPU测试，您可以全面验证DeepSeek V3模型的MLA和MOE组件在不同并行配置下的正确性。这些测试覆盖了从模型结构到权重转换的各个方面，确保模型在实际训练和推理场景中能够正常工作。

如果您只需要验证组件的基本功能而没有GPU资源，请参考CPU测试指南。
