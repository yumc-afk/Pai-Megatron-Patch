# DeepSeek V3 MLA/MOE CPU测试指南

本指南详细介绍如何在CPU环境下测试DeepSeek V3模型的MLA（Multi-Latent Attention）和MOE（Mixture of Experts）组件实现。这些测试不需要GPU硬件，可以在任何标准计算环境中运行。

## 目录

1. [环境安装](#环境安装)
2. [测试脚本概述](#测试脚本概述)
3. [运行测试](#运行测试)
4. [测试结果解读](#测试结果解读)
5. [常见问题](#常见问题)
6. [高级配置](#高级配置)

## 环境安装

### 自动安装

我们提供了自动安装脚本，可以一键安装所有必要的依赖：

```bash
# 克隆仓库（如果尚未克隆）
git clone https://github.com/haoweiliang1996/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch

# 运行安装脚本
bash examples/deepseek_v3/install_cpu_test.sh

# 设置环境变量
source examples/deepseek_v3/setup_env.sh
```

### 手动安装

如果您希望手动安装依赖，请确保安装以下组件：

1. Python 3.8+
2. PyTorch 2.0+（CPU版本即可）
3. Transformers库
4. Huggingface Hub库

```bash
# 安装基本依赖
pip install torch numpy transformers huggingface_hub tqdm matplotlib

# 设置PYTHONPATH
export PYTHONPATH=/path/to/Pai-Megatron-Patch:/path/to/Pai-Megatron-Patch/Megatron-LM-250328:$PYTHONPATH
```

## 测试脚本概述

我们提供了多个CPU测试脚本，每个脚本专注于测试不同的组件：

1. **test_mla_moe_cpu_simple.py**：
   - 最基础的测试脚本，验证MLA和MOE组件的基本功能
   - 使用随机生成的权重，不需要下载模型
   - 测试组件的前向传播和输出形状

2. **test_mla_moe_components.py**：
   - 更详细的组件测试，单独测试MLA和MOE的各个子组件
   - 包括旋转位置编码、多潜在注意力机制、MOE路由器等
   - 验证每个组件的数值稳定性和正确性

3. **verify_deepseek_v3_structure.py**：
   - 验证DeepSeek V3模型结构，特别是MLA和MOE相关配置
   - 只加载模型配置，不下载完整权重
   - 分析模型架构和参数设置

4. **verify_weight_conversion.py**：
   - 验证权重转换所需的结构参数
   - 分析模型配置中的MOE和MLA相关参数
   - 不执行实际的权重转换

## 运行测试

### 基本测试

运行基本的MLA/MOE组件测试：

```bash
# 运行简单CPU测试
python examples/deepseek_v3/test_mla_moe_cpu_simple.py

# 或使用提供的脚本运行所有CPU测试
bash examples/deepseek_v3/run_cpu_tests.sh
```

### 验证模型结构

验证DeepSeek V3模型结构：

```bash
# 验证DeepSeek V3结构
python examples/deepseek_v3/verify_deepseek_v3_structure.py --model-name deepseek-ai/deepseek-v3-7b --config-only

# 验证DeepSeek MOE结构
python examples/deepseek_v3/verify_mla_moe_structure.py --model-name deepseek-ai/deepseek-moe-16b-base --config-only
```

### 验证权重转换参数

验证权重转换所需的参数：

```bash
# 验证DeepSeek MOE 16B模型的权重转换参数
python examples/deepseek_v3/verify_weight_conversion.py --hf-model-name deepseek-ai/deepseek-moe-16b-base --use-cpu
```

## 测试结果解读

### test_mla_moe_cpu_simple.py 结果

此脚本会输出以下信息：

```
===== 测试MLA (Multi-Latent Attention)组件 =====
创建MLA测试模型...
输入形状: torch.Size([2, 16, 1024])
MLA输出形状: torch.Size([2, 16, 1024])
MLA测试通过!

===== 测试MOE (Mixture of Experts)组件 =====
创建MOE测试模型...
输入形状: torch.Size([2, 16, 1024])
MOE输出形状: torch.Size([2, 16, 1024])
MOE测试通过!

===== 所有测试完成 =====
```

关键检查点：
- 输出形状是否符合预期
- 是否有NaN或Inf值
- 是否有任何错误或警告消息

### verify_deepseek_v3_structure.py 结果

此脚本会输出模型结构信息：

```
===== 模型配置 =====
模型类型: deepseek
隐藏层大小: 7168
层数: 32
注意力头数: 32
qk_head_dim: 128
qk_pos_emb_head_dim: 64
v_head_dim: 128
模型包含MLA (Multi-Latent Attention)
```

关键检查点：
- 模型是否包含MLA
- 模型是否包含MOE
- 关键参数值是否符合预期

## 常见问题

### 1. ImportError: No module named 'megatron'

这通常是因为PYTHONPATH设置不正确。确保运行：

```bash
export PYTHONPATH=/path/to/Pai-Megatron-Patch:/path/to/Pai-Megatron-Patch/Megatron-LM-250328:$PYTHONPATH
```

### 2. 测试脚本运行缓慢

CPU测试可能比GPU测试慢，这是正常的。如果测试运行过慢，可以尝试：
- 减小测试批次大小和序列长度
- 减少测试迭代次数
- 使用更简单的模型配置

### 3. 内存错误

如果遇到内存错误，可以尝试：
- 减小模型大小（隐藏层大小、层数等）
- 减小批次大小
- 使用虚拟内存或增加系统内存

## 高级配置

### 自定义测试参数

您可以通过修改测试脚本中的参数来自定义测试：

```python
# 在test_mla_moe_cpu_simple.py中修改
BATCH_SIZE = 2          # 批次大小
SEQ_LENGTH = 16         # 序列长度
HIDDEN_SIZE = 1024      # 隐藏层大小
NUM_HEADS = 16          # 注意力头数
NUM_EXPERTS = 8         # 专家数量
```

### 添加新的测试用例

如果您想添加新的测试用例，可以参考现有测试脚本的结构，并添加您自己的测试函数。例如：

```python
def test_new_component():
    # 初始化组件
    # 运行测试
    # 验证结果
    pass

if __name__ == "__main__":
    # 添加您的新测试
    test_new_component()
```

### 调试技巧

如果测试失败，可以尝试以下调试技巧：

1. 添加更多的打印语句，输出中间结果
2. 使用较小的模型配置进行测试
3. 检查张量形状和数值范围
4. 比较与参考实现的差异

## 结论

通过这些CPU测试，您可以验证DeepSeek V3模型的MLA和MOE组件的基本功能和结构正确性，而无需GPU资源。这些测试主要关注组件的结构和前向传播功能，而不是性能优化或训练效果。

如果您需要更全面的测试，包括实际的权重转换和模型训练，请参考GPU测试指南。
