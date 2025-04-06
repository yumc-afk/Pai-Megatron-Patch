# DeepSeek V3 MLA/MOE 小型模型测试指南

本文档提供了如何使用小型配置测试DeepSeek V3模型中MLA（Multi-Latent Attention）和MOE（Mixture of Experts）实现正确性的指南。

## 概述

为了在有限的GPU资源（两张H800）上验证DeepSeek V3的MLA和MOE实现的正确性，我们创建了一个小型配置，具有以下特点：

- 减少层数：从原始的61层减少到6层
- 减少隐藏层大小：从7168减少到1024
- 减少专家数量：从256减少到16
- 保留MLA和MOE的核心功能

## 测试方法

我们的测试方法包括以下步骤：

1. 创建随机初始化的自定义HuggingFace风格模型作为参考
2. 在两种并行配置下运行模型：
   - TP=2, PP=1（张量并行度=2，流水线并行度=1）
   - TP=1, PP=2（张量并行度=1，流水线并行度=2）
3. 比较Megatron模型和自定义模型的结构和前向传播，验证实现的正确性

## 文件说明

- `run_mini_deepseek_test.sh`：小型DeepSeek V3配置脚本
- `test_mla_moe_correctness.py`：验证MLA和MOE实现正确性的测试脚本，包含自定义HF风格模型实现
- `run_mla_moe_tests.sh`：运行所有测试的包装脚本
- `test_mla_moe_cpu.py`：在CPU上验证MLA和MOE组件的测试脚本
- `test_mla_moe_cpu_simple.py`：简化版CPU测试脚本，更加稳健
- `run_cpu_tests.sh`：运行CPU测试的包装脚本

## 使用方法

### 准备工作

确保您已经安装了所有必要的依赖项。

### 运行测试

#### GPU测试

使用以下命令在GPU上运行测试：

```bash
bash run_mla_moe_tests.sh
```

测试结果将保存在`mini_model_test_results`目录中，包括：

- `tp2_pp1_test.log`：TP=2, PP=1配置的测试日志
- `tp1_pp2_test.log`：TP=1, PP=2配置的测试日志

#### CPU测试

在没有GPU的环境中，可以使用以下命令在CPU上验证MLA和MOE组件的基本功能：

```bash
bash run_cpu_tests.sh
```

CPU测试结果将保存在`cpu_test_results.log`文件中。

## 验证标准

### GPU测试验证标准

GPU测试脚本会比较以下组件的输出：

1. **MLA（Multi-Latent Attention）**：
   - 比较每一层的注意力输出
   - 验证位置编码和多潜在表示的正确性

2. **MOE（Mixture of Experts）**：
   - 比较每一层的MOE输出
   - 验证路由器输出和专家选择的正确性
   - 检查负载均衡和辅助损失计算

3. **最终输出**：
   - 比较模型的最终输出，确保整体功能正确

### CPU测试验证标准

CPU测试脚本会验证以下组件的基本功能：

1. **旋转位置编码（RotaryEmbedding）**：
   - 验证位置编码计算的正确性
   - 确保输出形状正确且不包含NaN或Inf值

2. **多潜在注意力（MLA Attention）**：
   - 验证注意力计算的正确性
   - 确保注意力分数和上下文向量计算正确

3. **MOE路由器（MOE Router）**：
   - 验证路由逻辑和专家选择的正确性
   - 确保路由概率和为1且不包含NaN或Inf值

4. **MOE专家（MOE Expert）**：
   - 验证专家网络前向传播的正确性
   - 确保输出形状正确且不包含NaN或Inf值

5. **MOE调度（MOE Dispatch）**：
   - 验证专家输出组合的正确性
   - 确保最终输出形状正确且不包含NaN或Inf值

由于使用随机权重，我们主要关注模型结构和前向传播的正确性，而不是具体的输出值。

## 自定义模型实现

测试脚本中包含了以下自定义组件的实现：

1. **CustomRotaryEmbedding**：实现旋转位置编码
2. **CustomAttention**：实现多头注意力机制
3. **CustomMoERouter**：实现专家路由器
4. **CustomMoEExpert**：实现单个专家网络
5. **CustomMoE**：实现混合专家层
6. **CustomTransformerLayer**：实现Transformer层
7. **CustomModel**：实现完整模型架构

这些组件模拟了HuggingFace风格的实现，用于与Megatron实现进行比较。

## 故障排除

如果测试失败，请检查：

1. 检查GPU内存是否足够
2. 查看日志中的具体错误信息，定位问题所在
3. 对于特定层的失败，可以调整`test_mla_moe_correctness.py`中的`epsilon`参数

## 注意事项

- 此测试配置仅用于验证实现的正确性，不适用于性能评估
- 小型配置可能无法完全反映大型模型的所有特性
- 测试结果可能受到随机初始化的影响，建议多次运行测试
