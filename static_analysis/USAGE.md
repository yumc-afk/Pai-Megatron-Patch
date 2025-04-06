# PyTorch LLM 分布式训练静态分析框架使用指南

本文档详细介绍如何使用静态分析框架验证PyTorch LLM分布式训练代码，特别是MLA（Multi-Latent Attention）和MOE（Mixture of Experts）组件的实现。

## 目录

1. [快速开始](#快速开始)
2. [工具详解](#工具详解)
   - [PyTea 张量形状分析](#pytea-张量形状分析)
   - [JaxType 类型增强](#jaxtype-类型增强)
   - [PyAssist 模式检测](#pyassist-模式检测)
   - [分布式同步验证](#分布式同步验证)
   - [符号执行工具](#符号执行工具)
3. [集成分析流程](#集成分析流程)
4. [常见问题](#常见问题)
5. [高级配置](#高级配置)

## 快速开始

### 安装

首先，安装所有必要的依赖和工具：

```bash
# 克隆仓库（如果尚未克隆）
git clone https://github.com/yumc-afk/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch

# 安装静态分析工具
bash static_analysis/install.sh

# 设置环境变量
source static_analysis/setup_env.sh
```

### 运行基本分析

对代码进行基本静态分析：

```bash
# 分析单个文件
python static_analysis/run_analysis.py --file examples/deepseek_v3/test_mla_moe_cpu_simple.py

# 分析整个目录
python static_analysis/run_analysis.py --dir examples/deepseek_v3/

# 分析特定组件
python static_analysis/run_analysis.py --component mla --dir examples/deepseek_v3/
```

### 查看分析报告

分析完成后，可以查看生成的报告：

```bash
# 查看最新的分析报告
cat static_analysis/reports/latest_report.md

# 查看特定日期的报告
cat static_analysis/reports/report_YYYY-MM-DD_HH-MM-SS.md
```

## 工具详解

### PyTea 张量形状分析

PyTea是一个专门针对PyTorch代码的静态分析器，可以检测潜在的张量形状不匹配错误。

#### 基本用法

```bash
# 使用PyTea分析单个文件
python static_analysis/tools/pytea_analyzer.py --file examples/deepseek_v3/test_mla_moe_cpu_simple.py

# 使用更详细的输出
python static_analysis/tools/pytea_analyzer.py --file examples/deepseek_v3/test_mla_moe_cpu_simple.py --verbose
```

#### 配置选项

PyTea支持多种配置选项，可以通过配置文件或命令行参数指定：

```bash
# 使用配置文件
python static_analysis/tools/pytea_analyzer.py --config static_analysis/configs/pytea_config.json

# 指定最大分析深度
python static_analysis/tools/pytea_analyzer.py --file examples/deepseek_v3/test_mla_moe_cpu_simple.py --max-depth 10
```

#### 解读结果

PyTea的分析结果包含以下几个部分：

1. **有效路径**：代码中所有张量形状兼容的执行路径
2. **无效路径**：可能导致张量形状不匹配的执行路径
3. **未知路径**：由于分析限制无法确定的执行路径

对于每个无效路径，PyTea会提供详细的错误信息，包括：
- 错误发生的位置（文件名和行号）
- 涉及的张量及其形状
- 导致错误的操作

### JaxType 类型增强

JaxType提供了增强的类型注解功能，允许在类型系统中指定张量的形状、维度和数据类型。作为TorchTyping的现代替代品，JaxType提供了更强大的类型检查能力和更好的与现代Python类型系统的集成。

#### 添加类型注解

在代码中添加JaxType类型注解：

```python
from jaxtyping import Array
from typeguard import typechecked
import torch

@typechecked
def process_batch(
    input_ids: Array["batch_size seq_len", torch.long],
    attention_mask: Array["batch_size seq_len", torch.bool]
) -> Array["batch_size seq_len hidden_size", torch.float]:
    # 函数实现...
    return output
```

#### 运行类型检查

使用我们的工具运行类型检查：

```bash
# 运行类型检查
python static_analysis/tools/type_checker.py --file examples/deepseek_v3/test_mla_moe_cpu_simple.py
```

#### 自动添加类型注解

我们的工具还可以自动为代码添加JaxType类型注解：

```bash
# 自动添加类型注解
python static_analysis/tools/type_annotator.py --file examples/deepseek_v3/test_mla_moe_cpu_simple.py
```

### PyAssist 模式检测

PyAssist是一个基于模式的静态分析工具，可以检测常见的PyTorch编程错误。

#### 运行模式检测

```bash
# 运行模式检测
python static_analysis/tools/pattern_detector.py --file examples/deepseek_v3/test_mla_moe_cpu_simple.py

# 检测特定模式
python static_analysis/tools/pattern_detector.py --file examples/deepseek_v3/test_mla_moe_cpu_simple.py --patterns device_mismatch,gradient_accumulation
```

#### 添加自定义模式

您可以添加自定义的错误模式：

```bash
# 添加自定义模式
python static_analysis/tools/pattern_manager.py --add --name my_pattern --description "描述" --pattern "正则表达式"
```

### 分布式同步验证

分布式同步验证工具可以分析分布式训练代码中的通信模式，检测潜在的死锁、不同步和资源竞争问题。

#### 运行同步验证

```bash
# 运行同步验证
python static_analysis/tools/dist_sync_checker.py --file examples/deepseek_v3/test_mla_moe_correctness.py
```

#### 分析结果

同步验证工具会生成一个通信图，显示各个进程之间的通信关系和潜在的同步问题。

### 符号执行工具

基于CrossHair的符号执行工具，可以对关键代码路径进行深入分析，验证在各种输入条件下的行为是否符合预期。

#### 添加契约

在代码中添加契约（前置条件和后置条件）：

```python
from crosshair import contracts

@contracts.ensure(lambda result, self: len(result) == len(self.data))
def process_data(self, data):
    # 函数实现...
    return result
```

#### 运行符号执行

```bash
# 运行符号执行
python static_analysis/tools/symbolic_executor.py --file examples/deepseek_v3/test_mla_moe_cpu_simple.py
```

## 集成分析流程

我们提供了一个集成的分析流程，可以一次性运行所有工具并生成综合报告：

```bash
# 运行集成分析
python static_analysis/run_analysis.py --file examples/deepseek_v3/test_mla_moe_cpu_simple.py --all

# 生成HTML报告
python static_analysis/run_analysis.py --file examples/deepseek_v3/test_mla_moe_cpu_simple.py --all --format html
```

## 常见问题

### 1. 分析超时

如果分析过程超时，可以尝试以下方法：

- 减小分析范围，只分析特定文件或函数
- 调整超时参数：`--timeout 300`（单位：秒）
- 降低分析深度：`--max-depth 5`

### 2. 误报处理

如果工具报告了误报（假阳性），可以：

- 添加注释标记忽略特定警告：`# static-analysis: ignore[shape-mismatch]`
- 更新工具配置，排除特定模式或路径
- 提交反馈，帮助我们改进工具

### 3. 工具兼容性

如果遇到工具兼容性问题，请检查：

- Python版本是否兼容（推荐Python 3.8+）
- PyTorch版本是否兼容（推荐PyTorch 1.10+）
- 是否安装了所有必要的依赖

## 高级配置

### 自定义分析规则

您可以创建自定义的分析规则：

```bash
# 创建自定义规则
python static_analysis/tools/rule_manager.py --create --name my_rule --type pytea --description "描述"
```

### 集成到CI/CD

将静态分析集成到CI/CD流程中：

```yaml
# .github/workflows/static-analysis.yml
name: Static Analysis

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        bash static_analysis/install.sh
    - name: Run analysis
      run: |
        source static_analysis/setup_env.sh
        python static_analysis/run_analysis.py --dir examples/ --all
```

### 性能优化

对于大型代码库，可以使用以下方法优化分析性能：

- 使用增量分析：`--incremental`
- 并行分析多个文件：`--parallel`
- 缓存分析结果：`--cache`
