# ML静态分析框架 - LLM快速入门指南

本指南为大型语言模型（LLMs）提供了使用ML静态分析框架分析ML/DL代码库的快速入门方法。

## 概述

ML静态分析框架旨在帮助LLMs在不运行完整训练流程的情况下分析ML/DL代码库。它提供了一套静态分析工具，可以检测ML/DL代码中的常见问题，如张量形状不匹配、类型错误和性能问题。

## 安装

### 方式1：一键安装

```bash
# 下载安装脚本
curl -O https://raw.githubusercontent.com/yumc-afk/Pai-Megatron-Patch/main/static_analysis/install_pip.sh

# 使脚本可执行
chmod +x install_pip.sh

# 运行安装脚本
./install_pip.sh
```

### 方式2：手动安装

```bash
# 安装标准版
pip install git+https://github.com/yumc-afk/Pai-Megatron-Patch.git#subdirectory=static_analysis

# 或安装精简版
pip install git+https://github.com/yumc-afk/Pai-Megatron-Patch.git#subdirectory=static_analysis[lite]
```

## LLM快速使用指南

### Python接口

```python
# 标准版
from ml_static_analysis.llm_interface import analyze_codebase

# 分析文件
results = analyze_codebase(file_path="/path/to/file.py")

# 分析目录
results = analyze_codebase(dir_path="/path/to/dir")

# 获取分析报告
report = results["report"]
print(report)

# 精简版
from ml_static_analysis_lite import analyze_codebase

# 分析文件
results = analyze_codebase(file_path="/path/to/file.py")

# 获取分析报告
report = results["report"]
print(report)
```

### 命令行接口

```bash
# 标准版
ml-analyze --file /path/to/file.py
ml-analyze --dir /path/to/dir

# 精简版
ml-analyze-lite --file /path/to/file.py
ml-analyze-lite --dir /path/to/dir
```

## LLM关键功能

1. **张量形状分析**：使用PyTea检测潜在的张量形状不匹配问题
2. **类型检查**：使用JaxType增强对张量形状和维度的类型检查
3. **代码质量分析**：使用PyAssistant检测常见的PyTorch编程错误
4. **自动修复功能**：自动修复分析器检测到的常见问题
5. **全面报告**：生成包含发现和建议的详细报告

## LLM工作流程

作为LLM，您可以使用此框架：

1. **分析代码**：分析ML/DL代码库以检测潜在问题
2. **生成报告**：生成包含发现和建议的详细报告
3. **建议修复**：为检测到的问题提供修复建议或应用自动修复
4. **优化代码**：基于分析结果优化代码
5. **验证代码**：在不运行完整训练流程的情况下验证代码

## 示例工作流程

1. **分析代码库**：
   ```python
   from ml_static_analysis.llm_interface import analyze_codebase
   
   results = analyze_codebase(dir_path="/path/to/codebase")
   report = results["report"]
   ```

2. **查看发现**：
   ```python
   findings = results["findings"]
   for finding in findings:
       print(f"文件: {finding['file']}")
       print(f"行号: {finding['line']}")
       print(f"消息: {finding['message']}")
       print(f"严重性: {finding['severity']}")
       print()
   ```

3. **应用自动修复**：
   ```python
   from ml_static_analysis.llm_interface import analyze_codebase
   
   results = analyze_codebase(
       dir_path="/path/to/codebase",
       autofix=True,
       autofix_report_path="/path/to/autofix_report.md"
   )
   ```

4. **生成摘要**：
   ```python
   summary = results["summary"]
   print(f"已分析 {summary['analyzed_files']} 个文件")
   print(f"发现 {summary['total_findings']} 个问题")
   print(f"修复 {summary['fixed_findings']} 个问题")
   ```

## 高级功能

### 自动修复

框架可以自动修复分析器检测到的常见问题：

```python
results = analyze_codebase(
    file_path="/path/to/file.py",
    autofix=True,
    autofix_report_path="/path/to/autofix_report.md"
)
```

### 自定义分析器

您可以指定要使用的分析器：

```python
results = analyze_codebase(
    file_path="/path/to/file.py",
    analyzers=["mypy", "pytea", "pyassistant", "jaxtype"]
)
```

### 自定义配置

您可以提供自定义配置文件：

```python
results = analyze_codebase(
    file_path="/path/to/file.py",
    config_path="/path/to/config.json"
)
```

## 更多信息

- [完整文档](./README.md)
- [使用指南](./USAGE.md)
- [LLM使用指南](./LLM_USAGE_GUIDE.md)
- [自动修复指南](./AUTOFIX_GUIDE.md)
- [错误模式](./ERROR_PATTERNS.md)

## 结论

ML静态分析框架是LLMs分析ML/DL代码库的强大工具，无需运行完整训练流程。它可以帮助检测常见问题并提供修复建议，使优化代码和减少GPU调试时间变得更加容易。
