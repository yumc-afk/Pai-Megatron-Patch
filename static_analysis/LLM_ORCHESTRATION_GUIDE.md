# LLM编排指南

## 概述

本文档提供了如何使用大型语言模型(LLM)来编排静态分析工作流程的详细指南。LLM在静态分析中扮演着关键角色，它可以理解代码上下文、解释分析结果、提出修复建议，并验证修复的正确性。

## LLM编排架构

### 核心组件

1. **LLM编排器**：负责协调整个分析流程，包括选择分析工具、解释结果、生成修复建议等
2. **分析工具适配器**：将各种静态分析工具集成到统一的接口中
3. **上下文管理器**：管理代码库的上下文信息，帮助LLM理解代码结构和意图
4. **修复生成器**：基于分析结果生成修复建议
5. **验证器**：验证修复的正确性和有效性

### 架构图

```
+-------------------+      +-------------------+      +-------------------+
|                   |      |                   |      |                   |
|  静态分析工具集成   |----->|    LLM编排器      |----->|    修复生成器     |
|                   |      |                   |      |                   |
+-------------------+      +-------------------+      +-------------------+
                                    ^                         |
                                    |                         |
                                    v                         v
                           +-------------------+      +-------------------+
                           |                   |      |                   |
                           |   上下文管理器     |      |      验证器       |
                           |                   |      |                   |
                           +-------------------+      +-------------------+
```

## 使用指南

### 1. 初始化LLM编排器

```python
from ml_static_analysis.llm_interface import LLMOrchestrator

# 创建LLM编排器
orchestrator = LLMOrchestrator(
    llm_provider="openai",  # 或其他支持的LLM提供商
    model_name="gpt-4",     # 或其他支持的模型
    api_key="your_api_key"  # 如果需要的话
)
```

### 2. 配置分析工具

```python
# 配置要使用的分析工具
orchestrator.configure_analyzers(
    analyzers=["mypy", "jaxtype"],  # 选择要使用的分析器
    config={
        "mypy": {
            "strict": True,
            "ignore_missing_imports": True
        },
        "jaxtype": {
            "check_shapes": True,
            "check_dtypes": True,
            "check_devices": True,
            "use_beartype": True
        }
    }
)
```

### 3. 分析代码库

```python
# 分析整个代码库
results = orchestrator.analyze_codebase(
    path="/path/to/codebase",
    recursive=True,
    file_extensions=[".py"]
)

# 或者分析单个文件
file_results = orchestrator.analyze_file("/path/to/file.py")
```

### 4. 解释分析结果

```python
# 获取人类可读的解释
explanations = orchestrator.explain_findings(results)

# 打印解释
for file_path, file_explanations in explanations.items():
    print(f"File: {file_path}")
    for explanation in file_explanations:
        print(f"  - {explanation}")
```

### 5. 生成修复建议

```python
# 生成修复建议
fixes = orchestrator.generate_fixes(results)

# 打印修复建议
for file_path, file_fixes in fixes.items():
    print(f"File: {file_path}")
    for fix in file_fixes:
        print(f"  Problem: {fix['problem']}")
        print(f"  Suggestion: {fix['suggestion']}")
        print(f"  Code: {fix['code']}")
```

### 6. 应用修复

```python
# 应用所有修复
orchestrator.apply_fixes(fixes)

# 或者应用特定文件的修复
orchestrator.apply_fixes(fixes, file_path="/path/to/specific/file.py")

# 或者应用特定的修复
orchestrator.apply_fix(fixes["/path/to/file.py"][0])
```

### 7. 验证修复

```python
# 验证所有修复
validation_results = orchestrator.validate_fixes(fixes)

# 打印验证结果
for file_path, validations in validation_results.items():
    print(f"File: {file_path}")
    for validation in validations:
        status = "成功" if validation["success"] else "失败"
        print(f"  修复 {validation['fix_id']}: {status}")
        if not validation["success"]:
            print(f"    原因: {validation['reason']}")
```

## 高级用法

### 自定义LLM提示

您可以自定义LLM的提示模板，以获得更好的结果：

```python
# 自定义解释提示
orchestrator.set_prompt_template(
    "explain",
    """
    你是一个专业的代码分析助手。请解释以下静态分析发现，使用简单明了的语言：
    
    {findings}
    
    请特别关注可能导致运行时错误的问题，并解释为什么这些问题是重要的。
    """
)

# 自定义修复提示
orchestrator.set_prompt_template(
    "fix",
    """
    你是一个专业的代码修复助手。请为以下问题提供修复建议：
    
    {problem}
    
    代码上下文：
    {context}
    
    请提供具体的修复代码，并解释你的修复如何解决问题。确保你的修复不会引入新的问题。
    """
)
```

### 集成自定义分析工具

您可以集成自己的分析工具：

```python
from ml_static_analysis.core.analyzer import BaseAnalyzer

# 创建自定义分析器
class MyCustomAnalyzer(BaseAnalyzer):
    def __init__(self, config):
        super().__init__(config)
        # 初始化自定义分析器
        
    def analyze(self):
        # 实现分析逻辑
        pass

# 注册自定义分析器
orchestrator.register_analyzer("my_custom", MyCustomAnalyzer)

# 使用自定义分析器
orchestrator.configure_analyzers(
    analyzers=["mypy", "jaxtype", "my_custom"],
    config={
        # 其他配置...
        "my_custom": {
            "enabled": True,
            # 自定义配置...
        }
    }
)
```

### 批处理分析

对于大型代码库，您可以使用批处理模式：

```python
# 批处理分析
batch_results = orchestrator.batch_analyze(
    path="/path/to/codebase",
    batch_size=10,  # 每批处理的文件数
    max_workers=4   # 并行工作进程数
)
```

## 最佳实践

1. **选择合适的LLM模型**：
   - 对于复杂代码库，使用更强大的模型（如GPT-4）
   - 对于简单任务，可以使用更轻量的模型以提高速度

2. **提供足够的上下文**：
   - 确保LLM有足够的代码上下文来做出准确判断
   - 使用`context_lines`参数控制上下文行数

3. **验证LLM建议**：
   - 始终验证LLM提供的修复建议
   - 使用`validate_fixes`函数检查修复是否解决了问题

4. **迭代改进**：
   - 使用LLM的反馈来改进分析配置
   - 记录成功的修复模式，用于未来的分析

5. **处理大型代码库**：
   - 使用批处理模式分析大型代码库
   - 优先分析关键模块和常见错误模式

## 故障排除

### 常见问题

1. **LLM返回不相关的结果**：
   - 检查提供的上下文是否足够
   - 尝试使用更具体的提示模板
   - 考虑使用更强大的LLM模型

2. **分析速度太慢**：
   - 减少分析的文件数量
   - 使用批处理模式
   - 考虑使用更轻量的LLM模型

3. **修复建议不正确**：
   - 提供更多代码上下文
   - 使用更专业的提示模板
   - 手动审查和调整修复建议

### 日志和调试

启用详细日志以帮助调试：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 或者在创建编排器时启用详细模式
orchestrator = LLMOrchestrator(verbose=True)
```

## 结论

LLM编排为静态分析工具带来了新的可能性，使其更加智能、更加易用、更加有效。通过将LLM的理解能力与传统静态分析工具的精确性相结合，我们可以创建更强大的代码质量保障系统，帮助开发者编写更好的代码。

有关更多信息，请参阅[LLM编排工作流程](./LLM_ORCHESTRATION_WORKFLOW.md)和[LLM使用指南](./LLM_USAGE_GUIDE.md)。
