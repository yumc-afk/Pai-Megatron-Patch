# PyTorch LLM 分布式训练静态分析报告

## 分析元数据

- **分析时间**: {{timestamp}}
- **分析目标**: {{target}}
- **分析器**: {{analyzers}}

## 分析摘要

- **总警告数**: {{total_warnings}}
- **总错误数**: {{total_errors}}
- **总建议数**: {{total_suggestions}}

{% for analyzer_name, analyzer_results in results.items() %}
## {{analyzer_name}} 分析结果

### 摘要

{{analyzer_results.summary}}

{% if analyzer_results.errors %}
### 错误 ({{analyzer_results.errors|length}})

{% for error in analyzer_results.errors %}
- **{{error.code}}**: {{error.message}}  [{{error.category}}]
  - 文件: `{{error.file_path}}`
  - 行号: {{error.line}}
{% if error.content %}
  - 代码: `{{error.content}}`
{% endif %}

{% endfor %}
{% endif %}

{% if analyzer_results.warnings %}
### 警告 ({{analyzer_results.warnings|length}})

{% for warning in analyzer_results.warnings %}
- **{{warning.code}}**: {{warning.message}}  [{{warning.category}}]
  - 文件: `{{warning.file_path}}`
  - 行号: {{warning.line}}
{% if warning.content %}
  - 代码: `{{warning.content}}`
{% endif %}

{% endfor %}
{% endif %}

{% if analyzer_results.suggestions %}
### 建议 ({{analyzer_results.suggestions|length}})

{% for suggestion in analyzer_results.suggestions %}
- **{{suggestion.code}}**: {{suggestion.message}}
  - 文件: `{{suggestion.file_path}}`
  - 行号: {{suggestion.line}}
{% if suggestion.content %}
  - 代码: `{{suggestion.content}}`
{% endif %}

{% endfor %}
{% endif %}

{% endfor %}

## 结论和建议

### 关键问题

以下是需要优先解决的关键问题:

{% for analyzer_name, analyzer_results in results.items() %}
{% for error in analyzer_results.errors %}
- [{{analyzer_name}}] **{{error.code}}**: {{error.message}}  [{{error.category}}]
  - 文件: `{{error.file_path}}`
  - 行号: {{error.line}}

{% endfor %}
{% endfor %}

### 改进建议

{% for analyzer_name, analyzer_results in results.items() %}
{% if analyzer_results.warnings %}
#### {{analyzer_name}} 改进建议

{% for warning in analyzer_results.warnings|slice:":5" %}
- {{warning.message}} ({{warning.file_path}}:{{warning.line}})
{% endfor %}
{% if analyzer_results.warnings|length > 5 %}
- ... 以及 {{analyzer_results.warnings|length - 5}} 个其他警告
{% endif %}

{% endif %}
{% endfor %}

### 自动修复

以下问题可以通过自动修复工具解决:

{% for analyzer_name, analyzer_results in results.items() %}
{% if analyzer_results.auto_fixable %}
#### {{analyzer_name}} 自动修复

{% for fixable in analyzer_results.auto_fixable|slice:":5" %}
- {{fixable.message}} ({{fixable.file_path}}:{{fixable.line}})
{% endfor %}
{% if analyzer_results.auto_fixable|length > 5 %}
- ... 以及 {{analyzer_results.auto_fixable|length - 5}} 个其他可自动修复的问题
{% endif %}

{% endif %}
{% endfor %}
