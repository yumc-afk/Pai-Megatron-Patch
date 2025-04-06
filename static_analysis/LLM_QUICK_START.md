# ML Static Analysis Framework - LLM Quick Start Guide

This guide provides a quick start for LLMs (Large Language Models) to use the ML Static Analysis Framework for analyzing ML/DL codebases.

## Overview

The ML Static Analysis Framework is designed to help LLMs analyze ML/DL codebases without running the full training pipeline. It provides a set of static analysis tools that can detect common issues in ML/DL code, such as tensor shape mismatches, type errors, and performance issues.

## Installation

### Option 1: One-click Installation

```bash
# Download the installation script
curl -O https://raw.githubusercontent.com/haoweiliang1996/Pai-Megatron-Patch/main/static_analysis/install_pip.sh

# Make it executable
chmod +x install_pip.sh

# Run the installation script
./install_pip.sh
```

### Option 2: Manual Installation

```bash
# Install the standard version
pip install git+https://github.com/haoweiliang1996/Pai-Megatron-Patch.git#subdirectory=static_analysis

# Or install the lite version
pip install git+https://github.com/haoweiliang1996/Pai-Megatron-Patch.git#subdirectory=static_analysis[lite]
```

## Quick Usage for LLMs

### Python Interface

```python
# Standard version
from ml_static_analysis.llm_interface import analyze_codebase

# Analyze a file
results = analyze_codebase(file_path="/path/to/file.py")

# Analyze a directory
results = analyze_codebase(dir_path="/path/to/dir")

# Get the analysis report
report = results["report"]
print(report)

# Lite version
from ml_static_analysis_lite import analyze_codebase

# Analyze a file
results = analyze_codebase(file_path="/path/to/file.py")

# Get the analysis report
report = results["report"]
print(report)
```

### Command Line Interface

```bash
# Standard version
ml-analyze --file /path/to/file.py
ml-analyze --dir /path/to/dir

# Lite version
ml-analyze-lite --file /path/to/file.py
ml-analyze-lite --dir /path/to/dir
```

## Key Features for LLMs

1. **Tensor Shape Analysis**: Detect potential tensor shape mismatches using PyTea
2. **Type Checking**: Enhance type checking with TorchTyping for tensor shapes and dimensions
3. **Code Quality Analysis**: Detect common PyTorch programming errors using PyAssistant
4. **Auto-fix Capabilities**: Automatically fix common issues detected by the analyzers
5. **Comprehensive Reports**: Generate detailed reports with findings and recommendations

## LLM Workflow

As an LLM, you can use this framework to:

1. **Analyze Code**: Analyze ML/DL codebases to detect potential issues
2. **Generate Reports**: Generate detailed reports with findings and recommendations
3. **Suggest Fixes**: Suggest fixes for detected issues or apply auto-fixes
4. **Optimize Code**: Optimize code based on the analysis results
5. **Validate Code**: Validate code without running the full training pipeline

## Example Workflow

1. **Analyze the codebase**:
   ```python
   from ml_static_analysis.llm_interface import analyze_codebase
   
   results = analyze_codebase(dir_path="/path/to/codebase")
   report = results["report"]
   ```

2. **Review the findings**:
   ```python
   findings = results["findings"]
   for finding in findings:
       print(f"File: {finding['file']}")
       print(f"Line: {finding['line']}")
       print(f"Message: {finding['message']}")
       print(f"Severity: {finding['severity']}")
       print()
   ```

3. **Apply auto-fixes**:
   ```python
   from ml_static_analysis.llm_interface import analyze_codebase
   
   results = analyze_codebase(
       dir_path="/path/to/codebase",
       autofix=True,
       autofix_report_path="/path/to/autofix_report.md"
   )
   ```

4. **Generate a summary**:
   ```python
   summary = results["summary"]
   print(f"Analyzed {summary['analyzed_files']} files")
   print(f"Found {summary['total_findings']} findings")
   print(f"Fixed {summary['fixed_findings']} findings")
   ```

## Advanced Features

### Auto-fix

The framework can automatically fix common issues detected by the analyzers:

```python
results = analyze_codebase(
    file_path="/path/to/file.py",
    autofix=True,
    autofix_report_path="/path/to/autofix_report.md"
)
```

### Custom Analyzers

You can specify which analyzers to use:

```python
results = analyze_codebase(
    file_path="/path/to/file.py",
    analyzers=["mypy", "pytea", "pyassistant", "torchtyping"]
)
```

### Custom Configuration

You can provide a custom configuration file:

```python
results = analyze_codebase(
    file_path="/path/to/file.py",
    config_path="/path/to/config.json"
)
```

## For More Information

- [Full Documentation](./README.md)
- [Usage Guide](./USAGE.md)
- [LLM Usage Guide](./LLM_USAGE_GUIDE.md)
- [Auto-fix Guide](./AUTOFIX_GUIDE.md)
- [Error Patterns](./ERROR_PATTERNS.md)

## Conclusion

The ML Static Analysis Framework is a powerful tool for LLMs to analyze ML/DL codebases without running the full training pipeline. It can help detect common issues and suggest fixes, making it easier to optimize code and reduce GPU debugging time.
