# Auto-Fix Guide for ML Static Analysis Framework

This guide explains how to use the auto-fix capabilities of the ML static analysis framework to automatically fix common issues in ML codebases.

## Overview

The ML static analysis framework includes an auto-fix feature that can automatically fix common issues detected by the analyzers. This feature is particularly useful for LLMs that want to quickly improve code quality without manually editing files.

The auto-fix feature supports fixing issues detected by the following analyzers:

- **MyPy**: Type annotation issues
- **PyTea**: Tensor shape and dimension issues
- **PyAssistant**: Code quality and best practices
- **JaxType**: Tensor type annotation issues
- **Pattern Analysis**: Common ML code patterns and anti-patterns

## Using Auto-Fix from the Command Line

To use the auto-fix feature from the command line, use the following options:

```bash
ml-analyze --file /path/to/file.py --autofix
```

This will analyze the file and automatically fix any issues that can be fixed.

To see what would be fixed without actually making changes (dry run):

```bash
ml-analyze --file /path/to/file.py --autofix-dry-run
```

To save a report of the fixes applied:

```bash
ml-analyze --file /path/to/file.py --autofix --autofix-report /path/to/report.md
```

## Using Auto-Fix from Python

To use the auto-fix feature from Python, use the `analyze_codebase` function from the `ml_static_analysis.llm_interface` module:

```python
from ml_static_analysis.llm_interface import analyze_codebase

results = analyze_codebase(
    file_path="/path/to/file.py",
    autofix=True,
    autofix_report_path="/path/to/report.md"
)

# Check if auto-fix was successful
if "autofix_report" in results:
    print(results["autofix_report"])
```

For a dry run:

```python
results = analyze_codebase(
    file_path="/path/to/file.py",
    autofix_dry_run=True,
    autofix_report_path="/path/to/report.md"
)
```

## Using Auto-Fix with LLMs

LLMs can use the auto-fix feature to automatically improve code quality. Here's a recommended workflow:

1. Analyze the codebase to identify issues:

```python
from ml_static_analysis.llm_interface import analyze_codebase

results = analyze_codebase(
    dir_path="/path/to/codebase",
    autofix_dry_run=True
)
```

2. Review the potential fixes in the results:

```python
if "autofix_data" in results:
    fixes = results["autofix_data"]["fixes"]
    for fix in fixes:
        print(f"File: {fix['file']}")
        print(f"Line: {fix['line']}")
        print(f"Original: {fix['original']}")
        print(f"Fixed: {fix['fixed']}")
        print()
```

3. Apply the fixes if they look good:

```python
results = analyze_codebase(
    dir_path="/path/to/codebase",
    autofix=True,
    autofix_report_path="/path/to/autofix_report.md"
)
```

## Types of Fixes

The auto-fix feature can fix various types of issues:

### MyPy Fixes

- Missing type annotations
- Incorrect type annotations
- Missing imports for types

### PyTea Fixes

- Tensor shape checks before reshape/view operations
- Tensor dtype checks before conversion operations
- Tensor device checks before device transfer operations

### PyAssistant Fixes

- Code style issues
- Best practices violations
- Performance optimizations

### JaxType Fixes

- Missing tensor type annotations
- Incorrect tensor type annotations
- Missing imports for JaxType
- Tensor shape constraints and validation

### Pattern Analysis Fixes

- Thread safety issues (e.g., proper use of `torch.no_grad()`)
- Error handling issues (e.g., proper exception handling)
- Performance issues (e.g., using `time.perf_counter()` instead of `time.time()`)
- Weight switching issues (e.g., using `torch.no_grad()` for parameter copying)

## Limitations

The auto-fix feature has some limitations:

- It can only fix issues that have a clear, deterministic solution
- It may not fix complex issues that require context-aware changes
- It may not fix issues that require significant refactoring
- It may not fix issues that require domain knowledge

## Customizing Auto-Fix

You can customize the auto-fix behavior by modifying the configuration file:

```json
{
  "autofix": {
    "enabled": true,
    "dry_run": false,
    "analyzers": ["mypy", "pytea", "pyassistant", "jaxtype", "pattern"],
    "severity_threshold": "warning"
  }
}
```

This configuration enables auto-fix for all analyzers and only fixes issues with severity "warning" or higher.

## Conclusion

The auto-fix feature is a powerful tool for improving code quality automatically. It's particularly useful for LLMs that want to quickly improve code quality without manually editing files.

For more information, see the [LLM Usage Guide](LLM_USAGE_GUIDE.md) and the [API Reference](API_REFERENCE.md).
