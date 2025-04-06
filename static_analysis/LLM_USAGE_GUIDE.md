# LLM Usage Guide for ML Static Analysis Framework

This guide provides instructions for Large Language Models (LLMs) like Devin to use the ML Static Analysis Framework to analyze machine learning codebases, identify potential issues, and provide recommendations for improvements.

## Overview

The ML Static Analysis Framework is designed to help LLMs analyze machine learning codebases, particularly those using PyTorch for distributed training. It provides a simple interface for LLMs to:

1. Analyze specific files, directories, or components
2. Run various static analysis tools (MyPy, PyTea, PyAssistant, TorchTyping, etc.)
3. Generate comprehensive reports with findings and recommendations
4. Identify potential issues before running code on GPUs

The framework supports a "static analysis → CPU validation → GPU execution" workflow that can significantly reduce GPU debugging time and improve development efficiency.

## Quick Start

After installation, LLMs can use the framework through the `ml_static_analysis.llm_interface` module:

```python
from ml_static_analysis.llm_interface import analyze_codebase

# Analyze a specific file
results = analyze_codebase(file_path="/path/to/file.py")

# Analyze a directory
results = analyze_codebase(dir_path="/path/to/dir")

# Analyze a component
results = analyze_codebase(component="deepseek_v3")

# Get the analysis report
report = results["report"]
print(report)

# Get the detailed findings
findings = results["findings"]
for analyzer_name, analyzer_findings in findings["analyzers"].items():
    print(f"Findings from {analyzer_name}:")
    for file_path, file_findings in analyzer_findings.get("findings", {}).items():
        print(f"  {file_path}:")
        for category, category_findings in file_findings.items():
            print(f"    {category}:")
            for finding in category_findings:
                print(f"      {finding['message']} (Line {finding['line']})")
```

## Installation

The framework can be installed using pip:

```bash
# Install standard version (all analyzers)
pip install ml-static-analysis

# Install lite version (core analyzers only)
pip install ml-static-analysis[lite]
```

Alternatively, you can install from source:

```bash
git clone https://github.com/yumc-afk/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch/static_analysis
pip install -e .
```

## LLM Workflow

As an LLM, you can follow this workflow to analyze ML codebases:

1. **Understand the task**: Determine what part of the codebase needs to be analyzed
2. **Select analysis targets**: Identify specific files, directories, or components to analyze
3. **Run analysis**: Use the `analyze_codebase` function to run the appropriate analyzers
4. **Interpret results**: Review the report and findings to identify potential issues
5. **Provide recommendations**: Based on the findings, provide recommendations for improvements
6. **Validate on CPU**: Suggest CPU-based validation tests before running on GPUs
7. **Monitor GPU execution**: If GPU execution is necessary, provide guidance on what to monitor

## API Reference

### Main Functions

#### `analyze_codebase`

```python
def analyze_codebase(
    file_path: Optional[str] = None,
    dir_path: Optional[str] = None,
    component: Optional[str] = None,
    component_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    output_path: Optional[str] = None,
    analyzers: Optional[List[str]] = None,
    lite_mode: bool = False,
    verbose: bool = False,
    autofix: bool = False,
    autofix_dry_run: bool = False,
    autofix_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze a codebase using the ML static analysis framework."""
```

#### `analyze_file`

```python
def analyze_file(
    file_path: str,
    analyzers: Optional[List[str]] = None,
    lite_mode: bool = False,
    verbose: bool = False,
    autofix: bool = False,
    autofix_dry_run: bool = False,
    autofix_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze a specific file using the ML static analysis framework."""
```

#### `analyze_directory`

```python
def analyze_directory(
    dir_path: str,
    analyzers: Optional[List[str]] = None,
    lite_mode: bool = False,
    verbose: bool = False,
    autofix: bool = False,
    autofix_dry_run: bool = False,
    autofix_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze a directory using the ML static analysis framework."""
```

#### `analyze_component`

```python
def analyze_component(
    component: str,
    component_dir: Optional[str] = None,
    analyzers: Optional[List[str]] = None,
    lite_mode: bool = False,
    verbose: bool = False,
    autofix: bool = False,
    autofix_dry_run: bool = False,
    autofix_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze a component using the ML static analysis framework."""
```

### Return Value

The analysis functions return a dictionary with the following structure:

```python
{
    "success": True,  # Whether the analysis was successful
    "report": "...",  # Markdown report with findings and recommendations
    "findings": {     # Detailed findings in JSON format
        "component": "...",
        "files_analyzed": 42,
        "analyzers": {
            "MyPy": {
                "success": True,
                "total_findings": 10,
                "findings_by_category": {...},
                "findings_by_severity": {...},
                "findings": {...}
            },
            # Other analyzers...
        }
    },
    "report_path": "/path/to/report.md",  # Path to the saved report
    "findings_path": "/path/to/findings.json",  # Path to the saved findings
    "autofix_report": "...",  # Only present if autofix or autofix_dry_run is True
    "autofix_report_path": "/path/to/autofix_report.md",  # Only present if autofix_report_path is provided
    "autofix_data": {  # Only present if autofix or autofix_dry_run is True and no autofix_report_path is provided
        "fixes_applied": 10,
        "fixes_by_analyzer": {...},
        "fixes": [
            {
                "file": "/path/to/file.py",
                "line": 42,
                "analyzer": "MyPy",
                "original": "def func():",
                "fixed": "def func() -> None:"
            },
            # Other fixes...
        ]
    }
}
```

## Available Analyzers

The framework includes the following analyzers:

- **MyPy**: Static type checking for Python
- **PyTea**: Tensor shape analysis for PyTorch
- **PyAssistant**: Code quality analysis for Python
- **JaxType**: Tensor type analysis for PyTorch (替代TorchTyping的现代选择)
- **Pattern Analysis**: Pattern-based code analysis for common issues
- **AutoFix**: Automatic fixing of issues detected by other analyzers

在精简版中，只有核心分析器（MyPy、PyTea、PyAssistant、JaxType）可用。

## Auto-Fix Functionality

The framework includes an auto-fix feature that can automatically fix common issues detected by the analyzers. This feature is particularly useful for LLMs that want to quickly improve code quality without manually editing files.

### Using Auto-Fix

To use the auto-fix feature, set the `autofix` parameter to `True` when calling the analysis functions:

```python
# Analyze a file and automatically fix issues
results = analyze_file(
    file_path="/path/to/file.py",
    autofix=True
)

# Check if auto-fix was successful
if "autofix_report" in results:
    print(results["autofix_report"])
```

To see what would be fixed without actually making changes (dry run):

```python
# Analyze a file and show what would be fixed
results = analyze_file(
    file_path="/path/to/file.py",
    autofix_dry_run=True
)

# Check the potential fixes
if "autofix_data" in results:
    fixes = results["autofix_data"]["fixes"]
    for fix in fixes:
        print(f"File: {fix['file']}")
        print(f"Line: {fix['line']}")
        print(f"Original: {fix['original']}")
        print(f"Fixed: {fix['fixed']}")
        print()
```

To save a report of the fixes applied:

```python
# Analyze a file, automatically fix issues, and save a report
results = analyze_file(
    file_path="/path/to/file.py",
    autofix=True,
    autofix_report_path="/path/to/autofix_report.md"
)

# The report is saved to the specified path
print(f"Auto-fix report saved to: {results['autofix_report_path']}")
```

### Types of Fixes

The auto-fix feature can fix various types of issues:

- **MyPy Fixes**: Missing type annotations, incorrect type annotations, missing imports for types
- **PyTea Fixes**: Tensor shape checks, tensor dtype checks, tensor device checks
- **PyAssistant Fixes**: Code style issues, best practices violations, performance optimizations
- **TorchTyping Fixes**: Missing tensor type annotations, incorrect tensor type annotations
- **Pattern Analysis Fixes**: Thread safety issues, error handling issues, performance issues

For more information, see the [Auto-Fix Guide](AUTOFIX_GUIDE.md).

## Common Analysis Patterns

### Analyzing MLA (Multi-Latent Attention) Components

```python
results = analyze_component(
    component="mla",
    component_dir="/path/to/mla",
    analyzers=["pytea", "jaxtype"],
)
```

### Analyzing MOE (Mixture of Experts) Components

```python
results = analyze_component(
    component="moe",
    component_dir="/path/to/moe",
    analyzers=["pytea", "jaxtype", "pattern"],
)
```

### Analyzing Distributed Training Code

```python
results = analyze_directory(
    dir_path="/path/to/distributed",
    analyzers=["mypy", "pattern"],
)
```

### Analyzing Weight Switching Code

```python
results = analyze_directory(
    dir_path="/path/to/weight_switching",
    analyzers=["pattern"],
)
```

## Interpreting Results

When interpreting the results, focus on:

1. **Critical issues**: Look for errors and warnings that could cause runtime failures
2. **Performance issues**: Identify patterns that could lead to poor performance
3. **Thread safety issues**: Check for potential race conditions in distributed code
4. **Tensor shape issues**: Verify that tensor shapes are properly validated
5. **Type issues**: Ensure that types are properly annotated and checked

## Recommending CPU Validation

After identifying potential issues, recommend CPU validation tests to verify the fixes before running on GPUs. For example:

```python
# Example CPU validation test for MLA component
def test_mla_component():
    # Create small test inputs
    batch_size = 2
    seq_len = 16
    hidden_size = 64
    num_heads = 4
    
    # Create random inputs
    import torch
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, 1, 1, seq_len)
    
    # Import the component
    from path.to.mla import MultiLatentAttention
    
    # Create the component with small dimensions
    mla = MultiLatentAttention(hidden_size, num_heads)
    
    # Run the component
    output = mla(hidden_states, attention_mask)
    
    # Verify output shape
    assert output.shape == (batch_size, seq_len, hidden_size)
    
    print("MLA component test passed!")
```

## Example Workflow

Here's an example workflow for analyzing a DeepSeek V3 model:

1. **Analyze the MLA component**:
   ```python
   mla_results = analyze_component(
       component="mla",
       component_dir="/path/to/deepseek_v3/mla",
   )
   ```

2. **Analyze the MOE component**:
   ```python
   moe_results = analyze_component(
       component="moe",
       component_dir="/path/to/deepseek_v3/moe",
   )
   ```

3. **Analyze the distributed training code**:
   ```python
   dist_results = analyze_directory(
       dir_path="/path/to/deepseek_v3/distributed",
   )
   ```

4. **Combine the results and provide recommendations**:
   ```python
   # Extract key findings
   mla_findings = mla_results["findings"]
   moe_findings = moe_results["findings"]
   dist_findings = dist_results["findings"]
   
   # Provide recommendations
   recommendations = []
   
   # Add MLA recommendations
   if mla_findings["analyzers"]["PyTea"]["total_findings"] > 0:
       recommendations.append("Fix tensor shape issues in MLA component")
   
   # Add MOE recommendations
   if moe_findings["analyzers"]["Pattern Analysis"]["total_findings"] > 0:
       recommendations.append("Fix pattern issues in MOE component")
   
   # Add distributed training recommendations
   if dist_findings["analyzers"]["MyPy"]["total_findings"] > 0:
       recommendations.append("Fix type issues in distributed training code")
   
   # Print recommendations
   for i, recommendation in enumerate(recommendations, 1):
       print(f"{i}. {recommendation}")
   ```

5. **Recommend CPU validation tests**:
   ```python
   print("Recommended CPU validation tests:")
   print("1. Test MLA component with small inputs")
   print("2. Test MOE component with small inputs")
   print("3. Test distributed training with single-node setup")
   ```

## Conclusion

By using the ML Static Analysis Framework, LLMs can help developers identify and fix potential issues in ML codebases before running them on GPUs. This can significantly reduce debugging time and improve development efficiency.

For more information, see the following resources:

- [README.md](README.md): Overview of the framework
- [USAGE.md](USAGE.md): Detailed usage instructions
- [ERROR_PATTERNS.md](ERROR_PATTERNS.md): Common error patterns and solutions
- [LLM_ORCHESTRATION_GUIDE.md](LLM_ORCHESTRATION_GUIDE.md): Guide for LLMs orchestrating static analysis
- [LLM_DECISION_FRAMEWORKS.md](LLM_DECISION_FRAMEWORKS.md): Decision frameworks for LLMs
