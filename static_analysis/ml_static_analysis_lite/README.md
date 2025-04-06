# ML Static Analysis Framework (Lite Version)

This is the lite version of the ML Static Analysis Framework, which includes only the core analyzers and LLM orchestration capabilities. It's designed to be lightweight and easy to use for LLMs that want to analyze ML codebases without the full set of analyzers.

## Features

- LLM orchestration for static analysis
- Core analyzers:
  - MyPy: Static type checking for Python
  - PyTea: Tensor shape analysis for PyTorch
  - PyAssistant: Code quality analysis for Python
  - JaxType: Enhanced tensor type analysis for PyTorch
- Simplified interface for LLMs
- Auto-fix capabilities for common issues

## Installation

```bash
pip install ml-static-analysis[lite]
```

## Usage

```python
from ml_static_analysis_lite import analyze_codebase

# Analyze a file
results = analyze_codebase(file_path="/path/to/file.py")

# Get the analysis report
report = results["report"]
print(report)
```

For more information, see the [LLM Usage Guide](../LLM_USAGE_GUIDE.md).
