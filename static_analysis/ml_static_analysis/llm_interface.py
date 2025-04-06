#
#
#

"""LLM interface for ML static analysis framework.

This module provides a simple interface for LLMs to use the ML static analysis framework.
It is designed to be used by LLMs like Devin to analyze ML codebases and provide
recommendations for improvements.

Example usage:
```python
from ml_static_analysis.llm_interface import analyze_codebase

results = analyze_codebase(file_path="/path/to/file.py")

results = analyze_codebase(dir_path="/path/to/dir")

results = analyze_codebase(component="deepseek_v3")

report = results["report"]
print(report)

findings = results["findings"]
for analyzer, analyzer_findings in findings.items():
    print(f"Findings from {analyzer}:")
    for file_path, file_findings in analyzer_findings.items():
        print(f"  {file_path}:")
        for finding in file_findings:
            print(f"    {finding}")
```
"""

import os
import sys
import json
import tempfile
from typing import Dict, List, Optional, Any, Union

from ml_static_analysis.core.config import AnalysisConfig
from ml_static_analysis.core.report import ReportGenerator


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
    """Analyze a codebase using the ML static analysis framework.
    
    This function provides a simple interface for LLMs to analyze ML codebases
    and get recommendations for improvements.
    
    Args:
        file_path: Path to a specific file to analyze.
        dir_path: Path to a directory to analyze.
        component: Name of a component to analyze.
        component_dir: Path to a directory containing the component to analyze.
        config_path: Path to a configuration file.
        output_path: Path to save the analysis report to.
        analyzers: List of analyzers to run. If None, all analyzers will be run.
        lite_mode: Whether to run in lite mode (only core analyzers).
        verbose: Whether to enable verbose output.
        autofix: Whether to automatically fix issues found by analyzers.
        autofix_dry_run: Whether to show what would be fixed without making changes.
        autofix_report_path: Path to save the auto-fix report to.
        
    Returns:
        A dictionary with the analysis results, including the report and detailed findings.
    """
    if not any([file_path, dir_path, component, component_dir]):
        raise ValueError("At least one of file_path, dir_path, component, or component_dir must be provided.")
    
    if not output_path:
        output_dir = tempfile.mkdtemp(prefix="ml_static_analysis_")
        output_path = os.path.join(output_dir, "analysis_report.md")
    else:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
    
    args = ["ml-analyze"]
    
    if file_path:
        args.extend(["--file", file_path])
    
    if dir_path:
        args.extend(["--dir", dir_path])
    
    if component:
        args.extend(["--component", component])
    
    if component_dir:
        args.extend(["--component-dir", component_dir])
    
    if config_path:
        args.extend(["--config", config_path])
    
    if output_path:
        args.extend(["--output", output_path])
    
    if lite_mode:
        args.append("--lite")
    
    if verbose:
        args.append("--verbose")
    
    if analyzers:
        for analyzer in analyzers:
            if analyzer.lower() == "mypy":
                args.append("--mypy")
            elif analyzer.lower() == "pytea":
                args.append("--pytea")
            elif analyzer.lower() == "pyassistant":
                args.append("--pyassistant")
            elif analyzer.lower() == "jaxtype":
                args.append("--jaxtype")
            elif analyzer.lower() == "pattern":
                args.append("--pattern")
    else:
        args.append("--all")
    
    if autofix:
        args.append("--autofix")
    
    if autofix_dry_run:
        args.append("--autofix-dry-run")
    
    if autofix_report_path:
        args.extend(["--autofix-report", autofix_report_path])
    
    args.append("--llm-mode")
    
    json_output_path = os.path.join(output_dir, "analysis_findings.json")
    args.extend(["--llm-output", json_output_path, "--llm-format", "json"])
    
    from ml_static_analysis.cli.main import main as cli_main
    
    original_argv = sys.argv.copy()
    
    try:
        sys.argv = args
        
        exit_code = cli_main()
        
        if exit_code != 0:
            return {
                "success": False,
                "error": f"Analysis failed with exit code {exit_code}",
                "report": "",
                "findings": {},
            }
        
        with open(output_path, "r", encoding="utf-8") as f:
            report = f.read()
        
        with open(json_output_path, "r", encoding="utf-8") as f:
            findings = json.load(f)
        
        result = {
            "success": True,
            "report": report,
            "findings": findings,
            "report_path": output_path,
            "findings_path": json_output_path,
        }
        
        if autofix or autofix_dry_run:
            if autofix_report_path and os.path.exists(autofix_report_path):
                with open(autofix_report_path, "r", encoding="utf-8") as f:
                    autofix_report = f.read()
                result["autofix_report"] = autofix_report
                result["autofix_report_path"] = autofix_report_path
            elif "AutoFix" in findings.get("analyzers", {}):
                autofix_data = findings["analyzers"]["AutoFix"]
                result["autofix_data"] = autofix_data
        
        return result
    finally:
        sys.argv = original_argv


def analyze_file(
    file_path: str,
    analyzers: Optional[List[str]] = None,
    lite_mode: bool = False,
    verbose: bool = False,
    autofix: bool = False,
    autofix_dry_run: bool = False,
    autofix_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze a specific file using the ML static analysis framework.
    
    This is a convenience function that calls analyze_codebase with file_path.
    
    Args:
        file_path: Path to the file to analyze.
        analyzers: List of analyzers to run. If None, all analyzers will be run.
        lite_mode: Whether to run in lite mode (only core analyzers).
        verbose: Whether to enable verbose output.
        autofix: Whether to automatically fix issues found by analyzers.
        autofix_dry_run: Whether to show what would be fixed without making changes.
        autofix_report_path: Path to save the auto-fix report to.
        
    Returns:
        A dictionary with the analysis results, including the report and detailed findings.
    """
    return analyze_codebase(
        file_path=file_path,
        analyzers=analyzers,
        lite_mode=lite_mode,
        verbose=verbose,
        autofix=autofix,
        autofix_dry_run=autofix_dry_run,
        autofix_report_path=autofix_report_path,
    )


def analyze_directory(
    dir_path: str,
    analyzers: Optional[List[str]] = None,
    lite_mode: bool = False,
    verbose: bool = False,
    autofix: bool = False,
    autofix_dry_run: bool = False,
    autofix_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze a directory using the ML static analysis framework.
    
    This is a convenience function that calls analyze_codebase with dir_path.
    
    Args:
        dir_path: Path to the directory to analyze.
        analyzers: List of analyzers to run. If None, all analyzers will be run.
        lite_mode: Whether to run in lite mode (only core analyzers).
        verbose: Whether to enable verbose output.
        autofix: Whether to automatically fix issues found by analyzers.
        autofix_dry_run: Whether to show what would be fixed without making changes.
        autofix_report_path: Path to save the auto-fix report to.
        
    Returns:
        A dictionary with the analysis results, including the report and detailed findings.
    """
    return analyze_codebase(
        dir_path=dir_path,
        analyzers=analyzers,
        lite_mode=lite_mode,
        verbose=verbose,
        autofix=autofix,
        autofix_dry_run=autofix_dry_run,
        autofix_report_path=autofix_report_path,
    )


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
    """Analyze a component using the ML static analysis framework.
    
    This is a convenience function that calls analyze_codebase with component and component_dir.
    
    Args:
        component: Name of the component to analyze.
        component_dir: Path to the directory containing the component to analyze.
        analyzers: List of analyzers to run. If None, all analyzers will be run.
        lite_mode: Whether to run in lite mode (only core analyzers).
        verbose: Whether to enable verbose output.
        autofix: Whether to automatically fix issues found by analyzers.
        autofix_dry_run: Whether to show what would be fixed without making changes.
        autofix_report_path: Path to save the auto-fix report to.
        
    Returns:
        A dictionary with the analysis results, including the report and detailed findings.
    """
    return analyze_codebase(
        component=component,
        component_dir=component_dir,
        analyzers=analyzers,
        lite_mode=lite_mode,
        verbose=verbose,
        autofix=autofix,
        autofix_dry_run=autofix_dry_run,
        autofix_report_path=autofix_report_path,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM interface for ML static analysis framework")
    parser.add_argument("--file", type=str, help="Path to a specific file to analyze")
    parser.add_argument("--dir", type=str, help="Path to a directory to analyze")
    parser.add_argument("--component", type=str, help="Name of a component to analyze")
    parser.add_argument("--component-dir", type=str, help="Path to a directory containing the component to analyze")
    parser.add_argument("--output", type=str, help="Path to save the analysis report to")
    parser.add_argument("--lite", action="store_true", help="Run in lite mode (only core analyzers)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if not any([args.file, args.dir, args.component, args.component_dir]):
        parser.error("At least one of --file, --dir, --component, or --component-dir must be provided.")
    
    results = analyze_codebase(
        file_path=args.file,
        dir_path=args.dir,
        component=args.component,
        component_dir=args.component_dir,
        output_path=args.output,
        lite_mode=args.lite,
        verbose=args.verbose,
    )
    
    if results["success"]:
        print(f"Analysis report saved to {results['report_path']}")
        print(f"Detailed findings saved to {results['findings_path']}")
    else:
        print(f"Analysis failed: {results['error']}")
