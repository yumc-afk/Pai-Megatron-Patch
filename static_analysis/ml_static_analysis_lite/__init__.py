"""ML Static Analysis Framework (Lite Version)

This is the lite version of the ML Static Analysis Framework, which includes only the core analyzers
and LLM orchestration capabilities. It's designed to be lightweight and easy to use for LLMs that
want to analyze ML codebases without the full set of analyzers.
"""

from typing import Dict, List, Optional, Any, Union
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from ml_static_analysis.llm_interface import (
    analyze_codebase as _analyze_codebase,
    analyze_file as _analyze_file,
    analyze_directory as _analyze_directory,
    analyze_component as _analyze_component,
)

LITE_ANALYZERS = ["mypy", "pytea", "pyassistant", "jaxtype"]

def analyze_codebase(
    file_path: Optional[str] = None,
    dir_path: Optional[str] = None,
    component: Optional[str] = None,
    component_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    output_path: Optional[str] = None,
    analyzers: Optional[List[str]] = None,
    verbose: bool = False,
    autofix: bool = False,
    autofix_dry_run: bool = False,
    autofix_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze a codebase using the ML static analysis framework (lite version).
    
    This is a wrapper around the full analyze_codebase function that enforces lite mode.
    
    Args:
        file_path: Path to a specific file to analyze.
        dir_path: Path to a directory to analyze.
        component: Name of a component to analyze.
        component_dir: Path to the component directory.
        config_path: Path to a configuration file.
        output_path: Path to save the output.
        analyzers: List of analyzers to use. If None, all lite analyzers will be used.
        verbose: Whether to enable verbose output.
        autofix: Whether to automatically fix issues.
        autofix_dry_run: Whether to show what would be fixed without making changes.
        autofix_report_path: Path to save the autofix report.
        
    Returns:
        A dictionary with analysis results.
    """
    if analyzers is None:
        analyzers = LITE_ANALYZERS
    else:
        analyzers = [a for a in analyzers if a in LITE_ANALYZERS]
    
    return _analyze_codebase(
        file_path=file_path,
        dir_path=dir_path,
        component=component,
        component_dir=component_dir,
        config_path=config_path,
        output_path=output_path,
        analyzers=analyzers,
        lite_mode=True,
        verbose=verbose,
        autofix=autofix,
        autofix_dry_run=autofix_dry_run,
        autofix_report_path=autofix_report_path,
    )

def analyze_file(
    file_path: str,
    analyzers: Optional[List[str]] = None,
    verbose: bool = False,
    autofix: bool = False,
    autofix_dry_run: bool = False,
    autofix_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze a specific file using the ML static analysis framework (lite version).
    
    Args:
        file_path: Path to the file to analyze.
        analyzers: List of analyzers to use. If None, all lite analyzers will be used.
        verbose: Whether to enable verbose output.
        autofix: Whether to automatically fix issues.
        autofix_dry_run: Whether to show what would be fixed without making changes.
        autofix_report_path: Path to save the autofix report.
        
    Returns:
        A dictionary with analysis results.
    """
    return analyze_codebase(
        file_path=file_path,
        analyzers=analyzers,
        verbose=verbose,
        autofix=autofix,
        autofix_dry_run=autofix_dry_run,
        autofix_report_path=autofix_report_path,
    )

def analyze_directory(
    dir_path: str,
    analyzers: Optional[List[str]] = None,
    verbose: bool = False,
    autofix: bool = False,
    autofix_dry_run: bool = False,
    autofix_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze a directory using the ML static analysis framework (lite version).
    
    Args:
        dir_path: Path to the directory to analyze.
        analyzers: List of analyzers to use. If None, all lite analyzers will be used.
        verbose: Whether to enable verbose output.
        autofix: Whether to automatically fix issues.
        autofix_dry_run: Whether to show what would be fixed without making changes.
        autofix_report_path: Path to save the autofix report.
        
    Returns:
        A dictionary with analysis results.
    """
    return analyze_codebase(
        dir_path=dir_path,
        analyzers=analyzers,
        verbose=verbose,
        autofix=autofix,
        autofix_dry_run=autofix_dry_run,
        autofix_report_path=autofix_report_path,
    )

def analyze_component(
    component: str,
    component_dir: Optional[str] = None,
    analyzers: Optional[List[str]] = None,
    verbose: bool = False,
    autofix: bool = False,
    autofix_dry_run: bool = False,
    autofix_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze a component using the ML static analysis framework (lite version).
    
    Args:
        component: Name of the component to analyze.
        component_dir: Path to the component directory.
        analyzers: List of analyzers to use. If None, all lite analyzers will be used.
        verbose: Whether to enable verbose output.
        autofix: Whether to automatically fix issues.
        autofix_dry_run: Whether to show what would be fixed without making changes.
        autofix_report_path: Path to save the autofix report.
        
    Returns:
        A dictionary with analysis results.
    """
    return analyze_codebase(
        component=component,
        component_dir=component_dir,
        analyzers=analyzers,
        verbose=verbose,
        autofix=autofix,
        autofix_dry_run=autofix_dry_run,
        autofix_report_path=autofix_report_path,
    )
