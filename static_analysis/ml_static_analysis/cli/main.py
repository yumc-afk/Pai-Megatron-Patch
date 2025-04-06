#
#
#

"""Command-line interface for ML static analysis framework."""

import os
import sys
import argparse
import json
from typing import Dict, List, Optional, Any, Union

from ml_static_analysis.core.config import AnalysisConfig
from ml_static_analysis.core.report import ReportGenerator


def setup_argparse() -> argparse.ArgumentParser:
    """Set up argument parser for the command-line interface.
    
    Returns:
        An argument parser.
    """
    parser = argparse.ArgumentParser(
        description="ML static analysis framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    file_group = parser.add_argument_group("File and directory selection")
    file_group.add_argument(
        "--file",
        type=str,
        help="Path to a specific file to analyze",
    )
    file_group.add_argument(
        "--dir",
        type=str,
        help="Directory to analyze",
    )
    file_group.add_argument(
        "--component",
        type=str,
        default="default",
        help="Name of the component to analyze",
    )
    file_group.add_argument(
        "--component-dir",
        type=str,
        help="Directory containing the component to analyze",
    )
    file_group.add_argument(
        "--exclude",
        type=str,
        help="Glob pattern to exclude files from analysis",
    )
    file_group.add_argument(
        "--include",
        type=str,
        default="*.py",
        help="Glob pattern to include files in analysis",
    )
    
    analyzer_group = parser.add_argument_group("Analyzer selection")
    analyzer_group.add_argument(
        "--mypy",
        action="store_true",
        help="Run MyPy type checking",
    )
    analyzer_group.add_argument(
        "--pytea",
        action="store_true",
        help="Run PyTea tensor shape analysis",
    )
    analyzer_group.add_argument(
        "--pyassistant",
        action="store_true",
        help="Run PyAssistant code quality analysis",
    )
    analyzer_group.add_argument(
        "--jaxtype",
        action="store_true",
        help="Run JaxType tensor type analysis",
    )
    analyzer_group.add_argument(
        "--all",
        action="store_true",
        help="Run all analyzers",
    )
    analyzer_group.add_argument(
        "--lite",
        action="store_true",
        help="Run only core analyzers (lite version)",
    )
    
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config",
        type=str,
        help="Path to a JSON configuration file",
    )
    config_group.add_argument(
        "--save-config",
        type=str,
        help="Path to save the configuration to",
    )
    
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output",
        type=str,
        help="Path to save the analysis report to",
    )
    output_group.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json", "text"],
        default="markdown",
        help="Format of the analysis report",
    )
    output_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    autofix_group = parser.add_argument_group("Auto-fix")
    autofix_group.add_argument(
        "--autofix",
        action="store_true",
        help="Automatically fix issues found by analyzers",
    )
    autofix_group.add_argument(
        "--autofix-dry-run",
        action="store_true",
        help="Show what would be fixed without making changes",
    )
    autofix_group.add_argument(
        "--autofix-report",
        type=str,
        help="Path to save the auto-fix report to",
    )
    
    llm_group = parser.add_argument_group("LLM integration")
    llm_group.add_argument(
        "--llm-mode",
        action="store_true",
        help="Enable LLM integration mode",
    )
    llm_group.add_argument(
        "--llm-output",
        type=str,
        help="Path to save the LLM-friendly output to",
    )
    llm_group.add_argument(
        "--llm-format",
        type=str,
        choices=["markdown", "json"],
        default="markdown",
        help="Format of the LLM-friendly output",
    )
    
    return parser


def get_files_to_analyze(args: argparse.Namespace) -> List[str]:
    """Get the list of files to analyze.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        A list of file paths to analyze.
    """
    import glob
    
    files_to_analyze = []
    
    if args.file:
        if os.path.exists(args.file):
            files_to_analyze.append(os.path.abspath(args.file))
        else:
            print(f"Warning: File {args.file} does not exist.")
    
    if args.dir:
        if os.path.exists(args.dir) and os.path.isdir(args.dir):
            pattern = os.path.join(args.dir, "**", args.include)
            files = glob.glob(pattern, recursive=True)
            
            if args.exclude:
                exclude_pattern = os.path.join(args.dir, "**", args.exclude)
                exclude_files = set(glob.glob(exclude_pattern, recursive=True))
                files = [f for f in files if f not in exclude_files]
            
            files_to_analyze.extend([os.path.abspath(f) for f in files])
        else:
            print(f"Warning: Directory {args.dir} does not exist.")
    
    if args.component_dir:
        if os.path.exists(args.component_dir) and os.path.isdir(args.component_dir):
            pattern = os.path.join(args.component_dir, "**", args.include)
            files = glob.glob(pattern, recursive=True)
            
            if args.exclude:
                exclude_pattern = os.path.join(args.component_dir, "**", args.exclude)
                exclude_files = set(glob.glob(exclude_pattern, recursive=True))
                files = [f for f in files if f not in exclude_files]
            
            files_to_analyze.extend([os.path.abspath(f) for f in files])
        else:
            print(f"Warning: Component directory {args.component_dir} does not exist.")
    
    return files_to_analyze


def run_analyzers(
    args: argparse.Namespace,
    files: List[str],
    config: AnalysisConfig,
) -> Dict[str, Dict[str, Any]]:
    """Run the selected analyzers on the specified files.
    
    Args:
        args: Command-line arguments.
        files: List of file paths to analyze.
        config: Analysis configuration.
        
    Returns:
        A dictionary mapping analyzer names to their results.
    """
    results = {}
    
    run_mypy = args.mypy or args.all or (args.lite and config.is_analyzer_enabled("mypy"))
    run_pytea = args.pytea or args.all or (args.lite and config.is_analyzer_enabled("pytea"))
    run_pyassistant = args.pyassistant or args.all or (args.lite and config.is_analyzer_enabled("pyassistant"))
    run_jaxtype = args.jaxtype or args.all or (args.lite and config.is_analyzer_enabled("jaxtype"))
    
    if run_mypy:
        try:
            from ml_static_analysis.analyzers.mypy_analyzer import run_mypy_analysis
            
            print("Running MyPy type checking...")
            mypy_results = run_mypy_analysis(
                files,
                config.get_analyzer_config("mypy"),
                verbose=args.verbose,
            )
            
            results["MyPy"] = mypy_results
        except ImportError:
            print("Warning: MyPy analyzer not available.")
            results["MyPy"] = {"success": False, "error": "MyPy analyzer not available."}
    
    if run_pytea:
        try:
            from ml_static_analysis.analyzers.pytea_analyzer import run_pytea_analysis
            
            print("Running PyTea tensor shape analysis...")
            pytea_results = run_pytea_analysis(
                files,
                config.get_analyzer_config("pytea"),
                verbose=args.verbose,
            )
            
            results["PyTea"] = pytea_results
        except ImportError:
            print("Warning: PyTea analyzer not available.")
            results["PyTea"] = {"success": False, "error": "PyTea analyzer not available."}
    
    if run_pyassistant:
        try:
            from ml_static_analysis.analyzers.pyassistant_analyzer import run_pyassistant_analysis
            
            print("Running PyAssistant code quality analysis...")
            pyassistant_results = run_pyassistant_analysis(
                files,
                config.get_analyzer_config("pyassistant"),
                verbose=args.verbose,
            )
            
            results["PyAssistant"] = pyassistant_results
        except ImportError:
            print("Warning: PyAssistant analyzer not available.")
            results["PyAssistant"] = {"success": False, "error": "PyAssistant analyzer not available."}
    
    if run_jaxtype:
        try:
            from ml_static_analysis.analyzers.jaxtype_analyzer import run_jaxtype_analysis
            
            print("Running JaxType tensor type analysis...")
            jaxtype_results = run_jaxtype_analysis(
                files,
                config.get_analyzer_config("jaxtype"),
                verbose=args.verbose,
            )
            
            results["JaxType"] = jaxtype_results
        except ImportError:
            print("Warning: JaxType analyzer not available.")
            results["JaxType"] = {"success": False, "error": "JaxType analyzer not available."}
    
    
    if args.autofix or args.autofix_dry_run:
        try:
            from ml_static_analysis.autofix.autofix_manager import AutoFixManager
            
            print("Applying auto-fixes..." if args.autofix else "Checking for potential auto-fixes (dry run)...")
            
            autofix_manager = AutoFixManager(verbose=args.verbose)
            
            if "MyPy" in results and results["MyPy"].get("success", False):
                autofix_manager.apply_fixes(
                    results["MyPy"],
                    "mypy",
                    dry_run=args.autofix_dry_run,
                )
            
            if "PyTea" in results and results["PyTea"].get("success", False):
                autofix_manager.apply_fixes(
                    results["PyTea"],
                    "pytea",
                    dry_run=args.autofix_dry_run,
                )
            
            if "PyAssistant" in results and results["PyAssistant"].get("success", False):
                autofix_manager.apply_fixes(
                    results["PyAssistant"],
                    "pyassistant",
                    dry_run=args.autofix_dry_run,
                )
            
            if "JaxType" in results and results["JaxType"].get("success", False):
                autofix_manager.apply_fixes(
                    results["JaxType"],
                    "jaxtype",
                    dry_run=args.autofix_dry_run,
                )
            
            
            if args.autofix_report:
                fix_report = autofix_manager.generate_fix_report()
                
                os.makedirs(os.path.dirname(args.autofix_report), exist_ok=True)
                
                with open(args.autofix_report, "w", encoding="utf-8") as f:
                    f.write(fix_report)
                
                print(f"Auto-fix report saved to {args.autofix_report}")
            elif autofix_manager.fixes_applied:
                print("\n=== AUTO-FIX REPORT ===\n")
                print(autofix_manager.generate_fix_report())
            
            results["AutoFix"] = {
                "success": True,
                "fixes_applied": len(autofix_manager.fixes_applied),
                "fixes": autofix_manager.fixes_applied,
            }
        except ImportError:
            print("Warning: Auto-fix manager not available.")
            results["AutoFix"] = {"success": False, "error": "Auto-fix manager not available."}
    
    return results


def generate_llm_output(
    results: Dict[str, Dict[str, Any]],
    files: List[str],
    component: str,
    format: str = "markdown",
) -> str:
    """Generate LLM-friendly output from the analysis results.
    
    Args:
        results: Dictionary mapping analyzer names to their results.
        files: List of files that were analyzed.
        component: Name of the component that was analyzed.
        format: Format of the output.
        
    Returns:
        A string containing the LLM-friendly output.
    """
    if format == "json":
        llm_output = {
            "component": component,
            "files_analyzed": len(files),
            "analyzers": {},
        }
        
        for analyzer_name, analyzer_results in results.items():
            if not analyzer_results:
                continue
                
            if analyzer_results.get("success") is False:
                llm_output["analyzers"][analyzer_name] = {
                    "success": False,
                    "error": analyzer_results.get("error", "Unknown error"),
                }
                continue
            
            summary = analyzer_results.get("summary", {})
            findings = analyzer_results.get("findings", {})
            
            llm_output["analyzers"][analyzer_name] = {
                "success": True,
                "total_findings": summary.get("total_findings", 0),
                "findings_by_category": summary.get("findings_by_category", {}),
                "findings_by_severity": summary.get("findings_by_severity", {}),
                "findings": findings,
            }
        
        return json.dumps(llm_output, indent=2)
    else:
        report_generator = ReportGenerator()
        
        return report_generator.generate_report(
            results,
            files,
            component,
            "ML Static Analysis",
        )


def main() -> int:
    """Main entry point for the command-line interface.
    
    Returns:
        Exit code.
    """
    parser = setup_argparse()
    args = parser.parse_args()
    
    config = AnalysisConfig(args.config, verbose=args.verbose)
    
    if args.save_config:
        config.save_config(args.save_config)
    
    files = get_files_to_analyze(args)
    
    if not files:
        print("No files to analyze.")
        return 1
    
    print(f"Analyzing {len(files)} files...")
    
    results = run_analyzers(args, files, config)
    
    report_generator = ReportGenerator(verbose=args.verbose)
    
    report = report_generator.generate_report(
        results,
        files,
        args.component,
        "ML Static Analysis",
    )
    
    if args.output:
        report_generator.save_report(report, args.output)
    else:
        print(report)
    
    if args.llm_mode:
        llm_output = generate_llm_output(
            results,
            files,
            args.component,
            args.llm_format,
        )
        
        if args.llm_output:
            os.makedirs(os.path.dirname(args.llm_output), exist_ok=True)
            
            with open(args.llm_output, "w", encoding="utf-8") as f:
                f.write(llm_output)
        else:
            print("\n\n=== LLM-FRIENDLY OUTPUT ===\n")
            print(llm_output)
    
    return 0


def main_lite() -> int:
    """Main entry point for the lite version of the command-line interface.
    
    Returns:
        Exit code.
    """
    parser = setup_argparse()
    args = parser.parse_args()
    
    args.lite = True
    args.all = False
    
    config = AnalysisConfig(args.config, verbose=args.verbose)
    
    files = get_files_to_analyze(args)
    
    if not files:
        print("No files to analyze.")
        return 1
    
    print(f"Analyzing {len(files)} files (lite mode)...")
    
    results = run_analyzers(args, files, config)
    
    report_generator = ReportGenerator(verbose=args.verbose)
    
    report = report_generator.generate_report(
        results,
        files,
        args.component,
        "ML Static Analysis (Lite)",
    )
    
    if args.output:
        report_generator.save_report(report, args.output)
    else:
        print(report)
    
    if args.llm_mode:
        llm_output = generate_llm_output(
            results,
            files,
            args.component,
            args.llm_format,
        )
        
        if args.llm_output:
            os.makedirs(os.path.dirname(args.llm_output), exist_ok=True)
            
            with open(args.llm_output, "w", encoding="utf-8") as f:
                f.write(llm_output)
        else:
            print("\n\n=== LLM-FRIENDLY OUTPUT ===\n")
            print(llm_output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
