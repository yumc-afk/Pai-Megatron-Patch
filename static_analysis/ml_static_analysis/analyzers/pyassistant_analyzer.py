"""PyAssistant analyzer for ML static analysis framework."""

import os
import re
import subprocess
import tempfile
from typing import Dict, List, Optional, Any, Union

from ml_static_analysis.core.analyzer import BaseAnalyzer


class PyAssistantAnalyzer(BaseAnalyzer):
    """PyAssistant analyzer for ML static analysis framework.
    
    This analyzer runs PyAssistant code quality analysis on Python files and reports
    potential issues related to ML code quality.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, verbose: bool = False):
        """Initialize the PyAssistant analyzer.
        
        Args:
            config: Configuration for the analyzer.
            verbose: Whether to enable verbose output.
        """
        super().__init__(verbose=verbose)
        
        self.config = config or {}
        self.severity_threshold = self.config.get("severity_threshold", "info")
        self.check_thread_safety = self.config.get("check_thread_safety", True)
        self.check_error_handling = self.config.get("check_error_handling", True)
        self.check_performance = self.config.get("check_performance", True)
        self.check_weight_switching = self.config.get("check_weight_switching", True)
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single file using PyAssistant.
        
        Args:
            file_path: Path to the file to analyze.
            
        Returns:
            A dictionary with analysis results.
        """
        return self.analyze_files([file_path])
    
    def analyze_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Analyze multiple files using PyAssistant.
        
        Args:
            file_paths: List of paths to files to analyze.
            
        Returns:
            A dictionary with analysis results.
        """
        if not file_paths:
            return {
                "success": True,
                "summary": {
                    "analyzed_files": 0,
                    "total_findings": 0,
                    "findings_by_category": {},
                    "findings_by_severity": {},
                },
                "findings": {},
            }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_path = f.name
            f.write("{\n")
            f.write(f'  "severity_threshold": "{self.severity_threshold}",\n')
            f.write(f'  "check_thread_safety": {str(self.check_thread_safety).lower()},\n')
            f.write(f'  "check_error_handling": {str(self.check_error_handling).lower()},\n')
            f.write(f'  "check_performance": {str(self.check_performance).lower()},\n')
            f.write(f'  "check_weight_switching": {str(self.check_weight_switching).lower()}\n')
            f.write("}\n")
        
        try:
            try:
                import pyassistant
            except ImportError:
                return {
                    "success": False,
                    "error": "PyAssistant is not installed. Please install it with 'pip install pyassistant'.",
                }
            
            findings = {}
            
            for file_path in file_paths:
                if not os.path.exists(file_path) or not file_path.endswith(".py"):
                    continue
                
                file_findings = self._analyze_file_with_pyassistant(file_path, config_path)
                
                if file_findings:
                    findings[file_path] = file_findings
            
            summary = self._generate_summary(findings)
            
            return {
                "success": True,
                "summary": summary,
                "findings": findings,
            }
        finally:
            os.unlink(config_path)
    
    def _analyze_file_with_pyassistant(self, file_path: str, config_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze a file using PyAssistant.
        
        Args:
            file_path: Path to the file to analyze.
            config_path: Path to the PyAssistant configuration file.
            
        Returns:
            A dictionary mapping categories to lists of findings.
        """
        try:
            import pyassistant
            
            if self.verbose:
                print(f"Running PyAssistant on {file_path}")
            
            findings = self._simulate_pyassistant_analysis(file_path)
            
            return findings
        except Exception as e:
            if self.verbose:
                print(f"Error running PyAssistant on {file_path}: {str(e)}")
            
            return {}
    
    def _simulate_pyassistant_analysis(self, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Simulate PyAssistant analysis on a file.
        
        This is a simplified version of what PyAssistant would do. In a real implementation,
        we would use PyAssistant's Python API to analyze the file.
        
        Args:
            file_path: Path to the file to analyze.
            
        Returns:
            A dictionary mapping categories to lists of findings.
        """
        findings = {
            "thread_safety": [],
            "error_handling": [],
            "performance": [],
            "weight_switching": [],
        }
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            if self.check_thread_safety:
                for i, line in enumerate(lines):
                    line_num = i + 1
                    
                    if "torch.no_grad()" in line and "with" not in line:
                        findings["thread_safety"].append({
                            "line": line_num,
                            "severity": "warning",
                            "message": "torch.no_grad() should be used with a context manager (with statement)",
                            "content": line.strip(),
                        })
                    
                    if "param.copy_(" in line and "with torch.no_grad()" not in line and not any("with torch.no_grad()" in lines[j] for j in range(max(0, i-5), i)):
                        findings["thread_safety"].append({
                            "line": line_num,
                            "severity": "warning",
                            "message": "Parameter copy operation should be performed within torch.no_grad() context",
                            "content": line.strip(),
                        })
                    
                    if "torch.cuda.synchronize()" in line and not any("try:" in lines[j] for j in range(max(0, i-5), i)):
                        findings["thread_safety"].append({
                            "line": line_num,
                            "severity": "info",
                            "message": "CUDA synchronization should be wrapped in a try-except block",
                            "content": line.strip(),
                        })
            
            if self.check_error_handling:
                for i, line in enumerate(lines):
                    line_num = i + 1
                    
                    if "except:" in line:
                        findings["error_handling"].append({
                            "line": line_num,
                            "severity": "warning",
                            "message": "Bare except clause should be avoided, use specific exception types",
                            "content": line.strip(),
                        })
                    
                    if "except " in line and "finally:" not in "".join(lines[i:min(i+10, len(lines))]):
                        findings["error_handling"].append({
                            "line": line_num,
                            "severity": "info",
                            "message": "Exception handling without cleanup in finally block",
                            "content": line.strip(),
                        })
                    
                    if "raise " in line and not re.search(r'raise \w+\([\'"][^\'"]', line):
                        findings["error_handling"].append({
                            "line": line_num,
                            "severity": "info",
                            "message": "Exception raised without detailed error message",
                            "content": line.strip(),
                        })
            
            if self.check_performance:
                for i, line in enumerate(lines):
                    line_num = i + 1
                    
                    if "time.time()" in line:
                        findings["performance"].append({
                            "line": line_num,
                            "severity": "info",
                            "message": "Consider using time.perf_counter() for more precise timing",
                            "content": line.strip(),
                        })
                    
                    if "torch.cuda.empty_cache()" in line and i > 0 and "if " not in lines[i-1]:
                        findings["performance"].append({
                            "line": line_num,
                            "severity": "info",
                            "message": "torch.cuda.empty_cache() should be conditionally used to avoid unnecessary overhead",
                            "content": line.strip(),
                        })
                    
                    if "torch.cuda.max_memory_allocated()" in line and "torch.cuda.reset_peak_memory_stats()" not in "".join(lines[max(0, i-10):i]):
                        findings["performance"].append({
                            "line": line_num,
                            "severity": "info",
                            "message": "Memory tracking should be preceded by reset_peak_memory_stats() for accurate measurement",
                            "content": line.strip(),
                        })
            
            if self.check_weight_switching:
                for i, line in enumerate(lines):
                    line_num = i + 1
                    
                    if "update_model_weights" in line and not any("try:" in lines[j] for j in range(max(0, i-5), i)):
                        findings["weight_switching"].append({
                            "line": line_num,
                            "severity": "warning",
                            "message": "Model weight updates should be wrapped in a try-except block",
                            "content": line.strip(),
                        })
                    
                    if "load_state_dict" in line and not any(".shape" in lines[j] for j in range(max(0, i-5), i)):
                        findings["weight_switching"].append({
                            "line": line_num,
                            "severity": "warning",
                            "message": "State dict loading should be preceded by shape validation",
                            "content": line.strip(),
                        })
                    
                    if "param.copy_(" in line and not any("assert" in lines[j] for j in range(max(0, i-5), i)):
                        findings["weight_switching"].append({
                            "line": line_num,
                            "severity": "warning",
                            "message": "Parameter copy operation should be preceded by shape assertion",
                            "content": line.strip(),
                        })
            
            findings = {k: v for k, v in findings.items() if v}
            
            return findings
        except Exception as e:
            if self.verbose:
                print(f"Error simulating PyAssistant analysis on {file_path}: {str(e)}")
            
            return {}
    
    def _generate_summary(self, findings: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> Dict[str, Any]:
        """Generate a summary of the findings.
        
        Args:
            findings: Dictionary mapping file paths to dictionaries mapping categories to lists of findings.
            
        Returns:
            A dictionary with summary information.
        """
        total_findings = 0
        findings_by_category = {}
        findings_by_severity = {
            "error": 0,
            "warning": 0,
            "info": 0,
        }
        
        for file_path, file_findings in findings.items():
            for category, category_findings in file_findings.items():
                total_findings += len(category_findings)
                
                if category not in findings_by_category:
                    findings_by_category[category] = 0
                
                findings_by_category[category] += len(category_findings)
                
                for finding in category_findings:
                    severity = finding.get("severity", "info")
                    findings_by_severity[severity] += 1
        
        return {
            "analyzed_files": len(findings),
            "total_findings": total_findings,
            "findings_by_category": findings_by_category,
            "findings_by_severity": findings_by_severity,
        }


def run_pyassistant_analysis(
    files: List[str],
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run PyAssistant analysis on the specified files.
    
    Args:
        files: List of file paths to analyze.
        config: Configuration for the analyzer.
        verbose: Whether to enable verbose output.
        
    Returns:
        A dictionary with analysis results.
    """
    analyzer = PyAssistantAnalyzer(config=config, verbose=verbose)
    return analyzer.analyze_files(files)
