"""JaxType analyzer for ML static analysis framework."""

import os
import re
import subprocess
import tempfile
from typing import Dict, List, Optional, Any, Union

from ml_static_analysis.core.analyzer import BaseAnalyzer
from ml_static_analysis.core.report import AnalysisReport, Finding, Severity


class JaxTypeAnalyzer(BaseAnalyzer):
    """JaxType analyzer for ML static analysis framework.
    
    This analyzer checks for proper tensor type annotations using JaxType
    and reports potential tensor shape and type issues.
    """
    
    def __init__(self, config):
        """Initialize the JaxType analyzer.
        
        Args:
            config: Configuration for the analyzer.
        """
        super().__init__(config)
        
        self.name = "JaxTypeAnalyzer"
        self.severity_threshold = self.config.get_analyzer_config("jaxtype").get("severity_threshold", "info")
        self.check_shapes = self.config.get_analyzer_config("jaxtype").get("check_shapes", True)
        self.check_dtypes = self.config.get_analyzer_config("jaxtype").get("check_dtypes", True)
        self.check_devices = self.config.get_analyzer_config("jaxtype").get("check_devices", True)
    
    def analyze(self) -> AnalysisReport:
        """Analyze the target specified in the configuration.
        
        Returns:
            An AnalysisReport object containing the analysis results.
        """
        target_path = self.config.target_path
        
        if not target_path:
            report = AnalysisReport(self.name)
            report.add_error(
                file_path="",
                line=0,
                message="No target path specified",
                code="jaxtype-001"
            )
            return report
        
        if os.path.isfile(target_path):
            return self._analyze_file(target_path)
        elif os.path.isdir(target_path):
            return self._analyze_directory(target_path)
        else:
            report = AnalysisReport(self.name)
            report.add_error(
                file_path=target_path,
                line=0,
                message=f"Target path does not exist: {target_path}",
                code="jaxtype-002"
            )
            return report
    
    def _analyze_file(self, file_path: str) -> AnalysisReport:
        """Analyze a single file.
        
        Args:
            file_path: Path to the file to analyze.
            
        Returns:
            An AnalysisReport object containing the analysis results.
        """
        if not file_path.endswith(".py"):
            report = AnalysisReport(self.name)
            report.add_warning(
                file_path=file_path,
                line=0,
                message=f"Skipping non-Python file: {file_path}",
                code="jaxtype-003"
            )
            return report
        
        findings = self._analyze_file_with_jaxtype(file_path)
        
        report = AnalysisReport(self.name)
        
        for finding in findings:
            severity = finding.get("severity", "info")
            message = finding.get("message", "")
            line = finding.get("line", 0)
            code = finding.get("code", "jaxtype-finding")
            
            if severity == "error":
                report.add_error(
                    file_path=file_path,
                    line=line,
                    message=message,
                    code=code
                )
            elif severity == "warning":
                report.add_warning(
                    file_path=file_path,
                    line=line,
                    message=message,
                    code=code
                )
            else:
                report.add_suggestion(
                    file_path=file_path,
                    line=line,
                    message=message,
                    code=code
                )
        
        return report
    
    def _analyze_directory(self, directory_path: str) -> AnalysisReport:
        """Analyze all Python files in a directory.
        
        Args:
            directory_path: Path to the directory to analyze.
            
        Returns:
            An AnalysisReport object containing the analysis results.
        """
        report = AnalysisReport(self.name)
        analyzed_files = 0
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    file_report = self._analyze_file(file_path)
                    
                    report.merge(file_report)
                    analyzed_files += 1
        
        report.set_analyzed_files(analyzed_files)
        return report
    
    def _analyze_file_with_jaxtype(self, file_path: str) -> List[Dict[str, Any]]:
        """Analyze a file using JaxType.
        
        Args:
            file_path: Path to the file to analyze.
            
        Returns:
            A list of findings.
        """
        try:
            try:
                import jaxtyping
            except ImportError:
                return [{
                    "line": 1,
                    "severity": "error",
                    "message": "JaxType is not installed. Please install it with 'pip install jaxtyping'."
                }]
            
            if self.verbose:
                print(f"Running JaxType analysis on {file_path}")
            
            findings = self._simulate_jaxtype_analysis(file_path)
            
            return findings
        except Exception as e:
            if self.verbose:
                print(f"Error running JaxType on {file_path}: {str(e)}")
            
            return [{
                "line": 1,
                "severity": "error",
                "message": f"Error analyzing file with JaxType: {str(e)}"
            }]
    
    def _simulate_jaxtype_analysis(self, file_path: str) -> List[Dict[str, Any]]:
        """Simulate JaxType analysis on a file.
        
        This is a simplified version of what JaxType would do. In a real implementation,
        we would use JaxType's Python API to analyze the file.
        
        Args:
            file_path: Path to the file to analyze.
            
        Returns:
            A list of findings.
        """
        findings = []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            has_jaxtype_import = False
            has_tensortype_import = False
            
            for i, line in enumerate(lines):
                if "import jaxtyping" in line or "from jaxtyping import" in line:
                    has_jaxtype_import = True
                    if "Array" in line or "Float" in line or "Int" in line:
                        has_tensortype_import = True
            
            for i, line in enumerate(lines):
                line_num = i + 1
                
                if "torch.Tensor" in line and ":" in line and "->" not in line:
                    if re.search(r"(\w+)\s*:\s*torch\.Tensor", line):
                        findings.append({
                            "line": line_num,
                            "severity": "warning",
                            "message": "Missing tensor type annotation, use jaxtyping.Array instead of torch.Tensor"
                        })
                
                if "->" in line and "torch.Tensor" in line and "def " in line:
                    findings.append({
                        "line": line_num,
                        "severity": "info",
                        "message": "Consider using jaxtyping.Array for return type annotation"
                    })
                
                if "torch." in line and any(op in line for op in ["reshape", "view", "permute", "transpose"]):
                    has_shape_check = False
                    for j in range(max(0, i-5), i):
                        if ".shape" in lines[j] or "assert" in lines[j]:
                            has_shape_check = True
                            break
                    
                    if not has_shape_check:
                        findings.append({
                            "line": line_num,
                            "severity": "info",
                            "message": "Tensor operation without shape check, consider adding JaxType annotations"
                        })
                
                if "torch." in line and any(dtype in line for dtype in ["float", "int", "long", "bool", "double"]):
                    has_dtype_check = False
                    for j in range(max(0, i-5), i):
                        if ".dtype" in lines[j] or "assert" in lines[j]:
                            has_dtype_check = True
                            break
                    
                    if not has_dtype_check:
                        findings.append({
                            "line": line_num,
                            "severity": "info",
                            "message": "Tensor dtype operation without dtype check, consider adding JaxType annotations"
                        })
            
            if not has_jaxtype_import and len(findings) > 0:
                findings.append({
                    "line": 1,
                    "severity": "info",
                    "message": "Consider importing jaxtyping for better tensor type annotations"
                })
            
            if has_jaxtype_import and not has_tensortype_import and len(findings) > 0:
                for i, line in enumerate(lines):
                    if "import jaxtyping" in line or "from jaxtyping import" in line:
                        findings.append({
                            "line": i + 1,
                            "severity": "info",
                            "message": "Consider importing Array, Float, Int from jaxtyping"
                        })
                        break
            
            return findings
        except Exception as e:
            if self.verbose:
                print(f"Error simulating JaxType analysis on {file_path}: {str(e)}")
            
            return [{
                "line": 1,
                "severity": "error",
                "message": f"Error during JaxType analysis: {str(e)}"
            }]


def run_jaxtype_analysis(
    files: List[str],
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run JaxType analysis on the specified files.
    
    Args:
        files: List of file paths to analyze.
        config: Configuration for the analyzer.
        verbose: Whether to enable verbose output.
        
    Returns:
        A dictionary with analysis results.
    """
    from ml_static_analysis.core.config import AnalysisConfig
    
    config_obj = AnalysisConfig(verbose=verbose)
    if config:
        config_obj.set_analyzer_config("jaxtype", config)
    
    analyzer = JaxTypeAnalyzer(config_obj)
    
    results = {}
    for file in files:
        report = analyzer._analyze_file(file)
        results[file] = report.to_dict()
    
    return results
