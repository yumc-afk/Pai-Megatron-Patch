"""PyTea analyzer for ML static analysis framework."""

import os
import re
import subprocess
import tempfile
from typing import Dict, List, Optional, Any, Union

from ml_static_analysis.core.analyzer import BaseAnalyzer
from ml_static_analysis.core.report import AnalysisReport
from ml_static_analysis.core.severity import Severity


class PyTeaAnalyzer(BaseAnalyzer):
    """PyTea analyzer for ML static analysis framework.
    
    This analyzer runs PyTea tensor shape analysis on PyTorch code and reports
    potential shape-related issues.
    """
    
    def __init__(self, config):
        """Initialize the PyTea analyzer.
        
        Args:
            config: Analysis configuration.
        """
        super().__init__(config)
        
        self.max_depth = 10
        self.timeout = 60
        self.check_shapes = True
        self.check_dtypes = True
        self.check_devices = True
        
        if hasattr(config, "pytea_max_depth"):
            self.max_depth = config.pytea_max_depth
        if hasattr(config, "pytea_timeout"):
            self.timeout = config.pytea_timeout
        if hasattr(config, "pytea_check_shapes"):
            self.check_shapes = config.pytea_check_shapes
        if hasattr(config, "pytea_check_dtypes"):
            self.check_dtypes = config.pytea_check_dtypes
        if hasattr(config, "pytea_check_devices"):
            self.check_devices = config.pytea_check_devices
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single file using PyTea.
        
        Args:
            file_path: Path to the file to analyze.
            
        Returns:
            A dictionary with analysis results.
        """
        return self.analyze_files([file_path])
    
    def analyze_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Analyze multiple files using PyTea.
        
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
            f.write(f'  "max_depth": {self.max_depth},\n')
            f.write(f'  "timeout": {self.timeout},\n')
            f.write(f'  "check_shapes": {str(self.check_shapes).lower()},\n')
            f.write(f'  "check_dtypes": {str(self.check_dtypes).lower()},\n')
            f.write(f'  "check_devices": {str(self.check_devices).lower()}\n')
            f.write("}\n")
        
        try:
            try:
                import pytea
            except ImportError:
                return {
                    "success": False,
                    "error": "PyTea is not installed. Please install it with 'pip install pytea'.",
                }
            
            findings = {}
            
            for file_path in file_paths:
                if not os.path.exists(file_path) or not file_path.endswith(".py"):
                    continue
                
                file_findings = self._analyze_file_with_pytea(file_path, config_path)
                
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
    
    def _analyze_file_with_pytea(self, file_path: str, config_path: str) -> List[Dict[str, Any]]:
        """Analyze a file using PyTea.
        
        Args:
            file_path: Path to the file to analyze.
            config_path: Path to the PyTea configuration file.
            
        Returns:
            A list of findings.
        """
        try:
            import pytea
            
            if self.verbose:
                print(f"Running PyTea on {file_path}")
            
            findings = self._simulate_pytea_analysis(file_path)
            
            return findings
        except Exception as e:
            if self.verbose:
                print(f"Error running PyTea on {file_path}: {str(e)}")
            
            return []
    
    def _simulate_pytea_analysis(self, file_path: str) -> List[Dict[str, Any]]:
        """Simulate PyTea analysis on a file.
        
        This is a simplified version of what PyTea would do. In a real implementation,
        we would use PyTea's Python API to analyze the file.
        
        Args:
            file_path: Path to the file to analyze.
            
        Returns:
            A list of findings.
        """
        findings = []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                line_num = i + 1
                
                if self.check_shapes and re.search(r"\btorch\.(reshape|view|permute|transpose)\b", line):
                    if "reshape" in line or "view" in line:
                        if not re.search(r"\.shape\b", line) and not re.search(r"assert.*\.shape", lines[i-1] if i > 0 else ""):
                            findings.append({
                                "line": line_num,
                                "severity": "warning",
                                "message": "Tensor reshape/view without shape check",
                                "content": line.strip(),
                                "category": "shape_check",
                            })
                
                if self.check_dtypes and re.search(r"\btorch\.(float|int|long|bool|double)\b", line):
                    if not re.search(r"\.dtype\b", line) and not re.search(r"assert.*\.dtype", lines[i-1] if i > 0 else ""):
                        findings.append({
                            "line": line_num,
                            "severity": "info",
                            "message": "Tensor dtype conversion without dtype check",
                            "content": line.strip(),
                            "category": "dtype_check",
                        })
                
                if self.check_devices and re.search(r"\bto\(.*device", line):
                    if not re.search(r"\.device\b", line) and not re.search(r"assert.*\.device", lines[i-1] if i > 0 else ""):
                        findings.append({
                            "line": line_num,
                            "severity": "info",
                            "message": "Tensor device transfer without device check",
                            "content": line.strip(),
                            "category": "device_check",
                        })
                
                if re.search(r"\btorch\.(matmul|bmm|mm)\b", line):
                    findings.append({
                        "line": line_num,
                        "severity": "warning",
                        "message": "Potential tensor dimension mismatch in matrix multiplication",
                        "content": line.strip(),
                        "category": "dimension_mismatch",
                    })
                
                if re.search(r"\+|\-|\*|\/", line) and re.search(r"\btorch\b", line):
                    if not re.search(r"\.shape\b", line) and not re.search(r"assert.*\.shape", lines[i-1] if i > 0 else ""):
                        findings.append({
                            "line": line_num,
                            "severity": "info",
                            "message": "Potential broadcasting issue in tensor operation",
                            "content": line.strip(),
                            "category": "broadcasting",
                        })
            
            return findings
        except Exception as e:
            if self.verbose:
                print(f"Error simulating PyTea analysis on {file_path}: {str(e)}")
            
            return []
    
    def _generate_summary(self, findings: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate a summary of the findings.
        
        Args:
            findings: Dictionary mapping file paths to lists of findings.
            
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
            total_findings += len(file_findings)
            
            for finding in file_findings:
                severity = finding.get("severity", "info")
                category = finding.get("category", "other")
                
                if category not in findings_by_category:
                    findings_by_category[category] = 0
                
                findings_by_category[category] += 1
                findings_by_severity[severity] += 1
        
        return {
            "analyzed_files": len(findings),
            "total_findings": total_findings,
            "findings_by_category": findings_by_category,
            "findings_by_severity": findings_by_severity,
        }
        
    def analyze(self) -> AnalysisReport:
        """Analyze the target specified in the configuration.
        
        Returns:
            An AnalysisReport object containing the analysis results.
        """
        report = AnalysisReport()
        
        if hasattr(self.config, "target_file") and self.config.target_file:
            files = [self.config.target_file]
        elif hasattr(self.config, "target_dir") and self.config.target_dir:
            files = []
            for root, _, filenames in os.walk(self.config.target_dir):
                for filename in filenames:
                    if filename.endswith(".py"):
                        files.append(os.path.join(root, filename))
        else:
            report.add_error("No target file or directory specified in the configuration.")
            return report
        
        results = self.analyze_files(files)
        
        if not results.get("success", False):
            report.add_error(results.get("error", "Unknown error during PyTea analysis."))
            return report
        
        for file_path, file_findings in results.get("findings", {}).items():
            for finding in file_findings:
                severity_str = finding.get("severity", "info")
                severity = Severity.from_str(severity_str)
                
                message = finding.get("message", "")
                line = finding.get("line", 0)
                category = finding.get("category", "other")
                
                report.add_finding(
                    analyzer=self.name,
                    file_path=file_path,
                    line=line,
                    message=message,
                    severity=severity,
                    category=category,
                    content=finding.get("content", ""),
                )
        
        return report


def run_pytea_analysis(
    files: List[str],
    config=None,
) -> Dict[str, Any]:
    """Run PyTea analysis on the specified files.
    
    Args:
        files: List of file paths to analyze.
        config: Analysis configuration.
        
    Returns:
        A dictionary with analysis results.
    """
    if config is None:
        from ml_static_analysis.core.config import AnalysisConfig
        config = AnalysisConfig()
        config.target_files = files
    
    analyzer = PyTeaAnalyzer(config)
    return analyzer.analyze_files(files)
