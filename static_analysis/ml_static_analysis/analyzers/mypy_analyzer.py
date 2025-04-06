"""MyPy analyzer for ML static analysis framework."""

import os
import re
import subprocess
import tempfile
from typing import Dict, List, Optional, Any, Union

from ml_static_analysis.core.analyzer import BaseAnalyzer


class MyPyAnalyzer(BaseAnalyzer):
    """MyPy analyzer for ML static analysis framework.
    
    This analyzer runs MyPy type checking on Python files and reports
    type-related issues.
    """
    
    def __init__(self, config):
        """Initialize the MyPy analyzer.
        
        Args:
            config: Configuration for the analyzer.
        """
        super().__init__(config)
        
        self.name = "MyPyAnalyzer"
        mypy_config = self.config.get_analyzer_config("mypy")
        self.strict = mypy_config.get("strict", False)
        self.ignore_missing_imports = mypy_config.get("ignore_missing_imports", True)
        self.python_version = mypy_config.get("python_version", "3.8")
        self.disallow_untyped_defs = mypy_config.get("disallow_untyped_defs", False)
        self.disallow_incomplete_defs = mypy_config.get("disallow_incomplete_defs", False)
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single file using MyPy.
        
        Args:
            file_path: Path to the file to analyze.
            
        Returns:
            A dictionary with analysis results.
        """
        return self.analyze_files([file_path])
    
    def analyze_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Analyze multiple files using MyPy.
        
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
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
            config_path = f.name
            f.write("[mypy]\n")
            
            if self.strict:
                f.write("strict = True\n")
            
            if self.ignore_missing_imports:
                f.write("ignore_missing_imports = True\n")
            
            f.write(f"python_version = {self.python_version}\n")
            
            if self.disallow_untyped_defs:
                f.write("disallow_untyped_defs = True\n")
            
            if self.disallow_incomplete_defs:
                f.write("disallow_incomplete_defs = True\n")
        
        try:
            cmd = ["mypy", "--config-file", config_path]
            cmd.extend(file_paths)
            
            if self.verbose:
                print(f"Running MyPy: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0 and not stdout:
                if self.verbose:
                    print(f"MyPy failed with error: {stderr}")
                
                if "No module named mypy" in stderr:
                    return {
                        "success": False,
                        "error": "MyPy is not installed. Please install it with 'pip install mypy'.",
                    }
                
                return {
                    "success": False,
                    "error": f"MyPy failed with error: {stderr}",
                }
            
            findings = self._parse_mypy_output(stdout)
            
            summary = self._generate_summary(findings)
            
            return {
                "success": True,
                "summary": summary,
                "findings": findings,
            }
        finally:
            os.unlink(config_path)
    
    def _parse_mypy_output(self, output: str) -> Dict[str, List[Dict[str, Any]]]:
        """Parse the output of MyPy.
        
        Args:
            output: Output of MyPy.
            
        Returns:
            A dictionary mapping file paths to lists of findings.
        """
        findings = {}
        
        pattern = r"^(.+?):(\d+): (\w+): (.+)$"
        
        for line in output.splitlines():
            match = re.match(pattern, line)
            
            if not match:
                continue
            
            file_path, line_num, severity, message = match.groups()
            
            line_num = int(line_num)
            
            content = ""
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    if 0 < line_num <= len(lines):
                        content = lines[line_num - 1].strip()
            except Exception as e:
                if self.verbose:
                    print(f"Error reading file {file_path}: {str(e)}")
            
            finding = {
                "line": line_num,
                "severity": "error" if severity == "error" else "warning",
                "message": message,
                "content": content,
            }
            
            if file_path not in findings:
                findings[file_path] = []
            
            findings[file_path].append(finding)
        
        return findings
    
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
                message = finding.get("message", "")
                
                category = self._categorize_finding(message)
                
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
    
    def _categorize_finding(self, message: str) -> str:
        """Categorize a finding based on its message.
        
        Args:
            message: The error message from MyPy.
            
        Returns:
            A category for the finding.
        """
        if "missing type annotation" in message.lower() or "has no annotation" in message.lower():
            return "missing_type_annotation"
        elif "incompatible type" in message.lower():
            return "incompatible_type"
        elif "undefined" in message.lower() or "not defined" in message.lower():
            return "undefined_name"
        elif "unused" in message.lower():
            return "unused_code"
        elif "import" in message.lower():
            return "import_error"
        elif "attribute" in message.lower():
            return "attribute_error"
        elif "call" in message.lower():
            return "call_error"
        else:
            return "other"
            
    def analyze(self):
        """Analyze the target specified in the configuration.
        
        Returns:
            An AnalysisReport object containing the analysis results.
        """
        from ml_static_analysis.core.report import AnalysisReport
        
        target_path = self.config.target_path
        
        if not target_path:
            report = AnalysisReport(self.name)
            report.add_error(
                file_path="",
                line=0,
                message="No target path specified",
                code="mypy-001"
            )
            return report
        
        if os.path.isfile(target_path):
            result = self.analyze_file(target_path)
        elif os.path.isdir(target_path):
            result = self.analyze_files([target_path])
        else:
            report = AnalysisReport(self.name)
            report.add_error(
                file_path=target_path,
                line=0,
                message=f"Target path does not exist: {target_path}",
                code="mypy-002"
            )
            return report
        
        report = AnalysisReport(self.name)
        
        if not result.get("success", False):
            error_message = result.get("error", "Unknown error")
            report.set_failure(error_message)
            return report
        
        findings = result.get("findings", {})
        
        for file_path, file_findings in findings.items():
            for finding in file_findings:
                severity = finding.get("severity", "info")
                message = finding.get("message", "")
                line = finding.get("line", 0)
                code = finding.get("code", "mypy-finding")
                
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


def run_mypy_analysis(
    files: List[str],
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run MyPy analysis on the specified files.
    
    Args:
        files: List of file paths to analyze.
        config: Configuration for the analyzer.
        verbose: Whether to enable verbose output.
        
    Returns:
        A dictionary with analysis results.
    """
    analyzer = MyPyAnalyzer(config=config, verbose=verbose)
    return analyzer.analyze_files(files)
