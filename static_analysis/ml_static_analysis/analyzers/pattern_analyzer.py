"""Pattern analyzer for ML static analysis framework."""

import os
import re
from typing import Dict, List, Optional, Any, Union

from ml_static_analysis.core.analyzer import BaseAnalyzer
from ml_static_analysis.core.report import AnalysisReport
from ml_static_analysis.core.severity import Severity


class PatternAnalyzer(BaseAnalyzer):
    """Pattern analyzer for ML static analysis framework.
    
    This analyzer uses pattern matching to identify common issues in ML code.
    """
    
    def __init__(self, config):
        """Initialize the Pattern analyzer.
        
        Args:
            config: Analysis configuration.
        """
        super().__init__(config)
        
        self.severity_threshold = "info"
        self.check_thread_safety = True
        self.check_error_handling = True
        self.check_performance = True
        self.check_tensor_operations = True
        self.check_weight_switching = True
        self.check_distributed_training = True
        
        self.custom_patterns = {}
        if hasattr(config, "pattern_patterns"):
            self.custom_patterns = {"custom": config.pattern_patterns}
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single file using pattern matching.
        
        Args:
            file_path: Path to the file to analyze.
            
        Returns:
            A dictionary with analysis results.
        """
        return self.analyze_files([file_path])
    
    def analyze_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Analyze multiple files using pattern matching.
        
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
        
        findings = {}
        
        for file_path in file_paths:
            if not os.path.exists(file_path) or not file_path.endswith(".py"):
                continue
            
            file_findings = self._analyze_file_with_patterns(file_path)
            
            if file_findings:
                findings[file_path] = file_findings
        
        summary = self._generate_summary(findings)
        
        return {
            "success": True,
            "summary": summary,
            "findings": findings,
        }
    
    def _analyze_file_with_patterns(self, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze a file using pattern matching.
        
        Args:
            file_path: Path to the file to analyze.
            
        Returns:
            A dictionary mapping categories to lists of findings.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.splitlines()
            
            findings = {
                "thread_safety": [],
                "error_handling": [],
                "performance": [],
                "tensor_operations": [],
                "weight_switching": [],
                "distributed_training": [],
            }
            
            if self.check_thread_safety:
                findings["thread_safety"] = self._check_thread_safety(lines, file_path)
            
            if self.check_error_handling:
                findings["error_handling"] = self._check_error_handling(lines, file_path)
            
            if self.check_performance:
                findings["performance"] = self._check_performance(lines, file_path)
            
            if self.check_tensor_operations:
                findings["tensor_operations"] = self._check_tensor_operations(lines, file_path)
            
            if self.check_weight_switching:
                findings["weight_switching"] = self._check_weight_switching(lines, file_path)
            
            if self.check_distributed_training:
                findings["distributed_training"] = self._check_distributed_training(lines, file_path)
            
            for category, patterns in self.custom_patterns.items():
                if category not in findings:
                    findings[category] = []
                
                for pattern in patterns:
                    pattern_regex = pattern.get("regex", "")
                    message = pattern.get("message", "Custom pattern match")
                    severity = pattern.get("severity", "info")
                    
                    for i, line in enumerate(lines):
                        if re.search(pattern_regex, line):
                            findings[category].append({
                                "line": i + 1,
                                "severity": severity,
                                "message": message,
                                "content": line.strip(),
                            })
            
            findings = {k: v for k, v in findings.items() if v}
            
            return findings
        except Exception as e:
            if self.verbose:
                print(f"Error analyzing file {file_path}: {str(e)}")
            
            return {}
    
    def _check_thread_safety(self, lines: List[str], file_path: str) -> List[Dict[str, Any]]:
        """Check for thread safety issues.
        
        Args:
            lines: Lines of the file.
            file_path: Path to the file.
            
        Returns:
            A list of findings.
        """
        findings = []
        
        patterns = [
            {
                "regex": r"torch\.no_grad\(\)",
                "not_regex": r"with\s+torch\.no_grad\(\)",
                "message": "torch.no_grad() should be used with a context manager (with statement)",
                "severity": "warning",
            },
            {
                "regex": r"param\.copy_\(",
                "not_regex": r"with\s+torch\.no_grad\(\)",
                "message": "Parameter copy operation should be performed within torch.no_grad() context",
                "severity": "warning",
            },
            {
                "regex": r"torch\.cuda\.synchronize\(\)",
                "not_regex": r"try\s*:",
                "message": "CUDA synchronization should be wrapped in a try-except block",
                "severity": "info",
            },
            {
                "regex": r"\.data\b",
                "message": "Using .data can lead to race conditions, use .detach() instead",
                "severity": "warning",
            },
            {
                "regex": r"torch\._C\._cuda_setDevice",
                "message": "Direct CUDA device setting can lead to race conditions, use torch.cuda.set_device() instead",
                "severity": "warning",
            },
        ]
        
        for i, line in enumerate(lines):
            for pattern in patterns:
                if re.search(pattern["regex"], line):
                    if "not_regex" in pattern:
                        if re.search(pattern["not_regex"], line):
                            continue
                        
                        found_not_regex = False
                        for j in range(max(0, i-5), i):
                            if re.search(pattern["not_regex"], lines[j]):
                                found_not_regex = True
                                break
                        
                        if found_not_regex:
                            continue
                    
                    findings.append({
                        "line": i + 1,
                        "severity": pattern["severity"],
                        "message": pattern["message"],
                        "content": line.strip(),
                    })
        
        return findings
    
    def _check_error_handling(self, lines: List[str], file_path: str) -> List[Dict[str, Any]]:
        """Check for error handling issues.
        
        Args:
            lines: Lines of the file.
            file_path: Path to the file.
            
        Returns:
            A list of findings.
        """
        findings = []
        
        patterns = [
            {
                "regex": r"except\s*:",
                "message": "Bare except clause should be avoided, use specific exception types",
                "severity": "warning",
            },
            {
                "regex": r"except\s+\w+",
                "not_regex": r"finally\s*:",
                "message": "Exception handling without cleanup in finally block",
                "severity": "info",
                "check_next_lines": 10,
            },
            {
                "regex": r"raise\s+\w+",
                "not_regex": r"raise\s+\w+\(['\"][^'\"]+",
                "message": "Exception raised without detailed error message",
                "severity": "info",
            },
            {
                "regex": r"assert\s+",
                "message": "Assert statements are removed in optimized mode, consider using proper error handling",
                "severity": "info",
            },
        ]
        
        for i, line in enumerate(lines):
            for pattern in patterns:
                if re.search(pattern["regex"], line):
                    if "not_regex" in pattern:
                        if re.search(pattern["not_regex"], line):
                            continue
                        
                        if "check_next_lines" in pattern:
                            found_not_regex = False
                            for j in range(i+1, min(i+pattern["check_next_lines"], len(lines))):
                                if re.search(pattern["not_regex"], lines[j]):
                                    found_not_regex = True
                                    break
                            
                            if found_not_regex:
                                continue
                    
                    findings.append({
                        "line": i + 1,
                        "severity": pattern["severity"],
                        "message": pattern["message"],
                        "content": line.strip(),
                    })
        
        return findings
    
    def _check_performance(self, lines: List[str], file_path: str) -> List[Dict[str, Any]]:
        """Check for performance issues.
        
        Args:
            lines: Lines of the file.
            file_path: Path to the file.
            
        Returns:
            A list of findings.
        """
        findings = []
        
        patterns = [
            {
                "regex": r"time\.time\(\)",
                "message": "Consider using time.perf_counter() for more precise timing",
                "severity": "info",
            },
            {
                "regex": r"torch\.cuda\.empty_cache\(\)",
                "not_regex": r"if\s+",
                "message": "torch.cuda.empty_cache() should be conditionally used to avoid unnecessary overhead",
                "severity": "info",
                "check_prev_lines": 1,
            },
            {
                "regex": r"torch\.cuda\.max_memory_allocated\(\)",
                "not_regex": r"torch\.cuda\.reset_peak_memory_stats\(\)",
                "message": "Memory tracking should be preceded by reset_peak_memory_stats() for accurate measurement",
                "severity": "info",
                "check_prev_lines": 10,
            },
            {
                "regex": r"for\s+\w+\s+in\s+range\(len\(",
                "message": "Use enumerate() instead of range(len()) for better performance and readability",
                "severity": "info",
            },
            {
                "regex": r"\.to\(device\)",
                "message": "Consider batching .to(device) operations for better performance",
                "severity": "info",
            },
        ]
        
        for i, line in enumerate(lines):
            for pattern in patterns:
                if re.search(pattern["regex"], line):
                    if "not_regex" in pattern:
                        if re.search(pattern["not_regex"], line):
                            continue
                        
                        if "check_prev_lines" in pattern:
                            found_not_regex = False
                            for j in range(max(0, i-pattern["check_prev_lines"]), i):
                                if re.search(pattern["not_regex"], lines[j]):
                                    found_not_regex = True
                                    break
                            
                            if found_not_regex:
                                continue
                    
                    findings.append({
                        "line": i + 1,
                        "severity": pattern["severity"],
                        "message": pattern["message"],
                        "content": line.strip(),
                    })
        
        return findings
    
    def _check_tensor_operations(self, lines: List[str], file_path: str) -> List[Dict[str, Any]]:
        """Check for tensor operation issues.
        
        Args:
            lines: Lines of the file.
            file_path: Path to the file.
            
        Returns:
            A list of findings.
        """
        findings = []
        
        patterns = [
            {
                "regex": r"\.(reshape|view|permute|transpose)\(",
                "not_regex": r"\.shape",
                "message": "Tensor reshape/view without shape check",
                "severity": "warning",
                "check_prev_lines": 5,
            },
            {
                "regex": r"\.to\(.*dtype",
                "not_regex": r"\.dtype",
                "message": "Tensor dtype conversion without dtype check",
                "severity": "info",
                "check_prev_lines": 5,
            },
            {
                "regex": r"\.to\(.*device",
                "not_regex": r"\.device",
                "message": "Tensor device transfer without device check",
                "severity": "info",
                "check_prev_lines": 5,
            },
            {
                "regex": r"\.(matmul|bmm|mm)\(",
                "message": "Potential tensor dimension mismatch in matrix multiplication",
                "severity": "info",
            },
            {
                "regex": r"torch\.(cat|stack)\(",
                "not_regex": r"dim\s*=",
                "message": "torch.cat/stack without explicit dimension parameter",
                "severity": "info",
            },
        ]
        
        for i, line in enumerate(lines):
            for pattern in patterns:
                if re.search(pattern["regex"], line):
                    if "not_regex" in pattern:
                        if re.search(pattern["not_regex"], line):
                            continue
                        
                        if "check_prev_lines" in pattern:
                            found_not_regex = False
                            for j in range(max(0, i-pattern["check_prev_lines"]), i):
                                if re.search(pattern["not_regex"], lines[j]):
                                    found_not_regex = True
                                    break
                            
                            if found_not_regex:
                                continue
                    
                    findings.append({
                        "line": i + 1,
                        "severity": pattern["severity"],
                        "message": pattern["message"],
                        "content": line.strip(),
                    })
        
        return findings
    
    def _check_weight_switching(self, lines: List[str], file_path: str) -> List[Dict[str, Any]]:
        """Check for weight switching issues.
        
        Args:
            lines: Lines of the file.
            file_path: Path to the file.
            
        Returns:
            A list of findings.
        """
        findings = []
        
        patterns = [
            {
                "regex": r"update_model_weights",
                "not_regex": r"try\s*:",
                "message": "Model weight updates should be wrapped in a try-except block",
                "severity": "warning",
                "check_prev_lines": 5,
            },
            {
                "regex": r"load_state_dict",
                "not_regex": r"\.shape",
                "message": "State dict loading should be preceded by shape validation",
                "severity": "warning",
                "check_prev_lines": 5,
            },
            {
                "regex": r"param\.copy_\(",
                "not_regex": r"assert",
                "message": "Parameter copy operation should be preceded by shape assertion",
                "severity": "warning",
                "check_prev_lines": 5,
            },
            {
                "regex": r"load_checkpoint",
                "not_regex": r"try\s*:",
                "message": "Checkpoint loading should be wrapped in a try-except block",
                "severity": "warning",
                "check_prev_lines": 5,
            },
            {
                "regex": r"model\.load_state_dict\(.*strict\s*=\s*False",
                "message": "Using strict=False with load_state_dict can hide weight loading issues",
                "severity": "info",
            },
        ]
        
        for i, line in enumerate(lines):
            for pattern in patterns:
                if re.search(pattern["regex"], line):
                    if "not_regex" in pattern:
                        if re.search(pattern["not_regex"], line):
                            continue
                        
                        if "check_prev_lines" in pattern:
                            found_not_regex = False
                            for j in range(max(0, i-pattern["check_prev_lines"]), i):
                                if re.search(pattern["not_regex"], lines[j]):
                                    found_not_regex = True
                                    break
                            
                            if found_not_regex:
                                continue
                    
                    findings.append({
                        "line": i + 1,
                        "severity": pattern["severity"],
                        "message": pattern["message"],
                        "content": line.strip(),
                    })
        
        return findings
    
    def _check_distributed_training(self, lines: List[str], file_path: str) -> List[Dict[str, Any]]:
        """Check for distributed training issues.
        
        Args:
            lines: Lines of the file.
            file_path: Path to the file.
            
        Returns:
            A list of findings.
        """
        findings = []
        
        patterns = [
            {
                "regex": r"torch\.distributed\.all_reduce\(",
                "not_regex": r"group\s*=",
                "message": "all_reduce without explicit process group parameter",
                "severity": "info",
            },
            {
                "regex": r"torch\.distributed\.broadcast\(",
                "not_regex": r"src\s*=",
                "message": "broadcast without explicit source rank parameter",
                "severity": "warning",
            },
            {
                "regex": r"torch\.distributed\.barrier\(",
                "not_regex": r"group\s*=",
                "message": "barrier without explicit process group parameter",
                "severity": "info",
            },
            {
                "regex": r"torch\.cuda\.set_device\(",
                "not_regex": r"local_rank",
                "message": "set_device should use local_rank in distributed training",
                "severity": "warning",
            },
            {
                "regex": r"torch\.distributed\.init_process_group\(",
                "not_regex": r"timeout\s*=",
                "message": "init_process_group without explicit timeout parameter",
                "severity": "info",
            },
        ]
        
        for i, line in enumerate(lines):
            for pattern in patterns:
                if re.search(pattern["regex"], line):
                    if "not_regex" in pattern:
                        if re.search(pattern["not_regex"], line):
                            continue
                    
                    findings.append({
                        "line": i + 1,
                        "severity": pattern["severity"],
                        "message": pattern["message"],
                        "content": line.strip(),
                    })
        
        return findings
    
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
            report.add_error(results.get("error", "Unknown error during pattern analysis."))
            return report
        
        for file_path, file_findings in results.get("findings", {}).items():
            for category, category_findings in file_findings.items():
                for finding in category_findings:
                    severity_str = finding.get("severity", "info")
                    severity = Severity.from_str(severity_str)
                    
                    message = finding.get("message", "")
                    line = finding.get("line", 0)
                    
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


def run_pattern_analysis(
    files: List[str],
    config=None,
) -> Dict[str, Any]:
    """Run pattern analysis on the specified files.
    
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
    
    analyzer = PatternAnalyzer(config)
    return analyzer.analyze_files(files)
