"""Report generation for ML static analysis framework."""

import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass

from ml_static_analysis.core.severity import Severity


@dataclass
class Finding:
    """A finding from an analyzer."""
    
    file_path: str
    line: int
    message: str
    code: Optional[str] = None
    severity: Severity = Severity.INFO
    content: Optional[str] = None
    category: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the finding to a dictionary.
        
        Returns:
            A dictionary representation of the finding.
        """
        return {
            "file_path": self.file_path,
            "line": self.line,
            "message": self.message,
            "code": self.code,
            "severity": str(self.severity),
            "content": self.content,
            "category": self.category
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Finding':
        """Create a finding from a dictionary.
        
        Args:
            data: Dictionary representation of a finding.
            
        Returns:
            A Finding instance.
        """
        severity_str = data.get("severity", "info")
        try:
            severity = Severity.from_str(severity_str)
        except ValueError:
            severity = Severity.INFO
        
        return cls(
            file_path=data.get("file_path", ""),
            line=data.get("line", 0),
            message=data.get("message", ""),
            code=data.get("code"),
            severity=severity,
            content=data.get("content"),
            category=data.get("category")
        )


class AnalysisReport:
    """Analysis report for ML static analysis framework.
    
    This class stores the results of an analysis run, including errors,
    warnings, suggestions, and other findings.
    """
    
    def __init__(self, analyzer_name: str):
        """Initialize the analysis report.
        
        Args:
            analyzer_name: Name of the analyzer that generated this report.
        """
        self.analyzer_name = analyzer_name
        self.errors = []
        self.warnings = []
        self.suggestions = []
        self.findings = {}
        self.summary = {
            "analyzed_files": 0,
            "total_findings": 0,
            "findings_by_category": {},
            "findings_by_severity": {
                "error": 0,
                "warning": 0,
                "info": 0
            }
        }
        self.success = True
        self.error_message = None
    
    def add_error(self, file_path: str, line: int, message: str, code: Optional[str] = None) -> None:
        """Add an error to the report.
        
        Args:
            file_path: Path to the file where the error was found.
            line: Line number where the error was found.
            message: Error message.
            code: Error code or identifier.
        """
        error = {
            "file_path": file_path,
            "line": line,
            "message": message,
            "code": code,
            "severity": "error"
        }
        self.errors.append(error)
        self._add_finding(file_path, error)
        self.summary["findings_by_severity"]["error"] += 1
        self.summary["total_findings"] += 1
    
    def add_warning(self, file_path: str, line: int, message: str, code: Optional[str] = None) -> None:
        """Add a warning to the report.
        
        Args:
            file_path: Path to the file where the warning was found.
            line: Line number where the warning was found.
            message: Warning message.
            code: Warning code or identifier.
        """
        warning = {
            "file_path": file_path,
            "line": line,
            "message": message,
            "code": code,
            "severity": "warning"
        }
        self.warnings.append(warning)
        self._add_finding(file_path, warning)
        self.summary["findings_by_severity"]["warning"] += 1
        self.summary["total_findings"] += 1
    
    def add_suggestion(self, file_path: str, line: int, message: str, code: Optional[str] = None) -> None:
        """Add a suggestion to the report.
        
        Args:
            file_path: Path to the file where the suggestion was found.
            line: Line number where the suggestion was found.
            message: Suggestion message.
            code: Suggestion code or identifier.
        """
        suggestion = {
            "file_path": file_path,
            "line": line,
            "message": message,
            "code": code,
            "severity": "info"
        }
        self.suggestions.append(suggestion)
        self._add_finding(file_path, suggestion)
        self.summary["findings_by_severity"]["info"] += 1
        self.summary["total_findings"] += 1
    
    def add_finding(self, file_path: str, category: str, finding: Dict[str, Any]) -> None:
        """Add a finding to the report.
        
        Args:
            file_path: Path to the file where the finding was found.
            category: Category of the finding.
            finding: Finding details.
        """
        if file_path not in self.findings:
            self.findings[file_path] = {}
        
        if category not in self.findings[file_path]:
            self.findings[file_path][category] = []
        
        self.findings[file_path][category].append(finding)
        
        if category not in self.summary["findings_by_category"]:
            self.summary["findings_by_category"][category] = 0
        
        self.summary["findings_by_category"][category] += 1
        self.summary["total_findings"] += 1
        
        severity = finding.get("severity", "info")
        self.summary["findings_by_severity"][severity] += 1
    
    def _add_finding(self, file_path: str, finding: Dict[str, Any]) -> None:
        """Add a finding to the findings dictionary.
        
        Args:
            file_path: Path to the file where the finding was found.
            finding: Finding details.
        """
        if file_path not in self.findings:
            self.findings[file_path] = []
        
        self.findings[file_path].append(finding)
    
    def set_analyzed_files(self, count: int) -> None:
        """Set the number of analyzed files.
        
        Args:
            count: Number of analyzed files.
        """
        self.summary["analyzed_files"] = count
    
    def set_failure(self, error_message: str) -> None:
        """Mark the analysis as failed.
        
        Args:
            error_message: Error message explaining the failure.
        """
        self.success = False
        self.error_message = error_message
    
    def merge(self, other: 'AnalysisReport') -> None:
        """Merge another report into this one.
        
        Args:
            other: Another AnalysisReport to merge into this one.
        """
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.suggestions.extend(other.suggestions)
        
        for file_path, file_findings in other.findings.items():
            if file_path not in self.findings:
                self.findings[file_path] = file_findings
            else:
                if isinstance(file_findings, dict) and isinstance(self.findings[file_path], dict):
                    for category, category_findings in file_findings.items():
                        if category not in self.findings[file_path]:
                            self.findings[file_path][category] = category_findings
                        else:
                            self.findings[file_path][category].extend(category_findings)
                elif isinstance(file_findings, list) and isinstance(self.findings[file_path], list):
                    self.findings[file_path].extend(file_findings)
        
        self.summary["analyzed_files"] += other.summary["analyzed_files"]
        self.summary["total_findings"] += other.summary["total_findings"]
        
        for category, count in other.summary.get("findings_by_category", {}).items():
            if category not in self.summary["findings_by_category"]:
                self.summary["findings_by_category"][category] = 0
            self.summary["findings_by_category"][category] += count
        
        for severity, count in other.summary.get("findings_by_severity", {}).items():
            if severity not in self.summary["findings_by_severity"]:
                self.summary["findings_by_severity"][severity] = 0
            self.summary["findings_by_severity"][severity] += count
        
        if not other.success:
            self.success = False
            if self.error_message is None:
                self.error_message = other.error_message
            elif other.error_message is not None:
                self.error_message += f"; {other.error_message}"
    
    def get_errors(self) -> List[Dict[str, Any]]:
        """Get the list of errors.
        
        Returns:
            A list of errors.
        """
        return self.errors
    
    def get_warnings(self) -> List[Dict[str, Any]]:
        """Get the list of warnings.
        
        Returns:
            A list of warnings.
        """
        return self.warnings
    
    def get_suggestions(self) -> List[Dict[str, Any]]:
        """Get the list of suggestions.
        
        Returns:
            A list of suggestions.
        """
        return self.suggestions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the report to a dictionary.
        
        Returns:
            A dictionary representation of the report.
        """
        return {
            "analyzer_name": self.analyzer_name,
            "success": self.success,
            "error": self.error_message,
            "summary": self.summary,
            "errors": self.errors,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "findings": self.findings
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisReport':
        """Create a report from a dictionary.
        
        Args:
            data: Dictionary representation of a report.
            
        Returns:
            An AnalysisReport instance.
        """
        report = cls(data.get("analyzer_name", "unknown"))
        report.success = data.get("success", True)
        report.error_message = data.get("error")
        report.summary = data.get("summary", {})
        report.errors = data.get("errors", [])
        report.warnings = data.get("warnings", [])
        report.suggestions = data.get("suggestions", [])
        report.findings = data.get("findings", {})
        
        return report


class ReportGenerator:
    """Report generator for ML static analysis framework.
    
    This class handles the generation of comprehensive analysis reports
    from the results of multiple analyzers.
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize the report generator.
        
        Args:
            verbose: Whether to enable verbose output.
        """
        self.verbose = verbose
    
    def generate_report(
        self,
        results: Dict[str, Dict[str, Any]],
        files: Optional[List[str]] = None,
        component: str = "default",
        project_name: str = "Project",
    ) -> str:
        """Generate a comprehensive analysis report.
        
        Args:
            results: Dictionary mapping analyzer names to their results.
            files: List of files that were analyzed.
            component: Name of the component that was analyzed.
            project_name: Name of the project.
            
        Returns:
            A string containing the generated report in Markdown format.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# {project_name} Static Analysis Report

**Generated:** {timestamp}
**Component:** {component}
**Files Analyzed:** {len(files) if files else 0}


"""
        
        for analyzer_name, analyzer_results in results.items():
            if not analyzer_results:
                continue
                
            report += f"### {analyzer_name}\n\n"
            
            if analyzer_results.get("success") is False:
                error = analyzer_results.get("error", "Unknown error")
                report += f"❌ Failed: {error}\n\n"
                continue
            
            summary = analyzer_results.get("summary", {})
            
            if "total_findings" in summary:
                report += f"- Total findings: {summary['total_findings']}\n"
            
            if "analyzed_files" in summary:
                report += f"- Files analyzed: {summary['analyzed_files']}\n"
            
            if "findings_by_category" in summary:
                report += "- Findings by category:\n"
                for category, count in summary["findings_by_category"].items():
                    report += f"  - {category.replace('_', ' ').title()}: {count}\n"
            
            if "findings_by_severity" in summary:
                severities = summary["findings_by_severity"]
                report += f"- Severity: {severities.get('error', 0)} errors, {severities.get('warning', 0)} warnings, {severities.get('info', 0)} info\n"
            
            report += "\n"
        
        report += "## Detailed Findings\n\n"
        
        for analyzer_name, analyzer_results in results.items():
            if not analyzer_results:
                continue
                
            report += f"### {analyzer_name}\n\n"
            
            if analyzer_results.get("success") is False:
                error = analyzer_results.get("error", "Unknown error")
                report += f"❌ Failed: {error}\n\n"
                continue
            
            findings = analyzer_results.get("findings", {})
            
            if not findings:
                report += f"No findings from {analyzer_name}.\n\n"
                continue
            
            for file_path, file_findings in findings.items():
                report += f"#### {os.path.basename(file_path)}\n\n"
                
                if isinstance(file_findings, dict):
                    for category, category_findings in file_findings.items():
                        report += f"**{category.replace('_', ' ').title()}**\n\n"
                        
                        for finding in category_findings:
                            self._format_finding(report, finding)
                        
                        report += "\n"
                elif isinstance(file_findings, list):
                    for finding in file_findings:
                        self._format_finding(report, finding)
                    
                    report += "\n"
        
        report += "## Recommendations\n\n"
        
        for analyzer_name, analyzer_results in results.items():
            if not analyzer_results or analyzer_results.get("success") is False:
                continue
                
            findings = analyzer_results.get("findings", {})
            
            if not findings:
                continue
            
            report += f"### {analyzer_name} Recommendations\n\n"
            
            if analyzer_name == "MyPy":
                report += "- Fix type annotations in the codebase\n"
                report += "- Add more specific type hints instead of using `Any`\n"
                report += "- Consider using more advanced typing features like Protocol and TypeVar\n\n"
            elif analyzer_name == "Pattern Analysis":
                has_thread_safety = False
                has_error_handling = False
                has_performance = False
                
                for file_findings in findings.values():
                    if "thread_safety" in file_findings:
                        has_thread_safety = True
                    if "error_handling" in file_findings:
                        has_error_handling = True
                    if "performance" in file_findings:
                        has_performance = True
                
                if has_thread_safety:
                    report += "- **Thread Safety:**\n"
                    report += "  - Ensure all `torch.no_grad()` operations are used within a context manager (`with` statement) for proper cleanup\n"
                    report += "  - Verify that parameter copy operations (`param.copy_()`) are properly synchronized\n"
                    report += "  - Consider adding more explicit synchronization around CUDA cache clearing\n\n"
                
                if has_error_handling:
                    report += "- **Error Handling:**\n"
                    report += "  - Add more specific exception types instead of catching generic exceptions\n"
                    report += "  - Ensure proper cleanup in exception handlers\n"
                    report += "  - Add more detailed error messages for debugging\n\n"
                
                if has_performance:
                    report += "- **Performance:**\n"
                    report += "  - Consider adding more granular performance tracking\n"
                    report += "  - Add memory usage tracking alongside timing information\n"
                    report += "  - Consider adding performance benchmarks for different model sizes\n\n"
            elif analyzer_name == "PyTea":
                report += "- Add shape assertions before critical tensor operations\n"
                report += "- Consider adding more explicit dimension checks\n"
                report += "- Document expected tensor shapes in function docstrings\n\n"
            elif analyzer_name == "TorchTyping":
                report += "- Add shape information to tensor type annotations\n"
                report += "- Use TorchTyping to specify tensor shapes, dimensions, and dtypes\n"
                report += "- Add runtime shape validation for critical tensor operations\n"
                report += "- Consider adding shape assertions at function boundaries\n\n"
            elif analyzer_name == "PyAssistant":
                has_thread_safety = False
                has_error_handling = False
                has_performance = False
                has_weight_switching = False
                
                for file_findings in findings.values():
                    if "thread_safety" in file_findings:
                        has_thread_safety = True
                    if "error_handling" in file_findings:
                        has_error_handling = True
                    if "performance" in file_findings:
                        has_performance = True
                    if "weight_switching" in file_findings:
                        has_weight_switching = True
                
                if has_thread_safety:
                    report += "- **Thread Safety:**\n"
                    report += "  - Ensure proper synchronization for shared resources\n"
                    report += "  - Use thread-safe data structures for concurrent access\n"
                    report += "  - Consider using atomic operations for critical updates\n\n"
                
                if has_error_handling:
                    report += "- **Error Handling:**\n"
                    report += "  - Add more specific exception types\n"
                    report += "  - Ensure proper cleanup in exception handlers\n"
                    report += "  - Improve error messages for better debugging\n\n"
                
                if has_performance:
                    report += "- **Performance:**\n"
                    report += "  - Optimize critical code paths\n"
                    report += "  - Consider batching operations where possible\n"
                    report += "  - Add performance metrics for key operations\n\n"
                
                if has_weight_switching:
                    report += "- **Weight Switching:**\n"
                    report += "  - Ensure proper validation of weight shapes\n"
                    report += "  - Add memory usage tracking during weight updates\n"
                    report += "  - Consider implementing gradual weight updates\n\n"
        
        return report
    
    def _format_finding(self, report: str, finding: Dict[str, Any]) -> None:
        """Format a single finding and append it to the report.
        
        Args:
            report: The report string to append to.
            finding: The finding to format.
        """
        severity = finding.get('severity', 'info')
        severity_marker = "ℹ️" if severity == "info" else "⚠️" if severity == "warning" else "❌"
        message = finding.get('message', '')
        line = finding.get('line', 0)
        content = finding.get('content', '')
        
        report += f"- {severity_marker} Line {line}: {message}\n"
        if content:
            report += f"  `{content}`\n"
    
    def save_report(self, report: str, output_file: Optional[str] = None) -> str:
        """Save the report to a file.
        
        Args:
            report: The report to save.
            output_file: Path to the output file. If None, a default path will be used.
            
        Returns:
            The path to the saved report.
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"analysis_report_{timestamp}.md"
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        
        if self.verbose:
            print(f"Report saved to {output_file}")
        
        return output_file
