"""Report generation for ML static analysis framework."""

import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Union


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
