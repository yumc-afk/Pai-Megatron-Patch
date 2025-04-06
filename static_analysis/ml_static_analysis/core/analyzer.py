"""Base analyzer class for ML static analysis framework."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union


class BaseAnalyzer(ABC):
    """Base class for all analyzers in the ML static analysis framework.
    
    This abstract class defines the common interface that all analyzers must implement.
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize the analyzer.
        
        Args:
            verbose: Whether to enable verbose output.
        """
        self.verbose = verbose
    
    @abstractmethod
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single file.
        
        Args:
            file_path: Path to the file to analyze.
            
        Returns:
            A dictionary with analysis results.
        """
        pass
    
    @abstractmethod
    def analyze_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Analyze multiple files.
        
        Args:
            file_paths: List of paths to files to analyze.
            
        Returns:
            A dictionary with analysis results.
        """
        pass
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of the analysis results.
        
        Args:
            results: Analysis results from analyze_file or analyze_files.
        """
        if "summary" not in results:
            print("No summary available.")
            return
        
        summary = results["summary"]
        
        print(f"Analyzed {summary.get('analyzed_files', 0)} files")
        print(f"Found {summary.get('total_findings', 0)} findings")
        
        if "findings_by_category" in summary:
            print("\nFindings by category:")
            for category, count in summary["findings_by_category"].items():
                print(f"  {category}: {count}")
        
        if "findings_by_severity" in summary:
            print("\nFindings by severity:")
            for severity, count in summary["findings_by_severity"].items():
                print(f"  {severity}: {count}")
