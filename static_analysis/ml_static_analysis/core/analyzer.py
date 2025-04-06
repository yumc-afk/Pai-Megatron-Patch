"""Base analyzer class for ML static analysis framework."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Type
import os
import importlib
import logging
import inspect

from ml_static_analysis.core.config import AnalysisConfig
from ml_static_analysis.core.report import AnalysisReport

logger = logging.getLogger(__name__)


class BaseAnalyzer(ABC):
    """Base class for all analyzers in the ML static analysis framework.
    
    This abstract class defines the common interface that all analyzers must implement.
    """
    
    def __init__(self, config: AnalysisConfig):
        """Initialize the analyzer.
        
        Args:
            config: Analysis configuration.
        """
        self.config = config
        self.verbose = config.verbose
        self.name = self.__class__.__name__
    
    @abstractmethod
    def analyze(self) -> AnalysisReport:
        """Analyze the target specified in the configuration.
        
        Returns:
            An AnalysisReport object containing the analysis results.
        """
        pass
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single file.
        
        Args:
            file_path: Path to the file to analyze.
            
        Returns:
            A dictionary with analysis results.
        """
        raise NotImplementedError("This analyzer does not support analyzing a single file.")
    
    def analyze_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Analyze multiple files.
        
        Args:
            file_paths: List of paths to files to analyze.
            
        Returns:
            A dictionary with analysis results.
        """
        raise NotImplementedError("This analyzer does not support analyzing multiple files.")
    
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


class AnalyzerRegistry:
    """Registry for all analyzers in the ML static analysis framework."""
    
    def __init__(self, config: AnalysisConfig):
        """Initialize the analyzer registry.
        
        Args:
            config: Analysis configuration.
        """
        self.config = config
        self.analyzers = {}
        self._discover_analyzers()
    
    def _discover_analyzers(self) -> None:
        """Discover all available analyzers."""
        analyzers_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "analyzers")
        
        if not os.path.exists(analyzers_dir):
            logger.warning(f"Analyzers directory not found: {analyzers_dir}")
            return
        
        from ml_static_analysis.analyzers.mypy_analyzer import MyPyAnalyzer
        from ml_static_analysis.analyzers.pytea_analyzer import PyTeaAnalyzer
        from ml_static_analysis.analyzers.pattern_analyzer import PatternAnalyzer
        from ml_static_analysis.analyzers.jaxtype_analyzer import JaxTypeAnalyzer
        
        self.analyzers["MyPyAnalyzer"] = MyPyAnalyzer
        self.analyzers["PyTeaAnalyzer"] = PyTeaAnalyzer
        self.analyzers["PatternAnalyzer"] = PatternAnalyzer
        self.analyzers["JaxTypeAnalyzer"] = JaxTypeAnalyzer
        
        try:
            from ml_static_analysis.analyzers.pyassistant_analyzer import PyAssistantAnalyzer
            self.analyzers["PyAssistantAnalyzer"] = PyAssistantAnalyzer
        except ImportError:
            logger.warning("PyAssistant analyzer not found")
        
        try:
            for filename in os.listdir(analyzers_dir):
                if filename.endswith("_analyzer.py") and filename != "__init__.py":
                    module_name = filename[:-3]  # 去掉.py后缀
                    
                    if module_name in ["mypy_analyzer", "pytea_analyzer", "pattern_analyzer", 
                                      "jaxtype_analyzer", "pyassistant_analyzer"]:
                        continue
                    
                    try:
                        module = importlib.import_module(f"ml_static_analysis.analyzers.{module_name}")
                        
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and 
                                issubclass(obj, BaseAnalyzer) and 
                                obj != BaseAnalyzer):
                                
                                analyzer_name = name
                                self.analyzers[analyzer_name] = obj
                                logger.debug(f"Registered analyzer: {analyzer_name}")
                    
                    except ImportError as e:
                        logger.error(f"Failed to import analyzer module {module_name}: {e}")
                    except Exception as e:
                        logger.error(f"Error while discovering analyzer in {module_name}: {e}")
        except Exception as e:
            logger.error(f"Error while discovering analyzers: {e}")
    
    def get_analyzer(self, name: str) -> Optional[BaseAnalyzer]:
        """Get an analyzer by name.
        
        Args:
            name: Name of the analyzer.
            
        Returns:
            An instance of the analyzer, or None if not found.
        """
        if name not in self.analyzers:
            logger.error(f"Analyzer not found: {name}")
            return None
        
        try:
            return self.analyzers[name](self.config)
        except Exception as e:
            logger.error(f"Failed to instantiate analyzer {name}: {e}")
            return None
    
    def get_analyzers(self) -> List[BaseAnalyzer]:
        """Get all available analyzers.
        
        Returns:
            A list of analyzer instances.
        """
        if self.config.analyzers == "all":
            analyzer_classes = list(self.analyzers.values())
        else:
            analyzer_name_map = {
                "mypy": "MyPyAnalyzer",
                "pytea": "PyTeaAnalyzer",
                "pattern": "PatternAnalyzer",
                "jaxtype": "JaxTypeAnalyzer",
                "pyassistant": "PyAssistantAnalyzer"
            }
            
            analyzer_classes = []
            for name in self.config.analyzers:
                if name in self.analyzers:
                    analyzer_classes.append(self.analyzers[name])
                elif name in analyzer_name_map and analyzer_name_map[name] in self.analyzers:
                    analyzer_classes.append(self.analyzers[analyzer_name_map[name]])
                else:
                    logger.warning(f"Analyzer not found: {name}")
        
        analyzers = []
        for analyzer_class in analyzer_classes:
            try:
                analyzer = analyzer_class(self.config)
                analyzers.append(analyzer)
            except Exception as e:
                logger.error(f"Failed to instantiate analyzer {analyzer_class.__name__}: {e}")
        
        return analyzers
