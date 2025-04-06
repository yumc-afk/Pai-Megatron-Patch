"""Configuration management for ML static analysis framework."""

import os
import json
from typing import Dict, List, Optional, Any, Union


class AnalysisConfig:
    """Configuration manager for ML static analysis framework.
    
    This class handles loading, validating, and providing access to
    configuration settings for the static analysis framework.
    """
    
    DEFAULT_CONFIG = {
        "mypy": {
            "enabled": True,
            "strict": False,
            "ignore_missing_imports": True
        },
        "pytea": {
            "enabled": False,
            "max_depth": 10,
            "timeout": 60
        },
        "pyassistant": {
            "enabled": False,
            "severity_threshold": "info"
        },
        "jaxtype": {
            "enabled": True,
            "severity_threshold": "info",
            "check_shapes": True,
            "check_dtypes": True,
            "check_devices": True
        },
        "pattern_analysis": {
            "enabled": True,
            "categories": ["thread_safety", "error_handling", "performance", "weight_switching"]
        }
    }
    
    def __init__(self, 
                 config_path: Optional[str] = None, 
                 verbose: bool = False,
                 target_path: Optional[str] = None,
                 analyzers: Union[List[str], str] = "all",
                 autofix_enabled: bool = False,
                 autofix_dry_run: bool = False):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to a JSON configuration file. If None, default configuration will be used.
            verbose: Whether to enable verbose output.
            target_path: Path to the target file or directory to analyze.
            analyzers: List of analyzer names to use, or "all" to use all available analyzers.
            autofix_enabled: Whether to enable automatic fixing of issues.
            autofix_dry_run: Whether to run autofix in dry-run mode (show fixes but don't apply them).
        """
        self.verbose = verbose
        self.config = self.DEFAULT_CONFIG.copy()
        self.target_path = target_path
        self.analyzers = analyzers
        self.autofix_enabled = autofix_enabled
        self.autofix_dry_run = autofix_dry_run
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from a JSON file.
        
        Args:
            config_path: Path to a JSON configuration file.
        """
        if not os.path.exists(config_path):
            if self.verbose:
                print(f"Warning: Configuration file {config_path} does not exist. Using default configuration.")
            return
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            
            for analyzer, settings in user_config.items():
                if analyzer in self.config:
                    if isinstance(settings, dict) and isinstance(self.config[analyzer], dict):
                        self.config[analyzer].update(settings)
                    else:
                        self.config[analyzer] = settings
                else:
                    self.config[analyzer] = settings
            
            if self.verbose:
                print(f"Loaded configuration from {config_path}")
        except Exception as e:
            if self.verbose:
                print(f"Error loading configuration from {config_path}: {str(e)}")
            print(f"Warning: Failed to load configuration from {config_path}. Using default configuration.")
    
    def save_config(self, config_path: str) -> None:
        """Save configuration to a JSON file.
        
        Args:
            config_path: Path to save the configuration to.
        """
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)
            
            if self.verbose:
                print(f"Saved configuration to {config_path}")
        except Exception as e:
            print(f"Error saving configuration to {config_path}: {str(e)}")
    
    def get_analyzer_config(self, analyzer: str) -> Dict[str, Any]:
        """Get configuration for a specific analyzer.
        
        Args:
            analyzer: Name of the analyzer.
            
        Returns:
            A dictionary with configuration settings for the analyzer.
        """
        return self.config.get(analyzer, {})
    
    def is_analyzer_enabled(self, analyzer: str) -> bool:
        """Check if an analyzer is enabled.
        
        Args:
            analyzer: Name of the analyzer.
            
        Returns:
            True if the analyzer is enabled, False otherwise.
        """
        analyzer_config = self.get_analyzer_config(analyzer)
        return analyzer_config.get("enabled", False)
    
    def get_enabled_analyzers(self) -> List[str]:
        """Get a list of enabled analyzers.
        
        Returns:
            A list of names of enabled analyzers.
        """
        return [
            analyzer
            for analyzer, settings in self.config.items()
            if isinstance(settings, dict) and settings.get("enabled", False)
        ]
    
    def set_analyzer_enabled(self, analyzer: str, enabled: bool) -> None:
        """Enable or disable an analyzer.
        
        Args:
            analyzer: Name of the analyzer.
            enabled: Whether to enable the analyzer.
        """
        if analyzer in self.config:
            if isinstance(self.config[analyzer], dict):
                self.config[analyzer]["enabled"] = enabled
            else:
                self.config[analyzer] = {"enabled": enabled}
        else:
            self.config[analyzer] = {"enabled": enabled}
    
    def set_analyzer_config(self, analyzer: str, settings: Dict[str, Any]) -> None:
        """Set configuration for a specific analyzer.
        
        Args:
            analyzer: Name of the analyzer.
            settings: Configuration settings for the analyzer.
        """
        if analyzer in self.config:
            if isinstance(self.config[analyzer], dict) and isinstance(settings, dict):
                self.config[analyzer].update(settings)
            else:
                self.config[analyzer] = settings
        else:
            self.config[analyzer] = settings
    
    def get_lite_config(self) -> Dict[str, Any]:
        """Get a lite version of the configuration.
        
        Returns:
            A dictionary with configuration settings for the lite version.
        """
        lite_config = {}
        
        core_analyzers = ["mypy", "pytea", "pyassistant", "jaxtype"]
        
        for analyzer in core_analyzers:
            if analyzer in self.config:
                lite_config[analyzer] = self.config[analyzer].copy()
        
        return lite_config
