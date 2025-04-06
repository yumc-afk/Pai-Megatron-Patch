"""JaxType analyzer for ML static analysis framework."""

import os
import re
import subprocess
import tempfile
from typing import Dict, List, Optional, Any, Union

from ml_static_analysis.core.analyzer import BaseAnalyzer


class JaxTypeAnalyzer(BaseAnalyzer):
    """JaxType analyzer for ML static analysis framework.
    
    This analyzer checks for proper tensor type annotations using JaxType
    and reports potential tensor shape and type issues.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, verbose: bool = False):
        """Initialize the JaxType analyzer.
        
        Args:
            config: Configuration for the analyzer.
            verbose: Whether to enable verbose output.
        """
        super().__init__(verbose=verbose)
        
        self.config = config or {}
        self.severity_threshold = self.config.get("severity_threshold", "info")
        self.check_shapes = self.config.get("check_shapes", True)
        self.check_dtypes = self.config.get("check_dtypes", True)
        self.check_devices = self.config.get("check_devices", True)
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single file using JaxType.
        
        Args:
            file_path: Path to the file to analyze.
            
        Returns:
            A dictionary with analysis results.
        """
        return self.analyze_files([file_path])
    
    def analyze_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Analyze multiple files using JaxType.
        
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
            f.write(f'  "check_shapes": {str(self.check_shapes).lower()},\n')
            f.write(f'  "check_dtypes": {str(self.check_dtypes).lower()},\n')
            f.write(f'  "check_devices": {str(self.check_devices).lower()}\n')
            f.write("}\n")
        
        try:
            try:
                import jaxtyping
            except ImportError:
                return {
                    "success": False,
                    "error": "JaxType is not installed. Please install it with 'pip install jaxtyping'.",
                }
            
            findings = {}
            
            for file_path in file_paths:
                if not os.path.exists(file_path) or not file_path.endswith(".py"):
                    continue
                
                file_findings = self._analyze_file_with_jaxtype(file_path, config_path)
                
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
    
    def _analyze_file_with_jaxtype(self, file_path: str, config_path: str) -> List[Dict[str, Any]]:
        """Analyze a file using JaxType.
        
        Args:
            file_path: Path to the file to analyze.
            config_path: Path to the JaxType configuration file.
            
        Returns:
            A list of findings.
        """
        try:
            import jaxtyping
            
            if self.verbose:
                print(f"Running JaxType on {file_path}")
            
            findings = self._simulate_jaxtype_analysis(file_path)
            
            return findings
        except Exception as e:
            if self.verbose:
                print(f"Error running JaxType on {file_path}: {str(e)}")
            
            return []
    
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
            has_array_import = False
            
            for i, line in enumerate(lines):
                if "import jaxtyping" in line or "from jaxtyping import" in line:
                    has_jaxtype_import = True
                    if "Array" in line:
                        has_array_import = True
            
            for i, line in enumerate(lines):
                line_num = i + 1
                
                if "torch.Tensor" in line and ":" in line and "->" not in line:
                    if re.search(r"(\w+)\s*:\s*torch\.Tensor", line):
                        findings.append({
                            "line": line_num,
                            "severity": "warning",
                            "message": "Missing tensor type annotation, use Array instead of torch.Tensor",
                            "content": line.strip(),
                            "category": "missing_tensor_type",
                        })
                
                if "->" in line and "torch.Tensor" in line and "def " in line:
                    findings.append({
                        "line": line_num,
                        "severity": "info",
                        "message": "Consider using Array for return type annotation",
                        "content": line.strip(),
                        "category": "return_type_annotation",
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
                            "message": "Tensor operation without shape check, consider adding Array annotations",
                            "content": line.strip(),
                            "category": "shape_check",
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
                            "message": "Tensor dtype operation without dtype check, consider adding Array annotations",
                            "content": line.strip(),
                            "category": "dtype_check",
                        })
            
            if not has_jaxtype_import and len(findings) > 0:
                findings.append({
                    "line": 1,
                    "severity": "info",
                    "message": "Consider importing JaxType for better tensor type annotations",
                    "content": lines[0].strip() if lines else "",
                    "category": "missing_import",
                })
            
            if has_jaxtype_import and not has_array_import and len(findings) > 0:
                for i, line in enumerate(lines):
                    if "import jaxtyping" in line or "from jaxtyping import" in line:
                        findings.append({
                            "line": i + 1,
                            "severity": "info",
                            "message": "Consider importing Array from jaxtyping",
                            "content": line.strip(),
                            "category": "missing_import",
                        })
                        break
            
            return findings
        except Exception as e:
            if self.verbose:
                print(f"Error simulating JaxType analysis on {file_path}: {str(e)}")
            
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
    analyzer = JaxTypeAnalyzer(config=config, verbose=verbose)
    return analyzer.analyze_files(files)
