"""Auto-fix manager for ML static analysis framework."""

import os
import re
import difflib
from typing import Dict, List, Optional, Any, Union, Tuple


class AutoFixManager:
    """Manager for auto-fixing issues found by static analyzers.
    
    This class provides functionality to automatically fix common issues
    found by static analyzers in ML codebases.
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize the auto-fix manager.
        
        Args:
            verbose: Whether to enable verbose output.
        """
        self.verbose = verbose
        self.fixes_applied = []
    
    def apply_fixes(
        self,
        findings: Dict[str, Any],
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """Apply auto-fixes to issues found by static analyzers.
        
        Args:
            findings: Dictionary with findings from static analyzers.
            dry_run: Whether to only simulate applying fixes without actually modifying files.
            
        Returns:
            A dictionary with information about applied fixes.
        """
        self.fixes_applied = []
        
        for analyzer_name, analyzer_results in findings.get("analyzers", {}).items():
            if not analyzer_results or analyzer_results.get("success") is False:
                continue
            
            analyzer_findings = analyzer_results.get("findings", {})
            
            if not analyzer_findings:
                continue
            
            if analyzer_name == "MyPy":
                self._apply_mypy_fixes(analyzer_findings, dry_run)
            elif analyzer_name == "PyTea":
                self._apply_pytea_fixes(analyzer_findings, dry_run)
            elif analyzer_name == "PyAssistant":
                self._apply_pyassistant_fixes(analyzer_findings, dry_run)
            elif analyzer_name == "TorchTyping":
                self._apply_torchtyping_fixes(analyzer_findings, dry_run)
            elif analyzer_name == "Pattern Analysis":
                self._apply_pattern_fixes(analyzer_findings, dry_run)
        
        return {
            "fixes_applied": len(self.fixes_applied),
            "fixes": self.fixes_applied,
        }
    
    def _apply_mypy_fixes(
        self,
        findings: Dict[str, Any],
        dry_run: bool = True,
    ) -> None:
        """Apply auto-fixes to issues found by MyPy.
        
        Args:
            findings: Dictionary with findings from MyPy.
            dry_run: Whether to only simulate applying fixes without actually modifying files.
        """
        for file_path, file_findings in findings.items():
            if not file_findings:
                continue
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception as e:
                if self.verbose:
                    print(f"Error reading file {file_path}: {str(e)}")
                continue
            
            file_modified = False
            
            for finding in file_findings:
                line_num = finding.get("line", 0)
                message = finding.get("message", "")
                
                if line_num <= 0 or line_num > len(lines):
                    continue
                
                fixed, new_line = self._fix_mypy_error(lines[line_num - 1], message)
                
                if fixed:
                    if self.verbose:
                        print(f"Fixed MyPy error in {file_path}:{line_num}: {message}")
                    
                    lines[line_num - 1] = new_line
                    file_modified = True
                    
                    self.fixes_applied.append({
                        "analyzer": "MyPy",
                        "file": file_path,
                        "line": line_num,
                        "message": message,
                        "original": lines[line_num - 1],
                        "fixed": new_line,
                    })
            
            if file_modified and not dry_run:
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.writelines(lines)
                except Exception as e:
                    if self.verbose:
                        print(f"Error writing file {file_path}: {str(e)}")
    
    def _fix_mypy_error(self, line: str, message: str) -> Tuple[bool, str]:
        """Fix a MyPy error in a line of code.
        
        Args:
            line: The line of code with the error.
            message: The error message from MyPy.
            
        Returns:
            A tuple of (fixed, new_line) where fixed is a boolean indicating
            whether the error was fixed, and new_line is the fixed line of code.
        """
        if "Function is missing a type annotation" in message:
            if "return type" in message:
                if "->" not in line and ":" in line:
                    match = re.match(r"(\s*def\s+\w+\s*\([^)]*\))(\s*:)", line)
                    if match:
                        new_line = f"{match.group(1)} -> Any{match.group(2)}{line[match.end(2):]}"
                        return True, new_line
            
            if "parameter" in message and "has no annotation" in message:
                param_name = re.search(r"parameter '(\w+)'", message)
                if param_name:
                    param_name = param_name.group(1)
                    pattern = rf"(\s*def\s+\w+\s*\([^)]*?)(\b{param_name}\b)([^:]*?)([,)])"
                    match = re.search(pattern, line)
                    if match:
                        new_line = f"{match.group(1)}{match.group(2)}: Any{match.group(3)}{match.group(4)}{line[match.end(4):]}"
                        return True, new_line
        
        if "Incompatible return value type" in message:
            if not line.strip().endswith("# type: ignore"):
                new_line = line.rstrip() + "  # type: ignore\n"
                return True, new_line
        
        if "Name 'Any' is not defined" in message:
            if "from typing import" in line:
                match = re.match(r"(\s*from\s+typing\s+import\s+)([^#\n]*)", line)
                if match:
                    imports = match.group(2).strip()
                    if "Any" not in imports:
                        if imports.endswith(","):
                            new_line = f"{match.group(1)}{imports} Any,\n"
                        else:
                            new_line = f"{match.group(1)}{imports}, Any\n"
                        return True, new_line
            else:
                new_line = "from typing import Any\n" + line
                return True, new_line
        
        return False, line
    
    def _apply_pytea_fixes(
        self,
        findings: Dict[str, Any],
        dry_run: bool = True,
    ) -> None:
        """Apply auto-fixes to issues found by PyTea.
        
        Args:
            findings: Dictionary with findings from PyTea.
            dry_run: Whether to only simulate applying fixes without actually modifying files.
        """
        for file_path, file_findings in findings.items():
            if not file_findings:
                continue
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception as e:
                if self.verbose:
                    print(f"Error reading file {file_path}: {str(e)}")
                continue
            
            file_modified = False
            
            for finding in file_findings:
                line_num = finding.get("line", 0)
                message = finding.get("message", "")
                
                if line_num <= 0 or line_num > len(lines):
                    continue
                
                fixed, new_line = self._fix_pytea_error(lines[line_num - 1], message)
                
                if fixed:
                    if self.verbose:
                        print(f"Fixed PyTea error in {file_path}:{line_num}: {message}")
                    
                    lines[line_num - 1] = new_line
                    file_modified = True
                    
                    self.fixes_applied.append({
                        "analyzer": "PyTea",
                        "file": file_path,
                        "line": line_num,
                        "message": message,
                        "original": lines[line_num - 1],
                        "fixed": new_line,
                    })
            
            if file_modified and not dry_run:
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.writelines(lines)
                except Exception as e:
                    if self.verbose:
                        print(f"Error writing file {file_path}: {str(e)}")
    
    def _fix_pytea_error(self, line: str, message: str) -> Tuple[bool, str]:
        """Fix a PyTea error in a line of code.
        
        Args:
            line: The line of code with the error.
            message: The error message from PyTea.
            
        Returns:
            A tuple of (fixed, new_line) where fixed is a boolean indicating
            whether the error was fixed, and new_line is the fixed line of code.
        """
        if "Tensor shape mismatch" in message:
            match = re.search(r"expected shape \[(.*?)\], got \[(.*?)\]", message)
            if match:
                expected_shape = match.group(1)
                var_name = re.search(r"(\w+)\s*=", line)
                if var_name:
                    var_name = var_name.group(1)
                    indent = re.match(r"(\s*)", line).group(1)
                    new_line = line + f"{indent}assert {var_name}.shape == [{expected_shape}], f\"Shape mismatch: expected [{expected_shape}], got {{list({var_name}.shape)}}\"\n"
                    return True, new_line
        
        if "Dimension mismatch" in message:
            match = re.search(r"(\w+)\.(\w+)\(", line)
            if match:
                var_name = match.group(1)
                method_name = match.group(2)
                indent = re.match(r"(\s*)", line).group(1)
                new_line = line + f"{indent}# TODO: Add dimension check for {var_name}.{method_name}\n"
                return True, new_line
        
        return False, line
    
    def _apply_pyassistant_fixes(
        self,
        findings: Dict[str, Any],
        dry_run: bool = True,
    ) -> None:
        """Apply auto-fixes to issues found by PyAssistant.
        
        Args:
            findings: Dictionary with findings from PyAssistant.
            dry_run: Whether to only simulate applying fixes without actually modifying files.
        """
        for file_path, file_findings in findings.items():
            if not file_findings:
                continue
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception as e:
                if self.verbose:
                    print(f"Error reading file {file_path}: {str(e)}")
                continue
            
            file_modified = False
            
            for category, category_findings in file_findings.items():
                for finding in category_findings:
                    line_num = finding.get("line", 0)
                    message = finding.get("message", "")
                    
                    if line_num <= 0 or line_num > len(lines):
                        continue
                    
                    fixed, new_line = self._fix_pyassistant_error(lines[line_num - 1], category, message)
                    
                    if fixed:
                        if self.verbose:
                            print(f"Fixed PyAssistant {category} issue in {file_path}:{line_num}: {message}")
                        
                        lines[line_num - 1] = new_line
                        file_modified = True
                        
                        self.fixes_applied.append({
                            "analyzer": "PyAssistant",
                            "category": category,
                            "file": file_path,
                            "line": line_num,
                            "message": message,
                            "original": lines[line_num - 1],
                            "fixed": new_line,
                        })
            
            if file_modified and not dry_run:
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.writelines(lines)
                except Exception as e:
                    if self.verbose:
                        print(f"Error writing file {file_path}: {str(e)}")
    
    def _fix_pyassistant_error(self, line: str, category: str, message: str) -> Tuple[bool, str]:
        """Fix a PyAssistant error in a line of code.
        
        Args:
            line: The line of code with the error.
            category: The category of the error.
            message: The error message from PyAssistant.
            
        Returns:
            A tuple of (fixed, new_line) where fixed is a boolean indicating
            whether the error was fixed, and new_line is the fixed line of code.
        """
        if category == "thread_safety":
            if "torch.no_grad()" in line and "with" not in line:
                indent = re.match(r"(\s*)", line).group(1)
                new_line = f"{indent}with torch.no_grad():\n"
                return True, new_line
        
        if category == "error_handling":
            if "except:" in line:
                indent = re.match(r"(\s*)", line).group(1)
                new_line = f"{indent}except Exception as e:\n"
                return True, new_line
        
        if category == "performance":
            if "time.time()" in line and "start_time" in line:
                indent = re.match(r"(\s*)", line).group(1)
                new_line = f"{indent}start_time = time.perf_counter()  # More precise timing\n"
                return True, new_line
        
        if category == "weight_switching":
            if "param.copy_(" in line and "with torch.no_grad()" not in line:
                indent = re.match(r"(\s*)", line).group(1)
                new_line = f"{indent}with torch.no_grad():\n{indent}    {line.lstrip()}"
                return True, new_line
        
        return False, line
    
    def _apply_torchtyping_fixes(
        self,
        findings: Dict[str, Any],
        dry_run: bool = True,
    ) -> None:
        """Apply auto-fixes to issues found by TorchTyping.
        
        Args:
            findings: Dictionary with findings from TorchTyping.
            dry_run: Whether to only simulate applying fixes without actually modifying files.
        """
        for file_path, file_findings in findings.items():
            if not file_findings:
                continue
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception as e:
                if self.verbose:
                    print(f"Error reading file {file_path}: {str(e)}")
                continue
            
            file_modified = False
            
            for finding in file_findings:
                line_num = finding.get("line", 0)
                message = finding.get("message", "")
                
                if line_num <= 0 or line_num > len(lines):
                    continue
                
                fixed, new_line = self._fix_torchtyping_error(lines[line_num - 1], message)
                
                if fixed:
                    if self.verbose:
                        print(f"Fixed TorchTyping error in {file_path}:{line_num}: {message}")
                    
                    lines[line_num - 1] = new_line
                    file_modified = True
                    
                    self.fixes_applied.append({
                        "analyzer": "TorchTyping",
                        "file": file_path,
                        "line": line_num,
                        "message": message,
                        "original": lines[line_num - 1],
                        "fixed": new_line,
                    })
            
            if file_modified and not dry_run:
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.writelines(lines)
                except Exception as e:
                    if self.verbose:
                        print(f"Error writing file {file_path}: {str(e)}")
    
    def _fix_torchtyping_error(self, line: str, message: str) -> Tuple[bool, str]:
        """Fix a TorchTyping error in a line of code.
        
        Args:
            line: The line of code with the error.
            message: The error message from TorchTyping.
            
        Returns:
            A tuple of (fixed, new_line) where fixed is a boolean indicating
            whether the error was fixed, and new_line is the fixed line of code.
        """
        if "Missing tensor type annotation" in message:
            match = re.search(r"(\w+):\s*(\w+)", line)
            if match:
                var_name = match.group(1)
                type_name = match.group(2)
                if type_name == "torch.Tensor":
                    new_line = line.replace(f"{var_name}: {type_name}", f"{var_name}: TensorType[..., torch.float]")
                    return True, new_line
        
        if "TensorType is not defined" in message:
            if "from torchtyping import" in line:
                match = re.match(r"(\s*from\s+torchtyping\s+import\s+)([^#\n]*)", line)
                if match:
                    imports = match.group(2).strip()
                    if "TensorType" not in imports:
                        if imports.endswith(","):
                            new_line = f"{match.group(1)}{imports} TensorType,\n"
                        else:
                            new_line = f"{match.group(1)}{imports}, TensorType\n"
                        return True, new_line
            else:
                new_line = "from torchtyping import TensorType\n" + line
                return True, new_line
        
        return False, line
    
    def _apply_pattern_fixes(
        self,
        findings: Dict[str, Any],
        dry_run: bool = True,
    ) -> None:
        """Apply auto-fixes to issues found by pattern analysis.
        
        Args:
            findings: Dictionary with findings from pattern analysis.
            dry_run: Whether to only simulate applying fixes without actually modifying files.
        """
        for file_path, file_findings in findings.items():
            if not file_findings:
                continue
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception as e:
                if self.verbose:
                    print(f"Error reading file {file_path}: {str(e)}")
                continue
            
            file_modified = False
            
            for category, category_findings in file_findings.items():
                for finding in category_findings:
                    line_num = finding.get("line", 0)
                    message = finding.get("message", "")
                    
                    if line_num <= 0 or line_num > len(lines):
                        continue
                    
                    fixed, new_line = self._fix_pattern_error(lines[line_num - 1], category, message)
                    
                    if fixed:
                        if self.verbose:
                            print(f"Fixed pattern {category} issue in {file_path}:{line_num}: {message}")
                        
                        lines[line_num - 1] = new_line
                        file_modified = True
                        
                        self.fixes_applied.append({
                            "analyzer": "Pattern Analysis",
                            "category": category,
                            "file": file_path,
                            "line": line_num,
                            "message": message,
                            "original": lines[line_num - 1],
                            "fixed": new_line,
                        })
            
            if file_modified and not dry_run:
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.writelines(lines)
                except Exception as e:
                    if self.verbose:
                        print(f"Error writing file {file_path}: {str(e)}")
    
    def _fix_pattern_error(self, line: str, category: str, message: str) -> Tuple[bool, str]:
        """Fix a pattern error in a line of code.
        
        Args:
            line: The line of code with the error.
            category: The category of the error.
            message: The error message from pattern analysis.
            
        Returns:
            A tuple of (fixed, new_line) where fixed is a boolean indicating
            whether the error was fixed, and new_line is the fixed line of code.
        """
        if category == "thread_safety":
            if "torch.no_grad()" in line and "with" not in line:
                indent = re.match(r"(\s*)", line).group(1)
                new_line = f"{indent}with torch.no_grad():\n"
                return True, new_line
        
        if category == "error_handling":
            if "except:" in line:
                indent = re.match(r"(\s*)", line).group(1)
                new_line = f"{indent}except Exception as e:\n"
                return True, new_line
        
        if category == "performance":
            if "time.time()" in line and "start_time" in line:
                indent = re.match(r"(\s*)", line).group(1)
                new_line = f"{indent}start_time = time.perf_counter()  # More precise timing\n"
                return True, new_line
        
        if category == "weight_switching":
            if "param.copy_(" in line and "with torch.no_grad()" not in line:
                indent = re.match(r"(\s*)", line).group(1)
                new_line = f"{indent}with torch.no_grad():\n{indent}    {line.lstrip()}"
                return True, new_line
        
        return False, line
    
    def generate_fix_report(self) -> str:
        """Generate a report of applied fixes.
        
        Returns:
            A string containing the report in Markdown format.
        """
        if not self.fixes_applied:
            return "No fixes applied."
        
        report = "# Auto-Fix Report\n\n"
        
        fixes_by_file = {}
        for fix in self.fixes_applied:
            file_path = fix["file"]
            if file_path not in fixes_by_file:
                fixes_by_file[file_path] = []
            fixes_by_file[file_path].append(fix)
        
        for file_path, fixes in fixes_by_file.items():
            report += f"## {os.path.basename(file_path)}\n\n"
            
            for fix in fixes:
                analyzer = fix["analyzer"]
                line = fix["line"]
                message = fix["message"]
                original = fix["original"].strip()
                fixed = fix["fixed"].strip()
                
                report += f"### Line {line}: {analyzer}\n\n"
                report += f"**Message:** {message}\n\n"
                report += "**Original:**\n```python\n"
                report += original + "\n"
                report += "```\n\n"
                report += "**Fixed:**\n```python\n"
                report += fixed + "\n"
                report += "```\n\n"
                
                diff = difflib.unified_diff(
                    [original + "\n"],
                    [fixed + "\n"],
                    n=0,
                )
                
                diff_text = "".join(list(diff)[2:])  # Skip the first two lines
                
                if diff_text:
                    report += "**Diff:**\n```diff\n"
                    report += diff_text
                    report += "```\n\n"
            
            report += "\n"
        
        return report
