"""
应用JaxType分析结果对DeepSeek V3项目进行改进。
"""

import os
import sys
import argparse
import re
from typing import Dict, List, Any, Optional, Union

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
sys.path.append(repo_root)
sys.path.append(current_dir)

from ml_static_analysis.autofix.autofix_manager import AutofixManager
from ml_static_analysis.core.config import AnalysisConfig


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="应用JaxType分析结果对DeepSeek V3项目进行改进")
    
    parser.add_argument(
        "--dir",
        type=str,
        default=os.path.join(repo_root, "examples/deepseek_v3"),
        help="要改进的目录路径"
    )
    
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="分析报告路径，如果不提供则使用最新的报告"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="是否输出详细信息"
    )
    
    return parser.parse_args()


def find_latest_report():
    """查找最新的分析报告"""
    reports_dir = os.path.join(current_dir, "reports")
    if not os.path.exists(reports_dir):
        return None
    
    reports = [os.path.join(reports_dir, f) for f in os.listdir(reports_dir) if f.endswith(".markdown") or f.endswith(".md")]
    if not reports:
        return None
    
    return max(reports, key=os.path.getmtime)


def add_jaxtype_imports(file_path: str) -> bool:
    """添加JaxType导入"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    if "import jaxtyping" in content or "from jaxtyping import" in content:
        return False
    
    if "import torch" not in content:
        return False
    
    new_content = re.sub(
        r"(import torch.*?)(\n+)",
        r"\1\n\nimport jaxtyping\nfrom jaxtyping import Array, Float, Int\n\2",
        content
    )
    
    if new_content != content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return True
    
    return False


def fix_tensor_annotations(file_path: str) -> bool:
    """修复张量类型注解"""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    modified = False
    new_lines = []
    
    for line in lines:
        if "torch.Tensor" in line and ":" in line:
            new_line = re.sub(
                r"(\w+)\s*:\s*torch\.Tensor",
                r"\1: Array",
                line
            )
            
            new_line = re.sub(
                r"(\w+\s*:\s*)torch\.Tensor",
                r"\1Array",
                new_line
            )
            
            new_line = re.sub(
                r"(def \w+\([^)]*\)\s*->\s*)torch\.Tensor",
                r"\1Array",
                new_line
            )
            
            if new_line != line:
                modified = True
                new_lines.append(new_line)
                continue
        
        new_lines.append(line)
    
    if modified:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    
    return modified


def add_shape_checks(file_path: str) -> bool:
    """添加形状检查"""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    modified = False
    new_lines = []
    
    for i, line in enumerate(lines):
        if re.search(r"\.(reshape|view|permute|transpose)\(", line):
            has_shape_check = False
            for j in range(max(0, i-5), i):
                if ".shape" in lines[j] or "assert" in lines[j]:
                    has_shape_check = True
                    break
            
            if not has_shape_check:
                indent = re.match(r"^(\s*)", line).group(1)
                tensor_name = re.search(r"(\w+)\.(reshape|view|permute|transpose)", line)
                if tensor_name:
                    tensor_name = tensor_name.group(1)
                    shape_check = f"{indent}# 确保张量形状正确\n{indent}assert {tensor_name}.shape, f\"Unexpected shape: {{{{tensor.shape}}}}\"\n"
                    new_lines.append(shape_check)
                    modified = True
        
        new_lines.append(line)
    
    if modified:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    
    return modified


def add_dtype_checks(file_path: str) -> bool:
    """添加数据类型检查"""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    modified = False
    new_lines = []
    
    for i, line in enumerate(lines):
        if re.search(r"torch\.(float|int|long|bool|double)", line):
            has_dtype_check = False
            for j in range(max(0, i-5), i):
                if ".dtype" in lines[j] or "assert" in lines[j]:
                    has_dtype_check = True
                    break
            
            if not has_dtype_check:
                indent = re.match(r"^(\s*)", line).group(1)
                tensor_name = re.search(r"(\w+)\s*=", line)
                if tensor_name:
                    tensor_name = tensor_name.group(1)
                    dtype_check = f"{indent}# 确保张量数据类型正确\n{indent}assert {tensor_name}.dtype, f\"Unexpected dtype: {{{{tensor.dtype}}}}\"\n"
                    new_lines.append(dtype_check)
                    modified = True
        
        new_lines.append(line)
    
    if modified:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    
    return modified


def fix_memory_history_issue(file_path: str) -> bool:
    """修复_record_memory_history参数问题"""
    if not os.path.basename(file_path) == "pretrain_deepseek.py":
        return False
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    if "_record_memory_history(True)" in content:
        new_content = content.replace(
            "_record_memory_history(True)",
            "_record_memory_history('state')"
        )
        
        if new_content != content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            return True
    
    return False


def fix_cuda_observer_issue(file_path: str) -> bool:
    """修复_cuda_attach_out_of_memory_observer问题"""
    if not os.path.basename(file_path) == "pretrain_deepseek.py":
        return False
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    if "_cuda_attach_out_of_memory_observer" in content:
        new_content = re.sub(
            r"torch\._cuda_attach_out_of_memory_observer\(\)",
            "# torch._cuda_attach_out_of_memory_observer() # 已弃用，使用新的内存分析方法",
            content
        )
        
        if new_content != content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            return True
    
    return False


def fix_moe_assignment_issue(file_path: str) -> bool:
    """修复MOE赋值类型不匹配问题"""
    if not os.path.basename(file_path) == "test_mla_moe_components.py":
        return False
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    if "CustomMoEExpert" in content and "CustomMoE" in content:
        match = re.search(r"(\s*)(\w+)\s*=\s*CustomMoEExpert\(", content)
        if match:
            indent = match.group(1)
            var_name = match.group(2)
            
            type_match = re.search(rf"{var_name}\s*:\s*CustomMoE", content)
            if type_match:
                new_content = re.sub(
                    rf"({var_name})\s*:\s*CustomMoE",
                    r"\1: CustomMoEExpert",
                    content
                )
                
                if new_content != content:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    return True
    
    return False


def apply_fixes_to_file(file_path: str, verbose: bool = False) -> Dict[str, int]:
    """应用所有修复到单个文件"""
    if not file_path.endswith(".py"):
        return {"total": 0, "fixed": 0}
    
    if verbose:
        print(f"正在处理文件: {file_path}")
    
    fixes_applied = 0
    
    if add_jaxtype_imports(file_path):
        fixes_applied += 1
        if verbose:
            print(f"  - 已添加JaxType导入")
    
    if fix_tensor_annotations(file_path):
        fixes_applied += 1
        if verbose:
            print(f"  - 已修复张量类型注解")
    
    if add_shape_checks(file_path):
        fixes_applied += 1
        if verbose:
            print(f"  - 已添加形状检查")
    
    if add_dtype_checks(file_path):
        fixes_applied += 1
        if verbose:
            print(f"  - 已添加数据类型检查")
    
    if fix_memory_history_issue(file_path):
        fixes_applied += 1
        if verbose:
            print(f"  - 已修复_record_memory_history参数问题")
    
    if fix_cuda_observer_issue(file_path):
        fixes_applied += 1
        if verbose:
            print(f"  - 已修复_cuda_attach_out_of_memory_observer问题")
    
    if fix_moe_assignment_issue(file_path):
        fixes_applied += 1
        if verbose:
            print(f"  - 已修复MOE赋值类型不匹配问题")
    
    return {"total": 1, "fixed": 1 if fixes_applied > 0 else 0}


def apply_fixes_to_directory(directory: str, verbose: bool = False) -> Dict[str, int]:
    """应用所有修复到目录中的所有Python文件"""
    results = {"total": 0, "fixed": 0}
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                file_results = apply_fixes_to_file(file_path, verbose)
                
                results["total"] += file_results["total"]
                results["fixed"] += file_results["fixed"]
    
    return results


def main():
    """主函数"""
    args = parse_args()
    
    report_path = args.report
    if not report_path:
        report_path = find_latest_report()
        if not report_path:
            print("未找到分析报告，将直接应用修复")
    else:
        if not os.path.exists(report_path):
            print(f"报告文件不存在: {report_path}")
            return 1
    
    if report_path and args.verbose:
        print(f"使用分析报告: {report_path}")
    
    print(f"正在对目录应用修复: {args.dir}")
    results = apply_fixes_to_directory(args.dir, args.verbose)
    
    print("\n修复摘要:")
    print(f"- 总文件数: {results['total']}")
    print(f"- 已修复文件数: {results['fixed']}")
    print(f"- 修复率: {results['fixed'] / results['total'] * 100:.2f}%")
    
    print("\n建议运行以下命令验证修复效果:")
    print(f"python analyze_dsv3_with_jaxtype.py --dir {args.dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
