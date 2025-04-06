"""
使用JaxType分析器分析DeepSeek V3项目，生成改进报告。
"""

import os
import sys
import argparse
import datetime
import json
from typing import Dict, List, Any, Optional, Union

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
sys.path.append(repo_root)
sys.path.append(current_dir)

from ml_static_analysis.analyzers.jaxtype_analyzer import JaxTypeAnalyzer
from ml_static_analysis.analyzers.mypy_analyzer import MyPyAnalyzer
from ml_static_analysis.analyzers.pattern_analyzer import PatternAnalyzer
from ml_static_analysis.analyzers.pytea_analyzer import PyTeaAnalyzer
from ml_static_analysis.core.config import AnalysisConfig
from ml_static_analysis.core.report import AnalysisReport
from ml_static_analysis.report_generator import ReportGenerator
from ml_static_analysis.autofix.autofix_manager import AutofixManager


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用JaxType分析DeepSeek V3项目")
    
    parser.add_argument(
        "--dir",
        type=str,
        default=os.path.join(repo_root, "examples/deepseek_v3"),
        help="要分析的目录路径"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(repo_root, f"dsv3_jaxtype_analysis_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"),
        help="分析报告输出路径"
    )
    
    parser.add_argument(
        "--autofix",
        action="store_true",
        help="是否自动修复发现的问题"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="是否输出详细信息"
    )
    
    return parser.parse_args()


def find_python_files(directory: str) -> List[str]:
    """查找目录中的所有Python文件"""
    python_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    
    return python_files


def analyze_with_jaxtype(files: List[str], verbose: bool = False) -> Dict[str, Any]:
    """使用JaxType分析器分析文件"""
    print(f"使用JaxType分析{len(files)}个文件...")
    
    config = AnalysisConfig()
    config.jaxtype_severity_threshold = "info"
    config.jaxtype_check_shapes = True
    config.jaxtype_check_dtypes = True
    config.jaxtype_check_devices = True
    config.verbose = verbose
    
    analyzer = JaxTypeAnalyzer(config)
    
    try:
        return analyzer.analyze_files(files)
    except Exception as e:
        print(f"JaxType分析器出错: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "summary": {
                "analyzed_files": 0,
                "total_findings": 0,
                "findings_by_category": {},
                "findings_by_severity": {},
            },
            "findings": {},
        }


def analyze_with_mypy(files: List[str], verbose: bool = False) -> Dict[str, Any]:
    """使用MyPy分析器分析文件"""
    print(f"使用MyPy分析{len(files)}个文件...")
    
    config = AnalysisConfig()
    config.mypy_python_version = "3.8"
    config.mypy_disallow_untyped_defs = False
    config.mypy_disallow_incomplete_defs = False
    config.verbose = verbose
    
    analyzer = MyPyAnalyzer(config)
    
    return analyzer.analyze_files(files)


def analyze_with_pattern(files: List[str], verbose: bool = False) -> Dict[str, Any]:
    """使用模式分析器分析文件"""
    print(f"使用模式分析器分析{len(files)}个文件...")
    
    config = AnalysisConfig()
    config.pattern_patterns = [
        {
            "name": "tensor_device_mismatch",
            "regex": r"\.to\(.*device.*\)",
            "description": "可能的张量设备不匹配",
            "severity": "warning",
        },
        {
            "name": "unchecked_tensor_shape",
            "regex": r"(reshape|view|permute|transpose)\(",
            "description": "未检查的张量形状操作",
            "severity": "warning",
        },
        {
            "name": "moe_routing_issue",
            "regex": r"(router|dispatch|combine).*token",
            "description": "MOE路由相关操作",
            "severity": "info",
        },
        {
            "name": "mla_attention_issue",
            "regex": r"multi_latent_attention|mla",
            "description": "MLA注意力相关操作",
            "severity": "info",
        },
    ]
    config.verbose = verbose
    
    analyzer = PatternAnalyzer(config)
    
    return analyzer.analyze_files(files)


def analyze_with_pytea(files: List[str], verbose: bool = False) -> Dict[str, Any]:
    """使用PyTea分析器分析文件"""
    print(f"使用PyTea分析{len(files)}个文件...")
    
    config = AnalysisConfig()
    config.pytea_max_depth = 5
    config.pytea_timeout = 60
    config.verbose = verbose
    
    analyzer = PyTeaAnalyzer(config)
    
    return analyzer.analyze_files(files)


def generate_report(
    jaxtype_results: Dict[str, Any],
    mypy_results: Dict[str, Any],
    pattern_results: Dict[str, Any],
    pytea_results: Dict[str, Any],
    output_path: str,
    files: List[str] = None,
    verbose: bool = False
) -> None:
    """生成分析报告"""
    print(f"生成分析报告到{output_path}...")
    
    report = AnalysisReport("MultiAnalyzer")
    
    if jaxtype_results.get("success", False):
        report.add_analyzer_results("JaxType", jaxtype_results)
    
    if mypy_results.get("success", False):
        report.add_analyzer_results("MyPy", mypy_results)
    
    if pattern_results.get("success", False):
        report.add_analyzer_results("Pattern", pattern_results)
    
    if pytea_results.get("success", False):
        report.add_analyzer_results("PyTea", pytea_results)
    
    config = AnalysisConfig()
    config.verbose = verbose
    
    report_generator = ReportGenerator(config)
    
    analysis_results = {}
    if jaxtype_results.get("success", False):
        analysis_results["JaxType"] = report
    if mypy_results.get("success", False):
        analysis_results["MyPy"] = report
    if pattern_results.get("success", False):
        analysis_results["Pattern"] = report
    if pytea_results.get("success", False):
        analysis_results["PyTea"] = report
    
    report_path = report_generator.generate_report(
        analysis_results=analysis_results,
        target_path=output_path,
        format="markdown"
    )
    
    print(f"报告已保存到 {report_path}")
    
    print(f"分析报告已生成到{output_path}")
    
    if verbose:
        print("\n分析摘要:")
        print(f"- 总发现问题数: {report.get_total_findings_count()}")
        print(f"- 错误数: {len(report.get_errors())}")
        print(f"- 警告数: {len(report.get_warnings())}")
        print(f"- 建议数: {len(report.get_suggestions())}")


def apply_autofix(
    jaxtype_results: Dict[str, Any],
    mypy_results: Dict[str, Any],
    pattern_results: Dict[str, Any],
    pytea_results: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """应用自动修复"""
    print("应用自动修复...")
    
    config = AnalysisConfig()
    config.verbose = verbose
    
    autofix_manager = AutofixManager(config)
    
    report = AnalysisReport("MultiAnalyzer")
    
    if jaxtype_results.get("success", False):
        report.add_analyzer_results("JaxType", jaxtype_results)
    
    if mypy_results.get("success", False):
        report.add_analyzer_results("MyPy", mypy_results)
    
    if pattern_results.get("success", False):
        report.add_analyzer_results("Pattern", pattern_results)
    
    if pytea_results.get("success", False):
        report.add_analyzer_results("PyTea", pytea_results)
    
    fix_results = autofix_manager.apply_fixes(report)
    
    if verbose:
        print("\n自动修复摘要:")
        print(f"- 总修复尝试数: {fix_results.get('total_attempts', 0)}")
        print(f"- 成功修复数: {fix_results.get('successful_fixes', 0)}")
        print(f"- 失败修复数: {fix_results.get('failed_fixes', 0)}")
    
    return fix_results


def main():
    """主函数"""
    args = parse_args()
    
    config = AnalysisConfig()
    config.target_dir = args.dir
    config.output_file = args.output
    config.verbose = args.verbose
    config.autofix = args.autofix
    
    config.jaxtype_severity_threshold = "info"
    config.jaxtype_check_shapes = True
    config.jaxtype_check_dtypes = True
    config.jaxtype_check_devices = True
    
    config.mypy_python_version = "3.8"
    config.mypy_disallow_untyped_defs = False
    config.mypy_disallow_incomplete_defs = False
    
    config.pattern_patterns = [
        {
            "name": "tensor_device_mismatch",
            "regex": r"\.to\(.*device.*\)",
            "description": "可能的张量设备不匹配",
            "severity": "warning",
        },
        {
            "name": "unchecked_tensor_shape",
            "regex": r"(reshape|view|permute|transpose)\(",
            "description": "未检查的张量形状操作",
            "severity": "warning",
        },
        {
            "name": "moe_routing_issue",
            "regex": r"(router|dispatch|combine).*token",
            "description": "MOE路由相关操作",
            "severity": "info",
        },
        {
            "name": "mla_attention_issue",
            "regex": r"multi_latent_attention|mla",
            "description": "MLA注意力相关操作",
            "severity": "info",
        },
    ]
    
    config.pytea_max_depth = 5
    config.pytea_timeout = 60
    
    files = find_python_files(args.dir)
    print(f"找到{len(files)}个Python文件")
    
    jaxtype_results = analyze_with_jaxtype(files, args.verbose)
    
    mypy_results = analyze_with_mypy(files, args.verbose)
    
    pattern_results = analyze_with_pattern(files, args.verbose)
    
    pytea_results = analyze_with_pytea(files, args.verbose)
    
    if args.autofix:
        fix_results = apply_autofix(
            jaxtype_results,
            mypy_results,
            pattern_results,
            pytea_results,
            args.verbose
        )
        
        print("\n重新分析修复后的文件...")
        jaxtype_results = analyze_with_jaxtype(files, args.verbose)
        mypy_results = analyze_with_mypy(files, args.verbose)
        pattern_results = analyze_with_pattern(files, args.verbose)
        pytea_results = analyze_with_pytea(files, args.verbose)
    
    generate_report(
        jaxtype_results,
        mypy_results,
        pattern_results,
        pytea_results,
        args.output,
        files=files,
        verbose=args.verbose
    )
    
    return 0


if __name__ == "__main__":
    main()
