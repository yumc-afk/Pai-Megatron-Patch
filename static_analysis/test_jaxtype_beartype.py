"""
测试JaxType与Beartype集成的脚本
"""

import os
import sys
import torch
import argparse
from typing import Dict, List, Any, Optional, Union

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from ml_static_analysis.analyzers.jaxtype_analyzer import JaxTypeAnalyzer
from ml_static_analysis.core.config import AnalysisConfig
from ml_static_analysis.core.report import AnalysisReport


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="测试JaxType与Beartype集成")
    
    parser.add_argument(
        "--file",
        type=str,
        default=os.path.join(current_dir, "ml_static_analysis/analyzers/jaxtype_beartype_examples.py"),
        help="要分析的文件路径"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="是否输出详细信息"
    )
    
    return parser.parse_args()


def test_jaxtype_beartype_integration(file_path: str, verbose: bool = False) -> Dict[str, Any]:
    """测试JaxType与Beartype集成"""
    print(f"使用JaxType分析器分析文件: {file_path}")
    
    config = AnalysisConfig()
    config.jaxtype_severity_threshold = "info"
    config.jaxtype_check_shapes = True
    config.jaxtype_check_dtypes = True
    config.jaxtype_check_devices = True
    config.jaxtype_use_beartype = True
    config.verbose = verbose
    
    analyzer = JaxTypeAnalyzer(config)
    
    try:
        try:
            import jaxtyping
            has_jaxtyping = True
        except ImportError:
            has_jaxtyping = False
            print("警告: JaxType未安装，请使用'pip install jaxtyping'安装")
        
        try:
            import beartype
            has_beartype = True
        except ImportError:
            has_beartype = False
            print("警告: Beartype未安装，请使用'pip install beartype'安装")
        
        if not has_jaxtyping or not has_beartype:
            print("缺少必要的依赖，无法继续测试")
            return {
                "success": False,
                "error": "缺少必要的依赖",
                "has_jaxtyping": has_jaxtyping,
                "has_beartype": has_beartype
            }
        
        results = analyzer.analyze_file(file_path)
        
        try:
            sys.path.append(os.path.dirname(file_path))
            module_name = os.path.basename(file_path).replace(".py", "")
            
            if verbose:
                print(f"导入模块: {module_name}")
            
            module = __import__(module_name)
            
            if hasattr(module, "main"):
                if verbose:
                    print("运行示例文件中的main函数")
                module.main()
        except Exception as e:
            print(f"运行示例文件时出错: {str(e)}")
        
        return results
    except Exception as e:
        print(f"JaxType分析器出错: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def main():
    """主函数"""
    args = parse_args()
    
    results = test_jaxtype_beartype_integration(args.file, args.verbose)
    
    if results.get("success", False):
        print("\n分析成功!")
        
        findings = results.get("findings", {})
        for file_path, file_findings in findings.items():
            print(f"\n文件: {file_path}")
            print(f"发现 {len(file_findings)} 个问题")
            
            for finding in file_findings:
                severity = finding.get("severity", "info")
                message = finding.get("message", "")
                line = finding.get("line", 0)
                category = finding.get("category", "other")
                
                print(f"  [{severity.upper()}] 行 {line}: {message} ({category})")
        
        summary = results.get("summary", {})
        total_findings = summary.get("total_findings", 0)
        
        print(f"\n总计: {total_findings} 个问题")
        
        if "has_beartype" in results and results["has_beartype"]:
            print("Beartype集成成功!")
        else:
            print("警告: Beartype未集成")
    else:
        print(f"\n分析失败: {results.get('error', '未知错误')}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
