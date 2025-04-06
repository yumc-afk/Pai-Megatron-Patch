#
#
#

"""
使用新框架分析DeepSeek V3项目的示例脚本
"""

import os
import sys
import logging
import argparse
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(repo_root)

from ml_static_analysis.core.config import AnalysisConfig
from ml_static_analysis.core.analyzer import AnalyzerRegistry
from ml_static_analysis.report_generator import ReportGenerator
from ml_static_analysis.autofix.autofix_manager import AutofixManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(current_dir, "dsv3_new_analysis.log"))
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="使用新框架分析DeepSeek V3项目",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--target", 
        type=str, 
        default=os.path.join(repo_root, "examples", "deepseek_v3"),
        help="要分析的目标路径"
    )
    
    parser.add_argument(
        "--analyzers", 
        type=str, 
        default="all",
        help="要使用的分析器，用逗号分隔，例如 'mypy,pytea,pattern'。使用 'all' 表示所有分析器"
    )
    
    parser.add_argument(
        "--report-format", 
        type=str, 
        choices=["markdown", "json", "html"], 
        default="markdown",
        help="报告格式"
    )
    
    parser.add_argument(
        "--autofix", 
        action="store_true",
        help="自动修复发现的问题"
    )
    
    parser.add_argument(
        "--autofix-dry-run", 
        action="store_true",
        help="显示自动修复建议但不应用"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="显示详细输出"
    )
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    config = AnalysisConfig(
        target_path=args.target,
        analyzers=args.analyzers.split(",") if args.analyzers != "all" else "all",
        verbose=args.verbose,
        autofix_enabled=args.autofix or args.autofix_dry_run,
        autofix_dry_run=args.autofix_dry_run
    )
    
    analyzer_registry = AnalyzerRegistry(config)
    
    analyzers = analyzer_registry.get_analyzers()
    
    logger.info(f"开始分析 {args.target}")
    logger.info(f"使用分析器: {', '.join([analyzer.name for analyzer in analyzers])}")
    
    analysis_results = {}
    for analyzer in analyzers:
        logger.info(f"运行分析器: {analyzer.name}")
        try:
            report = analyzer.analyze()
            analysis_results[analyzer.name] = report
            
            logger.info(f"{analyzer.name} 分析完成:")
            logger.info(f"  - 错误: {len(report.get_errors())}")
            logger.info(f"  - 警告: {len(report.get_warnings())}")
            logger.info(f"  - 建议: {len(report.get_suggestions())}")
        except Exception as e:
            logger.error(f"{analyzer.name} 分析失败: {e}")
    
    report_generator = ReportGenerator(config)
    report_path = report_generator.generate_report(
        analysis_results=analysis_results,
        target_path=args.target,
        format=args.report_format
    )
    
    logger.info(f"分析报告已生成: {report_path}")
    
    if args.autofix or args.autofix_dry_run:
        autofix_manager = AutofixManager(config)
        fixes_applied, fixes_failed, _ = autofix_manager.apply_fixes(analysis_results)
        
        autofix_report = autofix_manager.generate_autofix_report()
        autofix_report_path = os.path.join(
            os.path.dirname(report_path),
            f"autofix_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        
        with open(autofix_report_path, "w", encoding="utf-8") as f:
            f.write(autofix_report)
        
        logger.info(f"自动修复报告已生成: {autofix_report_path}")
        logger.info(f"应用的修复: {fixes_applied}, 失败的修复: {fixes_failed}")
    
    logger.info("分析完成")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
