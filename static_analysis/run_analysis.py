#
#
#

"""
静态分析调度器

此脚本协调多个静态分析工具，为PyTorch分布式训练代码提供综合分析。
它作为LLM调度者的执行工具，根据配置运行各种分析工具并生成结构化报告。
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("analysis_orchestrator")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
TOOLS_DIR = os.path.join(CURRENT_DIR, "tools")
REPORTS_DIR = os.path.join(CURRENT_DIR, "reports")

os.makedirs(TOOLS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

sys.path.append(TOOLS_DIR)
try:
    from pytea_analyzer import PyTeaAnalyzer
    HAS_PYTEA = True
except ImportError:
    logger.warning("无法导入PyTea分析器，将使用模拟模式")
    HAS_PYTEA = False

class AnalysisOrchestrator:
    """
    静态分析调度器
    
    协调多个静态分析工具，为PyTorch分布式训练代码提供综合分析。
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化分析调度器
        
        Args:
            config: 配置字典，可选
        """
        self.config = config or {}
        self.results = {}
        self.tools = {}
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        self._init_tools()
    
    def _init_tools(self):
        """初始化分析工具"""
        if HAS_PYTEA:
            self.tools["pytea"] = PyTeaAnalyzer(self.config.get("pytea", {}))
        else:
            logger.warning("PyTea未安装，将使用模拟模式")
            self.tools["pytea"] = None
        
    
    def analyze_file(self, file_path: str, **kwargs) -> Dict:
        """
        分析单个文件
        
        Args:
            file_path: 文件路径
            **kwargs: 额外参数
                
        Returns:
            分析结果字典
        """
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return {"error": f"文件不存在: {file_path}"}
        
        logger.info(f"开始分析文件: {file_path}")
        
        results = {}
        
        if "pytea" in self.tools and self.tools["pytea"]:
            logger.info("运行PyTea张量形状分析...")
            results["pytea"] = self.tools["pytea"].analyze_file(file_path, **kwargs)
        
        
        
        
        self.results[file_path] = results
        
        return results
    
    def analyze_dir(self, dir_path: str, **kwargs) -> Dict:
        """
        分析目录中的所有Python文件
        
        Args:
            dir_path: 目录路径
            **kwargs: 额外参数
                
        Returns:
            分析结果字典
        """
        if not os.path.exists(dir_path):
            logger.error(f"目录不存在: {dir_path}")
            return {"error": f"目录不存在: {dir_path}"}
        
        if not os.path.isdir(dir_path):
            logger.error(f"不是目录: {dir_path}")
            return {"error": f"不是目录: {dir_path}"}
        
        logger.info(f"开始分析目录: {dir_path}")
        
        python_files = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))
        
        logger.info(f"找到 {len(python_files)} 个Python文件")
        
        results = {}
        for file_path in python_files:
            results[file_path] = self.analyze_file(file_path, **kwargs)
        
        return results
    
    def analyze_component(self, component_name: str, dir_path: str, **kwargs) -> Dict:
        """
        分析特定组件
        
        Args:
            component_name: 组件名称
            dir_path: 目录路径
            **kwargs: 额外参数
                
        Returns:
            分析结果字典
        """
        logger.info(f"开始分析组件: {component_name}")
        
        if component_name == "mla":
            pattern = "multi_latent_attention"
        elif component_name == "moe":
            pattern = "moe"
        else:
            logger.error(f"未知组件: {component_name}")
            return {"error": f"未知组件: {component_name}"}
        
        python_files = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".py") and (pattern in file.lower() or pattern in root.lower()):
                    python_files.append(os.path.join(root, file))
        
        logger.info(f"找到 {len(python_files)} 个与组件 {component_name} 相关的文件")
        
        results = {}
        for file_path in python_files:
            results[file_path] = self.analyze_file(file_path, **kwargs)
        
        return results
    
    def generate_report(self, format: str = "markdown") -> str:
        """
        生成分析报告
        
        Args:
            format: 报告格式，支持"markdown"和"html"
                
        Returns:
            报告内容
        """
        if format not in ["markdown", "html"]:
            logger.warning(f"不支持的报告格式: {format}，将使用markdown")
            format = "markdown"
        
        logger.info(f"生成{format}格式报告...")
        
        if format == "markdown":
            return self._generate_markdown_report()
        else:
            return self._generate_html_report()
    
    def _generate_markdown_report(self) -> str:
        """生成Markdown格式报告"""
        report = f"# 静态分析报告\n\n"
        report += f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        total_files = len(self.results)
        total_issues = 0
        issues_by_tool = {}
        
        for file_path, file_results in self.results.items():
            for tool_name, tool_result in file_results.items():
                if "issues" in tool_result:
                    issues = tool_result["issues"]
                    total_issues += len(issues)
                    
                    if tool_name not in issues_by_tool:
                        issues_by_tool[tool_name] = 0
                    issues_by_tool[tool_name] += len(issues)
        
        report += f"## 总结\n\n"
        report += f"- 分析文件数: {total_files}\n"
        report += f"- 发现问题数: {total_issues}\n\n"
        
        if issues_by_tool:
            report += "### 各工具发现的问题\n\n"
            for tool_name, count in issues_by_tool.items():
                report += f"- {tool_name}: {count}个问题\n"
            report += "\n"
        
        report += f"## 详细结果\n\n"
        
        for file_path, file_results in self.results.items():
            relative_path = os.path.relpath(file_path, REPO_ROOT)
            report += f"### {relative_path}\n\n"
            
            file_has_issues = False
            
            for tool_name, tool_result in file_results.items():
                if "issues" in tool_result and tool_result["issues"]:
                    file_has_issues = True
                    report += f"#### {tool_name}分析结果\n\n"
                    
                    for i, issue in enumerate(tool_result["issues"], 1):
                        report += f"**问题{i}**: {issue['message']}\n"
                        if "location" in issue:
                            report += f"- 位置: {issue['location']}\n"
                        if "severity" in issue:
                            severity_map = {1: "低", 2: "中", 3: "高"}
                            severity = severity_map.get(issue['severity'], issue['severity'])
                            report += f"- 严重性: {severity}\n"
                        if "suggestion" in issue:
                            report += f"- 建议: {issue['suggestion']}\n"
                        report += "\n"
            
            if not file_has_issues:
                report += "未发现问题\n\n"
        
        report += f"## 改进建议\n\n"
        report += "根据分析结果，建议关注以下方面的改进：\n\n"
        
        report += "1. 检查张量形状兼容性，确保操作前的维度正确\n"
        report += "2. 确保分布式通信操作的参数类型正确\n"
        report += "3. 避免设备不一致问题，减少不必要的CPU-GPU数据传输\n"
        report += "4. 确保并行策略配置正确，避免潜在的死锁和通信不匹配\n\n"
        
        report_path = os.path.join(REPORTS_DIR, f"report_{self.timestamp}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        latest_report_path = os.path.join(REPORTS_DIR, "latest_report.md")
        if os.path.exists(latest_report_path):
            os.remove(latest_report_path)
        os.symlink(report_path, latest_report_path)
        
        logger.info(f"报告已保存: {report_path}")
        
        return report
    
    def _generate_html_report(self) -> str:
        """生成HTML格式报告"""
        markdown_report = self._generate_markdown_report()
        
        try:
            import markdown
            html = markdown.markdown(markdown_report, extensions=['tables', 'fenced_code'])
        except ImportError:
            logger.warning("未安装markdown模块，将使用简单HTML格式")
            html = f"<pre>{markdown_report}</pre>"
        
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>静态分析报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
                h1 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                h2 {{ color: #3498db; margin-top: 30px; }}
                h3 {{ color: #2980b9; }}
                h4 {{ color: #16a085; }}
                pre {{ background: #f8f8f8; border: 1px solid #ddd; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                code {{ background: #f8f8f8; padding: 2px 5px; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .severity-high {{ color: #e74c3c; }}
                .severity-medium {{ color: #f39c12; }}
                .severity-low {{ color: #27ae60; }}
            </style>
        </head>
        <body>
            {html}
        </body>
        </html>
        """
        
        report_path = os.path.join(REPORTS_DIR, f"report_{self.timestamp}.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_report)
        
        latest_report_path = os.path.join(REPORTS_DIR, "latest_report.html")
        if os.path.exists(latest_report_path):
            os.remove(latest_report_path)
        os.symlink(report_path, latest_report_path)
        
        logger.info(f"HTML报告已保存: {report_path}")
        
        return html_report
    
    def generate_refactoring_plan(self) -> str:
        """
        生成重构计划
        
        基于分析结果生成代码重构计划。
                
        Returns:
            重构计划内容
        """
        logger.info("生成重构计划...")
        
        plan = f"# 代码重构计划\n\n"
        plan += f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        correctness_issues = []
        performance_issues = []
        maintainability_issues = []
        
        for file_path, file_results in self.results.items():
            relative_path = os.path.relpath(file_path, REPO_ROOT)
            
            for tool_name, tool_result in file_results.items():
                if "issues" not in tool_result:
                    continue
                
                for issue in tool_result["issues"]:
                    if "shape" in issue.get("message", "").lower() or "type" in issue.get("message", "").lower():
                        correctness_issues.append({
                            "file": relative_path,
                            "message": issue.get("message", ""),
                            "location": issue.get("location", ""),
                            "tool": tool_name
                        })
                    elif "performance" in issue.get("message", "").lower() or "device" in issue.get("message", "").lower():
                        performance_issues.append({
                            "file": relative_path,
                            "message": issue.get("message", ""),
                            "location": issue.get("location", ""),
                            "tool": tool_name
                        })
                    else:
                        maintainability_issues.append({
                            "file": relative_path,
                            "message": issue.get("message", ""),
                            "location": issue.get("location", ""),
                            "tool": tool_name
                        })
        
        plan += f"## 重构优先级\n\n"
        plan += "根据分析结果，建议按以下优先级进行重构：\n\n"
        plan += "1. 正确性问题 - 影响代码功能正确性的问题\n"
        plan += "2. 性能问题 - 影响代码效率但不影响正确性的问题\n"
        plan += "3. 可维护性问题 - 影响代码可读性和未来修改难度的问题\n\n"
        
        plan += f"## 正确性问题 ({len(correctness_issues)}个)\n\n"
        if correctness_issues:
            for i, issue in enumerate(correctness_issues, 1):
                plan += f"### {i}. {issue['file']}\n\n"
                plan += f"**问题**: {issue['message']}\n"
                if issue['location']:
                    plan += f"**位置**: {issue['location']}\n"
                plan += f"**工具**: {issue['tool']}\n\n"
                plan += "**建议修复**:\n\n"
                plan += "```python\n# 待LLM提供具体修复建议\n```\n\n"
        else:
            plan += "未发现正确性问题。\n\n"
        
        plan += f"## 性能问题 ({len(performance_issues)}个)\n\n"
        if performance_issues:
            for i, issue in enumerate(performance_issues, 1):
                plan += f"### {i}. {issue['file']}\n\n"
                plan += f"**问题**: {issue['message']}\n"
                if issue['location']:
                    plan += f"**位置**: {issue['location']}\n"
                plan += f"**工具**: {issue['tool']}\n\n"
                plan += "**建议优化**:\n\n"
                plan += "```python\n# 待LLM提供具体优化建议\n```\n\n"
        else:
            plan += "未发现性能问题。\n\n"
        
        plan += f"## 可维护性问题 ({len(maintainability_issues)}个)\n\n"
        if maintainability_issues:
            for i, issue in enumerate(maintainability_issues, 1):
                plan += f"### {i}. {issue['file']}\n\n"
                plan += f"**问题**: {issue['message']}\n"
                if issue['location']:
                    plan += f"**位置**: {issue['location']}\n"
                plan += f"**工具**: {issue['tool']}\n\n"
                plan += "**建议改进**:\n\n"
                plan += "```python\n# 待LLM提供具体改进建议\n```\n\n"
        else:
            plan += "未发现可维护性问题。\n\n"
        
        plan += f"## 实施计划\n\n"
        plan += "建议按以下步骤实施重构：\n\n"
        plan += "1. 首先修复所有正确性问题，确保代码功能正确\n"
        plan += "2. 然后优化性能问题，提高代码效率\n"
        plan += "3. 最后改进可维护性问题，提高代码质量\n\n"
        plan += "对于每个修复：\n\n"
        plan += "1. 创建单独的分支\n"
        plan += "2. 实施修复\n"
        plan += "3. 运行测试确保修复有效\n"
        plan += "4. 提交代码并创建Pull Request\n"
        plan += "5. 代码审查后合并\n\n"
        
        plan_path = os.path.join(REPORTS_DIR, f"refactoring_plan_{self.timestamp}.md")
        with open(plan_path, "w", encoding="utf-8") as f:
            f.write(plan)
        
        logger.info(f"重构计划已保存: {plan_path}")
        
        return plan
    
    def save_results(self, output_path: str) -> bool:
        """
        保存分析结果为JSON文件
        
        Args:
            output_path: 输出路径
            
        Returns:
            是否成功保存
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"结果已保存: {output_path}")
            return True
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
            return False

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PyTorch分布式训练代码静态分析调度器")
    
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--file", type=str, help="要分析的文件路径")
    target_group.add_argument("--dir", type=str, help="要分析的目录路径")
    target_group.add_argument("--component", type=str, help="要分析的组件名称(mla, moe)")
    
    parser.add_argument("--component-dir", type=str, help="组件所在的目录路径，与--component一起使用")
    
    parser.add_argument("--max-depth", type=int, default=10, help="最大分析深度")
    parser.add_argument("--timeout", type=int, default=300, help="超时时间（秒）")
    parser.add_argument("--verbose", action="store_true", help="输出详细信息")
    parser.add_argument("--all", action="store_true", help="运行所有分析工具")
    
    parser.add_argument("--output", type=str, help="输出文件路径")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--format", type=str, choices=["markdown", "html"], default="markdown", help="报告格式")
    parser.add_argument("--generate-plan", action="store_true", help="生成重构计划")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    config = {}
    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
    
    orchestrator = AnalysisOrchestrator(config)
    
    analysis_options = {
        "max_depth": args.max_depth,
        "timeout": args.timeout,
        "verbose": args.verbose,
        "all": args.all
    }
    
    if args.file:
        orchestrator.analyze_file(args.file, **analysis_options)
    elif args.dir:
        orchestrator.analyze_dir(args.dir, **analysis_options)
    elif args.component:
        if not args.component_dir:
            logger.error("使用--component时必须指定--component-dir")
            return
        orchestrator.analyze_component(args.component, args.component_dir, **analysis_options)
    
    report = orchestrator.generate_report(format=args.format)
    
    if args.generate_plan:
        plan = orchestrator.generate_refactoring_plan()
    
    if args.output:
        orchestrator.save_results(args.output)
    
    print("\n=== 分析摘要 ===")
    print(f"分析文件数: {len(orchestrator.results)}")
    
    total_issues = 0
    for file_path, file_results in orchestrator.results.items():
        for tool_name, tool_result in file_results.items():
            if "issues" in tool_result:
                total_issues += len(tool_result["issues"])
    
    print(f"发现问题数: {total_issues}")
    print(f"报告已保存在: {os.path.join(REPORTS_DIR, 'latest_report.' + args.format)}")
    
    if args.generate_plan:
        print(f"重构计划已保存在: {os.path.join(REPORTS_DIR, f'refactoring_plan_{orchestrator.timestamp}.md')}")

if __name__ == "__main__":
    main()
