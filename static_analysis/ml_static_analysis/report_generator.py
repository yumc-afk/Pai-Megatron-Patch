#
#
#

"""
报告生成器模块，用于生成静态分析报告
"""

import os
import json
import datetime
import logging
from typing import Dict, List, Any, Optional, Union

from ml_static_analysis.core.config import AnalysisConfig
from ml_static_analysis.core.report import AnalysisReport

logger = logging.getLogger(__name__)

class ReportGenerator:
    """静态分析报告生成器"""
    
    def __init__(self, config: AnalysisConfig):
        """
        初始化报告生成器
        
        Args:
            config: 分析配置
        """
        self.config = config
        self.reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def generate_report(self, analysis_results: Dict[str, AnalysisReport], 
                        target_path: str, 
                        format: str = "markdown") -> str:
        """
        生成分析报告
        
        Args:
            analysis_results: 分析结果字典，键为分析器名称，值为分析报告
            target_path: 分析目标路径
            format: 报告格式，支持 "markdown", "json", "html"
            
        Returns:
            str: 报告文件路径
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"analysis_report_{timestamp}.{format}"
        report_path = os.path.join(self.reports_dir, report_filename)
        
        latest_report_path = os.path.join(self.reports_dir, f"latest_report.{format}")
        
        if format == "markdown":
            content = self._generate_markdown_report(analysis_results, target_path)
        elif format == "json":
            content = self._generate_json_report(analysis_results, target_path)
        elif format == "html":
            content = self._generate_html_report(analysis_results, target_path)
        else:
            raise ValueError(f"不支持的报告格式: {format}")
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        if os.path.exists(latest_report_path):
            os.remove(latest_report_path)
        
        try:
            os.symlink(report_path, latest_report_path)
        except Exception as e:
            logger.warning(f"创建符号链接失败: {e}")
            import shutil
            shutil.copy2(report_path, latest_report_path)
        
        logger.info(f"报告已生成: {report_path}")
        return report_path
    
    def _generate_markdown_report(self, analysis_results: Dict[str, AnalysisReport], target_path: str) -> str:
        """生成Markdown格式的报告"""
        lines = []
        
        lines.append("# PyTorch LLM 分布式训练静态分析报告")
        lines.append("")
        
        lines.append("## 分析元数据")
        lines.append("")
        lines.append(f"- **分析时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **分析目标**: `{target_path}`")
        lines.append(f"- **分析器**: {', '.join(analysis_results.keys())}")
        lines.append("")
        
        lines.append("## 分析摘要")
        lines.append("")
        
        total_warnings = sum(len(report.get_warnings()) for report in analysis_results.values())
        total_errors = sum(len(report.get_errors()) for report in analysis_results.values())
        total_suggestions = sum(len(report.get_suggestions()) for report in analysis_results.values())
        
        lines.append(f"- **总警告数**: {total_warnings}")
        lines.append(f"- **总错误数**: {total_errors}")
        lines.append(f"- **总建议数**: {total_suggestions}")
        lines.append("")
        
        for analyzer_name, report in analysis_results.items():
            lines.append(f"## {analyzer_name} 分析结果")
            lines.append("")
            
            if report.summary:
                lines.append(f"### {analyzer_name} 摘要")
                lines.append("")
                lines.append(str(report.summary))
                lines.append("")
            
            errors = report.get_errors()
            if errors:
                lines.append(f"### {analyzer_name} 错误 ({len(errors)})")
                lines.append("")
                for error in errors:
                    code = error.get("code", "")
                    message = error.get("message", "")
                    file_path = error.get("file_path", "")
                    line = error.get("line", 0)
                    
                    lines.append(f"- **{code}**: {message}")
                    if file_path:
                        lines.append(f"  - 文件: `{file_path}`")
                    if line:
                        lines.append(f"  - 行号: {line}")
                    lines.append("")
            
            warnings = report.get_warnings()
            if warnings:
                lines.append(f"### {analyzer_name} 警告 ({len(warnings)})")
                lines.append("")
                for warning in warnings:
                    code = warning.get("code", "")
                    message = warning.get("message", "")
                    file_path = warning.get("file_path", "")
                    line = warning.get("line", 0)
                    
                    lines.append(f"- **{code}**: {message}")
                    if file_path:
                        lines.append(f"  - 文件: `{file_path}`")
                    if line:
                        lines.append(f"  - 行号: {line}")
                    lines.append("")
            
            suggestions = report.get_suggestions()
            if suggestions:
                lines.append(f"### {analyzer_name} 建议 ({len(suggestions)})")
                lines.append("")
                for suggestion in suggestions:
                    code = suggestion.get("code", "")
                    message = suggestion.get("message", "")
                    file_path = suggestion.get("file_path", "")
                    line = suggestion.get("line", 0)
                    
                    lines.append(f"- **{code}**: {message}")
                    if file_path:
                        lines.append(f"  - 文件: `{file_path}`")
                    if line:
                        lines.append(f"  - 行号: {line}")
                    lines.append("")
        
        if hasattr(self.config, 'autofix_enabled') and self.config.autofix_enabled:
            lines.append("## 自动修复建议")
            lines.append("")
            
            autofix_count = 0
            for report in analysis_results.values():
                for item in report.get_errors():
                    if item.get("can_autofix", False) or item.get("severity", "") in ["error", "warning"]:
                        autofix_count += 1
                for item in report.get_warnings():
                    if item.get("can_autofix", False) or item.get("severity", "") in ["error", "warning"]:
                        autofix_count += 1
            
            if autofix_count > 0:
                lines.append(f"发现 {autofix_count} 个可能自动修复的问题。")
                lines.append("运行以下命令应用自动修复:")
                lines.append("")
                lines.append("```bash")
                lines.append(f"ml-analyze --target {target_path} --autofix")
                lines.append("```")
                lines.append("")
                lines.append("或者运行以下命令查看修复建议而不应用:")
                lines.append("")
                lines.append("```bash")
                lines.append(f"ml-analyze --target {target_path} --autofix-dry-run")
                lines.append("```")
            else:
                lines.append("未发现可自动修复的问题。")
            
            lines.append("")
        
        lines.append("## 结论和建议")
        lines.append("")
        
        if total_errors > 0:
            lines.append("### 关键问题")
            lines.append("")
            lines.append("以下是需要优先解决的关键问题:")
            lines.append("")
            
            for analyzer_name, report in analysis_results.items():
                for error in report.get_errors():
                    code = error.get("code", "")
                    message = error.get("message", "")
                    file_path = error.get("file_path", "")
                    line = error.get("line", 0)
                    
                    lines.append(f"- [{analyzer_name}] **{code}**: {message}")
                    if file_path:
                        lines.append(f"  - 文件: `{file_path}`")
                    if line:
                        lines.append(f"  - 行号: {line}")
                    lines.append("")
        
        performance_suggestions = []
        for analyzer_name, report in analysis_results.items():
            for suggestion in report.get_suggestions():
                tags = suggestion.get("tags", [])
                if isinstance(tags, str):
                    tags = [tags]
                if "performance" in tags or "memory" in tags:
                    performance_suggestions.append((analyzer_name, suggestion))
        
        if performance_suggestions:
            lines.append("### 性能优化建议")
            lines.append("")
            for analyzer_name, suggestion in performance_suggestions:
                code = suggestion.get("code", "")
                message = suggestion.get("message", "")
                file_path = suggestion.get("file_path", "")
                line = suggestion.get("line", 0)
                
                lines.append(f"- [{analyzer_name}] **{code}**: {message}")
                if file_path:
                    lines.append(f"  - 文件: `{file_path}`")
                if line:
                    lines.append(f"  - 行号: {line}")
                lines.append("")
        
        return "\n".join(lines)
    
    def _generate_json_report(self, analysis_results: Dict[str, AnalysisReport], target_path: str) -> str:
        """生成JSON格式的报告"""
        report_data = {
            "metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "target_path": target_path,
                "analyzers": list(analysis_results.keys())
            },
            "summary": {
                "total_warnings": sum(len(report.get_warnings()) for report in analysis_results.values()),
                "total_errors": sum(len(report.get_errors()) for report in analysis_results.values()),
                "total_suggestions": sum(len(report.get_suggestions()) for report in analysis_results.values())
            },
            "results": {}
        }
        
        for analyzer_name, report in analysis_results.items():
            report_data["results"][analyzer_name] = report.to_dict()
        
        return json.dumps(report_data, ensure_ascii=False, indent=2)
    
    def _generate_html_report(self, analysis_results: Dict[str, AnalysisReport], target_path: str) -> str:
        """生成HTML格式的报告"""
        
        html_parts = []
        
        html_parts.append("""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>PyTorch LLM 分布式训练静态分析报告</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #0366d6;
        }
        code {
            background-color: #f6f8fa;
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
        }
        pre {
            background-color: #f6f8fa;
            padding: 16px;
            border-radius: 3px;
            overflow: auto;
        }
        .error {
            color: #d73a49;
            font-weight: bold;
        }
        .warning {
            color: #e36209;
            font-weight: bold;
        }
        .suggestion {
            color: #0366d6;
        }
        .finding-item {
            margin-bottom: 15px;
            padding: 10px;
            border-left: 4px solid #dfe2e5;
        }
        .error-item {
            border-left-color: #d73a49;
            background-color: #ffeef0;
        }
        .warning-item {
            border-left-color: #e36209;
            background-color: #fff5e8;
        }
        .suggestion-item {
            border-left-color: #0366d6;
            background-color: #f1f8ff;
        }
        .file-path {
            font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
            color: #6a737d;
        }
        .summary-box {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .summary-item {
            flex: 1;
            text-align: center;
            padding: 10px;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            margin: 0 5px;
        }
        .summary-item.errors {
            background-color: #ffeef0;
        }
        .summary-item.warnings {
            background-color: #fff5e8;
        }
        .summary-item.suggestions {
            background-color: #f1f8ff;
        }
        .summary-count {
            font-size: 24px;
            font-weight: bold;
        }
    </style>
</head>
<body>
""")
        
        html_parts.append("<h1>PyTorch LLM 分布式训练静态分析报告</h1>")
        
        html_parts.append("<h2>分析元数据</h2>")
        html_parts.append("<ul>")
        html_parts.append(f"<li><strong>分析时间</strong>: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>")
        html_parts.append(f"<li><strong>分析目标</strong>: {target_path}</li>")
        html_parts.append(f"<li><strong>分析器</strong>: {', '.join(analysis_results.keys())}</li>")
        html_parts.append("</ul>")
        
        total_warnings = sum(len(report.get_warnings()) for report in analysis_results.values())
        total_errors = sum(len(report.get_errors()) for report in analysis_results.values())
        total_suggestions = sum(len(report.get_suggestions()) for report in analysis_results.values())
        
        html_parts.append("<h2>分析摘要</h2>")
        html_parts.append('<div class="summary-box">')
        html_parts.append(f'<div class="summary-item errors"><div class="summary-count">{total_errors}</div><div>错误</div></div>')
        html_parts.append(f'<div class="summary-item warnings"><div class="summary-count">{total_warnings}</div><div>警告</div></div>')
        html_parts.append(f'<div class="summary-item suggestions"><div class="summary-count">{total_suggestions}</div><div>建议</div></div>')
        html_parts.append('</div>')
        
        for analyzer_name, report in analysis_results.items():
            html_parts.append(f'<div class="analyzer-section">')
            html_parts.append(f'<h2>{analyzer_name} 分析结果</h2>')
            
            if report.summary:
                html_parts.append("<h3>摘要</h3>")
                html_parts.append(f"<pre>{report.summary}</pre>")
            
            errors = report.get_errors()
            if errors:
                html_parts.append(f"<h3>错误 ({len(errors)})</h3>")
                for error in errors:
                    code = error.get("code", "")
                    message = error.get("message", "")
                    file_path = error.get("file_path", "")
                    line = error.get("line", 0)
                    content = error.get("content", "")
                    
                    html_parts.append('<div class="finding-item error-item">')
                    html_parts.append(f'<div class="error"><strong>{code}</strong>: {message}</div>')
                    if file_path:
                        html_parts.append(f'<div class="file-path">文件: {file_path}</div>')
                    if line:
                        html_parts.append(f'<div>行号: {line}</div>')
                    if content:
                        html_parts.append(f'<pre><code>{content}</code></pre>')
                    html_parts.append('</div>')
            
            warnings = report.get_warnings()
            if warnings:
                html_parts.append(f"<h3>警告 ({len(warnings)})</h3>")
                for warning in warnings:
                    code = warning.get("code", "")
                    message = warning.get("message", "")
                    file_path = warning.get("file_path", "")
                    line = warning.get("line", 0)
                    content = warning.get("content", "")
                    
                    html_parts.append('<div class="finding-item warning-item">')
                    html_parts.append(f'<div class="warning"><strong>{code}</strong>: {message}</div>')
                    if file_path:
                        html_parts.append(f'<div class="file-path">文件: {file_path}</div>')
                    if line:
                        html_parts.append(f'<div>行号: {line}</div>')
                    if content:
                        html_parts.append(f'<pre><code>{content}</code></pre>')
                    html_parts.append('</div>')
            
            suggestions = report.get_suggestions()
            if suggestions:
                html_parts.append(f"<h3>建议 ({len(suggestions)})</h3>")
                for suggestion in suggestions:
                    code = suggestion.get("code", "")
                    message = suggestion.get("message", "")
                    file_path = suggestion.get("file_path", "")
                    line = suggestion.get("line", 0)
                    content = suggestion.get("content", "")
                    
                    html_parts.append('<div class="finding-item suggestion-item">')
                    html_parts.append(f'<div class="suggestion"><strong>{code}</strong>: {message}</div>')
                    if file_path:
                        html_parts.append(f'<div class="file-path">文件: {file_path}</div>')
                    if line:
                        html_parts.append(f'<div>行号: {line}</div>')
                    if content:
                        html_parts.append(f'<pre><code>{content}</code></pre>')
                    html_parts.append('</div>')
            
            html_parts.append('</div>')  # 结束analyzer-section
        
        html_parts.append("<h2>结论和建议</h2>")
        
        if total_errors > 0:
            html_parts.append("<h3>关键问题</h3>")
            html_parts.append("<p>以下是需要优先解决的关键问题:</p>")
            
            for analyzer_name, report in analysis_results.items():
                for error in report.get_errors():
                    code = error.get("code", "")
                    message = error.get("message", "")
                    file_path = error.get("file_path", "")
                    line = error.get("line", 0)
                    
                    html_parts.append('<div class="finding-item error-item">')
                    html_parts.append(f'<div class="error">[{analyzer_name}] <strong>{code}</strong>: {message}</div>')
                    if file_path:
                        html_parts.append(f'<div class="file-path">文件: {file_path}</div>')
                    if line:
                        html_parts.append(f'<div>行号: {line}</div>')
                    html_parts.append('</div>')
        
        autofix_items = []
        for analyzer_name, report in analysis_results.items():
            for item in report.get_errors() + report.get_warnings():
                if item.get("can_autofix", False):
                    autofix_items.append((analyzer_name, item))
        
        if autofix_items:
            html_parts.append("<h3>自动修复</h3>")
            html_parts.append("<p>以下问题可以通过自动修复工具解决:</p>")
            
            for analyzer_name, item in autofix_items[:5]:  # 只显示前5个
                message = item.get("message", "")
                file_path = item.get("file_path", "")
                line = item.get("line", 0)
                
                html_parts.append('<div class="finding-item suggestion-item">')
                html_parts.append(f'<div class="suggestion">{message} ({file_path}:{line})</div>')
                html_parts.append('</div>')
            
            if len(autofix_items) > 5:
                html_parts.append(f"<p>... 以及 {len(autofix_items) - 5} 个其他可自动修复的问题</p>")
        
        html_parts.append("</body></html>")
        
        return "\n".join(html_parts)
