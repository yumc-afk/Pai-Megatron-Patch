#
#
#

"""
DeepSeek V3 静态分析工具
用于分析DeepSeek V3模型中MLA和MOE组件的实现
"""

import os
import sys
import re
import ast
import argparse
import json
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from pathlib import Path
import logging
import traceback
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dsv3_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DSV3-Analyzer")

class BaseAnalyzer:
    """分析器基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.issues = []
        self.warnings = []
        self.suggestions = []
        
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """分析文件"""
        raise NotImplementedError("子类必须实现analyze方法")
    
    def add_issue(self, message: str, line: int, severity: str = "error"):
        """添加问题"""
        self.issues.append({
            "message": message,
            "line": line,
            "severity": severity
        })
        
    def add_warning(self, message: str, line: int):
        """添加警告"""
        self.warnings.append({
            "message": message,
            "line": line
        })
        
    def add_suggestion(self, message: str, line: int):
        """添加建议"""
        self.suggestions.append({
            "message": message,
            "line": line
        })
        
    def get_results(self) -> Dict[str, Any]:
        """获取分析结果"""
        return {
            "analyzer": self.name,
            "issues": self.issues,
            "warnings": self.warnings,
            "suggestions": self.suggestions
        }

class TensorShapeAnalyzer(BaseAnalyzer):
    """分析张量形状兼容性的分析器"""
    
    def __init__(self):
        super().__init__("TensorShapeAnalyzer")
        
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """分析文件中的张量形状兼容性"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
                
            tree = ast.parse(code)
            self._analyze_node(tree, code.split('\n'))
            
            return self.get_results()
        except Exception as e:
            logger.error(f"分析文件 {file_path} 时出错: {str(e)}")
            logger.error(traceback.format_exc())
            self.add_issue(f"分析器错误: {str(e)}", 0, "error")
            return self.get_results()
    
    def _analyze_node(self, node, lines):
        """递归分析AST节点"""
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            self._check_matmul(node, lines)
        
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == 'matmul' and hasattr(node.func.value, 'id') and node.func.value.id == 'torch':
                    self._check_torch_matmul(node, lines)
                elif node.func.attr == 'transpose' or node.func.attr == 'permute':
                    self._check_transpose(node, lines)
                elif node.func.attr == 'view' or node.func.attr == 'reshape':
                    self._check_reshape(node, lines)
                elif node.func.attr == 'softmax':
                    self._check_softmax(node, lines)
        
        elif isinstance(node, ast.Subscript):
            self._check_indexing(node, lines)
        
        for child in ast.iter_child_nodes(node):
            self._analyze_node(child, lines)
    
    def _check_matmul(self, node, lines):
        """检查矩阵乘法操作"""
        line_num = node.lineno
        line = lines[line_num - 1]
        
        if 'transpose' not in line and '.t()' not in line and 'permute' not in line:
            self.add_warning("矩阵乘法操作可能需要转置以确保形状匹配", line_num)
    
    def _check_torch_matmul(self, node, lines):
        """检查torch.matmul调用"""
        line_num = node.lineno
        if len(node.args) != 2:
            self.add_issue("torch.matmul需要两个参数", line_num, "error")
    
    def _check_transpose(self, node, lines):
        """检查转置操作"""
        line_num = node.lineno
        if len(node.args) < 2:
            self.add_warning("转置操作可能缺少维度参数", line_num)
    
    def _check_reshape(self, node, lines):
        """检查reshape操作"""
        line_num = node.lineno
        has_infer_dim = False
        for arg in node.args[1:]:
            if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                has_infer_dim = True
                break
            elif isinstance(arg, ast.Constant) and arg.value == -1:
                has_infer_dim = True
                break
        
        if not has_infer_dim:
            self.add_suggestion("考虑使用-1作为维度参数以自动推断形状", line_num)
    
    def _check_softmax(self, node, lines):
        """检查softmax操作"""
        line_num = node.lineno
        has_dim = False
        for kw in node.keywords:
            if kw.arg == 'dim':
                has_dim = True
                break
        
        if not has_dim and len(node.args) < 2:
            self.add_warning("softmax操作应该指定dim参数", line_num)
    
    def _check_indexing(self, node, lines):
        """检查索引操作"""
        line_num = node.lineno
        if isinstance(node.slice, ast.Index) and isinstance(node.slice.value, ast.Num):
            self.add_suggestion("考虑添加索引边界检查以避免潜在的越界错误", line_num)

class DistributedTrainingAnalyzer(BaseAnalyzer):
    """分析分布式训练相关问题的分析器"""
    
    def __init__(self):
        super().__init__("DistributedTrainingAnalyzer")
        
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """分析文件中的分布式训练问题"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
                lines = code.split('\n')
            
            dist_import_line = 0
            for i, line in enumerate(lines):
                if re.search(r'import\s+torch\.distributed', line) or re.search(r'import\s+torch\.distributed\s+as\s+dist', line):
                    dist_import_line = i + 1
                    break
            
            has_dist_import = re.search(r'import\s+torch\.distributed', code) is not None
            has_dist_alias = re.search(r'import\s+torch\.distributed\s+as\s+dist', code) is not None
            
            has_dist_init = re.search(r'dist\.init_process_group|init_process_group', code) is not None
            has_dist_barrier = re.search(r'dist\.barrier|barrier\(', code) is not None
            has_dist_allreduce = re.search(r'dist\.all_reduce|all_reduce\(', code) is not None
            
            tp_line = 0
            pp_line = 0
            ep_line = 0
            for i, line in enumerate(lines):
                if re.search(r'tensor[_-]parallel|TP', line):
                    tp_line = i + 1
                if re.search(r'pipeline[_-]parallel|PP', line):
                    pp_line = i + 1
                if re.search(r'expert[_-]parallel|EP', line):
                    ep_line = i + 1
            
            has_tp = tp_line > 0
            has_pp = pp_line > 0
            has_ep = ep_line > 0
            
            sync_line = 0
            for i, line in enumerate(lines):
                if re.search(r'torch\.cuda\.synchronize|synchronize\(', line):
                    sync_line = i + 1
                    break
            
            has_sync = sync_line > 0
            
            if has_dist_import or has_dist_alias:
                if not has_dist_init:
                    warning_line = dist_import_line if dist_import_line > 0 else 1
                    self.add_warning("导入了分布式模块但可能没有初始化进程组", warning_line)
                
                if not has_dist_barrier and (has_tp or has_pp or has_ep):
                    warning_line = max(tp_line, pp_line, ep_line) if max(tp_line, pp_line, ep_line) > 0 else 1
                    self.add_warning("使用并行计算但可能缺少同步障碍", warning_line)
                
                if not has_dist_allreduce and has_tp:
                    warning_line = tp_line if tp_line > 0 else 1
                    self.add_warning("使用张量并行但可能缺少全局规约操作", warning_line)
            
            if (has_tp or has_pp or has_ep) and not has_sync:
                suggestion_line = max(tp_line, pp_line, ep_line) if max(tp_line, pp_line, ep_line) > 0 else 1
                self.add_suggestion("考虑在关键点添加torch.cuda.synchronize()以确保设备同步", suggestion_line)
            
            moe_line = 0
            for i, line in enumerate(lines):
                if re.search(r'moe|expert|router', line, re.IGNORECASE):
                    moe_line = i + 1
                    break
            
            if moe_line > 0:
                has_load_balance = re.search(r'load[_-]balanc|aux[_-]loss', code, re.IGNORECASE) is not None
                if not has_load_balance:
                    self.add_suggestion("MOE模型可能需要负载均衡机制", moe_line)
            
            mla_line = 0
            for i, line in enumerate(lines):
                if re.search(r'multi[_-]latent|mla', line, re.IGNORECASE):
                    mla_line = i + 1
                    break
            
            if mla_line > 0:
                has_mask = re.search(r'mask|attention_mask|causal_mask', code, re.IGNORECASE) is not None
                if not has_mask:
                    self.add_warning("多潜在注意力机制可能需要注意力掩码", mla_line)
            
            return self.get_results()
        except Exception as e:
            logger.error(f"分析文件 {file_path} 时出错: {str(e)}")
            logger.error(traceback.format_exc())
            self.add_issue(f"分析器错误: {str(e)}", 0, "error")
            return self.get_results()

class MemoryOptimizationAnalyzer(BaseAnalyzer):
    """分析内存使用和优化的分析器"""
    
    def __init__(self):
        super().__init__("MemoryOptimizationAnalyzer")
        
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """分析文件中的内存使用和优化问题"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
                lines = code.split('\n')
            
            for i, line in enumerate(lines):
                if re.search(r'torch\.(rand|randn|zeros|ones|empty)', line):
                    size_match = re.search(r'(\d+)\s*,\s*(\d+)', line)
                    if size_match:
                        size1 = int(size_match.group(1))
                        size2 = int(size_match.group(2))
                        if size1 * size2 > 1000000:  # 超过百万元素
                            self.add_warning(f"创建了大型张量 ({size1}x{size2})，考虑内存优化", i+1)
                
                if '.clone()' in line or '.copy()' in line or '.detach()' in line:
                    self.add_suggestion(f"检测到张量复制操作，考虑是否必要", i+1)
                
                if 'with' in line and 'torch.no_grad' not in line and ('forward' in line or 'inference' in line):
                    self.add_suggestion(f"推理代码可以使用with torch.no_grad()减少内存使用", i+1)
                
                if '.backward(' in line and 'retain_graph=True' in line:
                    self.add_warning(f"使用retain_graph=True可能导致内存泄漏", i+1)
            
            has_fp16 = re.search(r'torch\.(float16|half)|amp|autocast|fp16', code) is not None
            if not has_fp16 and re.search(r'train|optimizer', code) is not None:
                self.add_suggestion("考虑使用混合精度训练(FP16/BF16)以减少内存使用并提高性能", 1)
            
            has_checkpoint = re.search(r'checkpoint|recompute|activation_checkpoint', code) is not None
            if not has_checkpoint and re.search(r'train|backward', code) is not None:
                self.add_suggestion("考虑使用梯度检查点(checkpoint)以减少内存使用", 1)
            
            return self.get_results()
        except Exception as e:
            logger.error(f"分析文件 {file_path} 时出错: {str(e)}")
            logger.error(traceback.format_exc())
            self.add_issue(f"分析器错误: {str(e)}", 0, "error")
            return self.get_results()

class MLAMOEAnalyzer(BaseAnalyzer):
    """专门分析MLA和MOE组件的分析器"""
    
    def __init__(self):
        super().__init__("MLAMOEAnalyzer")
        
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """分析文件中的MLA和MOE相关问题"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
                lines = code.split('\n')
            
            self._analyze_mla(code, lines)
            
            self._analyze_moe(code, lines)
            
            return self.get_results()
        except Exception as e:
            logger.error(f"分析文件 {file_path} 时出错: {str(e)}")
            logger.error(traceback.format_exc())
            self.add_issue(f"分析器错误: {str(e)}", 0, "error")
            return self.get_results()
    
    def _analyze_mla(self, code, lines):
        """分析MLA相关代码"""
        mla_pattern = re.compile(r'multi[_-]latent|mla|attention', re.IGNORECASE)
        mla_lines = []
        
        for i, line in enumerate(lines):
            if mla_pattern.search(line):
                mla_lines.append((i+1, line))
        
        if not mla_lines:
            return
        
        for line_num, line in mla_lines:
            if 'attention_score' in line or 'attention_weight' in line:
                if 'scale' not in line and 'sqrt' not in line:
                    self.add_warning("注意力分数计算可能缺少缩放因子", line_num)
            
            if 'mask' in line and ('add' not in line and '+' not in line):
                self.add_warning("注意力掩码应该通过加法应用", line_num)
            
            if 'softmax' in line and 'dim' not in line:
                self.add_warning("softmax操作应该指定dim参数", line_num)
            
            if 'head' in line and ('reshape' not in line and 'view' not in line and 'transpose' not in line):
                self.add_suggestion("多头注意力可能需要reshape或transpose操作", line_num)
        
        rope_line = 0
        for i, line in enumerate(lines):
            if re.search(r'rotary|rope|旋转', line, re.IGNORECASE):
                rope_line = i + 1
                break
                
        has_rope = rope_line > 0
        if has_rope:
            has_trig = re.search(r'sin|cos|三角|旋转', code, re.IGNORECASE) is not None
            if not has_trig:
                self.add_warning("使用旋转位置编码但可能缺少三角函数计算", rope_line)
    
    def _analyze_moe(self, code, lines):
        """分析MOE相关代码"""
        has_load_balance = re.search(r'load[_-]balanc|aux[_-]loss|balance_loss|router_z_loss', code, re.IGNORECASE) is not None
        
        if has_load_balance:
            has_router = False
            router_line = 0
            has_topk = False
            has_dispatch = False
            dispatch_line = 0
            has_combine = False
            
            for line_num, line in enumerate(lines):
                if 'router' in line.lower():
                    has_router = True
                    router_line = line_num + 1
                    if 'softmax' in line and 'sigmoid' not in line:
                        self.add_suggestion("考虑使用sigmoid而非softmax作为路由器激活函数", line_num + 1)
                
                if 'topk' in line.lower():
                    has_topk = True
                
                if 'dispatch' in line.lower():
                    has_dispatch = True
                    dispatch_line = line_num + 1
                
                if 'combine' in line.lower() or ('sum' in line.lower() and 'expert' in line.lower()):
                    has_combine = True
            
            if has_router and not has_topk:
                self.add_warning("MOE路由器可能缺少topk选择", router_line)
            
            if has_router and not has_dispatch:
                self.add_warning("MOE可能缺少分发机制", router_line)
            
            if has_dispatch and not has_combine:
                self.add_warning("MOE可能缺少专家输出组合机制", dispatch_line)
            
            return
        
        moe_pattern = re.compile(r'moe|expert|router|dispatch', re.IGNORECASE)
        first_moe_line = 0
        
        for i, line in enumerate(lines):
            if moe_pattern.search(line):
                first_moe_line = i + 1
                break
        
        if first_moe_line > 0:
            self.add_warning("MOE应该包含负载均衡机制", first_moe_line)
        
        has_router = False
        router_line = 0
        has_topk = False
        has_dispatch = False
        dispatch_line = 0
        has_combine = False
        
        for line_num, line in enumerate(lines):
            if 'router' in line.lower():
                has_router = True
                router_line = line_num + 1
                if 'softmax' in line and 'sigmoid' not in line:
                    self.add_suggestion("考虑使用sigmoid而非softmax作为路由器激活函数", line_num + 1)
            
            if 'topk' in line.lower():
                has_topk = True
            
            if 'dispatch' in line.lower():
                has_dispatch = True
                dispatch_line = line_num + 1
            
            if 'combine' in line.lower() or ('sum' in line.lower() and 'expert' in line.lower()):
                has_combine = True
        
        if has_router and not has_topk:
            self.add_warning("MOE路由器可能缺少topk选择", router_line)
        
        if has_router and not has_dispatch:
            self.add_warning("MOE可能缺少分发机制", router_line)
        
        if has_dispatch and not has_combine:
            self.add_warning("MOE可能缺少专家输出组合机制", dispatch_line)

class DSV3Analyzer:
    """DeepSeek V3分析器主类"""
    
    def __init__(self):
        self.analyzers = [
            TensorShapeAnalyzer(),
            DistributedTrainingAnalyzer(),
            MemoryOptimizationAnalyzer(),
            MLAMOEAnalyzer()
        ]
        self.results = {}
        
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """分析单个文件"""
        logger.info(f"分析文件: {file_path}")
        
        file_results = {}
        for analyzer in self.analyzers:
            logger.info(f"运行分析器: {analyzer.name}")
            try:
                result = analyzer.analyze(file_path)
                file_results[analyzer.name] = result
            except Exception as e:
                logger.error(f"分析器 {analyzer.name} 运行失败: {str(e)}")
                logger.error(traceback.format_exc())
        
        self.results[file_path] = file_results
        return file_results
    
    def analyze_directory(self, dir_path: str, pattern: str = "*.py") -> Dict[str, Dict[str, Any]]:
        """分析目录中的所有文件"""
        logger.info(f"分析目录: {dir_path}, 模式: {pattern}")
        
        path = Path(dir_path)
        files = list(path.glob(pattern))
        
        for file_path in files:
            self.analyze_file(str(file_path))
        
        return self.results
    
    def generate_report(self, output_path: str = None) -> str:
        """生成分析报告"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"dsv3_analysis_report_{timestamp}.md"
        
        logger.info(f"生成报告: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# DeepSeek V3 静态分析报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            total_issues = 0
            total_warnings = 0
            total_suggestions = 0
            
            for file_path, analyzers_results in self.results.items():
                f.write(f"## 文件: {os.path.basename(file_path)}\n\n")
                
                file_issues = 0
                file_warnings = 0
                file_suggestions = 0
                
                file_warning_types = {}
                file_suggestion_types = {}
                
                for analyzer_name, result in analyzers_results.items():
                    f.write(f"### {analyzer_name}\n\n")
                    
                    if result["issues"]:
                        f.write("#### 问题\n\n")
                        for issue in result["issues"]:
                            f.write(f"- **[{issue['severity']}]** 第{issue['line']}行: {issue['message']}\n")
                        f.write("\n")
                        file_issues += len(result["issues"])
                    
                    if result["warnings"]:
                        f.write("#### 警告\n\n")
                        
                        unique_warnings = []
                        for warning in result["warnings"]:
                            message = warning['message']
                            warning_type = None
                            
                            if "MOE应该包含负载均衡机制" in message:
                                warning_type = "MOE负载均衡"
                            elif "MOE可能缺少分发机制" in message:
                                warning_type = "MOE分发机制"
                            elif "MOE路由器可能缺少topk选择" in message:
                                warning_type = "MOE路由器topk"
                            elif "注意力分数计算可能缺少缩放因子" in message:
                                warning_type = "注意力缩放因子"
                            elif "注意力掩码应该通过加法应用" in message:
                                warning_type = "注意力掩码应用"
                            elif "softmax操作应该指定dim参数" in message:
                                warning_type = "softmax维度"
                            elif "使用旋转位置编码但可能缺少三角函数计算" in message:
                                warning_type = "旋转位置编码"
                            else:
                                warning_type = message  # 其他类型的警告使用完整消息作为类型
                            
                            if warning_type not in file_warning_types:
                                file_warning_types[warning_type] = warning
                                unique_warnings.append(warning)
                        
                        unique_warnings.sort(key=lambda x: x['line'])
                        
                        for warning in unique_warnings:
                            f.write(f"- 第{warning['line']}行: {warning['message']}\n")
                        f.write("\n")
                        file_warnings += len(unique_warnings)
                    
                    if result["suggestions"]:
                        f.write("#### 建议\n\n")
                        
                        unique_suggestions = []
                        for suggestion in result["suggestions"]:
                            message = suggestion['message']
                            suggestion_type = None
                            
                            if "多头注意力可能需要reshape或transpose操作" in message:
                                suggestion_type = "多头注意力reshape"
                                if suggestion_type in file_suggestion_types and len(file_suggestion_types[suggestion_type]) >= 3:
                                    continue
                                if suggestion_type not in file_suggestion_types:
                                    file_suggestion_types[suggestion_type] = []
                                file_suggestion_types[suggestion_type].append(suggestion)
                                unique_suggestions.append(suggestion)
                            elif "考虑使用-1作为维度参数以自动推断形状" in message:
                                suggestion_type = "自动推断形状"
                                if suggestion_type in file_suggestion_types and len(file_suggestion_types[suggestion_type]) >= 3:
                                    continue
                                if suggestion_type not in file_suggestion_types:
                                    file_suggestion_types[suggestion_type] = []
                                file_suggestion_types[suggestion_type].append(suggestion)
                                unique_suggestions.append(suggestion)
                            elif "考虑使用梯度检查点(checkpoint)以减少内存使用" in message:
                                suggestion_type = "梯度检查点"
                                if suggestion_type in file_suggestion_types:
                                    continue
                                file_suggestion_types[suggestion_type] = [suggestion]
                                unique_suggestions.append(suggestion)
                            elif "检测到张量复制操作，考虑是否必要" in message:
                                suggestion_type = "张量复制"
                                if suggestion_type in file_suggestion_types and len(file_suggestion_types[suggestion_type]) >= 3:
                                    continue
                                if suggestion_type not in file_suggestion_types:
                                    file_suggestion_types[suggestion_type] = []
                                file_suggestion_types[suggestion_type].append(suggestion)
                                unique_suggestions.append(suggestion)
                            else:
                                suggestion_type = message  # 其他类型的建议使用完整消息作为类型
                                if suggestion_type not in file_suggestion_types:
                                    file_suggestion_types[suggestion_type] = [suggestion]
                                    unique_suggestions.append(suggestion)
                        
                        unique_suggestions.sort(key=lambda x: x['line'])
                        
                        for suggestion in unique_suggestions:
                            f.write(f"- 第{suggestion['line']}行: {suggestion['message']}\n")
                        f.write("\n")
                        file_suggestions += len(unique_suggestions)
                
                f.write(f"### 文件统计\n\n")
                f.write(f"- 问题: {file_issues}\n")
                f.write(f"- 警告: {file_warnings}\n")
                f.write(f"- 建议: {file_suggestions}\n\n")
                
                total_issues += file_issues
                total_warnings += file_warnings
                total_suggestions += file_suggestions
            
            f.write("## 总体统计\n\n")
            f.write(f"- 分析文件数: {len(self.results)}\n")
            f.write(f"- 总问题数: {total_issues}\n")
            f.write(f"- 总警告数: {total_warnings}\n")
            f.write(f"- 总建议数: {total_suggestions}\n")
        
        logger.info(f"报告已生成: {output_path}")
        return output_path
    
    def generate_json_report(self, output_path: str = None) -> str:
        """生成JSON格式的分析报告"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"dsv3_analysis_report_{timestamp}.json"
        
        logger.info(f"生成JSON报告: {output_path}")
        
        total_issues = 0
        total_warnings = 0
        total_suggestions = 0
        
        for file_path, analyzers_results in self.results.items():
            for analyzer_name, result in analyzers_results.items():
                total_issues += len(result["issues"])
                total_warnings += len(result["warnings"])
                total_suggestions += len(result["suggestions"])
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "files_analyzed": len(self.results),
                "total_issues": total_issues,
                "total_warnings": total_warnings,
                "total_suggestions": total_suggestions
            },
            "results": self.results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON报告已生成: {output_path}")
        return output_path

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DeepSeek V3 静态分析工具")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', type=str, help='要分析的文件路径')
    group.add_argument('--dir', type=str, help='要分析的目录路径')
    
    parser.add_argument('--pattern', type=str, default="*.py", help='文件匹配模式，默认为*.py')
    parser.add_argument('--output', type=str, help='报告输出路径')
    parser.add_argument('--format', type=str, choices=['md', 'json'], default='md', help='报告格式，默认为md')
    parser.add_argument('--verbose', action='store_true', help='显示详细日志')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    analyzer = DSV3Analyzer()
    
    start_time = time.time()
    
    if args.file:
        analyzer.analyze_file(args.file)
    elif args.dir:
        analyzer.analyze_directory(args.dir, args.pattern)
    
    if args.format == 'md':
        report_path = analyzer.generate_report(args.output)
    else:
        report_path = analyzer.generate_json_report(args.output)
    
    end_time = time.time()
    
    logger.info(f"分析完成，耗时: {end_time - start_time:.2f}秒")
    logger.info(f"报告已保存至: {report_path}")

if __name__ == "__main__":
    main()
