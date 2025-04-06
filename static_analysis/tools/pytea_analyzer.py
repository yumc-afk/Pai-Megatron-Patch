#
#
#

"""
PyTea张量形状分析工具

此模块提供了基于PyTea的张量形状分析功能，可以检测PyTorch代码中潜在的张量形状不匹配错误。
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Optional, Set, Tuple, Union, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pytea_analyzer")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.dirname(CURRENT_DIR)
REPO_ROOT = os.path.dirname(TOOLS_DIR)

try:
    import pytea
    from pytea.pyteaconfig import PyteaConfig
    from pytea.analysis import analyze_file
    HAS_PYTEA = True
except ImportError:
    logger.warning("无法导入PyTea，将使用模拟模式")
    HAS_PYTEA = False

class PyTeaAnalyzer:
    """
    PyTea张量形状分析器
    
    使用PyTea静态分析工具检测PyTorch代码中潜在的张量形状不匹配错误。
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化PyTea分析器
        
        Args:
            config: 配置字典，可选
        """
        self.config = config or {}
        self.results = {}
        
        self.has_pytea = HAS_PYTEA
        if not self.has_pytea:
            logger.warning("PyTea未安装，将使用模拟模式")
    
    def analyze_file(self, file_path: str, **kwargs) -> Dict:
        """
        分析单个文件
        
        Args:
            file_path: 文件路径
            **kwargs: 额外参数
                - max_depth: 最大分析深度
                - timeout: 超时时间（秒）
                - verbose: 是否输出详细信息
                
        Returns:
            分析结果字典
        """
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return {"error": f"文件不存在: {file_path}"}
        
        if not file_path.endswith(".py"):
            logger.warning(f"非Python文件: {file_path}")
            return {"warning": f"非Python文件: {file_path}"}
        
        logger.info(f"开始分析文件: {file_path}")
        
        max_depth = kwargs.get("max_depth", 10)
        timeout = kwargs.get("timeout", 300)
        verbose = kwargs.get("verbose", False)
        
        if self.has_pytea:
            return self._analyze_with_pytea(file_path, max_depth, timeout, verbose)
        else:
            return self._mock_analyze(file_path, max_depth, timeout, verbose)
    
    def _analyze_with_pytea(self, file_path: str, max_depth: int, timeout: int, verbose: bool) -> Dict:
        """使用PyTea分析文件"""
        try:
            config = PyteaConfig(
                max_depth=max_depth,
                timeout=timeout,
                verbose=verbose
            )
            
            logger.info(f"使用PyTea分析文件: {file_path}")
            result = analyze_file(file_path, config)
            
            issues = []
            for path in result.invalid_paths:
                for error in path.errors:
                    issue = {
                        "message": error.message,
                        "location": f"{file_path}:{error.line}",
                        "severity": 2,  # 高严重度
                        "suggestion": self._generate_suggestion(error),
                        "analyzer": "pytea"
                    }
                    issues.append(issue)
            
            self.results[file_path] = {
                "status": "success",
                "issues": issues,
                "valid_paths": len(result.valid_paths),
                "invalid_paths": len(result.invalid_paths),
                "unknown_paths": len(result.unknown_paths)
            }
            
            logger.info(f"分析完成，发现 {len(issues)} 个问题")
            return self.results[file_path]
        
        except Exception as e:
            logger.error(f"PyTea分析失败: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _mock_analyze(self, file_path: str, max_depth: int, timeout: int, verbose: bool) -> Dict:
        """模拟PyTea分析"""
        logger.info(f"使用模拟模式分析文件: {file_path}")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"读取文件失败: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
        
        issues = []
        
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if "torch.matmul" in line or "@" in line or "torch.mm" in line:
                if verbose:
                    logger.debug(f"发现矩阵乘法: {line}")
                
                if "# shape mismatch" in line:
                    issue = {
                        "message": "可能的矩阵乘法形状不匹配",
                        "location": f"{file_path}:{i+1}",
                        "severity": 2,
                        "suggestion": "确保矩阵乘法的两个张量形状兼容，即A的列数等于B的行数",
                        "analyzer": "pytea"
                    }
                    issues.append(issue)
            
            if "+" in line or "-" in line or "*" in line or "/" in line:
                if verbose:
                    logger.debug(f"发现广播操作: {line}")
                
                if "# broadcast error" in line:
                    issue = {
                        "message": "可能的广播操作形状不匹配",
                        "location": f"{file_path}:{i+1}",
                        "severity": 2,
                        "suggestion": "确保广播操作的张量形状兼容，参考PyTorch的广播规则",
                        "analyzer": "pytea"
                    }
                    issues.append(issue)
            
            if "[" in line and "]" in line and "torch" in line:
                if verbose:
                    logger.debug(f"发现索引操作: {line}")
                
                if "# index out of bounds" in line:
                    issue = {
                        "message": "可能的索引越界",
                        "location": f"{file_path}:{i+1}",
                        "severity": 2,
                        "suggestion": "确保索引在张量的有效范围内",
                        "analyzer": "pytea"
                    }
                    issues.append(issue)
            
            if "view" in line or "reshape" in line or "permute" in line:
                if verbose:
                    logger.debug(f"发现维度操作: {line}")
                
                if "# dimension mismatch" in line:
                    issue = {
                        "message": "可能的维度操作形状不匹配",
                        "location": f"{file_path}:{i+1}",
                        "severity": 2,
                        "suggestion": "确保维度操作前后的元素总数保持不变",
                        "analyzer": "pytea"
                    }
                    issues.append(issue)
        
        self.results[file_path] = {
            "status": "success",
            "issues": issues,
            "valid_paths": 1,  # 模拟值
            "invalid_paths": len(issues),
            "unknown_paths": 0
        }
        
        logger.info(f"模拟分析完成，发现 {len(issues)} 个问题")
        return self.results[file_path]
    
    def _generate_suggestion(self, error: Any) -> str:
        """根据错误生成建议"""
        if "shape mismatch" in str(error):
            return "检查张量的形状是否兼容，确保操作前的维度正确"
        elif "broadcast" in str(error):
            return "检查广播操作的张量形状是否兼容，参考PyTorch的广播规则"
        elif "index" in str(error):
            return "确保索引在张量的有效范围内"
        elif "dimension" in str(error):
            return "确保维度操作前后的元素总数保持不变"
        else:
            return "检查张量操作的形状兼容性"
    
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
    
    def get_results(self) -> Dict:
        """获取分析结果"""
        return self.results
    
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
    parser = argparse.ArgumentParser(description="PyTea张量形状分析工具")
    
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--file", type=str, help="要分析的文件路径")
    target_group.add_argument("--dir", type=str, help="要分析的目录路径")
    
    parser.add_argument("--max-depth", type=int, default=10, help="最大分析深度")
    parser.add_argument("--timeout", type=int, default=300, help="超时时间（秒）")
    parser.add_argument("--verbose", action="store_true", help="输出详细信息")
    
    parser.add_argument("--output", type=str, help="输出文件路径")
    parser.add_argument("--config", type=str, help="配置文件路径")
    
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
    
    analyzer = PyTeaAnalyzer(config)
    
    analysis_options = {
        "max_depth": args.max_depth,
        "timeout": args.timeout,
        "verbose": args.verbose
    }
    
    if args.file:
        result = analyzer.analyze_file(args.file, **analysis_options)
    elif args.dir:
        result = analyzer.analyze_dir(args.dir, **analysis_options)
    
    if args.output:
        analyzer.save_results(args.output)
    else:
        total_issues = 0
        for file_path, file_result in analyzer.get_results().items():
            issues = file_result.get("issues", [])
            total_issues += len(issues)
            if issues:
                print(f"\n{file_path}: {len(issues)} 个问题")
                for i, issue in enumerate(issues, 1):
                    print(f"  {i}. {issue['message']} at {issue['location']}")
        
        print(f"\n总计: {total_issues} 个问题")

if __name__ == "__main__":
    main()
