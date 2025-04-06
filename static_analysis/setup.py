#
#

import os
import sys
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ml-static-analysis",
    version="0.1.0",
    author="Alibaba PAI Team",
    author_email="pai-team@alibaba-inc.com",
    description="PyTorch LLM分布式训练静态分析框架，支持LLM编排",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yumc-afk/Pai-Megatron-Patch",
    packages=find_packages() + ["ml_static_analysis_lite"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "mypy>=1.0.0",
        "typing-extensions>=4.5.0",
        "jinja2>=3.0.0",
    ],
    extras_require={
        "lite": [
            "mypy>=1.0.0",
            "typing-extensions>=4.5.0",
            "jaxtyping>=0.2.20",
            "jinja2>=3.0.0",
        ],
        "standard": [
            "mypy>=1.0.0",
            "typing-extensions>=4.5.0",
            "jaxtyping>=0.2.20",
            "pylint>=2.17.0",
            "flake8>=6.0.0",
            "pytest>=7.0.0",
            "jinja2>=3.0.0",
        ],
        "all": [
            "mypy>=1.0.0",
            "typing-extensions>=4.5.0",
            "jaxtyping>=0.2.20",
            "pylint>=2.17.0",
            "flake8>=6.0.0",
            "pytest>=7.0.0",
            "crosshair-tool>=0.0.31",
            "jinja2>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ml-analyze=ml_static_analysis.cli.main:main",
            "ml-analyze-lite=ml_static_analysis_lite:analyze_codebase",
        ],
    },
)
