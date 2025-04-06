#
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
    description="Static analysis framework for ML/DL codebases with LLM orchestration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/haoweiliang1996/Pai-Megatron-Patch",
    packages=find_packages(),
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
    ],
    extras_require={
        "lite": [
            "mypy>=1.0.0",
            "typing-extensions>=4.5.0",
        ],
        "standard": [
            "mypy>=1.0.0",
            "typing-extensions>=4.5.0",
            "pylint>=2.17.0",
            "flake8>=6.0.0",
            "pytest>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ml-analyze=ml_static_analysis.cli.main:main",
            "ml-analyze-lite=ml_static_analysis.cli.main:main_lite",
        ],
    },
)
