set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}开始安装PyTorch LLM分布式训练静态分析工具...${NC}"

python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${YELLOW}检测到Python版本: ${python_version}${NC}"

if ! command -v pip &> /dev/null; then
    echo -e "${RED}未检测到pip，正在安装...${NC}"
    sudo apt-get update
    sudo apt-get install -y python3-pip
fi

echo -e "${YELLOW}请选择安装版本:${NC}"
echo "1) 标准版 - 包含所有分析器和功能"
echo "2) 精简版 - 仅包含核心分析器和LLM编排功能"
read -p "请选择 [1/2] (默认: 1): " version_choice
version_choice=${version_choice:-1}

echo -e "${YELLOW}安装基本依赖...${NC}"
pip install torch numpy jinja2

if [[ $version_choice == "1" ]]; then
    echo -e "${YELLOW}安装标准版...${NC}"
    pip install -e .
    
    echo -e "${YELLOW}验证安装...${NC}"
    python3 -c "import ml_static_analysis; print('标准版已成功安装')"
    
    echo -e "${GREEN}标准版安装完成!${NC}"
else
    echo -e "${YELLOW}安装精简版...${NC}"
    pip install -e .[lite]
    
    echo -e "${YELLOW}验证安装...${NC}"
    python3 -c "import ml_static_analysis_lite; print('精简版已成功安装')"
    
    echo -e "${GREEN}精简版安装完成!${NC}"
fi

echo -e "${YELLOW}使用说明:${NC}"
if [[ $version_choice == "1" ]]; then
    echo -e "1. 运行 'ml-analyze --help' 查看分析命令帮助"
else
    echo -e "1. 运行 'ml-analyze-lite --help' 查看分析命令帮助"
fi

echo -e "2. 查看文档获取详细使用指南: https://github.com/yumc-afk/Pai-Megatron-Patch/tree/main/static_analysis"
