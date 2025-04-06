set -e

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
REPO_ROOT="$( cd "$( dirname "${CURRENT_DIR}" )" && pwd )"

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

read -p "是否创建虚拟环境? (y/n) " create_venv
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo -e "${YELLOW}创建虚拟环境...${NC}"
    
    if ! python3 -c "import venv" &> /dev/null; then
        echo -e "${RED}未检测到venv模块，正在安装...${NC}"
        sudo apt-get install -y python3-venv
    fi
    
    VENV_DIR="${REPO_ROOT}/venv_static_analysis"
    python3 -m venv ${VENV_DIR}
    
    source ${VENV_DIR}/bin/activate
    echo -e "${GREEN}已激活虚拟环境: ${VENV_DIR}${NC}"
    
    pip install --upgrade pip
fi

echo -e "${YELLOW}安装基本依赖...${NC}"
pip install torch numpy mypy

echo -e "${YELLOW}安装TorchTyping...${NC}"
pip install torchtyping

echo -e "${YELLOW}安装PyTea依赖...${NC}"
pip install z3-solver

echo -e "${YELLOW}克隆PyTea仓库...${NC}"
cd ${REPO_ROOT}
if [ ! -d "pytea" ]; then
    git clone https://github.com/ropas/pytea.git
    cd pytea
    pip install -e .
else
    echo -e "${YELLOW}PyTea仓库已存在，跳过克隆...${NC}"
    cd pytea
    git pull
    pip install -e .
fi

echo -e "${YELLOW}安装CrossHair...${NC}"
pip install crosshair-tool

echo -e "${YELLOW}安装自定义静态分析工具...${NC}"
cd ${CURRENT_DIR}
pip install -e .

echo -e "${YELLOW}设置环境变量...${NC}"
export PYTHONPATH=${REPO_ROOT}:${CURRENT_DIR}:$PYTHONPATH
echo "export PYTHONPATH=${REPO_ROOT}:${CURRENT_DIR}:\$PYTHONPATH" > ${CURRENT_DIR}/setup_env.sh

echo -e "${YELLOW}验证安装...${NC}"
python3 -c "import torch; import torchtyping; import z3; print('基本依赖已成功安装')"

echo -e "${GREEN}安装完成!${NC}"
echo -e "${YELLOW}使用说明:${NC}"
echo -e "1. 运行 'source ${CURRENT_DIR}/setup_env.sh' 设置环境变量"
echo -e "2. 运行 'python ${CURRENT_DIR}/run_analysis.py --help' 查看分析命令帮助"
echo -e "3. 查看 '${CURRENT_DIR}/USAGE.md' 获取详细使用指南"

if [[ -n $VIRTUAL_ENV ]]; then
    echo -e "4. 使用 'deactivate' 退出虚拟环境"
fi
