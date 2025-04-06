set -e


CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
REPO_ROOT="$( cd "$( dirname "$( dirname "${CURRENT_DIR}" )" )" && pwd )"
echo "仓库根目录: ${REPO_ROOT}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}开始安装DeepSeek V3 MLA/MOE CPU测试环境...${NC}"

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
    
    VENV_DIR="${REPO_ROOT}/venv_deepseek_test"
    python3 -m venv ${VENV_DIR}
    
    source ${VENV_DIR}/bin/activate
    echo -e "${GREEN}已激活虚拟环境: ${VENV_DIR}${NC}"
    
    pip install --upgrade pip
fi

echo -e "${YELLOW}安装基本依赖...${NC}"
pip install torch numpy transformers huggingface_hub

echo -e "${YELLOW}验证PyTorch安装...${NC}"
python3 -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"

echo -e "${YELLOW}设置PYTHONPATH...${NC}"
export PYTHONPATH=${REPO_ROOT}:${REPO_ROOT}/Megatron-LM-250328:$PYTHONPATH
echo "export PYTHONPATH=${REPO_ROOT}:${REPO_ROOT}/Megatron-LM-250328:\$PYTHONPATH" > ${CURRENT_DIR}/setup_env.sh

if [ ! -d "${REPO_ROOT}/Megatron-LM-250328" ]; then
    echo -e "${RED}未找到Megatron-LM-250328子模块，正在初始化...${NC}"
    cd ${REPO_ROOT}
    git submodule update --init --recursive
fi

echo -e "${YELLOW}安装测试特定依赖...${NC}"
pip install tqdm matplotlib

echo -e "${YELLOW}验证安装...${NC}"
python3 -c "import torch; import transformers; import huggingface_hub; print('所有依赖已成功安装')"

echo -e "${GREEN}安装完成!${NC}"
echo -e "${YELLOW}使用说明:${NC}"
echo -e "1. 运行 'source ${CURRENT_DIR}/setup_env.sh' 设置环境变量"
echo -e "2. 运行 'bash ${CURRENT_DIR}/run_cpu_tests.sh' 执行CPU测试"
echo -e "3. 查看 '${CURRENT_DIR}/CPU_TEST_GUIDE.md' 获取详细使用指南"

if [[ -n $VIRTUAL_ENV ]]; then
    echo -e "4. 使用 'deactivate' 退出虚拟环境"
fi
