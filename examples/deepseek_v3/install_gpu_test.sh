set -e


CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
REPO_ROOT="$( cd "$( dirname "$( dirname "${CURRENT_DIR}" )" )" && pwd )"
echo "仓库根目录: ${REPO_ROOT}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}开始安装DeepSeek V3 MLA/MOE GPU测试环境...${NC}"

if command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}检测到NVIDIA GPU:${NC}"
    nvidia-smi
else
    echo -e "${RED}警告: 未检测到NVIDIA GPU或nvidia-smi命令不可用${NC}"
    echo -e "${RED}GPU测试需要NVIDIA GPU和CUDA支持${NC}"
    read -p "是否继续安装? (y/n) " continue_install
    if [[ $continue_install != "y" && $continue_install != "Y" ]]; then
        echo "安装已取消"
        exit 1
    fi
fi

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
    
    VENV_DIR="${REPO_ROOT}/venv_deepseek_gpu_test"
    python3 -m venv ${VENV_DIR}
    
    source ${VENV_DIR}/bin/activate
    echo -e "${GREEN}已激活虚拟环境: ${VENV_DIR}${NC}"
    
    pip install --upgrade pip
fi

echo -e "${YELLOW}安装基本依赖...${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy transformers huggingface_hub

echo -e "${YELLOW}验证PyTorch安装...${NC}"
python3 -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available()); print('CUDA版本:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

echo -e "${YELLOW}安装Megatron-LM依赖...${NC}"
pip install regex nltk sentencepiece

echo -e "${YELLOW}设置PYTHONPATH...${NC}"
export PYTHONPATH=${REPO_ROOT}:${REPO_ROOT}/Megatron-LM-250328:$PYTHONPATH
echo "export PYTHONPATH=${REPO_ROOT}:${REPO_ROOT}/Megatron-LM-250328:\$PYTHONPATH" > ${CURRENT_DIR}/setup_gpu_env.sh

if [ ! -d "${REPO_ROOT}/Megatron-LM-250328" ]; then
    echo -e "${RED}未找到Megatron-LM-250328子模块，正在初始化...${NC}"
    cd ${REPO_ROOT}
    git submodule update --init --recursive
fi

echo -e "${YELLOW}安装测试特定依赖...${NC}"
pip install tqdm matplotlib tensorboard

read -p "是否安装NVIDIA APEX? (y/n) " install_apex
if [[ $install_apex == "y" || $install_apex == "Y" ]]; then
    echo -e "${YELLOW}安装NVIDIA APEX...${NC}"
    cd ${REPO_ROOT}
    if [ ! -d "apex" ]; then
        git clone https://github.com/NVIDIA/apex
    fi
    cd apex
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
fi

read -p "是否安装Flash Attention? (y/n) " install_flash_attn
if [[ $install_flash_attn == "y" || $install_flash_attn == "Y" ]]; then
    echo -e "${YELLOW}安装Flash Attention...${NC}"
    pip install flash-attn --no-build-isolation
fi

echo -e "${YELLOW}验证安装...${NC}"
python3 -c "import torch; import transformers; import huggingface_hub; print('所有依赖已成功安装')"

echo -e "${GREEN}安装完成!${NC}"
echo -e "${YELLOW}使用说明:${NC}"
echo -e "1. 运行 'source ${CURRENT_DIR}/setup_gpu_env.sh' 设置环境变量"
echo -e "2. 运行 'bash ${CURRENT_DIR}/run_mini_deepseek_test.sh dsw 2 1' 执行TP=2,PP=1测试"
echo -e "3. 运行 'bash ${CURRENT_DIR}/run_mini_deepseek_test.sh dsw 1 2' 执行TP=1,PP=2测试"
echo -e "4. 查看 '${CURRENT_DIR}/GPU_TEST_GUIDE.md' 获取详细使用指南"

if [[ -n $VIRTUAL_ENV ]]; then
    echo -e "5. 使用 'deactivate' 退出虚拟环境"
fi
