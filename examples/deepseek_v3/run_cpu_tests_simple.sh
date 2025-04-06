set -e

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
REPO_ROOT="$( cd "$( dirname "$( dirname "${CURRENT_DIR}" )" )" && pwd )"
export PYTHONPATH=${REPO_ROOT}:${REPO_ROOT}/Megatron-LM-250328:$PYTHONPATH

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}===== 运行CPU上的MLA/MOE组件验证 =====${NC}"

RESULTS_DIR="${CURRENT_DIR}/cpu_test_results"
mkdir -p ${RESULTS_DIR}

echo -e "${YELLOW}运行基本组件测试...${NC}"
python ${CURRENT_DIR}/test_mla_moe_cpu_simple.py 2>&1 | tee ${RESULTS_DIR}/basic_components_test.log

echo -e "${YELLOW}验证DeepSeek V3结构...${NC}"
python ${CURRENT_DIR}/verify_deepseek_v3_structure.py --model-name deepseek-ai/deepseek-v3-7b --config-only 2>&1 | tee ${RESULTS_DIR}/deepseek_v3_structure.log

echo -e "${YELLOW}验证DeepSeek MOE结构...${NC}"
python ${CURRENT_DIR}/verify_mla_moe_structure.py --model-name deepseek-ai/deepseek-moe-16b-base --config-only 2>&1 | tee ${RESULTS_DIR}/deepseek_moe_structure.log

echo -e "${YELLOW}验证权重转换参数...${NC}"
python ${CURRENT_DIR}/verify_weight_conversion.py --hf-model-name deepseek-ai/deepseek-moe-16b-base --use-cpu 2>&1 | tee ${RESULTS_DIR}/weight_conversion_params.log

echo -e "${GREEN}===== 测试完成 =====${NC}"
echo -e "结果保存在: ${RESULTS_DIR}"

echo -e "${YELLOW}生成测试报告...${NC}"
cat > ${RESULTS_DIR}/summary.txt << EOF

测试时间: $(date)


1. 基本组件测试 (test_mla_moe_cpu_simple.py)
2. DeepSeek V3结构验证 (verify_deepseek_v3_structure.py)
3. DeepSeek MOE结构验证 (verify_mla_moe_structure.py)
4. 权重转换参数验证 (verify_weight_conversion.py)


- 操作系统: $(uname -a)
- Python版本: $(python --version 2>&1)
- PyTorch版本: $(python -c "import torch; print(torch.__version__)" 2>&1)


请查看各个日志文件获取详细测试结果。
EOF

echo -e "${GREEN}测试报告已生成: ${RESULTS_DIR}/summary.txt${NC}"
