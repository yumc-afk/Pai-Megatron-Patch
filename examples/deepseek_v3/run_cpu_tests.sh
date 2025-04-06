set -e

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATH}/Megatron-LM-250328:$PYTHONPATH

echo "===== 运行CPU上的MLA/MOE组件验证 ====="
python ${CURRENT_DIR}/test_mla_moe_cpu.py 2>&1 | tee ${CURRENT_DIR}/cpu_test_results.log

echo "===== 测试完成 ====="
echo "结果保存在: ${CURRENT_DIR}/cpu_test_results.log"
