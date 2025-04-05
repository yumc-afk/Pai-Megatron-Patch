set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <hf_model_path>"
    exit 1
fi

HF_MODEL_PATH=$1
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))

OUTPUT_DIR="${CURRENT_DIR}/mini_model_test_results"
mkdir -p ${OUTPUT_DIR}

echo "===== 步骤1: 转换HuggingFace模型到小型Megatron模型 ====="
python ${CURRENT_DIR}/convert_mini_model.py \
    --hf-model-path ${HF_MODEL_PATH} \
    --num-target-layers 6 \
    --save-path "${CURRENT_DIR}/mini_model_ckpt" \
    --target-tensor-model-parallel-size 2 \
    --target-pipeline-model-parallel-size 2

echo "===== 步骤2: 测试 TP=2, PP=1 配置 ====="
bash ${CURRENT_DIR}/run_mini_deepseek_test.sh dsw 2 1 ${HF_MODEL_PATH} 2>&1 | tee ${OUTPUT_DIR}/tp2_pp1_test.log

echo "===== 步骤3: 测试 TP=1, PP=2 配置 ====="
bash ${CURRENT_DIR}/run_mini_deepseek_test.sh dsw 1 2 ${HF_MODEL_PATH} 2>&1 | tee ${OUTPUT_DIR}/tp1_pp2_test.log

echo "===== 测试完成 ====="
echo "结果保存在: ${OUTPUT_DIR}"
echo "请检查日志文件以查看测试结果"
