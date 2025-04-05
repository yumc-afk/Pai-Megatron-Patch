set -e
ENV=$1
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATH}/Megatron-LM-250328:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # for PyTorch >= 2.6

if [ $ENV = dsw ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    MASTER_ADDR=localhost
    MASTER_PORT=$(shuf -n 1 -i 10000-65535)
    NNODES=1
    NODE_RANK=0
    GPUS_PER_NODE=2
elif [ $ENV = dlc ]; then
    NNODES=${WORLD_SIZE}
    NODE_RANK=${RANK}
    GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU:-2}
fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

TP=$2
PP=$3
EP=1
ETP=1

NUM_LAYERS=6
HIDDEN_SIZE=1024
NUM_ATTENTION_HEADS=16
INTERMEDIATE_SIZE=2816
MOE_INTERMEDIATE_SIZE=512
MAX_POSITION_EMBEDDINGS=2048
EXTRA_VOCAB_SIZE=467
Q_LORA_RANK=512
KV_LORA_RANK=128
QK_NOPE_HEAD_DIM=128
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
ROPE_THETA=10000
SCALE_FACTOR=1.0
NUM_EXPERTS=16
ROUTER_TOPK=4
NUM_SHARED_EXPERTS=1
RMS_NORM_EPS=1e-6

moe_options=" \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall \
    --moe-router-topk ${ROUTER_TOPK} \
    --moe-router-group-topk 2 \
    --moe-router-num-groups 2 \
    --num-experts ${NUM_EXPERTS} \
    --expert-model-parallel-size ${EP} \
    --expert-tensor-parallel-size ${ETP} \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-shared-expert-overlap \
    --moe-router-enable-expert-bias \
    --mscale 1.0 \
    --mscale-all-dim 1.0 \
    --moe-router-score-function sigmoid \
    --moe-router-bias-update-rate 0.001 \
    --moe-aux-loss-coeff 0.001 \
    --moe-layer-freq '([0]*1+[1]*5)' \
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
    --q-lora-rank ${Q_LORA_RANK} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-nope-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-rope-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --mtp-num-layers 1 \
    "

if [ $TP -eq 2 ]; then
    sp_option="--sequence-parallel"
else
    sp_option=""
fi

megatron_options="  \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_ATTENTION_HEADS} \
    --ffn-hidden-size ${INTERMEDIATE_SIZE} \
    --seq-length 512 \
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
    --max-padding-length 512 \
    --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
    --patch-tokenizer-type DeepSeekV2Tokenizer \
    --swiglu \
    --normalization RMSNorm \
    --norm-epsilon ${RMS_NORM_EPS} \
    --use-rotary-position-embeddings \
    --no-rope-fusion \
    --position-embedding-type rope \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --rotary-base ${ROPE_THETA} \
    --rotary-scaling-factor ${SCALE_FACTOR} \
    --kv-channels ${V_HEAD_DIM} \
    --qk-layernorm \
    --multi-latent-attention \
    --ckpt-format torch \
    --transformer-impl transformer_engine \
    --no-masked-softmax-fusion \
    --use-rope-scaling \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    "

echo "Running test script for validating MLA/MOE implementation..."
run_cmd="torchrun $DISTRIBUTED_ARGS test_mla_moe_correctness.py ${megatron_options} ${moe_options} ${sp_option} --test-tp ${TP} --test-pp ${PP}"

echo ${run_cmd}
eval ${run_cmd}
