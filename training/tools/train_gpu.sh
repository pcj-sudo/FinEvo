#!/usr/bin/env bash
# Run inside the isolated, pinned GPU runtime container under the allocation lock.
set -euo pipefail
MODE=${1:?use train}
[[ "$MODE" == train ]] || { echo "Optimized runtime starts from HF; use train" >&2; exit 2; }
RSI_ROOT=${RSI_ROOT:-/work/rsi-finance-training}
SLIME_ROOT=${SLIME_ROOT:-$RSI_ROOT/training/backend/slime}
RSI_RUN_ID=${RSI_RUN_ID:-smoke-001}
case "$RSI_RUN_ID" in *[!A-Za-z0-9_.-]*|'') echo 'Invalid run ID' >&2; exit 2;; esac
export PYTHONPATH="$RSI_ROOT/training/src:$SLIME_ROOT:${MEGATRON_ROOT:-/opt/Megatron-LM}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_COMPILE_DISABLE=1
export RSI_RUN_DIR="$RSI_ROOT/runs/$RSI_RUN_ID"
export RSI_ROOT
mkdir -p "$RSI_RUN_DIR"
mkdir -p "$RSI_RUN_DIR/diagnostics"
# Per-process communication logs plus a bounded flight recorder. Avoid DETAIL
# collective checking, CUDA synchronization, or in-process signal stack dumps.
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_DEBUG_FILE="$RSI_RUN_DIR/diagnostics/nccl.%h.%p.log"
export TORCH_DISTRIBUTED_DEBUG=INFO TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000 TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_CPP_STACK=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$RSI_RUN_DIR/diagnostics/flight-"
export RAY_DEDUP_LOGS=0
# Applied before either node starts Ray; allow transient CPU offload peaks while
# retaining Ray's memory monitor. The allocation's hard memory limit is unchanged.
export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.99}
echo "Ray host memory threshold: $RAY_memory_usage_threshold (node ${RSI_NODE_RANK:-0})"
cd "$SLIME_ROOT"

# Verified against Qwen/Qwen3.6-27B revision 6a9e13... text_config.
MODEL_ARGS=(
  --spec slime_plugins.models.qwen3_5 get_qwen3_5_spec
  --disable-bias-linear --qk-layernorm --group-query-attention
  --num-attention-heads 24 --num-query-groups 4 --kv-channels 256
  --num-layers 64 --hidden-size 5120 --ffn-hidden-size 17408 --use-gated-attention
  --normalization RMSNorm --apply-layernorm-1p --position-embedding-type rope
  --norm-epsilon 1e-6 --rotary-percent 0.25 --swiglu
  --untie-embeddings-and-output-weights --vocab-size 248320
  --rotary-base 10000000 --attention-output-gate
)
HF_MODEL="$RSI_ROOT/models/Qwen3.6-27B-hf"
export RSI_NODE_IPS=${RSI_NODE_IPS:-$RSI_GPU_IP}
RSI_NODE_COUNT=$(python -c 'import os; print(len(os.environ["RSI_NODE_IPS"].split(",")))')
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*' no_proxy='*'
if [[ "$RSI_NODE_COUNT" -gt 1 ]]; then
  # Pin bootstrap and data traffic to the reachable allocation network.
  : "${NCCL_SOCKET_IFNAME:?Set the reachable interface for distributed training}"
  export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME#=}}
fi
if [[ "${RSI_NODE_RANK:-0}" != 0 ]]; then
  exec python "$RSI_ROOT/training/tools/ray_worker.py"
fi
: "${RSI_OPTIMIZATION_CONFIG:?Explicit optimization config required}"
# Evaluate before mapfile so invalid configuration cannot be swallowed.
RSI_PARALLEL_TEXT=$(python -m rsi_training.optimization parallel)
RSI_ROUTER_TEXT=$(python -m rsi_training.optimization router)
RSI_TRAIN_ENV_JSON=$(python -m rsi_training.optimization train-env)
mapfile -t PARALLEL_ARGS <<< "$RSI_PARALLEL_TEXT"
mapfile -t ROUTER_ARGS <<< "$RSI_ROUTER_TEXT"
RSI_POOL_TEXT=$(python -m rsi_training.optimization pool)
mapfile -t POOL_ARGS <<< "$RSI_POOL_TEXT"
# Shared with the CPU adapter and frozen manifest; avoid divergent token limits.
RSI_TOKEN_LIMITS=$(python -m rsi_training.limits)
read -r RSI_CONTEXT_LENGTH RSI_OUTPUT_TOKENS RSI_INPUT_TOKENS RSI_CONDENSER_TOKENS <<< "$RSI_TOKEN_LIMITS"
read -r RSI_STEPS RSI_QUERIES RSI_SAMPLES RSI_BATCH <<< "$(python -m rsi_training.profile)"
RSI_SAVE_INTERVAL=$(python -c 'import os; from rsi_training.profile import read_profile; p=read_profile(os.environ.get("RSI_TRAIN_PROFILE")); print(p["steps"] if p["save_only_final"] else p["checkpoint_interval_steps"])')
RSI_PROMPT_DATA=$(python -c 'import os; from pathlib import Path; from rsi_training.profile import read_profile; print(Path(os.environ["RSI_ROOT"]) / read_profile(os.environ.get("RSI_TRAIN_PROFILE"))["prompt_manifest"])')
if [[ -f "$RSI_ROOT/checkpoints/$RSI_RUN_ID/latest_checkpointed_iteration.txt" && ! -f "$RSI_RUN_DIR/frozen-config.json" ]]; then
  echo "Cannot adopt an optimizer checkpoint without its matching frozen configuration" >&2; exit 2
fi
python "$RSI_ROOT/training/tools/freeze_run.py"
: "${RSI_CONTROLLER_URL:?CPU controller must be ready}"
: "${RSI_CONTROLLER_TOKEN_FILE:?controller credential file required}"
export RSI_CONTROLLER_URL RSI_CONTROLLER_TOKEN_FILE
export RSI_GATEWAY_PORT=${RSI_GATEWAY_PORT:-18001}
export RSI_GATEWAY_BIND=0.0.0.0
SAVE="$RSI_ROOT/checkpoints/$RSI_RUN_ID"
python -m rsi_training.continuous_checkpoint
LOAD="$HF_MODEL"
START_ARGS=(--start-rollout-id 0)
if [[ -f "$SAVE/latest_checkpointed_iteration.txt" ]]; then
  LOAD="$SAVE"
  START_ARGS=(--use-checkpoint-opt-param-scheduler)  # Preserve native scheduler, Adam and RNG when extending constant-LR training.
fi
[[ -f "$LOAD/latest_checkpointed_iteration.txt" || -f "$LOAD/config.json" ]] || { echo "Missing HF/checkpoint weights" >&2; exit 2; }
mkdir -p "$SAVE"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*' no_proxy='*'
# Container PID isolation scopes Ray to this run; no host-wide pkill or ray stop.
ray start --head --node-ip-address="${RSI_GPU_IP:?}" --num-gpus=8 \
  --num-cpus=60 --object-store-memory=8589934592 --port=6389 --dashboard-host=127.0.0.1 --dashboard-port=8269 --disable-usage-stats
python -c 'import os,json,pathlib,time; p=pathlib.Path(os.environ["RSI_RUN_DIR"]); (p/"head-ready.json").write_text(json.dumps({"time":time.time()}))'
# Includes bounded worker registration and real cross-node NCCL communication.
timeout --kill-after=15s 600s python -u "$RSI_ROOT/training/tools/ray_preflight.py"
python "$RSI_ROOT/training/tools/train_driver.py" \
  --actor-num-gpus-per-node 8 "${POOL_ARGS[@]}" "${START_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$HF_MODEL" --load "$LOAD" \
  --save "$SAVE" --save-interval "$RSI_SAVE_INTERVAL" \
  --prompt-data "$RSI_PROMPT_DATA" --input-key prompt --metadata-key metadata \
  --apply-chat-template \
  --rollout-function-path rsi_training.rollout.generate_batch \
  --custom-generate-function-path rsi_training.rollout.generate \
  --custom-rm-path rsi_training.reward.reward \
  --custom-reward-post-process-path rsi_training.reward.post_process \
  --custom-megatron-before-train-step-hook-path rsi_training.diagnostics.before_train_step \
  --num-rollout "$RSI_STEPS" --rollout-batch-size "$RSI_QUERIES" --n-samples-per-prompt "$RSI_SAMPLES" \
  --global-batch-size "$RSI_BATCH" --rollout-temperature 1.0 --rollout-top-p 1 \
  --rollout-max-context-len "$RSI_CONTEXT_LENGTH" --rollout-max-response-len "$RSI_OUTPUT_TOKENS" \
  --seq-length "$RSI_CONTEXT_LENGTH" --max-position-embeddings "$RSI_CONTEXT_LENGTH" \
  --rollout-stop-token-ids 248046 248044 \
  "${PARALLEL_ARGS[@]}" --train-env-vars "$RSI_TRAIN_ENV_JSON" --balance-data --balance-by-flops \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --use-dynamic-batch-size --max-tokens-per-gpu 16384 \
  --advantage-estimator grpo --calculate-per-token-loss \
  --kl-loss-coef 0 --kl-coef 0 --entropy-coef 0 --eps-clip 0.2 \
  --optimizer adam --lr 4e-6 --lr-decay-style constant --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 \
  --use-precision-aware-optimizer \
  --rollout-num-gpus-per-engine 2 --sglang-mem-fraction-static 0.65 \
  "${ROUTER_ARGS[@]}" \
  --sglang-context-length "$RSI_CONTEXT_LENGTH" \
  --sglang-tool-call-parser qwen3_coder --sglang-reasoning-parser qwen3-thinking \
  --sglang-mamba-scheduler-strategy extra_buffer \
  --attention-dropout 0 --hidden-dropout 0 --attention-backend flash \
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
  --save-debug-rollout-data "$RSI_RUN_DIR/slime-rollout-{rollout_id}.pt"
