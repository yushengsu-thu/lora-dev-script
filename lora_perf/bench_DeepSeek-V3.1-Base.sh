#!/bin/bash
# Wrapper: benchmark DeepSeek-V3.1-Base with its LoRA adapter.
#
# Model-specific configuration notes:
#   * MLA attention → decode backend must be flashinfer (fa4 does not support MLA).
#   * Piecewise CUDA graphs crash at warmup on the non-absorbed MHA path for
#     the base scenario, so we pass --disable-piecewise-cuda-graph. (LoRA
#     scenarios implicitly disable it.)
#   * Shared-expert fusion conflicts with LoRA remapping on DS-V3 → the LoRA
#     scenarios pass --disable-shared-experts-fusion via --lora-extra-args,
#     while the base scenario keeps fusion enabled (it's the faster path).
#   * --moe-runner-backend triton makes base vs LoRA a fair apples-to-apples
#     comparison (LoRA scenarios use triton regardless).
#   * --experts-shared-outer-loras routes LoRA through the shared-experts
#     outer path (required for MoE + LoRA correctness; mirrors
#     run_acc_DeepSeek-V3.1-Base.sh).
#
# Override any flag at the CLI, e.g.:
#   ./bench_DeepSeek-V3.1-Base.sh --tp 8 --batch-sizes "1 64 256"
#   ./bench_DeepSeek-V3.1-Base.sh --scenarios "base"
#   REUSE=1 ./bench_DeepSeek-V3.1-Base.sh --scenarios "lora_opt"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Auto-pick TP from GPU type (count alone is ambiguous — a box may have
# 8 GB300 or 4 H200):
#   * GB300 -> tp=4  (uses 4 of however many GB300s are present)
#   * H200  -> tp=8  (requires all 8 H200s)
GPU_LIST=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)
GPU_NAME=$(printf '%s\n' "$GPU_LIST" | awk 'NR==1' | tr -s ' ')
GPU_COUNT=$(printf '%s\n' "$GPU_LIST" | grep -c . | tr -d ' ')
case "$GPU_NAME" in
    *GB300*) TP=4 ;;
    *H200*)  TP=8 ;;
    *) echo "ERROR: bench_DeepSeek-V3.1-Base expects GB300 or H200; detected '${GPU_NAME}'"; exit 1 ;;
esac
if [ "$GPU_COUNT" -lt "$TP" ]; then
    echo "ERROR: '${GPU_NAME}' run wants tp=${TP} but only ${GPU_COUNT} GPUs are available"
    exit 1
fi

# Meaningful EP values on this TP: {1, tp}. SGLang's default moe_dp_size=1
# enforces ep_size * moe_dp_size == tp_size when ep > 1, so only ep=1 and
# ep=tp launch cleanly; the sweep covers both.
exec "${SCRIPT_DIR}/run_perf_lora.sh" \
    --model        deepseek-ai/DeepSeek-V3.1-Base \
    --adapter      yushengsu/lora-diff-DeepSeek-V3.1-Base \
    --tp           "$TP" \
    --ep           "1 $TP" \
    --input-len    1024 \
    --output-len   2048 \
    --batch-sizes  "1 128 512" \
    --num-waves    2 \
    --min-samples  32 \
    --common-args  "--prefill-attention-backend fa4 --decode-attention-backend flashinfer --disable-piecewise-cuda-graph" \
    --lora-extra-args "--moe-runner-backend triton --experts-shared-outer-loras --disable-shared-experts-fusion" \
    "$@"
