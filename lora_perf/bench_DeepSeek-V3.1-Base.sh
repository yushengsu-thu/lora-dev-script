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

exec "${SCRIPT_DIR}/run_perf_lora.sh" \
    --model        deepseek-ai/DeepSeek-V3.1-Base \
    --adapter      yushengsu/lora-diff-DeepSeek-V3.1-Base \
    --tp           8 \
    --input-len    1024 \
    --output-len   2048 \
    --batch-sizes  "1 128 512" \
    --num-waves    2 \
    --min-samples  32 \
    --common-args  "--prefill-attention-backend fa4 --decode-attention-backend flashinfer --disable-piecewise-cuda-graph" \
    --lora-extra-args "--moe-runner-backend triton --experts-shared-outer-loras --disable-shared-experts-fusion" \
    "$@"
