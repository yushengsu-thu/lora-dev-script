#!/bin/bash
# Wrapper: benchmark Kimi-K2.5 (MoE) with its LoRA adapter.
#
# Params mirror run_acc_Kimi-K2.5.sh so accuracy and perf runs stay aligned.
#
# Model-specific notes:
#   * --trust-remote-code is required by Kimi-K2.5's custom modeling code and
#     applies to every scenario → goes in --common-args.
#   * --moe-runner-backend triton keeps base vs LoRA apples-to-apples (LoRA
#     scenarios use triton regardless).
#   * --decode-attention-backend flashinfer is the MLA-compatible decode path;
#     fa4 does not support MLA on this model. Prefill stays on fa4.
#   * --experts-shared-outer-loras is required for Kimi's shared-experts MoE
#     LoRA routing to run correctly → goes in --lora-extra-args so both the
#     'lora' and 'lora_opt' scenarios include it.
#   * SGLANG_DISABLE_CUDNN_CHECK=1 sidesteps a cuDNN version mismatch at
#     launch (matches the accuracy script).
#
# Troubleshooting:
#   * If the base scenario crashes during CUDA-graph warmup (as DeepSeek-V3.1
#     does on the non-absorbed MHA path), add --disable-piecewise-cuda-graph
#     to --common-args.
#
# Override any flag at the CLI, e.g.:
#   ./bench_Kimi-K2.5.sh                                  # defaults
#   ./bench_Kimi-K2.5.sh --batch-sizes "1 64 256"         # override BS
#   ./bench_Kimi-K2.5.sh --scenarios "base lora_opt"
#   REUSE=1 ./bench_Kimi-K2.5.sh --scenarios "lora_opt"   # reuse cached results

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SGLANG_DISABLE_CUDNN_CHECK=1

exec "${SCRIPT_DIR}/run_perf_lora.sh" \
    --model        moonshotai/Kimi-K2.5 \
    --adapter      yushengsu/lora-diff-Kimi-K2.5 \
    --tp           8 \
    --input-len    1024 \
    --output-len   2048 \
    --batch-sizes  "1 128 512" \
    --num-waves    2 \
    --min-samples  32 \
    --common-args  "--trust-remote-code --prefill-attention-backend fa4 --decode-attention-backend flashinfer" \
    --lora-extra-args "--moe-runner-backend triton --experts-shared-outer-loras" \
    "$@"
