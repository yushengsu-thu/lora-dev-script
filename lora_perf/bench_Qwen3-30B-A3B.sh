#!/bin/bash
# Wrapper: benchmark Qwen3-30B-A3B-Instruct-2507 with its LoRA adapter.
#
# Model-specific notes (mirrors run_perf_Qwen3-30B-A3B.sh /
# run_perf_Qwen3-30B-A3B-optimize.sh):
#   * --moe-runner-backend triton is LoRA-only per the reference — base
#     scenario uses the default MoE backend. If you want apples-to-apples,
#     move it into --common-args.
#   * --experts-shared-outer-loras routes LoRA through the shared-experts
#     outer path (required for MoE + LoRA correctness on this model).
#   * --max-lora-rank is intentionally omitted: with a single LoRA, SGLang
#     infers the rank from the adapter's config. The flag is only needed
#     when serving multiple adapters of varying rank.
#
# Any flag passed on the CLI overrides the default below (last-wins), e.g.:
#
#   ./bench_Qwen3-30B-A3B.sh                           # defaults
#   ./bench_Qwen3-30B-A3B.sh --batch-sizes "1 64 256"  # override BS
#   ./bench_Qwen3-30B-A3B.sh --scenarios "base lora_opt"
#   REUSE=1 ./bench_Qwen3-30B-A3B.sh --scenarios "lora_opt"
#
# If your adapter lives under a different HuggingFace repo, override with
# --adapter <your/repo>. The generic script only downloads when the local
# copy at ~/models/<basename> is missing.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/run_perf_lora.sh" \
    --model        Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --adapter      yushengsu/lora-diff-Qwen3-30B-A3B-Instruct-2507 \
    --tp           4 \
    --input-len    1024 \
    --output-len   2048 \
    --batch-sizes  "1 128 512" \
    --num-waves    2 \
    --min-samples  32 \
    --common-args  "--prefill-attention-backend fa4 --decode-attention-backend fa4" \
    --lora-extra-args "--moe-runner-backend triton --experts-shared-outer-loras" \
    "$@"
