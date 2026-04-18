#!/bin/bash
# Wrapper: benchmark Qwen3-30B-A3B-Instruct-2507 with 5 LoRA adapters
# (all pointing at the same HF repo -> same weights, different server-side
# slots lora0..lora4).
#
# Why 5 copies of one adapter?
#   * Exercises the multi-LoRA serving path (adapter selection, per-request
#     routing, memory budgeting) without introducing rank/shape variance
#     between adapters. Throughput differences vs. the single-adapter run
#     are then attributable to multi-LoRA overhead alone.
#   * Default sampling is `distinct`: prompts in a batch round-robin across
#     the 5 slots, so every batch (size >= 5) forces the server to fan out
#     across all adapters. This is the worst case for batched-LoRA kernels
#     and gives the cleanest signal for multi-LoRA scheduling overhead vs.
#     the single-adapter run. `uniform` only hits all slots statistically;
#     `skewed` (Zipf) isn't interesting with only 5 adapters.
#
# Model-specific notes (mirrors bench_Qwen3-30B-A3B.sh):
#   * --moe-runner-backend triton is LoRA-only per the reference — base
#     scenario uses the default MoE backend.
#   * --experts-shared-outer-loras routes LoRA through the shared-experts
#     outer path (required for MoE + LoRA correctness on this model).
#   * --max-lora-rank is NOT needed here because all 5 "adapters" are the
#     same weights (identical rank); if you swap in heterogeneous adapters
#     later, add `--max-lora-rank <N>` via --lora-extra-args.
#
# Result directory is auto-named by run_perf_lora.sh based on the adapter
# count + distribution (e.g. perf_results_Qwen3-30B-A3B-Instruct-2507_5lora_distinct/),
# so variants of this bench don't clobber each other. Override with
# --result-dir <path> if you want a specific location.
#
# Any flag passed on the CLI overrides the default below (last-wins), e.g.:
#
#   ./multi-lora-bench_Qwen3-30B-A3B.sh                              # defaults
#   ./multi-lora-bench_Qwen3-30B-A3B.sh --batch-sizes "1 64 256"     # override BS
#   ./multi-lora-bench_Qwen3-30B-A3B.sh --scenarios "lora lora_opt"  # skip base
#   ./multi-lora-bench_Qwen3-30B-A3B.sh --lora-request-distribution uniform
#   REUSE=1 ./multi-lora-bench_Qwen3-30B-A3B.sh --scenarios "lora_opt"
#
# To use a different adapter repo, override with --adapter "<repo> <repo> ..."
# (repeat the same repo to keep the "N copies of one adapter" setup).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 5 identical adapter slots -> server registers lora0..lora4, all backed by
# the same local directory. download_if_missing is idempotent on basename,
# so the repo is downloaded exactly once.
ADAPTER_REPO="yushengsu/lora-diff-Qwen3-30B-A3B-Instruct-2507"

exec "${SCRIPT_DIR}/run_perf_lora.sh" \
    --model        Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --adapter      "${ADAPTER_REPO} ${ADAPTER_REPO} ${ADAPTER_REPO} ${ADAPTER_REPO} ${ADAPTER_REPO}" \
    --tp           4 \
    --input-len    1024 \
    --output-len   2048 \
    --batch-sizes  "1 128 512" \
    --num-waves    2 \
    --min-samples  32 \
    --lora-request-distribution distinct \
    --common-args  "--prefill-attention-backend fa4 --decode-attention-backend fa4" \
    --lora-extra-args "--moe-runner-backend triton --experts-shared-outer-loras" \
    "$@"
