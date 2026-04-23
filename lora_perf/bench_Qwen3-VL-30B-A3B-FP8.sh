#!/bin/bash
# Wrapper: benchmark Qwen3-VL-30B-A3B-Instruct-FP8 with its LoRA adapter.
#
# Uses local paths for both the model and the adapter (no HF download):
#   model   = /root/models/Qwen3-VL-30B-A3B-Instruct-FP8
#   adapter = /root/models/riverai_lora/lora-diff-Qwen3-VL-30B-A3B
# run_perf_lora.sh detects the leading '/' and skips the HF download path.
#
# Model-specific notes (mirrors bench_Qwen3-30B-A3B.sh — same MoE family):
#   * --moe-runner-backend triton is LoRA-only per the reference — base
#     scenario uses the default MoE backend. If you want apples-to-apples,
#     move it into --common-args.
#   * --experts-shared-outer-loras routes LoRA through the shared-experts
#     outer path (required for MoE + LoRA correctness on this model).
#   * --max-lora-rank is intentionally omitted: with a single LoRA, SGLang
#     infers the rank from the adapter's config. The flag is only needed
#     when serving multiple adapters of varying rank.
#   * FP8 weights don't require any extra flag here — SGLang reads the
#     quantization config from the checkpoint.
#   * TP=4, EP=1 is NOT loadable on this checkpoint: FP8 block quant uses
#     block_n=128, and TP-sharding moe_intermediate_size=768 by 4 gives 192
#     (not a multiple of 128). With EP=tp, experts are sharded across ranks
#     (whole experts per rank) instead of slicing each expert's weights, so
#     per-expert output dim stays at 768 and loading works. Hence the EP
#     sweep below is "4" only — ep=1 is dropped for this variant.
#   * The Qwen3-VL base is multimodal, but bench_one_batch_server uses
#     `--dataset-name random` (text-only prompts), so this measures the
#     language-model path. If you later add multimodal prompts, revisit.
#
# Any flag passed on the CLI overrides the default below (last-wins), e.g.:
#
#   ./bench_Qwen3-VL-30B-A3B-FP8.sh                           # defaults
#   ./bench_Qwen3-VL-30B-A3B-FP8.sh --batch-sizes "1 64 256"  # override BS
#   ./bench_Qwen3-VL-30B-A3B-FP8.sh --scenarios "base lora_opt"
#   REUSE=1 ./bench_Qwen3-VL-30B-A3B-FP8.sh --scenarios "lora_opt"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/run_perf_lora.sh" \
    --model        /root/models/Qwen3-VL-30B-A3B-Instruct-FP8 \
    --adapter      /root/models/riverai_lora/lora-diff-Qwen3-VL-30B-A3B \
    --tp           4 \
    --ep           4 \
    --input-len    1024 \
    --output-len   2048 \
    --batch-sizes  "1 128 512" \
    --num-waves    2 \
    --min-samples  32 \
    --common-args  "--prefill-attention-backend fa4 --decode-attention-backend fa4" \
    --lora-extra-args "--moe-runner-backend triton" \
    "$@"
