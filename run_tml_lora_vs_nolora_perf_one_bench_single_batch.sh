#!/bin/bash
# Single-batch LoRA vs No-LoRA bench (minimal version):
#   For each scenario (LoRA / NoLoRA), launch ONE server and run all BSs in
#   one bench_one_batch_server call (`--batch-size 1 2 4 ... 64`). The bench
#   itself loops over BSs, calls /flush_cache between trials, and prints a
#   per-BS table via --show-report.
#
#   Why one server per scenario (instead of per BS): in single-batch mode
#   N == BS so the actual concurrent batch on GPU is exactly BS regardless
#   of running cap, as long as cap >= max(BATCH_SIZES). One server boot per
#   scenario is much faster than per-BS reboot.

pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 ray python 2>/dev/null || true
sleep 5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/sglang/python:${PYTHONPATH:-}"
export FLASHINFER_DISABLE_VERSION_CHECK=1

MODEL_PATH="Qwen/Qwen3-30B-A3B-Instruct-2507"
ADAPTER_PATH="${SCRIPT_DIR}/lora_test_cases/Qwen3-30B-A3B-Instruct-2507"
PORT=30000
TP=4
INPUT_LEN=8192
OUTPUT_LEN=1024
# First entry is a warmup pass (CUDA graph capture, KV alloc, first-shape
# prefill happen here); we drop its jsonl record after bench finishes so
# the report only contains production trials. Convention: NUM_WARMUP first
# entries of BATCH_SIZES are warmup. With BATCH_SIZES=(1 1 2 4 8 16 32 64)
# and NUM_WARMUP=1, the first BS=1 is warmup and the remaining 7 entries
# (BS=1, 2, 4, 8, 16, 32, 64) are production.
BATCH_SIZES=(1 1 2 4 8 16 32 64)
NUM_WARMUP=1

# Env overrides for cross-branch driver (run_two_branches_compare.sh style).
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/perf_results_tml_lora_vs_nolora}"
LORA_BACKEND="${LORA_BACKEND:-csgmv}"
SKIP_SERVER_WARMUP="${SKIP_SERVER_WARMUP:-0}"
mkdir -p "$RESULT_DIR"
rm -f "${RESULT_DIR}"/*.jsonl

SERVER_WARMUP_ARG=()
[[ "$SKIP_SERVER_WARMUP" == "1" ]] && SERVER_WARMUP_ARG=(--skip-server-warmup)

# Running cap = max(BATCH_SIZES) so every BS in the sweep fits in one wave.
MAX_BS=0
for bs in "${BATCH_SIZES[@]}"; do (( bs > MAX_BS )) && MAX_BS=$bs; done

echo "================================================================"
echo "  LoRA vs No-LoRA TML Perf (single-batch, one server per scenario)"
echo "  Model: ${MODEL_PATH} | TP=${TP}"
echo "  BSs: ${BATCH_SIZES[*]}  (first ${NUM_WARMUP} = warmup, max-running-requests=${MAX_BS})"
echo "  input_len=${INPUT_LEN}  output_len=${OUTPUT_LEN}"
echo "================================================================"

cleanup() {
    echo "  Cleaning up GPU/processes..."
    pkill -9 sglang python 2>/dev/null || true
    sleep 15
}

# Run one scenario: launches its own server, sweeps all BATCH_SIZES, prints
# the built-in --show-report table, dumps raw trials to ${SCENARIO}.jsonl.
# `--skip-warmup` skips the bench's internal small-shape (1024/16) warmup
# which used to trigger cudaErrorIllegalAddress on LoRA + virtual-experts;
# we instead use BATCH_SIZES[0..NUM_WARMUP-1] as our own real-shape warmup
# and strip those records from the jsonl after the run.
run_bench() {
    local LABEL="$1" SCENARIO="$2"; shift 2
    local RESULT_JSONL="${RESULT_DIR}/${SCENARIO}.jsonl"
    echo ""
    echo "──────────────────────────────────────────────"
    echo "  Running: ${LABEL}"
    echo "──────────────────────────────────────────────"
    python3 -m sglang.bench_one_batch_server \
        --model "$MODEL_PATH" \
        --tp "$TP" \
        --port "$PORT" \
        "${SERVER_WARMUP_ARG[@]}" \
        --max-running-requests "$MAX_BS" \
        --batch-size "${BATCH_SIZES[@]}" \
        --input-len "$INPUT_LEN" \
        --output-len "$OUTPUT_LEN" \
        --result-filename "$RESULT_JSONL" \
        --skip-warmup \
        --show-report \
        "$@"

    # Drop the first NUM_WARMUP records (warmup passes) from the jsonl so
    # downstream report tooling sees only production trials.
    if (( NUM_WARMUP > 0 )) && [[ -s "$RESULT_JSONL" ]]; then
        local total kept
        total=$(wc -l < "$RESULT_JSONL")
        kept=$(( total - NUM_WARMUP ))
        if (( kept > 0 )); then
            tail -n "+$((NUM_WARMUP + 1))" "$RESULT_JSONL" > "${RESULT_JSONL}.tmp" \
                && mv "${RESULT_JSONL}.tmp" "$RESULT_JSONL"
            echo "  Dropped first ${NUM_WARMUP} warmup record(s) from ${RESULT_JSONL} (kept ${kept}/${total})"
        else
            echo "  WARNING: jsonl has ${total} records but NUM_WARMUP=${NUM_WARMUP}; not dropping."
        fi
    fi

    cleanup
}

run_bench "LoRA (${LORA_BACKEND}, virtual experts)" "lora" \
    --mem-fraction-static 0.82 \
    --enable-lora \
    --lora-paths my_lora="$ADAPTER_PATH" \
    --max-lora-rank 32 \
    --lora-backend "$LORA_BACKEND" \
    --moe-runner-backend triton \
    --experts-shared-outer-loras \
    --lora-use-virtual-experts \
    --prefill-attention-backend fa4 \
    --decode-attention-backend fa4 \
    --lora-name my_lora

run_bench "Pure base model (no LoRA)" "nolora" \
    --prefill-attention-backend fa4 \
    --decode-attention-backend fa4

echo ""
echo "Done. Raw results: ${RESULT_DIR}/{lora,nolora}.jsonl"
echo "Per-scenario tables were printed inline above (--show-report)."
