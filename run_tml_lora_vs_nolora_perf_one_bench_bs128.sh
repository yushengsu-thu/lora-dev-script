#!/bin/bash
# Single-batch BS=128 LoRA vs No-LoRA reproduction bench. Sibling of
# run_tml_lora_vs_nolora_perf_one_bench_single_batch.sh — same harness, but
# BATCH_SIZES is fixed to (1 128) and input/output shrunk to (2048, 128) so
# 128 concurrent prompts comfortably fit on 4×GB300 memory while still
# exercising the codepath.
#
# Primary use: confirm whether lora-perf-optimize-2 hits the BS>=128
# illegal-memory-access flagged on PR #24262's align kernel, while
# 04-27-2026 (pre-opt-2) succeeds at the same shape.

pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 ray python 2>/dev/null || true
sleep 5

set -uo pipefail   # NOTE: no -e — we want to continue past a server crash
                   # (which is the entire point of this BS=128 repro run).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/sglang/python:${PYTHONPATH:-}"
export FLASHINFER_DISABLE_VERSION_CHECK=1

MODEL_PATH="Qwen/Qwen3-30B-A3B-Instruct-2507"
ADAPTER_PATH="${SCRIPT_DIR}/lora_test_cases/Qwen3-30B-A3B-Instruct-2507"
PORT=30000
TP=4
# INPUT_LEN / OUTPUT_LEN are overridable from the env.
INPUT_LEN="${INPUT_LEN:-8192}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
# (1) is warmup; the second entry is the production trial we actually want
# to read. Edit this to change the production BS.
# BATCH_SIZES=(1 128)
BATCH_SIZES=(1 64)
# BATCH_SIZES=(1 4)
NUM_WARMUP=1

# Compute MAX_BS up-front for --max-running-requests.
MAX_BS=0
for bs in "${BATCH_SIZES[@]}"; do (( bs > MAX_BS )) && MAX_BS=$bs; done

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/perf_results_comp/bench_${TIMESTAMP}}"
LORA_BACKEND="${LORA_BACKEND:-csgmv}"
SKIP_SERVER_WARMUP="${SKIP_SERVER_WARMUP:-0}"
SKIP_LORA="${SKIP_LORA:-0}"
SKIP_NOLORA="${SKIP_NOLORA:-0}"
mkdir -p "$RESULT_DIR"
rm -f "${RESULT_DIR}"/*.jsonl

SERVER_WARMUP_ARG=()
[[ "$SKIP_SERVER_WARMUP" == "1" ]] && SERVER_WARMUP_ARG=(--skip-server-warmup)

echo "================================================================"
echo "  BS=128 LoRA vs No-LoRA reproduction bench"
echo "  Model: ${MODEL_PATH} | TP=${TP}"
echo "  BSs: ${BATCH_SIZES[*]}  (first ${NUM_WARMUP} = warmup, max-running-requests=${MAX_BS})"
echo "  input_len=${INPUT_LEN}  output_len=${OUTPUT_LEN}"
echo "  LoRA backend: ${LORA_BACKEND}  skip_server_warmup=${SKIP_SERVER_WARMUP}"
echo "  skip_lora=${SKIP_LORA}  skip_nolora=${SKIP_NOLORA}"
echo "================================================================"

cleanup() {
    echo "  Cleaning up GPU/processes..."
    pkill -9 sglang python 2>/dev/null || true
    sleep 15
}

run_bench() {
    local LABEL="$1" SCENARIO="$2"; shift 2
    local RESULT_JSONL="${RESULT_DIR}/${SCENARIO}.jsonl"
    local SERVER_LOG="${RESULT_DIR}/${SCENARIO}.server.log"
    echo ""
    echo "──────────────────────────────────────────────"
    echo "  Running: ${LABEL}"
    echo "  Server stderr/stdout → ${SERVER_LOG}"
    echo "──────────────────────────────────────────────"

    set +e
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
        "$@" 2>&1 | tee "$SERVER_LOG"
    local RC=${PIPESTATUS[0]}
    set -e

    if (( RC != 0 )); then
        echo ""
        echo "  >>> bench_one_batch_server exited with code ${RC} for scenario '${SCENARIO}'."
        echo "      Last 30 lines of server log:"
        tail -n 30 "$SERVER_LOG" | sed 's/^/      | /'
    fi

    if (( NUM_WARMUP > 0 )) && [[ -s "$RESULT_JSONL" ]]; then
        local total kept
        total=$(wc -l < "$RESULT_JSONL")
        kept=$(( total - NUM_WARMUP ))
        if (( kept > 0 )); then
            tail -n "+$((NUM_WARMUP + 1))" "$RESULT_JSONL" > "${RESULT_JSONL}.tmp" \
                && mv "${RESULT_JSONL}.tmp" "$RESULT_JSONL"
            echo "  Dropped first ${NUM_WARMUP} warmup record(s) from ${RESULT_JSONL} (kept ${kept}/${total})"
        fi
    fi

    cleanup
    return 0
}

if [[ "$SKIP_LORA" != "1" ]]; then
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
fi

if [[ "$SKIP_NOLORA" != "1" ]]; then
    run_bench "Pure base model (no LoRA, triton MoE for apples-to-apples)" "nolora" \
        --mem-fraction-static 0.82 \
        --moe-runner-backend triton \
        --prefill-attention-backend fa4 \
        --decode-attention-backend fa4
fi

echo ""
echo "================================================================"
echo "  Done.  All results in: ${RESULT_DIR}"
echo "  Benchmark:  ${RESULT_DIR}/{lora,nolora}.jsonl"
echo "  Server logs: ${RESULT_DIR}/{lora,nolora}.server.log"
echo "================================================================"
