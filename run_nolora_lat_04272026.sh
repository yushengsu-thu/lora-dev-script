#!/bin/bash
set -euo pipefail

SCRIPT_DIR="/home/radixark/yushengsu"
export SCRIPT_DIR
export PYTHONPATH="${PYTHONPATH:-}"

MODEL_PATH="Qwen/Qwen3-30B-A3B-Instruct-2507"
PORT=30000
TP=4

NUM_RUNS=3
RESULT_DIR="${SCRIPT_DIR}/perf_results_04-27-2026_nolora_lat"
mkdir -p "$RESULT_DIR"
rm -f "${RESULT_DIR}"/*.jsonl

SCENARIOS=(
    "lat_prefill_short:1:1:256:2"
    "lat_prefill_long:1:1:4096:2"
    "lat_prefill_8k:1:1:8192:2"
    "lat_prefill_16k:1:1:16384:2"
    "lat_decode:1:1:256:256"
)

launch_and_wait() {
    local LABEL="$1"; shift
    echo "  Launching server: ${LABEL}"

    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    PYTHONPATH="${SCRIPT_DIR}/sglang/python:$PYTHONPATH" \
    python -m sglang.launch_server "$@" 2>&1 &
    SERVER_PID=$!

    local MAX_WAIT=600 ELAPSED=0
    while ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; do
        sleep 5; ELAPSED=$((ELAPSED + 5))
        if [ $ELAPSED -ge $MAX_WAIT ]; then
            echo "ERROR: Server failed to start within ${MAX_WAIT}s"
            kill -9 $SERVER_PID 2>/dev/null
            exit 1
        fi
        echo "  ... waiting (${ELAPSED}s / ${MAX_WAIT}s)"
    done
    echo "  Server ready! (PID: $SERVER_PID)"
}

kill_server() {
    echo "  Stopping server..."
    kill -9 $SERVER_PID 2>/dev/null || true
    pkill -9 sglang 2>/dev/null || true
    pkill -9 python 2>/dev/null || true
    sleep 15
}

run_warmup() {
    local LORA_ARG="$1"
    local NP="$2"
    local BS="$3"
    local IN_LEN="$4"
    local OUT_LEN="$5"
    echo "  >> Warmup: ${NP} reqs (BS=${BS}, in=${IN_LEN}, out=${OUT_LEN})..."
    PYTHONPATH="${SCRIPT_DIR}/sglang/python:$PYTHONPATH" \
    python -m sglang.bench_serving \
        --backend sglang \
        --port "$PORT" \
        --model "$MODEL_PATH" \
        --dataset-name random \
        --num-prompts "$NP" \
        --random-input-len "$IN_LEN" \
        --random-output-len "$OUT_LEN" \
        --request-rate inf \
        --max-concurrency "$BS" \
        ${LORA_ARG} \
        --disable-tqdm
    echo "  >> Warmup done."
}

run_bench_scenarios() {
    local PREFIX="$1"
    local LORA_ARG="$2"

    for SCENARIO in "${SCENARIOS[@]}"; do
        IFS=':' read -r NAME NP BS IN_LEN OUT_LEN <<< "$SCENARIO"
        local TAG="${PREFIX}_${NAME}"

        run_warmup "$LORA_ARG" "$NP" "$BS" "$IN_LEN" "$OUT_LEN"

        for RUN in $(seq 1 "$NUM_RUNS"); do
            echo "  >> [${TAG}] Run ${RUN}/${NUM_RUNS}"
            PYTHONPATH="${SCRIPT_DIR}/sglang/python:$PYTHONPATH" \
            python -m sglang.bench_serving \
                --backend sglang \
                --port "$PORT" \
                --model "$MODEL_PATH" \
                --dataset-name random \
                --num-prompts "$NP" \
                --random-input-len "$IN_LEN" \
                --random-output-len "$OUT_LEN" \
                --request-rate inf \
                --max-concurrency "$BS" \
                ${LORA_ARG} \
                --output-file "${RESULT_DIR}/${TAG}.jsonl" \
                --disable-tqdm
        done
    done
}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  No-LoRA Base Model Latency (BS=1) - 04-27-2026            ║"
echo "╚══════════════════════════════════════════════════════════════╝"

launch_and_wait "Pure base model (no LoRA)" \
    --model "$MODEL_PATH" \
    --tp "$TP" \
    --port "$PORT" \
    --prefill-attention-backend fa4 \
    --decode-attention-backend fa4

run_bench_scenarios "nolora" ""
kill_server

echo "Done. Results in: ${RESULT_DIR}/"
