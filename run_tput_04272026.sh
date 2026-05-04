#!/bin/bash
set -euo pipefail

SCRIPT_DIR="/home/radixark/yushengsu"
export SCRIPT_DIR
export PYTHONPATH="${PYTHONPATH:-}"

MODEL_PATH="Qwen/Qwen3-30B-A3B-Instruct-2507"
ADAPTER_PATH="${SCRIPT_DIR}/lora_test_cases/Qwen3-30B-A3B-Instruct-2507"
PORT=30000
TP=4

NUM_RUNS=3
RESULT_DIR="${SCRIPT_DIR}/perf_results_tml_lora_vs_nolora_04-27-2026_tput"
mkdir -p "$RESULT_DIR"
rm -f "${RESULT_DIR}"/*.jsonl

SCENARIOS=(
    "tput_bs64_np64:64:64:256:256"
    "tput_bs64_np128:128:64:256:256"
    "tput_bs64_np256:256:64:256:256"
)

launch_and_wait() {
    local LABEL="$1"; shift
    echo ""
    echo "────────────────────────────────────────────────"
    echo "  Launching server: ${LABEL}"
    echo "────────────────────────────────────────────────"

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
            echo "  >> [${TAG}] Run ${RUN}/${NUM_RUNS} num_prompts=${NP} BS=${BS} in=${IN_LEN} out=${OUT_LEN}"
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

# ══════════════════════════════════════════════════════════════
# Scenario 1: LoRA with triton + virtual experts (no mem-fraction-static to give more headroom)
# ══════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Throughput: LoRA (triton + virtual experts) - 04-27-2026   ║"
echo "╚══════════════════════════════════════════════════════════════╝"

launch_and_wait "LoRA (triton, virtual experts)" \
    --model "$MODEL_PATH" \
    --tp "$TP" \
    --port "$PORT" \
    --enable-lora \
    --lora-paths my_lora="$ADAPTER_PATH" \
    --max-lora-rank 32 \
    --lora-backend triton \
    --moe-runner-backend triton \
    --experts-shared-outer-loras \
    --lora-use-virtual-experts \
    --prefill-attention-backend fa4 \
    --decode-attention-backend fa4

run_bench_scenarios "lora" "--lora-name my_lora"
kill_server

# ══════════════════════════════════════════════════════════════
# Scenario 2: Pure base model (no LoRA)
# ══════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Throughput: Pure base model (no LoRA) - 04-27-2026         ║"
echo "╚══════════════════════════════════════════════════════════════╝"

launch_and_wait "Pure base model (no LoRA)" \
    --model "$MODEL_PATH" \
    --tp "$TP" \
    --port "$PORT" \
    --prefill-attention-backend fa4 \
    --decode-attention-backend fa4

run_bench_scenarios "nolora" ""
kill_server

echo ""
echo "Done. Results in: ${RESULT_DIR}/"
