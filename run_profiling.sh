#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/sglang/python:${PYTHONPATH:-}"
export FLASHINFER_DISABLE_VERSION_CHECK=1

MODEL_PATH="Qwen/Qwen3-30B-A3B-Instruct-2507"
ADAPTER_PATH="${SCRIPT_DIR}/lora_test_cases/Qwen3-30B-A3B-Instruct-2507"
PORT=30000
TP=4

PROFILE_BASE="/home/radixark/yushengsu/profiling_0423"

SERVE_INPUT_LEN=256
SERVE_OUTPUT_LEN=32
SERVE_NUM_PROMPTS=10
SERVE_MAX_CONCURRENCY=4

log() { echo -e "\n[$(date '+%H:%M:%S')] $*\n"; }

cleanup() {
    log "Cleaning up all processes..."
    pkill -9 sglang 2>/dev/null || true
    sleep 3
    ray stop --force 2>/dev/null || true
    pkill -9 ray 2>/dev/null || true
    pkill -9 python 2>/dev/null || true
    sleep 3
    pkill -9 ray 2>/dev/null || true
    pkill -9 python 2>/dev/null || true
    sleep 2
}

wait_for_server() {
    log "Waiting for server at http://localhost:${PORT} ..."
    local max_wait=600
    local waited=0
    while ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if [[ ${waited} -ge ${max_wait} ]]; then
            echo "[ERROR] Server did not start within ${max_wait}s"
            exit 1
        fi
        echo "  ... waiting (${waited}s / ${max_wait}s)"
    done
    log "Server is ready (waited ${waited}s)"
}

kill_server() {
    log "Stopping server..."
    pkill -9 sglang 2>/dev/null || true
    sleep 5
}

# ══════════════════════════════════════════════════════════════
#  Scenario 1: No LoRA — Base (CG), fa4 attention
# ══════════════════════════════════════════════════════════════
run_profile_no_lora() {
    local PROFILE_DIR="${PROFILE_BASE}/no_lora"
    mkdir -p "${PROFILE_DIR}"
    export SGLANG_TORCH_PROFILER_DIR="${PROFILE_DIR}"

    log "═══ Scenario: No LoRA (Base + CG) ═══"
    log "Profile output: ${PROFILE_DIR}"

    python -m sglang.launch_server \
        --model "$MODEL_PATH" \
        --tp "$TP" \
        --port "$PORT" \
        --host 0.0.0.0 \
        --prefill-attention-backend fa4 \
        --decode-attention-backend fa4 \
        &
    local SERVER_PID=$!

    wait_for_server

    log "Running bench_serving with --profile (no LoRA)..."
    python -m sglang.bench_serving \
        --backend sglang \
        --port "$PORT" \
        --model "$MODEL_PATH" \
        --dataset-name random \
        --random-input-len "$SERVE_INPUT_LEN" \
        --random-output-len "$SERVE_OUTPUT_LEN" \
        --num-prompts "$SERVE_NUM_PROMPTS" \
        --max-concurrency "$SERVE_MAX_CONCURRENCY" \
        --profile \
        --profile-output-dir "${PROFILE_DIR}" \
        2>&1 | tee "${PROFILE_DIR}/bench_serving.log"

    kill_server
    wait "${SERVER_PID}" 2>/dev/null || true

    log "No-LoRA profile saved to: ${PROFILE_DIR}"
}

# ══════════════════════════════════════════════════════════════
#  Scenario 2: LoRA (CG) — virtual experts, shared outer loras
# ══════════════════════════════════════════════════════════════
run_profile_lora() {
    local PROFILE_DIR="${PROFILE_BASE}/lora"
    mkdir -p "${PROFILE_DIR}"
    export SGLANG_TORCH_PROFILER_DIR="${PROFILE_DIR}"

    log "═══ Scenario: LoRA + CG (virtual experts) ═══"
    log "Profile output: ${PROFILE_DIR}"

    python -m sglang.launch_server \
        --model "$MODEL_PATH" \
        --tp "$TP" \
        --port "$PORT" \
        --host 0.0.0.0 \
        --enable-lora \
        --lora-paths my_lora="$ADAPTER_PATH" \
        --max-lora-rank 32 \
        --lora-backend triton \
        --moe-runner-backend triton \
        --experts-shared-outer-loras \
        --lora-use-virtual-experts \
        --prefill-attention-backend fa4 \
        --decode-attention-backend fa4 \
        &
    local SERVER_PID=$!

    wait_for_server

    log "Running bench_serving with --profile (LoRA)..."
    python -m sglang.bench_serving \
        --backend sglang \
        --port "$PORT" \
        --model "$MODEL_PATH" \
        --dataset-name random \
        --random-input-len "$SERVE_INPUT_LEN" \
        --random-output-len "$SERVE_OUTPUT_LEN" \
        --num-prompts "$SERVE_NUM_PROMPTS" \
        --max-concurrency "$SERVE_MAX_CONCURRENCY" \
        --lora-name my_lora \
        --profile \
        --profile-output-dir "${PROFILE_DIR}" \
        2>&1 | tee "${PROFILE_DIR}/bench_serving.log"

    kill_server
    wait "${SERVER_PID}" 2>/dev/null || true

    log "LoRA profile saved to: ${PROFILE_DIR}"
}

# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════
echo "================================================================"
echo "  Profiling: Qwen3-30B-A3B-Instruct-2507 | TP=${TP}"
echo "  input_len=${SERVE_INPUT_LEN}  output_len=${SERVE_OUTPUT_LEN}"
echo "  num_prompts=${SERVE_NUM_PROMPTS}  concurrency=${SERVE_MAX_CONCURRENCY}"
echo "  Output: ${PROFILE_BASE}"
echo "================================================================"

cleanup

MODE="${1:-all}"

case "${MODE}" in
    no_lora)
        run_profile_no_lora
        ;;
    lora)
        run_profile_lora
        ;;
    all)
        run_profile_no_lora
        cleanup
        run_profile_lora
        ;;
    *)
        echo "Usage: $0 {no_lora|lora|all}"
        exit 1
        ;;
esac

cleanup

echo ""
echo "================================================================"
echo "  Profiling Complete!"
echo "  Results: ${PROFILE_BASE}/"
echo "    no_lora/ — Base model profiling traces"
echo "    lora/    — LoRA model profiling traces"
echo ""
echo "  View traces at: https://ui.perfetto.dev/"
echo "================================================================"
