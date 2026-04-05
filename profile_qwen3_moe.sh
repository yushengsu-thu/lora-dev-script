#!/bin/bash
#
# Profiling script for Qwen3-30B-A3B-Instruct-2507 on SGLang
#
# Usage:
#   ./profile_qwen3_moe.sh [MODE]
#
# Modes:
#   serving        - Profile with bench_serving (realistic online scenario)
#   one_batch      - Profile with bench_one_batch (kernel-level)
#   offline        - Profile with bench_offline_throughput (max throughput)
#   nsight         - Profile with Nsight Systems
#   nsight_layer   - Nsight + layerwise NVTX markers
#   all            - Run one_batch, offline, and serving sequentially
#
# LoRA profiling (set LORA=1 to enable):
#   LORA=1 ./profile_qwen3_moe.sh serving
#   LORA=1 ./profile_qwen3_moe.sh nsight_layer

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/sglang/python:${PYTHONPATH:-}"
export FLASHINFER_DISABLE_VERSION_CHECK=1

# ============================================================
# Configuration (override via environment variables)
# ============================================================
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
TP="${TP:-4}"
EP_SIZE="${EP_SIZE:-1}"
MOE_BACKEND="${MOE_BACKEND:-auto}"
MOE_A2A="${MOE_A2A:-none}"

LORA="${LORA:-0}"
LORA_NAME="${LORA_NAME:-qwen3_moe_lora}"
LORA_PATH="${LORA_PATH:-/home/radixark/yushengsu/lora_test_cases/Qwen3-30B-A3B-Instruct-2507}"

PROFILE_DIR="${PROFILE_DIR:-/home/radixark/yushengsu/profile_output}"
SERVER_PORT="${SERVER_PORT:-30000}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"

SERVE_NUM_PROMPTS="${SERVE_NUM_PROMPTS:-10}"
SERVE_OUTPUT_LEN="${SERVE_OUTPUT_LEN:-100}"
SERVE_MAX_CONCURRENCY="${SERVE_MAX_CONCURRENCY:-4}"
SERVE_DATASET="${SERVE_DATASET:-random}"
SERVE_INPUT_LEN="${SERVE_INPUT_LEN:-256}"
SERVE_RANDOM_OUTPUT_LEN="${SERVE_RANDOM_OUTPUT_LEN:-32}"

BATCH_SIZE="${BATCH_SIZE:-32}"
INPUT_LEN="${INPUT_LEN:-256}"
OUTPUT_LEN="${OUTPUT_LEN:-32}"

OFFLINE_NUM_PROMPTS="${OFFLINE_NUM_PROMPTS:-10}"
MEM_FRAC="${MEM_FRAC:-0.8}"

DUMMY="${DUMMY:-0}"

PREFILL_ATTN_BACKEND="${PREFILL_ATTN_BACKEND:-}"
DECODE_ATTN_BACKEND="${DECODE_ATTN_BACKEND:-}"

# ============================================================
# Derived settings
# ============================================================
export SGLANG_TORCH_PROFILER_DIR="${PROFILE_DIR}"

LOAD_FMT_FLAG=""
if [[ "${DUMMY}" == "1" ]]; then
    LOAD_FMT_FLAG="--load-format dummy"
    echo "[INFO] Using dummy weights (no real model weights needed)"
fi

EP_FLAG=""
if [[ "${EP_SIZE}" -gt 1 ]]; then
    EP_FLAG="--ep-size ${EP_SIZE}"
fi

MOE_FLAGS=""
if [[ "${MOE_BACKEND}" != "auto" ]]; then
    MOE_FLAGS+=" --moe-runner-backend ${MOE_BACKEND}"
fi
if [[ "${MOE_A2A}" != "none" ]]; then
    MOE_FLAGS+=" --moe-a2a-backend ${MOE_A2A}"
fi

ATTN_FLAGS=""
if [[ -n "${PREFILL_ATTN_BACKEND}" ]]; then
    ATTN_FLAGS+=" --prefill-attention-backend ${PREFILL_ATTN_BACKEND}"
fi
if [[ -n "${DECODE_ATTN_BACKEND}" ]]; then
    ATTN_FLAGS+=" --decode-attention-backend ${DECODE_ATTN_BACKEND}"
fi

LORA_SERVER_FLAGS=""
LORA_BENCH_FLAGS=""
if [[ "${LORA}" == "1" ]]; then
    _LORA_PREFILL="${PREFILL_ATTN_BACKEND:-fa4}"
    _LORA_DECODE="${DECODE_ATTN_BACKEND:-fa4}"
    LORA_SERVER_FLAGS="--enable-lora --lora-paths ${LORA_NAME}=${LORA_PATH} --moe-runner-backend triton --experts-shared-outer-loras --prefill-attention-backend ${_LORA_PREFILL} --decode-attention-backend ${_LORA_DECODE} --enable-cudagraph-gc"
    LORA_BENCH_FLAGS="--lora-name ${LORA_NAME}"
    echo "[INFO] LoRA enabled: ${LORA_NAME} -> ${LORA_PATH}"
    echo "[INFO] LoRA backends: moe-runner=triton, prefill-attn=${_LORA_PREFILL}, decode-attn=${_LORA_DECODE}, experts-shared-outer-loras"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_URL="http://${SERVER_HOST}:${SERVER_PORT}"

mkdir -p "${PROFILE_DIR}"

# ============================================================
# Utility functions
# ============================================================
log() { echo -e "\n[$(date '+%H:%M:%S')] $*\n"; }

wait_for_server() {
    log "Waiting for server at ${BASE_URL} ..."
    local max_wait=600
    local waited=0
    while ! curl -s "${BASE_URL}/health" > /dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if [[ ${waited} -ge ${max_wait} ]]; then
            echo "[ERROR] Server did not start within ${max_wait}s"
            exit 1
        fi
    done
    log "Server is ready (waited ${waited}s)"
}

kill_server() {
    log "Stopping server ..."
    pkill -f "sglang.launch_server.*${SERVER_PORT}" 2>/dev/null || true
    sleep 3
}

print_config() {
    echo "========================================"
    echo " Qwen3-30B-A3B MoE Profiling"
    echo "========================================"
    echo " Model:          ${MODEL_PATH}"
    echo " TP:             ${TP}"
    echo " EP:             ${EP_SIZE}"
    echo " MoE runner:     ${MOE_BACKEND}"
    echo " MoE A2A:        ${MOE_A2A}"
    echo " Dummy weights:  ${DUMMY}"
    if [[ "${LORA}" == "1" ]]; then
    echo " LoRA:           ${LORA_NAME} -> ${LORA_PATH}"
    else
    echo " LoRA:           disabled"
    fi
    if [[ -n "${PREFILL_ATTN_BACKEND}" || -n "${DECODE_ATTN_BACKEND}" ]]; then
    echo " Prefill attn:   ${PREFILL_ATTN_BACKEND:-auto}"
    echo " Decode attn:    ${DECODE_ATTN_BACKEND:-auto}"
    fi
    echo " Profile dir:    ${PROFILE_DIR}"
    echo " Timestamp:      ${TIMESTAMP}"
    echo "========================================"
}

# ============================================================
# Mode: bench_serving
# ============================================================
profile_serving() {
    log "=== Profile Mode: bench_serving ==="

    local sub_dir="${PROFILE_DIR}/serving_${TIMESTAMP}"
    mkdir -p "${sub_dir}"
    export SGLANG_TORCH_PROFILER_DIR="${sub_dir}"

    log "Starting server ..."
    python -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --tp "${TP}" \
        ${EP_FLAG} \
        ${MOE_FLAGS} \
        ${ATTN_FLAGS} \
        ${LOAD_FMT_FLAG} \
        ${LORA_SERVER_FLAGS} \
        --port "${SERVER_PORT}" \
        --host 0.0.0.0 \
        &
    local SERVER_PID=$!

    wait_for_server

    log "Running bench_serving with profiling ..."
    if [[ "${SERVE_DATASET}" == "random" ]]; then
        python -m sglang.bench_serving \
            --backend sglang \
            --base-url "${BASE_URL}" \
            --model "${MODEL_PATH}" \
            --dataset-name random \
            --random-input-len "${SERVE_INPUT_LEN}" \
            --random-output-len "${SERVE_RANDOM_OUTPUT_LEN}" \
            --num-prompts "${SERVE_NUM_PROMPTS}" \
            --max-concurrency "${SERVE_MAX_CONCURRENCY}" \
            ${LORA_BENCH_FLAGS} \
            --profile \
            2>&1 | tee "${sub_dir}/bench_serving.log"
    else
        python -m sglang.bench_serving \
            --backend sglang \
            --base-url "${BASE_URL}" \
            --model "${MODEL_PATH}" \
            --dataset-name sharegpt \
            --sharegpt-output-len "${SERVE_OUTPUT_LEN}" \
            --num-prompts "${SERVE_NUM_PROMPTS}" \
            --max-concurrency "${SERVE_MAX_CONCURRENCY}" \
            ${LORA_BENCH_FLAGS} \
            --profile \
            2>&1 | tee "${sub_dir}/bench_serving.log"
    fi

    kill_server
    wait "${SERVER_PID}" 2>/dev/null || true

    log "Serving profile saved to: ${sub_dir}"
}

# ============================================================
# Mode: bench_one_batch (kernel-level)
# ============================================================
profile_one_batch() {
    log "=== Profile Mode: bench_one_batch ==="
    if [[ "${LORA}" == "1" ]]; then
        echo "[ERROR] bench_one_batch does not support LoRA. Use 'serving' or 'nsight_layer' mode instead."
        exit 1
    fi

    local sub_dir="${PROFILE_DIR}/one_batch_${TIMESTAMP}"
    mkdir -p "${sub_dir}"
    export SGLANG_TORCH_PROFILER_DIR="${sub_dir}"

    log "Running bench_one_batch (prefill + decode) ..."
    python -m sglang.bench_one_batch \
        --model-path "${MODEL_PATH}" \
        --tp "${TP}" \
        ${EP_FLAG} \
        ${MOE_FLAGS} \
        ${LOAD_FMT_FLAG} \
        --batch-size "${BATCH_SIZE}" \
        --input-len "${INPUT_LEN}" \
        --output-len "${OUTPUT_LEN}" \
        --profile \
        --profile-filename-prefix "${sub_dir}/qwen3_moe" \
        2>&1 | tee "${sub_dir}/bench_one_batch.log"

    log "Kernel-level profiles saved to: ${sub_dir}"
}

# ============================================================
# Mode: bench_offline_throughput
# ============================================================
profile_offline() {
    log "=== Profile Mode: bench_offline_throughput ==="
    if [[ "${LORA}" == "1" ]]; then
        echo "[ERROR] bench_offline_throughput does not support LoRA. Use 'serving' or 'nsight_layer' mode instead."
        exit 1
    fi

    local sub_dir="${PROFILE_DIR}/offline_${TIMESTAMP}"
    mkdir -p "${sub_dir}"
    export SGLANG_TORCH_PROFILER_DIR="${sub_dir}"

    log "Running bench_offline_throughput ..."
    python -m sglang.bench_offline_throughput \
        --model-path "${MODEL_PATH}" \
        --tp "${TP}" \
        ${EP_FLAG} \
        ${MOE_FLAGS} \
        ${LOAD_FMT_FLAG} \
        --dataset-name random \
        --random-input-len "${SERVE_INPUT_LEN}" \
        --random-output-len "${SERVE_RANDOM_OUTPUT_LEN}" \
        --num-prompts "${OFFLINE_NUM_PROMPTS}" \
        --profile \
        --mem-frac "${MEM_FRAC}" \
        2>&1 | tee "${sub_dir}/bench_offline.log"

    log "Offline throughput profiles saved to: ${sub_dir}"
}

# ============================================================
# Mode: Nsight Systems
# ============================================================
profile_nsight() {
    log "=== Profile Mode: Nsight Systems ==="
    if [[ "${LORA}" == "1" ]]; then
        echo "[ERROR] Nsight bench_one_batch mode does not support LoRA. Use 'nsight_layer' mode instead."
        exit 1
    fi

    local sub_dir="${PROFILE_DIR}/nsight_${TIMESTAMP}"
    mkdir -p "${sub_dir}"

    if ! command -v nsys &> /dev/null; then
        echo "[ERROR] nsys not found. Install Nsight Systems first."
        exit 1
    fi

    log "Running Nsight profile on bench_one_batch ..."
    nsys profile \
        --trace-fork-before-exec=true \
        --cuda-graph-trace=node \
        --force-overwrite true \
        -o "${sub_dir}/qwen3_moe_nsight" \
        python -m sglang.bench_one_batch \
            --model-path "${MODEL_PATH}" \
            --tp "${TP}" \
            ${EP_FLAG} \
            ${MOE_FLAGS} \
            ${LOAD_FMT_FLAG} \
            --batch-size "${BATCH_SIZE}" \
            --input-len "${INPUT_LEN}" \
            --output-len "${OUTPUT_LEN}" \
            2>&1 | tee "${sub_dir}/nsight.log"

    log "Nsight report saved to: ${sub_dir}"
}

# ============================================================
# Mode: Nsight + layerwise NVTX
# ============================================================
profile_nsight_layerwise() {
    log "=== Profile Mode: Nsight + Layerwise NVTX ==="

    local sub_dir="${PROFILE_DIR}/nsight_layer_${TIMESTAMP}"
    mkdir -p "${sub_dir}"

    if ! command -v nsys &> /dev/null; then
        echo "[ERROR] nsys not found. Install Nsight Systems first."
        exit 1
    fi

    log "Starting server under nsys with layerwise NVTX ..."
    nsys profile \
        --trace-fork-before-exec=true \
        --cuda-graph-trace=node \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --force-overwrite true \
        -o "${sub_dir}/qwen3_moe_layerwise" \
        python -m sglang.launch_server \
            --model-path "${MODEL_PATH}" \
            --tp "${TP}" \
            ${EP_FLAG} \
            ${MOE_FLAGS} \
            ${LOAD_FMT_FLAG} \
            ${LORA_SERVER_FLAGS} \
            --port "${SERVER_PORT}" \
            --host 0.0.0.0 \
            --enable-layerwise-nvtx-marker &
    local NSYS_PID=$!

    wait_for_server

    log "Starting CUDA profiler via HTTP API ..."
    curl -s -X POST "${BASE_URL}/start_profile" \
        -H "Content-Type: application/json" \
        -d '{
            "output_dir": "'"${sub_dir}"'",
            "start_step": 3,
            "num_steps": 10,
            "activities": ["CUDA_PROFILER"]
        }'

    log "Sending benchmark traffic ..."
    python -m sglang.bench_serving \
        --backend sglang \
        --base-url "${BASE_URL}" \
        --model "${MODEL_PATH}" \
        --dataset-name random \
        --random-input-len "${SERVE_INPUT_LEN}" \
        --random-output-len "${SERVE_RANDOM_OUTPUT_LEN}" \
        --num-prompts 50 \
        --max-concurrency 4 \
        ${LORA_BENCH_FLAGS} \
        2>&1 | tee "${sub_dir}/nsight_layer_bench.log"

    sleep 5
    kill_server
    wait "${NSYS_PID}" 2>/dev/null || true

    log "Nsight layerwise report saved to: ${sub_dir}"
}

# ============================================================
# Mode: all
# ============================================================
profile_all() {
    profile_one_batch
    profile_offline
    profile_serving
}

# ============================================================
# Summary
# ============================================================
print_summary() {
    echo ""
    echo "========================================"
    echo " Profiling Complete"
    echo "========================================"
    echo " Results:  ${PROFILE_DIR}"
    echo ""
    echo " View traces:"
    echo "   - https://ui.perfetto.dev/  (upload .trace.json.gz)"
    echo "   - chrome://tracing          (Chrome only)"
    echo "   - nsys-ui <file>.nsys-rep   (Nsight reports)"
    echo "========================================"
}

# ============================================================
# Main
# ============================================================
MODE="${1:-serving}"

print_config

case "${MODE}" in
    serving)         profile_serving ;;
    one_batch)       profile_one_batch ;;
    offline)         profile_offline ;;
    nsight)          profile_nsight ;;
    nsight_layer)    profile_nsight_layerwise ;;
    all)             profile_all ;;
    *)
        echo "Unknown mode: ${MODE}"
        echo "Usage: $0 {serving|one_batch|offline|nsight|nsight_layer|all}"
        exit 1
        ;;
esac

print_summary
