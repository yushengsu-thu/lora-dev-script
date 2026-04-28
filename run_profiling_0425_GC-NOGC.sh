#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/sglang/python:${PYTHONPATH:-}"
export FLASHINFER_DISABLE_VERSION_CHECK=1

MODEL_PATH="Qwen/Qwen3-30B-A3B-Instruct-2507"
ADAPTER_PATH="${SCRIPT_DIR}/lora_test_cases/Qwen3-30B-A3B-Instruct-2507"
PORT=30000
TP=4

PROFILE_BASE="${SCRIPT_DIR}/lora_profiling_0425_all"

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

convert_traces() {
    local TRACE_DIR="$1"
    log "Converting traces to Perfetto-compatible format in: ${TRACE_DIR}"
    for trace_subdir in "${TRACE_DIR}"/*/; do
        if [[ -d "${trace_subdir}" ]]; then
            for gz_file in "${trace_subdir}"/*.trace.json.gz; do
                if [[ -f "${gz_file}" ]]; then
                    local basename
                    basename="$(basename "${gz_file}")"
                    if [[ "${basename}" == perfetto-compatible-* ]]; then
                        continue
                    fi
                    log "  Converting: ${basename}"
                    python "${SCRIPT_DIR}/convert_to_perfetto_compatible.py" \
                        "${basename}" \
                        --dir-data "$(dirname "${gz_file}")"
                fi
            done
        fi
    done
}

# ══════════════════════════════════════════════════════════════
#  Scenario 1: No LoRA — with CUDA Graph, fa4 attention
# ══════════════════════════════════════════════════════════════
run_profile_no_lora_cg() {
    local PROFILE_DIR="${PROFILE_BASE}/no_lora_cg"
    mkdir -p "${PROFILE_DIR}"
    export SGLANG_TORCH_PROFILER_DIR="${PROFILE_DIR}"

    log "═══ Scenario: No LoRA + CUDA Graph ═══"
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

    log "Running bench_serving with --profile (no LoRA + CG)..."
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

    convert_traces "${PROFILE_DIR}"

    log "No-LoRA + CG profile saved to: ${PROFILE_DIR}"
}

# ══════════════════════════════════════════════════════════════
#  Scenario 2: No LoRA — without CUDA Graph, fa4 attention
# ══════════════════════════════════════════════════════════════
run_profile_no_lora_no_cg() {
    local PROFILE_DIR="${PROFILE_BASE}/no_lora_no_cg"
    mkdir -p "${PROFILE_DIR}"
    export SGLANG_TORCH_PROFILER_DIR="${PROFILE_DIR}"

    log "═══ Scenario: No LoRA + No CUDA Graph ═══"
    log "Profile output: ${PROFILE_DIR}"

    python -m sglang.launch_server \
        --model "$MODEL_PATH" \
        --tp "$TP" \
        --port "$PORT" \
        --host 0.0.0.0 \
        --prefill-attention-backend fa4 \
        --decode-attention-backend fa4 \
        --disable-cuda-graph \
        &
    local SERVER_PID=$!

    wait_for_server

    log "Running bench_serving with --profile (no LoRA + no CG)..."
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

    convert_traces "${PROFILE_DIR}"

    log "No-LoRA + No-CG profile saved to: ${PROFILE_DIR}"
}

# ══════════════════════════════════════════════════════════════
#  Scenario 3: LoRA — with CUDA Graph, virtual experts
# ══════════════════════════════════════════════════════════════
run_profile_lora_cg() {
    local PROFILE_DIR="${PROFILE_BASE}/lora_cg"
    mkdir -p "${PROFILE_DIR}"
    export SGLANG_TORCH_PROFILER_DIR="${PROFILE_DIR}"

    log "═══ Scenario: LoRA + CUDA Graph (virtual experts) ═══"
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

    log "Running bench_serving with --profile (LoRA + CG)..."
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

    convert_traces "${PROFILE_DIR}"

    log "LoRA + CG profile saved to: ${PROFILE_DIR}"
}

# ══════════════════════════════════════════════════════════════
#  Scenario 4: LoRA — without CUDA Graph, virtual experts
# ══════════════════════════════════════════════════════════════
run_profile_lora_no_cg() {
    local PROFILE_DIR="${PROFILE_BASE}/lora_no_cg"
    mkdir -p "${PROFILE_DIR}"
    export SGLANG_TORCH_PROFILER_DIR="${PROFILE_DIR}"

    log "═══ Scenario: LoRA + No CUDA Graph (virtual experts) ═══"
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
        --disable-cuda-graph \
        &
    local SERVER_PID=$!

    wait_for_server

    log "Running bench_serving with --profile (LoRA + no CG)..."
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

    convert_traces "${PROFILE_DIR}"

    log "LoRA + No-CG profile saved to: ${PROFILE_DIR}"
}

# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════
echo "================================================================"
echo "  Profiling (CG vs No-CG): Qwen3-30B-A3B-Instruct-2507 | TP=${TP}"
echo "  input_len=${SERVE_INPUT_LEN}  output_len=${SERVE_OUTPUT_LEN}"
echo "  num_prompts=${SERVE_NUM_PROMPTS}  concurrency=${SERVE_MAX_CONCURRENCY}"
echo "  Output: ${PROFILE_BASE}"
echo "================================================================"

cleanup

MODE="${1:-all}"

case "${MODE}" in
    no_lora_cg)
        run_profile_no_lora_cg
        ;;
    no_lora_no_cg)
        run_profile_no_lora_no_cg
        ;;
    lora_cg)
        run_profile_lora_cg
        ;;
    lora_no_cg)
        run_profile_lora_no_cg
        ;;
    no_lora)
        run_profile_no_lora_cg
        cleanup
        run_profile_no_lora_no_cg
        ;;
    lora)
        run_profile_lora_cg
        cleanup
        run_profile_lora_no_cg
        ;;
    all)
        run_profile_no_lora_cg
        cleanup
        run_profile_no_lora_no_cg
        cleanup
        run_profile_lora_cg
        cleanup
        run_profile_lora_no_cg
        ;;
    *)
        echo "Usage: $0 {no_lora_cg|no_lora_no_cg|lora_cg|lora_no_cg|no_lora|lora|all}"
        exit 1
        ;;
esac

cleanup

echo ""
echo "================================================================"
echo "  Profiling Complete!"
echo "  Results: ${PROFILE_BASE}/"
echo "    no_lora_cg/    — Base model + CUDA Graph"
echo "    no_lora_no_cg/ — Base model + No CUDA Graph"
echo "    lora_cg/       — LoRA model + CUDA Graph"
echo "    lora_no_cg/    — LoRA model + No CUDA Graph"
echo ""
echo "  View traces at: https://ui.perfetto.dev/"
echo "================================================================"
