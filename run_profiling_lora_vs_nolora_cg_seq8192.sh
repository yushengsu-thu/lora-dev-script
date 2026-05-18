#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/sglang/python:${PYTHONPATH:-}"
export FLASHINFER_DISABLE_VERSION_CHECK=1

MODEL_PATH="Qwen/Qwen3-30B-A3B-Instruct-2507"
ADAPTER_PATH="${SCRIPT_DIR}/lora_test_cases/Qwen3-30B-A3B-Instruct-2507"
PORT=30000
TP=4

PROFILE_BASE="${SCRIPT_DIR}/lora_profiling_0425_seq_8192"

SERVE_INPUT_LEN=8192
SERVE_OUTPUT_LEN=1024
SERVE_NUM_PROMPTS=10
SERVE_MAX_CONCURRENCY=4

MERGER_SCRIPT="${SCRIPT_DIR}/sglang_profiler_trace_merger.py"
CONVERT_SCRIPT="${SCRIPT_DIR}/convert_to_perfetto_compatible.py"

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
                    local bname
                    bname="$(basename "${gz_file}")"
                    if [[ "${bname}" == perfetto-compatible-* ]] || [[ "${bname}" == merged-* ]]; then
                        continue
                    fi
                    log "  Converting: ${bname}"
                    python "${CONVERT_SCRIPT}" "${bname}" --dir-data "$(dirname "${gz_file}")"
                fi
            done
        fi
    done
}

merge_traces_github() {
    local TRACE_DIR="$1"
    log "Merging TP traces (GitHub merger) in: ${TRACE_DIR}"
    for trace_subdir in "${TRACE_DIR}"/*/; do
        if [[ -d "${trace_subdir}" ]]; then
            local has_tp_traces=false
            for f in "${trace_subdir}"/*TP-*.trace.json.gz; do
                if [[ -f "$f" ]]; then
                    local bname
                    bname="$(basename "$f")"
                    if [[ "${bname}" != merged-* ]]; then
                        has_tp_traces=true
                        break
                    fi
                fi
            done
            if [[ "${has_tp_traces}" == "true" ]]; then
                log "  Merging traces in: ${trace_subdir}"
                python "${MERGER_SCRIPT}" --dir-data "${trace_subdir}"
            fi
        fi
    done
}

# ══════════════════════════════════════════════════════════════
#  Scenario 1: No LoRA + CUDA Graph
# ══════════════════════════════════════════════════════════════
run_no_lora_cg() {
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

    # Run 1: warmup (discard)
    log "Running bench_serving (run 1/2 - warmup, will discard)..."
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
        --profile-prefix no_lora_cg_run1 \
        --profile-output-dir "${PROFILE_DIR}" \
        2>&1 | tee "${PROFILE_DIR}/bench_run1.log"

    log "Deleting warmup traces..."
    sleep 3
    find "${PROFILE_DIR}" -name "*.trace.json.gz" -delete 2>/dev/null || true
    find "${PROFILE_DIR}" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true

    # Run 2: keep
    log "Running bench_serving (run 2/2 - keeping this profile)..."
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
        --profile-prefix no_lora_cg \
        --profile-output-dir "${PROFILE_DIR}" \
        2>&1 | tee "${PROFILE_DIR}/bench_run2.log"

    kill_server
    wait "${SERVER_PID}" 2>/dev/null || true

    log "No-LoRA + CG profile (2nd run) saved to: ${PROFILE_DIR}"
}

# ══════════════════════════════════════════════════════════════
#  Scenario 2: LoRA + CUDA Graph (virtual experts)
# ══════════════════════════════════════════════════════════════
run_lora_cg() {
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
        --lora-backend csgmv \
        --moe-runner-backend triton \
        --experts-shared-outer-loras \
        --lora-use-virtual-experts \
        --prefill-attention-backend fa4 \
        --decode-attention-backend fa4 \
        &
    local SERVER_PID=$!
    wait_for_server

    # Run 1: warmup (discard)
    log "Running bench_serving (run 1/2 - warmup, will discard)..."
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
        --profile-prefix lora_cg_run1 \
        --profile-output-dir "${PROFILE_DIR}" \
        2>&1 | tee "${PROFILE_DIR}/bench_run1.log"

    log "Deleting warmup traces..."
    sleep 3
    find "${PROFILE_DIR}" -name "*.trace.json.gz" -delete 2>/dev/null || true
    find "${PROFILE_DIR}" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true

    # Run 2: keep
    log "Running bench_serving (run 2/2 - keeping this profile)..."
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
        --profile-prefix lora_cg \
        --profile-output-dir "${PROFILE_DIR}" \
        2>&1 | tee "${PROFILE_DIR}/bench_run2.log"

    kill_server
    wait "${SERVER_PID}" 2>/dev/null || true

    log "LoRA + CG profile (2nd run) saved to: ${PROFILE_DIR}"
}

# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════
echo "================================================================"
echo "  Profiling: LoRA CG vs No-LoRA CG (seq_len=${SERVE_INPUT_LEN})"
echo "  Model: ${MODEL_PATH} | TP=${TP}"
echo "  input_len=${SERVE_INPUT_LEN}  output_len=${SERVE_OUTPUT_LEN}"
echo "  num_prompts=${SERVE_NUM_PROMPTS}  concurrency=${SERVE_MAX_CONCURRENCY}"
echo "  Each scenario: 2 runs, keep 2nd profile only"
echo "  Output: ${PROFILE_BASE}"
echo "================================================================"

mkdir -p "${PROFILE_BASE}"
cleanup

# --- Scenario 1: No LoRA + CG ---
run_no_lora_cg
cleanup

# --- Scenario 2: LoRA + CG ---
run_lora_cg
cleanup

# --- Post-processing ---
log "═══ Post-processing: Convert + Merge ═══"

for scenario in no_lora_cg lora_cg; do
    SCENARIO_DIR="${PROFILE_BASE}/${scenario}"
    log "Processing scenario: ${scenario}"

    # Step 1: Convert to Perfetto-compatible format
    convert_traces "${SCENARIO_DIR}"

    # Step 2: Merge TP traces using GitHub merger
    merge_traces_github "${SCENARIO_DIR}"
done

cleanup

echo ""
echo "================================================================"
echo "  Profiling Complete!"
echo "  Results: ${PROFILE_BASE}/"
echo "    no_lora_cg/ — Base model + CUDA Graph (2nd run)"
echo "    lora_cg/    — LoRA model + CUDA Graph (2nd run)"
echo ""
echo "  Each directory contains:"
echo "    - Original TP trace files"
echo "    - perfetto-compatible-* converted traces"
echo "    - merged-* combined TP trace"
echo ""
echo "  View traces at: https://ui.perfetto.dev/"
echo "================================================================"
