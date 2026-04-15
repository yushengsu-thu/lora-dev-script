#!/bin/bash
#
# Profile Qwen3-30B-A3B: Base (no LoRA) vs LoRA (CG)
# Then convert TP-0 traces to Perfetto-compatible format.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_SCRIPT="${SCRIPT_DIR}/profile_qwen3_moe.sh"
CONVERTER="${SCRIPT_DIR}/convert_to_perfetto_compatible.py"
PROFILE_DIR="${SCRIPT_DIR}/profile_output"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="${SCRIPT_DIR}/profile_perf_Qwen3-30B-A3B_${TIMESTAMP}"
mkdir -p "${RESULT_DIR}"

log() { echo -e "\n[$(date '+%H:%M:%S')] $*\n"; }

cleanup() {
    pkill -9 sglang 2>/dev/null || true
    sleep 3
    ray stop --force 2>/dev/null || true
    pkill -9 ray 2>/dev/null || true
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    sleep 3
}

# ══════════════════════════════════════════════════════════════
#  Scenario 1: Base (no LoRA), CUDA Graph ON, fa4
# ══════════════════════════════════════════════════════════════
log "====== Scenario 1: Base (no LoRA) ======"
cleanup

PREFILL_ATTN_BACKEND=fa4 \
DECODE_ATTN_BACKEND=fa4 \
PROFILE_DIR="${RESULT_DIR}/base_no_lora" \
SERVE_INPUT_LEN=1024 \
SERVE_RANDOM_OUTPUT_LEN=32 \
SERVE_NUM_PROMPTS=10 \
SERVE_MAX_CONCURRENCY=4 \
    bash "${PROFILE_SCRIPT}" serving

log "Scenario 1 done."

# ══════════════════════════════════════════════════════════════
#  Scenario 2: LoRA, CUDA Graph ON, fa4
# ══════════════════════════════════════════════════════════════
log "====== Scenario 2: LoRA (CG) ======"
cleanup

LORA=1 \
PREFILL_ATTN_BACKEND=fa4 \
DECODE_ATTN_BACKEND=fa4 \
PROFILE_DIR="${RESULT_DIR}/lora_cg" \
SERVE_INPUT_LEN=1024 \
SERVE_RANDOM_OUTPUT_LEN=32 \
SERVE_NUM_PROMPTS=10 \
SERVE_MAX_CONCURRENCY=4 \
    bash "${PROFILE_SCRIPT}" serving

log "Scenario 2 done."

# ══════════════════════════════════════════════════════════════
#  Convert TP-0 traces to Perfetto-compatible format
# ══════════════════════════════════════════════════════════════
log "====== Converting TP-0 traces ======"

convert_tp0() {
    local scenario_dir="$1"
    local label="$2"

    local trace_dir
    trace_dir=$(find "${scenario_dir}" -name "*-TP-0.trace.json.gz" -printf '%h\n' | head -1)

    if [[ -z "${trace_dir}" ]]; then
        echo "[WARN] No TP-0 trace found in ${scenario_dir}"
        return
    fi

    local tp0_file
    tp0_file=$(find "${scenario_dir}" -name "*-TP-0.trace.json.gz" | head -1)
    local tp0_basename
    tp0_basename=$(basename "${tp0_file}")

    log "Converting ${label}: ${tp0_basename}"
    python "${CONVERTER}" "${tp0_basename}" \
        --dir-data "${trace_dir}"

    local converted="${trace_dir}/perfetto-compatible-${tp0_basename}"
    if [[ -f "${converted}" ]]; then
        cp "${converted}" "${RESULT_DIR}/${label}-perfetto-TP-0.trace.json.gz"
        log "  -> ${RESULT_DIR}/${label}-perfetto-TP-0.trace.json.gz"
    fi
}

convert_tp0 "${RESULT_DIR}/base_no_lora" "base"
convert_tp0 "${RESULT_DIR}/lora_cg" "lora"

# ══════════════════════════════════════════════════════════════
#  Summary
# ══════════════════════════════════════════════════════════════
log "====== All Done ======"
echo ""
echo "Results in: ${RESULT_DIR}/"
echo ""
ls -lh "${RESULT_DIR}"/*.trace.json.gz 2>/dev/null || echo "(no converted traces found at top level)"
echo ""
echo "Upload the perfetto-TP-0 files to https://ui.perfetto.dev/"
echo ""
