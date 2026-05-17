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
INPUT_LEN=2048
OUTPUT_LEN=128
# (1) is warmup; (128) is the production trial we actually want to read.
BATCH_SIZES=(1 128)
NUM_WARMUP=1

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/perf_results_comp/${TIMESTAMP}}"
LORA_BACKEND="${LORA_BACKEND:-csgmv}"
SKIP_SERVER_WARMUP="${SKIP_SERVER_WARMUP:-0}"
SKIP_NOLORA="${SKIP_NOLORA:-0}"
mkdir -p "$RESULT_DIR"
rm -f "${RESULT_DIR}"/*.jsonl

TORCH_UTILS_DIR="${SCRIPT_DIR}/torch_utils"
CONVERT_SCRIPT="${TORCH_UTILS_DIR}/convert_to_perfetto_compatible.py"
MERGER_SCRIPT="${TORCH_UTILS_DIR}/sglang_profiler_trace_merger.py"
if [[ ! -f "$CONVERT_SCRIPT" ]] || [[ ! -f "$MERGER_SCRIPT" ]]; then
    echo "Downloading torch_utils scripts..."
    mkdir -p "$TORCH_UTILS_DIR"
    curl -sL "https://raw.githubusercontent.com/fzyzcjy/torch_utils/master/src/convert_to_perfetto_compatible/convert_to_perfetto_compatible.py" \
        -o "$CONVERT_SCRIPT"
    curl -sL "https://raw.githubusercontent.com/fzyzcjy/torch_utils/master/src/torch_profile_trace_merger/sglang_profiler_trace_merger.py" \
        -o "$MERGER_SCRIPT"
fi
pip install -q orjson typer 2>/dev/null || true

SERVER_WARMUP_ARG=()
[[ "$SKIP_SERVER_WARMUP" == "1" ]] && SERVER_WARMUP_ARG=(--skip-server-warmup)

MAX_BS=0
for bs in "${BATCH_SIZES[@]}"; do (( bs > MAX_BS )) && MAX_BS=$bs; done

echo "================================================================"
echo "  BS=128 LoRA vs No-LoRA reproduction bench"
echo "  Model: ${MODEL_PATH} | TP=${TP}"
echo "  BSs: ${BATCH_SIZES[*]}  (first ${NUM_WARMUP} = warmup, max-running-requests=${MAX_BS})"
echo "  input_len=${INPUT_LEN}  output_len=${OUTPUT_LEN}"
echo "  LoRA backend: ${LORA_BACKEND}  skip_server_warmup=${SKIP_SERVER_WARMUP}"
echo "================================================================"

cleanup() {
    echo "  Cleaning up GPU/processes..."
    pkill -9 sglang python 2>/dev/null || true
    sleep 15
}

process_profiles() {
    local SCENARIO="$1"
    local PROFILE_BASE="${RESULT_DIR}/profile_${SCENARIO}"

    if [[ ! -d "$PROFILE_BASE" ]]; then
        echo "  No profile directory found for ${SCENARIO}, skipping post-processing."
        return 0
    fi

    local found=0
    for PROFILE_SUBDIR in "$PROFILE_BASE"/*/; do
        [[ -d "$PROFILE_SUBDIR" ]] || continue
        found=1
        echo ""
        echo "  ── Profile post-processing: ${PROFILE_SUBDIR}"

        echo "  [1/4] Converting traces to Perfetto-compatible format (skipping bs-1)..."
        for f in "$PROFILE_SUBDIR"/*.trace.json.gz; do
            [[ -f "$f" ]] || continue
            [[ "$(basename "$f")" == *bs-1-* ]] && continue
            echo "    → $(basename "$f")"
            python3 "$CONVERT_SCRIPT" \
                "$(basename "$f")" \
                --dir-data "${PROFILE_SUBDIR%/}" || true
        done

        echo "  [2/4] Merging TP rank traces..."
        # The merger expects filenames starting with a numeric profile_id
        # (e.g. "1779022238.3597927-TP-0.trace.json.gz"), but sglang outputs
        # "bs-128-il-2048-1779022238.3597927-TP-0.trace.json.gz".
        # Create temporary symlinks with the expected naming pattern.
        local PROFILE_ID=""
        for f in "${PROFILE_SUBDIR%/}"/*-TP-0.trace.json.gz; do
            [[ -f "$f" ]] || continue
            [[ "$(basename "$f")" == *bs-1-* ]] && continue
            PROFILE_ID=$(basename "$f" | grep -oP '\d{10,}\.\d+')
            break
        done
        if [[ -n "$PROFILE_ID" ]]; then
            for f in "${PROFILE_SUBDIR%/}"/*-${PROFILE_ID}-TP-*.trace.json.gz; do
                [[ -f "$f" ]] || continue
                local bname
                bname=$(basename "$f")
                [[ "$bname" == perfetto-compatible-* ]] && continue
                [[ "$bname" == *bs-1-* ]] && continue
                local tp_part
                tp_part=$(echo "$bname" | grep -oP 'TP-\d+')
                ln -sf "$bname" "${PROFILE_SUBDIR%/}/${PROFILE_ID}-${tp_part}.trace.json.gz"
            done
            python3 "$MERGER_SCRIPT" --dir-data "${PROFILE_SUBDIR%/}" || true
            # Remove temporary symlinks
            for f in "${PROFILE_SUBDIR%/}"/${PROFILE_ID}-TP-*.trace.json.gz; do
                [[ -L "$f" ]] && rm -f "$f"
            done
        else
            echo "    Could not extract profile_id, skipping merge."
        fi

        echo "  [3/4] Converting merged trace to Perfetto-compatible format..."
        for f in "$PROFILE_SUBDIR"/merged-*.trace.json.gz; do
            [[ -f "$f" ]] || continue
            echo "    → $(basename "$f")"
            python3 "$CONVERT_SCRIPT" \
                "$(basename "$f")" \
                --dir-data "${PROFILE_SUBDIR%/}" || true
        done

        echo "  [4/4] Cleaning up intermediate files and bs-1 results..."
        for f in "${PROFILE_SUBDIR%/}"/*.trace.json.gz; do
            [[ -f "$f" ]] || continue
            local bname
            bname=$(basename "$f")
            if [[ "$bname" == *bs-1-* ]]; then
                rm -f "$f"
                echo "    rm $(basename "$f") (bs-1)"
                continue
            fi
            [[ "$bname" == perfetto-compatible-* ]] && continue
            rm -f "$f"
            echo "    rm $(basename "$f")"
        done
    done

    if (( found == 0 )); then
        echo "  No profile trace subdirectories found in ${PROFILE_BASE}"
    else
        echo "  Profile post-processing complete for ${SCENARIO}"
    fi
}

run_bench() {
    local LABEL="$1" SCENARIO="$2"; shift 2
    local RESULT_JSONL="${RESULT_DIR}/${SCENARIO}.jsonl"
    local SERVER_LOG="${RESULT_DIR}/${SCENARIO}.server.log"
    local PROFILE_OUTPUT="${RESULT_DIR}/profile_${SCENARIO}"
    mkdir -p "$PROFILE_OUTPUT"
    echo ""
    echo "──────────────────────────────────────────────"
    echo "  Running: ${LABEL}"
    echo "  Server stderr/stdout → ${SERVER_LOG}"
    echo "  Profile output       → ${PROFILE_OUTPUT}"
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
        --profile \
        --profile-output-dir "$PROFILE_OUTPUT" \
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
process_profiles "lora"

if [[ "$SKIP_NOLORA" != "1" ]]; then
    run_bench "Pure base model (no LoRA)" "nolora" \
        --prefill-attention-backend fa4 \
        --decode-attention-backend fa4
    process_profiles "nolora"
fi

echo ""
echo "================================================================"
echo "  Done.  All results in: ${RESULT_DIR}"
echo "  Benchmark:  ${RESULT_DIR}/{lora,nolora}.jsonl"
echo "  Server logs: ${RESULT_DIR}/{lora,nolora}.server.log"
echo "  Profiles:    ${RESULT_DIR}/profile_{lora,nolora}/"
echo "    (perfetto-compatible-*.json.gz       = per-TP converted traces)"
echo "    (perfetto-compatible-merged-*.json.gz = TP-merged + converted)"
echo "================================================================"
