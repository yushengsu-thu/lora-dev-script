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
SKIP_LORA="${SKIP_LORA:-0}"
SKIP_NOLORA="${SKIP_NOLORA:-0}"
# PROFILE_BY_STAGE=1 → split EXTEND/DECODE into separate trace files (sglang's
# --profile-by-stage). Default (0) keeps prefill+decode in a single profile so
# the whole forward pass is visible on one timeline, which is what we usually
# want for LoRA-vs-noLoRA comparisons.
PROFILE_BY_STAGE="${PROFILE_BY_STAGE:-0}"
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

PROFILE_BY_STAGE_ARG=()
[[ "$PROFILE_BY_STAGE" == "1" ]] && PROFILE_BY_STAGE_ARG=(--profile-by-stage)

MAX_BS=0
for bs in "${BATCH_SIZES[@]}"; do (( bs > MAX_BS )) && MAX_BS=$bs; done

echo "================================================================"
echo "  BS=128 LoRA vs No-LoRA reproduction bench"
echo "  Model: ${MODEL_PATH} | TP=${TP}"
echo "  BSs: ${BATCH_SIZES[*]}  (first ${NUM_WARMUP} = warmup, max-running-requests=${MAX_BS})"
echo "  input_len=${INPUT_LEN}  output_len=${OUTPUT_LEN}"
echo "  LoRA backend: ${LORA_BACKEND}  skip_server_warmup=${SKIP_SERVER_WARMUP}"
echo "  profile_by_stage=${PROFILE_BY_STAGE}  skip_lora=${SKIP_LORA}  skip_nolora=${SKIP_NOLORA}"
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

        echo "  [2/4] Merging TP rank traces (stage-aware)..."
        # The merger expects filenames starting with a numeric profile_id
        # (e.g. "1779022238.3597927-TP-0.trace.json.gz"), but sglang outputs
        # "bs-128-il-2048-1779022238.3597927-TP-0[-<stage>].trace.json.gz".
        # Create temporary symlinks with the expected naming pattern.
        declare -A MERGE_TARGETS=()
        for f in "${PROFILE_SUBDIR%/}"/*.trace.json.gz; do
            [[ -f "$f" ]] || continue
            local bname
            bname=$(basename "$f")
            [[ "$bname" == perfetto-compatible-* ]] && continue
            [[ "$bname" == merged-* ]] && continue
            [[ "$bname" == *bs-1-* ]] && continue
            local extracted_profile_id
            extracted_profile_id=$(echo "$bname" | grep -oP '\d{10,}\.\d+' | head -n 1)
            [[ -z "$extracted_profile_id" ]] && continue

            local stage_tag="all"
            if [[ "$bname" =~ -(EXTEND|DECODE|prefill|decode)\.trace\.json\.gz$ ]]; then
                stage_tag="${BASH_REMATCH[1]}"
            fi
            MERGE_TARGETS["${stage_tag}:${extracted_profile_id}"]=1
        done

        if (( ${#MERGE_TARGETS[@]} == 0 )); then
            echo "    Could not extract profile_id, skipping merge."
        else
            for merge_key in "${!MERGE_TARGETS[@]}"; do
                local stage_tag profile_id_for_merge
                IFS=':' read -r stage_tag profile_id_for_merge <<< "$merge_key"

                local merge_profile_id="$profile_id_for_merge"
                [[ "$stage_tag" != "all" ]] && merge_profile_id="${profile_id_for_merge}-${stage_tag}"

                local linked=0
                for f in "${PROFILE_SUBDIR%/}"/*-${profile_id_for_merge}-TP-*.trace.json.gz; do
                    [[ -f "$f" ]] || continue
                    local bname
                    bname=$(basename "$f")
                    [[ "$bname" == perfetto-compatible-* ]] && continue
                    [[ "$bname" == merged-* ]] && continue
                    [[ "$bname" == *bs-1-* ]] && continue

                    if [[ "$stage_tag" == "all" ]]; then
                        [[ "$bname" =~ -(EXTEND|DECODE|prefill|decode)\.trace\.json\.gz$ ]] && continue
                    else
                        [[ "$bname" =~ -${stage_tag}\.trace\.json\.gz$ ]] || continue
                    fi

                    local tp_part
                    tp_part=$(echo "$bname" | grep -oP 'TP-\d+')
                    [[ -z "$tp_part" ]] && continue
                    ln -sf "$bname" "${PROFILE_SUBDIR%/}/${merge_profile_id}-${tp_part}.trace.json.gz"
                    (( linked++ ))
                done

                if (( linked > 0 )); then
                    python3 "$MERGER_SCRIPT" \
                        --dir-data "${PROFILE_SUBDIR%/}" \
                        --profile-id "${merge_profile_id}" || true
                    # Remove temporary symlinks
                    for f in "${PROFILE_SUBDIR%/}"/${merge_profile_id}-TP-*.trace.json.gz; do
                        [[ -L "$f" ]] && rm -f "$f"
                    done
                fi
            done
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
        "${PROFILE_BY_STAGE_ARG[@]}" \
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
        --disable-piecewise-cuda-graph \
        --prefill-attention-backend fa4 \
        --decode-attention-backend fa4 \
        --lora-name my_lora
    # NOTE: subshell + "|| true" is required. process_profiles uses `set -u`-
    # sensitive constructs (BASH_REMATCH[1], associative arrays) which, if they
    # hit an unbound-variable, would kill the parent shell despite there being
    # no `set -e` — and we MUST reach the no-LoRA scenario below regardless.
    ( process_profiles "lora" ) || echo "  (process_profiles lora failed; continuing to no-LoRA scenario)"
fi

if [[ "$SKIP_NOLORA" != "1" ]]; then
    run_bench "Pure base model (no LoRA, triton MoE for apples-to-apples)" "nolora" \
        --mem-fraction-static 0.82 \
        --moe-runner-backend triton \
        --disable-piecewise-cuda-graph \
        --prefill-attention-backend fa4 \
        --decode-attention-backend fa4
    ( process_profiles "nolora" ) || echo "  (process_profiles nolora failed; raw traces still in profile_nolora/)"
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
