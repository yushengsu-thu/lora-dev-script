#!/bin/bash
# Generic LoRA perf benchmark for SGLang.
#
# Runs up to three scenarios against the same model, in one shot:
#   - base:     Base model, CUDA Graph on
#   - lora:     Base + LoRA, CUDA Graph on
#   - lora_opt: Base + LoRA, CUDA Graph on, virtual experts + cudagraph-gc
#
# Uses `sglang.bench_one_batch_server` as the client: each BS is measured with
# a single HTTP POST carrying (NUM_WAVES * BS) prompts, while the server is
# launched with --max-running-requests BS. This gives deterministic batching
# (no request staging) and matches the desired overall-throughput metric.
#
# Because --max-running-requests is fixed at launch, the server is relaunched
# for every (scenario, BS) pair.
#
# Downloads the model and LoRA adapter from HuggingFace into ~/models if not
# already present (LoRA adapters are pulled as dataset repos via
# --repo-type dataset).
#
# After all runs complete, invokes summarize_perf.py to produce pretty /
# markdown / tsv tables.
#
# Example:
#   ./run_perf_lora.sh \
#       --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
#       --adapter my-org/my-lora-qwen3-30b \
#       --batch-sizes "1 128 512" \
#       --common-args "--prefill-attention-backend fa4 --decode-attention-backend fa4"
#
#   # DeepSeek V3.1 (MLA + FP8):
#   ./run_perf_lora.sh \
#       --model deepseek-ai/DeepSeek-V3.1-Base \
#       --adapter org/my-lora-ds \
#       --tp 8 \
#       --batch-sizes "1 128 512" \
#       --common-args "--prefill-attention-backend fa4 --decode-attention-backend flashinfer --moe-runner-backend triton --disable-piecewise-cuda-graph" \
#       --lora-extra-args "--disable-shared-experts-fusion"

# ---- stale-process cleanup (before set -e so failures are tolerated) ----
pkill -9 sglang  2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 ray     2>/dev/null || true
pkill -9 python  2>/dev/null || true
sleep 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SCRIPT_DIR
export PYTHONPATH="${PYTHONPATH:-}"

# ---- defaults ----
TP=4
PORT=30000
INPUT_LEN=1024
OUTPUT_LEN=2048
BATCH_SIZES=(1 128 512)
NUM_WAVES=2
MIN_SAMPLES=32
SCENARIOS="base lora lora_opt"
COMMON_ARGS=""
LORA_EXTRA_ARGS=""
RESULT_DIR=""
MODEL_REPO=""
# Adapters: accepts one or many HF repos, space-separated. Server-side names
# are auto-assigned positionally as lora0, lora1, ... (matching input order),
# so we don't need to store them — they're derivable from the array index.
ADAPTER_REPOS=()
# Multi-LoRA sampling (pass-through to bench_one_batch_server). Only relevant
# when >1 adapter is supplied; ignored by the bench script for single-adapter
# runs because there's nothing to sample from.
LORA_REQUEST_DISTRIBUTION="uniform"
LORA_ZIPF_ALPHA="1.1"
MAX_WAIT=900         # server startup timeout; DS-V3 JIT can be slow on a cold cache

# When truthy, skip (scenario, BS) pairs whose result JSONL already exists and
# is non-empty. Useful when iterating on one scenario without re-running all.
# Override ad-hoc per invocation, e.g.:
#   REUSE=1 ./run_perf_lora.sh --model ... --scenarios "lora_opt"
REUSE="${REUSE:-false}"

usage() {
    cat <<EOF
Usage: $0 --model <hf_repo> [--adapter <hf_repo>] [options]

Required:
  --model <hf_repo>             HuggingFace repo for base model (downloaded to ~/models if missing).
  --adapter "<hf_repo> ..."     One or more HuggingFace repos for LoRA adapters
                                (each downloaded as a dataset). With multiple
                                adapters, each prompt in a batch gets one
                                sampled adapter per --lora-request-distribution.
                                Server-side names are auto-assigned lora0,
                                lora1, ... (matches the given order).
                                Required if 'lora' or 'lora_opt' is in --scenarios.

Workload:
  --tp <int>                    Tensor parallel size (default: $TP).
  --port <int>                  Server port (default: $PORT).
  --input-len <int>             Prompt length in tokens (default: $INPUT_LEN).
  --output-len <int>            Generated length in tokens (default: $OUTPUT_LEN).
  --batch-sizes "<a> <b> ..."   Space-separated batch sizes to sweep (default: "${BATCH_SIZES[*]}").
                                Server is relaunched per BS with --max-running-requests=BS.
  --num-waves <int>             Full-batch waves per measurement (default: $NUM_WAVES).
                                Each measurement sends total_prompts prompts in one request;
                                max-running-requests=BS serializes them into ~num_waves waves.
  --min-samples <int>           Floor on total_prompts per measurement (default: $MIN_SAMPLES).
                                total_prompts = max(min_samples, num_waves * BS).
                                Keeps BS=1 runs long enough for stable throughput.

Scenarios:
  --scenarios "<list>"          Subset of: base lora lora_opt (default: all three).

Pass-through server args:
  --common-args "<args>"        Extra SGLang server args appended to ALL scenarios
                                (e.g. attention/MoE backends).
  --lora-extra-args "<args>"    Extra SGLang server args appended to LoRA scenarios only
                                (e.g. --disable-shared-experts-fusion for DS-V3).

Multi-LoRA sampling (only meaningful when >1 adapter given):
  --lora-request-distribution <uniform|distinct|skewed>
                                How to sample a per-prompt adapter within a
                                batch (default: $LORA_REQUEST_DISTRIBUTION).
                                Forwarded to bench_one_batch_server.
  --lora-zipf-alpha <float>     Zipf exponent for 'skewed' sampling; must be
                                > 1 (default: $LORA_ZIPF_ALPHA).

Misc:
  --result-dir <path>           Output directory (default: ./perf_results_<model_basename>).
  --max-wait <sec>              Server startup timeout (default: $MAX_WAIT).
  -h, --help                    Show this help.

Result layout:
  Files are always named base_cg_bs<BS>.jsonl / lora_cg_bs<BS>.jsonl /
  lora_opt_cg_bs<BS>.jsonl. When --result-dir is NOT given, the default
  result-dir encodes #adapters + distribution so multi-LoRA variants can
  coexist side-by-side:
    no adapters           -> perf_results_<model>
    1 adapter             -> perf_results_<model>_1lora
    N>1, uniform|distinct -> perf_results_<model>_<N>lora_<dist>
    N>1, skewed           -> perf_results_<model>_<N>lora_skewed_zipf<alpha>
  Passing --result-dir explicitly disables this auto-suffixing.

Env vars:
  REUSE                         If truthy (1/true/yes), skip any (scenario, BS)
                                whose <tag>_bs<BS>.jsonl already exists and is
                                non-empty. Server launch is skipped too.
                                Example:
                                  REUSE=1 $0 --model ... --scenarios "lora_opt"
EOF
    exit "${1:-0}"
}

# ---- arg parsing ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)            MODEL_REPO="$2";       shift 2 ;;
        --adapter)          read -ra ADAPTER_REPOS <<< "$2"; shift 2 ;;
        --tp)               TP="$2";               shift 2 ;;
        --port)             PORT="$2";             shift 2 ;;
        --input-len)        INPUT_LEN="$2";        shift 2 ;;
        --output-len)       OUTPUT_LEN="$2";       shift 2 ;;
        --batch-sizes)      read -ra BATCH_SIZES <<< "$2"; shift 2 ;;
        --num-waves)        NUM_WAVES="$2";        shift 2 ;;
        --min-samples)      MIN_SAMPLES="$2";      shift 2 ;;
        --scenarios)        SCENARIOS="$2";        shift 2 ;;
        --common-args)      COMMON_ARGS="$2";      shift 2 ;;
        --lora-extra-args)  LORA_EXTRA_ARGS="$2";  shift 2 ;;
        --lora-request-distribution) LORA_REQUEST_DISTRIBUTION="$2"; shift 2 ;;
        --lora-zipf-alpha)  LORA_ZIPF_ALPHA="$2";  shift 2 ;;
        --result-dir)       RESULT_DIR="$2";       shift 2 ;;
        --max-wait)         MAX_WAIT="$2";         shift 2 ;;
        -h|--help)          usage 0 ;;
        *) echo "Unknown arg: $1"; usage 1 ;;
    esac
done

[ -z "$MODEL_REPO" ] && { echo "ERROR: --model is required"; usage 1; }

# Parse scenarios once, validate, and decide whether adapter is needed.
IFS=' ' read -ra SCENARIO_LIST <<< "$SCENARIOS"
NEEDS_ADAPTER=0
for SC in "${SCENARIO_LIST[@]}"; do
    case "$SC" in
        base) ;;
        lora|lora_opt) NEEDS_ADAPTER=1 ;;
        *) echo "ERROR: unknown scenario '$SC' (valid: base lora lora_opt)"; exit 1 ;;
    esac
done
if [ "$NEEDS_ADAPTER" = "1" ] && [ "${#ADAPTER_REPOS[@]}" -eq 0 ]; then
    echo "ERROR: --adapter is required when scenarios include 'lora' or 'lora_opt'"
    usage 1
fi

# Validate multi-LoRA sampling flags early so we fail before the first
# (expensive) server launch. Mirrors bench_one_batch_server's own checks.
case "$LORA_REQUEST_DISTRIBUTION" in
    uniform|distinct|skewed) ;;
    *) echo "ERROR: --lora-request-distribution must be uniform|distinct|skewed (got '$LORA_REQUEST_DISTRIBUTION')"; exit 1 ;;
esac
if [ "$LORA_REQUEST_DISTRIBUTION" != "uniform" ] && [ "${#ADAPTER_REPOS[@]}" -lt 2 ]; then
    echo "ERROR: --lora-request-distribution=$LORA_REQUEST_DISTRIBUTION requires >=2 adapters via --adapter"
    exit 1
fi
# Bash float comparison via awk.
if ! awk -v a="$LORA_ZIPF_ALPHA" 'BEGIN{exit !(a>1)}'; then
    echo "ERROR: --lora-zipf-alpha must be > 1 (got '$LORA_ZIPF_ALPHA')"
    exit 1
fi

# ---- LoRA result-dir suffix ----
# Encode N-adapters + distribution (+ zipf alpha for skewed) in the default
# result-dir name so multiple multi-LoRA variants can coexist side-by-side
# (e.g. perf_results_Qwen3_5lora_distinct/ vs ..._5lora_uniform/) without
# clobbering each other. Files inside the dir keep their simple tag names
# (base_cg_*, lora_cg_*, lora_opt_cg_*) so summarize_perf.py is unchanged.
# If --result-dir is given explicitly, the suffix is not applied — the user
# is in full control of the path.
# Examples:
#   no adapters           -> "" (base only)
#   1 adapter             -> "_1lora"          (distribution is a no-op)
#   5 adapters, distinct  -> "_5lora_distinct"
#   3 adapters, skewed    -> "_3lora_skewed_zipf1.1"
LORA_RESULT_SUFFIX=""
if [ "$NEEDS_ADAPTER" = "1" ]; then
    LORA_RESULT_SUFFIX="_${#ADAPTER_REPOS[@]}lora"
    if [ "${#ADAPTER_REPOS[@]}" -gt 1 ]; then
        LORA_RESULT_SUFFIX="${LORA_RESULT_SUFFIX}_${LORA_REQUEST_DISTRIBUTION}"
        if [ "$LORA_REQUEST_DISTRIBUTION" = "skewed" ]; then
            LORA_RESULT_SUFFIX="${LORA_RESULT_SUFFIX}_zipf${LORA_ZIPF_ALPHA}"
        fi
    fi
fi

MODEL_NAME=$(basename "$MODEL_REPO")
[ -z "$RESULT_DIR" ] && RESULT_DIR="${SCRIPT_DIR}/perf_results_${MODEL_NAME}${LORA_RESULT_SUFFIX}"
mkdir -p "$RESULT_DIR"

# ---- GPU auto-detect ----
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr -s ' ')
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
else
    GPU_NAME="unknown"
    GPU_COUNT=0
fi

# ---- model/adapter download ----
download_if_missing() {
    # $1 = hf repo, $2 = "model"|"dataset"
    local REPO="$1" TYPE="$2"
    local LOCAL_PATH="$HOME/models/$(basename "$REPO")"
    if [ -d "$LOCAL_PATH" ] && [ -n "$(ls -A "$LOCAL_PATH" 2>/dev/null || true)" ]; then
        echo "$LOCAL_PATH"
        return
    fi
    echo "Downloading $REPO (type=$TYPE) -> $LOCAL_PATH ..." >&2
    mkdir -p "$HOME/models"
    local EXTRA=()
    [ "$TYPE" = "dataset" ] && EXTRA=(--repo-type dataset)
    hf download "$REPO" "${EXTRA[@]}" --local-dir "$LOCAL_PATH" >&2
    echo "$LOCAL_PATH"
}

MODEL_PATH=$(download_if_missing "$MODEL_REPO" model)

# Resolve every adapter and build the two arg strings reused by both LoRA
# scenarios. Names are positional (lora0, lora1, ...) so duplicate basenames
# can't collide and the client --lora-name order mirrors the server's
# --lora-paths order.
#   SERVER_LORA_PATHS -> "--lora-paths lora0=/p0 lora1=/p1 ..."
#   CLIENT_LORA_ARGS  -> "--lora-name lora0 lora1 ... [--lora-request-distribution <d>
#                         [--lora-zipf-alpha <a>]]"
SERVER_LORA_PATHS=""
CLIENT_LORA_ARGS=""
if [ "$NEEDS_ADAPTER" = "1" ]; then
    for i in "${!ADAPTER_REPOS[@]}"; do
        local_path=$(download_if_missing "${ADAPTER_REPOS[$i]}" dataset)
        SERVER_LORA_PATHS="${SERVER_LORA_PATHS} lora${i}=${local_path}"
        CLIENT_LORA_ARGS="${CLIENT_LORA_ARGS} lora${i}"
    done
    SERVER_LORA_PATHS="--lora-paths${SERVER_LORA_PATHS}"
    CLIENT_LORA_ARGS="--lora-name${CLIENT_LORA_ARGS}"

    # Sampling flags only matter with >1 adapter (bench_one_batch_server
    # treats a single adapter as broadcast-to-all), so skip them otherwise
    # to keep the command line tidy.
    if [ "${#ADAPTER_REPOS[@]}" -gt 1 ]; then
        CLIENT_LORA_ARGS="${CLIENT_LORA_ARGS} --lora-request-distribution ${LORA_REQUEST_DISTRIBUTION}"
        if [ "$LORA_REQUEST_DISTRIBUTION" = "skewed" ]; then
            CLIENT_LORA_ARGS="${CLIENT_LORA_ARGS} --lora-zipf-alpha ${LORA_ZIPF_ALPHA}"
        fi
    fi
fi

# ---- server lifecycle ----
SERVER_PID=""

launch_and_wait() {
    local LABEL="$1"; shift
    echo ""
    echo "────────────────────────────────────────────────"
    echo "  Launching server: ${LABEL}"
    echo "────────────────────────────────────────────────"

    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    PYTHONPATH="${SCRIPT_DIR}/sglang/python:${PYTHONPATH}" \
    python -m sglang.launch_server "$@" 2>&1 &
    SERVER_PID=$!

    local ELAPSED=0
    while ! curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
        sleep 5; ELAPSED=$((ELAPSED + 5))
        if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
            echo ""
            echo "ERROR: Server did not become ready within ${MAX_WAIT}s"
            kill -9 "$SERVER_PID" 2>/dev/null || true
            exit 1
        fi
        printf "\r  ... waiting (%ds / %ds)" "$ELAPSED" "$MAX_WAIT"
    done
    echo ""
    echo "  Server ready (PID $SERVER_PID)"
}

kill_server() {
    echo "  Stopping server..."
    [ -n "$SERVER_PID" ] && kill -9 "$SERVER_PID" 2>/dev/null || true
    pkill -9 sglang 2>/dev/null || true
    SERVER_PID=""
    sleep 5
}

# Clean up on abnormal exit
trap 'echo ""; echo "Caught exit; cleaning up..."; kill_server 2>/dev/null || true' EXIT

# ---- reuse-existing-results gate ----
# Returns 0 (skip) when REUSE is truthy AND the target result file already
# exists and is non-empty; 1 (run) otherwise.
should_skip_bench() {
    local TAG="$1" BS="$2"
    local OUT_FILE="${RESULT_DIR}/${TAG}_bs${BS}.jsonl"
    case "$REUSE" in
        1|true|True|TRUE|yes|Yes|YES)
            if [ -s "$OUT_FILE" ]; then
                echo "  >> [${TAG}] BS=${BS}  SKIP: reusing $OUT_FILE (REUSE=$REUSE)"
                return 0
            fi
            ;;
    esac
    return 1
}

# ---- single measurement (one BS, one scenario) ----
run_one_bench() {
    # $1 = tag (e.g. "base_cg"), $2 = BS, $3 = LoRA client args string
    local TAG="$1" BS="$2" LORA_FLAG="$3"
    local TOTAL=$(( NUM_WAVES * BS ))
    [ "$TOTAL" -lt "$MIN_SAMPLES" ] && TOTAL="$MIN_SAMPLES"
    local OUT_FILE="${RESULT_DIR}/${TAG}_bs${BS}.jsonl"
    # bench_one_batch_server opens the result file in append mode, and
    # summarize_perf.py always reads the last non-empty line. Keep previous
    # records so reruns don't destroy data; the freshest run wins in the
    # summary, and older lines stay available for ad-hoc inspection.
    echo "  >> [${TAG}] BS=${BS}  waves=${NUM_WAVES}  total_prompts=${TOTAL}  (max-running-requests=${BS})"
    # shellcheck disable=SC2086
    PYTHONPATH="${SCRIPT_DIR}/sglang/python:${PYTHONPATH}" \
    python -m sglang.bench_one_batch_server \
        --model None \
        --base-url "http://localhost:${PORT}" \
        --batch-size "$TOTAL" \
        --input-len "$INPUT_LEN" \
        --output-len "$OUTPUT_LEN" \
        --dataset-name random \
        --skip-warmup \
        --enable-multi-batch \
        --run-name "${TAG}_bs${BS}" \
        --result-filename "$OUT_FILE" \
        ${LORA_FLAG}
}

# ---- scenario runners (relaunch server per BS) ----
run_scenario_base() {
    for BS in "${BATCH_SIZES[@]}"; do
        if should_skip_bench "base_cg" "$BS"; then continue; fi
        # shellcheck disable=SC2086
        launch_and_wait "Base (CG) BS=${BS}" \
            --model "$MODEL_PATH" \
            --tp "$TP" \
            --port "$PORT" \
            --max-running-requests "$BS" \
            ${COMMON_ARGS}
        run_one_bench "base_cg" "$BS" ""
        kill_server
    done
}

run_scenario_lora() {
    for BS in "${BATCH_SIZES[@]}"; do
        if should_skip_bench "lora_cg" "$BS"; then continue; fi
        # shellcheck disable=SC2086
        launch_and_wait "Base + LoRA (CG) BS=${BS}" \
            --model "$MODEL_PATH" \
            --tp "$TP" \
            --port "$PORT" \
            --max-running-requests "$BS" \
            --enable-lora \
            ${SERVER_LORA_PATHS} \
            --lora-backend triton \
            ${COMMON_ARGS} ${LORA_EXTRA_ARGS}
        run_one_bench "lora_cg" "$BS" "${CLIENT_LORA_ARGS}"
        kill_server
    done
}

run_scenario_lora_opt() {
    for BS in "${BATCH_SIZES[@]}"; do
        if should_skip_bench "lora_opt_cg" "$BS"; then continue; fi
        # shellcheck disable=SC2086
        launch_and_wait "Base + LoRA (CG, optimized) BS=${BS}" \
            --model "$MODEL_PATH" \
            --tp "$TP" \
            --port "$PORT" \
            --max-running-requests "$BS" \
            --enable-lora \
            ${SERVER_LORA_PATHS} \
            --lora-backend triton \
            --lora-use-virtual-experts \
            --enable-cudagraph-gc \
            ${COMMON_ARGS} ${LORA_EXTRA_ARGS}
        run_one_bench "lora_opt_cg" "$BS" "${CLIENT_LORA_ARGS}"
        kill_server
    done
}

# ---- banner ----
echo "================================================================"
echo "  SGLang LoRA Perf Benchmark (bench_one_batch_server)"
echo "  Model:        $MODEL_REPO  ($MODEL_PATH)"
if [ "${#ADAPTER_REPOS[@]}" -eq 0 ]; then
    echo "  Adapters:     <none>"
else
    for i in "${!ADAPTER_REPOS[@]}"; do
        echo "  Adapter[$i]:   lora${i} <- ${ADAPTER_REPOS[$i]}"
    done
    if [ "${#ADAPTER_REPOS[@]}" -gt 1 ]; then
        echo "  Multi-LoRA:   distribution=${LORA_REQUEST_DISTRIBUTION}$([ "$LORA_REQUEST_DISTRIBUTION" = "skewed" ] && echo "  zipf_alpha=${LORA_ZIPF_ALPHA}")"
    fi
fi
echo "  GPU:          ${GPU_NAME} x${GPU_COUNT}"
echo "  TP:           $TP"
echo "  Input/Output: ${INPUT_LEN} / ${OUTPUT_LEN}"
echo "  BatchSizes:   ${BATCH_SIZES[*]}"
echo "  Prompts/meas: total_prompts = max(${MIN_SAMPLES}, ${NUM_WAVES} * BS)"
echo "  Scenarios:    ${SCENARIOS}"
echo "  Common args:  ${COMMON_ARGS:-<none>}"
echo "  LoRA-extra:   ${LORA_EXTRA_ARGS:-<none>}"
echo "  Result dir:   $RESULT_DIR"
echo "  LoRA tag:     ${LORA_RESULT_SUFFIX:-<none>}   (encoded in default result-dir name)"
echo "  Reuse cache:  REUSE=${REUSE}"
echo "================================================================"

# ---- dispatch scenarios ----
for SC in "${SCENARIO_LIST[@]}"; do
    case "$SC" in
        base)     run_scenario_base ;;
        lora)     run_scenario_lora ;;
        lora_opt) run_scenario_lora_opt ;;
    esac
done

# Disable the EXIT trap before the summary step; servers are already killed.
trap - EXIT

# ---- summary ----
echo ""
echo "Generating summary tables..."
python3 "${SCRIPT_DIR}/summarize_perf.py" \
    --result-dir "$RESULT_DIR" \
    --batch-sizes "${BATCH_SIZES[*]}" \
    --scenarios  "$SCENARIOS" \
    --gpu        "$GPU_NAME" \
    --tp         "$TP" \
    --input-len  "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN"

echo ""
echo "Done. Raw JSONL: ${RESULT_DIR}/*.jsonl"
echo "     Markdown:   ${RESULT_DIR}/summary.md   (paste into Google Docs)"
echo "     TSV:        ${RESULT_DIR}/summary.tsv  (paste into Google Sheets)"
