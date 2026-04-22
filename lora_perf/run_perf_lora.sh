#!/bin/bash
# Generic LoRA perf benchmark for SGLang.
#
# Runs up to three scenarios per (tp, ep, batch-size) combo:
#   - base:     Base model, CUDA Graph on
#   - lora:     Base + LoRA, CUDA Graph on
#   - lora_opt: Base + LoRA, CUDA Graph on, virtual experts + cudagraph-gc
# (CUDA Graph is on for every scenario, so the filename no longer tags it.)
#
# --tp and --ep each accept a single value or a space-separated list. The
# full cartesian product runs back-to-back; each combo writes to its own
# subfolder under a shared outer result dir. When >1 combo runs, a combined
# sweep summary is written at the outer dir.
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
# --repo-type dataset). Absolute local paths (starting with "/") are used
# as-is — no download, no copy.
#
# Note: --ep 1 is SGLang's own default (ep_size=1, i.e. no expert parallelism).
# Pass --ep 1 when you want EP disabled — there is no "none" sentinel. For
# non-MoE models, EP is ignored anyway.
#
# Example (single combo):
#   ./run_perf_lora.sh \
#       --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
#       --adapter my-org/my-lora-qwen3-30b \
#       --tp 4 --ep 4 \
#       --batch-sizes "1 128 512"
#
# Example (sweep):
#   ./run_perf_lora.sh \
#       --model deepseek-ai/DeepSeek-V3.1-Base \
#       --adapter org/my-lora-ds \
#       --tp "4 8" --ep "1 8" \
#       --batch-sizes "1 128"
#   # -> 4 combos: (tp=4,ep=1), (tp=4,ep=8), (tp=8,ep=1), (tp=8,ep=8)

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
TPS=(4)
# Expert-parallel sizes. --ep 1 matches SGLang's default (ep_size=1, i.e.
# no expert parallelism); pass --ep 1 explicitly to disable EP. Ignored by
# non-MoE models.
EPS=(1)
PORT=30000
INPUT_LEN=1024
OUTPUT_LEN=2048
BATCH_SIZES=(1 128 512)
NUM_WAVES=2
MIN_SAMPLES=32
SCENARIOS="base lora lora_opt"
COMMON_ARGS=""
LORA_EXTRA_ARGS=""
OUTER_DIR=""
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
REUSE="${REUSE:-false}"

usage() {
    cat <<EOF
Usage: $0 --model <hf_repo> [--adapter <hf_repo>] [options]

Required:
  --model <hf_repo>             HuggingFace repo for base model (downloaded to ~/models if missing).
  --adapter "<hf_repo> ..."     One or more HuggingFace repos for LoRA adapters
                                (each downloaded as a dataset). Server-side
                                names are auto-assigned lora0, lora1, ...
                                Required if 'lora' or 'lora_opt' is in --scenarios.

Workload (each supports one value or a space-separated list; full product runs):
  --tp "<a> [b ...]"            Tensor parallel sizes (default: ${TPS[*]}).
  --ep "<a> [b ...]"            Expert parallel sizes (default: ${EPS[*]}).
                                Use --ep 1 to disable EP (SGLang's default).
                                Ignored by non-MoE models.
  --batch-sizes "<a> [b ...]"   Batch sizes to sweep (default: "${BATCH_SIZES[*]}").
                                Server is relaunched per BS with --max-running-requests=BS.

  --port <int>                  Server port (default: $PORT).
  --input-len <int>             Prompt length in tokens (default: $INPUT_LEN).
  --output-len <int>            Generated length in tokens (default: $OUTPUT_LEN).
  --num-waves <int>             Full-batch waves per measurement (default: $NUM_WAVES).
                                Each measurement sends total_prompts prompts in one request;
                                max-running-requests=BS serializes them into ~num_waves waves.
  --min-samples <int>           Floor on total_prompts per measurement (default: $MIN_SAMPLES).
                                total_prompts = max(min_samples, num_waves * BS).

Scenarios:
  --scenarios "<list>"          Subset of: base lora lora_opt (default: all three).

Pass-through server args:
  --common-args "<args>"        Extra SGLang server args for ALL scenarios.
  --lora-extra-args "<args>"    Extra SGLang server args for LoRA scenarios only.

Multi-LoRA sampling (only meaningful when >1 adapter given):
  --lora-request-distribution <uniform|distinct|skewed>   (default: $LORA_REQUEST_DISTRIBUTION)
  --lora-zipf-alpha <float>     Zipf exponent for 'skewed'; > 1 (default: $LORA_ZIPF_ALPHA).

Misc:
  --result-dir <path>           Override the OUTER result dir. Inner tp/ep
                                subdirs and file names are unaffected.
  --max-wait <sec>              Server startup timeout (default: $MAX_WAIT).
  -h, --help                    Show this help.

Result layout:
    <outer_dir>/
        tp<TP>_ep<EP>/
            base_bs<BS>.jsonl
            lora_bs<BS>.jsonl
            lora_opt_bs<BS>.jsonl
            summary.md
            summary.tsv
        ...                                (one subdir per combo)
        combined_summary.md                (only when >1 combo)
        combined_summary.tsv

  Default <outer_dir> is:
    perf_results_<model>[_<loratag>]

  Where _<loratag> is:
      1 adapter             -> _1lora
      N>1, uniform|distinct -> _<N>lora_<dist>
      N>1, skewed           -> _<N>lora_skewed_zipf<alpha>

Env vars:
  REUSE                         Truthy (1/true/yes) skips any (scenario, BS)
                                whose result file already exists and is non-empty.
EOF
    exit "${1:-0}"
}

# ---- arg parsing ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)            MODEL_REPO="$2";       shift 2 ;;
        --adapter)          read -ra ADAPTER_REPOS <<< "$2"; shift 2 ;;
        --tp)               read -ra TPS <<< "$2"; shift 2 ;;
        --ep)               read -ra EPS <<< "$2"; shift 2 ;;
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
        --result-dir)       OUTER_DIR="$2";        shift 2 ;;
        --max-wait)         MAX_WAIT="$2";         shift 2 ;;
        -h|--help)          usage 0 ;;
        *) echo "Unknown arg: $1"; usage 1 ;;
    esac
done

[ -z "$MODEL_REPO" ] && { echo "ERROR: --model is required"; usage 1; }
[ "${#TPS[@]}" -eq 0 ] && { echo "ERROR: --tp resolved to an empty list"; exit 1; }
[ "${#EPS[@]}" -eq 0 ] && { echo "ERROR: --ep resolved to an empty list"; exit 1; }

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
# Encode #adapters + distribution (+ zipf alpha for skewed) into the outer
# dir name so LoRA variants coexist side-by-side.
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
[ -z "$OUTER_DIR" ] && OUTER_DIR="${SCRIPT_DIR}/perf_results_${MODEL_NAME}${LORA_RESULT_SUFFIX}"
mkdir -p "$OUTER_DIR"

# ---- tee stdout/stderr to a per-run log so failures are post-mortem-able ----
# Timestamped so reruns coexist. Placed after OUTER_DIR creation so the log
# lives with the per-run results. Early arg-parse failures still surface to
# the terminal normally.
LOG_FILE="${OUTER_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log: $LOG_FILE"

# ---- GPU auto-detect ----
if command -v nvidia-smi >/dev/null 2>&1; then
    # Query once to avoid `nvidia-smi | head -1` SIGPIPE under `set -o pipefail`
    # (head closes the pipe, nvidia-smi gets SIGPIPE -> exit 141 -> script aborts).
    GPU_LIST=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)
    GPU_NAME=$(printf '%s\n' "$GPU_LIST" | awk 'NR==1' | tr -s ' ')
    GPU_COUNT=$(printf '%s\n' "$GPU_LIST" | grep -c . | tr -d ' ')
else
    GPU_NAME="unknown"
    GPU_COUNT=0
fi

# ---- filter (tp, ep) combos to legal ones ----
# Users can throw any mix of tp/ep values at the script; we drop (with a
# warning) the combos SGLang would reject at launch. Rules enforced:
#   * tp <= #gpus                        (TP shards weights across GPUs)
#   * ep <= tp                           (ep_size * moe_dp_size <= tp_size)
#   * ep == 1  OR  ep == tp              (moe_dp_size defaults to 1; when
#                                         ep > 1, SGLang asserts
#                                         ep_size * moe_dp_size == tp_size)
# The third rule can be relaxed by also setting moe_dp_size>1 via
# --common-args, but we don't auto-detect that — users who need such configs
# should run those combos explicitly.
VALID_TPS=()
VALID_EPS=()
SKIPPED_COUNT=0
for _tp in "${TPS[@]}"; do
    for _ep in "${EPS[@]}"; do
        _reason=""
        if [ "$GPU_COUNT" -gt 0 ] && [ "$_tp" -gt "$GPU_COUNT" ]; then
            _reason="tp=${_tp} > available GPUs (${GPU_COUNT})"
        elif [ "$_ep" -gt "$_tp" ]; then
            _reason="ep=${_ep} > tp=${_tp}; SGLang requires ep_size * moe_dp_size <= tp_size"
        elif [ "$_ep" -ne 1 ] && [ "$_ep" -ne "$_tp" ]; then
            _reason="ep=${_ep} is neither 1 nor tp=${_tp}; with default moe_dp_size=1, SGLang requires ep ∈ {1, tp}"
        fi
        if [ -n "$_reason" ]; then
            echo "WARN: skipping combo tp=${_tp} ep=${_ep} — ${_reason}"
            SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        else
            VALID_TPS+=("$_tp")
            VALID_EPS+=("$_ep")
        fi
    done
done
unset _tp _ep _reason
NUM_COMBOS=${#VALID_TPS[@]}
if [ "$NUM_COMBOS" -eq 0 ]; then
    echo "ERROR: no legal (tp, ep) combos to run (all ${SKIPPED_COUNT} requested combos were filtered out)"
    exit 1
fi

# ---- model/adapter download ----
download_if_missing() {
    # $1 = hf repo or absolute local path, $2 = "model"|"dataset"
    local REPO="$1" TYPE="$2"
    # Absolute local path: use it as-is (no download, no copy). Error out
    # if it doesn't exist rather than letting `hf download` try to resolve
    # a path as an HF repo id.
    if [[ "$REPO" == /* ]]; then
        if [ -d "$REPO" ]; then
            echo "$REPO"
            return
        fi
        echo "ERROR: local path '$REPO' does not exist" >&2
        exit 1
    fi
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
trap 'echo ""; echo "Caught exit; cleaning up..."; echo "Log: $LOG_FILE"; kill_server 2>/dev/null || true' EXIT

# ---- reuse-existing-results gate ----
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
    # $1 = tag (e.g. "base"), $2 = BS, $3 = LoRA client args string
    local TAG="$1" BS="$2" LORA_FLAG="$3"
    local TOTAL=$(( NUM_WAVES * BS ))
    [ "$TOTAL" -lt "$MIN_SAMPLES" ] && TOTAL="$MIN_SAMPLES"
    local OUT_FILE="${RESULT_DIR}/${TAG}_bs${BS}.jsonl"
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

# ---- scenario runners (relaunch server per BS; read $TP/$EP_ARG/$RESULT_DIR set by combo loop) ----
run_scenario_base() {
    for BS in "${BATCH_SIZES[@]}"; do
        if should_skip_bench "base" "$BS"; then continue; fi
        # shellcheck disable=SC2086
        launch_and_wait "Base TP=${TP} EP=${EP} BS=${BS}" \
            --model "$MODEL_PATH" \
            --tp "$TP" \
            ${EP_ARG} \
            --port "$PORT" \
            --max-running-requests "$BS" \
            ${COMMON_ARGS}
        run_one_bench "base" "$BS" ""
        kill_server
    done
}

run_scenario_lora() {
    for BS in "${BATCH_SIZES[@]}"; do
        if should_skip_bench "lora" "$BS"; then continue; fi
        # shellcheck disable=SC2086
        launch_and_wait "Base + LoRA TP=${TP} EP=${EP} BS=${BS}" \
            --model "$MODEL_PATH" \
            --tp "$TP" \
            ${EP_ARG} \
            --port "$PORT" \
            --max-running-requests "$BS" \
            --enable-lora \
            ${SERVER_LORA_PATHS} \
            --lora-backend triton \
            ${COMMON_ARGS} ${LORA_EXTRA_ARGS}
        run_one_bench "lora" "$BS" "${CLIENT_LORA_ARGS}"
        kill_server
    done
}

run_scenario_lora_opt() {
    for BS in "${BATCH_SIZES[@]}"; do
        if should_skip_bench "lora_opt" "$BS"; then continue; fi
        # shellcheck disable=SC2086
        launch_and_wait "Base + LoRA (optimized) TP=${TP} EP=${EP} BS=${BS}" \
            --model "$MODEL_PATH" \
            --tp "$TP" \
            ${EP_ARG} \
            --port "$PORT" \
            --max-running-requests "$BS" \
            --enable-lora \
            ${SERVER_LORA_PATHS} \
            --lora-backend triton \
            --lora-use-virtual-experts \
            --enable-cudagraph-gc \
            ${COMMON_ARGS} ${LORA_EXTRA_ARGS}
        run_one_bench "lora_opt" "$BS" "${CLIENT_LORA_ARGS}"
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
echo "  TP sweep:     ${TPS[*]}"
echo "  EP sweep:     ${EPS[*]}"
echo "  BatchSizes:   ${BATCH_SIZES[*]}"
echo "  Combos:       ${NUM_COMBOS} valid / $(( ${#TPS[@]} * ${#EPS[@]} )) requested (${SKIPPED_COUNT} skipped)"
echo "  Planned runs (tp, ep):"
for ((i = 0; i < NUM_COMBOS; i++)); do
    echo "                tp=${VALID_TPS[$i]}  ep=${VALID_EPS[$i]}"
done
echo "  Input/Output: ${INPUT_LEN} / ${OUTPUT_LEN}"
echo "  Prompts/meas: total_prompts = max(${MIN_SAMPLES}, ${NUM_WAVES} * BS)"
echo "  Scenarios:    ${SCENARIOS}"
echo "  Common args:  ${COMMON_ARGS:-<none>}"
echo "  LoRA-extra:   ${LORA_EXTRA_ARGS:-<none>}"
echo "  Outer dir:    $OUTER_DIR"
echo "  Log file:     $LOG_FILE"
echo "  Reuse cache:  REUSE=${REUSE}"
echo "================================================================"

# ---- dispatch: run each (TP, EP) combo ----
RESULT_DIRS=()
COMBO_ARGS=()

for ((i = 0; i < NUM_COMBOS; i++)); do
    TP="${VALID_TPS[$i]}"
    EP="${VALID_EPS[$i]}"
    EP_ARG="--ep $EP"
    RESULT_DIR="${OUTER_DIR}/tp${TP}_ep${EP}"
    mkdir -p "$RESULT_DIR"

    echo ""
    echo "################################################################"
    echo "  Combo: TP=${TP}  EP=${EP}  BS=${BATCH_SIZES[*]}"
    echo "  Result dir: $RESULT_DIR"
    echo "################################################################"

    for SC in "${SCENARIO_LIST[@]}"; do
        case "$SC" in
            base)     run_scenario_base ;;
            lora)     run_scenario_lora ;;
            lora_opt) run_scenario_lora_opt ;;
        esac
    done

    echo ""
    echo "Per-combo summary (TP=${TP} EP=${EP}):"
    python3 "${SCRIPT_DIR}/summarize_perf.py" \
        --result-dir "$RESULT_DIR" \
        --batch-sizes "${BATCH_SIZES[*]}" \
        --scenarios  "$SCENARIOS" \
        --gpu        "$GPU_NAME" \
        --tp         "$TP" \
        --ep         "$EP" \
        --input-len  "$INPUT_LEN" \
        --output-len "$OUTPUT_LEN"

    RESULT_DIRS+=("$RESULT_DIR")
    COMBO_ARGS+=(--combo "tp=${TP},ep=${EP},dir=${RESULT_DIR}")
done

# Disable the EXIT trap before the summary step; servers are already killed.
trap - EXIT

# ---- combined sweep summary (only when >1 combo) ----
if [ "$NUM_COMBOS" -gt 1 ]; then
    echo ""
    echo "Generating combined sweep summary across ${NUM_COMBOS} combos..."
    python3 "${SCRIPT_DIR}/summarize_perf.py" \
        --combine \
        "${COMBO_ARGS[@]}" \
        --batch-sizes "${BATCH_SIZES[*]}" \
        --scenarios  "$SCENARIOS" \
        --gpu        "$GPU_NAME" \
        --input-len  "$INPUT_LEN" \
        --output-len "$OUTPUT_LEN" \
        --out-dir    "$OUTER_DIR"
    echo ""
    echo "Combined summary:"
    echo "  Markdown: ${OUTER_DIR}/combined_summary.md"
    echo "  TSV:      ${OUTER_DIR}/combined_summary.tsv"
fi

echo ""
echo "Done. Per-combo result dirs:"
for D in "${RESULT_DIRS[@]}"; do
    echo "  $D"
done
