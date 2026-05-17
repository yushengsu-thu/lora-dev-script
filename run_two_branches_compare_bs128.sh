#!/bin/bash
# BS=128 sibling of run_two_branches_compare_single_batch.sh.
# Drives run_tml_lora_vs_nolora_perf_one_bench_bs128.sh on each branch in
# BRANCHES, keeps per-branch results in separate folders. Designed to NOT
# abort on a per-branch crash so we can capture both the failure (expected
# on lora-perf-optimize-2) and the comparison (expected to succeed on
# 04-27-2026) in a single run.

set -uo pipefail   # NOTE: no -e — we want to continue past a per-branch crash.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="${SCRIPT_DIR}/sglang"
BENCH_SCRIPT="${SCRIPT_DIR}/run_tml_lora_vs_nolora_perf_one_bench_bs128.sh"

BRANCHES=("lora-perf-optimize-2" "04-27-2026")

declare -A LORA_BACKEND_FOR
LORA_BACKEND_FOR["lora-perf-optimize-2"]="csgmv"
LORA_BACKEND_FOR["04-27-2026"]="triton"

declare -A SKIP_SERVER_WARMUP_FOR
SKIP_SERVER_WARMUP_FOR["lora-perf-optimize-2"]="0"
SKIP_SERVER_WARMUP_FOR["04-27-2026"]="1"

SKIP_BRANCHES="${SKIP_BRANCHES:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"
SKIP_NOLORA="${SKIP_NOLORA:-0}"   # passed through to the inner script

TOPLEVEL_RESULT_DIR="${SCRIPT_DIR}/perf_results_two_branches_bs128"
mkdir -p "$TOPLEVEL_RESULT_DIR"

echo "================================================================"
echo "  Two-branch BS=128 reproduction bench"
echo "  Branches: ${BRANCHES[*]}"
echo "  Top-level result dir: ${TOPLEVEL_RESULT_DIR}"
echo "================================================================"

cd "$SGLANG_DIR"
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: ${SGLANG_DIR} has uncommitted changes."
    git status --short
    exit 1
fi
ORIGINAL_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "Original branch (will restore on exit): ${ORIGINAL_BRANCH}"

restore_branch() {
    local rc=$?
    cd "$SGLANG_DIR"
    echo ""
    echo "Restoring sglang to branch: ${ORIGINAL_BRANCH}"
    git checkout "$ORIGINAL_BRANCH" 2>/dev/null || true
    exit $rc
}
trap restore_branch EXIT INT TERM

cd "$SCRIPT_DIR"

for BR in "${BRANCHES[@]}"; do
    echo ""
    echo "###############################################################"
    echo "#  Branch: ${BR}"
    echo "###############################################################"

    BR_RESULT_DIR="${TOPLEVEL_RESULT_DIR}/${BR}"

    if [[ ",${SKIP_BRANCHES//[ ]/,}," == *",${BR},"* ]]; then
        echo "  >> SKIP_BRANCHES contains '${BR}', skipping."
        continue
    fi
    if [[ "$FORCE_RERUN" != "1" && -s "${BR_RESULT_DIR}/lora.jsonl" ]]; then
        echo "  >> Existing results found at ${BR_RESULT_DIR}/lora.jsonl; skipping."
        echo "     (set FORCE_RERUN=1 to re-run anyway)"
        continue
    fi

    LORA_BE="${LORA_BACKEND_FOR[$BR]:-csgmv}"
    SKIP_SW="${SKIP_SERVER_WARMUP_FOR[$BR]:-0}"

    cd "$SGLANG_DIR"
    git checkout "$BR"
    git log --oneline -1
    cd "$SCRIPT_DIR"

    mkdir -p "$BR_RESULT_DIR"

    echo "  Result dir:          ${BR_RESULT_DIR}"
    echo "  LoRA backend:        ${LORA_BE}"
    echo "  Skip server warmup:  ${SKIP_SW}"
    echo "  Skip nolora:         ${SKIP_NOLORA}"

    set +e
    RESULT_DIR="$BR_RESULT_DIR" \
    LORA_BACKEND="$LORA_BE" \
    SKIP_SERVER_WARMUP="$SKIP_SW" \
    SKIP_NOLORA="$SKIP_NOLORA" \
        bash "$BENCH_SCRIPT"
    BENCH_RC=$?
    set -e
    if (( BENCH_RC != 0 )); then
        echo "  >>> Inner bench script for ${BR} exited ${BENCH_RC} (continuing to next branch)."
    fi
done

# ──────────────────────────────────────────────────────────────
#  Summary: did each branch succeed at BS=128 LoRA?
# ──────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  BS=128 reproduction summary"
echo "================================================================"
printf "%-28s | %-10s | %-10s | %s\n" "branch" "lora.jsonl" "nolora.jsonl" "lora server.log tail"
printf "%-28s-+-%-10s-+-%-10s-+-%s\n" "----------------------------" "----------" "----------" "------------------------"
for BR in "${BRANCHES[@]}"; do
    BR_RESULT_DIR="${TOPLEVEL_RESULT_DIR}/${BR}"
    LORA_JSONL="${BR_RESULT_DIR}/lora.jsonl"
    NOLORA_JSONL="${BR_RESULT_DIR}/nolora.jsonl"
    LORA_LOG="${BR_RESULT_DIR}/lora.server.log"

    LORA_OK="MISS"
    [[ -s "$LORA_JSONL" ]] && LORA_OK="$(wc -l < "$LORA_JSONL") rec"
    NOLORA_OK="MISS"
    [[ -s "$NOLORA_JSONL" ]] && NOLORA_OK="$(wc -l < "$NOLORA_JSONL") rec"

    LOG_HINT="—"
    if [[ -s "$LORA_LOG" ]]; then
        # surface the most likely failure marker
        if grep -qE "illegal memory access|CUDA error|RuntimeError|Traceback" "$LORA_LOG" 2>/dev/null; then
            LOG_HINT="$(grep -E 'illegal memory access|CUDA error|RuntimeError|Traceback' "$LORA_LOG" | head -1 | cut -c1-60)..."
        else
            LOG_HINT="ok"
        fi
    fi
    printf "%-28s | %-10s | %-10s | %s\n" "$BR" "$LORA_OK" "$NOLORA_OK" "$LOG_HINT"
done

echo ""
echo "Per-branch artifacts:"
for BR in "${BRANCHES[@]}"; do
    echo "  ${TOPLEVEL_RESULT_DIR}/${BR}/{lora,nolora}.{jsonl,server.log}"
done
