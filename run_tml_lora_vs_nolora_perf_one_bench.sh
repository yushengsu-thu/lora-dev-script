#!/bin/bash
# bench_one_batch_server self-launch (per-BS multi-batch hybrid):
#   For each BS in BATCH_SIZES, launch one server with --max-running-requests=BS
#   and run NUM_RUNS trials of `--batch-size N --enable-multi-batch` where
#   N = BS * WAVES. The first NUM_SKIP trials per BS are discarded as warmup.
#   Each BS gets its own server lifetime (own CUDA graph, own KV alloc) so
#   running-cap is exactly BS, but trials inside one server lifetime so we
#   only pay one server boot per BS.

pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 3
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SCRIPT_DIR
export PYTHONPATH="${SCRIPT_DIR}/sglang/python:${PYTHONPATH:-}"
export FLASHINFER_DISABLE_VERSION_CHECK=1

MODEL_PATH="Qwen/Qwen3-30B-A3B-Instruct-2507"
ADAPTER_PATH="${SCRIPT_DIR}/lora_test_cases/Qwen3-30B-A3B-Instruct-2507"
PORT=30000
TP=4

# RESULT_DIR / REPORT_TAG / LORA_BACKEND can be overridden by env
# (used by run_two_branches_compare.sh to keep per-branch results separate
# and to switch LoRA backend per branch — older branches don't allow
# csgmv + virtual-experts and need triton).
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/perf_results_tml_lora_vs_nolora}"
REPORT_FILE="${RESULT_DIR}/bench_report.md"
REPORT_TAG="${REPORT_TAG:-}"
LORA_BACKEND="${LORA_BACKEND:-csgmv}"

# Some older sglang branches (e.g. 04-27-2026) keep `last_batch` set after
# the internal server warmup, so the very first /flush_cache from
# bench_one_batch_server hits is_fully_idle()=False and returns 400.
# Pass --skip-server-warmup to bypass that internal warmup. Newer branches
# don't need it but it's harmless. Off by default to keep prior runs
# reproducible.
SKIP_SERVER_WARMUP="${SKIP_SERVER_WARMUP:-0}"
SERVER_WARMUP_ARG=()
if [[ "$SKIP_SERVER_WARMUP" == "1" ]]; then
    SERVER_WARMUP_ARG+=(--skip-server-warmup)
fi
mkdir -p "$RESULT_DIR"
rm -f "${RESULT_DIR}"/*.jsonl

# Each BS in BATCH_SIZES: BS = server's running cap (--max-running-requests).
# N = BS * WAVES = total prompts per trial; the surplus over the running cap
# is queued by the scheduler under --enable-multi-batch and promoted batch-by-
# batch as slots free. ext_tput = N*(il+ol)/wall is the only meaningful metric.
# Note: BS=128 had cuda graph illegal-memory bug on this branch, kept ≤64.
BATCH_SIZES=(1 2 4 8 16 32 64)
WAVES=5
INPUT_LEN=8192
OUTPUT_LEN=1024

# Per-BS server lifetime: one boot, NUM_RUNS trials inside, first NUM_SKIP
# trials discarded as warmup (CUDA graph capture for the running cap and
# real-shape prefill happens in the first trial).
NUM_RUNS=4
NUM_SKIP=1

echo "================================================================"
echo "  LoRA vs No-LoRA TML Perf (per-BS multi-batch hybrid)"
echo "  Model: Qwen3-30B-A3B | TP=${TP}"
echo "  Backend: csgmv + virtual experts"
echo "  BSs (running cap): ${BATCH_SIZES[*]}"
echo "  WAVES: ${WAVES}  →  N = BS × WAVES"
echo "  Input len: ${INPUT_LEN}  Output len: ${OUTPUT_LEN}"
echo "  Per BS: 1 server launch × ${NUM_RUNS} trials (first ${NUM_SKIP} discarded)"
echo "================================================================"

cleanup_after_run() {
    echo "  Cleaning up GPU/processes..."
    pkill -9 sglang 2>/dev/null || true
    pkill -9 python 2>/dev/null || true
    sleep 15
}

run_bench() {
    local LABEL="$1"; shift
    local SCENARIO="$1"; shift   # "lora" | "nolora" — used for jsonl filename
    local EXTRA_ARGS=("$@")

    echo ""
    echo "────────────────────────────────────────────────"
    echo "  Running: ${LABEL}"
    echo "────────────────────────────────────────────────"

    for BS in "${BATCH_SIZES[@]}"; do
        local N=$(( BS * WAVES ))
        local RESULT_FILE="${RESULT_DIR}/${SCENARIO}_bs${BS}.jsonl"
        rm -f "$RESULT_FILE"

        # Repeat N value NUM_RUNS times so each entry is one trial inside
        # the same server lifetime, e.g. NUM_RUNS=4 → "$N $N $N $N".
        # Plain bash loop (NOT `yes | head`) because under `set -o pipefail`
        # `yes` gets SIGPIPE → exit 141 → whole script aborts via `set -e`.
        local REPEATED_NS=""
        local _i
        for ((_i = 0; _i < NUM_RUNS; _i++)); do
            REPEATED_NS+="$N "
        done

        echo ""
        echo "  >> [BS=${BS} N=${N} waves=${WAVES}] ${NUM_RUNS} trials in one server"
        # NOTE:
        #   * No --base-url  => bench_one_batch_server launches its own server.
        #   * --max-running-requests=$BS pins the server's running cap so the
        #     `BS` column in the report = actual concurrent batch on GPU.
        #   * --batch-size $REPEATED_NS = one trial per repeated N entry,
        #     all inside the same server lifetime; ttft/cache get reset
        #     between trials by run_one_case via /flush_cache.
        #   * --enable-multi-batch lets N > running cap (= BS); surplus is
        #     scheduler-queued.
        #   * --skip-warmup: bench's internal warmup is hard-coded to
        #     input_len=1024/output_len=16 which is irrelevant at our shape;
        #     our NUM_RUNS≥2 + NUM_SKIP≥1 makes the first trial a real-shape
        #     warmup we discard. Keeps things simple and avoids the
        #     cudaErrorIllegalAddress that small-shape warmup at large bs
        #     used to trigger on the LoRA + virtual-experts kernels.
        python3 -m sglang.bench_one_batch_server \
            --model "$MODEL_PATH" \
            --tp "$TP" \
            --port "$PORT" \
            "${SERVER_WARMUP_ARG[@]}" \
            --max-running-requests "$BS" \
            --batch-size $REPEATED_NS \
            --enable-multi-batch \
            --input-len "$INPUT_LEN" \
            --output-len "$OUTPUT_LEN" \
            --result-filename "$RESULT_FILE" \
            --skip-warmup \
            --show-report \
            "${EXTRA_ARGS[@]}"
        cleanup_after_run
    done
    echo "  >> All BSs done for ${LABEL}."
}

# ══════════════════════════════════════════════════════════════
#  Scenario 1: LoRA enabled (LORA_BACKEND + virtual experts)
# ══════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Scenario 1: LoRA (${LORA_BACKEND} + virtual experts)"
echo "╚══════════════════════════════════════════════════════════════╝"

run_bench "LoRA (${LORA_BACKEND}, CG, virtual experts)" \
    "lora" \
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

# ══════════════════════════════════════════════════════════════
#  Scenario 2: Pure base model (no LoRA enabled)
# ══════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Scenario 2: Pure base model (no LoRA enabled)             ║"
echo "╚══════════════════════════════════════════════════════════════╝"

run_bench "Pure base model (no LoRA)" \
    "nolora" \
    --prefill-attention-backend fa4 \
    --decode-attention-backend fa4

# ══════════════════════════════════════════════════════════════
#  Generate markdown report (per-BS jsonl → unified report w/ BS + N cols)
# ══════════════════════════════════════════════════════════════
python3 - "$RESULT_DIR" "$REPORT_FILE" "$NUM_SKIP" "$REPORT_TAG" "${BATCH_SIZES[@]}" <<'PYEOF'
import json, os, sys

result_dir   = sys.argv[1]
report_file  = sys.argv[2]
num_skip     = int(sys.argv[3])
report_tag   = sys.argv[4]
batch_sizes  = [int(x) for x in sys.argv[5:]]

def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def enrich(r):
    lat_s    = r["latency"]
    ttft_s   = r["last_ttft"]
    ol       = r["output_len"]
    decode_s = max(lat_s - ttft_s, 0)
    tpot_ms  = (decode_s / ol * 1000) if ol > 0 else 0
    r["dur"]      = lat_s
    r["lat"]      = lat_s * 1000
    r["ttft"]     = ttft_s * 1000
    r["tpot"]     = tpot_ms
    r["in_tput"]  = r["input_throughput"]
    r["out_tput"] = r["output_throughput"]
    r["ext_tput"] = r["overall_throughput"]
    r["dec_tput"] = r["last_gen_throughput"]
    return r

AVG_KEYS = ["dur", "lat", "ttft", "tpot", "in_tput", "out_tput", "ext_tput", "dec_tput"]
P99_KEYS = ["lat", "ttft", "tpot"]

def aggregate_one_bs(records, num_skip, bs):
    """Skip first num_skip trials, avg the rest. records all share same N."""
    if not records:
        return None
    kept = records[num_skip:] or records
    n = len(kept)
    agg = {
        "BS": bs,
        "N":  kept[0]["batch_size"],
        "input_len":  kept[0]["input_len"],
        "output_len": kept[0]["output_len"],
        "num_trials": n,
    }
    for k in AVG_KEYS:
        agg[k] = sum(r[k] for r in kept) / n
    for k in P99_KEYS:
        agg[k + "99"] = max(r[k] for r in kept)
    return agg

def collect(scenario):
    rows = []
    for bs in batch_sizes:
        path = os.path.join(result_dir, f"{scenario}_bs{bs}.jsonl")
        records = [enrich(r) for r in load(path)]
        agg = aggregate_one_bs(records, num_skip, bs)
        if agg:
            rows.append(agg)
    return rows

lora_rows   = collect("lora")
nolora_rows = collect("nolora")

TABLE_HDR = ("| BS | N | input_len | output_len | dur(s) | lat(ms) | lat99(ms) "
             "| ttft(ms) | ttft99(ms) | tpot(ms) | tpot99(ms) "
             "| in_tput | out_tput | ext_tput | dec_tput |")
TABLE_SEP = ("| --- | --- | --- | --- | --- | --- | --- "
             "| --- | --- | --- | --- "
             "| --- | --- | --- | --- |")

def table_row(r):
    return (
        f"| {r['BS']} | {r['N']} | {r['input_len']} | {r['output_len']} "
        f"| {r['dur']:.4f} | {r['lat']:.2f} | {r['lat99']:.2f} "
        f"| {r['ttft']:.2f} | {r['ttft99']:.2f} "
        f"| {r['tpot']:.2f} | {r['tpot99']:.2f} "
        f"| {r['in_tput']:.2f} | {r['out_tput']:.2f} "
        f"| {r['ext_tput']:.2f} | {r['dec_tput']:.2f} |"
    )

probe = lora_rows[0] if lora_rows else (nolora_rows[0] if nolora_rows else None)
il = probe['input_len'] if probe else '?'
ol = probe['output_len'] if probe else '?'
total_trials = (num_skip + probe['num_trials']) if probe else '?'
waves = (probe['N'] // probe['BS']) if probe and probe['BS'] else '?'

lines = []
title_suffix = f" [{report_tag}]" if report_tag else ""
lines.append(f"# bench_one_batch_server: LoRA vs No-LoRA (per-BS multi-batch){title_suffix}")
if report_tag:
    lines.append(f"**Branch / tag:** `{report_tag}`")
lines.append(f"**tp=4 dp=1 ep=1 | input_len={il} output_len={ol}**")
lines.append("")
lines.append(
    f"Setup: per-BS server launched with `--max-running-requests=BS` and "
    f"`--batch-size N --enable-multi-batch` where N = BS × {waves}. "
    f"Ran {total_trials} trials per BS in a single server lifetime, "
    f"discarded the first {num_skip} as warmup, averaged the rest. "
    "lat99/ttft99/tpot99 = max of kept trials."
)
lines.append("")
lines.append(
    "> **Note on multi-batch metrics**: under `--enable-multi-batch` with "
    "N > BS, only `lat`/`dur` (wall time) and `ext_tput` (= N × (in+out) / "
    "wall) are physically meaningful. `ttft`, `tpot`, `in_tput`, `out_tput`, "
    "`dec_tput` assume one-shot batching (no queueing) and become misleading "
    "when waves > 1. Use them only for sanity checks, not for base-vs-LoRA "
    "comparison."
)
lines.append("")

for label, rows in [("Base Model (no LoRA)", nolora_rows), ("LoRA", lora_rows)]:
    lines.append(f"## {label}")
    lines.append("")
    if rows:
        lines.append(TABLE_HDR)
        lines.append(TABLE_SEP)
        for r in rows:
            lines.append(table_row(r))
    else:
        lines.append("_(no results)_")
    lines.append("")

report = "\n".join(lines)
print(report)
with open(report_file, "w") as f:
    f.write(report)
print(f"\nReport saved to: {report_file}")
PYEOF

echo ""
echo "Raw results saved in: ${RESULT_DIR}/  (per-BS jsonl: {lora,nolora}_bs<BS>.jsonl)"
echo "Markdown report: ${REPORT_FILE}"
echo "Done."
