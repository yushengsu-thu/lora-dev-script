pkill -9 sglang
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 3
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SCRIPT_DIR
export PYTHONPATH="${PYTHONPATH:-}"

MODEL_PATH="Qwen/Qwen3-30B-A3B-Instruct-2507"
ADAPTER_PATH="${SCRIPT_DIR}/lora_test_cases/Qwen3-30B-A3B-Instruct-2507"
PORT=30000
TP=4

NUM_RUNS=3
RESULT_DIR="${SCRIPT_DIR}/perf_results_tml_lora_vs_nolora"
REPORT_FILE="${RESULT_DIR}/bench_report.md"
mkdir -p "$RESULT_DIR"
rm -f "${RESULT_DIR}"/*.jsonl

# Test scenarios: NAME:NUM_PROMPTS:BS:INPUT_LEN:OUTPUT_LEN
echo "================================================================"
echo "  LoRA vs No-LoRA TML Perf Comparison (lora-perf-optimize-3)"
echo "  Model: Qwen3-30B-A3B | TP=${TP} | GPU: GB300"
echo "  Backend: csgmv + virtual experts"
echo "================================================================"
SCENARIOS=(
    # ── Low Latency (BS=1) ──
    "lat_prefill_short:1:1:256:2"
    "lat_prefill_long:1:1:4096:2"
    "lat_prefill_8k:1:1:8192:2"
    "lat_prefill_16k:1:1:16384:2"
    "lat_decode:1:1:256:256"
    # ── High Throughput (BS=64) ──
    "tput_bs64_np64:64:64:256:256"
    "tput_bs64_np128:128:64:256:256"
    "tput_bs64_np256:256:64:256:256"
)

launch_and_wait() {
    local LABEL="$1"; shift
    echo ""
    echo "────────────────────────────────────────────────"
    echo "  Launching server: ${LABEL}"
    echo "────────────────────────────────────────────────"

    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    PYTHONPATH="${SCRIPT_DIR}/sglang/python:$PYTHONPATH" \
    python -m sglang.launch_server "$@" 2>&1 &
    SERVER_PID=$!

    local MAX_WAIT=600 ELAPSED=0
    while ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; do
        sleep 5; ELAPSED=$((ELAPSED + 5))
        if [ $ELAPSED -ge $MAX_WAIT ]; then
            echo "ERROR: Server failed to start within ${MAX_WAIT}s"
            kill -9 $SERVER_PID 2>/dev/null
            exit 1
        fi
        echo "  ... waiting (${ELAPSED}s / ${MAX_WAIT}s)"
    done
    echo "  Server ready! (PID: $SERVER_PID)"
}

kill_server() {
    echo "  Stopping server..."
    kill -9 $SERVER_PID 2>/dev/null
    pkill -9 sglang 2>/dev/null
    pkill -9 python 2>/dev/null
    sleep 15
}

run_warmup() {
    local LORA_ARG="$1"
    local NP="$2"
    local BS="$3"
    local IN_LEN="$4"
    local OUT_LEN="$5"
    echo "  >> Warmup: ${NP} reqs (BS=${BS}, in=${IN_LEN}, out=${OUT_LEN})..."
    PYTHONPATH="${SCRIPT_DIR}/sglang/python:$PYTHONPATH" \
    python -m sglang.bench_serving \
        --backend sglang \
        --port "$PORT" \
        --model "$MODEL_PATH" \
        --dataset-name random \
        --num-prompts "$NP" \
        --random-input-len "$IN_LEN" \
        --random-output-len "$OUT_LEN" \
        --request-rate inf \
        --max-concurrency "$BS" \
        ${LORA_ARG} \
        --disable-tqdm
    echo "  >> Warmup done."
}

run_bench_scenarios() {
    local PREFIX="$1"
    local LORA_ARG="$2"

    for SCENARIO in "${SCENARIOS[@]}"; do
        IFS=':' read -r NAME NP BS IN_LEN OUT_LEN <<< "$SCENARIO"
        local TAG="${PREFIX}_${NAME}"

        run_warmup "$LORA_ARG" "$NP" "$BS" "$IN_LEN" "$OUT_LEN"

        for RUN in $(seq 1 "$NUM_RUNS"); do
            echo "  >> [${TAG}] Run ${RUN}/${NUM_RUNS} num_prompts=${NP} BS=${BS} in=${IN_LEN} out=${OUT_LEN}"
            PYTHONPATH="${SCRIPT_DIR}/sglang/python:$PYTHONPATH" \
            python -m sglang.bench_serving \
                --backend sglang \
                --port "$PORT" \
                --model "$MODEL_PATH" \
                --dataset-name random \
                --num-prompts "$NP" \
                --random-input-len "$IN_LEN" \
                --random-output-len "$OUT_LEN" \
                --request-rate inf \
                --max-concurrency "$BS" \
                ${LORA_ARG} \
                --output-file "${RESULT_DIR}/${TAG}.jsonl" \
                --disable-tqdm
        done
    done
}

# ══════════════════════════════════════════════════════════════
#  Scenario 1: LoRA enabled (csgmv + virtual experts)
# ══════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Scenario 1: LoRA (csgmv + virtual experts)                ║"
echo "╚══════════════════════════════════════════════════════════════╝"

launch_and_wait "LoRA (csgmv, CG, virtual experts)" \
    --model "$MODEL_PATH" \
    --tp "$TP" \
    --port "$PORT" \
    --mem-fraction-static 0.95 \
    --enable-lora \
    --lora-paths my_lora="$ADAPTER_PATH" \
    --max-lora-rank 32 \
    --lora-backend csgmv \
    --moe-runner-backend triton \
    --experts-shared-outer-loras \
    --lora-use-virtual-experts \
    --prefill-attention-backend fa4 \
    --decode-attention-backend fa4

run_bench_scenarios "lora" "--lora-name my_lora"
kill_server

# ══════════════════════════════════════════════════════════════
#  Scenario 2: Pure base model (no LoRA enabled)
# ══════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Scenario 2: Pure base model (no LoRA enabled)             ║"
echo "╚══════════════════════════════════════════════════════════════╝"

launch_and_wait "Pure base model (no LoRA)" \
    --model "$MODEL_PATH" \
    --tp "$TP" \
    --port "$PORT" \
    --prefill-attention-backend fa4 \
    --decode-attention-backend fa4

run_bench_scenarios "nolora" ""
kill_server

# ══════════════════════════════════════════════════════════════
#  Generate markdown report (gist-compatible format)
# ══════════════════════════════════════════════════════════════
python3 - "$RESULT_DIR" "$REPORT_FILE" <<'PYEOF'
import json, os, sys

result_dir = sys.argv[1]
report_file = sys.argv[2]

scenarios = [
    # (name, num_prompts, bs, input_len, output_len)
    # Low Latency (BS=1)
    ("lat_prefill_short",    1,    1,   256,   2),
    ("lat_prefill_long",     1,    1,  4096,   2),
    ("lat_prefill_8k",       1,    1,  8192,   2),
    ("lat_prefill_16k",      1,    1, 16384,   2),
    ("lat_decode",           1,    1,   256, 256),
    # High Throughput (BS=64)
    ("tput_bs64_np64",      64,   64,   256, 256),
    ("tput_bs64_np128",    128,   64,   256, 256),
    ("tput_bs64_np256",    256,   64,   256, 256),
]

METRIC_KEYS = [
    ("completed",  "completed"),
    ("duration",   "duration"),
    ("req_tput",   "request_throughput"),
    ("in_tput",    "input_throughput"),
    ("out_tput",   "output_throughput"),
    ("total_tput", "total_throughput"),
    ("mean_e2e",   "mean_e2e_latency_ms"),
    ("median_e2e", "median_e2e_latency_ms"),
    ("p99_e2e",    "p99_e2e_latency_ms"),
    ("mean_ttft",  "mean_ttft_ms"),
    ("median_ttft","median_ttft_ms"),
    ("p99_ttft",   "p99_ttft_ms"),
    ("mean_tpot",  "mean_tpot_ms"),
    ("median_tpot","median_tpot_ms"),
    ("p99_tpot",   "p99_tpot_ms"),
    ("mean_itl",   "mean_itl_ms"),
    ("median_itl", "median_itl_ms"),
    ("p99_itl",    "p99_itl_ms"),
]

def read_avg(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    if not records:
        return None
    avg = {}
    for short, raw in METRIC_KEYS:
        vals = [r[raw] for r in records if isinstance(r.get(raw), (int, float))]
        avg[short] = sum(vals) / len(vals) if vals else 0
    avg["num_runs"] = len(records)
    return avg

TABLE_HDR = "| number_of_prompts | BS | in_len | out_len | runs | completed | dur(s) | req/s | in_tok/s | out_tok/s | total_tok/s | mean_e2e(ms) | med_e2e | p99_e2e | mean_ttft(ms) | med_ttft | p99_ttft | mean_tpot(ms) | med_tpot | p99_tpot | mean_itl(ms) | med_itl | p99_itl |"
TABLE_SEP = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"

def table_row(np, bs, in_len, out_len, m):
    if not m:
        return f"| {np} | {bs} | {in_len} | {out_len} |" + " N/A |" * 19
    return (
        f"| {np} | {bs} | {in_len} | {out_len} "
        f"| {m['num_runs']} | {m['completed']:.0f} | {m['duration']:.2f} | {m['req_tput']:.2f} "
        f"| {m['in_tput']:.2f} | {m['out_tput']:.2f} | {m['total_tput']:.2f} "
        f"| {m['mean_e2e']:.2f} | {m['median_e2e']:.2f} | {m['p99_e2e']:.2f} "
        f"| {m['mean_ttft']:.2f} | {m['median_ttft']:.2f} | {m['p99_ttft']:.2f} "
        f"| {m['mean_tpot']:.2f} | {m['median_tpot']:.2f} | {m['p99_tpot']:.2f} "
        f"| {m['mean_itl']:.2f} | {m['median_itl']:.2f} | {m['p99_itl']:.2f} |"
    )

lines = []
lines.append(f"** tp=4 dp=1 ep=1 | All metrics are averaged over multiple runs per scenario")
lines.append("")

for prefix, title in [("lora", "LoRA"), ("nolora", "Base Model (no LoRA)")]:
    lines.append(f"## {title}")
    lines.append("")
    lines.append(TABLE_HDR)
    lines.append(TABLE_SEP)
    for name, np, bs, in_len, out_len in scenarios:
        tag = f"{prefix}_{name}"
        m = read_avg(os.path.join(result_dir, f"{tag}.jsonl"))
        lines.append(table_row(np, bs, in_len, out_len, m))
    lines.append("")

def ratio(a, b):
    return f"{a / b:.2f}x" if b else "—"

CMP_HDR = "| number_of_prompts | BS | in | out | lora_mean_e2e(ms) | base_mean_e2e(ms) | e2e_ratio | lora_mean_ttft(ms) | base_mean_ttft(ms) | ttft_ratio | lora_mean_tpot(ms) | base_mean_tpot(ms) | tpot_ratio | lora_mean_itl(ms) | base_mean_itl(ms) | itl_ratio | lora_total_tok/s | base_total_tok/s | tput_ratio |"
CMP_SEP = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"

def cmp_row(np, bs, in_len, out_len, lm, bm):
    return (
        f"| {np} | {bs} | {in_len} | {out_len} "
        f"| {lm['mean_e2e']:.2f} | {bm['mean_e2e']:.2f} | {ratio(lm['mean_e2e'], bm['mean_e2e'])} "
        f"| {lm['mean_ttft']:.2f} | {bm['mean_ttft']:.2f} | {ratio(lm['mean_ttft'], bm['mean_ttft'])} "
        f"| {lm['mean_tpot']:.2f} | {bm['mean_tpot']:.2f} | {ratio(lm['mean_tpot'], bm['mean_tpot'])} "
        f"| {lm['mean_itl']:.2f} | {bm['mean_itl']:.2f} | {ratio(lm['mean_itl'], bm['mean_itl'])} "
        f"| {lm['total_tput']:.2f} | {bm['total_tput']:.2f} | {ratio(lm['total_tput'], bm['total_tput'])} |"
    )

lat_scenarios = [(n, np, bs, il, ol) for n, np, bs, il, ol in scenarios if bs == 1]
tput_scenarios = [(n, np, bs, il, ol) for n, np, bs, il, ol in scenarios if bs > 1]

lines.append("")
lines.append("## Low Latency (BS=1) — LoRA vs Base")
lines.append("")
lines.append("> Key metric: `e2e_ratio` = lora_mean_e2e / base_mean_e2e — closer to 1.0x is better")
lines.append("")
lines.append(CMP_HDR)
lines.append(CMP_SEP)

for name, np, bs, in_len, out_len in lat_scenarios:
    lm = read_avg(os.path.join(result_dir, f"lora_{name}.jsonl"))
    bm = read_avg(os.path.join(result_dir, f"nolora_{name}.jsonl"))
    if lm and bm:
        lines.append(cmp_row(np, bs, in_len, out_len, lm, bm))

lines.append("")
lines.append("## High Throughput (BS=64) — LoRA vs Base")
lines.append("")
lines.append("> Key metric: `tput_ratio` = lora_total_tok/s / base_total_tok/s — closer to 1.0x is better")
lines.append("")
lines.append(CMP_HDR)
lines.append(CMP_SEP)

for name, np, bs, in_len, out_len in tput_scenarios:
    lm = read_avg(os.path.join(result_dir, f"lora_{name}.jsonl"))
    bm = read_avg(os.path.join(result_dir, f"nolora_{name}.jsonl"))
    if lm and bm:
        lines.append(cmp_row(np, bs, in_len, out_len, lm, bm))

lines.append("")

report = "\n".join(lines)
print(report)

with open(report_file, "w") as f:
    f.write(report)
print(f"\nReport saved to: {report_file}")
PYEOF

echo ""
echo "Raw results saved in: ${RESULT_DIR}/"
echo "Markdown report: ${REPORT_FILE}"
echo "Done."
