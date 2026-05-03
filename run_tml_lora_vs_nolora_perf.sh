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
NUM_PROMPTS=30
WARMUP_PROMPTS=5

RESULT_DIR="${SCRIPT_DIR}/perf_results_tml_lora_vs_nolora"
REPORT_FILE="${RESULT_DIR}/bench_report.md"
mkdir -p "$RESULT_DIR"

echo "================================================================"
echo "  LoRA vs No-LoRA TML Perf Comparison (lora-perf-optimize-3)"
echo "  Model: Qwen3-30B-A3B | TP=${TP} | GPU: GB300"
echo "  Backend: csgmv + virtual experts"
echo "  Test matrix: bs={1,256} x in={256,4096} x out={2,256}"
echo "  Warmup: ${WARMUP_PROMPTS} requests before each measurement"
echo "================================================================"

# Test scenarios: NAME:BATCH_SIZE:INPUT_LEN:OUTPUT_LEN
SCENARIOS=(
    "single:1:256:256"
    "prefill_short:1:256:2"
    "prefill_long:1:4096:2"
    "decode_medium:256:256:256"
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
    sleep 5
}

run_warmup() {
    local LORA_ARG="$1"
    local BS="$2"
    local IN_LEN="$3"
    local OUT_LEN="$4"
    echo "  >> Warmup: ${WARMUP_PROMPTS} reqs (BS=${BS}, in=${IN_LEN}, out=${OUT_LEN})..."
    PYTHONPATH="${SCRIPT_DIR}/sglang/python:$PYTHONPATH" \
    python -m sglang.bench_serving \
        --backend sglang \
        --port "$PORT" \
        --model "$MODEL_PATH" \
        --dataset-name random \
        --num-prompts "$WARMUP_PROMPTS" \
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
        IFS=':' read -r NAME BS IN_LEN OUT_LEN <<< "$SCENARIO"
        local TAG="${PREFIX}_${NAME}"

        run_warmup "$LORA_ARG" "$BS" "$IN_LEN" "$OUT_LEN"

        echo "  >> [${TAG}] BS=${BS} in=${IN_LEN} out=${OUT_LEN} (measurement)"
        PYTHONPATH="${SCRIPT_DIR}/sglang/python:$PYTHONPATH" \
        python -m sglang.bench_serving \
            --backend sglang \
            --port "$PORT" \
            --model "$MODEL_PATH" \
            --dataset-name random \
            --num-prompts "$NUM_PROMPTS" \
            --random-input-len "$IN_LEN" \
            --random-output-len "$OUT_LEN" \
            --request-rate inf \
            --max-concurrency "$BS" \
            ${LORA_ARG} \
            --output-file "${RESULT_DIR}/${TAG}.jsonl" \
            --disable-tqdm
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
    --moe-runner-backend triton \
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
    ("single",        1,   256, 256),
    ("prefill_short", 1,   256,   2),
    ("prefill_long",  1,  4096,   2),
    ("decode_medium", 256, 256, 256),
]

def read_last(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    return json.loads(lines[-1]) if lines else None

def fmt(v, suffix=""):
    return f"{v:,.2f}{suffix}" if v else "N/A"

def get_metrics(r):
    if not r:
        return {}
    return {
        "dur": r.get("duration", 0),
        "lat": r.get("mean_e2e_latency_ms", 0) / 1000 if r.get("mean_e2e_latency_ms") else r.get("mean_e2e_latency", 0),
        "lat99": r.get("p99_e2e_latency_ms", 0) / 1000 if r.get("p99_e2e_latency_ms") else r.get("p99_e2e_latency", 0),
        "ttft": r.get("mean_ttft_ms", 0) / 1000 if r.get("mean_ttft_ms") else r.get("mean_ttft", 0),
        "ttft99": r.get("p99_ttft_ms", 0) / 1000 if r.get("p99_ttft_ms") else r.get("p99_ttft", 0),
        "tpot": r.get("mean_tpot_ms", 0) / 1000 if r.get("mean_tpot_ms") else r.get("mean_tpot", 0),
        "tpot99": r.get("p99_tpot_ms", 0) / 1000 if r.get("p99_tpot_ms") else r.get("p99_tpot", 0),
        "in_tput": r.get("input_throughput", 0),
        "out_tput": r.get("output_throughput", 0),
        "ext_tput": r.get("total_throughput", 0),
        "dec_tput": r.get("decode_throughput", 0),
    }

TABLE_HDR = "| batch_size | input_len | output_len | dur | lat | lat99 | ttft | ttft99 | tpot | tpot99 | in_tput | out_tput | ext_tput | dec_tput |"
TABLE_SEP = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"

def table_row(bs, in_len, out_len, m):
    if not m:
        return f"| {bs} | {in_len} | {out_len} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
    return (
        f"| {bs} | {in_len} | {out_len} "
        f"| {m['dur']:.2f} | {m['lat']:.2f} | {m['lat99']:.2f} "
        f"| {m['ttft']:.4f} | {m['ttft99']:.4f} "
        f"| {m['tpot']:.4f} | {m['tpot99']:.4f} "
        f"| {fmt(m['in_tput'])} | {fmt(m['out_tput'])} "
        f"| {fmt(m['ext_tput'])} | {fmt(m['dec_tput'])} |"
    )

lines = []
lines.append(f"** tp=4 dp=1 ep=1")
lines.append("")

for prefix, label in [("lora", ""), ("nolora", "nolora_")]:
    for name, bs, in_len, out_len in scenarios:
        section = f"{label}{name}" if label else name
        tag = f"{prefix}_{name}"
        r = read_last(os.path.join(result_dir, f"{tag}.jsonl"))
        m = get_metrics(r)

        lines.append(f"## {section}")
        lines.append("")
        lines.append(TABLE_HDR)
        lines.append(TABLE_SEP)
        lines.append(table_row(bs, in_len, out_len, m if m else None))
        lines.append("")

lines.append("")
lines.append("## LoRA vs Base Model Comparison")
lines.append("")
lines.append("| scenario | batch_size | input_len | output_len | lora_ext_tput | base_ext_tput | tput % of base | lora_ttft | base_ttft | ttft % of base | lora_tpot | base_tpot | tpot % of base | lora_dec_tput | base_dec_tput | dec_tput % of base |")
lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")

for name, bs, in_len, out_len in scenarios:
    lr = read_last(os.path.join(result_dir, f"lora_{name}.jsonl"))
    br = read_last(os.path.join(result_dir, f"nolora_{name}.jsonl"))
    lm = get_metrics(lr)
    bm = get_metrics(br)

    if lm and bm:
        tput_pct = f"{lm['ext_tput'] / bm['ext_tput'] * 100:.1f}%" if bm.get("ext_tput") else "N/A"
        ttft_pct = f"{bm['ttft'] / lm['ttft'] * 100:.1f}%" if lm.get("ttft") else "N/A"
        tpot_pct = f"{bm['tpot'] / lm['tpot'] * 100:.1f}%" if lm.get("tpot") else "N/A"
        dec_pct  = f"{lm['dec_tput'] / bm['dec_tput'] * 100:.1f}%" if bm.get("dec_tput") else "N/A"
        lines.append(
            f"| {name} | {bs} | {in_len} | {out_len} "
            f"| {fmt(lm['ext_tput'])} | {fmt(bm['ext_tput'])} | {tput_pct} "
            f"| {lm['ttft']:.4f} | {bm['ttft']:.4f} | {ttft_pct} "
            f"| {lm['tpot']:.4f} | {bm['tpot']:.4f} | {tpot_pct} "
            f"| {fmt(lm['dec_tput'])} | {fmt(bm['dec_tput'])} | {dec_pct} |"
        )
    else:
        lines.append(f"| {name} | {bs} | {in_len} | {out_len} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")

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
