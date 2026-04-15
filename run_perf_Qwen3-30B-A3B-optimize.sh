pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SCRIPT_DIR
export PYTHONPATH="${PYTHONPATH:-}"

MODEL_PATH="Qwen/Qwen3-30B-A3B-Instruct-2507"
ADAPTER_PATH="${SCRIPT_DIR}/lora_test_cases/Qwen3-30B-A3B-Instruct-2507"
PORT=30000
TP=4
INPUT_LEN=1024
OUTPUT_LEN=2048
NUM_PROMPTS=30
BATCH_SIZES=(1 128 512)

RESULT_DIR="${SCRIPT_DIR}/perf_results_Qwen3-30B-A3B-optimize"
mkdir -p "$RESULT_DIR"

echo "================================================================"
echo "  Perf Benchmark (Optimized LoRA): Qwen3-30B-A3B | TP=${TP}"
echo "  input_len=${INPUT_LEN}  output_len=${OUTPUT_LEN}"
echo "  Scenarios:"
echo "    1) Base (CG)           — no LoRA, CUDA Graph ON"
echo "    2) Base + LoRA (no CG) — LoRA, CUDA Graph OFF, virtual experts"
echo "    3) Base + LoRA (CG)    — LoRA, CUDA Graph ON,  virtual experts"
echo "================================================================"

# ── Helper: launch server & wait ──────────────────────────────
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

    local MAX_WAIT=300 ELAPSED=0
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

# ── Helper: kill server ───────────────────────────────────────
kill_server() {
    echo "  Stopping server..."
    kill -9 $SERVER_PID 2>/dev/null
    pkill -9 sglang 2>/dev/null
    sleep 5
}

# ── Helper: run bench for all batch sizes ─────────────────────
run_bench() {
    local TAG="$1"
    local LORA_ARG="$2"

    for BS in "${BATCH_SIZES[@]}"; do
        echo "  >> BS=${BS} [${TAG}]"
        PYTHONPATH="${SCRIPT_DIR}/sglang/python:$PYTHONPATH" \
        python -m sglang.bench_serving \
            --backend sglang \
            --port "$PORT" \
            --model "$MODEL_PATH" \
            --dataset-name random \
            --num-prompts "$NUM_PROMPTS" \
            --random-input-len "$INPUT_LEN" \
            --random-output-len "$OUTPUT_LEN" \
            --request-rate inf \
            --max-concurrency "$BS" \
            ${LORA_ARG} \
            --output-file "${RESULT_DIR}/${TAG}_bs${BS}.jsonl" \
            --disable-tqdm
    done
}

# ══════════════════════════════════════════════════════════════
#  Scenario 1: Base (CG) — no LoRA, CUDA Graph ON
# ══════════════════════════════════════════════════════════════
launch_and_wait "Base (CG)" \
    --model "$MODEL_PATH" \
    --tp "$TP" \
    --port "$PORT" \
    --prefill-attention-backend fa4 \
    --decode-attention-backend fa4

run_bench "base_cg" ""
kill_server

# ══════════════════════════════════════════════════════════════
#  Scenario 2: Base + LoRA (no CG) — LoRA ON, CUDA Graph OFF
#  + virtual experts, shared outer loras
#  [COMMENTED OUT — not needed for this comparison]
# ══════════════════════════════════════════════════════════════
# launch_and_wait "Base + LoRA (no CG, virtual experts)" \
#     --model "$MODEL_PATH" \
#     --tp "$TP" \
#     --port "$PORT" \
#     --enable-lora \
#     --lora-paths my_lora="$ADAPTER_PATH" \
#     --max-lora-rank 32 \
#     --lora-backend triton \
#     --moe-runner-backend triton \
#     --experts-shared-outer-loras \
#     --lora-use-virtual-experts \
#     --disable-cuda-graph \
#     --prefill-attention-backend fa4 \
#     --decode-attention-backend fa4
#
# run_bench "lora_nocg" "--lora-name my_lora"
# kill_server

# ══════════════════════════════════════════════════════════════
#  Scenario 3: Base + LoRA (CG) — LoRA ON, CUDA Graph ON
#  + virtual experts, shared outer loras, cudagraph GC
# ══════════════════════════════════════════════════════════════
launch_and_wait "Base + LoRA (CG, virtual experts)" \
    --model "$MODEL_PATH" \
    --tp "$TP" \
    --port "$PORT" \
    --enable-lora \
    --lora-paths my_lora="$ADAPTER_PATH" \
    --max-lora-rank 32 \
    --lora-backend triton \
    --moe-runner-backend triton \
    --experts-shared-outer-loras \
    --lora-use-virtual-experts \
    --enable-cudagraph-gc \
    --prefill-attention-backend fa4 \
    --decode-attention-backend fa4

run_bench "lora_cg" "--lora-name my_lora"
kill_server

# ══════════════════════════════════════════════════════════════
#  Generate summary tables
# ══════════════════════════════════════════════════════════════
python3 - "$RESULT_DIR" <<'PYEOF'
import json, os, sys

result_dir = sys.argv[1]
batch_sizes = [1, 128, 512]
scenarios = [
    ("base_cg",   "Base (CG)"),
    # ("lora_nocg", "Base + LoRA (no CG, vexp)"),
    ("lora_cg",   "Base + LoRA (CG, vexp)"),
]

def read_last(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    return json.loads(lines[-1]) if lines else None

data = {}
for tag, _ in scenarios:
    data[tag] = {}
    for bs in batch_sizes:
        data[tag][bs] = read_last(os.path.join(result_dir, f"{tag}_bs{bs}.jsonl"))

W = 78
hdr = f"{'bs':>4} | {'in_tput':>14} | {'out_tput':>14} | {'e2e_tps':>14} | {'vs Base':>8}"
sep = f"{'-'*4}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}-+-{'-'*8}"

for tag, label in scenarios:
    print()
    print("=" * W)
    print(f"  {label}  (TP=4, in=1024, out=2048, NVIDIA GB300)")
    print("=" * W)
    print(hdr)
    print(sep)
    for bs in batch_sizes:
        r = data[tag].get(bs)
        base_r = data["base_cg"].get(bs)
        if r:
            inp = r.get("input_throughput", 0)
            out = r.get("output_throughput", 0)
            tot = r.get("total_throughput", inp + out)
            if tag == "base_cg":
                ratio = "---"
            elif base_r:
                base_tot = base_r.get("total_throughput",
                            base_r.get("input_throughput", 0) + base_r.get("output_throughput", 0))
                ratio = f"{tot / base_tot * 100:.1f}%" if base_tot > 0 else "N/A"
            else:
                ratio = "N/A"
            print(f"{bs:>4} | {inp:>12.1f}/s | {out:>12.1f}/s | {tot:>12.1f}/s | {ratio:>8}")
        else:
            print(f"{bs:>4} | {'N/A':>14} | {'N/A':>14} | {'N/A':>14} | {'N/A':>8}")
    print("=" * W)

print()
PYEOF

echo ""
echo "Raw results saved in: ${RESULT_DIR}/"
echo "Done."
