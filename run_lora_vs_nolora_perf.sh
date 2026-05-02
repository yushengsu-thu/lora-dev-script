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
INPUT_LEN=1024
OUTPUT_LEN=2048
NUM_PROMPTS=30
BATCH_SIZES=(1 128 512)
WARMUP_PROMPTS=5

RESULT_DIR="${SCRIPT_DIR}/perf_results_lora_vs_base"
mkdir -p "$RESULT_DIR"

echo "================================================================"
echo "  LoRA vs Base Model Perf (lora-perf-optimize-2)"
echo "  Model: Qwen3-30B-A3B | TP=${TP} | GPU: GB300"
echo "  input_len=${INPUT_LEN}  output_len=${OUTPUT_LEN}"
echo "  Warmup: ${WARMUP_PROMPTS} requests before each measurement"
echo "================================================================"

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

kill_server() {
    echo "  Stopping server..."
    kill -9 $SERVER_PID 2>/dev/null
    pkill -9 sglang 2>/dev/null
    sleep 5
}

run_warmup() {
    local LORA_ARG="$1"
    local BS="$2"
    echo "  >> Warmup: sending ${WARMUP_PROMPTS} requests (BS=${BS})..."
    PYTHONPATH="${SCRIPT_DIR}/sglang/python:$PYTHONPATH" \
    python -m sglang.bench_serving \
        --backend sglang \
        --port "$PORT" \
        --model "$MODEL_PATH" \
        --dataset-name random \
        --num-prompts "$WARMUP_PROMPTS" \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --request-rate inf \
        --max-concurrency "$BS" \
        ${LORA_ARG} \
        --disable-tqdm
    echo "  >> Warmup done."
}

run_bench() {
    local TAG="$1"
    local LORA_ARG="$2"

    for BS in "${BATCH_SIZES[@]}"; do
        run_warmup "$LORA_ARG" "$BS"

        echo "  >> BS=${BS} [${TAG}] (actual measurement)"
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
#  Scenario 1: LoRA (csgmv + virtual experts, best config)
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

run_bench "lora" "--lora-name my_lora"
kill_server

# ══════════════════════════════════════════════════════════════
#  Scenario 2: Pure base model (no LoRA at all)
# ══════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Scenario 2: Pure base model (no LoRA)                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"

launch_and_wait "Base model (no LoRA)" \
    --model "$MODEL_PATH" \
    --tp "$TP" \
    --port "$PORT" \
    --moe-runner-backend triton \
    --prefill-attention-backend fa4 \
    --decode-attention-backend fa4

run_bench "base" ""
kill_server

# ══════════════════════════════════════════════════════════════
#  Generate comparison table
# ══════════════════════════════════════════════════════════════
python3 - "$RESULT_DIR" <<'PYEOF'
import json, os, sys

result_dir = sys.argv[1]
batch_sizes = [1, 128, 512]
scenarios = [
    ("lora", "LoRA (csgmv+VE)"),
    ("base", "Base (no LoRA)"),
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

W = 105
print()
print("=" * W)
print(f"  LoRA vs Base Model  (TP=4, in=1024, out=2048, NVIDIA GB300)")
print(f"  Branch: lora-perf-optimize-2 | LoRA backend: csgmv + virtual experts")
print("=" * W)

hdr = (f"{'bs':>4} |"
       f" {'LoRA in_tps':>13} | {'LoRA out_tps':>14} | {'LoRA total':>12} |"
       f" {'Base in_tps':>13} | {'Base out_tps':>14} | {'Base total':>12} |"
       f" {'LoRA/Base':>10}")
sep = (f"{'-'*4}-+-"
       f"{'-'*13}-+-{'-'*14}-+-{'-'*12}-+-"
       f"{'-'*13}-+-{'-'*14}-+-{'-'*12}-+-"
       f"{'-'*10}")
print(hdr)
print(sep)

for bs in batch_sizes:
    rl = data["lora"].get(bs)
    rb = data["base"].get(bs)

    def get_tput(r):
        if not r:
            return 0, 0, 0
        inp = r.get("input_throughput", 0)
        out = r.get("output_throughput", 0)
        tot = r.get("total_throughput", inp + out)
        return inp, out, tot

    l_in, l_out, l_tot = get_tput(rl)
    b_in, b_out, b_tot = get_tput(rb)

    ratio = f"{l_tot/b_tot*100:.1f}%" if b_tot > 0 and l_tot > 0 else "N/A"

    print(f"{bs:>4} |"
          f" {l_in:>11.1f}/s | {l_out:>12.1f}/s | {l_tot:>10.1f}/s |"
          f" {b_in:>11.1f}/s | {b_out:>12.1f}/s | {b_tot:>10.1f}/s |"
          f" {ratio:>10}")

print("=" * W)
print()
PYEOF

echo ""
echo "Raw results saved in: ${RESULT_DIR}/"
echo "Done."
