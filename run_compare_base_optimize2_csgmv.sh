pkill -9 sglang 2>/dev/null || true
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

SGLANG_DIR="${SCRIPT_DIR}/sglang"
BRANCH="lora-perf-optimize-2"
RESULT_DIR="${SCRIPT_DIR}/perf_results_csgmv_${BRANCH}"
mkdir -p "$RESULT_DIR"

echo "================================================================"
echo "  csgmv Perf: ${BRANCH} | Qwen3-30B-A3B | TP=${TP}"
echo "  input_len=${INPUT_LEN}  output_len=${OUTPUT_LEN}"
echo "  Backend: csgmv  (CG ON, virtual experts)"
echo "  Warmup: ${WARMUP_PROMPTS} requests before each measurement"
echo "================================================================"

cd "$SGLANG_DIR"
git checkout "$BRANCH"
cd "$SCRIPT_DIR"

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
#  LoRA (csgmv, CG, virtual experts) on lora-perf-optimize-2
# ══════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Branch: ${BRANCH}  (csgmv, virtual experts)               ║"
echo "╚══════════════════════════════════════════════════════════════╝"

launch_and_wait "LoRA (csgmv, CG, virtual experts) [${BRANCH}]" \
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

run_bench "lora_csgmv" "--lora-name my_lora"
kill_server

# ══════════════════════════════════════════════════════════════
#  Print results
# ══════════════════════════════════════════════════════════════
python3 - "$RESULT_DIR" <<'PYEOF'
import json, os, sys

result_dir = sys.argv[1]
batch_sizes = [1, 128, 512]
tag = "lora_csgmv"

def read_last(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    return json.loads(lines[-1]) if lines else None

W = 82
print()
print("=" * W)
print(f"  LoRA (csgmv)  lora-perf-optimize-2  (TP=4, in=1024, out=2048, GB300)")
print("=" * W)
hdr = f"{'bs':>4} | {'in_tput':>14} | {'out_tput':>14} | {'e2e_tps':>14}"
sep = f"{'-'*4}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}"
print(hdr)
print(sep)
for bs in batch_sizes:
    r = read_last(os.path.join(result_dir, f"{tag}_bs{bs}.jsonl"))
    if r:
        inp = r.get("input_throughput", 0)
        out = r.get("output_throughput", 0)
        tot = r.get("total_throughput", inp + out)
        print(f"{bs:>4} | {inp:>12.1f}/s | {out:>12.1f}/s | {tot:>12.1f}/s")
    else:
        print(f"{bs:>4} | {'N/A':>14} | {'N/A':>14} | {'N/A':>14}")
print("=" * W)
print()
PYEOF

echo ""
echo "Raw results saved in: ${RESULT_DIR}/"
echo "Done."
