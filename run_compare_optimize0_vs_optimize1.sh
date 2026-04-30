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

BRANCHES=("lora-perf-optimize-0" "lora-perf-optimize-1")

echo "================================================================"
echo "  Cross-branch Perf Comparison: Qwen3-30B-A3B | TP=${TP}"
echo "  input_len=${INPUT_LEN}  output_len=${OUTPUT_LEN}"
echo "  Branches: ${BRANCHES[*]}"
echo "  Backends: triton, csgmv  (CG ON, virtual experts)"
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

run_bench() {
    local TAG="$1"
    local LORA_ARG="$2"
    local RESULT_DIR="$3"

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

SGLANG_DIR="${SCRIPT_DIR}/sglang"

for BRANCH in "${BRANCHES[@]}"; do
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Branch: ${BRANCH}"
    echo "╚══════════════════════════════════════════════════════════════╝"

    cd "$SGLANG_DIR"
    git checkout "$BRANCH"
    cd "$SCRIPT_DIR"

    RESULT_DIR="${SCRIPT_DIR}/perf_results_${BRANCH}"
    mkdir -p "$RESULT_DIR"

    # Scenario 1: triton
    launch_and_wait "LoRA (triton, CG, virtual experts) [${BRANCH}]" \
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
        --prefill-attention-backend fa4 \
        --decode-attention-backend fa4

    run_bench "lora_triton" "--lora-name my_lora" "$RESULT_DIR"
    kill_server

    # Scenario 2: csgmv
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

    run_bench "lora_csgmv" "--lora-name my_lora" "$RESULT_DIR"
    kill_server
done

# Restore to optimize-1 branch
cd "$SGLANG_DIR"
git checkout lora-perf-optimize-1
cd "$SCRIPT_DIR"

# ══════════════════════════════════════════════════════════════
#  Generate cross-branch comparison tables
# ══════════════════════════════════════════════════════════════
python3 - "${SCRIPT_DIR}" <<'PYEOF'
import json, os, sys

base_dir = sys.argv[1]
branches = ["lora-perf-optimize-0", "lora-perf-optimize-1"]
batch_sizes = [1, 128, 512]
scenarios = [
    ("lora_triton", "LoRA (triton)"),
    ("lora_csgmv",  "LoRA (csgmv)"),
]

def read_last(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    return json.loads(lines[-1]) if lines else None

data = {}
for branch in branches:
    result_dir = os.path.join(base_dir, f"perf_results_{branch}")
    data[branch] = {}
    for tag, _ in scenarios:
        data[branch][tag] = {}
        for bs in batch_sizes:
            data[branch][tag][bs] = read_last(
                os.path.join(result_dir, f"{tag}_bs{bs}.jsonl")
            )

W = 100
baseline = branches[0]
target = branches[1]

for tag, label in scenarios:
    print()
    print("=" * W)
    print(f"  {label}  (TP=4, in=1024, out=2048)")
    print(f"  Baseline: {baseline}  vs  Target: {target}")
    print("=" * W)
    hdr = (f"{'bs':>4} | {'baseline in_tps':>14} | {'target in_tps':>14} | "
           f"{'baseline out_tps':>15} | {'target out_tps':>15} | {'speedup':>8}")
    sep = (f"{'-'*4}-+-{'-'*14}-+-{'-'*14}-+-"
           f"{'-'*15}-+-{'-'*15}-+-{'-'*8}")
    print(hdr)
    print(sep)
    for bs in batch_sizes:
        rb = data[baseline][tag].get(bs)
        rt = data[target][tag].get(bs)
        if rb and rt:
            b_in = rb.get("input_throughput", 0)
            b_out = rb.get("output_throughput", 0)
            t_in = rt.get("input_throughput", 0)
            t_out = rt.get("output_throughput", 0)
            b_tot = rb.get("total_throughput", b_in + b_out)
            t_tot = rt.get("total_throughput", t_in + t_out)
            ratio = f"{t_tot / b_tot:.3f}x" if b_tot > 0 else "N/A"
            print(f"{bs:>4} | {b_in:>12.1f}/s | {t_in:>12.1f}/s | "
                  f"{b_out:>13.1f}/s | {t_out:>13.1f}/s | {ratio:>8}")
        else:
            print(f"{bs:>4} | {'N/A':>14} | {'N/A':>14} | "
                  f"{'N/A':>15} | {'N/A':>15} | {'N/A':>8}")
    print("=" * W)

print()
PYEOF

echo ""
echo "Raw results saved in:"
for BRANCH in "${BRANCHES[@]}"; do
    echo "  ${SCRIPT_DIR}/perf_results_${BRANCH}/"
done
echo "Done."
