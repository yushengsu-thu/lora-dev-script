#!/bin/bash
# BS doubling sweep on lora-perf-optimize-2 LoRA only, looking for the BS
# threshold where the #24262 align kernel triggers illegal memory access.
#
# All BSs are run from one server invocation (one model load), with
# /flush_cache between trials. If the server crashes at BS=N, BSs < N still
# get persisted to lora_sweep.jsonl and the crash trace lands in
# lora_sweep.server.log.

pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 ray python 2>/dev/null || true
sleep 5

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/sglang/python:${PYTHONPATH:-}"
export FLASHINFER_DISABLE_VERSION_CHECK=1

# Pin sglang to the buggy branch.
( cd "${SCRIPT_DIR}/sglang" && git checkout lora-perf-optimize-2 )

MODEL_PATH="Qwen/Qwen3-30B-A3B-Instruct-2507"
ADAPTER_PATH="${SCRIPT_DIR}/lora_test_cases/Qwen3-30B-A3B-Instruct-2507"
PORT=30000
TP=4
INPUT_LEN="${INPUT_LEN:-2048}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
LORA_BACKEND="csgmv"

# Sweep: warmup at BS=1, then 128 → 256 → 512 → 1024.
# Override via:  BATCH_SIZES_OVERRIDE="1 128 256 512 1024 2048"  bash this.sh
if [[ -n "${BATCH_SIZES_OVERRIDE:-}" ]]; then
    read -ra BATCH_SIZES <<< "$BATCH_SIZES_OVERRIDE"
else
    BATCH_SIZES=(1 128 256 512 1024)
fi
NUM_WARMUP=1

RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/perf_results_bs_sweep_opt2}"
mkdir -p "$RESULT_DIR"
rm -f "$RESULT_DIR"/*.jsonl "$RESULT_DIR"/*.log

MAX_BS=0
for bs in "${BATCH_SIZES[@]}"; do (( bs > MAX_BS )) && MAX_BS=$bs; done

# Memory headroom: mem-fraction climbs with the largest BS we want to fit.
# At IL=2048/OL=128, per-GPU KV demand on TP=4 is roughly 0.1 GB/req → safe
# up to BS=2048 with mem-fraction 0.88 on 284 GB GB300s. We bump it linearly.
if   (( MAX_BS <= 512 ));  then MEM_FRAC=0.82
elif (( MAX_BS <= 1024 )); then MEM_FRAC=0.85
else                            MEM_FRAC=0.88
fi

echo "=========================================================="
echo "  BS doubling sweep on lora-perf-optimize-2 (LoRA only)"
echo "  Model: ${MODEL_PATH} | TP=${TP}"
echo "  BSs:   ${BATCH_SIZES[*]}  (first ${NUM_WARMUP} = warmup, max=${MAX_BS})"
echo "  IL/OL: ${INPUT_LEN}/${OUTPUT_LEN}"
echo "  Backend: ${LORA_BACKEND}, virtual experts"
echo "  cuda-graph-max-bs: ${MAX_BS}   mem-fraction-static: ${MEM_FRAC}"
echo "  Result dir: ${RESULT_DIR}"
echo "=========================================================="

RESULT_JSONL="${RESULT_DIR}/lora_sweep.jsonl"
SERVER_LOG="${RESULT_DIR}/lora_sweep.server.log"

set +e
python3 -m sglang.bench_one_batch_server \
    --model "$MODEL_PATH" \
    --tp "$TP" \
    --port "$PORT" \
    --max-running-requests "$MAX_BS" \
    --cuda-graph-max-bs "$MAX_BS" \
    --batch-size "${BATCH_SIZES[@]}" \
    --input-len "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN" \
    --result-filename "$RESULT_JSONL" \
    --skip-warmup \
    --show-report \
    --mem-fraction-static "$MEM_FRAC" \
    --enable-lora \
    --lora-paths my_lora="$ADAPTER_PATH" \
    --max-lora-rank 32 \
    --lora-backend "$LORA_BACKEND" \
    --moe-runner-backend triton \
    --experts-shared-outer-loras \
    --lora-use-virtual-experts \
    --prefill-attention-backend fa4 \
    --decode-attention-backend fa4 \
    --lora-name my_lora 2>&1 | tee "$SERVER_LOG"
RC=${PIPESTATUS[0]}
set -e

echo
echo "=========================================================="
echo "  Sweep done. exit=${RC}"
echo "  jsonl records:"
wc -l "$RESULT_JSONL" 2>/dev/null || echo "    (no jsonl)"
echo
echo "  Per-BS records:"
if [[ -s "$RESULT_JSONL" ]]; then
    python3 - "$RESULT_JSONL" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    for ln in f:
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        print(f"  BS={r['batch_size']:>5}  IL={r['input_len']}  OL={r['output_len']}  "
              f"lat={r['latency']:.2f}s  ttft={r['last_ttft']:.3f}s  "
              f"out_tput={r['output_throughput']:.1f}")
PYEOF
fi
echo
echo "  Crash signatures in server log:"
grep -nE "illegal memory access|CUDA error|RuntimeError|Traceback|core dumped|out of memory|^OOM" \
    "$SERVER_LOG" | head -30 || echo "    (none found — sweep completed cleanly)"
echo "=========================================================="

pkill -9 sglang python 2>/dev/null || true
