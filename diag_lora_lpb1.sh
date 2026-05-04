#!/bin/bash
set -uo pipefail
export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}

SCRIPT_DIR="/home/radixark/yushengsu"
MODEL_PATH="Qwen/Qwen3-30B-A3B-Instruct-2507"
ADAPTER_PATH="${SCRIPT_DIR}/lora_test_cases/Qwen3-30B-A3B-Instruct-2507"
PORT=30000
TP=4
LOG="${SCRIPT_DIR}/diag_lora_lpb1_$(date +%Y%m%d_%H%M%S).log"
echo "log -> $LOG"

export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTHONPATH="${SCRIPT_DIR}/sglang/python:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

(
  python -m sglang.launch_server \
    --model "$MODEL_PATH" --tp "$TP" --port "$PORT" \
    --mem-fraction-static 0.82 \
    --enable-lora --lora-paths my_lora="$ADAPTER_PATH" \
    --max-lora-rank 32 --lora-backend csgmv \
    --max-loras-per-batch 1 \
    --moe-runner-backend triton \
    --experts-shared-outer-loras --lora-use-virtual-experts \
    --prefill-attention-backend fa4 --decode-attention-backend fa4 2>&1
) >> "$LOG" 2>&1 &
SERVER_PID=$!
echo "server pid $SERVER_PID, log $LOG"

for i in $(seq 1 240); do
  if curl -s -m 3 "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "ready after ${i}*5s"; break
  fi
  sleep 5
done

if ! curl -s -m 3 "http://localhost:${PORT}/health" >/dev/null 2>&1; then
  echo "NOT ready"; tail -120 "$LOG"; exit 1
fi

echo "" | tee -a "$LOG"
echo "=== bench BS=160, max_loras_per_batch=1, CG on ===" | tee -a "$LOG"
python3 -m sglang.bench_one_batch_server \
    --model None --tokenizer-path "$MODEL_PATH" \
    --base-url "http://localhost:${PORT}" \
    --batch-size 160 --input-len 8192 --output-len 1024 \
    --result-filename "${SCRIPT_DIR}/diag_lora_lpb1_result.jsonl" \
    --show-report --enable-multi-batch --lora-name my_lora 2>&1 | tee -a "$LOG"

echo "bench exit=$?" | tee -a "$LOG"
echo "log: $LOG"
