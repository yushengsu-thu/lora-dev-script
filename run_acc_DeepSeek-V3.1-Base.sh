pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="deepseek-ai/DeepSeek-V3.1-Base"
ADAPTER_PATH="${SCRIPT_DIR}/lora_test_cases/DeepSeek-V3.1-Base"

FLASHINFER_DISABLE_VERSION_CHECK=1 \
PYTHONPATH="${SCRIPT_DIR}/sglang/python:$PYTHONPATH" \
python "${SCRIPT_DIR}/check_sglang_lora_correctness.py" \
    --adapter-path "$ADAPTER_PATH" \
    --model-path "$MODEL_PATH" \
    --tp 4 \
    --moe-runner-backend triton \
    --experts-shared-outer-loras \
    --batch-input-ids \
    --disable-shared-experts-fusion \
    --prefill-attention-backend fa4 \
    --decode-attention-backend flashinfer