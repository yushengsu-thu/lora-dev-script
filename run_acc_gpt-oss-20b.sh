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
MODEL_PATH="lmsys/gpt-oss-20b-bf16"
ADAPTER_PATH="${SCRIPT_DIR}/lora_test_cases/gpt-oss-20b"

FLASHINFER_DISABLE_VERSION_CHECK=1 \
PYTHONPATH="${SCRIPT_DIR}/sglang/python:$PYTHONPATH" \
python "${SCRIPT_DIR}/check_sglang_lora_correctness.py" \
    --adapter-path "$ADAPTER_PATH" \
    --model-path "$MODEL_PATH" \
    --tp 8 \
    --experts-shared-outer-loras \
    --moe-runner-backend triton \
    --prefill-attention-backend fa4 \
    --decode-attention-backend fa4