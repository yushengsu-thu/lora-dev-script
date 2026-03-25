#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/base_model_diff_results"
SGLANG_DIR="${SCRIPT_DIR}/sglang"
SGLANG_TML_DIR="${SCRIPT_DIR}/sglang_tml"

mkdir -p "${RESULTS_DIR}/sglang" "${RESULTS_DIR}/sglang_tml"

# ── Helpers ───────────────────────────────────────────────

kill_all() {
    pkill -9 sglang 2>/dev/null || true
    sleep 2
    ray stop --force 2>/dev/null || true
    pkill -9 ray 2>/dev/null || true
    pkill -9 python 2>/dev/null || true
    sleep 3
    pkill -9 ray 2>/dev/null || true
    pkill -9 python 2>/dev/null || true
    sleep 2
}

run_dump() {
    local SGLANG_SRC="$1"
    local TAG="$2"          # "sglang" or "sglang_tml"
    local MODEL_NAME="$3"
    local MODEL_PATH="$4"
    shift 4
    local EXTRA_ARGS=("$@")
    local OUTPUT="${RESULTS_DIR}/${TAG}/${MODEL_NAME}.pt"

    echo ""
    echo ">>> [${TAG}] ${MODEL_NAME}  (${MODEL_PATH})"

    kill_all

    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    SGLANG_DISABLE_CUDNN_CHECK=1 \
    PYTHONPATH="${SGLANG_SRC}/python:${PYTHONPATH:-}" \
    python "${SCRIPT_DIR}/dump_base_logprobs.py" \
        --model-path "$MODEL_PATH" \
        --adapter-path "${SCRIPT_DIR}/lora_test_cases/${MODEL_NAME}" \
        --output-path "$OUTPUT" \
        "${EXTRA_ARGS[@]}"
}

compare() {
    local MODEL_NAME="$1"
    local FA="${RESULTS_DIR}/sglang/${MODEL_NAME}.pt"
    local FB="${RESULTS_DIR}/sglang_tml/${MODEL_NAME}.pt"

    python3 - "$FA" "$FB" "$MODEL_NAME" <<'PYEOF'
import torch, sys

fa, fb, name = sys.argv[1], sys.argv[2], sys.argv[3]
a = torch.load(fa, weights_only=False)
b = torch.load(fb, weights_only=False)
la, lb = a["logprobs"].float(), b["logprobs"].float()

diff = (la - lb).abs()
mean_d = diff.mean().item()
max_d  = diff.max().item()
identical = torch.equal(la, lb)

print(f"\n{'='*60}")
print(f"[COMPARE] {name}")
print(f"  identical   = {identical}")
print(f"  mean_diff   = {mean_d:.10f}")
print(f"  max_diff    = {max_d:.10f}")
print(f"  sglang      first5 = {la[:5].tolist()}")
print(f"  sglang_tml  first5 = {lb[:5].tolist()}")

if max_d > 1e-4:
    top_k = 10
    vals, idxs = diff.topk(min(top_k, diff.numel()))
    print(f"  [WARNING] Large diff! Top-{top_k} positions:")
    for v, i in zip(vals, idxs):
        print(f"    pos {i.item():5d}: sglang={la[i].item():.8f}  sglang_tml={lb[i].item():.8f}  diff={v.item():.8f}")
    sys.exit(1)
else:
    print(f"  [OK] Results match within tolerance (max_diff <= 1e-4).")
print(f"{'='*60}")
PYEOF
}

# ── Per-model runner ──────────────────────────────────────

run_both() {
    local MODEL_NAME="$1"
    local MODEL_PATH="$2"
    shift 2
    local EXTRA_ARGS=("$@")

    echo ""
    echo "############################################################"
    echo "#  ${MODEL_NAME}"
    echo "############################################################"

    if ! run_dump "$SGLANG_DIR" "sglang" "$MODEL_NAME" "$MODEL_PATH" "${EXTRA_ARGS[@]}"; then
        echo "[ERROR] sglang dump failed for ${MODEL_NAME}, skipping."
        return 1
    fi
    if ! run_dump "$SGLANG_TML_DIR" "sglang_tml" "$MODEL_NAME" "$MODEL_PATH" "${EXTRA_ARGS[@]}"; then
        echo "[ERROR] sglang_tml dump failed for ${MODEL_NAME}, skipping."
        return 1
    fi
    compare "$MODEL_NAME"
}

# ── Main ──────────────────────────────────────────────────

echo "============================================================"
echo "  Base-model logprob diff:  sglang/  vs  sglang_tml/"
echo "  Results -> ${RESULTS_DIR}"
echo "============================================================"

FAILED=()

# --- Qwen3-8B (dense, tp=1) ---
run_both "Qwen3-8B" "Qwen/Qwen3-8B" \
    --tp 1 \
    --prefill-attention-backend fa4 \
    --decode-attention-backend fa4 \
    || FAILED+=("Qwen3-8B")

# --- gpt-oss-20b (MoE, tp=4) ---
run_both "gpt-oss-20b" "lmsys/gpt-oss-20b-bf16" \
    --tp 4 \
    --prefill-attention-backend fa4 \
    --decode-attention-backend fa4 \
    || FAILED+=("gpt-oss-20b")

# --- Qwen3-30B-A3B-Instruct-2507 (MoE, tp=4) ---
run_both "Qwen3-30B-A3B-Instruct-2507" "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --tp 4 \
    --prefill-attention-backend fa4 \
    --decode-attention-backend fa4 \
    || FAILED+=("Qwen3-30B-A3B-Instruct-2507")

# --- DeepSeek-V3.1-Base (MoE, tp=4) ---
run_both "DeepSeek-V3.1-Base" "deepseek-ai/DeepSeek-V3.1-Base" \
    --tp 4 \
    --batch-input-ids \
    --disable-shared-experts-fusion \
    --prefill-attention-backend fa4 \
    --decode-attention-backend flashinfer \
    || FAILED+=("DeepSeek-V3.1-Base")

# --- Kimi-K2.5 (MoE, tp=4) ---
run_both "Kimi-K2.5" "moonshotai/Kimi-K2.5" \
    --tp 4 \
    --trust-remote-code \
    --prefill-attention-backend fa4 \
    --decode-attention-backend flashinfer \
    || FAILED+=("Kimi-K2.5")

# --- Qwen3-VL-30B-A3B-Instruct (MoE, tp=4) ---
run_both "Qwen3-VL-30B-A3B-Instruct" "Qwen/Qwen3-VL-30B-A3B-Instruct" \
    --tp 4 \
    --prefill-attention-backend fa4 \
    --decode-attention-backend fa4 \
    || FAILED+=("Qwen3-VL-30B-A3B-Instruct")

# ── Uncomment below when HF paths are confirmed ──────────
# run_both "Qwen3.5-35B-A3B" "Qwen/Qwen3.5-35B-A3B" \
#     --tp 4 --prefill-attention-backend fa4 --decode-attention-backend fa4 \
#     || FAILED+=("Qwen3.5-35B-A3B")
#
# run_both "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" \
#     --tp 4 --prefill-attention-backend fa4 --decode-attention-backend fa4 \
#     || FAILED+=("NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
#
# run_both "NVIDIA-Nemotron-3-Super-120B-A12B-BF16" "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16" \
#     --tp 8 --prefill-attention-backend fa4 --decode-attention-backend fa4 \
#     || FAILED+=("NVIDIA-Nemotron-3-Super-120B-A12B-BF16")
#
# run_both "DeepSeek-V3.1-tiny-debug-2025-08-26" "deepseek-ai/DeepSeek-V3.1-tiny-debug-2025-08-26" \
#     --tp 4 --batch-input-ids --disable-shared-experts-fusion \
#     --prefill-attention-backend fa4 --decode-attention-backend flashinfer \
#     || FAILED+=("DeepSeek-V3.1-tiny-debug-2025-08-26")

# ── Cleanup & Summary ────────────────────────────────────

kill_all

echo ""
echo "============================================================"
echo "  SUMMARY"
echo "============================================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "  ALL MODELS PASSED -- sglang/ and sglang_tml/ produce identical base-model logprobs."
else
    echo "  FAILED models (max_diff > 1e-4):"
    for m in "${FAILED[@]}"; do
        echo "    - $m"
    done
    echo ""
    echo "  Check detailed diffs in: ${RESULTS_DIR}/"
    exit 1
fi
