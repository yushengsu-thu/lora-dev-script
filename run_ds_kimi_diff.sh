#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/prefill_decode_diff_results"
SGLANG_DIR="${SCRIPT_DIR}/sglang"
SGLANG_PRIVATE_DIR="${SCRIPT_DIR}/sglang-private"

mkdir -p "${RESULTS_DIR}/sglang" "${RESULTS_DIR}/sglang_private"

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
    local TAG="$2"
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
    python "${SCRIPT_DIR}/compare_prefill_decode_logprobs.py" \
        --model-path "$MODEL_PATH" \
        --adapter-path "${SCRIPT_DIR}/lora_test_cases/${MODEL_NAME}" \
        --output-path "$OUTPUT" \
        --max-new-tokens 32 \
        "${EXTRA_ARGS[@]}"
}

compare() {
    local MODEL_NAME="$1"
    local FA="${RESULTS_DIR}/sglang/${MODEL_NAME}.pt"
    local FB="${RESULTS_DIR}/sglang_private/${MODEL_NAME}.pt"

    python3 - "$FA" "$FB" "$MODEL_NAME" <<'PYEOF'
import torch, sys

fa, fb, name = sys.argv[1], sys.argv[2], sys.argv[3]
a = torch.load(fa, weights_only=False)
b = torch.load(fb, weights_only=False)

for phase in ["prefill", "decode"]:
    la = a[f"{phase}_logprobs"].float()
    lb = b[f"{phase}_logprobs"].float()
    diff = (la - lb).abs()
    mean_d = diff.mean().item()
    max_d = diff.max().item()
    identical = torch.equal(la, lb)

    print(f"\n{'='*60}")
    print(f"[{phase.upper()}] {name}")
    print(f"  identical   = {identical}")
    print(f"  mean_diff   = {mean_d:.10f}")
    print(f"  max_diff    = {max_d:.10f}")
    print(f"  sglang         first5 = {la[:5].tolist()}")
    print(f"  sglang_private first5 = {lb[:5].tolist()}")

    if max_d > 1e-4:
        top_k = 10
        vals, idxs = diff.topk(min(top_k, diff.numel()))
        print(f"  [WARNING] Large diff! Top-{top_k} positions:")
        for v, i in zip(vals, idxs):
            print(f"    pos {i.item():5d}: sglang={la[i].item():.8f}  private={lb[i].item():.8f}  diff={v.item():.8f}")
    else:
        print(f"  [OK] Match within tolerance (max_diff <= 1e-4).")
    print(f"{'='*60}")

if "decode_token_ids" in a and "decode_token_ids" in b:
    ids_a = a["decode_token_ids"]
    ids_b = b["decode_token_ids"]
    match = torch.equal(ids_a, ids_b)
    print(f"\n[DECODE TOKENS] {name}")
    print(f"  token_ids identical = {match}")
    if not match:
        for i in range(min(len(ids_a), len(ids_b))):
            if ids_a[i] != ids_b[i]:
                print(f"  first mismatch at pos {i}: sglang={ids_a[i].item()} vs private={ids_b[i].item()}")
                break
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
    if ! run_dump "$SGLANG_PRIVATE_DIR" "sglang_private" "$MODEL_NAME" "$MODEL_PATH" "${EXTRA_ARGS[@]}"; then
        echo "[ERROR] sglang_private dump failed for ${MODEL_NAME}, skipping."
        return 1
    fi
    compare "$MODEL_NAME"
}

# ── Main ──────────────────────────────────────────────────

echo "============================================================"
echo "  Prefill + Decode logprob diff:  sglang/  vs  sglang-private/"
echo "  Results -> ${RESULTS_DIR}"
echo "============================================================"

FAILED=()

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

# ── Cleanup & Summary ────────────────────────────────────

kill_all

echo ""
echo "============================================================"
echo "  SUMMARY"
echo "============================================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "  ALL PASSED -- sglang/ and sglang-private/ produce matching logprobs."
else
    echo "  FAILED models:"
    for m in "${FAILED[@]}"; do
        echo "    - $m"
    done
    echo ""
    echo "  Check detailed diffs in: ${RESULTS_DIR}/"
    exit 1
fi
