#!/bin/bash
# Drive run_tml_lora_vs_nolora_perf_one_bench_single_batch.sh on each branch
# in BRANCHES, keep per-branch results in separate folders, and at the end
# emit a cross-branch markdown comparison report (base vs base, lora vs lora,
# and per-branch LoRA / Base ratios).
#
# This is the single-batch sibling of run_two_branches_compare.sh — same
# wrapper logic, but driving the simpler single-batch bench script (no waves,
# no multi-batch queueing, all metrics physically meaningful).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="${SCRIPT_DIR}/sglang"
BENCH_SCRIPT="${SCRIPT_DIR}/run_tml_lora_vs_nolora_perf_one_bench_single_batch.sh"

BRANCHES=("lora-perf-optimize-2" "04-27-2026")

# Per-branch LoRA backend: newer branches allow `csgmv + virtual-experts`,
# older branches only allow `triton + virtual-experts`.
declare -A LORA_BACKEND_FOR
LORA_BACKEND_FOR["lora-perf-optimize-2"]="csgmv"
LORA_BACKEND_FOR["04-27-2026"]="triton"

# 04-27-2026's flush_cache logic checks is_fully_idle() which stays False
# right after the server's internal warmup, so the first /flush_cache from
# bench_one_batch_server returns 400. Skipping the internal server warmup
# bypasses that race. lora-perf-optimize-2 doesn't need it.
declare -A SKIP_SERVER_WARMUP_FOR
SKIP_SERVER_WARMUP_FOR["lora-perf-optimize-2"]="0"
SKIP_SERVER_WARMUP_FOR["04-27-2026"]="1"

# SKIP_BRANCHES: space/comma-separated list of branch names to skip
# (useful when one branch already has results and you only want to rerun
# the others). Detection is also automatic: if the per-branch lora.jsonl
# already exists we skip unless FORCE_RERUN=1.
SKIP_BRANCHES="${SKIP_BRANCHES:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"

TOPLEVEL_RESULT_DIR="${SCRIPT_DIR}/perf_results_two_branches_single_batch"
mkdir -p "$TOPLEVEL_RESULT_DIR"

echo "================================================================"
echo "  Two-branch LoRA vs No-LoRA Bench (single-batch)"
echo "  Branches: ${BRANCHES[*]}"
echo "  Top-level result dir: ${TOPLEVEL_RESULT_DIR}"
echo "================================================================"

cd "$SGLANG_DIR"
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: ${SGLANG_DIR} has uncommitted changes."
    echo "Please commit/stash before running this wrapper."
    git status --short
    exit 1
fi
ORIGINAL_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "Original branch (will restore on exit): ${ORIGINAL_BRANCH}"

restore_branch() {
    local rc=$?
    cd "$SGLANG_DIR"
    echo ""
    echo "Restoring sglang to branch: ${ORIGINAL_BRANCH}"
    git checkout "$ORIGINAL_BRANCH" 2>/dev/null || true
    exit $rc
}
trap restore_branch EXIT INT TERM

cd "$SCRIPT_DIR"

for BR in "${BRANCHES[@]}"; do
    echo ""
    echo "###############################################################"
    echo "#  Branch: ${BR}"
    echo "###############################################################"

    BR_RESULT_DIR="${TOPLEVEL_RESULT_DIR}/${BR}"

    if [[ ",${SKIP_BRANCHES//[ ]/,}," == *",${BR},"* ]]; then
        echo "  >> SKIP_BRANCHES contains '${BR}', skipping."
        continue
    fi
    if [[ "$FORCE_RERUN" != "1" && -s "${BR_RESULT_DIR}/lora.jsonl" \
                                 && -s "${BR_RESULT_DIR}/nolora.jsonl" ]]; then
        echo "  >> Existing results found at ${BR_RESULT_DIR}/{lora,nolora}.jsonl; skipping."
        echo "     (set FORCE_RERUN=1 to re-run anyway)"
        continue
    fi

    LORA_BE="${LORA_BACKEND_FOR[$BR]:-csgmv}"
    SKIP_SW="${SKIP_SERVER_WARMUP_FOR[$BR]:-0}"

    cd "$SGLANG_DIR"
    git checkout "$BR"
    git log --oneline -1
    cd "$SCRIPT_DIR"

    mkdir -p "$BR_RESULT_DIR"

    echo "  Result dir:          ${BR_RESULT_DIR}"
    echo "  LoRA backend:        ${LORA_BE}"
    echo "  Skip server warmup:  ${SKIP_SW}"
    echo ""

    RESULT_DIR="$BR_RESULT_DIR" \
    LORA_BACKEND="$LORA_BE" \
    SKIP_SERVER_WARMUP="$SKIP_SW" \
        bash "$BENCH_SCRIPT"
done

# ──────────────────────────────────────────────────────────────
#  Cross-branch comparison report
# ──────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Generating cross-branch comparison report..."
echo "================================================================"

python3 - "$TOPLEVEL_RESULT_DIR" "${BRANCHES[@]}" <<'PYEOF'
import json, os, sys

toplevel = sys.argv[1]
branches = sys.argv[2:]

# single_batch.sh layout: {branch_dir}/{lora,nolora}.jsonl, one record per
# (BS, IL, OL) trial. In single-batch mode bench_one_batch_server runs each
# BS exactly once per scenario, so a {BS: record} mapping is enough. If the
# user adds duplicate BSs (e.g. for warmup), we average all records at that
# BS (no skip — we trust the bench's own --skip-warmup).

def load_per_bs(branch_dir, scenario):
    path = os.path.join(branch_dir, f"{scenario}.jsonl")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.setdefault(r["batch_size"], []).append(r)
    return out

def aggregate(records_by_bs):
    """records_by_bs: {BS: [records...]} -> {BS: agg}."""
    def _avg(runs, key, default=None, skip_negative=True):
        vals = []
        for r in runs:
            v = r.get(key)
            if v is None:
                continue
            if skip_negative and v < 0:
                # -1 sentinel = "unknown" (e.g. vllm backend or /server_info miss)
                continue
            vals.append(v)
        if not vals:
            return default
        return sum(vals) / len(vals)

    out = {}
    for bs in sorted(records_by_bs):
        runs = records_by_bs[bs]
        n = len(runs)
        ol = runs[0]["output_len"]
        agg = {
            "BS":           bs,
            "lat_ms":       sum(r["latency"] for r in runs) / n * 1000,
            "ttft_ms":      sum(r["last_ttft"] for r in runs) / n * 1000,
            "in_tput":      sum(r["input_throughput"] for r in runs) / n,
            "out_tput":     sum(r["output_throughput"] for r in runs) / n,
            "ext_tput":     sum(r["overall_throughput"] for r in runs) / n,
            "last_gen":     _avg(runs, "last_gen_throughput"),
            "acc_len":      _avg(runs, "acc_length"),
            "il":           runs[0]["input_len"],
            "ol":           ol,
            "n":            n,
        }
        decode_s = sum(max(r["latency"] - r["last_ttft"], 0) for r in runs) / n
        agg["tpot_ms"] = (decode_s / ol * 1000) if ol > 0 else 0
        out[bs] = agg
    return out

data = {}
for br in branches:
    br_dir = os.path.join(toplevel, br)
    data[br] = {
        "base": load_per_bs(br_dir, "nolora"),
        "lora": load_per_bs(br_dir, "lora"),
    }

agg = {br: {"base": aggregate(data[br]["base"]),
            "lora": aggregate(data[br]["lora"])} for br in branches}

all_bs = sorted({bs
                 for br in branches
                 for sc in ("base", "lora")
                 for bs in agg[br][sc].keys()})

def safe_div(x, y):
    return (x / y) if y else float("nan")

lines = []
lines.append(f"# Cross-Branch Single-Batch Bench: {' vs '.join(branches)}")
lines.append("")
lines.append("> Single-batch mode (N == BS, no multi-batch queueing). All metrics "
             "are physically meaningful: lat (ms), ttft (ms), tpot (ms/tok decode), "
             "ext_tput (= BS×(in+out)/wall, tok/s), out_tput (tok/s).")
lines.append("> Per-branch / per-scenario: one server launch, all BSs swept in a "
             "single bench_one_batch_server call (`--batch-size 1 2 4 ... 64`).")
lines.append("")

# ── Section 1: per-scenario side-by-side ──
# Per-branch columns (raw jsonl fields + derived tpot):
#   latency(ms), last_ttft(ms), tpot(ms), input_tput, output_tput,
#   overall_tput, last_gen_tput, acc_length
PER_BR_COLS = [
    ("lat_ms",   "lat(ms)",   "{:.2f}"),
    ("ttft_ms",  "ttft(ms)",  "{:.2f}"),
    ("tpot_ms",  "tpot(ms)",  "{:.2f}"),
    ("in_tput",  "in_tput",   "{:.2f}"),
    ("out_tput", "out_tput",  "{:.2f}"),
    ("ext_tput", "ext_tput",  "{:.2f}"),
    ("last_gen", "last_gen",  "{:.2f}"),
    ("acc_len",  "acc_len",   "{:.2f}"),
]

def _fmt(val, spec):
    if val is None:
        return "N/A"
    try:
        return spec.format(val)
    except (TypeError, ValueError):
        return str(val)

for sc, sc_label in [("base", "Base Model (no LoRA)"), ("lora", "LoRA")]:
    lines.append(f"## {sc_label}")
    lines.append("")
    lines.append("Columns: lat/ttft/tpot in ms (lower = faster); "
                 "in_tput/out_tput/ext_tput/last_gen in tok/s (higher = faster); "
                 "acc_len = spec-decode accept length (−1 if unused).")
    lines.append("")
    hdr = ["BS"]
    for br in branches:
        for _, label, _ in PER_BR_COLS:
            hdr.append(f"{br}<br>{label}")
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for bs in all_bs:
        row = [str(bs)]
        for br in branches:
            r = agg[br][sc].get(bs)
            if r:
                for key, _, spec in PER_BR_COLS:
                    row.append(_fmt(r.get(key), spec))
            else:
                row.extend(["N/A"] * len(PER_BR_COLS))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

# ── Section 2: head-to-head ratio (only when exactly 2 branches) ──
if len(branches) == 2:
    a, b = branches
    for sc, sc_label in [("base", "Base Model"), ("lora", "LoRA")]:
        lines.append(f"## {sc_label}: `{b}` vs `{a}` (ratio = b/a)")
        lines.append("")
        lines.append(f"- ext_tput / out_tput / in_tput ratios > 1.0 = `{b}` is faster")
        lines.append(f"- lat / tpot / ttft ratios < 1.0 = `{b}` is faster")
        lines.append("")
        lines.append("| BS | lat ratio | ttft ratio | tpot ratio "
                     "| ext_tput ratio | out_tput ratio | in_tput ratio |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for bs in all_bs:
            ra = agg[a][sc].get(bs)
            rb = agg[b][sc].get(bs)
            if ra and rb:
                row = [
                    str(bs),
                    f"{safe_div(rb['lat_ms'],   ra['lat_ms']):.3f}x",
                    f"{safe_div(rb['ttft_ms'],  ra['ttft_ms']):.3f}x",
                    f"{safe_div(rb['tpot_ms'],  ra['tpot_ms']):.3f}x",
                    f"{safe_div(rb['ext_tput'], ra['ext_tput']):.3f}x",
                    f"{safe_div(rb['out_tput'], ra['out_tput']):.3f}x",
                    f"{safe_div(rb['in_tput'],  ra['in_tput']):.3f}x",
                ]
                lines.append("| " + " | ".join(row) + " |")
            else:
                lines.append(f"| {bs} | " + " | ".join(["N/A"] * 6) + " |")
        lines.append("")

# ── Section 3: per-branch LoRA / Base ratio ──
for br in branches:
    lines.append(f"## `{br}` — LoRA / Base ratio")
    lines.append("")
    lines.append("- ext_tput / out_tput ratio < 1.0 = LoRA slower than base")
    lines.append("- lat / tpot / ttft ratio > 1.0 = LoRA slower than base")
    lines.append("")
    lines.append("| BS | lat ratio | ttft ratio | tpot ratio "
                 "| ext_tput ratio | out_tput ratio |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for bs in all_bs:
        ra = agg[br]["base"].get(bs)
        rb = agg[br]["lora"].get(bs)
        if ra and rb:
            row = [
                str(bs),
                f"{safe_div(rb['lat_ms'],   ra['lat_ms']):.3f}x",
                f"{safe_div(rb['ttft_ms'],  ra['ttft_ms']):.3f}x",
                f"{safe_div(rb['tpot_ms'],  ra['tpot_ms']):.3f}x",
                f"{safe_div(rb['ext_tput'], ra['ext_tput']):.3f}x",
                f"{safe_div(rb['out_tput'], ra['out_tput']):.3f}x",
            ]
            lines.append("| " + " | ".join(row) + " |")
        else:
            lines.append(f"| {bs} | " + " | ".join(["N/A"] * 5) + " |")
    lines.append("")

report = "\n".join(lines)
out_path = os.path.join(toplevel, "cross_branch_report.md")
with open(out_path, "w") as f:
    f.write(report)
print(report)
print(f"\n=> Cross-branch report saved to: {out_path}")
PYEOF

echo ""
echo "================================================================"
echo "  All done."
echo "================================================================"
echo "Per-branch raw jsonl:"
for BR in "${BRANCHES[@]}"; do
    echo "  ${TOPLEVEL_RESULT_DIR}/${BR}/{lora,nolora}.jsonl"
done
echo "Cross-branch report:"
echo "  ${TOPLEVEL_RESULT_DIR}/cross_branch_report.md"
