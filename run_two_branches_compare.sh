#!/bin/bash
# Run run_tml_lora_vs_nolora_perf_one_bench.sh on each branch in BRANCHES,
# keep per-branch results in separate folders, and at the end emit a
# cross-branch markdown comparison report (base vs base, lora vs lora).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="${SCRIPT_DIR}/sglang"
BENCH_SCRIPT="${SCRIPT_DIR}/run_tml_lora_vs_nolora_perf_one_bench.sh"

BRANCHES=("lora-perf-optimize-2" "04-27-2026")

# Per-branch LoRA backend. Newer branches allow `csgmv + virtual-experts`,
# older branches only allow `triton + virtual-experts`.
declare -A LORA_BACKEND_FOR
LORA_BACKEND_FOR["lora-perf-optimize-2"]="csgmv"
LORA_BACKEND_FOR["04-27-2026"]="triton"

# Per-branch --skip-server-warmup: 04-27-2026's flush_cache logic checks
# is_fully_idle() which stays False right after the server's internal
# warmup, so the first /flush_cache from bench_one_batch_server returns
# 400. Skipping the internal server warmup bypasses that race.
declare -A SKIP_SERVER_WARMUP_FOR
SKIP_SERVER_WARMUP_FOR["lora-perf-optimize-2"]="0"
SKIP_SERVER_WARMUP_FOR["04-27-2026"]="1"

# SKIP_BRANCHES: space/comma-separated list of branch names to skip
# (useful when one branch already has results and you only want to
# rerun the others). Detection is also automatic: if the per-branch
# bench_report.md already exists we skip unless FORCE_RERUN=1.
SKIP_BRANCHES="${SKIP_BRANCHES:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"

TOPLEVEL_RESULT_DIR="${SCRIPT_DIR}/perf_results_two_branches"
mkdir -p "$TOPLEVEL_RESULT_DIR"

NUM_SKIP=1

echo "================================================================"
echo "  Two-branch LoRA vs No-LoRA Bench"
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
    BR_REPORT="${BR_RESULT_DIR}/bench_report.md"

    # Skip logic
    if [[ ",${SKIP_BRANCHES//[ ]/,}," == *",${BR},"* ]]; then
        echo "  >> SKIP_BRANCHES contains '${BR}', skipping."
        continue
    fi
    if [[ "$FORCE_RERUN" != "1" && -s "$BR_REPORT" ]]; then
        echo "  >> Existing results found at ${BR_REPORT}; skipping."
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

    RESULT_DIR="$BR_RESULT_DIR" REPORT_TAG="$BR" \
    LORA_BACKEND="$LORA_BE" SKIP_SERVER_WARMUP="$SKIP_SW" \
        bash "$BENCH_SCRIPT"
done

# ──────────────────────────────────────────────────────────────
#  Cross-branch comparison report
# ──────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Generating cross-branch comparison report..."
echo "================================================================"

python3 - "$TOPLEVEL_RESULT_DIR" "$NUM_SKIP" "${BRANCHES[@]}" <<'PYEOF'
import json, os, re, sys, glob
from collections import defaultdict

toplevel = sys.argv[1]
num_skip = int(sys.argv[2])
branches = sys.argv[3:]

# Per-BS jsonl naming from run_tml_lora_vs_nolora_perf_one_bench.sh:
#   {scenario}_bs{BS}.jsonl  where scenario = "lora" | "nolora".
# Each line is one trial; record's `batch_size` field stores N (total prompts
# under --enable-multi-batch), so BS comes from the filename and N from the
# record. We aggregate per (branch, scenario, BS) over kept trials.
BS_FILE_RE = re.compile(r"_bs(\d+)\.jsonl$")

def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def load_per_bs(branch_dir, scenario):
    """Return {BS: [records...]} loaded from {branch_dir}/{scenario}_bs*.jsonl."""
    out = {}
    for path in sorted(glob.glob(os.path.join(branch_dir, f"{scenario}_bs*.jsonl"))):
        m = BS_FILE_RE.search(path)
        if not m:
            continue
        bs = int(m.group(1))
        recs = load(path)
        if recs:
            out[bs] = recs
    return out

data = {}
for br in branches:
    br_dir = os.path.join(toplevel, br)
    data[br] = {
        "base": load_per_bs(br_dir, "nolora"),
        "lora": load_per_bs(br_dir, "lora"),
    }

def aggregate(records_by_bs, num_skip):
    """records_by_bs: {BS: [records of trials at that BS]} -> {BS: agg}."""
    out = {}
    for bs in sorted(records_by_bs):
        runs = records_by_bs[bs]
        kept = runs[num_skip:] or runs
        n = len(kept)
        ol = kept[0]["output_len"]
        agg = {
            "BS":       bs,
            "N":        kept[0]["batch_size"],
            "lat_ms":   sum(r["latency"] for r in kept) / n * 1000,
            "ttft_ms":  sum(r["last_ttft"] for r in kept) / n * 1000,
            "in_tput":  sum(r["input_throughput"] for r in kept) / n,
            "out_tput": sum(r["output_throughput"] for r in kept) / n,
            "ext_tput": sum(r["overall_throughput"] for r in kept) / n,
            "il":       kept[0]["input_len"],
            "ol":       ol,
            "n_kept":   n,
        }
        # ITL = (lat - ttft) / ol * 1000  (per-token decode time);
        # under multi-batch (N > BS) this is misleading — kept for sanity only.
        decode_s = sum(max(r["latency"] - r["last_ttft"], 0) for r in kept) / n
        agg["itl_ms"] = (decode_s / ol * 1000) if ol > 0 else 0
        out[bs] = agg
    return out

agg = {br: {"base": aggregate(data[br]["base"], num_skip),
            "lora": aggregate(data[br]["lora"], num_skip)} for br in branches}

all_bs = sorted({bs
                 for br in branches
                 for sc in ("base", "lora")
                 for bs in agg[br][sc].keys()})

# Build {BS: N} mapping (should be consistent across branches/scenarios for a
# given BS since both scripts compute N = BS * WAVES). If branches disagree
# we fall back to "?".
n_for_bs = {}
for bs in all_bs:
    seen = {agg[br][sc][bs]["N"]
            for br in branches
            for sc in ("base", "lora")
            if bs in agg[br][sc]}
    n_for_bs[bs] = next(iter(seen)) if len(seen) == 1 else "?"

def safe_div(x, y):
    return (x / y) if y else float("nan")

lines = []
lines.append(f"# Cross-Branch Bench: {' vs '.join(branches)}")
lines.append("")
lines.append(f"> Skipped first {num_skip} trial(s) per (BS, branch, scenario), averaged the rest.")
lines.append("> Per-BS server launch with `--max-running-requests=BS`; "
             "N = total prompts sent per trial under `--enable-multi-batch` "
             "(N > BS → surplus is scheduler-queued, run as multiple waves).")
lines.append("> Columns: lat (ms), ttft (ms), itl (ms/tok), ext_tput "
             "(= N×(in+out)/wall, tok/s), out_tput (tok/s).")
lines.append("> **Note**: under multi-batch (waves > 1), only `lat` and `ext_tput` "
             "are physically meaningful; `ttft`/`itl`/`out_tput` assume one-shot "
             "batching and are misleading — keep for sanity, not for ranking.")
lines.append("")

# ── Section 1: per-scenario side-by-side ──
for sc, sc_label in [("base", "Base Model (no LoRA)"), ("lora", "LoRA")]:
    lines.append(f"## {sc_label}")
    lines.append("")
    hdr = ["BS", "N"]
    for br in branches:
        hdr.extend([f"{br}<br>lat", f"{br}<br>ttft", f"{br}<br>itl",
                    f"{br}<br>ext_tput", f"{br}<br>out_tput"])
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for bs in all_bs:
        row = [str(bs), str(n_for_bs[bs])]
        for br in branches:
            r = agg[br][sc].get(bs)
            if r:
                row.extend([
                    f"{r['lat_ms']:.2f}",
                    f"{r['ttft_ms']:.2f}",
                    f"{r['itl_ms']:.2f}",
                    f"{r['ext_tput']:.2f}",
                    f"{r['out_tput']:.2f}",
                ])
            else:
                row.extend(["N/A"] * 5)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

# ── Section 2: head-to-head ratio (only when exactly 2 branches) ──
if len(branches) == 2:
    a, b = branches
    for sc, sc_label in [("base", "Base Model"), ("lora", "LoRA")]:
        lines.append(f"## {sc_label}: `{b}` vs `{a}` (ratio = b/a)")
        lines.append("")
        lines.append("- ext_tput / out_tput / in_tput ratios > 1.0 = `b` is faster")
        lines.append("- lat / itl / ttft ratios < 1.0 = `b` is faster")
        lines.append("")
        lines.append("| BS | N | lat ratio | ttft ratio | itl ratio "
                     "| ext_tput ratio | out_tput ratio | in_tput ratio |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for bs in all_bs:
            ra = agg[a][sc].get(bs)
            rb = agg[b][sc].get(bs)
            if ra and rb:
                row = [
                    str(bs),
                    str(n_for_bs[bs]),
                    f"{safe_div(rb['lat_ms'],   ra['lat_ms']):.3f}x",
                    f"{safe_div(rb['ttft_ms'],  ra['ttft_ms']):.3f}x",
                    f"{safe_div(rb['itl_ms'],   ra['itl_ms']):.3f}x",
                    f"{safe_div(rb['ext_tput'], ra['ext_tput']):.3f}x",
                    f"{safe_div(rb['out_tput'], ra['out_tput']):.3f}x",
                    f"{safe_div(rb['in_tput'],  ra['in_tput']):.3f}x",
                ]
                lines.append("| " + " | ".join(row) + " |")
            else:
                lines.append(f"| {bs} | {n_for_bs[bs]} | "
                             + " | ".join(["N/A"] * 6) + " |")
        lines.append("")

# ── Section 3: per-branch LoRA / base ratio ──
for br in branches:
    lines.append(f"## `{br}` — LoRA / Base ratio")
    lines.append("")
    lines.append("- ext_tput / out_tput ratio < 1.0 = LoRA slower than base")
    lines.append("- lat / itl ratio > 1.0 = LoRA slower than base")
    lines.append("")
    lines.append("| BS | N | lat ratio | ttft ratio | itl ratio "
                 "| ext_tput ratio | out_tput ratio |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for bs in all_bs:
        ra = agg[br]["base"].get(bs)
        rb = agg[br]["lora"].get(bs)
        if ra and rb:
            row = [
                str(bs),
                str(n_for_bs[bs]),
                f"{safe_div(rb['lat_ms'],   ra['lat_ms']):.3f}x",
                f"{safe_div(rb['ttft_ms'],  ra['ttft_ms']):.3f}x",
                f"{safe_div(rb['itl_ms'],   ra['itl_ms']):.3f}x",
                f"{safe_div(rb['ext_tput'], ra['ext_tput']):.3f}x",
                f"{safe_div(rb['out_tput'], ra['out_tput']):.3f}x",
            ]
            lines.append("| " + " | ".join(row) + " |")
        else:
            lines.append(f"| {bs} | {n_for_bs[bs]} | "
                         + " | ".join(["N/A"] * 5) + " |")
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
echo "Per-branch reports:"
for BR in "${BRANCHES[@]}"; do
    echo "  ${TOPLEVEL_RESULT_DIR}/${BR}/bench_report.md"
done
echo "Cross-branch report:"
echo "  ${TOPLEVEL_RESULT_DIR}/cross_branch_report.md"
