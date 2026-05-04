import json, os, sys

METRIC_KEYS = [
    ("completed",  "completed"),
    ("duration",   "duration"),
    ("req_tput",   "request_throughput"),
    ("in_tput",    "input_throughput"),
    ("out_tput",   "output_throughput"),
    ("total_tput", "total_throughput"),
    ("mean_e2e",   "mean_e2e_latency_ms"),
    ("median_e2e", "median_e2e_latency_ms"),
    ("p99_e2e",    "p99_e2e_latency_ms"),
    ("mean_ttft",  "mean_ttft_ms"),
    ("median_ttft","median_ttft_ms"),
    ("p99_ttft",   "p99_ttft_ms"),
    ("mean_tpot",  "mean_tpot_ms"),
    ("median_tpot","median_tpot_ms"),
    ("p99_tpot",   "p99_tpot_ms"),
    ("mean_itl",   "mean_itl_ms"),
    ("median_itl", "median_itl_ms"),
    ("p99_itl",    "p99_itl_ms"),
]

def read_avg(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    if not records:
        return None
    records = records[-3:]  # take last 3 runs
    avg = {}
    for short, raw in METRIC_KEYS:
        vals = [r[raw] for r in records if isinstance(r.get(raw), (int, float))]
        avg[short] = sum(vals) / len(vals) if vals else 0
    avg["num_runs"] = len(records)
    return avg

scenarios_lat = [
    ("lat_prefill_short", 1, 1, 256, 2),
    ("lat_prefill_long", 1, 1, 4096, 2),
    ("lat_prefill_8k", 1, 1, 8192, 2),
    ("lat_prefill_16k", 1, 1, 16384, 2),
    ("lat_decode", 1, 1, 256, 256),
]

scenarios_tput = [
    ("tput_bs64_np64", 64, 64, 256, 256),
    ("tput_bs64_np128", 128, 64, 256, 256),
    ("tput_bs64_np256", 256, 64, 256, 256),
]

TABLE_HDR = "| number_of_prompts | BS | in_len | out_len | runs | completed | dur(s) | req/s | in_tok/s | out_tok/s | total_tok/s | mean_e2e(ms) | med_e2e | p99_e2e | mean_ttft(ms) | med_ttft | p99_ttft | mean_tpot(ms) | med_tpot | p99_tpot | mean_itl(ms) | med_itl | p99_itl |"
TABLE_SEP = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"

def table_row(np_val, bs, in_len, out_len, m):
    if not m:
        return f"| {np_val} | {bs} | {in_len} | {out_len} |" + " N/A |" * 19
    return (
        f"| {np_val} | {bs} | {in_len} | {out_len} "
        f"| {m['num_runs']} | {m['completed']:.0f} | {m['duration']:.2f} | {m['req_tput']:.2f} "
        f"| {m['in_tput']:.2f} | {m['out_tput']:.2f} | {m['total_tput']:.2f} "
        f"| {m['mean_e2e']:.2f} | {m['median_e2e']:.2f} | {m['p99_e2e']:.2f} "
        f"| {m['mean_ttft']:.2f} | {m['median_ttft']:.2f} | {m['p99_ttft']:.2f} "
        f"| {m['mean_tpot']:.2f} | {m['median_tpot']:.2f} | {m['p99_tpot']:.2f} "
        f"| {m['mean_itl']:.2f} | {m['median_itl']:.2f} | {m['p99_itl']:.2f} |"
    )

# Data sources
# Branch: lora-perf-optimize-2
opt2_dir = "/home/radixark/yushengsu/perf_results_tml_lora_vs_nolora_branch_lora-perf-optimize-2"

# Branch: 04-27-2026
# LoRA latency from the earlier run
branch2_lora_lat_dir = "/home/radixark/yushengsu/perf_results_tml_lora_vs_nolora"
# No-LoRA latency from the dedicated run
branch2_nolora_lat_dir = "/home/radixark/yushengsu/perf_results_04-27-2026_nolora_lat"
# Throughput (both LoRA and no-LoRA) from the dedicated run
branch2_tput_dir = "/home/radixark/yushengsu/perf_results_tml_lora_vs_nolora_04-27-2026_tput"

lines = []

# ========== A. 04-27-2026 ==========
lines.append("## A. 04-27-2026 (triton backend)")
lines.append("")
lines.append("### Low Latency (BS=1)")
lines.append("")

# 1. Base model
lines.append("#### 1. Base Model (no LoRA)")
lines.append("")
lines.append(TABLE_HDR)
lines.append(TABLE_SEP)
for name, np_val, bs, in_len, out_len in scenarios_lat:
    m = read_avg(os.path.join(branch2_nolora_lat_dir, f"nolora_{name}.jsonl"))
    lines.append(table_row(np_val, bs, in_len, out_len, m))
lines.append("")

# 2. LoRA
lines.append("#### 2. LoRA (triton + virtual experts)")
lines.append("")
lines.append(TABLE_HDR)
lines.append(TABLE_SEP)
for name, np_val, bs, in_len, out_len in scenarios_lat:
    m = read_avg(os.path.join(branch2_lora_lat_dir, f"lora_{name}.jsonl"))
    lines.append(table_row(np_val, bs, in_len, out_len, m))
lines.append("")

# High Throughput
lines.append("### High Throughput (BS=64)")
lines.append("")

# 1. Base model
lines.append("#### 1. Base Model (no LoRA)")
lines.append("")
lines.append(TABLE_HDR)
lines.append(TABLE_SEP)
for name, np_val, bs, in_len, out_len in scenarios_tput:
    m = read_avg(os.path.join(branch2_tput_dir, f"nolora_{name}.jsonl"))
    lines.append(table_row(np_val, bs, in_len, out_len, m))
lines.append("")

# 2. LoRA
lines.append("#### 2. LoRA (triton + virtual experts)")
lines.append("")
lines.append(TABLE_HDR)
lines.append(TABLE_SEP)
for name, np_val, bs, in_len, out_len in scenarios_tput:
    m = read_avg(os.path.join(branch2_tput_dir, f"lora_{name}.jsonl"))
    lines.append(table_row(np_val, bs, in_len, out_len, m))
lines.append("")

# ========== B. lora-perf-optimize-2 ==========
lines.append("## B. lora-perf-optimize-2 (csgmv backend)")
lines.append("")
lines.append("### Low Latency (BS=1)")
lines.append("")

# 1. Base model
lines.append("#### 1. Base Model (no LoRA)")
lines.append("")
lines.append(TABLE_HDR)
lines.append(TABLE_SEP)
for name, np_val, bs, in_len, out_len in scenarios_lat:
    m = read_avg(os.path.join(opt2_dir, f"nolora_{name}.jsonl"))
    lines.append(table_row(np_val, bs, in_len, out_len, m))
lines.append("")

# 2. LoRA
lines.append("#### 2. LoRA (csgmv + virtual experts)")
lines.append("")
lines.append(TABLE_HDR)
lines.append(TABLE_SEP)
for name, np_val, bs, in_len, out_len in scenarios_lat:
    m = read_avg(os.path.join(opt2_dir, f"lora_{name}.jsonl"))
    lines.append(table_row(np_val, bs, in_len, out_len, m))
lines.append("")

# High Throughput
lines.append("### High Throughput (BS=64)")
lines.append("")

# 1. Base model
lines.append("#### 1. Base Model (no LoRA)")
lines.append("")
lines.append(TABLE_HDR)
lines.append(TABLE_SEP)
for name, np_val, bs, in_len, out_len in scenarios_tput:
    m = read_avg(os.path.join(opt2_dir, f"nolora_{name}.jsonl"))
    lines.append(table_row(np_val, bs, in_len, out_len, m))
lines.append("")

# 2. LoRA
lines.append("#### 2. LoRA (csgmv + virtual experts)")
lines.append("")
lines.append(TABLE_HDR)
lines.append(TABLE_SEP)
for name, np_val, bs, in_len, out_len in scenarios_tput:
    m = read_avg(os.path.join(opt2_dir, f"lora_{name}.jsonl"))
    lines.append(table_row(np_val, bs, in_len, out_len, m))
lines.append("")

# ========== Comparison ==========
lines.append("## 04-27-2026 vs lora-perf-optimize-2")
lines.append("")

def ratio(a, b):
    return f"{a / b:.2f}x" if b else "—"

CMP_HDR = "| number_of_prompts | BS | in | out | lora_mean_e2e(ms) | base_mean_e2e(ms) | e2e_ratio | lora_mean_ttft(ms) | base_mean_ttft(ms) | ttft_ratio | lora_mean_tpot(ms) | base_mean_tpot(ms) | tpot_ratio | lora_mean_itl(ms) | base_mean_itl(ms) | itl_ratio | lora_total_tok/s | base_total_tok/s | tput_ratio |"
CMP_SEP = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"

# Compare LoRA latency across branches
lines.append("### LoRA Low Latency (BS=1) — 04-27-2026 (triton) vs optimize-2 (csgmv)")
lines.append("")
lines.append("> Speedup = 04-27-2026_e2e / optimize-2_e2e — higher means csgmv is faster")
lines.append("")
lines.append("| Scenario | np | BS | in | out | 04-27-2026 mean_e2e(ms) | optimize-2 mean_e2e(ms) | Speedup | 04-27-2026 mean_ttft(ms) | optimize-2 mean_ttft(ms) | TTFT Speedup | 04-27-2026 mean_tpot(ms) | optimize-2 mean_tpot(ms) | TPOT Speedup | 04-27-2026 total_tok/s | optimize-2 total_tok/s | Tput Ratio |")
lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")

for name, np_val, bs, in_len, out_len in scenarios_lat:
    m1 = read_avg(os.path.join(branch2_lora_lat_dir, f"lora_{name}.jsonl"))
    m2 = read_avg(os.path.join(opt2_dir, f"lora_{name}.jsonl"))
    if m1 and m2:
        lines.append(
            f"| {name} | {np_val} | {bs} | {in_len} | {out_len} "
            f"| {m1['mean_e2e']:.2f} | {m2['mean_e2e']:.2f} | {ratio(m1['mean_e2e'], m2['mean_e2e'])} "
            f"| {m1['mean_ttft']:.2f} | {m2['mean_ttft']:.2f} | {ratio(m1['mean_ttft'], m2['mean_ttft'])} "
            f"| {m1['mean_tpot']:.2f} | {m2['mean_tpot']:.2f} | {ratio(m1['mean_tpot'], m2['mean_tpot'])} "
            f"| {m1['total_tput']:.2f} | {m2['total_tput']:.2f} | {ratio(m2['total_tput'], m1['total_tput'])} |"
        )
lines.append("")

# Compare LoRA throughput across branches
lines.append("### LoRA High Throughput (BS=64) — 04-27-2026 (triton) vs optimize-2 (csgmv)")
lines.append("")
lines.append("| Scenario | np | BS | in | out | 04-27-2026 mean_e2e(ms) | optimize-2 mean_e2e(ms) | Speedup | 04-27-2026 total_tok/s | optimize-2 total_tok/s | Tput Ratio |")
lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")

for name, np_val, bs, in_len, out_len in scenarios_tput:
    m1 = read_avg(os.path.join(branch2_tput_dir, f"lora_{name}.jsonl"))
    m2 = read_avg(os.path.join(opt2_dir, f"lora_{name}.jsonl"))
    if m1 and m2:
        lines.append(
            f"| {name} | {np_val} | {bs} | {in_len} | {out_len} "
            f"| {m1['mean_e2e']:.2f} | {m2['mean_e2e']:.2f} | {ratio(m1['mean_e2e'], m2['mean_e2e'])} "
            f"| {m1['total_tput']:.2f} | {m2['total_tput']:.2f} | {ratio(m2['total_tput'], m1['total_tput'])} |"
        )
lines.append("")

report = "\n".join(lines)
output_path = "/tmp/full_bench_report.md"
with open(output_path, "w") as f:
    f.write(report)
print(f"Report written to {output_path}")
print(f"Total lines: {len(lines)}")
