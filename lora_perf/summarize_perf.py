#!/usr/bin/env python3
"""Summarize SGLang bench_one_batch_server JSONL results.

Reads JSONL files named <scenario_tag>_bs<BS>.jsonl under --result-dir, prints
formatted tables to stdout, and writes both a markdown and TSV summary to the
result directory for easy copy-paste into Google Docs / Sheets.

Each JSONL file is written by sglang.bench_one_batch_server and contains one
record per measurement with these fields:
    run_name, batch_size, input_len, output_len,
    latency, input_throughput, output_throughput, overall_throughput,
    last_ttft, last_gen_throughput, acc_length, cache_hit_rate

NOTE on metrics semantics when num_waves > 1:
  - `overall_throughput` (batch_size * (in+out) / wall_latency) is the
    authoritative metric: total tokens / total wall time.
  - `last_gen_throughput` is a server-side instantaneous decode TPS.
  - `output_throughput` / `input_throughput` are computed from wall time and
    so include queueing across waves; they under-represent pure decode/prefill.
  - `last_ttft` is the TTFT of the *last* prompt to start, not the first wave;
    it's meaningful only for num_waves=1.

Can also be run standalone after a bench has completed:
    python summarize_perf.py --result-dir ./perf_results_X --batch-sizes "1 128 512"

Deps: `tabulate` (pip install tabulate). Used for pretty + markdown rendering;
TSV is hand-rolled to keep Google Sheets paste byte-stable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tabulate import tabulate


SCENARIO_TAGS: Dict[str, Tuple[str, str]] = {
    # logical_name: (file_tag,   pretty_label)
    # CUDA Graph is on for every scenario, so the file tag no longer marks it.
    "base":     ("base",     "Base"),
    "lora":     ("lora",     "LoRA"),
    "lora_opt": ("lora_opt", "LoRA optimized"),
}


@dataclass(frozen=True)
class Metric:
    key: str                        # key in the JSONL record
    header: str                     # short column header
    spec: str                       # bare format spec (tabulate floatfmt), e.g. ".1f"
    higher_is_better: bool = True   # for per-metric comparison tables


# We only report overall_throughput from bench_one_batch_server: it's the only
# metric that is unambiguously accurate under multi-wave batching
# (total_tokens / wall_time). output_throughput / input_throughput divide
# token counts by wall time that also includes the other phase + cross-wave
# queueing, and last_ttft is the TTFT of whichever prompt started last rather
# than the head of a wave. We keep `N` and `latency` as pure context columns
# (sample count and wall clock) — overall_throughput is derived from them.
DETAIL_METRICS: List[Metric] = [
    Metric("batch_size",          "N",          "d",   True),   # total prompts (= max(min_samples, num_waves*BS))
    Metric("latency",             "wall(s)",    ".2f", False),
    # overall_throughput = N * (input_len + output_len) / wall_latency
    # i.e. total (input + output) tokens per second across all prompts.
    Metric("overall_throughput",  "tokens/s",   ".1f", True),
]

# Cross-scenario comparison tables: overall_throughput only.
COMPARISON_METRICS: List[Metric] = [
    m for m in DETAIL_METRICS if m.key == "overall_throughput"
]


# --------------------------------------------------------------------------- #
# Table data model: hold raw values, defer formatting to tabulate (pretty/md)
# or to a tiny hand-rolled emitter (tsv, which must stay unpadded for Sheets).
# --------------------------------------------------------------------------- #

@dataclass
class Table:
    title: str
    headers: List[str] = field(default_factory=list)
    rows: List[List[Any]] = field(default_factory=list)
    # Per-column format spec for floats (tabulate `floatfmt`), e.g. ".1f".
    # Non-float cells are passed through untouched by tabulate.
    # Empty/short list falls back to "" (tabulate default).
    floatfmt: List[str] = field(default_factory=list)

    def add(self, *vals) -> None:
        self.rows.append(list(vals))

    def _tabulate(self, tablefmt: str) -> str:
        return tabulate(
            self.rows, self.headers,
            tablefmt=tablefmt,
            floatfmt=self.floatfmt or "",
            missingval="N/A",
        )

    def pretty(self) -> str:
        return f"{self.title}\n{self._tabulate('simple')}"

    def markdown(self) -> str:
        return f"### {self.title}\n\n{self._tabulate('pipe')}"

    def tsv(self) -> str:
        # Hand-rolled: tabulate's "tsv" pads cells with spaces, which
        # Google Sheets imports literally. We want unpadded cells.
        lines = [f"# {self.title}", "\t".join(self.headers)]
        for r in self.rows:
            lines.append("\t".join(self._tsv_cell(i, v) for i, v in enumerate(r)))
        return "\n".join(lines)

    def _tsv_cell(self, col: int, v: Any) -> str:
        if v is None:
            return "N/A"
        if isinstance(v, float) and col < len(self.floatfmt) and self.floatfmt[col]:
            return format(v, self.floatfmt[col])
        return str(v)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_last_record(path: str) -> Optional[dict]:
    """Each bench run appends one JSON line; pick the last non-empty line."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = [ln for ln in f if ln.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        print(f"WARN: could not parse last line of {path}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------- #
# Table builders
# --------------------------------------------------------------------------- #

def build_run_info(args, scenarios: List[str], batch_sizes: List[int]) -> Table:
    t = Table("Run Info")
    t.headers = ["GPU", "TP", "EP", "input_len", "output_len", "batch_sizes", "scenarios"]
    t.add(
        args.gpu,
        args.tp,
        args.ep,
        args.input_len,
        args.output_len,
        ",".join(map(str, batch_sizes)),
        " ".join(scenarios),
    )
    return t


def build_detail_table(label: str, records_by_bs: Dict[int, Optional[dict]]) -> Table:
    t = Table(f"{label} - detail  (BS = --max-running-requests; N = total prompts sent)")
    t.headers = ["BS"] + [m.header for m in DETAIL_METRICS]
    # BS column is integer; detail metrics carry their own specs.
    t.floatfmt = ["d"] + [m.spec for m in DETAIL_METRICS]
    for bs in sorted(records_by_bs):
        r = records_by_bs[bs]
        row: List[Any] = [bs]
        for m in DETAIL_METRICS:
            row.append(None if r is None else r.get(m.key))
        t.add(*row)
    return t


def build_comparison_table(
    metric: Metric,
    scenarios: List[str],
    all_data: Dict[str, Dict[int, Optional[dict]]],
    batch_sizes: List[int],
) -> Table:
    direction = "higher is better" if metric.higher_is_better else "lower is better"
    t = Table(f"{metric.key} by scenario ({direction})")

    # Column headers: one per scenario, then ratio columns vs base (if base present).
    t.headers = ["BS"] + [SCENARIO_TAGS[s][1] for s in scenarios]
    ratio_scenarios = [s for s in scenarios if s != "base"] if "base" in scenarios else []
    for s in ratio_scenarios:
        t.headers.append(f"{s}/base")
    # Ratio columns are preformatted strings ("99.5%"); no floatfmt needed there.
    t.floatfmt = ["d"] + [metric.spec] * len(scenarios) + [""] * len(ratio_scenarios)

    for bs in batch_sizes:
        values: List[Optional[float]] = []
        for s in scenarios:
            rec = all_data.get(s, {}).get(bs)
            values.append(None if rec is None else rec.get(metric.key))

        row: List[Any] = [bs] + list(values)

        if ratio_scenarios:
            base_val = values[scenarios.index("base")]
            for s in ratio_scenarios:
                v = values[scenarios.index(s)]
                if v is not None and base_val not in (None, 0):
                    row.append(f"{v / base_val * 100:.1f}%")
                else:
                    row.append("N/A")
        t.add(*row)
    return t


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-dir",  default=None,
                   help="Per-combo dir (required unless --combine is set).")
    p.add_argument("--batch-sizes", required=True,
                   help='Space-separated, e.g. "1 128 512"')
    p.add_argument("--scenarios",   default="base lora lora_opt",
                   help='Subset of: base lora lora_opt')
    p.add_argument("--gpu",         default="?")
    p.add_argument("--tp",          default="?")
    p.add_argument("--ep",          default="?")
    p.add_argument("--input-len",   default="?")
    p.add_argument("--output-len",  default="?")
    p.add_argument("--format",      default="pretty",
                   choices=["pretty", "markdown", "tsv"],
                   help="Console output format")
    p.add_argument("--md-file",     default=None,
                   help="Path to write markdown summary (default: <result-dir>/summary.md)")
    p.add_argument("--tsv-file",    default=None,
                   help="Path to write TSV summary (default: <result-dir>/summary.tsv)")
    # ---- combined sweep mode ----
    p.add_argument("--combine", action="store_true",
                   help="Aggregate multiple per-combo dirs into one sweep summary.")
    p.add_argument("--combo",   action="append", default=[],
                   help="With --combine: repeatable 'tp=<n>,ep=<n>,dir=<path>'.")
    p.add_argument("--out-dir", default=None,
                   help="With --combine: where to write combined_summary.{md,tsv}.")
    args = p.parse_args()
    if args.combine:
        if not args.combo:
            p.error("--combine requires at least one --combo spec")
    else:
        if not args.result_dir:
            p.error("--result-dir is required unless --combine is set")
    return args


def parse_combo_spec(spec: str) -> Dict[str, str]:
    """'tp=4,ep=1,dir=/path' -> {'tp': '4', 'ep': '1', 'dir': '/path'}."""
    out: Dict[str, str] = {}
    for part in spec.split(","):
        if "=" not in part:
            raise ValueError(f"--combo spec missing '=' in part {part!r} (full: {spec!r})")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    missing = {"tp", "ep", "dir"} - out.keys()
    if missing:
        raise ValueError(f"--combo spec missing keys {sorted(missing)} in {spec!r}")
    return out


def build_sweep_table(
    combos: List[Dict[str, str]],
    scenarios: List[str],
    batch_sizes: List[int],
) -> Table:
    """Flat table: one row per (combo, BS). Shows overall_throughput for
    each scenario plus per-scenario ratios vs base when 'base' is in play."""
    t = Table("Sweep: overall_throughput (tokens/s) by (TP, EP, BS) & scenario")
    scenario_labels = [SCENARIO_TAGS[s][1] for s in scenarios]
    ratio_scs = [s for s in scenarios if s != "base"] if "base" in scenarios else []
    t.headers = (
        ["TP", "EP", "BS"]
        + scenario_labels
        + [f"{SCENARIO_TAGS[s][1]}/base" for s in ratio_scs]
    )
    t.floatfmt = (
        ["d", "d", "d"]
        + [".1f"] * len(scenarios)
        + [""] * len(ratio_scs)
    )

    for combo in combos:
        tp_i, ep_i = int(combo["tp"]), int(combo["ep"])
        for bs in batch_sizes:
            vals: Dict[str, Optional[float]] = {}
            for s in scenarios:
                tag = SCENARIO_TAGS[s][0]
                rec = load_last_record(os.path.join(combo["dir"], f"{tag}_bs{bs}.jsonl"))
                vals[s] = rec.get("overall_throughput") if rec else None
            row: List[Any] = [tp_i, ep_i, bs] + [vals[s] for s in scenarios]
            base_v = vals.get("base")
            for s in ratio_scs:
                v = vals[s]
                if v is not None and base_v not in (None, 0):
                    row.append(f"{v / base_v * 100:.1f}%")
                else:
                    row.append("N/A")
            t.add(*row)
    return t


def run_combine(args: argparse.Namespace) -> None:
    combos = [parse_combo_spec(c) for c in args.combo]
    batch_sizes = [int(x) for x in args.batch_sizes.split()]
    scenarios = [s for s in args.scenarios.split() if s in SCENARIO_TAGS]
    if not scenarios:
        print(f"ERROR: no valid scenarios in '{args.scenarios}'", file=sys.stderr)
        sys.exit(2)

    out_dir = args.out_dir or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    info = Table("Sweep Run Info")
    info.headers = ["GPU", "input_len", "output_len", "batch_sizes", "scenarios", "combos"]
    info.add(
        args.gpu,
        args.input_len,
        args.output_len,
        ",".join(map(str, batch_sizes)),
        " ".join(scenarios),
        len(combos),
    )
    tables = [info, build_sweep_table(combos, scenarios, batch_sizes)]

    render = {"pretty": Table.pretty, "markdown": Table.markdown, "tsv": Table.tsv}[args.format]
    print("\n\n".join(render(t) for t in tables))

    md_path = os.path.join(out_dir, "combined_summary.md")
    tsv_path = os.path.join(out_dir, "combined_summary.tsv")
    with open(md_path, "w") as f:
        f.write("\n\n".join(t.markdown() for t in tables) + "\n")
    with open(tsv_path, "w") as f:
        f.write("\n\n".join(t.tsv() for t in tables) + "\n")

    print(f"\nMarkdown summary: {md_path}", file=sys.stderr)
    print(f"TSV      summary: {tsv_path}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    if args.combine:
        run_combine(args)
        return

    batch_sizes = [int(x) for x in args.batch_sizes.split()]
    scenarios = [s for s in args.scenarios.split() if s in SCENARIO_TAGS]
    if not scenarios:
        print(f"ERROR: no valid scenarios in '{args.scenarios}'", file=sys.stderr)
        sys.exit(2)

    # Load all records up front: all_data[scenario][bs] = record dict or None.
    # LoRA-variant info (adapter count, distribution) is encoded in the
    # result-dir name by run_perf_lora.sh, so filenames stay simple.
    all_data: Dict[str, Dict[int, Optional[dict]]] = {}
    for s in scenarios:
        tag = SCENARIO_TAGS[s][0]
        all_data[s] = {
            bs: load_last_record(os.path.join(args.result_dir, f"{tag}_bs{bs}.jsonl"))
            for bs in batch_sizes
        }

    tables: List[Table] = []
    tables.append(build_run_info(args, scenarios, batch_sizes))
    for s in scenarios:
        tables.append(build_detail_table(SCENARIO_TAGS[s][1], all_data[s]))
    for m in COMPARISON_METRICS:
        tables.append(build_comparison_table(m, scenarios, all_data, batch_sizes))

    # Console output
    render = {"pretty": Table.pretty, "markdown": Table.markdown, "tsv": Table.tsv}[args.format]
    print("\n\n".join(render(t) for t in tables))

    # Always persist md + tsv for later copy-paste.
    md_path = args.md_file or os.path.join(args.result_dir, "summary.md")
    tsv_path = args.tsv_file or os.path.join(args.result_dir, "summary.tsv")
    with open(md_path, "w") as f:
        f.write("\n\n".join(t.markdown() for t in tables) + "\n")
    with open(tsv_path, "w") as f:
        f.write("\n\n".join(t.tsv() for t in tables) + "\n")

    print(f"\nMarkdown summary: {md_path}", file=sys.stderr)
    print(f"TSV      summary: {tsv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
