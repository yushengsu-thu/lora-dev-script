# perf_results_comp

LoRA vs No-LoRA benchmark + profiling results.

Each run creates a timestamped subdirectory (`YYYY-MM-DD_HH-MM-SS`).

## Directory Structure

```
perf_results_comp/<timestamp>/
├── lora.jsonl                            # Benchmark results (LoRA)
├── nolora.jsonl                          # Benchmark results (no LoRA)
├── lora.server.log                       # Server stdout/stderr (LoRA)
├── nolora.server.log                     # Server stdout/stderr (no LoRA)
├── profile_lora/<profile_id>/
│   ├── server_args.json                  # Server config snapshot
│   ├── <id>-TP-0-*.trace.json.gz        # Raw torch profiler trace (TP rank 0)
│   ├── <id>-TP-1-*.trace.json.gz        # Raw torch profiler trace (TP rank 1)
│   ├── ...                               # (one per TP rank)
│   ├── perfetto-compatible-*.json.gz     # Perfetto-compatible traces
│   ├── merged-<id>.trace.json.gz        # All TP ranks merged into one trace
│   └── perfetto-compatible-merged-*.gz   # Merged trace, Perfetto-compatible
└── profile_nolora/<profile_id>/
    └── (same structure as above)
```

## How to View Traces

Open any `.trace.json.gz` file in [Perfetto UI](https://ui.perfetto.dev):

- **Individual TP rank**: `<id>-TP-<rank>-*.trace.json.gz`
- **All TP ranks merged**: `merged-<id>.trace.json.gz`
- **Perfetto-compatible** (recommended): `perfetto-compatible-*.json.gz` — fixes overlapping GPU events for cleaner visualization

## Post-processing Scripts

Traces are automatically post-processed by two scripts from [fzyzcjy/torch_utils](https://github.com/fzyzcjy/torch_utils):

1. **convert_to_perfetto_compatible.py** — Fixes overlapping GPU kernel events so Perfetto renders them correctly on separate swim lanes
2. **sglang_profiler_trace_merger.py** — Merges per-TP-rank traces into a single file with `[TP00]`, `[TP01]`, … prefixed process names

Scripts are auto-downloaded to `torch_utils/` on first run.

## Re-running

```bash
bash run_tml_lora_vs_nolora_perf_one_bench_bs128.sh
```

Environment overrides:

| Variable | Default | Description |
|---|---|---|
| `RESULT_DIR` | `perf_results_comp/<timestamp>` | Override output directory |
| `LORA_BACKEND` | `csgmv` | LoRA backend to use |
| `SKIP_SERVER_WARMUP` | `0` | Set `1` to skip server warmup |
| `SKIP_NOLORA` | `0` | Set `1` to skip the no-LoRA baseline run |
