# LoRA vs No-LoRA Profile Analysis

> **Qwen3-30B-A3B-Instruct-2507** | TP=4 | BS=128 | input=2048, output=128 | TP-0 GPU trace | 2026-05-17

## Summary

|  | LoRA | No-LoRA | Delta |
|--|------|---------|-------|
| **Latency** | 13.64s | 6.62s | **2.06x slower** |
| **Overall Throughput** | 20,413 tok/s | 42,072 tok/s | -51.5% |
| **TTFT** | 11.97s | 5.57s | +6.40s |
| **Output Throughput** | 9,763 tok/s | 15,543 tok/s | -37.2% |
| **GPU Kernel Time (TP-0)** | 829 ms | 452 ms | **+377 ms (+83.5%)** |

## GPU Kernel Time by Category (TP-0)

| Category | LoRA (ms) | No-LoRA (ms) | Delta (ms) | Delta % |
|----------|----------:|-------------:|-----------:|--------:|
| NCCL Comm | 371.1 | 303.9 | +67.2 | +22.1% |
| MoE Routing/GEMM | 291.9 | 97.3 | +194.6 | +200.0% |
| LoRA Kernels | 104.9 | 0.0 | +104.9 | NEW |
| Linear GEMM (nvjet) | 16.4 | 16.5 | -0.1 | -0.5% |
| LayerNorm/RMSNorm | 12.3 | 12.2 | +0.0 | +0.3% |
| Attention | 11.1 | 11.1 | +0.0 | +0.3% |
| Activation | 4.4 | 0.0 | +4.4 | NEW |
| QK Norm | 3.1 | 3.5 | -0.4 | -11.8% |
| RoPE | 2.0 | 2.0 | -0.0 | -2.0% |
| Other | 12.0 | 5.4 | +6.6 | +122.5% |

## Primary Bottleneck: MoE + LoRA Path (+299 ms, 79% of total overhead)

The MoE computation path accounts for **79% of the total overhead** (299ms out of 377ms delta). LoRA mode uses the Triton `fused_moe_kernel` instead of cuBLAS `bmm`, and adds dedicated LoRA shrink/expand kernels.

### LoRA MoE Path — 396.8 ms

| Kernel | Time (ms) | Calls | % of MoE |
|--------|----------:|------:|---------:|
| **fused_moe_kernel (Triton)** | **228.8** | 768 | 57.7% |
| **_chunked_lora_expand** | **70.1** | 392 | 17.7% |
| count_and_sort_expert_tokens | 24.7 | 576 | 6.2% |
| _chunked_lora_shrink | 19.0 | 388 | 4.8% |
| moe_align_block_size | 17.8 | 576 | 4.5% |
| moe_sum_reduce | 16.4 | 192 | 4.1% |
| _moe_lora_shrink_splitk | 15.7 | 384 | 4.0% |
| topkGatingSoftmax | 2.9 | 192 | 0.7% |
| virtual_topk_ids | 0.8 | 384 | 0.2% |
| sanitize_expert_ids | 0.6 | 384 | 0.1% |

### No-LoRA MoE Path — 97.3 ms

| Kernel | Time (ms) | Calls | % of MoE |
|--------|----------:|------:|---------:|
| bmm (cuBLAS gate_up) | 45.5 | 192 | 46.8% |
| bmm (cuBLAS down) | 30.4 | 192 | 31.2% |
| finalize (reduce) | 17.3 | 192 | 17.8% |
| routingIndicesHistogram | 2.0 | 192 | 2.1% |
| routingIndicesCoop | 1.8 | 192 | 1.8% |
| routingInitExpertCounts | 0.3 | 192 | 0.3% |

## Overhead Decomposition (LoRA − No-LoRA = +377 ms)

| Source | Delta (ms) | Share |
|--------|----------:|------:|
| fused_moe vs cuBLAS bmm | +135.6 | 36.0% |
| LoRA expand kernel | +70.1 | 18.6% |
| NCCL AllReduce | +67.2 | 17.8% |
| MoE routing overhead | +59.1 | 15.7% |
| LoRA shrink kernel | +19.0 | 5.0% |
| LoRA shrink_splitk kernel | +15.7 | 4.2% |
| Other | +10.3 | 2.7% |

## Key Findings

### 1. Triton `fused_moe_kernel` is 2.4x slower than cuBLAS `bmm`

`fused_moe_kernel`: 228.8ms (768 calls, avg 298μs) vs cuBLAS `bmm` + `finalize`: 93.2ms (576 calls, avg 162μs). The Triton MoE kernel runs **4x per layer per step** (vs 2x for cuBLAS), suggesting virtual-expert splitting doubles calls. This single kernel is the **#1 bottleneck** at +135.6ms delta.

### 2. LoRA expand/shrink kernels add 104.9ms

`_chunked_lora_expand` (70.1ms, avg 179μs) is the heaviest LoRA kernel — it runs once per expert group per layer. `_chunked_lora_shrink` (19.0ms) and `_moe_lora_shrink_splitk` (15.7ms) together add another 34.7ms. These are inherent to LoRA but could potentially be fused with the MoE kernel.

### 3. NCCL AllReduce is 22% slower (+67ms)

Same call count (388), but avg latency rises from 781μs to 952μs (+22%). Likely because LoRA increases the tensor sizes being reduced, or GPU-side compute delays push AllReduce into less favorable scheduling windows.

### 4. MoE routing overhead adds ~59ms

LoRA mode uses a different routing path: `count_and_sort_expert_tokens` (24.7ms), `moe_align_block_size` (17.8ms), `moe_sum_reduce` (16.4ms) are all absent or much cheaper in no-LoRA. The virtual-expert mapping adds extra sorting and alignment passes.

## Optimization Opportunities

| Priority | Target | Potential Saving | Approach |
|----------|--------|----------------:|----------|
| **P0** | `fused_moe_kernel` | ~136 ms | Switch to cuBLAS grouped GEMM backend or optimize Triton kernel tile sizes for BS=128 |
| **P1** | `_chunked_lora_expand` | ~70 ms | Fuse with MoE GEMM; reduce kernel launches by batching across experts |
| **P2** | MoE routing (sort/align) | ~59 ms | Use the v2 routing path (`routingCustom`) that no-LoRA uses; reduce double-sort overhead |
| **P3** | NCCL AllReduce | ~67 ms | Profile whether LoRA increases tensor size; check scheduling overlap with compute |
| **P4** | LoRA shrink kernels | ~35 ms | Fuse shrink + splitk into a single kernel; reduce launch overhead |

> Total theoretical max saving: ~367ms → would bring LoRA within ~10ms of no-LoRA GPU time.
> Realistic P0+P1 target: **~200ms saving** (bringing overhead from 2.06x to ~1.3x).
