"""
Benchmark: _align_block_size_jit (CUDA JIT v2) vs _align_block_size_torch (PyTorch fallback)

Measures the wall-clock time of the align_block_size operation for large
num_experts (> 1024), which is the hot path for LoRA virtual expert routing.

Usage:
    PYTHONPATH=/home/radixark/yushengsu/sglang/python python bench_align_block_size.py
"""

import time
import torch
import triton

torch.set_grad_enabled(False)

# ─── Reference: pure-PyTorch implementation (the old slow path) ───

@torch.compile(dynamic=True)
def _align_block_size_torch(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = topk_ids.device
    flat_topk_ids = topk_ids.reshape(-1).to(torch.int64)
    num_total_tokens = flat_topk_ids.numel()

    sentinel = num_experts
    valid_mask = (flat_topk_ids >= 0) & (flat_topk_ids < num_experts)
    safe_topk_ids = torch.where(
        valid_mask,
        flat_topk_ids,
        torch.full_like(flat_topk_ids, sentinel),
    )

    bucket_count = num_experts + 1
    max_total_padded_tokens = (
        (num_total_tokens + bucket_count * (block_size - 1) + block_size - 1)
        // block_size
    ) * block_size
    max_num_blocks = max_total_padded_tokens // block_size

    sorted_token_ids = torch.full(
        (max_total_padded_tokens,), num_total_tokens, dtype=torch.int32, device=device,
    )
    expert_ids = torch.full(
        (max_num_blocks,), -1, dtype=torch.int32, device=device,
    )

    if num_total_tokens == 0:
        num_tokens_post_padded = torch.zeros((1,), dtype=torch.int32, device=device)
        return sorted_token_ids, expert_ids, num_tokens_post_padded

    sorted_order = torch.argsort(safe_topk_ids)
    sorted_expert_ids = safe_topk_ids[sorted_order]
    expert_range = torch.arange(bucket_count, device=device, dtype=torch.int64)
    counts_offsets = torch.searchsorted(sorted_expert_ids, expert_range, right=False)
    counts_end = torch.searchsorted(sorted_expert_ids, expert_range, right=True)
    counts = counts_end - counts_offsets
    padded_counts = ((counts + block_size - 1) // block_size) * block_size
    total_padded_tokens = padded_counts.sum().to(torch.int32).reshape(1)
    padded_offsets = torch.cumsum(padded_counts, dim=0) - padded_counts

    token_ranks = (
        torch.arange(num_total_tokens, device=device, dtype=torch.int64)
        - counts_offsets[sorted_expert_ids]
    )
    output_positions = padded_offsets[sorted_expert_ids] + token_ranks
    sorted_token_ids.scatter_(
        0, output_positions.to(torch.int64), sorted_order.to(torch.int32),
    )

    block_counts = padded_counts // block_size
    real_block_counts = block_counts.clone()
    real_block_counts[sentinel] = 0
    actual_num_blocks = real_block_counts.sum()

    if max_num_blocks <= 0:
        return sorted_token_ids, expert_ids, total_padded_tokens

    block_offsets = torch.cumsum(real_block_counts, dim=0)
    all_block_positions = torch.arange(max_num_blocks, device=device, dtype=torch.int64)
    assigned_experts = torch.searchsorted(
        block_offsets, all_block_positions, right=True
    ).to(torch.int32)
    expert_ids.copy_(
        torch.where(
            all_block_positions < actual_num_blocks,
            assigned_experts,
            torch.full_like(assigned_experts, -1),
        )
    )

    return sorted_token_ids, expert_ids, total_padded_tokens


# ─── JIT v1: original (function-level import, 4 separate allocs) ───

def _align_block_size_jit_v1(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from sglang.jit_kernel.moe_align import (
        moe_align_block_size as jit_moe_align_block_size,
    )

    device = topk_ids.device
    flat_topk_ids = topk_ids.reshape(-1)
    if flat_topk_ids.dtype == torch.int64:
        flat_topk_ids = flat_topk_ids.to(torch.int32)
    num_total_tokens = flat_topk_ids.numel()

    jit_num_experts = num_experts + 1

    if num_total_tokens < jit_num_experts:
        max_num_tokens_padded = num_total_tokens * block_size
    else:
        max_num_tokens_padded = num_total_tokens + jit_num_experts * (block_size - 1)

    sorted_token_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=device
    )
    num_tokens_post_padded = torch.empty((1,), dtype=torch.int32, device=device)
    cumsum_buffer = torch.empty(
        (jit_num_experts + 1,), dtype=torch.int32, device=device
    )

    jit_moe_align_block_size(
        flat_topk_ids,
        jit_num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        cumsum_buffer,
        True,
    )

    return sorted_token_ids, expert_ids, num_tokens_post_padded


# ─── JIT v2: optimized (module-level import, fused alloc, inline cdiv) ───

from sglang.jit_kernel.moe_align import (
    moe_align_block_size as _jit_moe_align_block_size,
)

def _align_block_size_jit_v2(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = topk_ids.device
    flat_topk_ids = topk_ids.reshape(-1)
    if flat_topk_ids.dtype == torch.int64:
        flat_topk_ids = flat_topk_ids.to(torch.int32)
    num_total_tokens = flat_topk_ids.numel()

    jit_num_experts = num_experts + 1

    if num_total_tokens < jit_num_experts:
        max_num_tokens_padded = num_total_tokens * block_size
    else:
        max_num_tokens_padded = num_total_tokens + jit_num_experts * (block_size - 1)

    # Align to 4: CUDA kernel fills sorted_token_ids with int4 vec writes;
    # last write can spill up to 3 int32s into adjacent expert_ids region.
    max_num_tokens_padded = (max_num_tokens_padded + 3) & ~3

    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size

    total_buf = max_num_tokens_padded + max_num_m_blocks + 1 + (jit_num_experts + 1)
    buf = torch.empty(total_buf, dtype=torch.int32, device=device)
    off = 0
    sorted_token_ids = buf[off : off + max_num_tokens_padded]
    off += max_num_tokens_padded
    expert_ids = buf[off : off + max_num_m_blocks]
    off += max_num_m_blocks
    num_tokens_post_padded = buf[off : off + 1]
    off += 1
    cumsum_buffer = buf[off : off + jit_num_experts + 1]

    _jit_moe_align_block_size(
        flat_topk_ids,
        jit_num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        cumsum_buffer,
        True,
    )

    return sorted_token_ids, expert_ids, num_tokens_post_padded


# ─── Benchmark harness ───

def make_topk_ids(num_tokens: int, top_k: int, num_experts: int,
                  sentinel_ratio: float = 0.05, device: str = "cuda"):
    """Generate realistic topk_ids with some sentinel (-1) values."""
    ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device)
    mask = torch.rand(num_tokens, top_k, device=device) < sentinel_ratio
    ids[mask] = -1
    return ids


def bench_fn(fn, topk_ids, block_size, num_experts, warmup=5, repeats=20):
    """Time a function with CUDA synchronization."""
    for _ in range(warmup):
        fn(topk_ids, block_size, num_experts)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end_events   = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

    for i in range(repeats):
        start_events[i].record()
        fn(topk_ids, block_size, num_experts)
        end_events[i].record()
    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times_ms.sort()
    trimmed = times_ms[2:-2] if len(times_ms) > 4 else times_ms
    avg = sum(trimmed) / len(trimmed)
    return avg, min(times_ms), max(times_ms)


def verify_correctness(topk_ids, block_size, num_experts):
    """Check that JIT v2 and PyTorch produce compatible outputs."""
    s_jit, e_jit, n_jit = _align_block_size_jit_v2(topk_ids, block_size, num_experts)
    s_ref, e_ref, n_ref = _align_block_size_torch(topk_ids, block_size, num_experts)

    n_jit_val = n_jit.item()
    n_ref_val = n_ref.item()

    n_jit_blocks = triton.cdiv(n_jit_val, block_size)
    n_ref_blocks = triton.cdiv(n_ref_val, block_size)

    # Collect which token ids land in each expert for both implementations
    def get_expert_token_sets(sorted_ids, expert_ids, n_padded, bs):
        n_blocks = triton.cdiv(n_padded, bs)
        result = {}
        for b in range(min(n_blocks, expert_ids.numel())):
            eid = expert_ids[b].item()
            if eid < 0:
                continue
            tokens = set()
            for j in range(bs):
                idx = b * bs + j
                if idx < sorted_ids.numel():
                    tid = sorted_ids[idx].item()
                    num_total = topk_ids.numel()
                    if tid < num_total:
                        tokens.add(tid)
            if eid not in result:
                result[eid] = set()
            result[eid].update(tokens)
        return result

    jit_map = get_expert_token_sets(s_jit, e_jit, n_jit_val, block_size)
    ref_map = get_expert_token_sets(s_ref, e_ref, n_ref_val, block_size)

    all_experts = set(jit_map.keys()) | set(ref_map.keys())
    mismatches = 0
    for eid in all_experts:
        jit_tokens = jit_map.get(eid, set())
        ref_tokens = ref_map.get(eid, set())
        if jit_tokens != ref_tokens:
            mismatches += 1
            if mismatches <= 3:
                print(f"    MISMATCH expert {eid}: "
                      f"jit has {len(jit_tokens)} tokens, ref has {len(ref_tokens)} tokens, "
                      f"diff: {jit_tokens.symmetric_difference(ref_tokens)}")

    return mismatches == 0


def main():
    device = "cuda"
    print("=" * 100)
    print("  Benchmark: JIT v2 (optimized) vs JIT v1 (original) vs PyTorch fallback")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("  v1 = function-level import + 4x torch.empty + triton.cdiv")
    print("  v2 = module-level import + fused alloc + inline cdiv")
    print("=" * 100)

    configs = [
        (64,   2, 1024,  64, "8E×128L, 64tok"),
        (256,  2, 1024,  64, "8E×128L, 256tok"),
        (1024, 2, 1024,  64, "8E×128L, 1024tok"),
        (64,   2, 2048,  64, "8E×256L, 64tok"),
        (256,  2, 2048,  64, "8E×256L, 256tok"),
        (1024, 2, 2048,  64, "8E×256L, 1024tok"),
        (64,   2, 4096,  64, "64E×64L, 64tok"),
        (256,  2, 4096,  64, "64E×64L, 256tok"),
        (1024, 2, 4096,  64, "64E×64L, 1024tok"),
        (1,    2, 2048,  64, "8E×256L, 1tok (decode)"),
        (4,    2, 2048,  64, "8E×256L, 4tok (decode)"),
        (16,   2, 2048,  64, "8E×256L, 16tok (decode)"),
    ]

    # ── Phase 1: Correctness ──
    print("\n── Phase 1: Correctness verification (v2 vs torch) ──")
    print("  (Warming up torch.compile, first call may be slow...)\n")
    test_ids = make_topk_ids(128, 2, 2048, device=device)
    _align_block_size_torch(test_ids, 64, 2048)
    _align_block_size_torch(test_ids, 64, 2048)

    all_correct = True
    for num_tokens, top_k, num_experts, block_size, label in configs:
        topk_ids = make_topk_ids(num_tokens, top_k, num_experts, device=device)
        ok = verify_correctness(topk_ids, block_size, num_experts)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label:30s}  (E={num_experts}, N={num_tokens}×{top_k}, BS={block_size})")
        if not ok:
            all_correct = False

    if not all_correct:
        print("\n  WARNING: Some correctness checks failed!")
    else:
        print("\n  All correctness checks passed.")

    # ── Phase 2: Performance ──
    print("\n── Phase 2: Performance benchmark (50 repeats each) ──")
    print(f"  {'Config':<30s} | {'v2(opt) avg':>10s} | {'v1(orig) avg':>11s} | {'Torch avg':>10s} | {'v2/v1':>6s} | {'v2/Torch':>8s}")
    print("  " + "-" * 30 + "-+-" + "-" * 10 + "-+-" + "-" * 11 + "-+-" + "-" * 10 + "-+-" + "-" * 6 + "-+-" + "-" * 8)

    for num_tokens, top_k, num_experts, block_size, label in configs:
        topk_ids = make_topk_ids(num_tokens, top_k, num_experts, device=device)

        v2_avg, v2_min, v2_max = bench_fn(
            _align_block_size_jit_v2, topk_ids, block_size, num_experts,
            warmup=10, repeats=50,
        )
        v1_avg, v1_min, v1_max = bench_fn(
            _align_block_size_jit_v1, topk_ids, block_size, num_experts,
            warmup=10, repeats=50,
        )
        torch_avg, torch_min, torch_max = bench_fn(
            _align_block_size_torch, topk_ids, block_size, num_experts,
            warmup=5, repeats=30,
        )

        v1_speedup = v1_avg / v2_avg if v2_avg > 0 else float('inf')
        torch_speedup = torch_avg / v2_avg if v2_avg > 0 else float('inf')

        print(f"  {label:<30s} | "
              f"{v2_avg:>8.3f}ms | "
              f"{v1_avg:>9.3f}ms | "
              f"{torch_avg:>8.3f}ms | "
              f"{v1_speedup:>5.2f}x | "
              f"{torch_speedup:>6.2f}x")

    # ── Phase 3: High-frequency simulation ──
    print("\n── Phase 3: 100K-call simulation (BS=1 decode scenario) ──")
    print("  Simulating 48 MoE layers × 2048 decode steps = ~100K calls\n")

    topk_ids = make_topk_ids(1, 2, 2048, device=device)
    num_calls = 100_000

    for name, fn in [("JIT v2 (optimized)", _align_block_size_jit_v2),
                     ("JIT v1 (original)",  _align_block_size_jit_v1)]:
        # warmup
        for _ in range(100):
            fn(topk_ids, 64, 2048)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_calls):
            fn(topk_ids, 64, 2048)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(f"  {name:<25s}: {elapsed*1000:>8.1f}ms total, {elapsed/num_calls*1e6:>6.2f}µs/call")

    print("\n" + "=" * 100)
    print("  Done.")


if __name__ == "__main__":
    main()
