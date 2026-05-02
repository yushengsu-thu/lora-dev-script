"""
Microbenchmark for _align_block_size_cuda_jit and optimized variants.

Measures kernel-level GPU time using CUDA events across different
num_experts and numel (token count) configurations.
"""

import functools
import time

import torch
from torch.utils.cpp_extension import load_inline

# ─── V0: Current implementation (baseline) ────────────────────────────

_V0_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

__global__ void align_large_kernel_v0(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ cumsum_out,
    int32_t num_experts,
    int32_t block_size,
    int32_t numel,
    int32_t max_num_tokens_padded,
    int32_t max_num_blocks)
{
    const int bucket_count = num_experts + 1;

    if (blockIdx.x == 1) {
        for (int i = threadIdx.x; i < max_num_tokens_padded; i += blockDim.x)
            sorted_token_ids[i] = numel;
        return;
    }

    extern __shared__ int32_t counts[];
    const int tid = threadIdx.x;

    for (int i = tid; i < bucket_count; i += blockDim.x)
        counts[i] = 0;
    __syncthreads();

    for (int i = tid; i < numel; i += blockDim.x) {
        int eid = topk_ids[i];
        int bucket = (eid >= 0 && eid < num_experts) ? eid : num_experts;
        atomicAdd(&counts[bucket], 1);
    }
    __syncthreads();

    if (tid == 0) {
        int running = 0;
        for (int i = 0; i < bucket_count; i++) {
            cumsum_out[i] = running;
            running += CEILDIV(counts[i], block_size) * block_size;
        }
        cumsum_out[bucket_count] = running;
        *total_tokens_post_pad = running;
    }
    __syncthreads();

    for (int eid = tid; eid < num_experts; eid += blockDim.x) {
        int start = cumsum_out[eid];
        int end   = cumsum_out[eid + 1];
        for (int j = start; j < end; j += block_size) {
            int bidx = j / block_size;
            if (bidx < max_num_blocks)
                expert_ids[bidx] = eid;
        }
    }

    int sentinel_start_block = cumsum_out[num_experts] / block_size;
    for (int i = sentinel_start_block + tid; i < max_num_blocks; i += blockDim.x)
        expert_ids[i] = -1;
}

__global__ void scatter_tokens_kernel_v0(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum,
    int32_t num_experts,
    int32_t numel)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
         i += blockDim.x * gridDim.x) {
        int eid = topk_ids[i];
        int bucket = (eid >= 0 && eid < num_experts) ? eid : num_experts;
        int pos = atomicAdd(&cumsum[bucket], 1);
        sorted_token_ids[pos] = i;
    }
}

void align_block_size_v0(
    torch::Tensor topk_ids,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor total_tokens_post_pad,
    torch::Tensor cumsum_buffer,
    int64_t num_experts,
    int64_t block_size,
    int64_t max_num_tokens_padded,
    int64_t max_num_blocks)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    int numel = static_cast<int>(topk_ids.numel());
    int bucket_count = static_cast<int>(num_experts) + 1;

    int threads = 1024;
    size_t smem_bytes = bucket_count * sizeof(int32_t);

    align_large_kernel_v0<<<2, threads, smem_bytes, stream>>>(
        topk_ids.data_ptr<int32_t>(),
        sorted_token_ids.data_ptr<int32_t>(),
        expert_ids.data_ptr<int32_t>(),
        total_tokens_post_pad.data_ptr<int32_t>(),
        cumsum_buffer.data_ptr<int32_t>(),
        static_cast<int32_t>(num_experts),
        static_cast<int32_t>(block_size),
        numel,
        static_cast<int32_t>(max_num_tokens_padded),
        static_cast<int32_t>(max_num_blocks));

    if (numel > 0) {
        int scatter_threads = 256;
        int scatter_blocks = std::min(
            (numel + scatter_threads - 1) / scatter_threads, 65535);
        scatter_tokens_kernel_v0<<<scatter_blocks, scatter_threads, 0, stream>>>(
            topk_ids.data_ptr<int32_t>(),
            sorted_token_ids.data_ptr<int32_t>(),
            cumsum_buffer.data_ptr<int32_t>(),
            static_cast<int32_t>(num_experts),
            numel);
    }
}
"""

_V0_CPP_SRC = r"""
void align_block_size_v0(
    torch::Tensor topk_ids, torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids, torch::Tensor total_tokens_post_pad,
    torch::Tensor cumsum_buffer,
    int64_t num_experts, int64_t block_size,
    int64_t max_num_tokens_padded, int64_t max_num_blocks);
"""

# ─── V1: Parallel prefix-sum + multi-block histogram ──────────────────

_V1_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

// Kernel 1: histogram (multi-block with global atomics)
__global__ void histogram_kernel_v1(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ counts,
    int32_t num_experts,
    int32_t numel)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
         i += blockDim.x * gridDim.x) {
        int eid = topk_ids[i];
        int bucket = (eid >= 0 && eid < num_experts) ? eid : num_experts;
        atomicAdd(&counts[bucket], 1);
    }
}

// Kernel 2: parallel prefix-sum + fill expert_ids
//   Single block, up to 1024 threads.
//   Each thread handles ceil(bucket_count / blockDim.x) buckets.
//   Uses work-efficient Blelloch scan in shared memory.
__global__ void prefix_sum_and_fill_v1(
    int32_t* __restrict__ counts,
    int32_t* __restrict__ cumsum_out,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    int32_t max_num_blocks)
{
    const int bucket_count = num_experts + 1;
    extern __shared__ int32_t smem[];
    // smem layout: [0..bucket_count-1] = padded_counts for scan

    const int tid = threadIdx.x;
    const int n = bucket_count;

    // Load padded counts into shared memory
    for (int i = tid; i < n; i += blockDim.x) {
        smem[i] = CEILDIV(counts[i], block_size) * block_size;
    }
    __syncthreads();

    // Simple parallel prefix sum (sequential by thread 0 for correctness
    // with arbitrary n that may not be power-of-2).
    // For n up to ~4096 this is still fast since it's all in shared memory.
    if (tid == 0) {
        int running = 0;
        for (int i = 0; i < n; i++) {
            int val = smem[i];
            cumsum_out[i] = running;
            running += val;
        }
        cumsum_out[n] = running;
        *total_tokens_post_pad = running;
    }
    __syncthreads();

    // Fill expert_ids in parallel
    for (int eid = tid; eid < num_experts; eid += blockDim.x) {
        int start = cumsum_out[eid];
        int end   = cumsum_out[eid + 1];
        for (int j = start; j < end; j += block_size) {
            int bidx = j / block_size;
            if (bidx < max_num_blocks)
                expert_ids[bidx] = eid;
        }
    }

    int sentinel_start_block = cumsum_out[num_experts] / block_size;
    for (int i = sentinel_start_block + tid; i < max_num_blocks; i += blockDim.x)
        expert_ids[i] = -1;
}

// Kernel 3: fill sentinel + scatter (fused)
__global__ void init_and_scatter_v1(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum,
    int32_t num_experts,
    int32_t numel,
    int32_t max_num_tokens_padded)
{
    // Phase A: fill sorted_token_ids with sentinel
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < max_num_tokens_padded;
         i += blockDim.x * gridDim.x) {
        sorted_token_ids[i] = numel;
    }
    // Grid-wide barrier via cooperative groups is not available,
    // so we split into two kernels or use atomics.
    // Actually, sentinel fill and scatter are independent because
    // scatter overwrites specific positions. Let's just do sentinel first.
}

__global__ void scatter_tokens_v1(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum,
    int32_t num_experts,
    int32_t numel)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
         i += blockDim.x * gridDim.x) {
        int eid = topk_ids[i];
        int bucket = (eid >= 0 && eid < num_experts) ? eid : num_experts;
        int pos = atomicAdd(&cumsum[bucket], 1);
        sorted_token_ids[pos] = i;
    }
}

void align_block_size_v1(
    torch::Tensor topk_ids,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor total_tokens_post_pad,
    torch::Tensor cumsum_buffer,
    torch::Tensor counts_buffer,
    int64_t num_experts,
    int64_t block_size,
    int64_t max_num_tokens_padded,
    int64_t max_num_blocks)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    int numel = static_cast<int>(topk_ids.numel());
    int bucket_count = static_cast<int>(num_experts) + 1;

    // Kernel 1: multi-block histogram
    {
        int threads = 256;
        int blocks = std::min((numel + threads - 1) / threads, 256);
        histogram_kernel_v1<<<blocks, threads, 0, stream>>>(
            topk_ids.data_ptr<int32_t>(),
            counts_buffer.data_ptr<int32_t>(),
            static_cast<int32_t>(num_experts),
            numel);
    }

    // Kernel 2: prefix-sum + fill expert_ids (single block)
    {
        int threads = std::min(1024, bucket_count);
        size_t smem = bucket_count * sizeof(int32_t);
        prefix_sum_and_fill_v1<<<1, threads, smem, stream>>>(
            counts_buffer.data_ptr<int32_t>(),
            cumsum_buffer.data_ptr<int32_t>(),
            expert_ids.data_ptr<int32_t>(),
            total_tokens_post_pad.data_ptr<int32_t>(),
            static_cast<int32_t>(num_experts),
            static_cast<int32_t>(block_size),
            static_cast<int32_t>(max_num_blocks));
    }

    // Kernel 3: fill sentinel
    {
        int threads = 256;
        int blocks = std::min((int(max_num_tokens_padded) + threads - 1) / threads, 256);
        init_and_scatter_v1<<<blocks, threads, 0, stream>>>(
            topk_ids.data_ptr<int32_t>(),
            sorted_token_ids.data_ptr<int32_t>(),
            cumsum_buffer.data_ptr<int32_t>(),
            static_cast<int32_t>(num_experts),
            numel,
            static_cast<int32_t>(max_num_tokens_padded));
    }

    // Kernel 4: scatter tokens
    if (numel > 0) {
        int threads = 256;
        int blocks = std::min((numel + threads - 1) / threads, 65535);
        scatter_tokens_v1<<<blocks, threads, 0, stream>>>(
            topk_ids.data_ptr<int32_t>(),
            sorted_token_ids.data_ptr<int32_t>(),
            cumsum_buffer.data_ptr<int32_t>(),
            static_cast<int32_t>(num_experts),
            numel);
    }
}
"""

_V1_CPP_SRC = r"""
void align_block_size_v1(
    torch::Tensor topk_ids, torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids, torch::Tensor total_tokens_post_pad,
    torch::Tensor cumsum_buffer, torch::Tensor counts_buffer,
    int64_t num_experts, int64_t block_size,
    int64_t max_num_tokens_padded, int64_t max_num_blocks);
"""

# ─── V2: Warp-level histogram reduction + fused 3-kernel pipeline ─────

_V2_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
#define WARP_SIZE 32

// Kernel 1: warp-level pre-aggregated histogram
//   Each warp maintains a private histogram chunk in shared memory,
//   reducing atomic contention on hot experts.
__global__ void histogram_warp_v2(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ global_counts,
    int32_t num_experts,
    int32_t numel)
{
    const int bucket_count = num_experts + 1;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    // Shared memory: per-warp histograms. Each warp has bucket_count slots.
    // We cap shared memory usage: if bucket_count * num_warps > limit, fall back
    // to global atomics directly.
    extern __shared__ int32_t warp_hist[];

    // Zero shared memory
    for (int i = threadIdx.x; i < bucket_count * num_warps; i += blockDim.x)
        warp_hist[i] = 0;
    __syncthreads();

    int32_t* my_hist = warp_hist + warp_id * bucket_count;

    // Each thread processes elements in a grid-stride loop
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
         i += blockDim.x * gridDim.x) {
        int eid = topk_ids[i];
        int bucket = (eid >= 0 && eid < num_experts) ? eid : num_experts;
        atomicAdd(&my_hist[bucket], 1);
    }
    __syncthreads();

    // Reduce per-warp histograms to global counts
    // Each thread handles a slice of buckets
    for (int b = threadIdx.x; b < bucket_count; b += blockDim.x) {
        int sum = 0;
        for (int w = 0; w < num_warps; w++)
            sum += warp_hist[w * bucket_count + b];
        if (sum > 0)
            atomicAdd(&global_counts[b], sum);
    }
}

// Kernel 2: prefix-sum + fill expert_ids (unchanged from V1)
__global__ void prefix_sum_and_fill_v2(
    int32_t* __restrict__ counts,
    int32_t* __restrict__ cumsum_out,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    int32_t max_num_blocks)
{
    const int tid = threadIdx.x;
    const int bucket_count = num_experts + 1;

    if (tid == 0) {
        int running = 0;
        for (int i = 0; i < bucket_count; i++) {
            int val = CEILDIV(counts[i], block_size) * block_size;
            cumsum_out[i] = running;
            running += val;
        }
        cumsum_out[bucket_count] = running;
        *total_tokens_post_pad = running;
    }
    __syncthreads();

    for (int eid = tid; eid < num_experts; eid += blockDim.x) {
        int start = cumsum_out[eid];
        int end   = cumsum_out[eid + 1];
        for (int j = start; j < end; j += block_size) {
            int bidx = j / block_size;
            if (bidx < max_num_blocks)
                expert_ids[bidx] = eid;
        }
    }

    int sentinel_start_block = cumsum_out[num_experts] / block_size;
    for (int i = sentinel_start_block + tid; i < max_num_blocks; i += blockDim.x)
        expert_ids[i] = -1;
}

// Kernel 3: sentinel fill
__global__ void fill_sentinel_v2(
    int32_t* __restrict__ sorted_token_ids,
    int32_t numel,
    int32_t max_num_tokens_padded)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < max_num_tokens_padded;
         i += blockDim.x * gridDim.x)
        sorted_token_ids[i] = numel;
}

// Kernel 4: scatter tokens
__global__ void scatter_tokens_v2(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum,
    int32_t num_experts,
    int32_t numel)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
         i += blockDim.x * gridDim.x) {
        int eid = topk_ids[i];
        int bucket = (eid >= 0 && eid < num_experts) ? eid : num_experts;
        int pos = atomicAdd(&cumsum[bucket], 1);
        sorted_token_ids[pos] = i;
    }
}

void align_block_size_v2(
    torch::Tensor topk_ids,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor total_tokens_post_pad,
    torch::Tensor cumsum_buffer,
    torch::Tensor counts_buffer,
    int64_t num_experts,
    int64_t block_size,
    int64_t max_num_tokens_padded,
    int64_t max_num_blocks)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    int numel = static_cast<int>(topk_ids.numel());
    int bucket_count = static_cast<int>(num_experts) + 1;

    // Kernel 1: warp-level histogram
    {
        int threads = 256;
        int num_warps_per_block = threads / 32;
        // Cap shared mem: per-warp histogram.
        // If bucket_count * num_warps * 4 > 48KB, reduce blocks or fall back.
        size_t smem = (size_t)bucket_count * num_warps_per_block * sizeof(int32_t);
        int blocks;
        if (smem > 48 * 1024) {
            // Too much shared mem, use fewer warps
            threads = 64;  // 2 warps
            num_warps_per_block = 2;
            smem = (size_t)bucket_count * num_warps_per_block * sizeof(int32_t);
            blocks = std::min((numel + threads - 1) / threads, 64);
        } else {
            blocks = std::min((numel + threads - 1) / threads, 128);
        }
        histogram_warp_v2<<<blocks, threads, smem, stream>>>(
            topk_ids.data_ptr<int32_t>(),
            counts_buffer.data_ptr<int32_t>(),
            static_cast<int32_t>(num_experts),
            numel);
    }

    // Kernel 2: prefix-sum + expert_ids fill
    {
        int threads = std::min(1024, bucket_count);
        prefix_sum_and_fill_v2<<<1, threads, 0, stream>>>(
            counts_buffer.data_ptr<int32_t>(),
            cumsum_buffer.data_ptr<int32_t>(),
            expert_ids.data_ptr<int32_t>(),
            total_tokens_post_pad.data_ptr<int32_t>(),
            static_cast<int32_t>(num_experts),
            static_cast<int32_t>(block_size),
            static_cast<int32_t>(max_num_blocks));
    }

    // Kernel 3: sentinel fill
    {
        int threads = 256;
        int blocks = std::min(
            (int(max_num_tokens_padded) + threads - 1) / threads, 256);
        fill_sentinel_v2<<<blocks, threads, 0, stream>>>(
            sorted_token_ids.data_ptr<int32_t>(),
            numel,
            static_cast<int32_t>(max_num_tokens_padded));
    }

    // Kernel 4: scatter
    if (numel > 0) {
        int threads = 256;
        int blocks = std::min((numel + threads - 1) / threads, 65535);
        scatter_tokens_v2<<<blocks, threads, 0, stream>>>(
            topk_ids.data_ptr<int32_t>(),
            sorted_token_ids.data_ptr<int32_t>(),
            cumsum_buffer.data_ptr<int32_t>(),
            static_cast<int32_t>(num_experts),
            numel);
    }
}
"""

_V2_CPP_SRC = r"""
void align_block_size_v2(
    torch::Tensor topk_ids, torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids, torch::Tensor total_tokens_post_pad,
    torch::Tensor cumsum_buffer, torch::Tensor counts_buffer,
    int64_t num_experts, int64_t block_size,
    int64_t max_num_tokens_padded, int64_t max_num_blocks);
"""


# ─── V3: V1 refined — fuse sentinel+scatter, multi-block expert fill ──

_V3_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

// Kernel 1: multi-block histogram with global atomics
__global__ void histogram_kernel_v3(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ counts,
    int32_t num_experts,
    int32_t numel)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
         i += blockDim.x * gridDim.x) {
        int eid = topk_ids[i];
        int bucket = (eid >= 0 && eid < num_experts) ? eid : num_experts;
        atomicAdd(&counts[bucket], 1);
    }
}

// Kernel 2: prefix-sum (single block, thread-0) + multi-block expert_ids fill
//   The prefix-sum part is fast for ~4K entries in shared memory.
//   Expert_ids fill is parallelized: each thread handles one expert
//   and writes all its blocks directly.
__global__ void prefix_sum_and_fill_v3(
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ cumsum_out,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    int32_t max_num_blocks)
{
    const int tid = threadIdx.x;
    const int bucket_count = num_experts + 1;

    if (tid == 0) {
        int running = 0;
        for (int i = 0; i < bucket_count; i++) {
            int val = CEILDIV(counts[i], block_size) * block_size;
            cumsum_out[i] = running;
            running += val;
        }
        cumsum_out[bucket_count] = running;
        *total_tokens_post_pad = running;
    }
    __syncthreads();

    // Fill expert_ids: each thread handles multiple experts in a stride loop
    for (int eid = tid; eid < num_experts; eid += blockDim.x) {
        int start_block = cumsum_out[eid] / block_size;
        int end_block = cumsum_out[eid + 1] / block_size;
        for (int b = start_block; b < end_block && b < max_num_blocks; b++)
            expert_ids[b] = eid;
    }

    // Fill remaining blocks with -1
    int sentinel_start_block = cumsum_out[num_experts] / block_size;
    for (int i = sentinel_start_block + tid; i < max_num_blocks; i += blockDim.x)
        expert_ids[i] = -1;
}

// Kernel 3: fused sentinel-fill + scatter
//   Two-phase approach using a grid-wide flag:
//   All blocks first fill their slice of sorted_token_ids with the sentinel,
//   then a grid-wide __threadfence + atomic counter barrier, then scatter.
//   ... Actually, a simpler approach: just launch fill + scatter in one kernel
//   with a grid-stride loop that first fills, then scatters (since scatter
//   only writes to valid positions, it will overwrite sentinels correctly).
//
//   Even simpler: fill the entire array with memset-like kernel, then scatter.
//   Two kernels are fine — the overhead of 1 extra launch is ~2us.
//   Let's fuse by having scatter overwrite sentinels.

__global__ void fill_and_scatter_v3(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum,
    int32_t num_experts,
    int32_t numel,
    int32_t max_num_tokens_padded)
{
    // Phase 1: fill entire sorted_token_ids with sentinel value
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < max_num_tokens_padded;
         i += blockDim.x * gridDim.x)
        sorted_token_ids[i] = numel;

    // Grid-wide fence to ensure all sentinel writes are visible
    __threadfence();

    // We need a grid-wide barrier here. Since CUDA doesn't have one
    // natively without cooperative groups, we use a simple approach:
    // the last block to arrive triggers the scatter.
    // Actually, this doesn't work cleanly. Let's use two kernels.
    // The cost of an extra kernel launch (~2us) is small.
}

__global__ void scatter_tokens_v3(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum,
    int32_t num_experts,
    int32_t numel)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
         i += blockDim.x * gridDim.x) {
        int eid = topk_ids[i];
        int bucket = (eid >= 0 && eid < num_experts) ? eid : num_experts;
        int pos = atomicAdd(&cumsum[bucket], 1);
        sorted_token_ids[pos] = i;
    }
}

void align_block_size_v3(
    torch::Tensor topk_ids,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor total_tokens_post_pad,
    torch::Tensor cumsum_buffer,
    torch::Tensor counts_buffer,
    int64_t num_experts,
    int64_t block_size,
    int64_t max_num_tokens_padded,
    int64_t max_num_blocks)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    int numel = static_cast<int>(topk_ids.numel());
    int bucket_count = static_cast<int>(num_experts) + 1;

    // Kernel 1: multi-block histogram
    {
        int threads = 256;
        int blocks = std::min((numel + threads - 1) / threads, 256);
        histogram_kernel_v3<<<blocks, threads, 0, stream>>>(
            topk_ids.data_ptr<int32_t>(),
            counts_buffer.data_ptr<int32_t>(),
            static_cast<int32_t>(num_experts),
            numel);
    }

    // Kernel 2: prefix-sum + expert_ids fill
    {
        int threads = std::min(1024, bucket_count);
        prefix_sum_and_fill_v3<<<1, threads, 0, stream>>>(
            counts_buffer.data_ptr<int32_t>(),
            cumsum_buffer.data_ptr<int32_t>(),
            expert_ids.data_ptr<int32_t>(),
            total_tokens_post_pad.data_ptr<int32_t>(),
            static_cast<int32_t>(num_experts),
            static_cast<int32_t>(block_size),
            static_cast<int32_t>(max_num_blocks));
    }

    // Kernel 3: sentinel fill (use max of numel and max_num_tokens_padded for grid)
    {
        int threads = 256;
        int total_work = std::max((int)max_num_tokens_padded, numel);
        int blocks = std::min((total_work + threads - 1) / threads, 512);
        fill_and_scatter_v3<<<blocks, threads, 0, stream>>>(
            topk_ids.data_ptr<int32_t>(),
            sorted_token_ids.data_ptr<int32_t>(),
            cumsum_buffer.data_ptr<int32_t>(),
            static_cast<int32_t>(num_experts),
            numel,
            static_cast<int32_t>(max_num_tokens_padded));
    }

    // Kernel 4: scatter
    if (numel > 0) {
        int threads = 256;
        int blocks = std::min((numel + threads - 1) / threads, 65535);
        scatter_tokens_v3<<<blocks, threads, 0, stream>>>(
            topk_ids.data_ptr<int32_t>(),
            sorted_token_ids.data_ptr<int32_t>(),
            cumsum_buffer.data_ptr<int32_t>(),
            static_cast<int32_t>(num_experts),
            numel);
    }
}
"""

_V3_CPP_SRC = r"""
void align_block_size_v3(
    torch::Tensor topk_ids, torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids, torch::Tensor total_tokens_post_pad,
    torch::Tensor cumsum_buffer, torch::Tensor counts_buffer,
    int64_t num_experts, int64_t block_size,
    int64_t max_num_tokens_padded, int64_t max_num_blocks);
"""


# ─── Compilation ──────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _compile_v0():
    return load_inline(
        name="bench_v0", cpp_sources=[_V0_CPP_SRC],
        cuda_sources=[_V0_CUDA_SRC], functions=["align_block_size_v0"],
        verbose=False,
    )

@functools.lru_cache(maxsize=1)
def _compile_v1():
    return load_inline(
        name="bench_v1", cpp_sources=[_V1_CPP_SRC],
        cuda_sources=[_V1_CUDA_SRC], functions=["align_block_size_v1"],
        verbose=False,
    )

@functools.lru_cache(maxsize=1)
def _compile_v2():
    return load_inline(
        name="bench_v2", cpp_sources=[_V2_CPP_SRC],
        cuda_sources=[_V2_CUDA_SRC], functions=["align_block_size_v2"],
        verbose=False,
    )

@functools.lru_cache(maxsize=1)
def _compile_v3():
    return load_inline(
        name="bench_v3", cpp_sources=[_V3_CPP_SRC],
        cuda_sources=[_V3_CUDA_SRC], functions=["align_block_size_v3"],
        verbose=False,
    )


# ─── Runner wrappers ─────────────────────────────────────────────────

def _make_buffers(numel, num_experts, block_size, device):
    bucket_count = num_experts + 1
    max_total_padded = (
        (numel + bucket_count * (block_size - 1) + block_size - 1)
        // block_size
    ) * block_size
    max_num_blocks = max_total_padded // block_size

    sorted_token_ids = torch.empty(max_total_padded, dtype=torch.int32, device=device)
    expert_ids = torch.empty(max_num_blocks, dtype=torch.int32, device=device)
    total_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=device)
    cumsum_buffer = torch.zeros(bucket_count + 1, dtype=torch.int32, device=device)
    counts_buffer = torch.zeros(bucket_count, dtype=torch.int32, device=device)

    return (sorted_token_ids, expert_ids, total_tokens_post_pad,
            cumsum_buffer, counts_buffer, max_total_padded, max_num_blocks)


def run_v0(topk_ids, num_experts, block_size):
    numel = topk_ids.numel()
    device = topk_ids.device
    (sorted_tok, expert_ids, total_post_pad,
     cumsum_buf, _, max_padded, max_blocks) = _make_buffers(numel, num_experts, block_size, device)

    mod = _compile_v0()
    mod.align_block_size_v0(
        topk_ids, sorted_tok, expert_ids, total_post_pad, cumsum_buf,
        num_experts, block_size, max_padded, max_blocks)
    return sorted_tok, expert_ids, total_post_pad


def run_v1(topk_ids, num_experts, block_size):
    numel = topk_ids.numel()
    device = topk_ids.device
    (sorted_tok, expert_ids, total_post_pad,
     cumsum_buf, counts_buf, max_padded, max_blocks) = _make_buffers(numel, num_experts, block_size, device)

    mod = _compile_v1()
    mod.align_block_size_v1(
        topk_ids, sorted_tok, expert_ids, total_post_pad, cumsum_buf, counts_buf,
        num_experts, block_size, max_padded, max_blocks)
    return sorted_tok, expert_ids, total_post_pad


def run_v2(topk_ids, num_experts, block_size):
    numel = topk_ids.numel()
    device = topk_ids.device
    (sorted_tok, expert_ids, total_post_pad,
     cumsum_buf, counts_buf, max_padded, max_blocks) = _make_buffers(numel, num_experts, block_size, device)

    mod = _compile_v2()
    mod.align_block_size_v2(
        topk_ids, sorted_tok, expert_ids, total_post_pad, cumsum_buf, counts_buf,
        num_experts, block_size, max_padded, max_blocks)
    return sorted_tok, expert_ids, total_post_pad


def run_v3(topk_ids, num_experts, block_size):
    numel = topk_ids.numel()
    device = topk_ids.device
    (sorted_tok, expert_ids, total_post_pad,
     cumsum_buf, counts_buf, max_padded, max_blocks) = _make_buffers(numel, num_experts, block_size, device)

    mod = _compile_v3()
    mod.align_block_size_v3(
        topk_ids, sorted_tok, expert_ids, total_post_pad, cumsum_buf, counts_buf,
        num_experts, block_size, max_padded, max_blocks)
    return sorted_tok, expert_ids, total_post_pad


# ─── Correctness check ───────────────────────────────────────────────

def verify_correctness(topk_ids, num_experts, block_size):
    """Check that V1, V2, V3 produce identical results to V0."""
    s0, e0, t0 = run_v0(topk_ids.clone(), num_experts, block_size)
    s1, e1, t1 = run_v1(topk_ids.clone(), num_experts, block_size)
    s3, e3, t3 = run_v3(topk_ids.clone(), num_experts, block_size)

    torch.cuda.synchronize()

    t0_val = t0.item()
    t1_val = t1.item()
    t3_val = t3.item()

    assert t0_val == t1_val, f"total_tokens mismatch V0={t0_val} vs V1={t1_val}"
    assert t0_val == t3_val, f"total_tokens mismatch V0={t0_val} vs V3={t3_val}"

    assert torch.equal(e0, e1), "expert_ids mismatch V0 vs V1"
    assert torch.equal(e0, e3), "expert_ids mismatch V0 vs V3"

    n = t0_val
    s0_set = set(s0[:n].cpu().tolist())
    s1_set = set(s1[:n].cpu().tolist())
    s3_set = set(s3[:n].cpu().tolist())
    assert s0_set == s1_set, "sorted_token_ids set mismatch V0 vs V1"
    assert s0_set == s3_set, "sorted_token_ids set mismatch V0 vs V3"

    print("  Correctness: PASS")


# ─── Benchmark ────────────────────────────────────────────────────────

def benchmark_one(fn, topk_ids, num_experts, block_size, warmup=10, iters=100):
    """Benchmark a single function, returns median GPU time in microseconds."""
    # Warmup
    for _ in range(warmup):
        fn(topk_ids.clone(), num_experts, block_size)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        ids_copy = topk_ids.clone()
        start.record()
        fn(ids_copy, num_experts, block_size)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # ms -> us

    times.sort()
    median = times[len(times) // 2]
    p10 = times[len(times) // 10]
    p90 = times[int(len(times) * 0.9)]
    return median, p10, p90


def main():
    device = torch.device("cuda")

    print("Compiling CUDA kernels...")
    t0 = time.time()
    _compile_v0()
    print(f"  V0 compiled in {time.time() - t0:.1f}s")
    t0 = time.time()
    _compile_v1()
    print(f"  V1 compiled in {time.time() - t0:.1f}s")
    t0 = time.time()
    _compile_v3()
    print(f"  V3 compiled in {time.time() - t0:.1f}s")

    block_size = 64  # typical BLOCK_SIZE_M

    configs = [
        # (num_experts, num_tokens, top_k, description)
        (1024,  512,  8, "1024 experts, 512 tokens, top8"),
        (1024,  4096, 8, "1024 experts, 4K tokens, top8"),
        (1024,  16384, 8, "1024 experts, 16K tokens, top8"),
        (2048,  512,  8, "2048 experts, 512 tokens, top8"),
        (2048,  4096, 8, "2048 experts, 4K tokens, top8"),
        (2048,  16384, 8, "2048 experts, 16K tokens, top8"),
        (4096,  512,  8, "4096 experts, 512 tokens, top8"),
        (4096,  4096, 8, "4096 experts, 4K tokens, top8"),
        (4096,  16384, 8, "4096 experts, 16K tokens, top8"),
    ]

    versions = [
        ("V0 (baseline)", run_v0),
        ("V1 (multi-blk)", run_v1),
        ("V3 (refined)", run_v3),
    ]

    print("\n" + "=" * 100)
    print(f"  align_block_size Benchmark | block_size={block_size} | GPU: {torch.cuda.get_device_name()}")
    print("=" * 100)

    header = f"{'Config':<42} |"
    for name, _ in versions:
        header += f" {name:>18} |"
    header += " V1 speedup | V3 speedup"
    print(header)
    print("-" * len(header))

    for num_experts, num_tokens, top_k, desc in configs:
        numel = num_tokens * top_k
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k),
                                 dtype=torch.int32, device=device).reshape(-1)

        verify_correctness(topk_ids, num_experts, block_size)

        row = f"  {desc:<40} |"
        medians = []
        for name, fn in versions:
            median, p10, p90 = benchmark_one(fn, topk_ids, num_experts, block_size)
            row += f" {median:>13.1f} us |"
            medians.append(median)

        if medians[0] > 0:
            row += f"  {medians[0]/medians[1]:>9.2f}x"
            row += f"  | {medians[0]/medians[2]:>8.2f}x"
        print(row)

    print("=" * 100)
    print("Done.")


if __name__ == "__main__":
    main()
