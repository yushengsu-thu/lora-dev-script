import {
  BarChart,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
} from "cursor/canvas";

export default function LoRAProfileAnalysis() {
  const theme = useHostTheme();

  return (
    <Stack gap={24} style={{ padding: 24 }}>
      <Stack gap={4}>
        <H1>LoRA vs No-LoRA Profile Analysis</H1>
        <Text tone="secondary" size="small">
          Qwen3-30B-A3B-Instruct-2507 | TP=4 | BS=128 | input=2048, output=128
          | TP-0 GPU trace | 2026-05-17
        </Text>
      </Stack>

      {/* Top-level throughput comparison */}
      <Grid columns={4} gap={12}>
        <Stat label="LoRA Latency" value="13.64s" tone="danger" />
        <Stat label="No-LoRA Latency" value="6.62s" tone="success" />
        <Stat label="LoRA Throughput" value="20,413 tok/s" tone="danger" />
        <Stat label="No-LoRA Throughput" value="42,072 tok/s" tone="success" />
      </Grid>

      <Grid columns={3} gap={12}>
        <Stat label="LoRA TTFT" value="11.97s" tone="danger" />
        <Stat label="No-LoRA TTFT" value="5.57s" tone="success" />
        <Stat label="LoRA Overhead" value="2.06x slower" tone="warning" />
      </Grid>

      <Divider />

      {/* GPU kernel time breakdown */}
      <H2>GPU Kernel Time by Category (TP-0)</H2>
      <Grid columns={2} gap={12}>
        <Stat label="LoRA Total GPU Time" value="829 ms" />
        <Stat label="No-LoRA Total GPU Time" value="452 ms" />
      </Grid>

      <BarChart
        categories={[
          "NCCL Comm",
          "MoE Routing/GEMM",
          "LoRA Kernels",
          "Linear GEMM",
          "LayerNorm",
          "Attention",
          "Activation",
          "QK Norm",
          "RoPE",
          "Other",
        ]}
        series={[
          {
            name: "LoRA",
            data: [371.1, 291.9, 104.9, 16.4, 12.3, 11.1, 4.4, 3.1, 2.0, 12.0],
            tone: "danger",
          },
          {
            name: "No-LoRA",
            data: [303.9, 97.3, 0, 16.5, 12.2, 11.1, 0, 3.5, 2.0, 5.4],
            tone: "success",
          },
        ]}
        height={280}
        valueSuffix=" ms"
      />
      <Text tone="secondary" size="small">
        Source: perfetto-compatible TP-0 trace, 4 profiled steps (1 prefill + 3
        decode)
      </Text>

      <Divider />

      {/* The big bottleneck: MoE path */}
      <H2>Primary Bottleneck: MoE + LoRA Path (+299 ms)</H2>
      <Text>
        The MoE computation path accounts for 79% of the total overhead
        (299ms out of 377ms delta). LoRA mode uses the Triton fused_moe_kernel
        instead of cuBLAS bmm, and adds dedicated LoRA shrink/expand kernels.
      </Text>

      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader>LoRA MoE Path — 396.8 ms</CardHeader>
          <CardBody>
            <Table
              headers={["Kernel", "Time (ms)", "Calls", "% of MoE"]}
              rows={[
                ["fused_moe_kernel (Triton)", "228.8", "768", "57.7%"],
                ["_chunked_lora_expand", "70.1", "392", "17.7%"],
                ["count_and_sort_expert_tokens", "24.7", "576", "6.2%"],
                ["_chunked_lora_shrink", "19.0", "388", "4.8%"],
                ["moe_align_block_size", "17.8", "576", "4.5%"],
                ["moe_sum_reduce", "16.4", "192", "4.1%"],
                ["_moe_lora_shrink_splitk", "15.7", "384", "4.0%"],
                ["topkGatingSoftmax", "2.9", "192", "0.7%"],
                ["virtual_topk_ids", "0.8", "384", "0.2%"],
                ["sanitize_expert_ids", "0.6", "384", "0.1%"],
              ]}
              rowTone={["danger", "danger", "warning", "warning", undefined, undefined, "warning", undefined, undefined, undefined]}
            />
          </CardBody>
        </Card>

        <Card>
          <CardHeader>No-LoRA MoE Path — 97.3 ms</CardHeader>
          <CardBody>
            <Table
              headers={["Kernel", "Time (ms)", "Calls", "% of MoE"]}
              rows={[
                ["bmm (cuBLAS gate_up)", "45.5", "192", "46.8%"],
                ["bmm (cuBLAS down)", "30.4", "192", "31.2%"],
                ["finalize (reduce)", "17.3", "192", "17.8%"],
                ["routingIndicesHistogram", "2.0", "192", "2.1%"],
                ["routingIndicesCoop", "1.8", "192", "1.8%"],
                ["routingInitExpertCounts", "0.3", "192", "0.3%"],
              ]}
            />
          </CardBody>
        </Card>
      </Grid>

      <Divider />

      {/* Decomposition of the delta */}
      <H2>Overhead Decomposition (LoRA − No-LoRA = +377 ms)</H2>

      <BarChart
        categories={[
          "fused_moe vs cuBLAS",
          "LoRA expand",
          "NCCL AllReduce",
          "MoE routing overhead",
          "LoRA shrink",
          "LoRA shrink_splitk",
          "Other",
        ]}
        series={[
          {
            name: "Delta (ms)",
            data: [
              228.8 - 75.9 - 17.3,
              70.1,
              67.2,
              24.7 + 17.8 + 16.4 + 2.9 + 0.8 + 0.6 - 4.1,
              19.0,
              15.7,
              377 - (228.8 - 75.9 - 17.3) - 70.1 - 67.2 - (24.7 + 17.8 + 16.4 + 2.9 + 0.8 + 0.6 - 4.1) - 19.0 - 15.7,
            ],
          },
        ]}
        height={240}
        valueSuffix=" ms"
      />

      <Divider />

      {/* Key findings */}
      <H2>Key Findings</H2>
      <Stack gap={12}>
        <Card>
          <CardHeader>1. Triton fused_moe_kernel is 2.4x slower than cuBLAS bmm</CardHeader>
          <CardBody>
            <Text>
              fused_moe_kernel: 228.8ms (768 calls, avg 298us) vs cuBLAS bmm +
              finalize: 93.2ms (576 calls, avg 162us). The Triton MoE kernel
              runs 4x per layer per step (vs 2x for cuBLAS), suggesting
              virtual-expert splitting doubles calls. This single kernel is the
              #1 bottleneck at +135.6ms delta.
            </Text>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>2. LoRA expand/shrink kernels add 104.9ms</CardHeader>
          <CardBody>
            <Text>
              _chunked_lora_expand (70.1ms, avg 179us) is the heaviest LoRA
              kernel — it runs once per expert group per layer. _chunked_lora_shrink
              (19.0ms) and _moe_lora_shrink_splitk (15.7ms) together add another
              34.7ms. These are inherent to LoRA but could potentially be fused
              with the MoE kernel.
            </Text>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>3. NCCL AllReduce is 22% slower (+67ms)</CardHeader>
          <CardBody>
            <Text>
              Same call count (388), but avg latency rises from 781us to 952us
              (+22%). Likely because LoRA increases the tensor sizes being
              reduced, or GPU-side compute delays push AllReduce into less
              favorable scheduling windows.
            </Text>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>4. MoE routing overhead adds ~59ms</CardHeader>
          <CardBody>
            <Text>
              LoRA mode uses a different routing path: count_and_sort_expert_tokens
              (24.7ms), moe_align_block_size (17.8ms), moe_sum_reduce (16.4ms)
              are all absent or much cheaper in no-LoRA. The virtual-expert
              mapping adds extra sorting and alignment passes.
            </Text>
          </CardBody>
        </Card>
      </Stack>

      <Divider />

      {/* Actionable suggestions */}
      <H2>Optimization Opportunities</H2>
      <Table
        headers={["Priority", "Target", "Potential Saving", "Approach"]}
        rows={[
          ["P0", "fused_moe_kernel", "~136 ms", "Switch to cuBLAS grouped GEMM backend or optimize Triton kernel tile sizes for BS=128"],
          ["P1", "_chunked_lora_expand", "~70 ms", "Fuse with MoE GEMM; reduce the number of kernel launches by batching across experts"],
          ["P2", "MoE routing (sort/align)", "~59 ms", "Use the v2 routing path (routingCustom) that no-LoRA uses; reduce double-sort overhead"],
          ["P3", "NCCL AllReduce", "~67 ms", "Profile whether LoRA increases tensor size; check scheduling overlap with compute"],
          ["P4", "LoRA shrink kernels", "~35 ms", "Fuse shrink + splitk into a single kernel; reduce launch overhead"],
        ]}
        rowTone={["danger", "danger", "warning", "warning", undefined]}
      />

      <Text tone="secondary" size="small" style={{ marginTop: 8 }}>
        Total theoretical max saving: ~367ms → would bring LoRA within ~1ms of
        no-LoRA GPU time. Realistic P0+P1 target: ~200ms saving (bringing
        overhead from 2.06x to ~1.3x).
      </Text>
    </Stack>
  );
}
