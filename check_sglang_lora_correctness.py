import sys
import sglang as sgl
import torch
import os
import argparse


def kl_v2(a, b):
    a = torch.tensor(a) if not torch.is_tensor(a) else a
    b = torch.tensor(b) if not torch.is_tensor(b) else b
    return (((a - b) ** 2) * 0.5).mean().item()


def get_prompt_logprobs(engine, input_ids, lora_path, batch_input_ids=False):
    if batch_input_ids:
        if isinstance(input_ids, torch.Tensor):
            input_ids = [input_ids.tolist()]
        elif not isinstance(input_ids[0], list):
            input_ids = [input_ids]
    out = engine.generate(
        input_ids=input_ids,
        sampling_params={"max_new_tokens": 4, "temperature": 0.0},
        return_logprob=True,
        logprob_start_len=0,
        lora_path=lora_path,
    )
    if batch_input_ids and isinstance(out, list):
        out = out[0]
    return [logprob for logprob, _, _ in out["meta_info"]["input_token_logprobs"]][1:]


def parse_args():
    parser = argparse.ArgumentParser(
        description="SGLang Tinker Lora Adapter Correctness Test"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to lora adapter, including also the compare_sample_train_data.pt file",
    )
    parser.add_argument(
        "--model-path", type=str, help="Path or name of HF model to test"
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument(
        "--disable-cuda-graph", action="store_true", help="Disable CUDA graph (enabled by default)"
    )
    parser.add_argument(
        "--moe-runner-backend", type=str, default="auto", help="MoE runner backend"
    )
    parser.add_argument(
        "--experts-shared-outer-loras",
        action="store_true",
        help="Enable shared outer LoRA mode for MoE models",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="flashinfer",
        help="Attention backend (flashinfer, triton, fa3, etc.)",
    )
    parser.add_argument(
        "--prefill-attention-backend",
        type=str,
        default=None,
        help="Prefill attention backend (e.g. fa4). Overrides --attention-backend for prefill.",
    )
    parser.add_argument(
        "--decode-attention-backend",
        type=str,
        default=None,
        help="Decode attention backend (e.g. triton, flashinfer). Overrides --attention-backend for decode.",
    )
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        nargs="+",
        default=None,
        help="Target modules for LoRA. If not specified, auto-detected from adapter config.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading custom model code from HuggingFace",
    )
    parser.add_argument(
        "--disable-shared-experts-fusion",
        action="store_true",
        help="Disable shared expert fusion to keep shared experts as separate modules (needed for LoRA on MoE shared experts)",
    )
    parser.add_argument(
        "--batch-input-ids",
        action="store_true",
        help="Wrap input_ids as a batch (list of lists) for models that require it (e.g. DeepSeek-V3)",
    )
    parser.add_argument(
        "--enable-torch-compile",
        action="store_true",
        help="Optimize the model with torch.compile",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    engine_kwargs = dict(
        model_path=args.model_path,
        served_model_name="model",
        tp_size=args.tp,
        enable_lora=True,
        max_lora_rank=32,
        lora_paths={"my_lora": args.adapter_path},
        # lora_backend="triton",
        lora_backend="csgmv",
        attention_backend=args.attention_backend,
        disable_cuda_graph=args.disable_cuda_graph,
        prefill_attention_backend=args.prefill_attention_backend,
        decode_attention_backend=args.decode_attention_backend,
    )
    if args.lora_target_modules is not None:
        engine_kwargs["lora_target_modules"] = args.lora_target_modules
    if args.moe_runner_backend != "auto":
        engine_kwargs["moe_runner_backend"] = args.moe_runner_backend
    if args.experts_shared_outer_loras:
        engine_kwargs["experts_shared_outer_loras"] = True
    if args.trust_remote_code:
        engine_kwargs["trust_remote_code"] = True
    if args.disable_shared_experts_fusion:
        engine_kwargs["disable_shared_experts_fusion"] = True
    if args.enable_torch_compile:
        engine_kwargs["enable_torch_compile"] = True

    engine = sgl.Engine(**engine_kwargs)

    cdata = torch.load(
        os.path.join(args.adapter_path, "compare_sample_train_data.pt")
    )
    base_logprobs = get_prompt_logprobs(engine, cdata["tokens"], lora_path=None, batch_input_ids=args.batch_input_ids)
    logprobs = get_prompt_logprobs(engine, cdata["tokens"], lora_path="my_lora", batch_input_ids=args.batch_input_ids)

    base_t = torch.tensor(base_logprobs)
    lora_t = torch.tensor(logprobs)
    diff = (base_t - lora_t).abs()
    print(f"[VERIFY] base vs lora: mean_diff={diff.mean().item():.6f}, "
          f"max_diff={diff.max().item():.6f}, "
          f"identical={torch.equal(base_t, lora_t)}")
    print(f"[VERIFY] base first5={base_logprobs[:5]}")
    print(f"[VERIFY] lora first5={logprobs[:5]}")

    print(
        f"KL(orig_sampler, trainer)={kl_v2(cdata['training_logprobs'], cdata['sampling_logprobs'])}"
    )
    print(f"KL(sglang, trainer)={kl_v2(cdata['training_logprobs'], logprobs)}")
    print(f"KL(sglang, orig_sampler)={kl_v2(logprobs, cdata['sampling_logprobs'])}")


if __name__ == "__main__":
    main()
