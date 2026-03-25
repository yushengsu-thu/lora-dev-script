import sglang as sgl
import torch
import os
import argparse


def get_prompt_logprobs(engine, input_ids, batch_input_ids=False):
    if batch_input_ids:
        if isinstance(input_ids, torch.Tensor):
            input_ids = [input_ids.tolist()]
        elif not isinstance(input_ids[0], list):
            input_ids = [input_ids]
    out = engine.generate(
        input_ids=input_ids,
        sampling_params={"max_new_tokens": 0, "temperature": 0.0},
        return_logprob=True,
        logprob_start_len=0,
    )
    if batch_input_ids and isinstance(out, list):
        out = out[0]
    return [logprob for logprob, _, _ in out["meta_info"]["input_token_logprobs"]][1:]


def main():
    parser = argparse.ArgumentParser(description="Dump base model logprobs (no LoRA)")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--adapter-path", type=str, required=True,
                        help="Path containing compare_sample_train_data.pt")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--batch-input-ids", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attention-backend", type=str, default="flashinfer")
    parser.add_argument("--prefill-attention-backend", type=str, default=None)
    parser.add_argument("--decode-attention-backend", type=str, default=None)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--disable-shared-experts-fusion", action="store_true")
    args = parser.parse_args()

    engine_kwargs = dict(
        model_path=args.model_path,
        tp_size=args.tp,
        attention_backend=args.attention_backend,
        disable_cuda_graph=args.disable_cuda_graph,
        prefill_attention_backend=args.prefill_attention_backend,
        decode_attention_backend=args.decode_attention_backend,
    )
    if args.trust_remote_code:
        engine_kwargs["trust_remote_code"] = True
    if args.disable_shared_experts_fusion:
        engine_kwargs["disable_shared_experts_fusion"] = True

    engine = sgl.Engine(**engine_kwargs)

    cdata = torch.load(
        os.path.join(args.adapter_path, "compare_sample_train_data.pt"),
        weights_only=False,
    )
    logprobs = get_prompt_logprobs(engine, cdata["tokens"], batch_input_ids=args.batch_input_ids)

    logprobs_tensor = torch.tensor(logprobs)
    tokens = cdata["tokens"] if isinstance(cdata["tokens"], torch.Tensor) else torch.tensor(cdata["tokens"])

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save({
        "logprobs": logprobs_tensor,
        "tokens": tokens,
        "model_path": args.model_path,
    }, args.output_path)

    print(f"[DUMP] Saved to {args.output_path}")
    print(f"[DUMP] logprobs shape={logprobs_tensor.shape}")
    print(f"[DUMP] first5={logprobs[:5]}")
    print(f"[DUMP] mean={logprobs_tensor.mean().item():.6f}")

    engine.shutdown()


if __name__ == "__main__":
    main()
