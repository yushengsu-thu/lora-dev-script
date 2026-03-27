import sglang as sgl
import torch
import os
import argparse


def get_prefill_logprobs(engine, input_ids, batch_input_ids=False):
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


def get_decode_logprobs(engine, input_ids, max_new_tokens=32, batch_input_ids=False):
    if batch_input_ids:
        if isinstance(input_ids, torch.Tensor):
            input_ids = [input_ids.tolist()]
        elif not isinstance(input_ids[0], list):
            input_ids = [input_ids]
    out = engine.generate(
        input_ids=input_ids,
        sampling_params={"max_new_tokens": max_new_tokens, "temperature": 0.0},
        return_logprob=True,
        logprob_start_len=0,
    )
    if batch_input_ids and isinstance(out, list):
        out = out[0]
    meta = out["meta_info"]
    decode_lps = [logprob for logprob, _, _ in meta["output_token_logprobs"]]
    decode_ids = [tid for _, tid, _ in meta["output_token_logprobs"]]
    return decode_lps, decode_ids


def main():
    parser = argparse.ArgumentParser(description="Dump prefill + decode logprobs")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--adapter-path", type=str, required=True,
                        help="Path containing compare_sample_train_data.pt")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
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

    print("[PHASE] Running prefill ...")
    prefill_lps = get_prefill_logprobs(engine, cdata["tokens"], batch_input_ids=args.batch_input_ids)

    print("[PHASE] Running decode ...")
    decode_lps, decode_ids = get_decode_logprobs(
        engine, cdata["tokens"],
        max_new_tokens=args.max_new_tokens,
        batch_input_ids=args.batch_input_ids,
    )

    tokens = cdata["tokens"] if isinstance(cdata["tokens"], torch.Tensor) else torch.tensor(cdata["tokens"])

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save({
        "prefill_logprobs": torch.tensor(prefill_lps),
        "decode_logprobs": torch.tensor(decode_lps),
        "decode_token_ids": torch.tensor(decode_ids),
        "tokens": tokens,
        "model_path": args.model_path,
    }, args.output_path)

    print(f"[DUMP] Saved to {args.output_path}")
    print(f"[DUMP] prefill_logprobs len={len(prefill_lps)}, first5={prefill_lps[:5]}")
    print(f"[DUMP] decode_logprobs  len={len(decode_lps)},  first5={decode_lps[:5]}")
    print(f"[DUMP] decode_token_ids first5={decode_ids[:5]}")

    engine.shutdown()


if __name__ == "__main__":
    main()
