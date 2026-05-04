import asyncio, aiohttp, time, sys

URL = "http://localhost:30000/generate"

async def one(session, i, prompt_tokens, output_len):
    payload = {
        "input_ids": [(i * 31 + 7) % 50000] * prompt_tokens,
        "sampling_params": {
            "temperature": 0.0, "max_new_tokens": output_len,
            "ignore_eos": True, "stop": []
        },
        "lora_path": "my_lora",
    }
    t0 = time.time()
    try:
        async with session.post(URL, json=payload, timeout=600) as r:
            text = await r.text()
            return i, r.status, time.time() - t0, len(text)
    except Exception as e:
        return i, -1, time.time() - t0, repr(e)[:200]

async def main(n, plen, olen):
    timeout = aiohttp.ClientTimeout(total=900)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [one(session, i, plen, olen) for i in range(n)]
        for fut in asyncio.as_completed(tasks):
            i, st, dur, info = await fut
            print(f"req {i:3d} status={st} dur={dur:.2f}s info_len_or_err={info}", flush=True)

if __name__ == "__main__":
    n = int(sys.argv[1])
    plen = int(sys.argv[2])
    olen = int(sys.argv[3])
    print(f"sending {n} concurrent reqs, prompt_len={plen}, output_len={olen}", flush=True)
    asyncio.run(main(n, plen, olen))
