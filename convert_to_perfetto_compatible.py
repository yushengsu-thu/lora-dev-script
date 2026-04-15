import gzip
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import orjson
import typer


def main(
        filename: str,
        dir_data: str = "/Users/tom/temp/temp_sglang_server2local",
):
    dir_data = Path(dir_data)
    path_input = dir_data / filename
    path_output = dir_data / f"perfetto-compatible-{filename}"
    print(f"{path_input=} {path_output=}")

    with (gzip.open(path_input, 'rt', encoding='utf-8') as f):
        trace = orjson.loads(f.read())
        output = {key: value for key, value in trace.items() if key != 'traceEvents'}
        output['traceEvents'] = _process_events(trace.get('traceEvents', []))

    with gzip.open(path_output, 'wb') as f:
        f.write(orjson.dumps(output))


def _process_events(events):
    print(f"process_events start {len(events)=}")

    last_end_time_of_pid_tid = defaultdict(lambda: -1)

    for e in events:
        if e["ph"] == "X" and _is_interest_event(e):
            while e["ts"] < last_end_time_of_pid_tid[(e["pid"], e["tid"])]:
                e["tid"] = str(e["tid"]) + "_hack"
            last_end_time_of_pid_tid[(e["pid"], e["tid"])] = e["ts"] + e["dur"]

    return events


def _is_interest_event(e):
    return "registers per thread" in e.get("args", {})


if __name__ == "__main__":
    typer.run(main)
