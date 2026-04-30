import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import orjson
import typer

def main(
 profile_id: Optional[str] = None,
 start_time_ms: Optional[int] = 0,
 end_time_ms: Optional[int] = 999999999,
 thread_filters: str = None,
 dir_data: str = "/Users/tom/temp/temp_sglang_server2local",
):
 dir_data = Path(dir_data)

 if profile_id is None:
  pattern = re.compile(r"(\d+\.\d+)")
  candidates = []
  for p in dir_data.glob("*.json.gz"):
   bn = p.name
   if bn.startswith("perfetto-compatible-") or bn.startswith("merged-"):
    continue
   m = pattern.search(bn)
   if m is not None:
    candidates.append(m.group(1))
  profile_id = max(candidates)
  print(f"{profile_id=}")

 interesting_paths = sorted(
  [p for p in dir_data.glob("*.json.gz")
   if profile_id in p.name
   and not p.name.startswith("perfetto-compatible-")
   and not p.name.startswith("merged-")
   and "TP-" in p.name],
  key=_get_tp_rank_of_path,
 )
 print(f"{interesting_paths=}")

 output_path = dir_data / f"merged-{profile_id}.trace.json.gz"
 _merge_chrome_traces(
  interesting_paths, output_path,
  config=Config(
   start_time_ms=start_time_ms,
   end_time_ms=end_time_ms,
   thread_filters=thread_filters,
  ),
 )

@dataclass
class Config:
 start_time_ms: int
 end_time_ms: int
 thread_filters: str

def _merge_chrome_traces(interesting_paths: List[Path], output_path: Path, config: Config):
 merged_trace = {"traceEvents": []}

 for output_raw in map(_handle_file, interesting_paths, [config] * len(interesting_paths)):
  merged_trace['traceEvents'].extend(output_raw['traceEvents'])
  for key, value in output_raw.items():
   if key != 'traceEvents' and key not in merged_trace:
    merged_trace[key] = value

 print(f"Write output to {output_path}")
 with gzip.open(output_path, 'wb') as f:
  f.write(orjson.dumps(merged_trace))

def _handle_file(path, config: Config):
 print(f"handle_file START {path=}")
 tp_rank = _get_tp_rank_of_path(path)

 with gzip.open(path, 'rt', encoding='utf-8') as f:
  trace = orjson.loads(f.read())
  print(f"handle_file {path=} {tp_rank=} {list(trace)=}")

 output = {key: value for key, value in trace.items() if key != 'traceEvents'}
 output['traceEvents'] = _process_events(trace.get('traceEvents', []), config, tp_rank)

 print(f"handle_file END {path=}")
 return output

def _process_events(events, config, tp_rank):
 print(f"{len(events)=}")

 min_ts = min(e["ts"] for e in events)
 ts_interest_start = min_ts + 1000 * config.start_time_ms
 ts_interest_end = min_ts + 1000 * config.end_time_ms
 events = [e for e in events if ts_interest_start <= e["ts"] <= ts_interest_end]
 print(f"after filtering by timestamp {len(events)=} ({ts_interest_start=} {ts_interest_end})")

 if config.thread_filters is not None:
  thread_filters_list = config.thread_filters.split(',')

  thread_name_of_tid = {
   e["tid"]: e["args"]["name"]
   for e in events
   if e["name"] == "thread_name"
  }

  def _thread_name_filter_fn(thread_id):
   ans = False
   if 'gpu' in thread_filters_list:
    ans |= "stream" in str(thread_id)
   return ans

  remove_tids = [
   tid
   for tid, thread_name in thread_name_of_tid.items()
   if not _thread_name_filter_fn(thread_name)
  ]
  print(f"{remove_tids=}")

  events = [e for e in events if e["tid"] not in remove_tids]
  print(f"after filtering by thread_filters {len(events)=}")

 for e in events:
  if e["name"] == "process_sort_index":
   pid = _maybe_cast_int(e["pid"])
   if pid is not None and pid < 1000:
    e["args"]["sort_index"] = 100 * tp_rank + int(e["pid"])
  e["pid"] = f"[TP{tp_rank:02d}] {e['pid']}"

 return events

def _maybe_cast_int(x):
 try:
  return int(x)
 except ValueError:
  return None

def _get_tp_rank_of_path(p: Path):
 return int(re.search(r"TP-(\d+)", p.name).group(1))

if __name__ == "__main__":
 typer.run(main)
