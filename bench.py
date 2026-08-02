"""Run ONE benchmark configuration and write ONE result file, then exit.

This runs as its own process on purpose. The whole point of the experiment is to
push a model until it runs out of memory, so this process is expected to die
sometimes. sweep.py runs it as a subprocess and records the death as a result.

Memory is read from /proc/meminfo, NOT nvidia-smi. On GB10 the CPU and GPU share
one pool of memory, so nvidia-smi reports N/A for memory.total and gives you
nothing useful.

    python3 bench.py --model /path/model-00001-of-00003.gguf \
                     --ctx 4096 --label UD-IQ3_S --out results/x.json
"""
import argparse
import json
import os
import re
import subprocess
import threading
import time

MEMINFO = "/proc/meminfo"


def meminfo_kb(key):
    with open(MEMINFO) as f:
        for line in f:
            if line.startswith(key):
                return int(re.search(r"(\d+)", line).group(1))
    return 0


def gib(kb):
    return round(kb / 1048576, 2)


class MemorySampler(threading.Thread):
    """Track the lowest MemAvailable seen while the benchmark runs."""

    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_flag = threading.Event()
        self.baseline_kb = meminfo_kb("MemAvailable")
        self.lowest_kb = self.baseline_kb

    def run(self):
        while not self.stop_flag.is_set():
            cur = meminfo_kb("MemAvailable")
            if cur < self.lowest_kb:
                self.lowest_kb = cur
            time.sleep(self.interval)

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=2)

    @property
    def peak_used_gib(self):
        return gib(self.baseline_kb - self.lowest_kb)


def was_oom_killed(returncode, stderr):
    """SIGKILL (-9) plus a kernel OOM line is the reliable signal."""
    if returncode == -9 or returncode == 137:
        return True
    markers = ("out of memory", "failed to allocate", "cudaErrorMemoryAllocation",
               "ggml_backend_alloc", "std::bad_alloc", "CUDA error: out of memory")
    return any(m.lower() in (stderr or "").lower() for m in markers)


def dmesg_oom_tail():
    try:
        out = subprocess.run(["dmesg", "--since", "-2min"], capture_output=True,
                             text=True, timeout=10).stdout
        hits = [l for l in out.splitlines() if "Killed process" in l or "Out of memory" in l]
        return hits[-2:]
    except Exception:
        return []


def parse_llama_bench_json(stdout):
    """llama-bench -o json prints a JSON array of runs."""
    try:
        start = stdout.index("[")
        rows = json.loads(stdout[start:])
    except (ValueError, json.JSONDecodeError):
        return {}
    out = {}
    for r in rows:
        kind = r.get("n_prompt", 0) and "prefill" or "decode"
        out[kind] = {
            "tokens_per_second": round(r.get("avg_ts", 0), 2),
            "stddev_ts": round(r.get("stddev_ts", 0), 2),
            "samples": r.get("samples_ns") and len(r["samples_ns"]) or None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to first GGUF shard")
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--label", required=True, help="e.g. UD-IQ3_S")
    ap.add_argument("--out", required=True)
    ap.add_argument("--llama-bench", default=os.path.expanduser("~/llama.cpp/build/bin/llama-bench"))
    ap.add_argument("--n-prompt", type=int, default=512)
    ap.add_argument("--n-gen", type=int, default=128)
    ap.add_argument("--ngl", type=int, default=999, help="layers on GPU")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--load-mode", default="mmap",
                    choices=["mmap", "mlock", "mmap+mlock", "none", "dio"],
                    help="mmap (default) lets a model bigger than RAM still run by paging "
                         "from disk, slowly. mlock forces it resident and fails if it does "
                         "not fit. Recorded in the result either way.")
    args = ap.parse_args()

    size_bytes = 0
    d = os.path.dirname(args.model)
    base = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", os.path.basename(args.model))
    for f in os.listdir(d or "."):
        if f.startswith(base) and f.endswith(".gguf"):
            size_bytes += os.path.getsize(os.path.join(d, f))

    result = {
        "label": args.label,
        "ctx": args.ctx,
        "model_path": args.model,
        "model_bytes": size_bytes,
        "model_gib": gib(size_bytes // 1024),
        "model_gb_decimal": round(size_bytes / 1e9, 1),
        "n_prompt": args.n_prompt,
        "n_gen": args.n_gen,
        "ngl": args.ngl,
        "load_mode": args.load_mode,
        # llama-bench has no -c. Context comes from -d/--n-depth, which is how much
        # KV cache gets allocated. Effective context is depth + prompt + generation.
        "n_depth": args.ctx,
        "effective_ctx": args.ctx + args.n_prompt + args.n_gen,
        "mem_total_gib": gib(meminfo_kb("MemTotal")),
        "mem_available_before_gib": gib(meminfo_kb("MemAvailable")),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    cmd = [args.llama_bench, "-m", args.model, "-d", str(args.ctx),
           "-p", str(args.n_prompt), "-n", str(args.n_gen),
           "-ngl", str(args.ngl), "-lm", args.load_mode, "-r", "3", "-o", "json"]
    result["command"] = " ".join(cmd)

    sampler = MemorySampler()
    sampler.start()
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
        timed_out = False
    except subprocess.TimeoutExpired as e:
        rc, out, err, timed_out = -1, e.stdout or "", e.stderr or "", True
    finally:
        sampler.stop()

    result["wall_seconds"] = round(time.time() - t0, 1)
    result["peak_memory_used_gib"] = sampler.peak_used_gib
    result["mem_available_low_gib"] = gib(sampler.lowest_kb)
    result["returncode"] = rc
    result["timed_out"] = timed_out

    oom = was_oom_killed(rc, err)
    result["oom"] = oom
    result["loaded"] = (rc == 0 and not timed_out)
    result["status"] = ("ok" if result["loaded"]
                        else "oom" if oom
                        else "timeout" if timed_out else "error")
    if oom:
        result["dmesg"] = dmesg_oom_tail()
    if rc == 0:
        result.update(parse_llama_bench_json(out))
    result["stderr_tail"] = (err or "")[-1500:]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"{args.label} ctx={args.ctx} -> {result['status']} "
          f"peak={result['peak_memory_used_gib']}GiB "
          f"decode={result.get('decode', {}).get('tokens_per_second', '-')} tok/s")
    return 0 if result["loaded"] else 0  # never fail the sweep


if __name__ == "__main__":
    raise SystemExit(main())
