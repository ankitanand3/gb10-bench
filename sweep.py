"""Run every benchmark configuration, one subprocess at a time.

Each test runs as its own process. When a model runs out of memory the child
dies, this keeps going, and the crash is recorded as a result instead of ending
the run. That matters here: finding where a model stops fitting IS the
experiment.

Safe to stop and restart. Finished tests are skipped.

    python3 sweep.py --config configs/deepseek-v4-flash.json
    python3 sweep.py --config configs/deepseek-v4-flash.json --only UD-Q2_K_XL
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    with open(path) as f:
        return json.load(f)


def result_path(results_dir, label, ctx):
    return os.path.join(results_dir, f"{label}__ctx{ctx}.json")


def already_done(results_dir, label, ctx):
    p = result_path(results_dir, label, ctx)
    if not os.path.exists(p):
        return None
    try:
        return load(p)
    except json.JSONDecodeError:
        return None


def find_first_shard(model_dir, label):
    hits = sorted(glob.glob(os.path.join(model_dir, "**", "*.gguf"), recursive=True))
    hits = [h for h in hits if label in h]
    first = [h for h in hits if "-00001-of-" in h] or hits
    return first[0] if first else None


def download(repo, label, model_dir, hf_bin):
    """Pull one quant folder. HF stores these in Xet now, so plain curl gets a stub."""
    target = os.path.join(model_dir, label)
    if find_first_shard(target, label):
        print(f"  [{label}] already downloaded")
        return target
    os.makedirs(target, exist_ok=True)
    cmd = [hf_bin, "download", repo, "--include", f"{label}/*",
           "--local-dir", model_dir]
    print(f"  [{label}] downloading: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        print(f"  [{label}] DOWNLOAD FAILED rc={proc.returncode}")
        return None
    print(f"  [{label}] downloaded in {round(time.time()-t0)}s")
    return target


def dir_size_gib(path):
    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(path) for f in fs)
    return round(total / 1048576 ** 1 / 1024, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models-dir", default=os.path.expanduser("~/models"))
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--hf", default=os.path.expanduser("~/gb10-venv/bin/hf"))
    ap.add_argument("--only", help="run just this one quant label")
    ap.add_argument("--keep", action="store_true",
                    help="keep model files after testing (default: delete)")
    args = ap.parse_args()

    cfg = load(args.config)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    quants = [q for q in cfg["quants"]
              if not args.only or q["label"] == args.only]

    print(f"repo: {cfg['repo']}")
    print(f"{len(quants)} quant(s), context steps: {cfg['context_steps']}\n")

    for q in quants:
        label = q["label"]
        print(f"=== {label} (listed {q.get('listed_size', '?')}) ===")

        steps = cfg["context_steps"]
        pending = [c for c in steps if not already_done(args.results_dir, label, c)]
        if not pending:
            print(f"  all context steps already done, skipping\n")
            continue

        model_dir = download(cfg["repo"], label, args.models_dir, args.hf)
        if not model_dir:
            continue
        shard = find_first_shard(model_dir, label)
        if not shard:
            print(f"  [{label}] no .gguf found after download, skipping\n")
            continue
        print(f"  on disk: {dir_size_gib(model_dir)} GiB")

        stop_after_oom = False
        for ctx in steps:
            done = already_done(args.results_dir, label, ctx)
            if done:
                print(f"  ctx={ctx}: cached ({done['status']})")
                if done["status"] in ("oom", "timeout"):
                    stop_after_oom = True
                    break
                continue
            if stop_after_oom:
                break
            out = result_path(args.results_dir, label, ctx)
            cmd = [sys.executable, os.path.join(HERE, "bench.py"),
                   "--model", shard, "--ctx", str(ctx),
                   "--label", label, "--out", out]
            subprocess.run(cmd)
            r = already_done(args.results_dir, label, ctx)
            # Once it stops fitting, larger context will not fit either.
            if r and r["status"] in ("oom", "timeout"):
                print(f"  stopped at ctx={ctx} ({r['status']}) — larger will not fit")
                stop_after_oom = True

        if not args.keep:
            print(f"  removing {model_dir} to free disk")
            shutil.rmtree(model_dir, ignore_errors=True)
        print()

    print("sweep complete. build the table with:  python3 report.py")


if __name__ == "__main__":
    main()
