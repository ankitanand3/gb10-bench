"""Turn raw results into a table you can read and a post you can publish.

    python3 report.py                  # print to screen
    python3 report.py --out RESULTS.md # write the file
"""
import argparse
import glob
import json
import os
import platform
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def load_all(results_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(p) as f:
                rows.append(json.load(f))
        except json.JSONDecodeError:
            print(f"  (skipping unreadable {os.path.basename(p)})")
    return rows


def machine_facts():
    facts = {"host": platform.node(), "arch": platform.machine(),
             "kernel": platform.release()}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    facts["mem_total_gib"] = round(kb / 1048576, 1)
                    facts["mem_total_gb_decimal"] = round(kb * 1024 / 1e9, 1)
                    break
    except OSError:
        pass
    for cmd, key in ((["nvidia-smi", "--query-gpu=name,driver_version",
                       "--format=csv,noheader"], "gpu"),
                     (["git", "-C", os.path.expanduser("~/llama.cpp"),
                       "rev-parse", "--short", "HEAD"], "llama_cpp_commit")):
        try:
            facts[key] = subprocess.run(cmd, capture_output=True, text=True,
                                        timeout=15).stdout.strip()
        except Exception:
            facts[key] = "unknown"
    return facts


def summarise(rows):
    """One line per quant: biggest context that worked, speed there, where it died."""
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    out = []
    for label, rs in by_label.items():
        rs.sort(key=lambda x: x["ctx"])
        ok = [r for r in rs if r["status"] == "ok"]
        failed = [r for r in rs if r["status"] in ("oom", "timeout")]
        best = ok[-1] if ok else None
        out.append({
            "label": label,
            "model_gib": rs[0]["model_gib"],
            "model_gb": rs[0]["model_gb_decimal"],
            "loads": bool(ok),
            "max_ctx_ok": best["ctx"] if best else None,
            "decode_tps": (best or {}).get("decode", {}).get("tokens_per_second"),
            "prefill_tps": (best or {}).get("prefill", {}).get("tokens_per_second"),
            "peak_gib": best["peak_memory_used_gib"] if best else None,
            "died_at_ctx": failed[0]["ctx"] if failed else None,
            "died_how": failed[0]["status"] if failed else None,
        })
    out.sort(key=lambda x: x["model_gib"])
    return out


def render(rows, facts):
    s = summarise(rows)
    L = ["# Results — Unsloth UD quants on one GB10", ""]
    L.append(f"Machine: {facts.get('gpu','?')} · {facts.get('mem_total_gib','?')} GiB "
             f"({facts.get('mem_total_gb_decimal','?')} GB) unified memory · "
             f"{facts.get('arch','?')} · kernel {facts.get('kernel','?')}")
    L.append(f"llama.cpp commit `{facts.get('llama_cpp_commit','?')}` · "
             f"{len(rows)} test runs")
    L.append("")
    L.append("Memory is read from `/proc/meminfo`. `nvidia-smi` reports N/A for memory "
             "on GB10 because the CPU and GPU share one pool.")
    L.append("")
    L.append("| Quant | Size on disk | Loads | Best context | Decode | Prefill | Peak mem | Died at |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in s:
        L.append(
            f"| {r['label']} | {r['model_gib']} GiB ({r['model_gb']} GB) | "
            f"{'yes' if r['loads'] else 'no'} | "
            f"{r['max_ctx_ok'] or '—'} | "
            f"{str(r['decode_tps'])+' tok/s' if r['decode_tps'] else '—'} | "
            f"{str(r['prefill_tps'])+' tok/s' if r['prefill_tps'] else '—'} | "
            f"{str(r['peak_gib'])+' GiB' if r['peak_gib'] else '—'} | "
            f"{(str(r['died_at_ctx'])+' ('+r['died_how']+')') if r['died_at_ctx'] else '—'} |")
    L.append("")
    L.append("Same box, same batch size, same prompts. Only the quant and the context changed.")
    L.append("")

    L.append("## Draft post")
    L.append("")
    L.append("```")
    L.append("Unsloth listed 5 shrunk versions of DeepSeek V4 Flash by size and quality.")
    L.append("They did not list speed, and did not say which ones still run once the")
    L.append("conversation starts taking memory.")
    L.append("")
    L.append(f"I ran them on one GB10 box, {facts.get('mem_total_gib','?')} GiB usable.")
    L.append("")
    for r in s:
        if r["loads"]:
            L.append(f"{r['label']:<12} {r['model_gb']:>6} GB   {r['decode_tps']} tok/s, "
                     f"dies at {r['died_at_ctx'] or '>tested'} ctx")
        else:
            L.append(f"{r['label']:<12} {r['model_gb']:>6} GB   does not load")
    L.append("")
    L.append("Same box, same settings, same prompts. Only the version changed.")
    L.append("")
    L.append("Full method and raw logs: <repo link>")
    L.append("```")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = load_all(args.results_dir)
    if not rows:
        print(f"No results in {args.results_dir}. Run sweep.py first.")
        return
    text = render(rows, machine_facts())
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
