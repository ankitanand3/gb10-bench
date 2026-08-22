# gb10-bench

Measure which quantized models actually run on an NVIDIA GB10 box, how fast, and where they run
out of memory.

**Results so far:**
- [RESULTS.md](RESULTS.md) — one box: the DeepSeek V4 Flash quant ladder, 4K to 128K context.
- [RESULTS-TWOBOX.md](RESULTS-TWOBOX.md) — two boxes over 200GbE: the 129GB rung that fits on
  neither box alone, llama.cpp RPC (it negotiated RDMA on its own), plus the split-cost
  control, a determinism check, and KV-per-token across architectures.

## Why this exists

People publish model sizes. A 117 GB model on a 128 GB machine looks like it fits.

It often does not, and the reason is that a model needs memory for two things:

1. **The weights.** That is the number everyone publishes.
2. **The KV cache.** This holds the conversation. It grows as the text gets longer, and it
   cannot be moved out of memory while the model is answering.

So "fits on disk" and "runs at a useful context length" are different questions. Almost nobody
publishes the second one, and the published sizes rarely say whether they mean GB or GiB — a
gap of about 7% at these sizes, which is enough to decide whether a model loads at all.

This tool answers the second question, for one machine, with the raw logs attached so you can
disagree with it.

## What it measures

For each quantized model, at increasing context lengths:

- does it load at all
- decode speed (tokens/second)
- prefill speed (tokens/second)
- peak memory used
- the context length where it stops fitting

## Requirements

- An NVIDIA GB10 machine (DGX Spark, MSI EdgeXpert, or similar), Linux, ARM64
- CUDA toolkit (13.x tested)
- `llama.cpp` built with CUDA
- Python 3.10+
- Enough disk for the models you test — they are 90–170 GB each

## Setup

```bash
# Hugging Face download tool. Models are stored in Xet now,
# so plain curl gets a 5 MB stub instead of the weights.
python3 -m venv ~/gb10-venv
~/gb10-venv/bin/pip install -U "huggingface_hub[hf_xet]"

# llama.cpp with CUDA. 121 is the GB10 compute capability.
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp
export PATH=/usr/local/cuda/bin:$PATH
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)" --target llama-cli llama-bench llama-server
```

## Run it

```bash
# one quant first, to check everything works
python3 sweep.py --config configs/deepseek-v4-flash.json --only UD-Q2_K_XL

# then the rest
python3 sweep.py --config configs/deepseek-v4-flash.json

# build the table
python3 report.py --out RESULTS.md
```

Runs take hours and download a lot. Use `tmux` so an SSH drop does not kill it:

```bash
tmux new -s bench
python3 sweep.py --config configs/deepseek-v4-flash.json
# Ctrl-b d to detach, tmux attach -t bench to come back
```

By default each model is deleted after testing to save disk. Pass `--keep` to keep them.

## How it works, and why

**Each test runs as its own process.** The point of this tool is to push a model until it runs
out of memory. That kills the process. If everything ran in one process — or in a notebook — the
crash would take the results with it. Instead `sweep.py` runs `bench.py` as a subprocess, and
records the crash as a result.

**A crash is a data point, not a failure.** "Died at 16,384 tokens" is the answer to the
question, so it gets written down like any other measurement.

**Memory comes from `/proc/meminfo`, not `nvidia-smi`.** On GB10 the CPU and GPU share one pool
of memory. `nvidia-smi --query-gpu=memory.total` returns `N/A`. Anyone reading GPU memory the
usual way gets nothing and may not notice.

**Speed comes from `llama-bench`**, which ships with llama.cpp. It is the tool most people in
this space already use, so the numbers are comparable to what others publish. This repo does not
reimplement timing.

**Stop and restart any time.** Finished tests are skipped. Once a model fails at some context
length, larger lengths are skipped too.

## Output

Raw JSON per test in `results/`, one file per model and context length. Each file records the
exact command, the llama.cpp commit, memory before and during, the return code, and the last of
stderr. `report.py` turns those into a table.

## What this does not tell you

- One box, one model at a time, batch size 1. Nothing here is about serving many users at once.
- Speed on a GB10 is limited by memory bandwidth, not compute. Do not read these numbers as
  general GPU performance.
- Quality loss from quantization is not measured yet. Size and speed only.

## Licence

MIT.
