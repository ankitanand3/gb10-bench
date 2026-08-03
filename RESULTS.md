# Results — DeepSeek V4 Flash 0731 on one NVIDIA GB10

**Machine:** NVIDIA GB10, driver 580.159.03, 121.7 GiB (130.7 GB) unified memory, aarch64,
kernel 6.17.0-1026-nvidia
**Engine:** llama.cpp commit `221f0f6`, CUDA 13.0, built for compute capability 121
**Command:** `llama-bench -p 512 -n 128 -ngl 999 -lm mmap -r 3 -d <context>`
**Runs:** 12 (2 quants × 6 context lengths), all successful
**Date:** 2026-08-02

Memory is read from `/proc/meminfo`. `nvidia-smi` reports `N/A` for memory on GB10 because the
CPU and GPU share one pool.

---

## Headline

**Both quants ran at every context length up to 128K. Neither ran out of memory.**

The 116 GB quant works on a 121.7 GiB machine, at full speed, with 5.4 GiB to spare at 128K.

---

## UD-Q2_K_XL — 90.18 GiB (96.8 GB decimal), listed 79% quality

| Context | Decode | Prefill | Peak memory | Free at peak |
|---|---|---|---|---|
| 4K | 17.05 tok/s | 336.65 tok/s | 90.46 GiB | 25.20 GiB |
| 8K | 16.78 | 321.68 | 92.44 GiB | 25.65 GiB |
| 16K | 16.57 | 310.19 | 92.71 GiB | 25.42 GiB |
| 32K | 15.92 | 276.50 | 92.32 GiB | 25.50 GiB |
| 64K | 14.97 | 223.22 | 93.51 GiB | 24.66 GiB |
| 128K | 13.14 | 161.69 | 94.30 GiB | 23.72 GiB |

4K → 128K: memory **+3.84 GiB**, decode **−23%**, prefill **−52%**

## UD-IQ3_S — 108.1 GiB (116.1 GB decimal), listed 83% quality

| Context | Decode | Prefill | Peak memory | Free at peak |
|---|---|---|---|---|
| 4K | 17.76 tok/s | 349.24 tok/s | 109.18 GiB | 7.93 GiB |
| 8K | 15.96 | 338.10 | 110.34 GiB | 7.90 GiB |
| 16K | 15.80 | 318.66 | 110.62 GiB | 7.66 GiB |
| 32K | 15.07 | 283.35 | 110.60 GiB | 7.53 GiB |
| 64K | 14.10 | 226.05 | 111.41 GiB | 6.82 GiB |
| 128K | 12.50 | 165.41 | 112.74 GiB | 5.38 GiB |

4K → 128K: memory **+3.56 GiB**, decode **−30%**, prefill **−53%**

---

## Three findings

### 1. The bigger quant costs about 5% speed, not more

| Context | Q2_K_XL | IQ3_S | Difference |
|---|---|---|---|
| 4K | 17.05 | 17.76 | **+4.2%** |
| 8K | 16.78 | 15.96 | −4.9% |
| 16K | 16.57 | 15.80 | −4.6% |
| 32K | 15.92 | 15.07 | −5.3% |
| 64K | 14.97 | 14.10 | −5.8% |
| 128K | 13.14 | 12.50 | −4.9% |

IQ3_S is 20 GB larger and rated 4 points higher on quality, and it runs about 5% slower. At 4K
it is *faster*.

**On a 128 GB-class box there is no throughput reason to choose Q2_K_XL over IQ3_S.**

### 2. Long context is cheap in memory and expensive in time

Going from 4K to 128K is 32× the context. It cost:

- **Memory:** +3.84 GiB (Q2_K_XL) / +3.56 GiB (IQ3_S) — roughly 0.03 GiB per 1,000 tokens
- **Decode:** −23% / −30%
- **Prefill:** −52% / −53%

The common assumption is that KV cache growth breaks a memory-limited box. On this model it is
not close. What long context actually costs is time to first token.

DeepSeek uses Multi-head Latent Attention, which compresses the KV cache. That is the likely
explanation, but this benchmark did not measure KV cache size directly, so treat it as an
inference and not a measurement.

### 3. The published sizes mean decimal GB

Unsloth lists UD-IQ3_S as "117GB". Measured on disk it is 116.1 GB decimal = 108.1 GiB. Their
numbers are decimal GB and they are accurate.

This matters. 117 **GiB** would not fit in 121.7 GiB alongside a KV cache. 117 **GB** does, with
room to spare. The unit is the difference between working and not working.

---

## Limits of this test

- One box. Nothing here is about two-box operation.
- Batch size 1, one request at a time. Says nothing about serving several users.
- `mmap` load mode, the default. A model larger than memory can appear to run by paging from
  disk. The speed numbers rule that out here, since both quants held full speed.
- Quality was not measured. Sizes and speeds only; the quality percentages are Unsloth's own.
- Speed on GB10 is limited by memory bandwidth, not compute. Do not read these as general GPU
  numbers.
- Downloads ran over WiFi and averaged 20–65 MB/s. That affects setup time, not results.

## Reproducing

```bash
python3 sweep.py --config configs/deepseek-v4-flash.json --only UD-IQ3_S
python3 report.py --out RESULTS.md
```

Raw per-run JSON is in `results/`, one file per quant and context length. Each records the exact
command, the llama.cpp commit, memory before and during, the return code, and stderr.
