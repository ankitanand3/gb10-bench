# Results — the rung that fits on neither box: 2x GB10 over 200GbE

**Machines:** two identical NVIDIA GB10 boxes (MSI EdgeXpert), 121.7 GiB (130.7 GB) unified
memory each, aarch64, linked directly over their ConnectX 200GbE ports (10.100.216.0/24
point-to-point).
**Engine:** llama.cpp commit `221f0f6` on both boxes, CUDA 13.0, compute capability 121,
built with `-DGGML_RPC=ON`.
**Command:** `llama-bench -p 512 -n 128 -ngl 999 -lm mmap -r 3 -d <context>` plus
`--rpc 10.100.216.1:50052`. Host box reads the GGUF; `ggml-rpc-server -c` runs on the other.
**Model:** DeepSeek-V4-Flash-0731 UD-Q3_K_XL — 128.2 GB decimal = **119.4 GiB** on disk,
284.3B total parameters, 13B active. Unsloth lists it at "129GB", 87% quality retained.
**Date:** 2026-08-22. Raw JSON in `results/twobox/`.

---

## Headline

**The 129GB rung of the ladder does not fit in one 128GB box. Split across two with llama.cpp
RPC it runs at 16.8 tok/s — about 5% behind the biggest quant that fits on one box. Past 32K
context it is the fastest configuration measured.**

One more thing nobody told us would happen: the RPC link came up as **RDMA (RoCEv2)** on its
own. No tuning, no config. The kernel Ethernet counters read ~0 for the whole run because RDMA
bypasses them; the InfiniBand counters carry the traffic.

## Two-box sweep, UD-Q3_K_XL (119.4 GiB)

| Context | Decode | Prefill |
|---|---|---|
| 0 | 16.70 tok/s | 343.6 tok/s |
| 4K | 16.78 | 348.8 |
| 8K | 16.67 | 350.5 |
| 16K | 16.48 | 331.8 |
| 32K | 15.75 | 293.5 |
| 64K | 15.48 | 231.3 |
| 128K | 13.63 | 164.4 |

64K and 128K ran with `-r 2`, everything else `-r 3`.

## The ladder, complete

Decode tok/s. One-box numbers are the 2026-08-02 runs, same command, same llama.cpp commit.

| Context | Q2_K_XL 97GB, 1 box | IQ3_S 117GB, 1 box | Q3_K_XL 129GB, 2 boxes |
|---|---|---|---|
| 4K | 17.05 | 17.76 | 16.78 |
| 8K | 16.78 | 15.96 | 16.67 |
| 16K | 16.57 | 15.80 | 16.48 |
| 32K | 15.92 | 15.07 | 15.75 |
| 64K | 14.97 | 14.10 | **15.48** |
| 128K | 13.14 | 12.50 | **13.63** |

At short context the split model pays ~5%. At 64K and beyond it wins, because the KV cache is
split too: each box walks half of it with its own memory bandwidth.

## What the wire actually carries

- First load: ~65 GB of layers streamed to the far box (69.5 GB on the RDMA transmit counter,
  including warmup traffic). This is a one-time cost per model load.
- Reload with the rpc-server tensor cache (`-c`): 12.6 GB.
- Decode: ~2.4 GB total across a ~26,000-token metering run, both directions — **tens of
  kilobytes per token**. The link idles during generation.

## The control: splitting a model that fits anyway

Qwen3.8-27B UD-Q4_K_XL (16.34 GiB, dense 27.3B) on one box, then split across both. Same
command, same depths.

| Depth | 1 box decode | 2 boxes decode | 1 box prefill | 2 boxes prefill |
|---|---|---|---|---|
| 0 | 12.33 | 12.41 | 837.3 | 817.9 |
| 4K | 12.14 | 12.18 | 815.9 | 806.2 |
| 16K | 11.54 | 11.48 | 752.0 | 721.6 |
| 32K | 10.91 | 10.70 | 680.9 | 634.5 |
| 64K | 9.80 | 9.52 | 578.9 | 511.5 |

Decode changes by under 3% at every depth. Prefill loses 2-12%, growing with depth. **At batch
1, a second box buys capacity, not speed.** If the model fits on one box, run it on one box.

Side table: the same model's UD-Q8_K_XL (29.3 GiB) on one box decodes at 7.40 tok/s at 4K vs
12.14 for Q4_K_XL (16.3 GiB) — 39% slower for 80% more bytes. Decode speed on this box is the
quant size; prefill barely moves (759 vs 816 tok/s).

## The one-box verdict on the 129GB rung

Same file, same command, second box removed: `llama_bench: error: failed to load model`,
exit 1. 119.4 GiB of weights do not load on a box with 121.7 GiB total and ~115 GiB usable
once the CUDA buffers are requested up front. That is the fits/doesn't-fit line the published
ladder never drew: **UD-Q3_K_XL needs two boxes. It then runs at full speed.**

## Determinism across the split

Greedy decoding (temperature 0, fixed seed, 64 tokens) on Qwen3.8-27B Q4: one box, then split
across two boxes over RPC. **Outputs byte-identical.** Splitting layers across machines does
not change results at batch 1 — the same kernels run in the same order; the boundary only
moves activations.

## KV cache per token, two architectures

- DeepSeek V4 Flash: ~0.03 GiB per 1K tokens, measured on Aug 2 via memory deltas → ~4 GiB
  at 128K next to ~110 GiB of weights.
- Qwen3.8-27B: 65 layers × 4 KV heads × 256 dims × (K+V) × f16 = 0.254 MiB per token,
  computed from the GGUF header → **~32 GiB at 128K, about twice its own weights.**

About 8x apart per token. On memory-limited boxes, long-context cost is an architecture
choice, not a constant.

## A note on dense vs MoE

The 284B model decodes faster than the 27B model on the same silicon: 16.8 vs 12.1 tok/s at
4K. DeepSeek V4 Flash activates 13B parameters per token; the Qwen model is dense, so all 27B
weights are read for every token. Total size decides what fits. Active bytes decide speed.

## Limits of this test

- Batch 1, one request at a time. Says nothing about serving several users.
- The RDMA path is whatever llama.cpp negotiated on its own (mtu=1024 in the probe line). Not
  tuned. A tuned setup may do better; a plain TCP setup may do worse. Neither was measured.
- Quality percentages are Unsloth's own. Not independently measured here.
- One-box ladder numbers come from box A on Aug 2; two-box runs hosted on box B on Aug 22. The
  boxes are the same hardware and the control run reproduced box A's Qwen numbers on box B
  within noise, but it is a cross-box, cross-date comparison.
- GB10 is memory-bandwidth-limited. Do not read these as general GPU numbers.

## Reproducing

On the far box:

```bash
ggml-rpc-server -H 10.100.216.1 -p 50052 -c
```

On the host box:

```bash
llama-bench -m DeepSeek-V4-Flash-0731-UD-Q3_K_XL-00001-of-00004.gguf \
  -ngl 999 -p 512 -n 128 -lm mmap -r 3 -d 0,4096,8192,16384,32768 \
  --rpc 10.100.216.1:50052 -o json
```

That is the whole setup. Both binaries come from the same build with `-DGGML_RPC=ON`.
