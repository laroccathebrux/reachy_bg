# Model Sizing on the Development Mac

The robot's brain runs locally through Ollama on the project owner's **work** MacBook Pro. That
machine is shared with a full workday of applications, so the model has to fit in what is
left, not in the whole machine. This document records the measurements the decision was based
on and the resulting choices. Re-measure when the hardware or the workload changes.

## Hardware

| Item | Value |
|---|---|
| Machine | MacBook Pro, Apple M1 Max |
| CPU | 10 cores (8 performance + 2 efficiency) |
| GPU | 32 cores, shared unified memory |
| Unified memory | 64 GB |
| Disk | 926 GB, ~75 GB free at the time of writing |
| Ollama | 0.33.3, models under the default location |

## Snapshot of a normal workday (2026-09-14, 11:00)

Taken while the owner was working, with Docker Desktop, Teams, Chrome, WhatsApp, Claude, a
virtual machine and the corporate management agent all running.

| Metric | Value | Reading |
|---|---|---|
| Load average (1 / 5 / 15 min) | 74 / 58 / 45 | CPU heavily oversubscribed on 10 cores |
| Memory: active + wired | 33.6 GB | actually in use by processes |
| Memory: inactive + speculative | 29.7 GB | mostly file cache, reclaimable |
| Memory: compressed | 4.4 GB | pressure is already present |
| Swap | 8.5 GB used of 10 GB | the system has been paging |
| Docker VM | 3.0 GB resident | Qdrant, Postgres, Redis containers |
| Largest single apps | 0.5 - 1.0 GB each | browser, Teams, chat clients, Claude |

Conclusion: on a working day, planning for **20-25 GB of free memory for the model** is
realistic. Planning for 45 GB or more is not.

## Candidate models

| Model | Architecture | Disk (quantized) | Tokens/s on M1 Max (estimate) | Fits on a workday? |
|---|---|---|---|---|
| Qwen 2.5 72B (original plan) | dense, 72B | ~47 GB at 4-bit | 5-8 | **No**: would swap constantly and stall the owner's work apps |
| Qwen 2.5 32B | dense, 32B | ~20 GB at 4-bit | 10-14 | Barely; too slow for a spoken turn |
| **Qwen 3.6 35B-A3B** | mixture of experts, 35B total, **3B active** per token | ~22 GB (nvfp4) | 30-50 | **Yes**: already pulled as `qwen3.6:35b-mlx` |
| Qwen 2.5 3B | dense, 3B | 1.9 GB | 100+ | Yes, for unit tests and smoke runs only |

Memory bandwidth, not compute, bounds decoding speed on Apple Silicon: a dense 32B model reads
its 20 GB of weights for every token, while the 35B-A3B mixture only touches the 8 experts
routed for that token (about 2 GB), which is why it is 3-4x faster at similar quality. The
model also has vision and tool-calling capabilities and a 262k context window, all of which
the strategy layer can use later.

## Decision

- **Reasoning model: `qwen3.6:35b-mlx`** (Qwen 3.6 35B-A3B). Default of `OLLAMA_MODEL` in
  [src/config.py](../src/config.py).
- **Embeddings: `bge-m3`** (1024 dimensions), the model the Qdrant collections were built with.
  Changing it means re-embedding every collection.
- **Development / CI fallback: `qwen2.5:3b`** for tests that need a live model without
  loading 22 GB.

## First measurements (2026-09-14, machine saturated: load average ~70, 8 GB swap)

| Call | Output | Time | Throughput |
|---|---|---|---|
| First call of the session (loads the 22 GB model) | 3 sentences | 41 s | 1.4 tok/s incl. load |
| Warm call, English | 3 sentences | 12.4 s | 4.3 tok/s |
| Warm call, Portuguese | 3 sentences | 7.8 s | 7.7 tok/s |

Far below the 30-50 tok/s a 3B-active model should reach, but the machine was also
compiling, syncing and running a VM at the time. Repeat on a quiet evening before drawing
conclusions; if it stays under 10 tok/s, shorten in-game answers to one or two sentences
and keep `think=False`.

## Operating rules

1. Keep at most one large model resident: `OLLAMA_KEEP_ALIVE` short (5 minutes) during
   development so the 22 GB is released when idle; long (1 hour) during a game session.
2. Before a game session, close the heaviest work apps or accept slower turns. Check with
   `memory_pressure` and `vm_stat`; if swap is above ~6 GB, expect stalls.
3. Do not raise the Ollama context window beyond what a turn needs (8k-16k tokens); KV
   cache for the full 262k window would not fit.
4. Re-run the snapshot above when moving to a different machine and update this file.

## Sources

- Qwen 3.6 35B-A3B architecture and Ollama availability:
  [Ollama library](https://ollama.com/library/qwen3.6:35b-a3b),
  [deployment guide](https://baeseokjae.github.io/posts/qwen-3-6-local-deployment-2026/),
  [local run guide](https://insiderllm.com/guides/best-way-run-qwen-3-6-35b-moe-locally/)
