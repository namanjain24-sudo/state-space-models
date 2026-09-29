# State Space Models / Mamba — from scratch

Learning project by two people to implement the **Mamba (selective state space model)**
architecture from scratch in PyTorch — no `mamba-ssm` package, no CUDA kernels, plain PyTorch so
the math stays visible.

Goal is understanding, not beating benchmarks: we verify correctness with toy tasks and baseline
comparisons at small scale, not by matching paper-scale SOTA numbers.

## Structure

- `part1_core/` — the SSM/Mamba math: discretization, selective scan, Mamba block.
- `part2_pipeline/` — data preprocessing, training loop, evaluation, baselines.

## Datasets

1. **Speech Commands (SC09)** — raw audio digit classification. Real S4/Mamba benchmark, no
   ready-made PyTorch dataset — preprocessing done by hand.
2. **Self-assembled text corpus** (Project Gutenberg) — stretch goal, long-context language
   modeling vs a small Transformer baseline.

## Full plan

See [`ROADMAP.md`](./ROADMAP.md) for the phase-by-phase checklist, prerequisites, and
how work is split between the two of us.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
