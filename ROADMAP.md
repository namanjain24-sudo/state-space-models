# Mamba / Selective SSM — Team Implementation Roadmap

Two-person team, both new-ish to this. Goal is **not** "get a working repo fast" — goal is
**both people deeply understand every piece**, using existing GitHub implementations only as a
reference/sanity-check, never as copy-paste source.

---

## 0. How we're splitting work

We're **not** splitting by "person A does half the paper, person B does the other half forever" —
that means only one of you actually learns the SSM math. Instead:

- **Part 1 — Core Engine**: the actual SSM/Mamba math (discretization, selective scan, block).
- **Part 2 — Pipeline**: dataset preprocessing, training loop, evaluation, baselines.

We run this in **two rounds** and **swap** who owns which part, so both of you own both parts by
the end:

| | Part 1 (Core Engine) | Part 2 (Pipeline) |
|---|---|---|
| **Round 1** (Audio dataset) | Person A | Person B |
| **Round 2** (Text dataset, stretch) | Person B | Person A |

**Hard rule:** at the end of every phase, the owner explains their code to the other person
line-by-line before moving on (15-20 min). If you can't explain it, you don't understand it yet —
go back. This is not optional, this is the actual learning mechanism here.

Fill in names here: Person A = ________, Person B = ________.

---

## 1. Prerequisites (do this before writing any code)

Don't skip this — Mamba will feel like magic without it.

- [ ] Understand a classical (continuous) state space model: `x'(t) = Ax(t) + Bu(t)`,
      `y(t) = Cx(t) + Du(t)`. This is control-theory 101, not deep learning — get comfortable
      with it independent of neural nets first.
- [ ] Understand **discretization** (zero-order hold): why/how a continuous SSM becomes a
      discrete recurrence `h_t = Ā h_{t-1} + B̄ x_t` usable step-by-step like an RNN.
- [ ] Understand why this recurrence can *also* be computed as a **convolution** (this is the S4
      insight) — and why that matters for training speed.
- [ ] Read the **Mamba paper's core idea in one sentence**: make A, B, C, and the step size
      Δ *depend on the input* (instead of being fixed/static like in S4) — this is "selectivity,"
      the whole point of the paper. Everything else is engineering around this idea.
- [ ] Skim (don't deep-dive yet) one existing minimal implementation for vocabulary, e.g.
      `alxndrTL/mamba.py` or `johnma2006/mamba-minimal`. You're allowed to *read*, not copy.

**Both people do this step together**, discuss it out loud, don't split it.

---

## 2. Dataset — chosen on purpose, not the usual tutorial dataset

Every "Mamba from scratch" tutorial on GitHub uses **tinyshakespeare** or a one-line
`load_dataset(...)` call. That's fine for checking your code runs, but it does zero work for you
and doesn't show *why* Mamba/SSMs matter (long sequences, no attention quadratic cost).

We use two datasets, matching the two rounds above.

### Round 1 dataset — Speech Commands (raw audio, classification)
- What: Google Speech Commands dataset — 1-second raw audio clips of spoken words
  ("zero"–"nine", "yes", "no", etc., ~16kHz).
- Why this one: this is the *actual* benchmark the S4 paper (Mamba's predecessor) became famous
  for — classifying **raw waveforms directly**, no spectrogram/MFCC feature engineering, which is
  exactly what a CNN/RNN struggles with and an SSM is built for (long raw sequences, ~8000-16000
  steps per clip).
- Why it's real work: it does **not** come as a ready PyTorch `Dataset` — you have to download the
  raw `.wav` files (`torchaudio.datasets.SPEECHCOMMANDS` gives you files, not tensors), handle
  variable lengths, resample/pad/truncate, normalize amplitude, and build the label mapping
  yourself.
- Scope it down: use the **SC09 subset** (digit words only, 10 classes) — small enough to train on
  a laptop/free Colab GPU in reasonable time.
- Link: https://www.tensorflow.org/datasets/catalog/speech_commands (data itself is framework
  agnostic, use `torchaudio.datasets.SPEECHCOMMANDS` to fetch it).

### Round 2 dataset (stretch) — self-assembled long-text corpus (language modeling)
- What: **you build this yourselves** — pick 6-10 out-of-copyright books/texts from Project
  Gutenberg (pick a theme/author you actually like, not a random default) and concatenate them
  into one raw text file.
- Why: forces you to write your own byte/char-level tokenizer and long-context chunking (train on
  sequences of length 1000-4000+ chars) instead of using a pre-tokenized HF dataset — and long
  context is exactly where Mamba is supposed to beat a small Transformer baseline, so it makes the
  comparison experiment meaningful.
- This dataset choice is a discussion for both of you once you reach Round 2 — pick something
  neither of you has used in a tutorial before.

---

## 3. Environment setup (both, together)

- [ ] Repo has a `part1_core/` and `part2_pipeline/` folder (or similar), plus a shared
      `README.md` explaining the split above.
- [ ] `requirements.txt`: `torch`, `torchaudio`, `numpy`, `matplotlib`, `tqdm`. No `mamba-ssm`
      pip package, no `causal-conv1d` — we're not using the official CUDA kernels, we want plain
      PyTorch so the math stays visible.
- [ ] Confirm both of you can run a trivial PyTorch script (CPU is fine for now).

---

## Round 1

### Phase 1 — Naive linear SSM layer *(Person A — Part 1)*
- [ ] Implement fixed (non-selective) discrete SSM recurrence as a plain Python for-loop over
      time steps: `h_t = A h_{t-1} + B x_t`, `y_t = C h_t`. No batching tricks yet — clarity over
      speed.
- [ ] Implement zero-order-hold discretization: turn continuous `(A, B)` + step size `Δ` into
      discrete `(Ā, B̄)`.
- [ ] Unit test: feed a simple known input (e.g. an impulse), check output matches hand-computed
      values for a tiny 2-state SSM.
- [ ] Explain to Person B: what each of A, B, C represents; why ZOH discretization; why fixed A
      limits what a static SSM layer "remembers."

### Phase 2 — Data pipeline v1: audio loading *(Person B — Part 2)*
- [ ] Download Speech Commands (SC09 subset) via `torchaudio`.
- [ ] Write a custom `Dataset` class: load `.wav`, resample if needed, pad/truncate to fixed
      length, normalize amplitude to roughly [-1, 1].
- [ ] Build label vocabulary (10 digit classes), train/val/test split **by speaker ID** (not
      random) to avoid leakage — check the dataset's speaker metadata for this.
- [ ] Sanity-check: plot a few raw waveforms + listen to a couple with playback, confirm labels
      are correct.
- [ ] Explain to Person A: why speaker-based split matters, what padding/normalization choices
      you made and why.

### Phase 3 — Selective SSM (S6), the actual Mamba idea *(Person A — Part 1)*
- [ ] Make Δ, B, C **functions of the input** `x_t` (small linear projections), instead of fixed
      parameters. This is the one-sentence idea from Phase 1 prerequisites, now in code.
- [ ] Implement the selective scan as a sequential loop first (correctness before speed).
- [ ] Compare against the non-selective layer from Phase 1 on a toy synthetic task (e.g. a
      "copying task": remember a token seen early in the sequence) — selective version should
      clearly do better. This comparison IS your proof you implemented selectivity correctly.
- [ ] Explain to Person B: what changed vs Phase 1, why making B/C/Δ input-dependent lets the
      model "choose" what to remember/forget per-token.

### Phase 4 — Full Mamba block *(Person A — Part 1, B reviews)*
- [ ] Assemble the block: input projection → causal 1D conv → SiLU activation → selective SSM →
      gating (multiply with a second SiLU-activated branch) → output projection → residual
      connection. (Check the paper's Figure 3 / block diagram for the exact wiring.)
- [ ] Stack 2-4 blocks + an embedding-equivalent front end (for audio: a small linear/conv
      projection from raw waveform to model dimension) + a classification head (mean-pool over
      time + linear layer to 10 classes).
- [ ] Forward pass runs end-to-end on a dummy batch, shapes check out.

### Phase 5 — Training + evaluation *(Person B — Part 2, A reviews)*
- [ ] Training loop: cross-entropy loss, Adam/AdamW, basic LR schedule, gradient clipping.
- [ ] Logging: loss curve + val accuracy per epoch (matplotlib plot, saved to disk).
- [ ] Baseline to compare against: a small 1D-CNN or GRU on the same raw waveform, same
      train/val split. This is what proves your Mamba implementation is actually doing something
      right, not just "code runs, loss goes down."
- [ ] Milestone: Mamba model beats or matches the baseline on val accuracy. If it doesn't, debug
      before moving on — don't just accept a broken run.

**Round 1 done when:** both of you can independently explain the full pipeline start to finish,
and you have a trained model + baseline comparison plot.

---

## Round 2 (stretch) — roles swapped

### Phase 6 — Text corpus pipeline *(Person A — Part 2 now)*
- [ ] Pick and download 6-10 Project Gutenberg texts, concatenate into one raw text file.
- [ ] Build a byte-level or char-level tokenizer (simple — a fixed vocab of bytes/chars, no
      need for BPE).
- [ ] Chunk into long training sequences (1000-4000+ chars), build train/val split by document
      (not randomly across the concatenated blob, to avoid trivial leakage).

### Phase 7 — Extend the core engine *(Person B — Part 1 now)*
- [ ] Take Person A's (Round 1) sequential selective scan and implement a **parallelized**
      version (chunked/blocked scan, or PyTorch's `torch.cumsum`-based associative scan trick) —
      motivated by: the sequential for-loop from Phase 3 will be painfully slow on 2000+ length
      text sequences.
- [ ] Add a language-modeling head (linear layer to vocab size) and causal masking behavior
      (should be automatic given the recurrence, but verify — no future leakage).
- [ ] Add the "O(1) recurrent generation" mode: generate text one token at a time using the
      carried hidden state, instead of recomputing the full sequence each step. Compare
      generation speed vs a Transformer baseline at long context lengths — this is the actual
      headline efficiency claim of the paper, verify it yourselves.

### Phase 8 — Train + long-context comparison *(joint)*
- [ ] Train the Mamba LM and a small Transformer baseline (same param count, roughly) on the same
      corpus.
- [ ] Compare: validation loss/perplexity, AND generation latency as context length grows (plot
      time-per-token vs context length for both models) — Mamba should stay flat, Transformer
      should grow.
- [ ] Both of you write a short shared summary (1 page) of what you found, referencing your own
      plots — not what the paper claims, what *your* numbers show.

---

## 4. Reference implementations (read only, after you're stuck — not before)

- `alxndrTL/mamba.py` — clean PyTorch reference with a real parallel scan.
- `johnma2006/mamba-minimal` — minimal educational implementation.
- Official `state-spaces/mamba` — has the real CUDA kernels, useful to see the "production"
  version once your naive version works, not before.

Rule: if you look at these before finishing your own attempt at a phase, you're allowed — but
then immediately close the tab and rewrite from memory. Don't have it open while typing.
