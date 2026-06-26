# OEIS RLVR: Reinforcement Learning for Integer Sequence Reasoning

**OEIS RLVR** trains large language models to synthesize closed-form functions from integer sequences, using **reinforcement learning with verifiable rewards** (RLVR) over a synthetic dataset based on information from the [On-Line Encyclopedia of Integer Sequences](https://oeis.org/).

## Vision

Integer sequences encode diverse mathematical relationships — combinatorial structures, analytic patterns, number-theoretic properties, and algorithmic processes. A model trained to *reverse-engineer* these relationships (i.e., infer the closed-form rule `a(n)` from observed terms) develops stronger:

- **Logical reasoning** — rules must be internally consistent across all terms.
- **Pattern recognition** — the model learns to abstract from limited examples.
- **Mathematical intuition** — it encounters deep mathematical structures at scale.

These abilities transfer to other reasoning tasks requiring rational thought and problem-solving via neural superposition... the patterns in the OEIS reinforce similar patterns elsewhere in the model's understanding of the language and through it, reality.

At least, that's the idea.

## Overview

The pipeline:

1. **Dataset synthesis** (`oeis_rlvr.py:build_dataset`): Filter OEIS sequences by keywords, split each into shown terms (prompt input) and held-out terms (reward signal).
2. **Training** (`gemma4_oeis_rlvr.py`): Fine-tune Gemma-4-E2B-it with LoRA via GRPO, optimizing three rewards:
   - `function_works`: does the emitted `def a(n)` parse and compile?
   - `no_cheating`: does it avoid imports (stays in sandboxed builtins)?
   - `sequence_matches`: do the function's outputs match held-out terms?
3. **Evaluation** (`eval_oeis.py`): Score base and trained models on held-out sequences (greedy and/or sampled at temperature 1.0).
4. **Curriculum** (`outcomes.json`): After training, identify always-solved sequences (zero GRPO signal) and omit them from the next run.

## Setup

### Prerequisites

- **GPU**: NVIDIA RTX 3090 Ti (or similar; 24 GB VRAM). The pipeline uses `unsloth` for memory-efficient LoRA training.
- **Python 3.12+** with `uv` package manager.
- **OEIS data**: Clone the OEIS sequences (see below).

### Clone and Install

```bash
# Clone this repo
git clone https://github.com/un1tz3r0/oeisrlvr
cd oeisrlvr

# Create venv and install deps
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Get OEIS Data

The OEIS sequences are too large to include in the repo. Clone them separately:

```bash
# Clone OEIS internal format (.seq files, ~250 MB, ~350K sequences)
# This takes a few minutes.
git clone --depth=1 https://github.com/OEIS/oeisdata.git

# The pipeline expects them at ./oeisdata/seq/A???/A??????.seq
```

After cloning, the directory structure should look like:
```
oeisrlvr/
  oeisdata/
    seq/
      A000/
        A000001.seq
        A000002.seq
        ...
```

### Preview the Dataset

Before training, preview how sequences are filtered and split:

```bash
# Build a small dataset (100 sequences) and inspect tasks
uv run python -c "
from oeis_rlvr import build_dataset

tasks = build_dataset(limit=100)
print(f'Built {len(tasks)} tasks.')
t = tasks[0]
print(f'\nExample: {t.anum} ({t.name})')
print(f'  Offset: {t.offset}')
print(f'  Shown terms (input): {t.shown[:5]}...')
print(f'  Holdout terms (eval): {t.holdout[:5]}...')
print(f'\nPrompt:\n{t.prompt_text()[:300]}...')
"
```

Output (example):
```
Built 100 tasks.

Example: A000045 (Fibonacci numbers)
  Offset: 0
  Shown terms (input): [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
  Holdout terms (eval): [55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765]

Prompt:
Write a Python function a(n) that returns the n-th term of this integer sequence.

Name: Fibonacci numbers
The sequence is indexed starting at n=0. Known terms:
a(0)=0, a(1)=1, a(2)=1, a(3)=2, a(4)=3, a(5)=5, a(6)=8, a(7)=13, a(8)=21, a(9)=34
...
```

### Run Training

```bash
# Train a LoRA adapter on 5000 OEIS sequences, 60 GRPO steps
# Logs to run7_final.log (or similar), saves adapter to gemma_4_oeis_lora/
uv run python gemma4_oeis_rlvr.py

# Output:
# - Built 5000 OEIS tasks.
# - Baseline generation (before training)
# - 60-step GRPO loop with per-step loss, reward, grad norm
# - Saves gemma_4_oeis_lora/adapter_model.safetensors
# - Writes outcomes.json (always-solved sequences for curriculum next run)
```

Typical run: **~3.5 hours on RTX 3090 Ti**.

### Run Evaluation

#### Greedy Mode (Argmax Decoding)

Score base and trained models on held-out sequences with greedy decoding:

```bash
# Compare base vs trained on 40 unseen sequences, 3072-token cap
uv run python eval_oeis.py -n 40 -k 1 -t 3072

# Output:
# HELD-OUT COMPARISON (40 unseen sequences, greedy, max_new_tokens=3072)
# BASE     mean_holdout_match=0.573  fully_solved=21/40  produced_valid_fn=33/40
# TRAINED  mean_holdout_match=0.474  fully_solved=15/40  produced_valid_fn=40/40
# Δ mean_holdout_match (trained - base) = -0.099
# Δ fully_solved                        = -6
```

#### Sampled Mode (Temperature 1.0, Pass@K)

Score models with multiple samples per task (fairer test of GRPO's sampled objective):

```bash
# 24 tasks, 4 samples each, temp=1.0
uv run python eval_oeis.py -n 24 -k 4 -t 3072

# Output:
# HELD-OUT COMPARISON (24 unseen sequences, k=4 temp=1.0, max_new_tokens=3072)
# BASE     mean_holdout_match=0.563  pass@4=22/24  produced_valid_fn=...
# TRAINED  mean_holdout_match=0.587  pass@4=23/24  produced_valid_fn=...
# Δ mean_holdout_match (trained - base) = +0.024
# Δ pass@4                        = +1
```

### Curriculum Training (Next Run)

After the first training run, a file `outcomes.json` lists sequences the model always solved (zero GRPO gradient):

```bash
# outcomes.json
{
  "seen": 60,                    # 60 sequences sampled during training
  "always_solved": ["A000045", "A000290", ...],  # 15 sequences, skip next run
  "never_matched": ["A020xxx", ...] # 8 sequences, too hard for now
}
```

On the next run, the script automatically **excludes `always_solved` sequences**, concentrating GRPO signal on harder, more educational tasks:

```bash
# gemma4_oeis_rlvr.py automatically reads outcomes.json and excludes always_solved
uv run python gemma4_oeis_rlvr.py
# Curriculum: excluding 15 always-solved sequences (outcomes.json).
# Built 4985 OEIS tasks.  # (5000 - 15)
```

## Project Structure

```
oeisrlvr/
  ├── gemma4_oeis_rlvr.py       # Training script (GRPO, LoRA, 60 steps)
  ├── eval_oeis.py              # Evaluation: base vs trained, greedy & sampled
  ├── oeis_rlvr.py              # Core env: task build, sandbox, rewards
  ├── oeis_parser.py            # Parse OEIS .seq internal format
  ├── test_oeis_rlvr.py         # Reward separation tests
  ├── test_parser.py            # Parser tests
  ├── oeisdata/                 # OEIS sequences (clone separately)
  ├── gemma_4_oeis_lora/        # Saved LoRA adapter (after training)
  ├── outcomes.json             # Curriculum signals (after training)
  └── README.md                 # This file
```

## Key Design Decisions

- **GRPO over SFT**: Reinforcement learning with multiple reward signals teaches the model to reason, not memorize.
- **Held-out evaluation**: Same sequences are scored on base and trained to isolate model improvement from difficulty variance.
- **Curriculum**: Omitting always-solved sequences concentrates gradient on educational tasks.
- **Sampled evaluation**: Pass@k at temperature 1.0 measures what GRPO was trained for (sampled behavior), not argmax mode.
- **Greedy token cap (3072)**: Matches the training regime (`max_seq_length=4096 − prompt_tokens`).

## Extending This Work

### TUI for Training Workflows

See `TODO_TUI_APP.md` for a planned interactive terminal UI to manage train → eval → curriculum cycles.

### Reward Customization

Edit `REWARD_FUNCS` in `oeis_rlvr.py` to add new signals (e.g., penalizing slow algorithms, rewarding brevity).

### Model Swap

Change `"unsloth/gemma-4-E2B-it"` in `gemma4_oeis_rlvr.py` to other Unsloth-supported models (e.g., `"unsloth/Llama-3.2-8B-it"`).

### Dataset Filtering

Adjust `build_dataset` parameters:
- `require_keywords`: Filter by keyword ("easy", "nonn", etc.)
- `min_terms`: Minimum sequence length
- `max_eval_terms`: Cap evaluation cost on fast-growing sequences
- `limit`: Pool size (5000 default)

## References

- [OEIS](https://oeis.org/)
- [Unsloth](https://github.com/unslothai/unsloth) — Fast LoRA training
- [TRL](https://github.com/huggingface/trl) — GRPO implementation
- [Gemma](https://ai.google.dev/gemma/) — Base model

## License

MIT

## Author

Victor C (un1tz3r0@gmail.com)

---

**Status**: Early research. Models are learning to emit valid functions but correctness on unseen sequences is a work in progress. Curriculum and longer training expected to improve generalization.
