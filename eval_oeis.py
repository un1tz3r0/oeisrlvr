# -*- coding: utf-8 -*-
"""Held-out evaluation: did GRPO training improve a(n) synthesis?

Scores the BASE Gemma-4-E2B model and the TRAINED LoRA adapter on the SAME fixed
sample of OEIS sequences the model never trained on (qualifying sequences just
beyond the 5000 in the training pool). Because both conditions see identical
sequences, the before/after difference controls for sequence difficulty -- unlike
the per-step training reward, which is confounded by which sequences got sampled.

Run AFTER training finishes: needs the GPU free and `gemma_4_oeis_lora/` saved.

Usage:
  uv run python eval_oeis.py [-n N] [-k K] [-t MAX_NEW_TOKENS]
    -n  number of held-out sequences (default 40)
    -k  samples per task (default 1 = greedy/argmax; >1 samples at temp 1.0 and
        reports pass@k -- the fairer test of what GRPO optimized)
    -t  max new tokens (default 3072, ~the training cap; 4096 ctx - prompt)

Greedy (-k 1) measures the argmax mode; GRPO optimized sampled behavior, so a gain
can hide in the distribution -- use -k 4 (with a smaller -n, e.g. 24) to surface it.
"""

# Same libnvJitLink preload as the training script (bitsandbytes / adamw_8bit).
try:
    import ctypes, os
    import nvidia.cu13
    ctypes.CDLL(
        os.path.join(list(nvidia.cu13.__path__)[0], "lib", "libnvJitLink.so.13"),
        mode=ctypes.RTLD_GLOBAL,
    )
except Exception:
    pass

import argparse
import torch
from unsloth import FastVisionModel
from peft import PeftModel

from oeis_rlvr import build_dataset, evaluate, extract_function

ap = argparse.ArgumentParser()
ap.add_argument("-n", type=int, default=40, help="held-out sequences")
ap.add_argument("-k", type=int, default=1, help="samples per task (1=greedy)")
ap.add_argument("-t", type=int, default=3072, help="max new tokens")
ap.add_argument("--adapter-a", default="base",
                help="first condition: 'base' or an adapter dir (default: base)")
ap.add_argument("--adapter-b", default="gemma_4_oeis_lora",
                help="second condition: 'base' or an adapter dir (default: gemma_4_oeis_lora)")
ap.add_argument("--adapters", nargs="+", default=None,
                help="N conditions to compare ('base'/dir each); overrides -a/-b")
ap.add_argument("--jsonl", default=None, help="write results summary as JSON to this path")
args = ap.parse_args()
N_EVAL, K, MAX_NEW_TOKENS = args.n, args.k, args.t
max_seq_length = 4096

# Held-out set: qualifying sequences just beyond the 5000 the training pool used.
# build_dataset iterates sorted paths, so the first-5000 anum set is a strict
# subset of the first-(5000+k) set; the difference is genuinely unseen.
train_anums = {t.anum for t in build_dataset(limit=5000)}
pool = build_dataset(limit=5000 + 4 * N_EVAL)
heldout = [t for t in pool if t.anum not in train_anums][:N_EVAL]
print(f"Held-out: {len(heldout)} unseen sequences | k={K} | max_new_tokens={MAX_NEW_TOKENS}")

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "unsloth/gemma-4-E2B-it",
    max_seq_length = max_seq_length,
    load_in_4bit = False,
    fast_inference = False,
)
FastVisionModel.for_inference(model)


@torch.no_grad()
def generate(task) -> list[str]:
    """Return K completions for `task` (greedy if K==1, else temp-1.0 samples)."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": task.prompt_text()}],
        tokenize = False, add_generation_prompt = True,
    )
    inputs = tokenizer(text = text, add_special_tokens = False,
                       return_tensors = "pt").to("cuda")
    kw = dict(max_new_tokens = MAX_NEW_TOKENS, use_cache = True)
    if K == 1:
        kw["do_sample"] = False
    else:
        kw.update(do_sample = True, temperature = 1.0, top_p = 0.95, top_k = 64,
                  num_return_sequences = K)
    out = model.generate(**inputs, **kw)
    plen = inputs["input_ids"].shape[1]
    return [tokenizer.decode(o[plen:], skip_special_tokens=True) for o in out]


def score(tasks) -> dict:
    """Per task: mean holdout-match over K samples; pass@k = any sample fully solves."""
    holdout_means, passk, valid_any = [], 0, 0
    for j, t in enumerate(tasks, 1):
        per_sample, any_valid, any_solved = [], False, False
        for completion in generate(t):
            fx = extract_function(completion)
            if fx is None:
                per_sample.append(0.0)
                continue
            any_valid = True
            try:
                shown_ok, hold_ok = evaluate(fx, t)
            except Exception:  # noqa: BLE001 -- compile/exec failure scores zero
                per_sample.append(0.0)
                continue
            per_sample.append(hold_ok / max(1, len(t.holdout)))
            if shown_ok == len(t.shown) and hold_ok == len(t.holdout):
                any_solved = True
        holdout_means.append(sum(per_sample) / len(per_sample))
        passk += any_solved
        valid_any += any_valid
        print(f"  [{j}/{len(tasks)}] {t.anum} mean_holdout={holdout_means[-1]:.2f}"
              f"{' PASS' if any_solved else ''}", flush=True)
    n = len(tasks)
    return dict(mean_holdout = sum(holdout_means) / n, passk = passk,
                valid = valid_any, n = n)


# Conditions to compare. 'base' = no adapter; otherwise an adapter dir.
# Attach each distinct adapter once (PEFT multi-adapter) over a single base load;
# score 'base' by disabling adapters on the wrapped model.
specs = args.adapters if args.adapters else [args.adapter_a, args.adapter_b]


def _label(spec):
    return "BASE" if spec == "base" else os.path.basename(spec.rstrip("/"))


adapter_name = {}
for spec in specs:
    if spec == "base" or spec in adapter_name:
        continue
    name = f"ad{len(adapter_name)}"
    if not isinstance(model, PeftModel):
        model = PeftModel.from_pretrained(model, spec, adapter_name=name)
    else:
        model.load_adapter(spec, adapter_name=name)
    adapter_name[spec] = name
if isinstance(model, PeftModel):
    FastVisionModel.for_inference(model)


def score_spec(spec):
    if spec == "base":
        if isinstance(model, PeftModel):
            with model.disable_adapter():
                return score(heldout)
        return score(heldout)
    model.set_adapter(adapter_name[spec])
    return score(heldout)


results = []
for spec in specs:
    print(f"Scoring {_label(spec)} ({'no adapter' if spec == 'base' else spec}) ...")
    results.append((spec, score_spec(spec)))

passk_label = f"pass@{K}" if K > 1 else "fully_solved"
n_seq = results[0][1]["n"]
print("\n" + "=" * 64)
print(f"HELD-OUT COMPARISON ({n_seq} unseen sequences, "
      f"{'greedy' if K == 1 else f'k={K} temp=1.0'}, max_new_tokens={MAX_NEW_TOKENS})")
print("=" * 64)
for spec, r in results:
    print(f"  {_label(spec):<24}  mean_holdout_match={r['mean_holdout']:.3f}  "
          f"{passk_label}={r['passk']}/{r['n']}  produced_valid_fn={r['valid']}/{r['n']}")
# Deltas of the last condition (newest adapter) vs each earlier one.
print("-" * 64)
last_spec, last_r = results[-1]
for spec, r in results[:-1]:
    print(f"  Δ {_label(last_spec)} - {_label(spec)}:  "
          f"mean_holdout={last_r['mean_holdout'] - r['mean_holdout']:+.3f}  "
          f"{passk_label}={last_r['passk'] - r['passk']:+d}  "
          f"valid_fn={last_r['valid'] - r['valid']:+d}")

if args.jsonl:
    import json
    with open(args.jsonl, "w") as f:
        json.dump({
            "n": n_seq, "k": K, "max_new_tokens": MAX_NEW_TOKENS,
            "passk_label": passk_label,
            "conditions": [
                {"spec": spec, "label": _label(spec), **r} for spec, r in results
            ],
        }, f, indent=2)
    print(f"Wrote results -> {args.jsonl}")
