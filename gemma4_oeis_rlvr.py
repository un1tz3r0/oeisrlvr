# -*- coding: utf-8 -*-
"""Gemma 4 (E2B) RLVR: learn to write a(n) programs for OEIS sequences.

Adapted from Unsloth's "Gemma4 (E2B) Reinforcement Learning Sudoku" notebook.
The Unsloth/LoRA/GRPO scaffold is unchanged; the Sudoku environment is replaced
by the OEIS program-synthesis environment in `oeis_rlvr.py`:

  * The model is shown a sequence's name + first terms and must emit `def a(n)`.
  * Rewards run the function in a sandbox and score it on held-out later terms,
    so hardcoding the shown terms does not pay off.

Run on a GPU (Colab T4 works). The reward functions and dataset are CPU-only and
can be exercised without a GPU via `test_oeis_rlvr.py`.
"""

# ---------------------------------------------------------------------------
# Installation (Colab) -- same as the source notebook. Gemma 4 needs
# transformers >= 5.5.0; do NOT pin to 4.x.
# ---------------------------------------------------------------------------
# %%capture
# import os, importlib.util
# !pip install --upgrade -qqq uv
# if importlib.util.find_spec("torch") is None or "COLAB_" in "".join(os.environ.keys()):
#     try: import numpy, PIL; _numpy = f"numpy=={numpy.__version__}"; _pil = f"pillow=={PIL.__version__}"
#     except: _numpy = "numpy"; _pil = "pillow"
#     !uv pip install -qqq \
#         "torch>=2.8.0" "triton>=3.4.0" {_numpy} {_pil} torchvision bitsandbytes \
#         "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
#         "unsloth[base] @ git+https://github.com/unslothai/unsloth" \
#         git+https://github.com/triton-lang/triton.git@0add68262ab0a2e33b84524346cb27cbb2787356#subdirectory=python/triton_kernels
# elif importlib.util.find_spec("unsloth") is None:
#     !uv pip install -qqq unsloth
# !uv pip install --upgrade --no-deps "transformers>=5.5.0" "tokenizers>=0.22.0,<=0.23.0" "trl>=0.28.0" unsloth unsloth_zoo

# bitsandbytes (used by optim="adamw_8bit") needs CUDA 13's libnvJitLink, which
# isn't on the loader path under some local installs. Preload it with global
# symbol visibility so bnb's dlopen resolves. No-op where unavailable (e.g. Colab).
try:
    import ctypes, os
    import nvidia.cu13
    # cu13 is a namespace package, so __file__ is None; use __path__ instead.
    ctypes.CDLL(
        os.path.join(list(nvidia.cu13.__path__)[0], "lib", "libnvJitLink.so.13"),
        mode=ctypes.RTLD_GLOBAL,
    )
except Exception:
    pass

# ---------------------------------------------------------------------------
# CLI configuration. Parsed before the heavy imports so --help is instant and
# an orchestrating TUI can validate args without loading the model. All values
# were previously hardcoded constants; defaults reproduce the original run.
# ---------------------------------------------------------------------------
import argparse, glob as _glob, re as _re, os as _os


def _next_stem(warmstart: str) -> str:
    """Next free incrementally-numbered adapter dir off `warmstart`'s base stem.

    'gemma_4_oeis_lora' or 'gemma_4_oeis_lora_03' both have base
    'gemma_4_oeis_lora'; returns 'gemma_4_oeis_lora_NN', NN one past the highest
    existing numbered sibling (01 if none yet).
    """
    base = _re.sub(r"_\d+$", "", warmstart.rstrip("/"))
    nums = [int(m.group(1)) for d in _glob.glob(f"{base}_*")
            if (m := _re.fullmatch(rf"{_re.escape(base)}_(\d+)", d))]
    return f"{base}_{(max(nums) + 1) if nums else 1:02d}"


_ap = argparse.ArgumentParser(description="GRPO-train a LoRA adapter on OEIS sequences.")
_ap.add_argument("--warmstart", default="gemma_4_oeis_lora",
                 help="adapter dir to warm-start from (skipped if missing)")
_ap.add_argument("--out", default=None,
                 help="adapter dir to save to (default: next numbered stem off --warmstart)")
_ap.add_argument("--max-steps", type=int, default=250)
_ap.add_argument("--save-steps", type=int, default=50)
_ap.add_argument("--n-show", type=int, default=10, help="terms shown to the model")
_ap.add_argument("--min-terms", type=int, default=20, help="min terms to qualify (for holdout)")
_ap.add_argument("--max-eval-terms", type=int, default=40, help="cap terms used per sequence")
_ap.add_argument("--limit", type=int, default=5000, help="training pool size")
_ap.add_argument("--curriculum", default=None,
                 help="outcomes JSON to read always-solved exclusions from "
                      "(default: --warmstart's outcomes file, else legacy outcomes.json)")
_ap.add_argument("--trace", default=None, help="JSONL trace path (default: trace_<out>.jsonl)")
_ap.add_argument("--outcomes", default=None,
                 help="outcomes JSON to write (default: outcomes_<out>.json)")
_ap.add_argument("--output-dir", default=None, help="checkpoint dir (default: outputs_<out>)")
# Model / LoRA / optimizer params (previously hardcoded; driven by model.toml).
_ap.add_argument("--base-model", default="unsloth/gemma-4-E2B-it")
_ap.add_argument("--max-seq-length", type=int, default=4096)
_ap.add_argument("--lora-rank", type=int, default=32)
_ap.add_argument("--lora-alpha", type=int, default=None, help="default: 2*lora_rank")
_ap.add_argument("--learning-rate", type=float, default=5e-5)
_ap.add_argument("--num-generations", type=int, default=4)
_ap.add_argument("--max-grad-norm", type=float, default=0.1)
args = _ap.parse_args()

if args.lora_alpha is None:
    args.lora_alpha = 2 * args.lora_rank

if args.out is None:
    args.out = _next_stem(args.warmstart)
_stem = _os.path.basename(args.out.rstrip("/"))
if args.trace is None:
    args.trace = f"trace_{_stem}.jsonl"
if args.outcomes is None:
    args.outcomes = f"outcomes_{_stem}.json"
if args.output_dir is None:
    args.output_dir = f"outputs_{_stem}"
if args.curriculum is None:
    _ws_outcomes = f"outcomes_{_os.path.basename(args.warmstart.rstrip('/'))}.json"
    args.curriculum = _ws_outcomes if _os.path.exists(_ws_outcomes) else "outcomes.json"
print(f"Config: warmstart={args.warmstart} out={args.out} max_steps={args.max_steps} "
      f"curriculum={args.curriculum} trace={args.trace} outcomes={args.outcomes}")

from unsloth import FastVisionModel
import torch

max_seq_length = args.max_seq_length   # room for reasoning + the function
lora_rank = args.lora_rank

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = args.base_model,
    max_seq_length = max_seq_length,
    load_in_4bit = False,    # False for LoRA 16bit
    fast_inference = False,  # vLLM unsupported for Gemma4 vision (siglip+gemma4)
)

model = FastVisionModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = args.lora_alpha,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# Warm-start: if the chosen adapter exists, load its weights over the
# freshly-initialised LoRA so training continues from where it left off.
if _os.path.exists(f"{args.warmstart}/adapter_model.safetensors"):
    try:
        from peft import set_peft_model_state_dict as _set_peft_sd
        from safetensors.torch import load_file as _load_sf
        _set_peft_sd(model, _load_sf(f"{args.warmstart}/adapter_model.safetensors"))
        print(f"Warm-started LoRA from {args.warmstart}/")
    except Exception as _e:
        print(f"Warm-start failed ({_e}), starting with fresh LoRA.")

# ---------------------------------------------------------------------------
# OEIS environment, dataset and rewards (see oeis_rlvr.py).
# ---------------------------------------------------------------------------
from oeis_rlvr import build_dataset, REWARD_FUNCS

# Filter to easy + nonn sequences (non-dead) with enough terms for a holdout.
# Curriculum: if a previous run left outcomes.json, drop the sequences it always
# solved (zero-variance groups give no GRPO signal). No-op on the first run.
import json, os
exclude_anums = frozenset()
if os.path.exists(args.curriculum):
    with open(args.curriculum) as f:
        exclude_anums = frozenset(json.load(f)["always_solved"])
    print(f"Curriculum: excluding {len(exclude_anums)} always-solved sequences ({args.curriculum}).")

tasks = build_dataset(
    n_show = args.n_show,                 # terms shown to the model
    min_terms = args.min_terms,           # so there are >= (min_terms - n_show) holdout terms
    max_eval_terms = args.max_eval_terms, # cap eval cost on fast-growing sequences
    limit = args.limit,                   # size of the training pool
    exclude_anums = exclude_anums,
)
print(f"Built {len(tasks)} OEIS tasks.")

from datasets import Dataset

# shown/holdout are JSON-encoded: OEIS terms can be arbitrary-precision integers
# that overflow Arrow's int64. _tasks_from_kwargs decodes them in the rewards.
dataset = Dataset.from_list([
    {
        "prompt": [{"role": "user", "content": t.prompt_text()}],
        "anum": t.anum,
        "name": t.name,
        "offset": t.offset,
        "shown": json.dumps(t.shown),
        "holdout": json.dumps(t.holdout),
    }
    for t in tasks
])

# Size the completion budget from the longest prompt in the pool.
maximum_length = max(
    len(tokenizer.apply_chat_template(row["prompt"], add_generation_prompt = True))
    for row in dataset
)
print(f"Maximum prompt length: {maximum_length}")
max_completion_length = max_seq_length - (maximum_length + 1)

# ---------------------------------------------------------------------------
# Baseline: prompt the model once before training.
# ---------------------------------------------------------------------------
text = tokenizer.apply_chat_template(
    dataset[0]["prompt"], tokenize = False, add_generation_prompt = True,
)
from transformers import TextStreamer
print("=" * 50, "\nBASE MODEL OUTPUT (before RL training):\n", "=" * 50)
inputs = tokenizer(text = text, add_special_tokens = False, return_tensors = "pt").to("cuda")
model.generate(**inputs, streamer = TextStreamer(tokenizer, skip_prompt = True),
               max_new_tokens = 256, use_cache = True,
               temperature = 1.0, top_p = 0.95, top_k = 64)

# ---------------------------------------------------------------------------
# Train with GRPO. Hyperparameters mirror the source notebook.
# ---------------------------------------------------------------------------
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    temperature = 1.0,
    learning_rate = args.learning_rate,
    weight_decay = 0.001,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    # One prompt-group per optimizer step: accumulate over exactly the prompt's
    # num_generations completions. Also keeps batch (1*grad_accum) divisible by
    # num_generations, which GRPO requires.
    gradient_accumulation_steps = args.num_generations,
    num_generations = args.num_generations,
    max_completion_length = max_completion_length,
    max_grad_norm = args.max_grad_norm,
    max_steps = args.max_steps,
    save_steps = args.save_steps,
    report_to = "none",
    output_dir = args.output_dir,
    epsilon = 0.2,
    epsilon_high = 0.28,
    delta = 1.5,
    loss_type = "bnpo",
    mask_truncated_completions = True,
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = REWARD_FUNCS,   # function_works, no_cheating, sequence_matches
    args = training_args,
    train_dataset = dataset,
)
import oeis_rlvr as _oeis_rlvr_mod
_oeis_rlvr_mod.open_trace(args.trace)
trainer.train()
_oeis_rlvr_mod.close_trace()

# Record which sampled sequences were always-solved (drop them next run via
# build_dataset(exclude_anums=...) -- zero-variance groups give no GRPO signal)
# or never matched a single holdout term (too hard).
from oeis_rlvr import export_outcomes
_summary = export_outcomes(args.outcomes)
print(f"Outcomes for {_summary['seen']} sampled sequences -> {args.outcomes}: "
      f"{len(_summary['always_solved'])} always-solved (omit next run), "
      f"{len(_summary['never_matched'])} never-matched.")

# ---------------------------------------------------------------------------
# Save the trained LoRA.
# ---------------------------------------------------------------------------
model.save_pretrained(args.out)
tokenizer.save_pretrained(args.out)
print(f"Saved adapter -> {args.out}/")

# Inference with the trained adapter:
text = tokenizer.apply_chat_template(
    dataset[0]["prompt"], tokenize = False, add_generation_prompt = True,
)
model.generate(
    **tokenizer(images = None, text = text, return_tensors = "pt").to("cuda"),
    temperature = 1.0, max_new_tokens = 512,
    streamer = TextStreamer(tokenizer, skip_prompt = False),
)
