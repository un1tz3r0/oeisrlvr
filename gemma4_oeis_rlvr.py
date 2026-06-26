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

from unsloth import FastVisionModel
import torch

max_seq_length = 4096   # room for reasoning + the function
lora_rank = 32

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "unsloth/gemma-4-E2B-it",
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
    lora_alpha = lora_rank * 2,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# ---------------------------------------------------------------------------
# OEIS environment, dataset and rewards (see oeis_rlvr.py).
# ---------------------------------------------------------------------------
from oeis_rlvr import build_dataset, REWARD_FUNCS

# Filter to easy + nonn sequences (non-dead) with enough terms for a holdout.
# Curriculum: if a previous run left outcomes.json, drop the sequences it always
# solved (zero-variance groups give no GRPO signal). No-op on the first run.
import json, os
exclude_anums = frozenset()
if os.path.exists("outcomes.json"):
    with open("outcomes.json") as f:
        exclude_anums = frozenset(json.load(f)["always_solved"])
    print(f"Curriculum: excluding {len(exclude_anums)} always-solved sequences (outcomes.json).")

tasks = build_dataset(
    n_show = 10,            # terms shown to the model
    min_terms = 20,        # so there are >= 10 holdout terms
    max_eval_terms = 40,   # cap eval cost on fast-growing sequences
    limit = 5000,          # size of the training pool
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
    learning_rate = 5e-5,
    weight_decay = 0.001,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 2,
    num_generations = 2,
    max_completion_length = max_completion_length,
    max_steps = 60,
    save_steps = 100,
    report_to = "none",
    output_dir = "outputs",
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
trainer.train()

# Record which sampled sequences were always-solved (drop them next run via
# build_dataset(exclude_anums=...) -- zero-variance groups give no GRPO signal)
# or never matched a single holdout term (too hard).
from oeis_rlvr import export_outcomes
_summary = export_outcomes("outcomes.json")
print(f"Outcomes for {_summary['seen']} sampled sequences -> outcomes.json: "
      f"{len(_summary['always_solved'])} always-solved (omit next run), "
      f"{len(_summary['never_matched'])} never-matched.")

# ---------------------------------------------------------------------------
# Save the trained LoRA.
# ---------------------------------------------------------------------------
model.save_pretrained("gemma_4_oeis_lora")
tokenizer.save_pretrained("gemma_4_oeis_lora")

# Inference with the trained adapter:
text = tokenizer.apply_chat_template(
    dataset[0]["prompt"], tokenize = False, add_generation_prompt = True,
)
model.generate(
    **tokenizer(images = None, text = text, return_tensors = "pt").to("cuda"),
    temperature = 1.0, max_new_tokens = 512,
    streamer = TextStreamer(tokenizer, skip_prompt = False),
)
