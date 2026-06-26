# TUI Application for Training & Eval Workflows

## Prompt for subagent

```
okay can we make a nice TUI application for starting and monitoring the alternating training and eval runs? i'm thinking a full-screen multi-page wizard style flow, which presents the user with a brief introductory message with a continue button that leads to a main menu for selecting from operating modes including "auto train: run or resume loop of alternating train and eval stages", "train step: train a new or existing adapter for configured steps, writing log to a new incrementally-numbered stem according to the base adapter's stem by default, saving a new adapter with the new stem when done", "eval step: run an eval comparing two adapters, or the base model and a saved adapter, of given size, and show/save a table of the results when done", and "configure: configure hyperparameters for dataset generation, training and eval run steps and max-token cap etc... settings are saved either as defaults or a named configuration preset, which can be selected from a list of presets shown (if there are any saved named presets other than the default settings) at the beginning of an auto, train or eval run". all three long running operations (auto, train and eval runs) should show detailed, animated progress with stats and collapsible log output while running, allow canceling and suspending process to return to the shell prompt that we were invoked from (fg the job to resume).
```

## Context

This is for the OEIS RLVR (program synthesis via GRPO) training pipeline. The script `gemma4_oeis_rlvr.py` trains a LoRA adapter on 5000 OEIS sequences, and `eval_oeis.py` evaluates base vs trained on held-out sequences with configurable token budgets and sampling strategies.

The goal is to streamline the workflow: users can toggle between training runs (which feed into `outcomes.json` for curriculum pruning) and evals (which measure improvement), without typing CLI commands repeatedly.

## Key files involved
- `gemma4_oeis_rlvr.py` - training script, accepts CLI args for dataset/training config
- `eval_oeis.py` - evaluation, accepts `-n` (tasks), `-k` (samples), `-t` (token cap)
- `outcomes.json` - generated after each training run, lists always-solved sequences to exclude next run
- Saved adapters: `gemma_4_oeis_lora/`, `gemma_4_oeis_lora_01/`, etc. (incrementally numbered)

## Suggested implementation notes
- Use a library like `textual` (Python TUI framework) for the full-screen wizard and progress displays
- Config presets can be JSON files in a `.oeis_rlvr_configs/` directory
- Progress/stats should pull from log files in real-time (tail + parse step metrics for training, task results for eval)
- Suspend/resume via job control (`fg`, `bg`) — the TUI should spawn subprocesses that survive backgrounding
