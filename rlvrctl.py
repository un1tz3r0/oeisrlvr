#!/usr/bin/env python3
"""Orchestrator for the OEIS RLVR train/eval workflow.

Owns the on-disk layout under ``training/<model-name>/`` and drives the
path-based scripts ``gemma4_oeis_rlvr.py`` / ``eval_oeis.py``. The Textual TUI
(later) is a front-end over this CLI; everything here is usable by hand too.

Layout::

    training/<model>/
        model.toml          model + LoRA params, [train_defaults], [eval_defaults]
        training.toml       persistent train overrides + [ephemeral] one-shot
        eval.toml           persistent eval overrides  + [ephemeral] one-shot
        001/ 002/ ...        one numbered dir per stage (a train OR an eval run)
            config.toml      effective config + [status] this stage ran with
            parent           symlink to the warm-start adapter (train only)
            adapter/         saved LoRA (train only)
            checkpoints/     intermediate checkpoints (train only)
            trace.jsonl      per-sample reward trace (train only)
            eval.json        results (eval only)
            run.log          stdout+stderr
            run.pid          supervisor pid; removed when the run exits

Commands::

    rlvrctl.py init   --model NAME [param overrides]
    rlvrctl.py train  --model NAME [-o key=val ...] [--ephemeral key=val ...]
    rlvrctl.py eval   --model NAME [--adapter-a SPEC] [--adapter-b SPEC] [-o ...]
    rlvrctl.py status --model NAME
    rlvrctl.py migrate --model NAME      (one-shot legacy import)
    rlvrctl.py _exec  ...                (internal detached supervisor)

An adapter SPEC is ``base``, a 3-digit cycle number, or a path.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from datetime import datetime

import tomlkit

REPO = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(REPO, "training")
PRESETS_DIR = os.path.join(REPO, ".oeis_rlvr_configs")

# Parameters that come from the model (not per-run). Passed as flags by name.
MODEL_PARAMS = ("base_model", "max_seq_length", "lora_rank", "lora_alpha")
TRAIN_PARAMS = ("learning_rate", "num_generations", "max_grad_norm", "max_steps",
                "save_steps", "n_show", "min_terms", "max_eval_terms", "limit")
EVAL_PARAMS = ("n", "k", "max_new_tokens")

DEFAULTS = {
    "base_model": "unsloth/gemma-4-E2B-it",
    "max_seq_length": 4096,
    "lora_rank": 32,
    "lora_alpha": 64,
    "train_defaults": {
        "learning_rate": 5e-5, "num_generations": 4, "max_grad_norm": 0.1,
        "max_steps": 250, "save_steps": 50,
        "n_show": 10, "min_terms": 20, "max_eval_terms": 40, "limit": 5000,
    },
    "eval_defaults": {"n": 40, "k": 1, "max_new_tokens": 3072},
}


# --- small toml/path helpers ------------------------------------------------

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_toml(path: str):
    if not os.path.exists(path):
        return tomlkit.document()
    with open(path) as f:
        return tomlkit.parse(f.read())


def dump_toml(path: str, doc) -> None:
    with open(path, "w") as f:
        f.write(tomlkit.dumps(doc))


def model_dir(name: str) -> str:
    return os.path.join(ROOT, name)


def require_model(name: str) -> str:
    d = model_dir(name)
    if not os.path.exists(os.path.join(d, "model.toml")):
        sys.exit(f"No model '{name}' under {ROOT}/ (run: rlvrctl.py init --model {name})")
    return d


def cycles(d: str) -> list[tuple[int, str]]:
    """(number, path) for every NNN cycle dir, ascending."""
    out = []
    for entry in os.listdir(d):
        if entry.isdigit() and os.path.isdir(os.path.join(d, entry)):
            out.append((int(entry), os.path.join(d, entry)))
    return sorted(out)


def next_cycle(d: str) -> str:
    nums = [n for n, _ in cycles(d)]
    return os.path.join(d, f"{(max(nums) + 1) if nums else 1:03d}")


def latest_train_adapter(d: str) -> str | None:
    """Path to the most recent train cycle's adapter/ dir, or None."""
    for _, path in reversed(cycles(d)):
        cfg = load_toml(os.path.join(path, "config.toml"))
        if cfg.get("kind") == "train" and os.path.isdir(os.path.join(path, "adapter")):
            return os.path.join(path, "adapter")
    return None


# --- config merge -----------------------------------------------------------

def effective(model_doc, overrides_doc, defaults_key: str, params: tuple) -> dict:
    """Merge model[defaults_key] <- overrides top-level <- overrides[ephemeral]."""
    eff = dict(model_doc.get(defaults_key, {}))
    eph = dict(overrides_doc.get("ephemeral", {}))
    for k, v in overrides_doc.items():
        if k != "ephemeral":
            eff[k] = v
    eff.update(eph)
    return {k: eff[k] for k in params if k in eff}


def model_params(model_doc) -> dict:
    return {k: model_doc[k] for k in MODEL_PARAMS if k in model_doc}


def clear_ephemeral(path: str) -> None:
    doc = load_toml(path)
    if "ephemeral" in doc:
        doc["ephemeral"] = tomlkit.table()
        dump_toml(path, doc)


# --- cycle creation + argv --------------------------------------------------

def write_config(cycle_dir: str, kind: str, model: str, params: dict,
                 warmstart: str | None = None, **extra) -> None:
    doc = tomlkit.document()
    doc["kind"] = kind
    doc["model"] = model
    doc["created"] = _now()
    if warmstart is not None:
        doc["warmstart"] = warmstart
    for k, v in extra.items():
        doc[k] = v
    p = tomlkit.table()
    for k, v in params.items():
        p[k] = v
    doc["params"] = p
    st = tomlkit.table()
    st["state"] = "pending"
    doc["status"] = st
    dump_toml(os.path.join(cycle_dir, "config.toml"), doc)


def set_status(cycle_dir: str, **fields) -> None:
    cfgpath = os.path.join(cycle_dir, "config.toml")
    doc = load_toml(cfgpath)
    st = doc.get("status", tomlkit.table())
    for k, v in fields.items():
        st[k] = v
    doc["status"] = st
    dump_toml(cfgpath, doc)


def resolve_adapter(model_d: str, spec: str) -> str:
    """'base' -> 'base'; NNN -> that cycle's adapter; else a path as given."""
    if spec == "base":
        return "base"
    if spec.isdigit():
        p = os.path.join(model_d, f"{int(spec):03d}", "adapter")
        if not os.path.isdir(p):
            sys.exit(f"cycle {spec} has no adapter/ ({p})")
        return p
    return spec


def train_argv(cycle_dir: str, model_p: dict, train_c: dict, warmstart: str | None,
               curriculum: str | None) -> list[str]:
    argv = [sys.executable, os.path.join(REPO, "gemma4_oeis_rlvr.py"),
            "--out", os.path.join(cycle_dir, "adapter"),
            "--output-dir", os.path.join(cycle_dir, "checkpoints"),
            "--trace", os.path.join(cycle_dir, "trace.jsonl"),
            "--outcomes", os.path.join(cycle_dir, "outcomes.json"),
            "--warmstart", warmstart if warmstart else "__base__"]
    if curriculum:
        argv += ["--curriculum", curriculum]
    for k, v in {**model_p, **train_c}.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return argv


def eval_argv(cycle_dir: str, eval_c: dict, specs: list[str]) -> list[str]:
    return [sys.executable, os.path.join(REPO, "eval_oeis.py"),
            "-n", str(eval_c.get("n", 40)),
            "-k", str(eval_c.get("k", 1)),
            "-t", str(eval_c.get("max_new_tokens", 3072)),
            "--adapters", *specs,
            "--jsonl", os.path.join(cycle_dir, "eval.json")]


# --- launching --------------------------------------------------------------

def spawn(cycle_dir: str, kind: str, overrides_file: str | None) -> int:
    """Start the detached supervisor (_exec) for an already-prepared cycle dir."""
    logf = open(os.path.join(cycle_dir, "run.log"), "w")
    cmd = [sys.executable, os.path.abspath(__file__), "_exec",
           "--cycle-dir", cycle_dir, "--kind", kind]
    if overrides_file:
        cmd += ["--overrides-file", overrides_file]
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            cwd=REPO, start_new_session=True)
    logf.close()
    return proc.pid


def start_train(model: str) -> tuple[str, int]:
    """Create the next train cycle (warm-started from the latest) and launch it.

    Reads the model's persisted training.toml; the TUI/CLI mutate that first.
    """
    d = require_model(model)
    model_doc = load_toml(os.path.join(d, "model.toml"))
    training = load_toml(os.path.join(d, "training.toml"))
    train_c = effective(model_doc, training, "train_defaults", TRAIN_PARAMS)
    model_p = model_params(model_doc)
    warm = latest_train_adapter(d)

    cycle = next_cycle(d)
    os.makedirs(cycle)
    if warm:
        os.symlink(os.path.relpath(warm, cycle), os.path.join(cycle, "parent"))
    write_config(cycle, "train", model, {**model_p, **train_c},
                 warmstart=os.path.relpath(warm, cycle) if warm else "base")
    pid = spawn(cycle, "train", os.path.join(d, "training.toml"))
    return cycle, pid


def start_eval(model: str, specs: list[str] | None = None) -> tuple[str, int]:
    """Create the next eval cycle comparing `specs` and launch it.

    specs are 'base' | cycle-number | path; default base vs latest-train.
    """
    d = require_model(model)
    model_doc = load_toml(os.path.join(d, "model.toml"))
    evald = load_toml(os.path.join(d, "eval.toml"))
    eval_c = effective(model_doc, evald, "eval_defaults", EVAL_PARAMS)
    if not specs:
        latest = latest_train_adapter(d)
        specs = ["base", os.path.relpath(latest, d) if latest else "base"]
    resolved = [resolve_adapter(d, s) for s in specs]

    cycle = next_cycle(d)
    os.makedirs(cycle)
    write_config(cycle, "eval", model, eval_c, adapters=resolved)
    pid = spawn(cycle, "eval", os.path.join(d, "eval.toml"))
    return cycle, pid


def cmd_train(a) -> None:
    d = require_model(a.model)
    training = load_toml(os.path.join(d, "training.toml"))
    apply_cli_overrides(training, a.override, a.ephemeral)
    dump_toml(os.path.join(d, "training.toml"), training)
    cycle, pid = start_train(a.model)
    print(f"train cycle {os.path.basename(cycle)} started (pid {pid}) -> {cycle}")


def cmd_eval(a) -> None:
    d = require_model(a.model)
    evald = load_toml(os.path.join(d, "eval.toml"))
    apply_cli_overrides(evald, a.override, a.ephemeral)
    dump_toml(os.path.join(d, "eval.toml"), evald)
    specs = a.adapters or [s for s in (a.adapter_a, a.adapter_b) if s] or None
    cycle, pid = start_eval(a.model, specs)
    print(f"eval cycle {os.path.basename(cycle)} started (pid {pid}) -> {cycle}")


def apply_cli_overrides(doc, overrides: list[str], ephemeral: list[str]) -> None:
    for kv in overrides or []:
        k, v = kv.split("=", 1)
        doc[k.strip()] = _coerce(v.strip())
    if ephemeral:
        eph = doc.get("ephemeral", tomlkit.table())
        for kv in ephemeral:
            k, v = kv.split("=", 1)
            eph[k.strip()] = _coerce(v.strip())
        doc["ephemeral"] = eph


def _coerce(s: str):
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


# --- detached supervisor ----------------------------------------------------

def cmd_exec(a) -> None:
    cycle = a.cycle_dir
    pidfile = os.path.join(cycle, "run.pid")
    with open(pidfile, "w") as f:
        f.write(str(os.getpid()))

    cfg = load_toml(os.path.join(cycle, "config.toml"))
    params = dict(cfg.get("params", {}))
    if a.kind == "train":
        model_p = {k: params[k] for k in MODEL_PARAMS if k in params}
        train_c = {k: params[k] for k in TRAIN_PARAMS if k in params}
        warm = os.path.join(cycle, "parent")
        warm = warm if os.path.islink(warm) else None
        curriculum = None
        if warm:
            cand = os.path.normpath(os.path.join(os.path.realpath(warm), "..", "outcomes.json"))
            curriculum = cand if os.path.exists(cand) else None
        argv = train_argv(cycle, model_p, train_c, warm, curriculum)
    else:
        specs = list(cfg.get("adapters") or [cfg.get("adapter_a"), cfg.get("adapter_b")])
        argv = eval_argv(cycle, params, specs)

    child = {"proc": None}

    def _term(_sig, _frm):
        if child["proc"] and child["proc"].poll() is None:
            child["proc"].terminate()
    signal.signal(signal.SIGTERM, _term)

    set_status(cycle, state="running", pid=os.getpid(), started=_now())
    rc = 1
    try:
        child["proc"] = subprocess.Popen(argv, cwd=REPO)
        rc = child["proc"].wait()
    finally:
        state = "done" if rc == 0 else ("canceled" if rc < 0 else "failed")
        set_status(cycle, state=state, exit_code=rc, finished=_now())
        if rc == 0 and a.overrides_file:
            clear_ephemeral(a.overrides_file)
        if os.path.exists(pidfile):
            os.remove(pidfile)
    sys.exit(0 if rc == 0 else 1)


# --- init / status / migrate ------------------------------------------------

def list_models() -> list[str]:
    """Names of models (dirs under training/ that have a model.toml)."""
    if not os.path.isdir(ROOT):
        return []
    return sorted(n for n in os.listdir(ROOT)
                  if os.path.exists(os.path.join(ROOT, n, "model.toml")))


def create_model(name: str, overrides: dict | None = None) -> str:
    """Create training/<name>/ with model.toml + empty training/eval overrides."""
    d = model_dir(name)
    if os.path.exists(os.path.join(d, "model.toml")):
        raise FileExistsError(f"model '{name}' already exists at {d}")
    os.makedirs(d, exist_ok=True)
    overrides = overrides or {}
    doc = tomlkit.document()
    doc["name"] = name
    for k in MODEL_PARAMS:
        v = overrides.get(k)
        doc[k] = (_coerce(v) if isinstance(v, str) else v) if v is not None else DEFAULTS[k]
    td = tomlkit.table(); td.update(DEFAULTS["train_defaults"]); doc["train_defaults"] = td
    ed = tomlkit.table(); ed.update(DEFAULTS["eval_defaults"]); doc["eval_defaults"] = ed
    dump_toml(os.path.join(d, "model.toml"), doc)
    for fn in ("training.toml", "eval.toml"):
        od = tomlkit.document(); od["ephemeral"] = tomlkit.table()
        dump_toml(os.path.join(d, fn), od)
    return d


def cycle_summary(model: str) -> list[dict]:
    """One row per cycle for status/dashboard views."""
    d = model_dir(model)
    rows = []
    for n, path in cycles(d):
        cfg = load_toml(os.path.join(path, "config.toml"))
        st = cfg.get("status", {})
        rows.append({
            "num": f"{n:03d}", "kind": cfg.get("kind", ""),
            "state": st.get("state", ""),
            "live": is_live(path),
            "created": cfg.get("created", ""), "path": path,
        })
    return rows


# --- run liveness / cancel / progress (used by the TUI monitor) -------------

def is_live(cycle_dir: str) -> bool:
    pf = os.path.join(cycle_dir, "run.pid")
    if not os.path.exists(pf):
        return False
    try:
        pid = int(open(pf).read().strip())
    except (ValueError, OSError):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cancel(cycle_dir: str) -> bool:
    """SIGTERM the whole supervisor process group for a live cycle."""
    pf = os.path.join(cycle_dir, "run.pid")
    if not os.path.exists(pf):
        return False
    try:
        pid = int(open(pf).read().strip())
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ValueError, OSError, ProcessLookupError):
        return False
    return True


def find_live_cycles(model: str) -> list[dict]:
    return [r for r in cycle_summary(model) if r["live"]]


def cycle_status(cycle_dir: str) -> dict:
    cfg = load_toml(os.path.join(cycle_dir, "config.toml"))
    st = dict(cfg.get("status", {}))
    st["kind"] = cfg.get("kind", "")
    st["live"] = is_live(cycle_dir)
    return st


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # partial last line during a live write
    return out


def _fnum(rec: dict, key: str) -> float:
    try:
        return float(rec.get(key))
    except (TypeError, ValueError):
        return 0.0


def train_progress(cycle_dir: str) -> dict:
    """Live training stats parsed from this cycle's trace.jsonl + config."""
    cfg = load_toml(os.path.join(cycle_dir, "config.toml"))
    total = int(cfg.get("params", {}).get("max_steps", 0) or 0)
    recs = _read_jsonl(os.path.join(cycle_dir, "trace.jsonl"))
    if not recs:
        return {"step": 0, "total": total, "samples": 0, "solved": 0, "valid": 0,
                "reward_mean": 0.0, "reward_recent": 0.0, "last_anum": "", "last_name": ""}
    step = max(int(r.get("step", 0) or 0) for r in recs)
    cur = [r for r in recs if int(r.get("step", 0) or 0) == step]
    n = len(recs)
    last = recs[-1]
    return {
        "step": step, "total": total, "samples": n,
        "solved": sum(1 for r in recs if "SOLVED" in str(r.get("sm_reason", ""))),
        "valid": sum(1 for r in recs if _fnum(r, "fw_score") > 0),
        "reward_mean": sum(_fnum(r, "reward_total") for r in recs) / n,
        "reward_recent": (sum(_fnum(r, "reward_total") for r in cur) / len(cur)) if cur else 0.0,
        "last_anum": last.get("anum", ""), "last_name": last.get("name", ""),
    }


def eval_progress(cycle_dir: str) -> dict:
    """Live eval stats: final eval.json if present, else parsed from run.log."""
    ej = os.path.join(cycle_dir, "eval.json")
    if os.path.exists(ej):
        try:
            return {"done": True, **json.load(open(ej))}
        except (json.JSONDecodeError, OSError):
            pass
    conds: list[dict] = []
    cur = None
    log = os.path.join(cycle_dir, "run.log")
    if os.path.exists(log):
        for line in open(log):
            if line.startswith("Scoring "):
                cur = {"label": line.split("Scoring ", 1)[1].split(" (")[0].strip(),
                       "scores": [], "passk": 0, "total": None}
                conds.append(cur)
            m = re.match(r"\s*\[(\d+)/(\d+)\]\s+\S+\s+mean_holdout=([0-9.]+)(\s+PASS)?", line)
            if m and cur is not None:
                cur["total"] = int(m.group(2))
                cur["scores"].append(float(m.group(3)))
                if m.group(4):
                    cur["passk"] += 1
    return {"done": False, "conditions": conds}


# --- config editing (used by the TUI Configure screen) ----------------------

def read_overrides(model: str, which: str) -> tuple[dict, dict]:
    """(persistent, ephemeral) override dicts from training.toml / eval.toml."""
    doc = load_toml(os.path.join(model_dir(model), f"{which}.toml"))
    persistent = {k: v for k, v in doc.items() if k != "ephemeral"}
    return persistent, dict(doc.get("ephemeral", {}))


def write_overrides(model: str, which: str, persistent: dict, ephemeral: dict) -> None:
    doc = tomlkit.document()
    for k, v in persistent.items():
        doc[k] = v
    et = tomlkit.table(); et.update(ephemeral); doc["ephemeral"] = et
    dump_toml(os.path.join(model_dir(model), f"{which}.toml"), doc)


def read_model_config(model: str):
    return load_toml(os.path.join(model_dir(model), "model.toml"))


def write_model_config(model: str, top: dict, train_defaults: dict,
                       eval_defaults: dict) -> None:
    """Update model.toml top-level params + the two defaults tables."""
    path = os.path.join(model_dir(model), "model.toml")
    doc = load_toml(path)
    for k, v in top.items():
        doc[k] = v
    td = tomlkit.table(); td.update(train_defaults); doc["train_defaults"] = td
    ed = tomlkit.table(); ed.update(eval_defaults); doc["eval_defaults"] = ed
    dump_toml(path, doc)


# --- presets (reusable config bundles, applyable across models) --------------

def list_presets() -> list[str]:
    if not os.path.isdir(PRESETS_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(PRESETS_DIR) if f.endswith(".toml"))


def save_preset(name: str, train: dict, eval_: dict) -> None:
    os.makedirs(PRESETS_DIR, exist_ok=True)
    doc = tomlkit.document()
    t = tomlkit.table(); t.update(train); doc["train"] = t
    e = tomlkit.table(); e.update(eval_); doc["eval"] = e
    dump_toml(os.path.join(PRESETS_DIR, f"{name}.toml"), doc)


def load_preset(name: str) -> tuple[dict, dict]:
    doc = load_toml(os.path.join(PRESETS_DIR, f"{name}.toml"))
    return dict(doc.get("train", {})), dict(doc.get("eval", {}))


def apply_preset(model: str, name: str) -> None:
    """Merge a preset's train/eval keys into the model's persistent overrides."""
    train, eval_ = load_preset(name)
    for which, data in (("training", train), ("eval", eval_)):
        persistent, ephemeral = read_overrides(model, which)
        persistent.update(data)
        write_overrides(model, which, persistent, ephemeral)


def cmd_init(a) -> None:
    overrides = {k: getattr(a, k) for k in MODEL_PARAMS if getattr(a, k, None) is not None}
    try:
        d = create_model(a.model, overrides)
    except FileExistsError as e:
        sys.exit(str(e))
    print(f"initialized model '{a.model}' at {d}")


def cmd_status(a) -> None:
    require_model(a.model)
    print(f"model: {a.model}  ({model_dir(a.model)})")
    for r in cycle_summary(a.model):
        print(f"  {r['num']}  {r['kind']:5} {r['state']:8} "
              f"{'[LIVE]' if r['live'] else '':6} {r['created']}")


def cmd_migrate(a) -> None:
    """One-shot import of the legacy flat artifacts into training/<model>/."""
    d = model_dir(a.model)
    if os.path.exists(d):
        sys.exit(f"{d} already exists; refusing to migrate over it")
    os.makedirs(d)

    # model.toml from current defaults
    doc = tomlkit.document()
    doc["name"] = a.model
    for k in MODEL_PARAMS:
        doc[k] = DEFAULTS[k]
    td = tomlkit.table(); td.update(DEFAULTS["train_defaults"]); doc["train_defaults"] = td
    ed = tomlkit.table(); ed.update(DEFAULTS["eval_defaults"]); doc["eval_defaults"] = ed
    dump_toml(os.path.join(d, "model.toml"), doc)
    for fn in ("training.toml", "eval.toml"):
        od = tomlkit.document(); od["ephemeral"] = tomlkit.table()
        dump_toml(os.path.join(d, fn), od)

    adapter_files = ("adapter_config.json", "adapter_model.safetensors",
                     "chat_template.jinja", "processor_config.json",
                     "tokenizer_config.json", "tokenizer.json", "README.md")

    def copy_adapter(src: str, dst: str) -> None:
        os.makedirs(dst, exist_ok=True)
        for fn in adapter_files:
            s = os.path.join(src, fn)
            if os.path.exists(s):
                shutil.copy2(s, os.path.join(dst, fn))

    def mk(num: int):
        c = os.path.join(d, f"{num:03d}")
        os.makedirs(c)
        return c

    p = lambda *x: os.path.join(REPO, *x)
    train_params = {**{k: DEFAULTS[k] for k in MODEL_PARAMS}, **DEFAULTS["train_defaults"]}

    # 001 train: adapter from run1's checkpoint-250, log from train.log
    c1 = mk(1)
    if os.path.isdir(p("outputs_run1", "checkpoint-250")):
        copy_adapter(p("outputs_run1", "checkpoint-250"), os.path.join(c1, "adapter"))
    if os.path.exists(p("train.log")):
        shutil.copy2(p("train.log"), os.path.join(c1, "run.log"))
    write_config(c1, "train", a.model, train_params, warmstart="base")
    set_status(c1, state="done", exit_code=0, finished=_now(), note="migrated")

    # 002 eval: pass@4 base vs 001, log from eval_run1.log
    c2 = mk(2)
    if os.path.exists(p("eval_run1.log")):
        shutil.copy2(p("eval_run1.log"), os.path.join(c2, "run.log"))
    write_config(c2, "eval", a.model, {"n": 24, "k": 4, "max_new_tokens": 3072},
                 adapter_a="base", adapter_b="../001/adapter")
    set_status(c2, state="done", exit_code=0, finished=_now(), note="migrated")

    # 003 train: current adapter, trace, outcomes, log from train_run2.log
    c3 = mk(3)
    if os.path.isdir(p("gemma_4_oeis_lora")):
        copy_adapter(p("gemma_4_oeis_lora"), os.path.join(c3, "adapter"))
    for src, dst in [("trace.jsonl", "trace.jsonl"),
                     ("outcomes.json", "outcomes.json"),
                     ("train_run2.log", "run.log")]:
        if os.path.exists(p(src)):
            shutil.copy2(p(src), os.path.join(c3, dst))
    os.symlink("../001/adapter", os.path.join(c3, "parent"))
    write_config(c3, "train", a.model, train_params, warmstart="../001/adapter")
    set_status(c3, state="done", exit_code=0, finished=_now(), note="migrated")

    print(f"migrated 3 cycles into {d}")
    print("Legacy originals left in place. After verifying, remove with:")
    print("  rm -rf outputs outputs_run1 gemma_4_oeis_lora trace.jsonl "
          "outcomes.json train.log train_run2.log eval_run1.log run_next.log")


# --- arg parsing ------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init")
    pi.add_argument("--model", required=True)
    for k in MODEL_PARAMS:
        pi.add_argument(f"--{k.replace('_', '-')}", default=None)
    pi.set_defaults(func=cmd_init)

    for name, fn in (("train", cmd_train), ("eval", cmd_eval)):
        ps = sub.add_parser(name)
        ps.add_argument("--model", required=True)
        ps.add_argument("-o", "--override", action="append",
                        help="persistent override key=val (repeatable)")
        ps.add_argument("--ephemeral", action="append",
                        help="one-shot override key=val, cleared after run (repeatable)")
        if name == "eval":
            ps.add_argument("--adapter-a", default=None, help="base | NNN | path")
            ps.add_argument("--adapter-b", default=None, help="base | NNN | path")
            ps.add_argument("--adapters", nargs="+", default=None,
                            help="N specs to compare (overrides -a/-b)")
        ps.set_defaults(func=fn)

    pst = sub.add_parser("status")
    pst.add_argument("--model", required=True)
    pst.set_defaults(func=cmd_status)

    pm = sub.add_parser("migrate")
    pm.add_argument("--model", required=True)
    pm.set_defaults(func=cmd_migrate)

    pe = sub.add_parser("_exec")
    pe.add_argument("--cycle-dir", required=True)
    pe.add_argument("--kind", required=True, choices=["train", "eval"])
    pe.add_argument("--overrides-file", default=None)
    pe.set_defaults(func=cmd_exec)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
