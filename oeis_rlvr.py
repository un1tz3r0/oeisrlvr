"""OEIS program-synthesis RLVR environment.

The task: given a sequence's name and its first ``n_show`` terms, the model must
emit ``def a(n): ...`` (Python, no imports) that reproduces the *n*-th term. We
reward generalization: the function is scored on held-out later terms it never
saw, so a function that merely hardcodes the shown terms scores poorly.

This module has no GPU/Unsloth dependency so it can be unit-tested directly. The
training script (``gemma4_oeis_rlvr.py``) imports the reward functions from here.

Pieces:
  * ``build_dataset``        -- filter sequences and split terms into shown/holdout
  * ``compile_candidate``    -- parse + sandbox a model-written ``a(n)``
  * ``evaluate``             -- run it over the term range, return per-term matches
  * ``function_works`` / ``no_cheating`` / ``sequence_matches`` -- GRPO rewards
"""

from __future__ import annotations

import ast
import datetime as _dt
import glob as _glob
import json
import random
import signal
import threading as _threading
from dataclasses import dataclass

from oeis_parser import parse_file

# --- task construction ------------------------------------------------------

DEFAULT_GLOB = "oeisdata/seq/A???/A??????.seq"
# Keywords that make a(n)-from-terms unfair or meaningless.
EXCLUDE_KEYWORDS = {"dead", "dupe", "base"}


@dataclass
class Task:
    anum: str
    name: str
    offset: int                # n of the first term
    shown: list[int]           # terms given to the model
    holdout: list[int]         # later terms used only for scoring

    def prompt_text(self) -> str:
        pairs = ", ".join(
            f"a({self.offset + i})={t}" for i, t in enumerate(self.shown)
        )
        return PROMPT_TEMPLATE.format(name=self.name, offset=self.offset, pairs=pairs)


PROMPT_TEMPLATE = """
Write a Python function a(n) that returns the n-th term of this integer sequence.

Name: {name}
The sequence is indexed starting at n={offset}. Known terms:
{pairs}

Requirements:
- Define exactly: def a(n): ...
- All code, including any imports, must be inside a(n).
- You may import these modules: math, cmath, fractions, decimal, numbers,
  itertools, functools, operator, statistics, sympy, mpmath. numpy is allowed
  too, but its integers are fixed-width -- avoid it for large terms and prefer
  exact tools (math.comb, math.factorial, math.isqrt, math.gcd,
  fractions.Fraction, sympy). No other imports (no os, sys, etc.).
- Find the general rule; do NOT hardcode a lookup table of the terms above.

Output only the function in a single Python code block:
```python
def a(n):
    # your logic here
    return ...
```
""".strip()


def build_dataset(
    glob_pattern: str = DEFAULT_GLOB,
    n_show: int = 10,
    min_terms: int = 20,
    max_eval_terms: int = 40,
    limit: int | None = None,
    require_keywords: tuple[str, ...] = ("easy", "nonn"),
    exclude_keywords: set[str] = EXCLUDE_KEYWORDS,
    exclude_anums: frozenset[str] = frozenset(),
    seed: int = 3407,
) -> list[Task]:
    """Select sequences and split their terms into shown/holdout.

    A sequence qualifies if it has all ``require_keywords``, none of
    ``exclude_keywords``, and at least ``min_terms`` terms (so there are at least
    ``min_terms - n_show`` holdout terms). At most ``max_eval_terms`` terms total
    are used, to bound evaluation cost on fast-growing sequences.
    """
    tasks: list[Task] = []
    rk = set(require_keywords)
    for path in sorted(_glob.glob(glob_pattern)):
        seq = parse_file(path)
        if seq.anum in exclude_anums:
            continue
        kws = set(seq.keywords)
        if not rk.issubset(kws) or kws & exclude_keywords:
            continue
        if seq.name is None or seq.offset is None or len(seq.data) < min_terms:
            continue
        terms = seq.data[:max_eval_terms]
        tasks.append(
            Task(
                anum=seq.anum,
                name=seq.name,
                offset=seq.offset.first,
                shown=terms[:n_show],
                holdout=terms[n_show:],
            )
        )
        if limit and len(tasks) >= limit:
            break
    random.Random(seed).shuffle(tasks)
    return tasks


# --- sandboxed evaluation ---------------------------------------------------

# Modules a candidate may import. Curated math/scientific set only: enough to
# express real formulas without handing model-generated code os/subprocess/etc.
# Restricted builtins are not a hard security boundary, but this keeps the
# common accidental footgun -- the model reaches for `import` during RL
# exploration -- off the training host.
_ALLOWED_MODULES = frozenset({
    "math", "cmath", "fractions", "decimal", "numbers",
    "itertools", "functools", "operator", "statistics",
    "numpy", "sympy", "mpmath",
})

# Pre-load the heavy allowed modules once so a candidate's first `import sympy`
# is a cached sys.modules hit, not a multi-hundred-ms cost under the eval alarm.
for _m in ("numpy", "sympy", "mpmath"):
    try:
        __import__(_m)
    except ImportError:
        pass


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """__import__ replacement for the sandbox: only whitelisted top-level modules."""
    top = name.split(".")[0]
    if level != 0 or top not in _ALLOWED_MODULES:
        raise ImportError(f"import of {name!r} not allowed")
    return __import__(name, globals, locals, fromlist, level)


# A deliberately small allow-list of builtins the candidate may use.
_SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        "range", "len", "abs", "min", "max", "sum", "pow", "divmod", "round",
        "int", "float", "bool", "str", "list", "tuple", "dict", "set", "frozenset",
        "sorted", "reversed", "enumerate", "zip", "map", "filter", "all", "any",
        "True", "False", "None",
    )
}
_SAFE_BUILTINS["__import__"] = _safe_import


class _Timeout(Exception):
    pass


def _alarm(_signum, _frame):
    raise _Timeout()


def _disallowed_imports(code: str) -> set[str]:
    """Top-level module names imported by `code` that are NOT on the allow-list."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    bad: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for alias in n.names:
                top = alias.name.split(".")[0]
                if top not in _ALLOWED_MODULES:
                    bad.add(top)
        elif isinstance(n, ast.ImportFrom):
            if n.level:                       # relative import (from . import x)
                bad.add(".")
            else:
                top = (n.module or "").split(".")[0]
                if top not in _ALLOWED_MODULES:
                    bad.add(top)
    return bad


def _as_int(got):
    """Coerce an exact-integer result to a Python int, or return None.

    Accepts Python int and exact integers from allowed libs (sympy.Integer,
    numpy.int64, ...). Rejects bool and float (incl. numpy.float64, a float
    subclass) so the original "must return an int" contract is unchanged, plus
    non-integral values. Avoids float() so arbitrary-precision bignums survive.
    """
    if isinstance(got, (bool, float)):
        return None
    if isinstance(got, int):
        return got
    try:
        gi = int(got)
        return gi if gi == got else None     # gi == got is False for Fraction(1,2), Decimal('2.5')
    except (TypeError, ValueError, OverflowError):
        return None


def extract_function(text: str) -> str | None:
    """Pull a ``def a(n):`` body out of a markdown code block."""
    if text.count("```") >= 2:
        first = text.find("```") + 3
        second = text.find("```", first)
        fx = text[first:second].strip()
        fx = fx.removeprefix("python\n")
        idx = fx.find("def ")
        if idx != -1:
            fx = fx[idx:]
            if fx.startswith("def a(n"):
                return fx
    return None


def compile_candidate(code: str):
    """Compile ``code`` and return its ``a`` callable, or raise.

    Rejects non-whitelisted imports and runs in a namespace with only the safe
    builtins (whose ``__import__`` permits the allow-listed modules).
    """
    bad = _disallowed_imports(code)
    if bad:
        raise ValueError(f"import not allowed: {', '.join(sorted(bad))}")
    namespace: dict = {"__builtins__": _SAFE_BUILTINS}
    compiled = compile(code, "<candidate>", "exec")
    exec(compiled, namespace)  # noqa: S102 -- sandboxed builtins, no imports
    fn = namespace.get("a")
    if not callable(fn):
        raise ValueError("no callable a(n) defined")
    return fn


def evaluate(code: str, task: Task, timeout: int = 5, *, _detail: list | None = None) -> tuple[int, int]:
    """Run ``a(n)`` across the task's terms.

    Returns ``(shown_correct, holdout_correct)``. The whole evaluation shares one
    wall-clock ``timeout`` to defend against infinite loops.
    If ``_detail`` is a list, per-term results are appended to it for trace logging.
    """
    fn = compile_candidate(code)
    terms = task.shown + task.holdout
    shown_correct = holdout_correct = 0
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    try:
        for i, expected in enumerate(terms):
            phase = "shown" if i < len(task.shown) else "holdout"
            try:
                got = fn(task.offset + i)
            except _Timeout:
                if _detail is not None:
                    _detail.append({"n": task.offset + i, "phase": phase,
                                    "expected": expected, "got": None,
                                    "match": False, "error": "timeout"})
                raise
            except Exception as e:  # noqa: BLE001 -- a bad term is just a mismatch
                if _detail is not None:
                    _detail.append({"n": task.offset + i, "phase": phase,
                                    "expected": expected, "got": None,
                                    "match": False, "error": repr(e)})
                continue
            got_int = _as_int(got)
            if got_int is None:
                if _detail is not None:
                    _detail.append({"n": task.offset + i, "phase": phase,
                                    "expected": expected, "got": repr(got),
                                    "match": False, "error": "wrong_type"})
                continue
            match = (got_int == expected)
            if _detail is not None:
                _detail.append({"n": task.offset + i, "phase": phase,
                                "expected": expected, "got": got_int,
                                "match": match, "error": None})
            if match:
                if i < len(task.shown):
                    shown_correct += 1
                else:
                    holdout_correct += 1
    except _Timeout:
        pass
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    return shown_correct, holdout_correct


# --- GRPO reward functions --------------------------------------------------
# TRL passes each dataset column through as a keyword list aligned with
# `completions`. We rely on columns: shown, holdout, offset, name, anum.
# shown/holdout are JSON-encoded term lists (OEIS bignums overflow Arrow int64).


def _tasks_from_kwargs(kwargs, i) -> Task:
    return Task(
        anum=kwargs["anum"][i],
        name=kwargs["name"][i],
        offset=kwargs["offset"][i],
        shown=json.loads(kwargs["shown"][i]),
        holdout=json.loads(kwargs["holdout"][i]),
    )


# --- per-example outcome tracking (for next-run curriculum) -----------------
# Records, per anum, each scored completion's (solved, matched_any_holdout).
# sequence_matches populates it; export_outcomes() distills it into the anums to
# drop next run (always solved -> zero-variance group -> no GRPO signal) and the
# ones that look too hard (never matched a single holdout term).
_OUTCOMES: dict[str, list[tuple[bool, bool]]] = {}


def _record_outcome(kwargs, i, *, solved: bool, any_holdout: bool) -> None:
    anum = kwargs.get("anum")
    if anum is not None:
        _OUTCOMES.setdefault(anum[i], []).append((solved, any_holdout))


def export_outcomes(path: str) -> dict:
    """Write JSON of always-solved / never-matched anums seen during training.

    `always_solved` are candidates to omit from the next run's dataset via
    ``build_dataset(exclude_anums=...)``. Only covers sequences actually sampled.
    """
    always_solved = sorted(a for a, rs in _OUTCOMES.items() if all(s for s, _ in rs))
    never_matched = sorted(a for a, rs in _OUTCOMES.items() if not any(h for _, h in rs))
    summary = {
        "seen": len(_OUTCOMES),
        "always_solved": always_solved,
        "never_matched": never_matched,
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# Trace logging — enabled by open_trace(); no-op otherwise.
# Each GRPO step emits one JSONL record per completion with: prompt, completion,
# extracted function, per-reward scores + reasons, full eval term detail.
# ---------------------------------------------------------------------------
_trace_lock = _threading.Lock()
_trace_file = None
_trace_step = 0          # incremented once per batch in function_works
_trace_accum: dict = {}  # (step, idx) -> partial record; flushed by sequence_matches


def open_trace(path: str) -> None:
    global _trace_file
    _trace_file = open(path, "a", buffering=1)  # line-buffered


def close_trace() -> None:
    global _trace_file
    if _trace_file is not None:
        _trace_file.close()
        _trace_file = None


def _trace_add(step: int, idx: int, **fields) -> None:
    if _trace_file is None:
        return
    with _trace_lock:
        _trace_accum.setdefault((step, idx), {}).update(fields)


def _trace_emit(step: int, idx: int, **fields) -> None:
    """Merge accumulated partial fields, add final fields, write one JSONL line."""
    if _trace_file is None:
        return
    with _trace_lock:
        rec = _trace_accum.pop((step, idx), {})
    rec.update(fields)
    rec["step"] = step
    rec["ts"] = _dt.datetime.now().isoformat()
    rec["reward_total"] = rec.get("fw_score", 0) + rec.get("nc_score", 0) + rec.get("sm_score", 0)
    with _trace_lock:
        _trace_file.write(json.dumps(rec) + "\n")


def function_works(completions, **kwargs):
    """+1 if a valid, compilable a(n) was produced; negative otherwise."""
    global _trace_step
    if _trace_file is not None:
        with _trace_lock:
            _trace_step += 1
    step = _trace_step
    scores = []
    for i, completion in enumerate(completions):
        text = completion[0]["content"]
        fx = extract_function(text)
        if fx is None:
            scores.append(-2.0)
            _trace_add(step, i, completion=text, function=None,
                       fw_score=-2.0, fw_reason="no_function_found")
            continue
        try:
            compile_candidate(fx)
            scores.append(1.0)
            _trace_add(step, i, completion=text, function=fx,
                       fw_score=1.0, fw_reason="ok")
        except Exception as e:  # noqa: BLE001
            scores.append(-1.0)
            _trace_add(step, i, completion=text, function=fx,
                       fw_score=-1.0, fw_reason=repr(e))
    return scores


def no_cheating(completions, **kwargs):
    """Penalize imports of non-whitelisted modules (allowed math/sci ones are fine)."""
    step = _trace_step
    scores = []
    for i, completion in enumerate(completions):
        fx = extract_function(completion[0]["content"])
        if fx is None:
            scores.append(-1.0)
            _trace_add(step, i, nc_score=-1.0, nc_reason="no_function_found")
            continue
        bad = _disallowed_imports(fx)
        if bad:
            scores.append(-20.0)
            _trace_add(step, i, nc_score=-20.0, nc_reason=f"bad_import:{','.join(sorted(bad))}")
        else:
            scores.append(1.0)
            _trace_add(step, i, nc_score=1.0, nc_reason="ok")
    return scores


def sequence_matches(completions, **kwargs):
    """Reward generalization, scored mostly on unseen holdout terms.

    A hardcoded lookup of the shown terms tops out near +1; a genuinely correct
    rule earns the shown credit, the (heavily weighted) holdout credit, and a
    large bonus for matching every term.
    """
    step = _trace_step
    scores = []
    for i, completion in enumerate(completions):
        fx = extract_function(completion[0]["content"])
        task = _tasks_from_kwargs(kwargs, i)
        if fx is None:
            scores.append(0.0)
            _record_outcome(kwargs, i, solved=False, any_holdout=False)
            _trace_emit(step, i, anum=task.anum, name=task.name,
                        prompt=task.prompt_text(), sm_score=0.0,
                        sm_reason="no_function_found",
                        shown_ok=0, hold_ok=0, eval_detail=None)
            continue
        detail: list | None = [] if _trace_file else None
        try:
            shown_ok, hold_ok = evaluate(fx, task, _detail=detail)
        except Exception as e:  # noqa: BLE001 -- compile/exec failure
            scores.append(-2.0)
            _record_outcome(kwargs, i, solved=False, any_holdout=False)
            _trace_emit(step, i, anum=task.anum, name=task.name,
                        prompt=task.prompt_text(), sm_score=-2.0,
                        sm_reason=repr(e), shown_ok=0, hold_ok=0,
                        eval_detail=detail)
            continue
        shown_frac = shown_ok / max(1, len(task.shown))
        hold_frac = hold_ok / max(1, len(task.holdout))
        solved = shown_ok == len(task.shown) and hold_ok == len(task.holdout)
        lookup = shown_ok == len(task.shown) and hold_ok == 0
        if solved:
            reward = 1.0 * shown_frac + 10.0 * hold_frac + 20.0  # fully general
        elif lookup:
            reward = -5.0  # passes every shown term, zero holdout -> hardcoded lookup
        else:
            reward = 1.0 * shown_frac + 10.0 * hold_frac
        scores.append(reward)
        _record_outcome(kwargs, i, solved=solved, any_holdout=hold_ok > 0)
        _trace_emit(step, i, anum=task.anum, name=task.name,
                    prompt=task.prompt_text(), sm_score=reward,
                    sm_reason=(f"shown={shown_ok}/{len(task.shown)} "
                               f"holdout={hold_ok}/{len(task.holdout)}"
                               + (" SOLVED" if solved else " LOOKUP" if lookup else "")),
                    shown_ok=shown_ok, hold_ok=hold_ok, eval_detail=detail)
    return scores


REWARD_FUNCS = [function_works, no_cheating, sequence_matches]
