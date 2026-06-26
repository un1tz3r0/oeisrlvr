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
import glob as _glob
import json
import random
import signal
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
- Use only native Python built-ins. No import statements.
- Find the general rule; do NOT hardcode a lookup table of the terms above.
- All helper code must be inside a(n).

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


class _Timeout(Exception):
    pass


def _alarm(_signum, _frame):
    raise _Timeout()


def _has_import(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree))


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

    Rejects imports and runs in a namespace with only the safe builtins.
    """
    if _has_import(code):
        raise ValueError("import statement not allowed")
    namespace: dict = {"__builtins__": _SAFE_BUILTINS}
    compiled = compile(code, "<candidate>", "exec")
    exec(compiled, namespace)  # noqa: S102 -- sandboxed builtins, no imports
    fn = namespace.get("a")
    if not callable(fn):
        raise ValueError("no callable a(n) defined")
    return fn


def evaluate(code: str, task: Task, timeout: int = 5) -> tuple[int, int]:
    """Run ``a(n)`` across the task's terms.

    Returns ``(shown_correct, holdout_correct)``. The whole evaluation shares one
    wall-clock ``timeout`` to defend against infinite loops.
    """
    fn = compile_candidate(code)
    terms = task.shown + task.holdout
    shown_correct = holdout_correct = 0
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    try:
        for i, expected in enumerate(terms):
            try:
                got = fn(task.offset + i)
            except _Timeout:
                raise
            except Exception:  # noqa: BLE001 -- a bad term is just a mismatch
                continue
            if isinstance(got, bool) or not isinstance(got, int):
                continue
            if got == expected:
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


def function_works(completions, **kwargs):
    """+1 if a valid, compilable a(n) was produced; negative otherwise."""
    scores = []
    for completion in completions:
        fx = extract_function(completion[0]["content"])
        if fx is None:
            scores.append(-2.0)
            continue
        try:
            compile_candidate(fx)
            scores.append(1.0)
        except Exception:  # noqa: BLE001
            scores.append(-1.0)
    return scores


def no_cheating(completions, **kwargs):
    """Penalize import statements (the only sandbox escape we forbid)."""
    scores = []
    for completion in completions:
        fx = extract_function(completion[0]["content"])
        if fx is None:
            scores.append(-1.0)
        elif _has_import(fx):
            scores.append(-20.0)
        else:
            scores.append(1.0)
    return scores


def sequence_matches(completions, **kwargs):
    """Reward generalization, scored mostly on unseen holdout terms.

    A hardcoded lookup of the shown terms tops out near +1; a genuinely correct
    rule earns the shown credit, the (heavily weighted) holdout credit, and a
    large bonus for matching every term.
    """
    scores = []
    for i, completion in enumerate(completions):
        fx = extract_function(completion[0]["content"])
        if fx is None:
            scores.append(0.0)
            _record_outcome(kwargs, i, solved=False, any_holdout=False)
            continue
        task = _tasks_from_kwargs(kwargs, i)
        try:
            shown_ok, hold_ok = evaluate(fx, task)
        except Exception:  # noqa: BLE001 -- compile/exec failure
            scores.append(-2.0)
            _record_outcome(kwargs, i, solved=False, any_holdout=False)
            continue
        shown_frac = shown_ok / max(1, len(task.shown))
        hold_frac = hold_ok / max(1, len(task.holdout))
        reward = 1.0 * shown_frac + 8.0 * hold_frac
        solved = shown_ok == len(task.shown) and hold_ok == len(task.holdout)
        if solved:
            reward += 20.0  # fully general solution
        scores.append(reward)
        _record_outcome(kwargs, i, solved=solved, any_holdout=hold_ok > 0)
    return scores


REWARD_FUNCS = [function_works, no_cheating, sequence_matches]
