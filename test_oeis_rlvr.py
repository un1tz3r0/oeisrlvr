"""Verify the OEIS RLVR environment: dataset build + reward separation."""

import json

from oeis_rlvr import (
    Task,
    build_dataset,
    evaluate,
    function_works,
    no_cheating,
    sequence_matches,
)


def _completion(code: str):
    # Mimic TRL's completion shape: a list with one {"role","content"} dict.
    return [[{"role": "assistant", "content": f"```python\n{code}\n```"}]]


def _kwargs_for(task: Task):
    # shown/holdout are JSON-encoded, matching how the training dataset stores them.
    return dict(
        anum=[task.anum], name=[task.name], offset=[task.offset],
        shown=[json.dumps(task.shown)], holdout=[json.dumps(task.holdout)],
    )


def reward_row(code: str, task: Task):
    c = _completion(code)
    kw = _kwargs_for(task)
    return (
        function_works(c)[0],
        no_cheating(c)[0],
        sequence_matches(c, **kw)[0],
    )


# A000045 Fibonacci-style task built by hand (offset 0): 0,1,1,2,3,5,8,...
FIB = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]
fib_task = Task(anum="A000045", name="Fibonacci numbers",
                offset=0, shown=FIB[:10], holdout=FIB[10:])

CORRECT = """
def a(n):
    x, y = 0, 1
    for _ in range(n):
        x, y = y, x + y
    return x
"""

HARDCODED = """
def a(n):
    table = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    return table[n]
"""

WRONG = """
def a(n):
    return n * n
"""

IMPORTS = """
def a(n):
    import math
    return n
"""

print("=== reward separation on a hand-built Fibonacci task ===")
for label, code in [("correct", CORRECT), ("hardcoded", HARDCODED),
                    ("wrong", WRONG), ("imports", IMPORTS)]:
    fw, nc, sm = reward_row(code, fib_task)
    print(f"  {label:10s} function_works={fw:+.1f}  no_cheating={nc:+.1f}  "
          f"sequence_matches={sm:+.2f}  total={fw+nc+sm:+.2f}")

print("\n  sanity: evaluate(correct) shown/holdout =", evaluate(CORRECT, fib_task))
print("  sanity: evaluate(hardcoded) shown/holdout =", evaluate(HARDCODED, fib_task))

print("\n=== dataset build (limit 50 from A000-A001 shard) ===")
tasks = build_dataset(glob_pattern="oeisdata/seq/A00[01]/A??????.seq", limit=50)
print(f"  built {len(tasks)} tasks")
for t in tasks[:3]:
    print(f"\n  --- {t.anum}: {t.name[:60]}")
    print(f"      offset={t.offset} shown={len(t.shown)} holdout={len(t.holdout)}")
print("\n  example prompt:\n")
print("  " + tasks[0].prompt_text().replace("\n", "\n  "))
