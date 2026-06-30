# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); the project is pre-release and not
yet versioned.

## [Unreleased]

### Added
- **Curated import allow-list** for model-written `a(n)`: `math`, `cmath`,
  `fractions`, `decimal`, `numbers`, `itertools`, `functools`, `operator`,
  `statistics`, `numpy`, `sympy`, `mpmath` — enforced at parse time and via a
  sandboxed `__import__` (`os`/`subprocess`/etc. stay blocked). Replaces the
  previous blanket "no imports" rule.
- `sympy`/`numpy` integer results are accepted in evaluation (coerced to `int`,
  arbitrary-precision-safe) so the allow-listed tools aren't penalized.
- `CHANGELOG.md`.

### Changed
- **Renamed** `oeisctl.py` → `rlvrctl.py` and `oeis_tui.py` → `rlvrtui.py`.
- **Output directory layout.** Training runs are now gap-free numbered checkpoints
  `training/<model>/NNN/`; each eval nests under the highest checkpoint it compares
  (`<latest>/evals/<others>[.N]/`), with relative symlinks from every earlier
  participant — including the base model under `base/`. Existing runs were migrated.
- **GRPO signal.** `num_generations` 2 → 4, with `gradient_accumulation_steps` tied
  to it so TRL's divisibility constraint always holds (one prompt-group per step).
- **Reward shaping.** Held-out term weight 8 → 10; pure lookup-table completions
  (all shown terms correct, zero held-out) now score −5 (net negative) instead of a
  small positive, and are tagged `LOOKUP` in the trace.
- **Prompt** now describes the allowed modules and steers toward exact-integer tools
  (`math.comb`/`factorial`/`isqrt`/`gcd`, `fractions`, `sympy`); notes `numpy` is
  fixed-width and to avoid it for large terms.
- Repo reorganized: `attic/` (gitignored) holds scratch/archived files, `docs/` holds
  help notes; README brought in line with the current tooling and layout.

### Fixed
- `eval_oeis.py`: an undefined `ra` raised `NameError` when writing the `--jsonl`
  summary (i.e. on every orchestrated eval). Now uses the held-out sequence count.
