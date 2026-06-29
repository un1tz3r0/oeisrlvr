# TUI Configure screen — per-field UX polish (deferred)

The Configure form is functional but monolithic/sparse. Each hyperparameter
deserves more individual attention. Ideas to revisit later (not blocking):

## Field presentation
- **Tailored field widths** — size each input to the expected max width of its
  value (e.g. `lora_rank` is short, `base_model` is long) instead of a uniform
  width. Less wasted space, easier scanning.
- Less "boring": group/visually distinguish related params; give each field some
  individual identity rather than a uniform list.

## Per-field controls
- **+/- stepper buttons** — increment/decrement pair for discrete scalar values
  (steps, ranks, counts), with sensible step sizes per field.
- **Reset-to-default button** — a recycle/↺ icon that snaps the field back to its
  inherited default value (shortcut vs. clearing manually).
- **"Auto" button** — for values that have a heuristic default (e.g. derive
  `lora_alpha = 2*lora_rank`, or size `max_completion_length` from the pool),
  a button that fills the heuristic value.

## Override awareness (the layered config: model defaults → persistent → ephemeral)
- **Overridden indicator** — when a field is shadowed by a higher-priority
  setting elsewhere, show an "overridden" icon, or auto-disable + gray out the
  control *and its label*.
- **Jump-to-override** — a button on a shadowed field that navigates to the
  screen/tab holding the overriding value, with that field auto-focused — so the
  user can change it at the right level without hunting.

## Help / discoverability
- **Collapsible per-field help** — detailed description of what the field does,
  its effects, normative ranges/values, and directional guidance
  ("increase → X, which has effect A; decrease → Y, which has effect B").

## Further ideas (mine)
- **Inline validation** — flag out-of-range / nonsensical values as you type
  (e.g. `num_generations < 2` is invalid for GRPO).
- **Effective-value preview** — show the resolved effective value per field after
  the full merge (defaults ← persistent ← ephemeral), so you see what a run will
  actually use without mentally merging layers.
- **Dirty/unsaved indicator** — mark changed-but-unsaved fields; warn on leaving.
- **Diff vs. last run** — highlight params that changed since the most recent
  cycle's snapshotted `config.toml`.
