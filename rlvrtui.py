#!/usr/bin/env python3
"""Textual front-end over rlvrctl for the OEIS RLVR train/eval workflow.

Wizard flow: Intro -> select/create model -> main menu (Auto / Train / Eval /
Configure / Status). Long-running train/eval stages run as detached subprocesses
(via rlvrctl), so quitting the TUI leaves them running; relaunching reattaches.

Run:  uv run python rlvrtui.py
"""
from __future__ import annotations

import os

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Center, Horizontal, Middle, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button, Collapsible, DataTable, Footer, Header, Input, Label, ListItem,
    ListView, ProgressBar, RichLog, Select, Static, TabbedContent, TabPane,
)

import rlvrctl

MODEL_PARAMS = rlvrctl.MODEL_PARAMS
TRAIN_PARAMS = rlvrctl.TRAIN_PARAMS
EVAL_PARAMS = rlvrctl.EVAL_PARAMS


def _coerce_or_none(s: str):
    s = s.strip()
    return rlvrctl._coerce(s) if s else None


def field(label: str, fid: str, value: str = "", placeholder: str = "") -> Horizontal:
    return Horizontal(
        Label(label, classes="flabel"),
        Input(value=value, placeholder=placeholder, id=fid),
        classes="frow",
    )

CREATE_SENTINEL = "__create__"


# --- intro ------------------------------------------------------------------

class IntroScreen(Screen):
    BINDINGS = [("enter", "go", "Continue"), ("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Middle(Center(Vertical(
            Static("[b]OEIS RLVR — Training Dashboard[/b]", classes="title"),
            Static(
                "Orchestrate the load → train → save → eval → tweak loop for "
                "LoRA adapters on OEIS program synthesis.\n\n"
                "Train and eval stages run detached: you can quit and reattach "
                "without interrupting a run.",
                classes="blurb",
            ),
            Center(Button("Continue", variant="primary", id="continue")),
            classes="panel",
        )))
        yield Footer()

    @on(Button.Pressed, "#continue")
    def action_go(self) -> None:
        self.app.push_screen(ModelSelectScreen())


# --- model selection / creation ---------------------------------------------

class ModelSelectScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Back"), ("r", "refresh", "Refresh")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static("[b]Select a model[/b]", classes="title"),
            ListView(id="models"),
            classes="panel",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.action_refresh()

    def on_screen_resume(self) -> None:
        self.action_refresh()

    def action_refresh(self) -> None:
        lv = self.query_one("#models", ListView)
        lv.clear()
        for name in rlvrctl.list_models():
            n_cycles = len(rlvrctl.cycle_summary(name))
            lv.append(ListItem(Label(f"{name}   [dim]({n_cycles} cycles)[/dim]"), name=name))
        lv.append(ListItem(Label("[green]+ Create new model…[/green]"), name=CREATE_SENTINEL))

    @on(ListView.Selected)
    def chosen(self, ev: ListView.Selected) -> None:
        name = ev.item.name
        if name == CREATE_SENTINEL:
            self.app.push_screen(NewModelScreen())
        else:
            self.app.push_screen(MainMenuScreen(name))


class NewModelScreen(ModalScreen):
    BINDINGS = [("escape", "app.pop_screen", "Cancel")]

    def compose(self) -> ComposeResult:
        d = rlvrctl.DEFAULTS
        yield Vertical(
            Static("[b]New model[/b]", classes="title"),
            Label("Name"), Input(placeholder="e.g. gemma4-oeis-e2b", id="name"),
            Label("Base model"), Input(value=d["base_model"], id="base_model"),
            Label("LoRA rank"), Input(value=str(d["lora_rank"]), id="lora_rank"),
            Label("LoRA alpha"), Input(value=str(d["lora_alpha"]), id="lora_alpha"),
            Horizontal(
                Button("Create", variant="primary", id="create"),
                Button("Cancel", id="cancel"),
                classes="row",
            ),
            classes="panel",
        )

    @on(Button.Pressed, "#cancel")
    def cancel(self) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#create")
    def create(self) -> None:
        name = self.query_one("#name", Input).value.strip()
        if not name:
            self.notify("Name is required", severity="error")
            return
        overrides = {k: self.query_one(f"#{k}", Input).value.strip()
                     for k in ("base_model", "lora_rank", "lora_alpha")}
        try:
            rlvrctl.create_model(name, overrides)
        except FileExistsError as e:
            self.notify(str(e), severity="error")
            return
        self.notify(f"Created model '{name}'")
        self.app.pop_screen()  # back to ModelSelect (refreshes on resume)


# --- main menu --------------------------------------------------------------

class MainMenuScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Back"), ("q", "quit", "Quit")]

    def __init__(self, model: str) -> None:
        super().__init__()
        self.model = model

    def on_mount(self) -> None:
        live = rlvrctl.find_live_cycles(self.model)
        if live:
            self.notify(f"{len(live)} live run(s) — open Status to attach")

    def compose(self) -> ComposeResult:
        yield Header()
        yield Middle(Center(Vertical(
            Static(f"[b]{self.model}[/b]", classes="title"),
            Button("Auto  — alternating train/eval loop", id="auto"),
            Button("Train — one training cycle", id="train"),
            Button("Eval  — compare adapters", id="eval"),
            Button("Configure — hyperparameters & presets", id="configure"),
            Button("Status — cycle history", id="status"),
            Button("Back", id="back"),
            classes="panel menu",
        )))
        yield Footer()

    @on(Button.Pressed, "#status")
    def status(self) -> None:
        self.app.push_screen(StatusScreen(self.model))

    @on(Button.Pressed, "#configure")
    def configure(self) -> None:
        self.app.push_screen(ConfigureScreen(self.model))

    @on(Button.Pressed, "#train")
    def train(self) -> None:
        if rlvrctl.find_live_cycles(self.model):
            self.notify("A run is already live; attach via Status", severity="warning")
            return
        cycle, _pid = rlvrctl.start_train(self.model)
        self.app.push_screen(RunMonitorScreen(cycle, "train"))

    @on(Button.Pressed, "#eval")
    def evaluate(self) -> None:
        self.app.push_screen(EvalSetupScreen(self.model))

    @on(Button.Pressed, "#auto")
    def auto(self) -> None:
        self.app.push_screen(AutoScreen(self.model))

    @on(Button.Pressed, "#back")
    def back(self) -> None:
        self.app.pop_screen()


# --- status / cycle history -------------------------------------------------

class StatusScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Back"), ("r", "refresh", "Refresh")]

    def __init__(self, model: str) -> None:
        super().__init__()
        self.model = model

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static(f"[b]{self.model} — cycles[/b]", classes="title"),
            DataTable(id="cycles", zebra_stripes=True),
            classes="panel",
        )
        yield Footer()

    def on_mount(self) -> None:
        t = self.query_one("#cycles", DataTable)
        t.add_columns("Cycle", "Kind", "State", "Live", "Created")
        self.action_refresh()

    def action_refresh(self) -> None:
        t = self.query_one("#cycles", DataTable)
        t.clear()
        self._rows = rlvrctl.cycle_summary(self.model)
        for r in self._rows:
            t.add_row(r["num"], r["kind"], r["state"],
                      "● LIVE" if r["live"] else "", r["created"])

    @on(DataTable.RowSelected)
    def open_cycle(self, ev: DataTable.RowSelected) -> None:
        r = self._rows[ev.cursor_row]
        self.app.push_screen(RunMonitorScreen(r["path"], r["kind"]))


# --- configure (guided tabbed form) -----------------------------------------

class ConfigureScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def __init__(self, model: str) -> None:
        super().__init__()
        self.model = model
        self.mc = rlvrctl.read_model_config(model)
        self.td_def = dict(self.mc.get("train_defaults", {}))
        self.ed_def = dict(self.mc.get("eval_defaults", {}))

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Model", id="tab-model"):
                with VerticalScroll(classes="formpane"):
                    yield Static("[b]Model & LoRA[/b]")
                    for k in MODEL_PARAMS:
                        yield field(k, f"m_{k}", str(self.mc.get(k, "")))
                    yield Static("[b]Train defaults[/b]")
                    for k in TRAIN_PARAMS:
                        yield field(k, f"td_{k}", str(self.td_def.get(k, "")))
                    yield Static("[b]Eval defaults[/b]")
                    for k in EVAL_PARAMS:
                        yield field(k, f"ed_{k}", str(self.ed_def.get(k, "")))
                    yield self._save_revert("model")
            with TabPane("Training", id="tab-train"):
                with VerticalScroll(classes="formpane"):
                    yield Static("[b]Persistent overrides[/b] [dim](blank = inherit default)[/dim]")
                    for k in TRAIN_PARAMS:
                        yield field(k, f"tp_{k}", placeholder=f"default {self.td_def.get(k, '')}")
                    with Collapsible(title="Ephemeral (one-shot, cleared after run)"):
                        for k in TRAIN_PARAMS:
                            yield field(k, f"te_{k}")
                    yield self._save_revert("train")
            with TabPane("Eval", id="tab-eval"):
                with VerticalScroll(classes="formpane"):
                    yield Static("[b]Persistent overrides[/b] [dim](blank = inherit default)[/dim]")
                    for k in EVAL_PARAMS:
                        yield field(k, f"ep_{k}", placeholder=f"default {self.ed_def.get(k, '')}")
                    with Collapsible(title="Ephemeral (one-shot, cleared after run)"):
                        for k in EVAL_PARAMS:
                            yield field(k, f"ee_{k}")
                    yield self._save_revert("eval")
            with TabPane("Presets", id="tab-presets"):
                with VerticalScroll(classes="formpane"):
                    yield Static("[b]Apply a preset[/b] [dim](merges into persistent overrides)[/dim]")
                    yield Select([(p, p) for p in rlvrctl.list_presets()],
                                 prompt="Select preset…", id="preset_sel", allow_blank=True)
                    yield Button("Apply to this model", id="preset_apply")
                    yield Static("[b]Save current overrides as preset[/b]")
                    yield field("name", "preset_name", placeholder="e.g. quick-eval")
                    yield Button("Save preset", id="preset_save", variant="primary")
        yield Footer()

    def _save_revert(self, which: str) -> Horizontal:
        return Horizontal(
            Button("Save", variant="primary", id=f"save_{which}"),
            Button("Revert", id=f"revert_{which}"),
            classes="row",
        )

    def on_mount(self) -> None:
        self._load(("train", "eval"))

    def _load(self, whichs) -> None:
        """Fill the override inputs from disk."""
        for which, pre in (("training", "tp"), ("eval", "ep")):
            if which[:4] not in whichs and which not in whichs:
                continue
            persistent, ephemeral = rlvrctl.read_overrides(self.model, which)
            eph_pre = {"tp": "te", "ep": "ee"}[pre]
            params = TRAIN_PARAMS if pre == "tp" else EVAL_PARAMS
            for k in params:
                self.query_one(f"#{pre}_{k}", Input).value = str(persistent.get(k, ""))
                self.query_one(f"#{eph_pre}_{k}", Input).value = str(ephemeral.get(k, ""))

    # --- saves ---
    @on(Button.Pressed, "#save_model")
    def save_model(self) -> None:
        top = {k: rlvrctl._coerce(self.query_one(f"#m_{k}", Input).value.strip())
               for k in MODEL_PARAMS}
        td = {k: rlvrctl._coerce(self.query_one(f"#td_{k}", Input).value.strip())
              for k in TRAIN_PARAMS if self.query_one(f"#td_{k}", Input).value.strip()}
        ed = {k: rlvrctl._coerce(self.query_one(f"#ed_{k}", Input).value.strip())
              for k in EVAL_PARAMS if self.query_one(f"#ed_{k}", Input).value.strip()}
        rlvrctl.write_model_config(self.model, top, td, ed)
        self.notify("Saved model.toml")

    @on(Button.Pressed, "#save_train")
    def save_train(self) -> None:
        self._save_overrides("training", "tp", "te", TRAIN_PARAMS)

    @on(Button.Pressed, "#save_eval")
    def save_eval(self) -> None:
        self._save_overrides("eval", "ep", "ee", EVAL_PARAMS)

    def _save_overrides(self, which, pre, eph_pre, params) -> None:
        persistent, ephemeral = {}, {}
        for k in params:
            v = _coerce_or_none(self.query_one(f"#{pre}_{k}", Input).value)
            if v is not None:
                persistent[k] = v
            e = _coerce_or_none(self.query_one(f"#{eph_pre}_{k}", Input).value)
            if e is not None:
                ephemeral[k] = e
        rlvrctl.write_overrides(self.model, which, persistent, ephemeral)
        self.notify(f"Saved {which}.toml")

    # --- reverts ---
    @on(Button.Pressed, "#revert_train")
    def revert_train(self) -> None:
        self._load(("train",)); self.notify("Reverted training")

    @on(Button.Pressed, "#revert_eval")
    def revert_eval(self) -> None:
        self._load(("eval",)); self.notify("Reverted eval")

    @on(Button.Pressed, "#revert_model")
    def revert_model(self) -> None:
        for k in MODEL_PARAMS:
            self.query_one(f"#m_{k}", Input).value = str(self.mc.get(k, ""))
        for k in TRAIN_PARAMS:
            self.query_one(f"#td_{k}", Input).value = str(self.td_def.get(k, ""))
        for k in EVAL_PARAMS:
            self.query_one(f"#ed_{k}", Input).value = str(self.ed_def.get(k, ""))
        self.notify("Reverted model")

    # --- presets ---
    @on(Button.Pressed, "#preset_apply")
    def preset_apply(self) -> None:
        sel = self.query_one("#preset_sel", Select).value
        if sel is Select.BLANK:
            self.notify("Pick a preset first", severity="warning")
            return
        rlvrctl.apply_preset(self.model, sel)
        self._load(("train", "eval"))
        self.notify(f"Applied preset '{sel}'")

    @on(Button.Pressed, "#preset_save")
    def preset_save(self) -> None:
        name = self.query_one("#preset_name", Input).value.strip()
        if not name:
            self.notify("Preset name required", severity="error")
            return
        train, _ = rlvrctl.read_overrides(self.model, "training")
        eval_, _ = rlvrctl.read_overrides(self.model, "eval")
        rlvrctl.save_preset(name, train, eval_)
        sel = self.query_one("#preset_sel", Select)
        sel.set_options([(p, p) for p in rlvrctl.list_presets()])
        self.notify(f"Saved preset '{name}' (from saved overrides)")


# --- run monitor (shared by Train/Eval/Auto) --------------------------------

def _train_stats_text(p: dict, state: str) -> str:
    n = max(1, p["samples"])
    return (
        f"step [b]{p['step']}/{p['total']}[/b]    "
        f"reward recent [b]{p['reward_recent']:.2f}[/b]  (mean {p['reward_mean']:.2f})\n"
        f"solved {p['solved']}/{p['samples']} ({100 * p['solved'] / n:.0f}%)    "
        f"valid fn {p['valid']}/{p['samples']} ({100 * p['valid'] / n:.0f}%)\n"
        f"last: {p['last_anum']} {str(p['last_name'])[:50]}\n"
        f"state: {state}"
    )


def _eval_stats_text(p: dict, n: int) -> str:
    lines = []
    for c in p.get("conditions", []):
        sc = c["scores"]
        mean = sum(sc) / len(sc) if sc else 0.0
        lines.append(f"{c['label']:24} {len(sc)}/{c['total'] or n}   "
                     f"mean_holdout [b]{mean:.3f}[/b]   pass {c['passk']}")
    if p.get("done"):
        lines.append("[green]complete[/green]")
    return "\n".join(lines) or "starting…"


class RunMonitorScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Back"), ("c", "cancel", "Cancel run")]

    def __init__(self, cycle_dir: str, kind: str) -> None:
        super().__init__()
        self.cycle_dir = cycle_dir
        self.kind = kind
        self._log_pos = 0
        self._finished = False
        cfg = rlvrctl.load_toml(os.path.join(cycle_dir, "config.toml"))
        self.params = dict(cfg.get("params", {}))
        self.n_conditions = max(1, len(cfg.get("adapters", []) or [1]))

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static(id="rm_title", classes="title"),
            ProgressBar(id="rm_bar", total=100, show_eta=False),
            Static(id="rm_stats", classes="stats"),
            Collapsible(RichLog(id="rm_log", wrap=True, markup=False, highlight=False),
                        title="Log", collapsed=False),
            Horizontal(Button("Cancel run", variant="error", id="rm_cancel"),
                       Button("Back", id="rm_back"), classes="row"),
            classes="panel monitor",
        )
        yield Footer()

    def on_mount(self) -> None:
        name = os.path.basename(self.cycle_dir)
        self.query_one("#rm_title", Static).update(f"[b]{self.kind} cycle {name}[/b]")
        self.timer = self.set_interval(1.0, self.tick)
        self.tick()

    def _tail(self) -> None:
        log = os.path.join(self.cycle_dir, "run.log")
        if not os.path.exists(log):
            return
        with open(log) as f:
            f.seek(self._log_pos)
            new = f.read()
            self._log_pos = f.tell()
        if new:
            w = self.query_one("#rm_log", RichLog)
            for line in new.splitlines():
                w.write(line)

    def tick(self) -> None:
        self._tail()
        st = rlvrctl.cycle_status(self.cycle_dir)
        if self.kind == "train":
            p = rlvrctl.train_progress(self.cycle_dir)
            self.query_one("#rm_bar", ProgressBar).update(
                total=max(1, p["total"]), progress=p["step"])
            self.query_one("#rm_stats", Static).update(
                _train_stats_text(p, st.get("state", "")))
        else:
            p = rlvrctl.eval_progress(self.cycle_dir)
            n = int(self.params.get("n", 0) or 0)
            done = sum(len(c["scores"]) for c in p.get("conditions", []))
            self.query_one("#rm_bar", ProgressBar).update(
                total=max(1, n * self.n_conditions), progress=done)
            self.query_one("#rm_stats", Static).update(_eval_stats_text(p, n))
        if st.get("state", "") in ("done", "failed", "canceled"):
            self._finish(st["state"])

    def _finish(self, state: str) -> None:
        if self._finished:
            return
        self._finished = True
        self.timer.stop()
        self._tail()
        color = {"done": "green", "failed": "red", "canceled": "yellow"}.get(state, "white")
        self.query_one("#rm_cancel", Button).disabled = True
        name = os.path.basename(self.cycle_dir)
        self.query_one("#rm_title", Static).update(
            f"[b]{self.kind} cycle {name}[/b] — [{color}]{state}[/{color}]")
        self.notify(f"Run {state}",
                    severity="information" if state == "done" else "warning")

    def action_cancel(self) -> None:
        if rlvrctl.cancel(self.cycle_dir):
            self.notify("Sent cancel (SIGTERM)")
        else:
            self.notify("No live run to cancel", severity="warning")

    @on(Button.Pressed, "#rm_cancel")
    def _cancel_btn(self) -> None:
        self.action_cancel()

    @on(Button.Pressed, "#rm_back")
    def _back_btn(self) -> None:
        self.app.pop_screen()


class EvalSetupScreen(ModalScreen):
    BINDINGS = [("escape", "app.pop_screen", "Cancel")]

    def __init__(self, model: str) -> None:
        super().__init__()
        self.model = model

    def compose(self) -> ComposeResult:
        trains = [r["num"] for r in rlvrctl.cycle_summary(self.model)
                  if r["kind"] == "train" and os.path.isdir(os.path.join(r["path"], "adapter"))]
        opts = [("base", "base")] + [(f"cycle {t}", t) for t in trains]
        latest = trains[-1] if trains else "base"
        yield Vertical(
            Static("[b]Eval setup[/b]", classes="title"),
            Label("Condition A"),
            Select(opts, value="base", id="ev_a", allow_blank=False),
            Label("Condition B"),
            Select(opts, value=latest, id="ev_b", allow_blank=False),
            Horizontal(Button("Run eval", variant="primary", id="ev_run"),
                       Button("Cancel", id="ev_cancel"), classes="row"),
            classes="panel",
        )

    @on(Button.Pressed, "#ev_cancel")
    def _cancel(self) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#ev_run")
    def _run(self) -> None:
        a = self.query_one("#ev_a", Select).value
        b = self.query_one("#ev_b", Select).value
        cycle, _pid = rlvrctl.start_eval(self.model, [a, b])
        self.app.pop_screen()
        self.app.push_screen(RunMonitorScreen(cycle, "eval"))


# --- auto loop (TUI-driven train -> eval -> train ...) -----------------------

class AutoScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def __init__(self, model: str) -> None:
        super().__init__()
        self.model = model
        self.timer = None
        self.cycle_dir = None
        self.stage = None          # "train" | "eval"
        self.round = 0
        self.rounds_max = 0        # 0 = unlimited
        self._log_pos = 0
        self.running = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static(f"[b]Auto loop — {self.model}[/b]", classes="title"),
            Horizontal(Label("Rounds (0 = unlimited)", classes="flabel"),
                       Input(value="0", id="auto_rounds"), classes="frow"),
            Horizontal(Button("Start", variant="primary", id="auto_start"),
                       Button("Stop", variant="error", id="auto_stop"), classes="row"),
            Static("idle", id="auto_state", classes="stats"),
            ProgressBar(id="auto_bar", total=100, show_eta=False),
            Static(id="auto_stats", classes="stats"),
            Collapsible(RichLog(id="auto_log", wrap=True, markup=False, highlight=False),
                        title="Log", collapsed=False),
            classes="panel monitor",
        )
        yield Footer()

    @on(Button.Pressed, "#auto_start")
    def start(self) -> None:
        if self.running:
            return
        if rlvrctl.find_live_cycles(self.model):
            self.notify("A run is already live; stop it first", severity="warning")
            return
        try:
            self.rounds_max = int(self.query_one("#auto_rounds", Input).value or "0")
        except ValueError:
            self.rounds_max = 0
        self.running = True
        self.round = 1
        self._begin_train()
        self.timer = self.set_interval(1.0, self.tick)

    def _begin_train(self) -> None:
        self.stage = "train"
        self.cycle_dir, _ = rlvrctl.start_train(self.model)
        self._log_pos = 0
        self._set_state(f"round {self.round}: training ({os.path.basename(self.cycle_dir)})")

    def _begin_eval(self) -> None:
        # compare base + previous-train + latest-train (the "both" baseline choice)
        trains = [r["num"] for r in rlvrctl.cycle_summary(self.model)
                  if r["kind"] == "train" and os.path.isdir(os.path.join(r["path"], "adapter"))]
        specs = ["base"]
        if len(trains) >= 2:
            specs.append(trains[-2])
        specs.append(trains[-1])
        self.stage = "eval"
        self.cycle_dir, _ = rlvrctl.start_eval(self.model, specs)
        self._log_pos = 0
        self._set_state(f"round {self.round}: evaluating ({' vs '.join(specs)})")

    def _set_state(self, msg: str) -> None:
        self.query_one("#auto_state", Static).update(msg)

    def _tail(self) -> None:
        if not self.cycle_dir:
            return
        log = os.path.join(self.cycle_dir, "run.log")
        if not os.path.exists(log):
            return
        with open(log) as f:
            f.seek(self._log_pos)
            new = f.read()
            self._log_pos = f.tell()
        if new:
            w = self.query_one("#auto_log", RichLog)
            for line in new.splitlines():
                w.write(line)

    def tick(self) -> None:
        if not self.running or not self.cycle_dir:
            return
        self._tail()
        st = rlvrctl.cycle_status(self.cycle_dir)
        if self.stage == "train":
            p = rlvrctl.train_progress(self.cycle_dir)
            self.query_one("#auto_bar", ProgressBar).update(
                total=max(1, p["total"]), progress=p["step"])
            self.query_one("#auto_stats", Static).update(_train_stats_text(p, st.get("state", "")))
        else:
            p = rlvrctl.eval_progress(self.cycle_dir)
            n = int(rlvrctl.read_model_config(self.model).get("eval_defaults", {}).get("n", 0) or 0)
            done = sum(len(c["scores"]) for c in p.get("conditions", []))
            self.query_one("#auto_bar", ProgressBar).update(
                total=max(1, n * 3), progress=done)
            self.query_one("#auto_stats", Static).update(_eval_stats_text(p, n))
        state = st.get("state", "")
        if state in ("failed", "canceled"):
            self._stop(f"{self.stage} {state} — loop halted")
        elif state == "done":
            self._advance()

    def _advance(self) -> None:
        if self.stage == "train":
            self._begin_eval()
        else:  # finished an eval round
            if self.rounds_max and self.round >= self.rounds_max:
                self._stop(f"completed {self.round} round(s)")
            else:
                self.round += 1
                self._begin_train()

    def _stop(self, msg: str) -> None:
        self.running = False
        if self.timer:
            self.timer.stop()
        self._set_state(f"stopped — {msg}")
        self.notify(msg)

    @on(Button.Pressed, "#auto_stop")
    def stop(self) -> None:
        if self.cycle_dir:
            rlvrctl.cancel(self.cycle_dir)
        self._stop("stopped by user")


# --- app --------------------------------------------------------------------

class OeisTUI(App):
    TITLE = "OEIS RLVR"
    CSS = """
    .panel { width: 80%; max-width: 90; padding: 1 2; border: round $primary; }
    .title { text-style: bold; padding-bottom: 1; }
    .blurb { color: $text-muted; padding-bottom: 1; }
    .menu Button { width: 100%; margin-bottom: 1; }
    .row { height: auto; align: center middle; }
    .row Button { margin: 1 1 0 1; }
    ListView { height: auto; max-height: 20; }
    DataTable { height: auto; }
    TabbedContent { height: 1fr; }
    .formpane { height: 1fr; padding: 0 1; }
    .frow { height: auto; }
    .flabel { width: 24; padding: 1 1 0 0; color: $text-muted; }
    .frow Input { width: 1fr; }
    .monitor { height: 1fr; }
    .stats { padding: 1 0; }
    RichLog { height: 16; border: round $panel; background: $surface; }
    ProgressBar { width: 100%; }
    """

    def on_mount(self) -> None:
        self.push_screen(IntroScreen())


if __name__ == "__main__":
    OeisTUI().run()
