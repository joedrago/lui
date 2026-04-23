// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

//! `Setting` is the declarative record that describes one CLI/TOML option.
//! Built with fluent chaining in `registry::declare_all_settings`. Once the
//! registry is built, `Setting` is read-only — parsing, help, TOML IO,
//! build_args, and the UI payload all walk `Registry` and ask each `Setting`
//! for its metadata.
//!
//! Adjacent function calls per entry are the intended authoring style (see
//! `registry.rs`); that's why every "setter" returns `Self` by value.

use super::store::Effective;
use super::value::{Value, ValueKind};

/// Where in the TOML a setting is stored, and who is allowed to set it.
///
/// - `Global`: lives in `[server]` only. `--this --port 9` is rejected at
///   parse time.
/// - `PerModel`: lives in `[models."X"]` only. Used by the (future) `type`
///   field that pins a model as huggingface vs local.
/// - `Both`: honors the sticky `--this`/`--global` cursor — writes to either
///   section depending on scope. Effective value is per-model-over-global.
/// - `Ephemeral`: not persisted; lives only for the duration of one
///   invocation. Mode flags (`--list`, `--cmd`, `--ssh`, ...) use this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    Global,
    PerModel,
    Both,
    Ephemeral,
}

/// How a setting contributes to the llama-server command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassthroughMode {
    /// lui handles this value itself; it is not forwarded.
    None,
    /// Forward as `<llama_flag> <value>`. Skipped when the effective value
    /// is not set (neither store has a value and no registry default).
    /// Settings that should always reach llama-server declare a registry
    /// default so the value is always present and thus always emitted.
    FlagValue,
    /// Bool setting: emit the bare `<llama_flag>` iff value is true. Used
    /// for presence-only flags that llama-server flips on by name alone.
    BoolFlagIfTrue,
    /// String-array contents appended literally (no `--long` prefix) to
    /// llama-server argv. Used for post-`--` passthrough tokens.
    LiteralTokens,
}

/// One declared setting. Construct via `Setting::new(name)` then chain the
/// fluent setters; `Registry` stores an immutable `Vec<Setting>`.
#[derive(Debug, Clone)]
pub struct Setting {
    /// Canonical key — matches the TOML key and the store's HashMap key.
    /// Snake_case to match Rust/serde convention. Must be unique.
    pub name: &'static str,

    pub kind: ValueKind,

    /// Short flag, e.g. `'c'` for `-c`. None for long-only flags.
    pub short: Option<char>,
    /// Primary long flag body without leading `--`, e.g. `"ctx-size"`.
    /// None for settings that only exist for TOML persistence (the
    /// identity keys `active_model` / `type`, say).
    pub long: Option<&'static str>,
    /// Additional long forms that should resolve to this setting. Parser
    /// walks these alongside `long`. Empty slice when there are none.
    pub long_aliases: &'static [&'static str],
    /// Placeholder shown after the flag in help output (`--temp <F>`).
    /// Empty for zero-arg flags; ignored for Bool with BoolFlagIfTrue.
    pub placeholder: &'static str,

    pub scope: Scope,

    /// Registry-level default — used when neither the per-model nor the
    /// global store has a value for this setting. `None` means "unset is
    /// meaningful" (e.g. `temp` without any override passes `--temp` to
    /// llama-server = not at all).
    pub default: Option<Value>,

    /// Help lines. Emitted one-per-line with auto-aligned continuation.
    pub help: &'static [&'static str],

    /// Top-level grouping for `--help` output, e.g. "MODEL", "SETTINGS",
    /// "MACHINE", "REMOTE", "OTHER". Stable identifiers — the help printer
    /// consults a fixed order to keep section ordering deterministic.
    pub section: &'static str,

    /// Visual group inside the UI payload, e.g. "sampling", "tuning".
    /// `None` means "don't surface in the UI's grouped sections" (either a
    /// hidden setting or one the UI shows in its own typed blocks). The
    /// TUI renderer filters `settings` by this label at draw time.
    pub group: Option<&'static str>,

    /// Minimum allowed value for `ValueKind::Integer`. Enforced at parse
    /// time and TOML-load time. Ignored for other kinds.
    pub min: Option<i64>,
    pub max: Option<i64>,

    /// The flag passed to llama-server for `PassthroughMode::*`. Some lui
    /// names differ from llama-server's (we say `ctx_size`, llama-server
    /// takes `-c`); `llama_flag` carries that mapping.
    pub llama_flag: Option<&'static str>,
    pub passthrough: PassthroughMode,

    /// Bool-only: when true (default), the parser also accepts `--no-<long>`
    /// as the inverted form. Flip to false for bools that don't have a
    /// meaningful negation (e.g. mode flags like `--list`).
    pub no_form: bool,

    /// Machine-parseable tag for downstream consumers that need to mark a
    /// setting without extending the main type. Today it only carries the
    /// two `type_flag_*` markers the parser uses for the `-m` / `--hf`
    /// zero-arg type flags.
    pub tag: Option<&'static str>,

    /// Human-friendly label for UI rendering. Falls back to the long flag
    /// (with `-` replaced by space, first letter uppercased) when unset.
    pub ui_label: Option<&'static str>,

    /// Custom formatter for UI display. Receives the effective value (or
    /// `None` if absent) plus the full `Effective` + this setting. Returns:
    ///   - `None` → skip this row entirely (group-rendered contexts) or
    ///     render the `ui_unset` phrase instead (single-row contexts).
    ///   - `Some("")` → label only, no `=value` suffix.
    ///   - `Some(s)` → `label=s`.
    pub ui_format: Option<UiFormatter>,

    /// Phrase rendered in the "Current config" list when the effective
    /// value is absent (or the formatter returns `None`). Typical values:
    /// `"model default"`, `"server default"`, `"auto"`, `"normal"`.
    pub ui_unset: Option<&'static str>,
}

/// Formatter hook for UI rendering of a setting's value. See
/// `Setting::ui_format`.
pub type UiFormatter =
    fn(value: Option<&Value>, eff: &Effective, setting: &Setting) -> Option<String>;

impl Setting {
    pub fn new(name: &'static str) -> Self {
        Setting {
            name,
            kind: ValueKind::Bool,
            short: None,
            long: None,
            long_aliases: &[],
            placeholder: "",
            scope: Scope::Global,
            default: None,
            help: &[],
            section: "OTHER",
            group: None,
            min: None,
            max: None,
            llama_flag: None,
            passthrough: PassthroughMode::None,
            no_form: true,
            tag: None,
            ui_label: None,
            ui_format: None,
            ui_unset: None,
        }
    }

    pub fn kind(mut self, k: ValueKind) -> Self {
        self.kind = k;
        // Bool settings without a declared negation form default to keeping
        // the negation. Non-bools don't care about `no_form`.
        if k != ValueKind::Bool {
            self.no_form = false;
        }
        self
    }
    pub fn short(mut self, c: char) -> Self {
        self.short = Some(c);
        self
    }
    pub fn long(mut self, s: &'static str) -> Self {
        self.long = Some(s);
        self
    }
    pub fn long_aliases(mut self, ss: &'static [&'static str]) -> Self {
        self.long_aliases = ss;
        self
    }
    pub fn placeholder(mut self, s: &'static str) -> Self {
        self.placeholder = s;
        self
    }
    pub fn scope(mut self, s: Scope) -> Self {
        self.scope = s;
        self
    }
    pub fn default(mut self, v: Value) -> Self {
        self.default = Some(v);
        self
    }
    pub fn help(mut self, lines: &'static [&'static str]) -> Self {
        self.help = lines;
        self
    }
    pub fn section(mut self, s: &'static str) -> Self {
        self.section = s;
        self
    }
    pub fn group(mut self, g: &'static str) -> Self {
        self.group = Some(g);
        self
    }
    pub fn min(mut self, n: i64) -> Self {
        self.min = Some(n);
        self
    }
    pub fn max(mut self, n: i64) -> Self {
        self.max = Some(n);
        self
    }
    pub fn llama_flag(mut self, s: &'static str) -> Self {
        self.llama_flag = Some(s);
        self
    }
    pub fn passthrough(mut self, p: PassthroughMode) -> Self {
        self.passthrough = p;
        self
    }
    pub fn no_form(mut self, b: bool) -> Self {
        self.no_form = b;
        self
    }
    pub fn tag(mut self, t: &'static str) -> Self {
        self.tag = Some(t);
        self
    }
    pub fn ui_label(mut self, s: &'static str) -> Self {
        self.ui_label = Some(s);
        self
    }
    pub fn ui_format(mut self, f: UiFormatter) -> Self {
        self.ui_format = Some(f);
        self
    }
    pub fn ui_unset(mut self, s: &'static str) -> Self {
        self.ui_unset = Some(s);
        self
    }

    /// Derived label — prefer the explicit `ui_label`, else derive from
    /// the long flag with `-` → space and first-letter uppercase. Falls
    /// back to the setting's canonical name for entries without a long
    /// flag (storage-only settings that still show up in the per-model
    /// override view).
    pub fn derived_ui_label(&self) -> String {
        if let Some(l) = self.ui_label {
            return l.to_string();
        }
        let source = self.long.unwrap_or(self.name);
        let spaced = source.replace('-', " ").replace('_', " ");
        let mut chars = spaced.chars();
        match chars.next() {
            Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
            None => String::new(),
        }
    }

    /// Does this setting take a value on the command line? Bool flags and
    /// zero-arg ephemerals (`--list`, `--public`) don't.
    pub fn takes_value(&self) -> bool {
        !matches!(self.kind, ValueKind::Bool)
    }

    /// True iff `--no-<long>` should be accepted as an inverted bool.
    pub fn has_no_form(&self) -> bool {
        self.kind == ValueKind::Bool && self.no_form && self.long.is_some()
    }
}

// --- UI value formatters -------------------------------------------------
//
// Each fn below implements `UiFormatter` — wired onto a setting via
// `.ui_format(format_...)` in the registry. They handle the settings whose
// display diverges from the generic "stringify the value, or show
// `ui_unset` when absent" path: unit suffixes, bare/off bools, zero-as-
// unset sentinels, and count-aggregate rows.
//
// Return convention:
//   - `None`     → row is skipped in grouped-line contexts; single-row
//                  contexts render the setting's `ui_unset` phrase.
//   - `Some("")` → render the label only, no `=value` suffix.
//   - `Some(s)`  → render `label=s`.

/// Integer values, rendered with a trailing ` MiB` unit. Used by
/// memory-sized settings whose raw value is ambiguous without the unit.
pub fn format_mib(value: Option<&Value>, _eff: &Effective, _s: &Setting) -> Option<String> {
    match value {
        Some(Value::Integer(n)) => Some(format!("{} MiB", n)),
        _ => None,
    }
}

/// Integer "count of layers" renderer — `-1` maps to the string `"all"`,
/// everything else to the plain integer.
pub fn format_negative_as_all(
    value: Option<&Value>,
    _eff: &Effective,
    _s: &Setting,
) -> Option<String> {
    match value {
        Some(Value::Integer(-1)) => Some("all".to_string()),
        Some(Value::Integer(n)) => Some(n.to_string()),
        _ => None,
    }
}

/// Context-size renderer — `0` is the "unset" sentinel and maps to `None`
/// so the caller falls back to the setting's `ui_unset` phrase.
pub fn format_nonzero_int(value: Option<&Value>, _eff: &Effective, _s: &Setting) -> Option<String> {
    match value {
        Some(Value::Integer(0)) | None => None,
        Some(Value::Integer(n)) => Some(n.to_string()),
        _ => None,
    }
}

/// Presence-only bool with an off-form: `true` → label alone (no `=value`
/// suffix), `false` → `label=off`. Matches llama-server's `--swa-full`
/// surface where mere presence flips it on.
pub fn format_bare_or_off(value: Option<&Value>, _eff: &Effective, _s: &Setting) -> Option<String> {
    match value {
        Some(Value::Bool(true)) => Some(String::new()),
        Some(Value::Bool(false)) => Some("off".to_string()),
        _ => None,
    }
}

/// Aggregate row for `StringArray` settings whose UI surface is a count,
/// not the individual tokens: renders as `"+N <label>"`. Returns `None`
/// for empty arrays so the row is omitted entirely. The full extras list
/// (global + per-model, concatenated) drives the count — same rule
/// `build_args` uses.
pub fn format_count_aggregate(
    _value: Option<&Value>,
    eff: &Effective,
    setting: &Setting,
) -> Option<String> {
    let n = eff.merged_string_array(setting.name).len();
    if n == 0 {
        None
    } else {
        Some(format!("+{} {}", n, setting.ui_label.unwrap_or("items")))
    }
}
