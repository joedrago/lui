// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

//! Value formatters referenced by `Setting::ui_format`.
//!
//! Each formatter is a free function the registry can wire onto a
//! setting via `.ui_format(ui::format_...)`. They handle the few
//! settings whose display diverges from the generic "stringify the
//! value, or show `ui_unset` when absent" path — unit suffixes, bare /
//! three-state bools, cross-setting fallback arithmetic, and count-
//! aggregate rows.
//!
//! A formatter returns `Option<String>`:
//!   - `None` → row is skipped in grouped-line contexts; single-row
//!     contexts render the setting's `ui_unset` phrase instead.
//!   - `Some("")` → render the label only, no `=value` suffix.
//!   - `Some(s)` → render `label=s`.

use super::setting::Setting;
use super::store::Effective;
use super::value::Value;

/// Integer values, rendered with a trailing ` MiB` unit. Used by
/// memory-sized settings whose raw value is ambiguous without the unit.
pub fn format_mib(value: Option<&Value>, _eff: &Effective, _s: &Setting) -> Option<String> {
    match value {
        Some(Value::Integer(n)) => Some(format!("{} MiB", n)),
        _ => None,
    }
}

/// Integer "count of layers" renderer — `-1` maps to the string
/// `"all"`, everything else to the plain integer.
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

/// Context-size renderer — `0` is the "unset" sentinel and maps to
/// `None` so the caller falls back to the setting's `ui_unset` phrase.
pub fn format_nonzero_int(
    value: Option<&Value>,
    _eff: &Effective,
    _s: &Setting,
) -> Option<String> {
    match value {
        Some(Value::Integer(0)) | None => None,
        Some(Value::Integer(n)) => Some(n.to_string()),
        _ => None,
    }
}

/// Presence-only bool with an off-form: `true` → label alone (no
/// `=value` suffix), `false` → `label=off`. Matches llama-server's
/// `--swa-full` surface where mere presence flips it on.
pub fn format_bare_or_off(
    value: Option<&Value>,
    _eff: &Effective,
    _s: &Setting,
) -> Option<String> {
    match value {
        Some(Value::Bool(true)) => Some(String::new()),
        Some(Value::Bool(false)) => Some("off".to_string()),
        _ => None,
    }
}

/// Aggregate row for `StringArray` settings whose UI surface is a
/// count, not the individual tokens: renders as `"+N extra"` using the
/// setting's label for the `extra` word. Returns `None` for empty
/// arrays so the row is omitted entirely.
///
/// The full extras list (global + per-model, concatenated) is what
/// drives the count — same rule `build_args` uses.
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
