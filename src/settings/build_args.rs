// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

//! Registry-driven llama-server argv assembly.
//!
//! Walks the `Registry` once and, for each setting whose `passthrough` is
//! not `None`, contributes zero or more tokens to an argv vector. The
//! always-on lui policy args (`--host`, `--port`, `--metrics`, `--jinja`,
//! `--log-colors off`, `-v`, `-fa on`, `--cache-reuse 256`, `-kvu`) stay
//! hard-coded in `server::build_args` — they are lui policy, not user
//! settings.
//!
//! No setting needs a per-setting emitter anymore; every passthrough
//! setting fits one of the four data-driven modes (FlagValue,
//! BoolFlagIfTrue, LiteralTokens, None).

use super::registry::Registry;
use super::setting::PassthroughMode;
use super::store::Effective;
use super::value::Value;

/// Append every passthrough-contributing setting's forwarded args to `out`.
/// The caller is responsible for the always-on args and for any non-
/// registry emissions (synthesizing `-hf <repo>` / `-m <path>` from the
/// active model's identity).
///
/// The inner loop is the one place that knows about passthrough modes at
/// all — each setting's mode alone decides what tokens flow onto the
/// argv. Adding a new passthrough flag is a single `.passthrough(...)`
/// declaration in the registry; no edits here.
pub fn append_passthrough(eff: &Effective, reg: &Registry, out: &mut Vec<String>) {
    for s in reg.settings() {
        match s.passthrough {
            PassthroughMode::None => {}
            PassthroughMode::FlagValue => {
                let Some(val) = eff.get(s.name) else { continue };
                let Some(flag) = s.llama_flag else { continue };
                out.push(flag.to_string());
                out.push(format_value(val));
            }
            PassthroughMode::BoolFlagIfTrue => {
                if matches!(eff.get(s.name), Some(Value::Bool(true))) {
                    if let Some(flag) = s.llama_flag {
                        out.push(flag.to_string());
                    }
                }
            }
            PassthroughMode::LiteralTokens => {
                // Append-semantics: global entries, then active-model entries.
                // Set by `extra_args` today; any future StringArray setting
                // with this mode gets the same treatment automatically.
                out.extend(eff.merged_string_array(s.name));
            }
        }
    }
}

/// Stringify a `Value` as llama-server would expect it on the argv.
/// Integers and floats get their natural display; strings forward
/// verbatim. Maps are serialised as a JSON object (llama-server's
/// --chat-template-kwargs takes this shape). Bool and StringArray aren't
/// `FlagValue`-valid — the registry should never route them here.
fn format_value(v: &Value) -> String {
    match v {
        Value::Integer(n) => n.to_string(),
        Value::Float(f) => {
            // Match the old `f32.to_string()` output shape (no trailing
            // zeros, no scientific notation for typical sampler ranges).
            format!("{}", f)
        }
        Value::String(s) => s.clone(),
        Value::Map(m) => crate::server::toml_map_to_json_object(m).to_string(),
        Value::Bool(_) | Value::StringArray(_) => String::new(),
    }
}
