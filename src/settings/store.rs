// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

//! Runtime value containers for declared settings. A `Store` is a bag of
//! `Value`s keyed by setting name; `Config` pairs the global store with a
//! per-model map plus the unified alias pool; `Effective` presents the
//! three-layer resolved view (per-model → global → registry default) to
//! downstream consumers.

use std::collections::{BTreeMap, HashMap};

use super::registry::Registry;
use super::value::{self, Value, ValueKind};

/// A bag of setting values keyed by `Setting::name`. Unset settings are
/// absent from the map; "unset" is canonical for "use the registry
/// default if any, else omit downstream".
#[derive(Debug, Default, Clone)]
pub struct Store {
    values: HashMap<String, Value>,
}

impl Store {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.values.contains_key(name)
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.values.get(name)
    }

    pub fn set(&mut self, name: impl Into<String>, v: Value) {
        self.values.insert(name.into(), v);
    }

    pub fn unset(&mut self, name: &str) {
        self.values.remove(name);
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Value)> {
        self.values.iter()
    }

    /// Serialize the store as a toml::Table in canonical (registry) order.
    /// Unknown keys (ones not present in the registry) are dropped.
    pub fn to_toml_table(&self, registry: &Registry) -> toml::value::Table {
        let mut t = toml::value::Table::new();
        for s in registry.settings() {
            if let Some(v) = self.values.get(s.name) {
                t.insert(s.name.to_string(), value::to_toml(v));
            }
        }
        t
    }

    /// Load a store from a toml::Table. Type-mismatched keys are reported
    /// through `warn` (printed to stderr by the caller in production; the
    /// tests capture and assert). Absent keys stay absent; they rely on
    /// the registry default when read through `Effective`.
    pub fn from_toml_table(
        registry: &Registry,
        table: &toml::value::Table,
        mut warn: impl FnMut(String),
    ) -> Self {
        let mut store = Store::new();
        for (k, v) in table {
            let Some(setting) = registry.get(k) else {
                // Unknown key — silently dropped. Ignore rather than error
                // so old or hand-edited keys don't break the load path.
                continue;
            };
            match value::from_toml(setting.kind, v) {
                Some(val) => {
                    store.values.insert(k.clone(), val);
                }
                None => warn(format!(
                    "settings: TOML key {:?} expected {} but got {}; ignoring",
                    k,
                    setting.kind.name(),
                    toml_kind_name(v)
                )),
            }
        }
        store
    }
}

fn toml_kind_name(v: &toml::Value) -> &'static str {
    match v {
        toml::Value::Boolean(_) => "bool",
        toml::Value::Integer(_) => "integer",
        toml::Value::Float(_) => "float",
        toml::Value::String(_) => "string",
        toml::Value::Datetime(_) => "datetime",
        toml::Value::Array(_) => "array",
        toml::Value::Table(_) => "table",
    }
}

/// The full persisted lui config in runtime form: one global store, a
/// per-model map, and the unified alias pool. `load_config` returns one
/// of these; `save_config` takes one.
#[derive(Debug, Default, Clone)]
pub struct Config {
    pub global: Store,
    pub per_model: BTreeMap<String, Store>,
    pub aliases: BTreeMap<String, String>,
}

impl Config {
    pub fn new() -> Self {
        Self::default()
    }

    /// Effective view for a given model key. When `model_key` is None (no
    /// active model, e.g. `lui --list`), per-model layer is skipped.
    pub fn effective<'a>(
        &'a self,
        registry: &'a Registry,
        model_key: Option<&str>,
    ) -> Effective<'a> {
        let per_model = model_key.and_then(|k| self.per_model.get(k));
        Effective {
            registry,
            global: &self.global,
            per_model,
        }
    }
}

/// Precedence-resolved view over the three layers. Read-only.
///
/// Lookup order is: per-model store → global store → registry default.
/// `None` means "no value at any layer"; downstream code should treat this
/// as "omit the flag from llama-server / skip the UI row".
pub struct Effective<'a> {
    pub registry: &'a Registry,
    pub global: &'a Store,
    pub per_model: Option<&'a Store>,
}

impl<'a> Effective<'a> {
    /// Raw `Value` accessor, following the three-layer precedence.
    pub fn get(&self, name: &str) -> Option<&Value> {
        if let Some(pm) = self.per_model {
            if let Some(v) = pm.get(name) {
                return Some(v);
            }
        }
        if let Some(v) = self.global.get(name) {
            return Some(v);
        }
        self.registry.get(name).and_then(|s| s.default.as_ref())
    }

    pub fn get_bool(&self, name: &str) -> Option<bool> {
        self.get(name).and_then(Value::as_bool)
    }
    pub fn get_i64(&self, name: &str) -> Option<i64> {
        self.get(name).and_then(Value::as_i64)
    }
    pub fn get_string(&self, name: &str) -> Option<&str> {
        self.get(name).and_then(Value::as_str)
    }
    /// Append-semantics accessor for `StringArray` settings: concatenate
    /// the global layer's entries with the per-model layer's entries. For
    /// scalar settings the regular precedence rule (`get`) applies; this
    /// path is only relevant for arrays where "additive" is the desired
    /// merge rather than "last layer wins".
    pub fn merged_string_array(&self, name: &str) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        if let Some(Value::StringArray(v)) = self.global.get(name) {
            out.extend(v.iter().cloned());
        }
        if let Some(pm) = self.per_model {
            if let Some(Value::StringArray(v)) = pm.get(name) {
                out.extend(v.iter().cloned());
            }
        }
        out
    }

    /// True iff any layer has explicitly set this — useful for UI dim styling
    /// (distinguishes "showing the default" from "user set this to the default").
    pub fn is_explicitly_set(&self, name: &str) -> bool {
        if let Some(pm) = self.per_model {
            if pm.contains(name) {
                return true;
            }
        }
        self.global.contains(name)
    }
}

/// Clamp a parsed integer to the setting's min/max. Returns Err(reason)
/// so the caller can format `"--flag: value N out of range"` consistently.
pub fn validate_integer(
    registry: &Registry,
    name: &str,
    n: i64,
) -> Result<i64, String> {
    let Some(s) = registry.get(name) else {
        return Ok(n);
    };
    if s.kind != ValueKind::Integer {
        return Ok(n);
    }
    if let Some(min) = s.min {
        if n < min {
            return Err(format!("{} below minimum {}", n, min));
        }
    }
    if let Some(max) = s.max {
        if n > max {
            return Err(format!("{} above maximum {}", n, max));
        }
    }
    Ok(n)
}
