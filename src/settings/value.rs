// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

//! `Value` is the type-erased payload a `Store` holds for each declared
//! setting. Kinds are declared in the registry (`ValueKind`); actual values
//! are stored here. Everything else — TOML IO, CLI parse, passthrough to
//! llama-server, UI payload — walks the registry and asks the matching
//! `Value` for its typed view.

use std::collections::BTreeMap;

/// The in-memory representation of a setting's value. One variant per
/// `ValueKind`. Stored directly in `Store::values` keyed by `Setting::name`.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
    StringArray(Vec<String>),
    /// Only `chat_template_kwargs` uses this today. Raw toml values are kept
    /// so numeric/bool entries round-trip without a lossy String conversion.
    Map(BTreeMap<String, toml::Value>),
}

/// The declarative counterpart to `Value`: kinds live on `Setting`, values
/// live in `Store`. Registry code uses `ValueKind` to decide how to parse
/// an argv token, validate a TOML load, and render a UI entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueKind {
    Bool,
    Integer,
    Float,
    String,
    StringArray,
    Map,
}

impl ValueKind {
    pub fn name(self) -> &'static str {
        match self {
            ValueKind::Bool => "bool",
            ValueKind::Integer => "integer",
            ValueKind::Float => "float",
            ValueKind::String => "string",
            ValueKind::StringArray => "string array",
            ValueKind::Map => "map",
        }
    }
}

impl Value {
    pub fn as_bool(&self) -> Option<bool> {
        if let Value::Bool(b) = self {
            Some(*b)
        } else {
            None
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        if let Value::Integer(n) = self {
            Some(*n)
        } else {
            None
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Value::String(s) = self {
            Some(s.as_str())
        } else {
            None
        }
    }

    /// Render as a wire-safe string for UI payloads and --cmd / --list
    /// output. Lossless for scalars; maps/arrays are formatted inline.
    pub fn display(&self) -> String {
        match self {
            Value::Bool(b) => b.to_string(),
            Value::Integer(n) => n.to_string(),
            Value::Float(f) => format!("{}", f),
            Value::String(s) => s.clone(),
            Value::StringArray(v) => {
                if v.is_empty() {
                    String::new()
                } else {
                    v.join(" ")
                }
            }
            Value::Map(m) => {
                let mut pieces: Vec<String> = Vec::new();
                for (k, v) in m {
                    pieces.push(format!("{}={}", k, v));
                }
                format!("{{{}}}", pieces.join(", "))
            }
        }
    }
}

/// Convert a toml::Value into a registry Value for load. `kind` tells us
/// what we expect; mismatches return None so the caller can report the
/// offending key with a clear message.
pub fn from_toml(kind: ValueKind, tv: &toml::Value) -> Option<Value> {
    match (kind, tv) {
        (ValueKind::Bool, toml::Value::Boolean(b)) => Some(Value::Bool(*b)),
        (ValueKind::Integer, toml::Value::Integer(n)) => Some(Value::Integer(*n)),
        (ValueKind::Float, toml::Value::Float(f)) => Some(Value::Float(*f)),
        (ValueKind::Float, toml::Value::Integer(n)) => Some(Value::Float(*n as f64)),
        (ValueKind::String, toml::Value::String(s)) => Some(Value::String(s.clone())),
        (ValueKind::StringArray, toml::Value::Array(a)) => {
            let mut out = Vec::with_capacity(a.len());
            for v in a {
                if let toml::Value::String(s) = v {
                    out.push(s.clone());
                } else {
                    return None;
                }
            }
            Some(Value::StringArray(out))
        }
        (ValueKind::Map, toml::Value::Table(t)) => {
            let mut out = BTreeMap::new();
            for (k, v) in t {
                out.insert(k.clone(), v.clone());
            }
            Some(Value::Map(out))
        }
        _ => None,
    }
}

pub fn to_toml(v: &Value) -> toml::Value {
    match v {
        Value::Bool(b) => toml::Value::Boolean(*b),
        Value::Integer(n) => toml::Value::Integer(*n),
        Value::Float(f) => toml::Value::Float(*f),
        Value::String(s) => toml::Value::String(s.clone()),
        Value::StringArray(a) => {
            toml::Value::Array(a.iter().cloned().map(toml::Value::String).collect())
        }
        Value::Map(m) => {
            let mut t = toml::value::Table::new();
            for (k, v) in m {
                t.insert(k.clone(), v.clone());
            }
            toml::Value::Table(t)
        }
    }
}
