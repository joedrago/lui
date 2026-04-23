// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

use std::path::{Path, PathBuf};

use crate::settings::migrate;
use crate::settings::registry::Registry;
use crate::settings::store::{Config, Store};
use crate::settings::value::Value;

/// Infer a model's type from its key. Path-shaped keys (contains '/' AND
/// starts with '/', './', '~', OR ends in '.gguf') are `"local"`;
/// everything else is `"huggingface"`. Used by the migrator to backfill
/// per-model `type` entries and by display / save-path logic that picks
/// between `-m` and `--hf` for a model that hasn't had its tag recorded
/// yet.
pub fn infer_model_type(key: &str) -> &'static str {
    let looks_like_path = key.contains('/')
        && (key.starts_with('/')
            || key.starts_with("./")
            || key.starts_with('~')
            || key.ends_with(".gguf"));
    if looks_like_path {
        "local"
    } else {
        "huggingface"
    }
}

/// Resolve the port the local websearch HTTP server binds to.
/// Defaults to `llama_port + 1` unless the user set `--web-port`.
pub fn websearch_port(eff: &crate::settings::store::Effective) -> u16 {
    if let Some(wp) = eff.get_i64("web_port") {
        return wp as u16;
    }
    let port = eff.get_i64("port").unwrap_or(8080) as u16;
    port.saturating_add(1)
}

/// Render the "source" line shown under the Model KV: `--hf ORG/REPO`,
/// `-m /path/to.gguf`, or `none`. Lives here (not in display.rs) so the
/// server can pre-format it into the UiSnapshot and any renderer — local or
/// client — shows an identical line.
pub fn format_source(eff: &crate::settings::store::Effective) -> String {
    let Some(active) = active_model_key(eff) else {
        return "none".to_string();
    };
    // Type from the per-model store ("huggingface" or "local"). Fall back
    // to shape-based inference if the entry's type didn't land (new
    // invocations that haven't been saved yet, etc.).
    let ty = eff
        .per_model
        .and_then(|s| match s.get("type") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_else(|| infer_model_type(&active).to_string());
    match ty.as_str() {
        "local" => format!("-m {}", active),
        _ => format!("--hf {}", active),
    }
}

pub fn config_path() -> PathBuf {
    // XDG-style path (~/.config/lui.toml on all platforms). On macOS this
    // is not the system "config_dir" (~/Library/Application Support/) but
    // it's what most CLI tools actually use and what users expect.
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".config").join("lui.toml")
}

/// Path written once when the migrator fires on an old-shape file. Sits
/// next to lui.toml so a user who hit "I regret this" has a one-step
/// rollback right there in the config directory.
fn pre_migration_backup_path(path: &Path) -> PathBuf {
    let mut os = path.as_os_str().to_os_string();
    os.push(".pre-migration.bak");
    PathBuf::from(os)
}

pub fn load_config() -> Config {
    load_config_from(&config_path())
}

/// Testable variant of `load_config`. Reads from `path`, runs the migrator,
/// writes `<path>.pre-migration.bak` on first migration, and rewrites `path`
/// in the new canonical shape so the next load is a no-op.
pub fn load_config_from(path: &Path) -> Config {
    if !path.exists() {
        return Config::new();
    }

    let contents = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Warning: failed to read {}: {}", path.display(), e);
            return Config::new();
        }
    };

    let mut root_table: toml::value::Table = match contents.parse::<toml::Value>() {
        Ok(toml::Value::Table(t)) => t,
        Ok(_) => {
            eprintln!("Warning: {} is not a toml table at the root", path.display());
            return Config::new();
        }
        Err(e) => {
            eprintln!("Warning: failed to parse {}: {}", path.display(), e);
            return Config::new();
        }
    };

    // One-shot migrator runs on the raw `toml::Value` before the Store
    // parser sees it. If anything actually changed, drop a one-time backup
    // next to lui.toml and rewrite the file in the canonical shape below.
    let outcome = migrate::migrate(&mut root_table);
    if outcome.did_migrate {
        let backup = pre_migration_backup_path(path);
        if !backup.exists() {
            if let Err(e) = std::fs::write(&backup, &contents) {
                eprintln!(
                    "Warning: failed to write {}: {}",
                    backup.display(),
                    e
                );
            }
        }
        for w in &outcome.warnings {
            eprintln!("{}", w);
        }
    }

    let config = config_from_table(&root_table);

    if outcome.did_migrate {
        save_config_to(&config, path);
    }

    config
}

/// Build a `Config` view straight from a migrated `toml::Value::Table`.
/// Every section flows through `Store::from_toml_table` so registry
/// declarations are the single source of truth for key names and kinds.
fn config_from_table(table: &toml::value::Table) -> Config {
    let reg = Registry::build();
    let mut config = Config::new();

    if let Some(t) = table.get("server").and_then(|v| v.as_table()) {
        config.global = Store::from_toml_table(&reg, t, |w| eprintln!("lui: {}", w));
    }

    if let Some(models_tbl) = table.get("models").and_then(|v| v.as_table()) {
        for (key, entry_val) in models_tbl {
            let Some(entry_tbl) = entry_val.as_table() else {
                continue;
            };
            let per_model_store =
                Store::from_toml_table(&reg, entry_tbl, |w| eprintln!("lui: {}", w));
            config.per_model.insert(key.clone(), per_model_store);
        }
    }

    if let Some(aliases_tbl) = table.get("aliases").and_then(|v| v.as_table()) {
        for (name, target_val) in aliases_tbl {
            if let Some(target) = target_val.as_str() {
                config.aliases.insert(name.clone(), target.to_string());
            }
        }
    }

    config
}

pub fn save_config(config: &Config) {
    save_config_to(config, &config_path())
}

/// Testable variant of `save_config`. Writes `config` to `path` in the
/// canonical shape — creating the parent directory if needed. Every key
/// name and serialization rule is driven by the registry via
/// `Store::to_toml_table`; this function only handles section ordering
/// and the type-tag backfill for new history entries.
pub fn save_config_to(config: &Config, path: &Path) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let reg = Registry::build();
    let mut config = config.clone();

    // Make sure the active model has an entry in [models."X"]. This turns
    // the tail of the toml into a history of every model the user has
    // tried; users can delete stale ones by hand.
    if let Some(Value::String(active)) = config.global.get("active_model").cloned() {
        if !active.is_empty() {
            let entry = config.per_model.entry(active.clone()).or_default();
            if !entry.contains("type") {
                entry.set(
                    "type",
                    Value::String(infer_model_type(&active).to_string()),
                );
            }
        }
    }
    // Belt-and-suspenders: any per-model entry without a `type` tag gets
    // one inferred from its key. The migrator fills this on load, so this
    // path only matters for in-memory configs that never touched disk.
    let keys: Vec<String> = config.per_model.keys().cloned().collect();
    for k in keys {
        let needs_type = config
            .per_model
            .get(&k)
            .map(|s| !s.contains("type"))
            .unwrap_or(false);
        if needs_type {
            let tag = infer_model_type(&k).to_string();
            if let Some(entry) = config.per_model.get_mut(&k) {
                entry.set("type", Value::String(tag));
            }
        }
    }

    // Section ordering: [server] → [aliases] → [models.*] with real
    // overrides before history-only (type-tag-only) entries, so a user
    // scanning the file sees the interesting stuff at the top.
    let mut output = String::new();

    let server_tbl = config.global.to_toml_table(&reg);
    output.push_str(&emit_section("server", &server_tbl));

    if !config.aliases.is_empty() {
        let mut at = toml::value::Table::new();
        for (k, v) in &config.aliases {
            at.insert(k.clone(), toml::Value::String(v.clone()));
        }
        output.push('\n');
        output.push_str(&emit_section("aliases", &at));
    }

    let mut has_overrides: Vec<&String> = Vec::new();
    let mut history_only: Vec<&String> = Vec::new();
    for (k, store) in &config.per_model {
        if store.iter().any(|(name, _)| name != "type") {
            has_overrides.push(k);
        } else {
            history_only.push(k);
        }
    }
    for k in has_overrides.iter().chain(history_only.iter()) {
        let Some(store) = config.per_model.get(*k) else {
            continue;
        };
        let entry = store.to_toml_table(&reg);
        output.push('\n');
        output.push_str(&emit_model_section(k, &entry));
    }

    if let Err(e) = std::fs::write(path, output) {
        eprintln!("Warning: failed to write {}: {}", path.display(), e);
    }
}

/// Render a top-level `[name]` section with its body. Empty bodies still
/// emit the bare header line, so consumers relying on the presence of a
/// section can rely on it.
fn emit_section(name: &str, body: &toml::value::Table) -> String {
    if body.is_empty() {
        return format!("[{}]\n", name);
    }
    let mut root = toml::value::Table::new();
    root.insert(name.to_string(), toml::Value::Table(body.clone()));
    match toml::to_string_pretty(&toml::Value::Table(root)) {
        Ok(s) => {
            let mut out = s.trim_end_matches('\n').to_string();
            out.push('\n');
            out
        }
        Err(e) => {
            eprintln!("Warning: failed to serialize [{}]: {}", name, e);
            format!("[{}]\n", name)
        }
    }
}

/// Render one `[models."X"]` section. Delegated to `toml::to_string_pretty`
/// so the toml crate handles key quoting for us — keys with slashes,
/// colons, or other non-bare characters get wrapped in `"..."` correctly.
fn emit_model_section(key: &str, entry: &toml::value::Table) -> String {
    let mut models = toml::value::Table::new();
    models.insert(key.to_string(), toml::Value::Table(entry.clone()));
    let mut root = toml::value::Table::new();
    root.insert("models".to_string(), toml::Value::Table(models));
    match toml::to_string_pretty(&toml::Value::Table(root)) {
        Ok(s) => {
            let mut out = s.trim_end_matches('\n').to_string();
            out.push('\n');
            out
        }
        Err(e) => {
            eprintln!(
                "Warning: failed to serialize [models.{:?}]: {}",
                key, e
            );
            String::new()
        }
    }
}


/// The identity string used to key per-model overrides. Deliberately the
/// EXACT text the user typed after `--hf` (or `-m`) — including org prefix
/// and quantization — so e.g. `unsloth/Foo:Q4_K_M` and `unsloth/Foo:Q8_0`
/// get separate override entries, and `unsloth/Bar` vs `zai-org/Bar` don't
/// collide. Distinct from `derive_model_name`, which produces the short
/// cosmetic name used for opencode's provider/model ID.
pub fn model_key(config: &Config) -> Option<String> {
    match config.global.get("active_model") {
        Some(Value::String(v)) if !v.is_empty() => Some(v.clone()),
        _ => None,
    }
}

/// Derive a short model name for opencode from the active model key.
/// HF-shape keys compress as "Org/Name-GGUF:Q4_K_M" -> "Name"; local paths
/// drop the extension. Falls back to "unknown" when there's no active
/// model at all.
pub fn derive_model_name(eff: &crate::settings::store::Effective) -> String {
    let Some(active) = active_model_key(eff) else {
        return "unknown".to_string();
    };
    // Path-shaped (contains slash + is path-like) → local; shape-infer
    // avoids having to plumb per-model `type` down here.
    if is_path_shaped(&active) {
        return PathBuf::from(&active)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
    }
    let name = active.split('/').next_back().unwrap_or(&active);
    let name = name.split(':').next().unwrap_or(name);
    let name = name.strip_suffix("-GGUF").unwrap_or(name);
    name.to_string()
}

/// Active-model key as stored in `[server].active_model`. Returns `None`
/// when no model is pinned at all.
fn active_model_key(eff: &crate::settings::store::Effective) -> Option<String> {
    eff.get_string("active_model")
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

fn is_path_shaped(s: &str) -> bool {
    s.contains('/')
        && (s.starts_with('/') || s.starts_with("./") || s.starts_with('~') || s.ends_with(".gguf"))
}

