// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

//! One-shot TOML migrator. Runs at load time on the raw `toml::Value`
//! before any typed parsing. Each step is an independent `fn` operating
//! in place on the table — none of them depend on each other's output, so
//! order only matters for readability. `did_migrate` flips true if any
//! step mutated; callers use that to decide whether to rewrite the file
//! and drop the one-shot `.pre-migration.bak` backup.
//!
//! Current steps (see `migrate` below): unify split alias pools, populate
//! per-model `type` tags, drop the `ctx_size=0` and `gpu_layers=0` legacy
//! sentinels, rename `[server].model`/`hf_repo` to `active_model`, and
//! flip the stored `websearch_disabled` bool to the positive-form
//! `websearch`.

use crate::config::infer_model_type;

/// Result of running the migrator. Warnings are surfaced by the caller;
/// we hand them back as strings so the caller can `eprintln!` them once
/// per load rather than have them echoed per call to the migrator itself.
#[derive(Debug, Default)]
pub struct MigrateOutcome {
    pub did_migrate: bool,
    pub warnings: Vec<String>,
}

/// Migrate `table` in place. Idempotent — running on already-migrated input
/// sets `did_migrate = false`.
pub fn migrate(table: &mut toml::value::Table) -> MigrateOutcome {
    let mut outcome = MigrateOutcome::default();

    outcome.did_migrate |= migrate_aliases(table, &mut outcome.warnings);
    outcome.did_migrate |= populate_model_types(table);
    outcome.did_migrate |= drop_zero_ctx_size(table);
    outcome.did_migrate |= drop_zero_gpu_layers(table);
    outcome.did_migrate |= rename_model_identity_to_active_model(table);
    outcome.did_migrate |= flip_websearch_sense(table);

    outcome
}

/// Rewrite `[server].websearch_disabled = X` to `websearch = !X`. Flipping
/// the sense in the stored key matches the registry rename to a positive-form
/// bool with `default(true)`.
fn flip_websearch_sense(table: &mut toml::value::Table) -> bool {
    let Some(server_val) = table.get_mut("server") else {
        return false;
    };
    let Some(server_tbl) = server_val.as_table_mut() else {
        return false;
    };
    let Some(toml::Value::Boolean(disabled)) = server_tbl.remove("websearch_disabled") else {
        return false;
    };
    server_tbl.insert("websearch".to_string(), toml::Value::Boolean(!disabled));
    true
}

/// Merge `[aliases.hf]` and `[aliases.model]` into a unified `[aliases]`.
/// HF wins on collision; one warning per collision.
fn migrate_aliases(table: &mut toml::value::Table, warnings: &mut Vec<String>) -> bool {
    let Some(aliases_val) = table.get("aliases") else {
        return false;
    };
    let Some(aliases_tbl) = aliases_val.as_table() else {
        return false;
    };

    // Detect legacy shape: the `[aliases]` table contains sub-tables `hf`
    // and/or `model`. The new shape is a flat `name = "target"` map, so we
    // bail out if no sub-tables are present.
    let hf_tbl = aliases_tbl.get("hf").and_then(|v| v.as_table()).cloned();
    let model_tbl = aliases_tbl.get("model").and_then(|v| v.as_table()).cloned();
    if hf_tbl.is_none() && model_tbl.is_none() {
        return false;
    }

    // Start with any pre-existing flat entries (defensive: the user may
    // have hand-edited a partial migration).
    let mut merged = toml::value::Table::new();
    for (k, v) in aliases_tbl {
        if k == "hf" || k == "model" {
            continue;
        }
        merged.insert(k.clone(), v.clone());
    }

    // HF wins on collision. Insert HF first, then only fill model entries
    // that don't already exist.
    if let Some(hf) = hf_tbl {
        for (k, v) in hf {
            merged.insert(k, v);
        }
    }
    if let Some(model) = model_tbl {
        for (k, v) in model {
            if merged.contains_key(&k) {
                warnings.push(format!(
                    "lui: alias {:?} existed in both pools; kept HF entry.",
                    k
                ));
                continue;
            }
            merged.insert(k, v);
        }
    }

    table.insert("aliases".to_string(), toml::Value::Table(merged));
    true
}

/// Populate `type = "..."` on every `[models."X"]` entry that lacks it.
/// Inference: path-shaped keys land as `local`, everything else as
/// `huggingface`. Matches `infer_model_type`.
fn populate_model_types(table: &mut toml::value::Table) -> bool {
    let Some(models_val) = table.get_mut("models") else {
        return false;
    };
    let Some(models_tbl) = models_val.as_table_mut() else {
        return false;
    };

    let mut changed = false;
    for (key, entry) in models_tbl.iter_mut() {
        let Some(entry_tbl) = entry.as_table_mut() else {
            continue;
        };
        if entry_tbl.contains_key("type") {
            continue;
        }
        entry_tbl.insert(
            "type".to_string(),
            toml::Value::String(infer_model_type(key).to_string()),
        );
        changed = true;
    }
    changed
}

/// Collapse the legacy `[server].model` / `[server].hf_repo` identity
/// split into `[server].active_model`. HF wins the tiebreaker when both
/// are set (matches the alias-pool convention). Removes the legacy keys
/// after the rename so stale values can't resurface on a later load.
fn rename_model_identity_to_active_model(table: &mut toml::value::Table) -> bool {
    let Some(server_val) = table.get_mut("server") else {
        return false;
    };
    let Some(server_tbl) = server_val.as_table_mut() else {
        return false;
    };
    let already_new_shape = server_tbl.contains_key("active_model");
    let had_legacy = server_tbl.contains_key("hf_repo") || server_tbl.contains_key("model");

    if already_new_shape && !had_legacy {
        return false;
    }

    // Decide the active_model value. Priority: existing active_model
    // (preserve), then hf_repo, then model. Empty strings don't count.
    let active: Option<String> = if already_new_shape {
        match server_tbl.get("active_model") {
            Some(toml::Value::String(s)) if !s.is_empty() => Some(s.clone()),
            _ => None,
        }
    } else {
        None
    };
    let active = active.or_else(|| match server_tbl.get("hf_repo") {
        Some(toml::Value::String(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    });
    let active = active.or_else(|| match server_tbl.get("model") {
        Some(toml::Value::String(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    });

    let removed_any = server_tbl.remove("hf_repo").is_some() | server_tbl.remove("model").is_some();
    if let Some(a) = active {
        server_tbl.insert("active_model".to_string(), toml::Value::String(a));
    }
    removed_any || !already_new_shape && server_tbl.contains_key("active_model")
}

/// Drop `[server].ctx_size = 0` entries. Legacy TOMLs default `ctx_size` to
/// zero to mean "unset"; on the registry side "unset" is encoded as
/// absence, so a stored zero would incorrectly emit `-c 0` on the next run.
fn drop_zero_ctx_size(table: &mut toml::value::Table) -> bool {
    let Some(server_val) = table.get_mut("server") else {
        return false;
    };
    let Some(server_tbl) = server_val.as_table_mut() else {
        return false;
    };
    match server_tbl.get("ctx_size") {
        Some(toml::Value::Integer(0)) => {
            server_tbl.remove("ctx_size");
            true
        }
        _ => false,
    }
}

/// Drop `[server].gpu_layers = 0` entries. Legacy shape treated zero as
/// "skip the -ngl flag entirely"; with the registry default of -1, we
/// now trust any stored value to be intentional. A user who genuinely
/// wants CPU-only can pass `--ngl 0` and the emission will survive.
fn drop_zero_gpu_layers(table: &mut toml::value::Table) -> bool {
    let Some(server_val) = table.get_mut("server") else {
        return false;
    };
    let Some(server_tbl) = server_val.as_table_mut() else {
        return false;
    };
    match server_tbl.get("gpu_layers") {
        Some(toml::Value::Integer(0)) => {
            server_tbl.remove("gpu_layers");
            true
        }
        _ => false,
    }
}
