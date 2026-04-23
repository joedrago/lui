// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

//! opencode harness: writes `~/.config/opencode/opencode.{jsonc,json}` and
//! manages the `permission.bash` curl allowlist for lui's web-search port.
//! Respects `harness_opencode_disable_prune` when set.

use jsonc_parser::cst::{CstInputValue, CstObject, CstObjectProp, CstRootNode};
use jsonc_parser::ParseOptions;
use regex::Regex;

use super::{arr, b, n, obj, s, ConfigFile, Harness, HarnessInputs};
use crate::settings::store::Effective;
use crate::ssh_tunnel::{ssh_run, SshTarget};

pub const HARNESS: Harness = Harness {
    name: "opencode",
    setting_name: "harness_opencode",
    flag_long: "harness-opencode",
    default_on: true,
    help: &[
        "Manage opencode.{jsonc,json} and the lui-web-search skill",
        "Leave opencode's config alone (for external harnesses)",
    ],
    config: ConfigFile {
        dir: ".config/opencode",
        candidates: &["opencode.jsonc", "opencode.json"],
    },
    apply,
    needs_backup,
    preflight_ssh: Some(preflight_ssh),
};

fn apply(root: &CstObject, eff: &Effective, inputs: &HarnessInputs) {
    set_provider_lui(root, &inputs.model_name, &inputs.base_url, inputs.ctx_size);
    set_permission_bash(root, inputs.web_port, inputs.websearch);
    if eff
        .get_bool("harness_opencode_disable_prune")
        .unwrap_or(false)
    {
        set_compaction_prune_false_if_absent(root);
    }
}

/// True iff the text has content but no `provider.lui` entry. Decides
/// whether to drop a `.luibackup` before writing. Empty/absent → false
/// (nothing to back up). Unparseable → true (preserve it).
fn needs_backup(existing: &str) -> bool {
    if existing.trim().is_empty() {
        return false;
    }
    let Ok(root) = CstRootNode::parse(existing, &ParseOptions::default()) else {
        return true;
    };
    let Some(root_obj) = root.object_value() else {
        return true;
    };
    let Some(provider) = root_obj.object_value("provider") else {
        return true;
    };
    provider.get("lui").is_none()
}

/// Verify `opencode` is installed on the remote before we start editing
/// its config. Non-interactive SSH runs a non-login shell, so PATH edits
/// the opencode installer makes in `~/.bashrc` / `~/.zshrc` aren't
/// sourced. We OR three checks together so any single success suffices:
/// `command -v opencode` (default PATH), `bash -lc 'command -v opencode'`
/// (login-shell PATH — picks up `~/.bash_profile` / `~/.profile` edits),
/// and the installer's canonical location `~/.opencode/bin/opencode`.
/// Any one succeeding means opencode is usable once the user SSHes in.
fn preflight_ssh(target: &SshTarget) -> Result<(), String> {
    let probe = "command -v opencode \
        || bash -lc 'command -v opencode' \
        || { [ -x \"$HOME/.opencode/bin/opencode\" ] && echo \"$HOME/.opencode/bin/opencode\"; }";
    match ssh_run(target, &[probe], None) {
        Ok(out) if !out.trim().is_empty() => Ok(()),
        Ok(_) | Err(_) => Err(format!(
            "opencode not found on {}. Install it there first.",
            target.spec()
        )),
    }
}

fn set_provider_lui(root_obj: &CstObject, model_name: &str, llama_base_url: &str, ctx_size: u32) {
    let provider = root_obj.object_value_or_set("provider");

    let model_entry = CstInputValue::Object(vec![(
        model_name.to_string(),
        obj([
            ("name", s(model_name)),
            ("supportsToolCalls", b(true)),
            (
                "limit",
                obj([
                    ("context", n(ctx_size)),
                    ("input", n(ctx_size)),
                    ("output", n(8192)),
                ]),
            ),
        ]),
    )]);

    let lui_value = obj([
        ("name", s("lui")),
        ("npm", s("@ai-sdk/openai-compatible")),
        (
            "options",
            obj([
                ("baseURL", s(llama_base_url)),
                (
                    "toolParser",
                    arr(vec![
                        obj([("type", s("raw-function-call"))]),
                        obj([("type", s("json"))]),
                    ]),
                ),
            ]),
        ),
        ("models", model_entry),
    ]);

    match provider.get("lui") {
        Some(prop) => prop.set_value(lui_value),
        None => {
            provider.append("lui", lui_value);
        }
    }
}

/// Manage `permission.bash["curl*http://127.0.0.1:<port>/*"]` so opencode
/// doesn't prompt for every search curl. Cleans stale port wildcards so a
/// `--web-port` change doesn't leave a dead allowlist entry behind. When
/// websearch is disabled we refuse to create parent objects (`permission`
/// / `bash`) just to insert nothing.
fn set_permission_bash(root_obj: &CstObject, web_port: u16, websearch: bool) {
    let current_pattern = format!("curl*http://127.0.0.1:{}/*", web_port);

    let has_permission = root_obj.get("permission").is_some();
    if !websearch && !has_permission {
        return;
    }
    let permission = root_obj.object_value_or_set("permission");

    let has_bash = permission.get("bash").is_some();
    if !websearch && !has_bash {
        return;
    }
    let bash = permission.object_value_or_set("bash");

    let stale_re = Regex::new(r"^curl\*http://127\.0\.0\.1:\d+/\*$").unwrap();
    let to_remove: Vec<CstObjectProp> = bash
        .properties()
        .into_iter()
        .filter(|p| {
            let Some(name) = p.name() else { return false };
            let Ok(key) = name.decoded_value() else {
                return false;
            };
            stale_re.is_match(&key) && key != current_pattern
        })
        .collect();
    for prop in to_remove {
        prop.remove();
    }

    if !websearch {
        if let Some(prop) = bash.get(&current_pattern) {
            prop.remove();
        }
    } else {
        match bash.get(&current_pattern) {
            Some(prop) => prop.set_value(s("allow")),
            None => {
                bash.append(&current_pattern, s("allow"));
            }
        }
    }
}

/// Set `compaction.prune = false` only if the key is absent — never
/// clobbers a user-set value. Creates the parent `compaction` object
/// only if missing.
fn set_compaction_prune_false_if_absent(root_obj: &CstObject) {
    let compaction = root_obj.object_value_or_set("compaction");
    if compaction.get("prune").is_none() {
        compaction.append("prune", b(false));
    }
}
