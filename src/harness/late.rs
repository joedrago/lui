// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

//! late-cli harness: writes `~/.config/late/config.json` (Linux) or
//! `~/Library/Application Support/late/config.json` (macOS) and manages
//! the lui-web-search skill in late's skills directory.
//!
//! late-cli uses `os.UserConfigDir()` which resolves to:
//!   - Linux: `~/.config/late`
//!   - macOS: `~/Library/Application Support/late`

use jsonc_parser::cst::CstObject;

use super::{s, ConfigFile, Harness, HarnessInputs};
use crate::settings::store::Effective;
use crate::ssh_tunnel::{ssh_run, SshTarget};

pub const HARNESS: Harness = Harness {
    name: "late",
    setting_name: "harness_late",
    flag_long: "harness-late",
    default_on: false,
    help: &[
        "Manage late-cli config.json and the lui-web-search skill",
        "Leave late-cli's config alone (default)",
    ],
    config: ConfigFile {
        dir: ".config/late",
        dir_macos: Some("Library/Application Support/late"),
        candidates: &["config.json"],
    },
    apply,
    needs_backup,
    preflight_ssh: Some(preflight_ssh),
};

fn apply(root: &CstObject, _eff: &Effective, inputs: &HarnessInputs) {
    set_openai_settings(root, &inputs.model_name, &inputs.base_url);
}

/// True for any non-empty existing file. late-cli has no lui-specific
/// namespace, so we preserve any pre-existing config on first touch.
fn needs_backup(existing: &str) -> bool {
    !existing.trim().is_empty()
}

/// Verify `late` is installed on the remote before editing its config.
fn preflight_ssh(target: &SshTarget) -> Result<(), String> {
    let probe = "command -v late \
        || bash -lc 'command -v late' \
        || { [ -x \"$HOME/.local/bin/late\" ] && echo \"$HOME/.local/bin/late\"; }";
    match ssh_run(target, &[probe], None) {
        Ok(out) if !out.trim().is_empty() => Ok(()),
        Ok(_) | Err(_) => Err(format!(
            "late not found on {}. Install it there first.",
            target.spec()
        )),
    }
}

fn set_openai_settings(root_obj: &CstObject, model_name: &str, base_url: &str) {
    // late-cli's client appends `/v1/...` to the base URL internally,
    // so strip the `/v1` suffix that lui's harness inputs include.
    let late_base_url = base_url.strip_suffix("/v1").unwrap_or(base_url);

    match root_obj.get("openai_base_url") {
        Some(prop) => prop.set_value(s(late_base_url)),
        None => {
            root_obj.append("openai_base_url", s(late_base_url));
        }
    }

    match root_obj.get("openai_model") {
        Some(prop) => prop.set_value(s(model_name)),
        None => {
            root_obj.append("openai_model", s(model_name));
        }
    }
}
