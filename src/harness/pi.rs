// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

//! pi harness: writes `~/.pi/agent/models.json` and manages the
//! lui-web-search skill in `~/.pi/agent/skills/lui-web-search`.
//!
//! pi's models.json is plain JSON, but jsonc-parser happily parses that as
//! a subset of JSONC — so any comments or formatting the user adds to
//! `~/.pi/agent/models.json` survive a round-trip.

use jsonc_parser::cst::{CstInputValue, CstObject, CstRootNode};
use jsonc_parser::ParseOptions;

use super::{b, n, obj, s, ConfigFile, Harness, HarnessInputs};
use crate::settings::store::Effective;

pub const HARNESS: Harness = Harness {
    name: "pi",
    setting_name: "harness_pi",
    flag_long: "harness-pi",
    default_on: false,
    help: &[
        "Manage ~/.pi/agent/models.json and the lui-web-search skill",
        "Leave pi's models.json alone (default)",
    ],
    config: ConfigFile {
        dir: ".pi/agent",
        candidates: &["models.json"],
    },
    apply,
    needs_backup,
    preflight_ssh: None,
};

fn apply(root: &CstObject, _eff: &Effective, inputs: &HarnessInputs) {
    set_providers_lui(root, &inputs.model_name, &inputs.base_url, inputs.ctx_size);
}

/// True iff the text has content but no `providers.lui` entry. Matches
/// the opencode pattern: the file itself is preserved via `.luibackup`
/// only the first time lui touches it.
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
    let Some(providers) = root_obj.object_value("providers") else {
        return true;
    };
    providers.get("lui").is_none()
}

fn set_providers_lui(
    root_obj: &CstObject,
    model_name: &str,
    base_url: &str,
    ctx_size: u32,
) {
    let providers = root_obj.object_value_or_set("providers");

    let lui_value = obj([
        ("baseUrl", s(base_url)),
        ("api", s("openai-completions")),
        ("apiKey", s("lui")),
        (
            "models",
            CstInputValue::Array(vec![obj([
                ("id", s(model_name)),
                ("name", s(model_name)),
                ("contextWindow", n(ctx_size)),
                ("maxTokens", n(8192)),
                ("supportsToolCalls", b(true)),
            ])]),
        ),
    ]);

    match providers.get("lui") {
        Some(prop) => prop.set_value(lui_value),
        None => {
            providers.append("lui", lui_value);
        }
    }
}
