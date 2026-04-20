//! Pure text-in, text-out transformation of an opencode config (JSON or
//! JSONC). Used by all three write paths:
//!   - local `update_opencode_config`
//!   - `--ssh` remote (pipes result over SSH)
//!   - `--remote` client-side local write
//!
//! Surgical edits via `jsonc-parser`'s CST feature — comments, key order, and
//! formatting of everything we don't touch are preserved. We only mutate our
//! own managed keys: `model` (conditionally), `provider.lui`,
//! `permission.bash["curl*http://127.0.0.1:<port>/*"]`, and optionally
//! `compaction.prune`.
//!
//! `build_opencode_json` is gone — the "parse to serde_json::Value, mutate,
//! re-serialize" path dropped comments on the floor and reordered keys, which
//! destroyed elaborate hand-maintained `opencode.jsonc` files.

use jsonc_parser::cst::{CstInputValue, CstObject, CstObjectProp, CstRootNode};
use jsonc_parser::ParseOptions;
use regex::Regex;

// We'd like to use jsonc_parser's `json!` macro here, but it has an arm-order
// issue in 0.32.3 that trips E0425 on object keys in some call contexts.
// Build `CstInputValue` via the crate's `From` impls instead — equally safe,
// no magic.
fn s(v: impl Into<String>) -> CstInputValue {
    CstInputValue::String(v.into())
}
fn b(v: bool) -> CstInputValue {
    CstInputValue::Bool(v)
}
fn n(v: u32) -> CstInputValue {
    CstInputValue::from(v)
}
fn arr(v: Vec<CstInputValue>) -> CstInputValue {
    CstInputValue::Array(v)
}
fn obj<I: IntoIterator<Item = (&'static str, CstInputValue)>>(props: I) -> CstInputValue {
    CstInputValue::Object(props.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

/// Parse `existing`, apply lui's managed edits, and return the new text. Empty
/// input is treated as `{}`. If the text parses but its root isn't an object,
/// we overwrite the root with a fresh `{}` (matching the historical behavior
/// of `build_opencode_json`).
pub fn update_opencode_config_text(
    existing: &str,
    model_name: &str,
    llama_base_url: &str,
    web_port: u16,
    ctx_size: u32,
    websearch_disabled: bool,
    set_prune_false: bool,
) -> Result<String, String> {
    let source = if existing.trim().is_empty() {
        "{}".to_string()
    } else {
        existing.to_string()
    };

    let parse_options = ParseOptions::default();
    let root = CstRootNode::parse(&source, &parse_options)
        .map_err(|e| format!("parse opencode config: {}", e))?;

    let root_obj = root.object_value_or_set();

    update_provider_lui(&root_obj, model_name, llama_base_url, ctx_size);
    update_permission_bash(&root_obj, web_port, websearch_disabled);
    if set_prune_false {
        set_compaction_prune_false_if_absent(&root_obj);
    }

    Ok(root.to_string())
}

/// True iff the text has content but no `provider.lui` entry. The caller uses
/// this to decide whether to drop a `.luibackup` before writing. An empty/absent
/// file → false (nothing to back up). Unparseable input → true (preserve it).
pub fn needs_backup(existing: &str) -> bool {
    if existing.trim().is_empty() {
        return false;
    }
    let parse_options = ParseOptions::default();
    let Ok(root) = CstRootNode::parse(existing, &parse_options) else {
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

fn update_provider_lui(
    root_obj: &CstObject,
    model_name: &str,
    llama_base_url: &str,
    ctx_size: u32,
) {
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
/// websearch is disabled we refuse to create parent objects (`permission` /
/// `bash`) just to insert nothing.
fn update_permission_bash(root_obj: &CstObject, web_port: u16, disabled: bool) {
    let current_pattern = format!("curl*http://127.0.0.1:{}/*", web_port);

    let has_permission = root_obj.get("permission").is_some();
    if disabled && !has_permission {
        return;
    }
    let permission = root_obj.object_value_or_set("permission");

    let has_bash = permission.get("bash").is_some();
    if disabled && !has_bash {
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

    if disabled {
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

/// Sets `compaction.prune: false` only if the key is absent — never clobbers a
/// user-set value. Creates the parent `compaction` object only if missing.
fn set_compaction_prune_false_if_absent(root_obj: &CstObject) {
    let compaction = root_obj.object_value_or_set("compaction");
    if compaction.get("prune").is_none() {
        compaction.append("prune", b(false));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply(input: &str, set_prune_false: bool) -> String {
        update_opencode_config_text(
            input,
            "Test-Model",
            "http://localhost:8080/v1",
            8081,
            160000,
            false,
            set_prune_false,
        )
        .unwrap()
    }

    fn apply_disabled(input: &str) -> String {
        update_opencode_config_text(
            input,
            "Test-Model",
            "http://localhost:8080/v1",
            8081,
            160000,
            true,
            false,
        )
        .unwrap()
    }

    #[test]
    fn preserves_line_and_block_comments() {
        let input = r#"{
  // top-level comment
  "theme": "dracula",
  /* block comment */
  "autoupdate": true
}"#;
        let out = apply(input, false);
        assert!(
            out.contains("// top-level comment"),
            "line comment gone:\n{}",
            out
        );
        assert!(
            out.contains("/* block comment */"),
            "block comment gone:\n{}",
            out
        );
        assert!(out.contains("\"theme\": \"dracula\""));
        assert!(out.contains("\"autoupdate\": true"));
        assert!(out.contains("\"provider\""));
        assert!(out.contains("\"lui\""));
    }

    #[test]
    fn preserves_sibling_providers() {
        let input = r#"{
  "provider": {
    "anthropic": { "npm": "@ai-sdk/anthropic" }
  }
}"#;
        let out = apply(input, false);
        assert!(
            out.contains("\"anthropic\""),
            "sibling provider removed:\n{}",
            out
        );
        assert!(out.contains("\"@ai-sdk/anthropic\""));
        assert!(out.contains("\"lui\""));
    }

    #[test]
    fn inserts_into_empty_object() {
        let out = apply("{}", false);
        assert!(out.contains("\"provider\""));
        assert!(out.contains("\"lui\""));
        // top-level `model` is intentionally NOT written (opencode's
        // most-recent-used-model logic handles defaults)
        assert!(!out.contains("\"model\":"));
    }

    #[test]
    fn inserts_into_empty_file() {
        let out = apply("", false);
        assert!(out.contains("\"provider\""));
        assert!(out.contains("\"lui\""));
    }

    #[test]
    fn replaces_existing_lui_provider() {
        let input = r#"{
  "provider": {
    "lui": { "npm": "old-npm" },
    "anthropic": { "npm": "@ai-sdk/anthropic" }
  }
}"#;
        let out = apply(input, false);
        assert!(!out.contains("old-npm"), "stale lui value kept:\n{}", out);
        assert!(out.contains("\"@ai-sdk/openai-compatible\""));
        assert!(out.contains("\"anthropic\""));
    }

    #[test]
    fn strips_stale_port_wildcard() {
        let input = r#"{
  "permission": {
    "bash": {
      "curl*http://127.0.0.1:9999/*": "allow",
      "rm -rf /": "deny"
    }
  }
}"#;
        let out = apply(input, false);
        assert!(
            !out.contains("curl*http://127.0.0.1:9999/*"),
            "stale port wildcard kept:\n{}",
            out
        );
        assert!(out.contains("curl*http://127.0.0.1:8081/*"));
        assert!(
            out.contains("\"rm -rf /\": \"deny\""),
            "unrelated perm removed:\n{}",
            out
        );
    }

    #[test]
    fn disabled_websearch_removes_current_pattern() {
        let input = r#"{
  "permission": {
    "bash": {
      "curl*http://127.0.0.1:8081/*": "allow"
    }
  }
}"#;
        let out = apply_disabled(input);
        assert!(
            !out.contains("curl*http://127.0.0.1:8081/*"),
            "current pattern kept when disabled:\n{}",
            out
        );
    }

    #[test]
    fn disabled_websearch_no_permission_creates_nothing() {
        let out = apply_disabled("{}");
        assert!(
            !out.contains("\"permission\""),
            "permission created while disabled:\n{}",
            out
        );
    }

    #[test]
    fn prune_flag_off_does_not_write_compaction() {
        let out = apply("{}", false);
        assert!(
            !out.contains("\"compaction\""),
            "compaction written without flag:\n{}",
            out
        );
    }

    #[test]
    fn prune_flag_on_writes_compaction_prune_false() {
        let out = apply("{}", true);
        assert!(out.contains("\"compaction\""));
        assert!(out.contains("\"prune\": false"));
    }

    #[test]
    fn prune_flag_on_does_not_clobber_user_set_true() {
        let input = r#"{ "compaction": { "prune": true } }"#;
        let out = apply(input, true);
        assert!(
            out.contains("\"prune\": true"),
            "user prune setting clobbered:\n{}",
            out
        );
        assert!(!out.contains("\"prune\": false"));
    }

    #[test]
    fn needs_backup_empty_is_false() {
        assert!(!needs_backup(""));
        assert!(!needs_backup("   \n  "));
    }

    #[test]
    fn needs_backup_with_lui_provider_is_false() {
        let input = r#"{ "provider": { "lui": { "name": "lui" } } }"#;
        assert!(!needs_backup(input));
    }

    #[test]
    fn needs_backup_missing_lui_is_true() {
        assert!(needs_backup(r#"{ "a": 1 }"#));
        assert!(needs_backup(r#"{ "provider": {} }"#));
        assert!(needs_backup(r#"{ "provider": { "anthropic": {} } }"#));
    }

    #[test]
    fn needs_backup_just_comments_is_true() {
        assert!(needs_backup("// only a comment\n"));
    }

    #[test]
    fn needs_backup_unparseable_is_true() {
        assert!(needs_backup("{ not valid json"));
    }

    #[test]
    fn model_is_never_touched() {
        // lui does not manage opencode's top-level `model` key. When absent,
        // opencode's most-recent-used-model logic picks — nicer UX than pinning
        // a default. Existing values (including stale `lui/*` from older lui
        // versions) are left alone; the release notes tell upgraders to
        // hand-delete if they want the most-recent-wins behavior.
        assert!(!apply("{}", false).contains("\"model\""));

        let out = apply(r#"{ "model": "anthropic/claude-sonnet-4-6" }"#, false);
        assert!(out.contains("\"model\": \"anthropic/claude-sonnet-4-6\""));

        let out = apply(r#"{ "model": "lui/Qwen2.5-Coder-7B" }"#, false);
        assert!(out.contains("\"model\": \"lui/Qwen2.5-Coder-7B\""));
    }

    /// Runs the surgical edit against a real JSONC file path given in
    /// $LUI_E2E_JSONC and writes the result to $LUI_E2E_OUT. Gated — only
    /// fires when both env vars are set, so it's a no-op in CI.
    #[test]
    fn e2e_from_env_files() {
        let Ok(inp_path) = std::env::var("LUI_E2E_JSONC") else {
            return;
        };
        let out_path = std::env::var("LUI_E2E_OUT").expect("LUI_E2E_OUT not set");
        let input = std::fs::read_to_string(&inp_path).unwrap();
        let out = update_opencode_config_text(
            &input,
            "Qwen2.5-Coder-7B",
            "http://localhost:8080/v1",
            8081,
            160000,
            false,
            false,
        )
        .unwrap();
        std::fs::write(&out_path, &out).unwrap();
    }

    #[test]
    fn e2e_jsonc_elaborate_preserves_user_state() {
        // End-to-end: a realistic user .jsonc with comments, sibling providers,
        // user-picked non-lui model, and user-set compaction.prune. After lui's
        // edit, none of the user's decisions should be overwritten.
        let input = r#"{
  // Team-wide config maintained by hand. Comments below matter.
  "$schema": "https://opencode.ai/config.json",
  "theme": "dracula",
  /* our active model is Anthropic by default */
  "model": "anthropic/claude-sonnet-4-6",
  "autoupdate": true,
  "provider": {
    // Anthropic via direct API
    "anthropic": {
      "npm": "@ai-sdk/anthropic",
      "options": {
        "apiKey": "{env:ANTHROPIC_API_KEY}"
      }
    }
  },
  "permission": {
    "bash": {
      // Allow gh CLI; deny destructive rm
      "gh *": "allow",
      "rm -rf *": "deny"
    }
  },
  // compaction: user hasn't touched it — should remain untouched after lui edits
  "compaction": {
    "prune": true
  }
}"#;
        let out = apply(input, false);

        // Comments preserved
        assert!(
            out.contains("// Team-wide config"),
            "top comment gone:\n{}",
            out
        );
        assert!(
            out.contains("/* our active model"),
            "block comment gone:\n{}",
            out
        );
        assert!(out.contains("// Anthropic via direct API"));
        assert!(out.contains("// Allow gh CLI"));
        assert!(out.contains("// compaction: user hasn't touched it"));

        // User's non-lui model NOT clobbered
        assert!(
            out.contains("\"model\": \"anthropic/claude-sonnet-4-6\""),
            "user model clobbered:\n{}",
            out
        );

        // Sibling provider preserved
        assert!(out.contains("\"anthropic\""));
        assert!(out.contains("\"@ai-sdk/anthropic\""));
        assert!(out.contains("{env:ANTHROPIC_API_KEY}"));

        // User's bash perms preserved
        assert!(out.contains("\"gh *\": \"allow\""));
        assert!(out.contains("\"rm -rf *\": \"deny\""));

        // Our own lui bits added
        assert!(out.contains("\"lui\""));
        assert!(out.contains("\"@ai-sdk/openai-compatible\""));
        assert!(out.contains("curl*http://127.0.0.1:8081/*"));

        // User's prune=true not clobbered (flag was off AND user had it set)
        assert!(
            out.contains("\"prune\": true"),
            "user prune clobbered:\n{}",
            out
        );
        assert!(!out.contains("\"prune\": false"));
    }

    #[test]
    fn preserves_trailing_comma() {
        let input = r#"{
  "theme": "dracula",
}"#;
        let out = apply(input, false);
        assert!(out.contains("\"theme\": \"dracula\""));
        assert!(out.contains("\"lui\""));
    }
}
