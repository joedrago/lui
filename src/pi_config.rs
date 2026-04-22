//! Pure text-in, text-out transformation of a pi models.json.
//!
//! Pi stores custom model definitions in `~/.pi/agent/models.json`. Each
//! provider is a keyed object with `baseUrl`, `api`, `apiKey`, and a
//! `models` array.
//!
//! This module provides surgical edit functions that preserve the user's
//! formatting and comments (where possible) while inserting/updating the
//! `lui` provider entry.

use std::path::PathBuf;

use serde_json::{json, Value};

/// The directory where pi stores agent config: `~/.pi/agent/`.
pub fn pi_agent_dir() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".pi").join("agent")
}

/// The models.json path: `~/.pi/agent/models.json`.
pub fn pi_models_json_path() -> PathBuf {
    pi_agent_dir().join("models.json")
}

/// Parse existing text into a JSON Value, returning {} for empty/invalid.
fn parse_existing(text: &str) -> Value {
    match text.trim() {
        "" => json!({}),
        _ => serde_json::from_str(text).unwrap_or(json!({})),
    }
}

/// Build the `lui` provider entry for pi's models.json.
fn build_lui_provider(model_name: &str, base_url: &str, ctx_size: u32) -> Value {
    json!({
        "baseUrl": base_url,
        "api": "openai-completions",
        "apiKey": "lui",
        "models": [
            {
                "id": model_name,
                "name": model_name,
                "contextWindow": ctx_size,
                "maxTokens": 8192
            }
        ]
    })
}

/// Update the pi models.json with the lui provider entry.
/// Returns the new file contents as a string.
pub fn update_pi_models_json(
    existing: &str,
    model_name: &str,
    base_url: &str,
    ctx_size: u32,
) -> Result<String, String> {
    let mut root = parse_existing(existing);

    // Ensure root is an object
    if !root.is_object() {
        root = json!({});
    }

    let obj = root.as_object_mut().unwrap();
    let providers = obj.entry("providers").or_insert_with(|| json!({}));

    // Ensure providers is an object
    if !providers.is_object() {
        *providers = json!({});
    }

    let providers_obj = providers.as_object_mut().unwrap();
    let base_url = format!("{}/v1", base_url);
    providers_obj.insert(
        "lui".to_string(),
        build_lui_provider(model_name, &base_url, ctx_size),
    );

    // Pretty-print with 2-space indent to match common JSON conventions
    Ok(serde_json::to_string_pretty(&root).unwrap_or_else(|_| "{}".to_string()) + "\n")
}

/// True iff the text has content but no `providers.lui` entry. Used to decide
/// whether to create a backup before writing.
pub fn pi_needs_backup(existing: &str) -> bool {
    if existing.trim().is_empty() {
        return false;
    }
    let Ok(root) = serde_json::from_str::<Value>(existing) else {
        return true;
    };
    match root.get("providers") {
        Some(Value::Object(map)) => map.get("lui").is_none(),
        _ => true,
    }
}

/// Render the SKILL.md body for pi's web search skill.
pub fn render_pi_websearch_skill(port: u16) -> String {
    format!(
        r#"---
name: lui-web-search
description: Browser-mediated web search for local/self-hosted models that have no native web-search tool. If you already have a native search tool (e.g. web_search, browse), use that instead — this skill requires the user to click a browser bookmarklet and blocks on their interaction. Returns JSON results with title, url, and snippet.
license: BSD-2-Clause
---

# lui-web-search

> **When to use this skill.** Only when you have no native web-search tool
> available. If you have one (Anthropic's `web_search`, OpenAI's browse, a
> native `search` tool, etc.), prefer that — it's faster, doesn't require a
> browser tab, and doesn't block on user interaction. This skill is the
> fallback for local/self-hosted models (llama.cpp, etc.) whose provider
> doesn't expose web search.

lui's search endpoint opens a Google search tab in the user's real
browser. The user clicks a one-time-installed `lui-grab` bookmarklet on
the resulting page; the bookmarklet POSTs the rendered results back to
lui, which returns them to you.

## Endpoint

```
GET http://127.0.0.1:{port}/bsearch?q=<URL-ENCODED QUERY>
```

- `q` (required): the search query. URL-encode it.

The request **blocks for up to 120 seconds** while waiting for the
user to click the bookmarklet. On timeout you'll get HTTP 504.

## Response

JSON array of objects:

```json
[
  {{"title": "...", "url": "https://...", "snippet": "..."}}
]
```

An HTTP 504 means the user did not click the bookmarklet in time
(probably they were AFK or the browser tab got buried). Other 4xx/5xx
or an empty array means the search failed — say so plainly rather than
fabricating answers.

## How to invoke

```sh
curl -sG 'http://127.0.0.1:{port}/bsearch' \
  --data-urlencode 'q=rust async traits 2026'
```

On Windows (PowerShell):

```powershell
$q = [uri]::EscapeDataString('rust async traits 2026')
curl.exe -s "http://127.0.0.1:{port}/bsearch?q=$q"
```

Read the JSON, then write your answer as normal prose with markdown
links. Do not paste the raw JSON back into the chat. If you need the
body of a specific page, fetch that page separately.

## When to use

- User asks to "search the web", "look up", "google", "find recent", etc.
- You need information that post-dates your training cutoff.
- You need a canonical URL for documentation, a release, a spec, or an issue.

Do not use this for fetching content from a URL the user already gave
you — just fetch that URL directly.

## Important: this requires user action

Each call pops a browser tab the user must click on. Before invoking
this for the first time in a conversation, **tell the user what's about
to happen** so they can be ready, e.g.:

> "I'm going to search the web for that. When I do, a Google tab will
> open in your browser — click the **lui-grab** bookmarklet on it. If
> you haven't installed lui-grab yet, visit
> `http://127.0.0.1:{port}/setup` and drag it to your bookmarks bar
> first. (This URL is also shown in the lui status panel.)"

Then call `/bsearch` and wait. If the call returns HTTP 504, the user
didn't click in time — most likely they don't have the bookmarklet
installed yet. Stop, point them at the setup page, wait for them to
say it's ready, then retry.

Be deliberate about when to search:
- One search at a time. Do not fire parallel searches.
- Pick the best query first instead of iterating with small variations.
- Don't search for things you already know.
"#
    )
}

/// Write (or rewrite) the `lui-web-search` skill for pi so the agent knows
/// how to invoke lui's local /search endpoint. Removes the skill when
/// websearch is disabled.
pub fn update_pi_websearch_skill(port: u16, disabled: bool) {
    let skill_dir = pi_agent_dir().join("skills").join("lui-web-search");
    let skill_path = skill_dir.join("SKILL.md");

    if disabled {
        if skill_path.exists() {
            let _ = std::fs::remove_file(&skill_path);
        }
        if skill_dir.exists() {
            let _ = std::fs::remove_dir(&skill_dir);
        }
        return;
    }

    let body = render_pi_websearch_skill(port);

    if let Err(e) = std::fs::create_dir_all(&skill_dir) {
        eprintln!("Warning: failed to create {}: {}", skill_dir.display(), e);
        return;
    }
    if let Err(e) = std::fs::write(&skill_path, body) {
        eprintln!("Warning: failed to write {}: {}", skill_path.display(), e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn updates_empty_file() {
        let out = update_pi_models_json("", "TestModel", "http://localhost:8080", 160000).unwrap();
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert!(parsed.get("providers").is_some());
        let lui = &parsed["providers"]["lui"];
        assert_eq!(lui["baseUrl"], "http://localhost:8080/v1");
        assert_eq!(lui["api"], "openai-completions");
        assert_eq!(lui["apiKey"], "lui");
        assert_eq!(lui["models"][0]["id"], "TestModel");
        assert_eq!(lui["models"][0]["contextWindow"], 160000);
    }

    #[test]
    fn preserves_existing_providers() {
        let input = r#"{
  "providers": {
    "anthropic": {
      "baseUrl": "https://api.anthropic.com/v1",
      "api": "anthropic-messages",
      "apiKey": "sk-ant-xxx",
      "models": [{"id": "claude-sonnet-4-20250514"}]
    }
  }
}"#;
        let out =
            update_pi_models_json(input, "TestModel", "http://localhost:8080", 160000).unwrap();
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert!(parsed["providers"]["anthropic"].is_object());
        assert!(parsed["providers"]["lui"].is_object());
        assert_eq!(parsed["providers"]["anthropic"]["apiKey"], "sk-ant-xxx");
    }

    #[test]
    fn replaces_existing_lui_provider() {
        let input = r#"{
  "providers": {
    "lui": {
      "baseUrl": "http://localhost:9999/v1",
      "api": "openai-completions",
      "apiKey": "old",
      "models": [{"id": "OldModel"}]
    }
  }
}"#;
        let out =
            update_pi_models_json(input, "NewModel", "http://localhost:8080", 200000).unwrap();
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(
            parsed["providers"]["lui"]["baseUrl"],
            "http://localhost:8080/v1"
        );
        assert_eq!(parsed["providers"]["lui"]["models"][0]["id"], "NewModel");
        assert_eq!(
            parsed["providers"]["lui"]["models"][0]["contextWindow"],
            200000
        );
    }

    #[test]
    fn needs_backup_empty_is_false() {
        assert!(!pi_needs_backup(""));
        assert!(!pi_needs_backup("  \n  "));
    }

    #[test]
    fn needs_backup_with_lui_is_false() {
        let input = r#"{ "providers": { "lui": {} } }"#;
        assert!(!pi_needs_backup(input));
    }

    #[test]
    fn needs_backup_missing_lui_is_true() {
        assert!(pi_needs_backup(r#"{ "providers": {} }"#));
        assert!(pi_needs_backup(r#"{ "providers": { "anthropic": {} } }"#));
    }
}
