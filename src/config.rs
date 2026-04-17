// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LuiConfig {
    pub server: ServerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub hf_repo: String,
    #[serde(default = "default_ctx_size")]
    pub ctx_size: u32,
    #[serde(default = "default_gpu_layers")]
    pub gpu_layers: i32,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temp: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,

    // Performance knobs. Overriding these changes the value actually passed to
    // llama-server; leaving them unset uses lui's defaults (see DEFAULT_* below).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ubatch_size: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads_batch: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_type_k: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_type_v: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub swa_full: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_ram: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prio_batch: Option<i32>,

    #[serde(default)]
    pub extra_args: Vec<String>,

    // Local web-search HTTP endpoint (see src/websearch.rs). On by default;
    // disable with --no-websearch. Port defaults to llama-server port + 1
    // when `web_port` is None.
    #[serde(default)]
    pub websearch_disabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_port: Option<u16>,
}

// Defaults applied when the user hasn't set a value in CLI or TOML.
// Kept narrow on purpose: a default that's great at 32k context can OOM a
// memory-constrained GPU at 200k, so we only default values where the win
// is unambiguous regardless of context/model size.
pub const DEFAULT_PARALLEL: i32 = 1;

// Logical batch size for prefill. llama.cpp's own default is 2048, which
// means a progress update every ~2048 tokens decoded. At 512 we get ~4x
// more frequent progress (smoother progress bar) for a negligible prefill
// throughput cost on single-user workloads.
pub const DEFAULT_BATCH_SIZE: u32 = 512;

fn default_ctx_size() -> u32 {
    0
}
fn default_gpu_layers() -> i32 {
    -1
}
fn default_port() -> u16 {
    8080
}
fn default_host() -> String {
    "127.0.0.1".to_string()
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            model: String::new(),
            hf_repo: String::new(),
            ctx_size: default_ctx_size(),
            gpu_layers: default_gpu_layers(),
            port: default_port(),
            host: default_host(),
            temp: None,
            top_p: None,
            top_k: None,
            min_p: None,
            ubatch_size: None,
            batch_size: None,
            parallel: None,
            threads_batch: None,
            cache_type_k: None,
            cache_type_v: None,
            swa_full: None,
            cache_ram: None,
            prio_batch: None,
            extra_args: Vec::new(),
            websearch_disabled: false,
            web_port: None,
        }
    }
}

/// Resolve the port the local websearch HTTP server binds to.
/// Defaults to `llama_port + 1` unless the user set `--web-port`.
pub fn websearch_port(config: &ServerConfig) -> u16 {
    config.web_port.unwrap_or_else(|| config.port.saturating_add(1))
}

impl Default for LuiConfig {
    fn default() -> Self {
        LuiConfig {
            server: ServerConfig::default(),
        }
    }
}

pub fn config_path() -> PathBuf {
    // XDG-style path (~/.config/lui.toml on all platforms). On macOS this
    // is not the system "config_dir" (~/Library/Application Support/) but
    // it's what most CLI tools actually use and what users expect.
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".config").join("lui.toml")
}

pub fn load_config() -> LuiConfig {
    let path = config_path();
    if path.exists() {
        match std::fs::read_to_string(&path) {
            Ok(contents) => match toml::from_str(&contents) {
                Ok(config) => return config,
                Err(e) => eprintln!("Warning: failed to parse {}: {}", path.display(), e),
            },
            Err(e) => eprintln!("Warning: failed to read {}: {}", path.display(), e),
        }
    }
    LuiConfig::default()
}

pub fn save_config(config: &LuiConfig) {
    let path = config_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match toml::to_string_pretty(config) {
        Ok(contents) => {
            if let Err(e) = std::fs::write(&path, contents) {
                eprintln!("Warning: failed to write {}: {}", path.display(), e);
            }
        }
        Err(e) => eprintln!("Warning: failed to serialize config: {}", e),
    }
}

/// Derive a short model name for opencode from the hf_repo or model path.
/// e.g. "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M" -> "Qwen2.5-Coder-7B-Instruct"
/// e.g. "/path/to/qwen2.5-coder-7b-instruct-q4_k_m.gguf" -> "qwen2.5-coder-7b-instruct-q4_k_m"
pub fn derive_model_name(config: &ServerConfig) -> String {
    if !config.hf_repo.is_empty() {
        let repo = &config.hf_repo;
        // Take the part after '/' and before ':'
        let name = repo.split('/').last().unwrap_or(repo);
        let name = name.split(':').next().unwrap_or(name);
        // Strip common suffixes like -GGUF
        let name = name.strip_suffix("-GGUF").unwrap_or(name);
        name.to_string()
    } else if !config.model.is_empty() {
        // Extract filename without extension
        PathBuf::from(&config.model)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string()
    } else {
        "unknown".to_string()
    }
}

fn opencode_config_path() -> PathBuf {
    // opencode uses ~/.config/opencode/opencode.json (XDG-style), not ~/Library/Application Support/
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".config").join("opencode").join("opencode.json")
}

fn websearch_bash_pattern(port: u16) -> String {
    // opencode's permission.bash keys use simple `*` / `?` wildcard matching
    // (see https://opencode.ai/docs/permissions/). This narrowly allows only
    // curl calls to lui's local search endpoint on the configured port.
    format!("curl*http://127.0.0.1:{}/*", port)
}

/// Manage the `permission.bash` entry that allows curl to lui's local search
/// endpoint, so opencode doesn't prompt for every search. We only touch keys
/// shaped like our pattern (any port) — anything else the user put there is
/// left untouched. When websearch is disabled, we remove stale entries.
fn update_websearch_permission(root: &mut serde_json::Map<String, serde_json::Value>, config: &ServerConfig) {
    let current_pattern = websearch_bash_pattern(websearch_port(config));

    if !root.contains_key("permission") {
        if config.websearch_disabled {
            return;
        }
        root.insert("permission".to_string(), serde_json::json!({}));
    }
    let permission = match root.get_mut("permission").and_then(|v| v.as_object_mut()) {
        Some(p) => p,
        None => return,
    };

    if !permission.contains_key("bash") {
        if config.websearch_disabled {
            return;
        }
        permission.insert("bash".to_string(), serde_json::json!({}));
    }
    let bash = match permission.get_mut("bash").and_then(|v| v.as_object_mut()) {
        Some(b) => b,
        None => return,
    };

    // Drop any prior lui-webseach keys (e.g. stale port after --web-port
    // change) so we never leave a dead allowlist entry behind.
    bash.retain(|k, _| {
        !(k.starts_with("curl*http://127.0.0.1:") && k.ends_with("/*"))
            || k == &current_pattern
    });

    if config.websearch_disabled {
        bash.remove(&current_pattern);
    } else {
        bash.insert(current_pattern, serde_json::json!("allow"));
    }
}

pub fn update_opencode_config(config: &ServerConfig) {
    let path = opencode_config_path();
    let model_name = derive_model_name(config);

    // Read existing config or start with empty object
    let mut json: serde_json::Value = if path.exists() {
        match std::fs::read_to_string(&path) {
            Ok(contents) => serde_json::from_str(&contents).unwrap_or(serde_json::json!({})),
            Err(_) => serde_json::json!({}),
        }
    } else {
        serde_json::json!({})
    };

    let obj = json.as_object_mut().unwrap();

    // Set top-level model
    obj.insert(
        "model".to_string(),
        serde_json::json!(format!("lui/{}", model_name)),
    );

    // Disable opencode's tool-output pruning. Pruning replaces old tool results
    // with "[Old tool result content cleared]" between turns, which mutates the
    // prompt prefix and invalidates llama-server's prompt cache - on a 160k
    // context that's tens of thousands of tokens re-prefilled per turn.
    // Only set if the user hasn't explicitly configured compaction, so we
    // don't clobber their override.
    if !obj.contains_key("compaction") {
        obj.insert("compaction".to_string(), serde_json::json!({}));
    }
    let compaction = obj
        .get_mut("compaction")
        .and_then(|v| v.as_object_mut())
        .unwrap();
    if !compaction.contains_key("prune") {
        compaction.insert("prune".to_string(), serde_json::json!(false));
    }

    // Ensure provider section exists
    if !obj.contains_key("provider") {
        obj.insert("provider".to_string(), serde_json::json!({}));
    }

    let providers = obj.get_mut("provider").unwrap().as_object_mut().unwrap();

    // Create/update the "lui" provider
    providers.insert(
        "lui".to_string(),
        serde_json::json!({
            "name": "lui",
            "npm": "@ai-sdk/openai-compatible",
            "options": {
                "baseURL": format!("http://localhost:{}/v1", config.port),
                "toolParser": [
                    { "type": "raw-function-call" },
                    { "type": "json" }
                ]
            },
            "models": {
                &model_name: {
                    "name": &model_name,
                    "supportsToolCalls": true
                }
            }
        }),
    );

    // Pre-allow the web-search curl pattern so opencode doesn't prompt the
    // user for every search call. Pattern is narrow — only curl to lui's
    // local search port is allowed, nothing else. We manage *only* entries
    // matching our port so the user can freely add their own bash rules.
    update_websearch_permission(obj, config);

    // Write back
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(contents) = serde_json::to_string_pretty(&json) {
        if let Err(e) = std::fs::write(&path, contents) {
            eprintln!("Warning: failed to write {}: {}", path.display(), e);
        }
    }
}

fn websearch_skill_dir() -> PathBuf {
    // Skills live under ~/.config/opencode/skills/<name>/SKILL.md — see
    // https://opencode.ai/docs/skills/. The directory name must match the
    // `name:` field in frontmatter.
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".config")
        .join("opencode")
        .join("skills")
        .join("lui-web-search")
}

/// Write (or rewrite) the `lui-web-search` skill so opencode knows how to
/// invoke lui's local /search endpoint. The port is baked in at write time
/// so the skill stays correct across restarts/port changes. Removes the
/// skill instead when websearch is disabled, so turning it off cleanly
/// withdraws the capability rather than leaving a stale SKILL.md pointing
/// at a dead port.
pub fn update_websearch_skill(config: &ServerConfig) {
    let dir = websearch_skill_dir();
    let skill_path = dir.join("SKILL.md");

    if config.websearch_disabled {
        // Remove SKILL.md first, then the now-empty directory. We intentionally
        // only remove the dir if it's empty, so we don't wipe out anything a
        // user may have stashed alongside.
        if skill_path.exists() {
            let _ = std::fs::remove_file(&skill_path);
        }
        if dir.exists() {
            let _ = std::fs::remove_dir(&dir);
        }
        return;
    }

    let port = websearch_port(config);
    let body = format!(
        r#"---
name: lui-web-search
description: Search the web via lui's local search endpoint. Use whenever the user asks to search the web, look up current information, or find documentation online. Returns JSON results with title, url, and snippet.
license: BSD-2-Clause
---

# lui-web-search

lui runs a small local HTTP server that performs web searches for you.
No API keys required; the server scrapes a public search frontend and
returns structured results.

## Endpoint

```
GET http://127.0.0.1:{port}/search?q=<URL-ENCODED QUERY>&n=<COUNT>
```

- `q` (required): the search query. URL-encode it.
- `n` (optional, default 10, max 25): how many results to return.

## Response

JSON array of objects:

```json
[
  {{"title": "...", "url": "https://...", "snippet": "..."}}
]
```

A 4xx/5xx response or an empty array means the search failed or produced
no results — say so plainly rather than fabricating answers.

## How to invoke

Use `curl` directly. On Linux/macOS:

```sh
curl -sG 'http://127.0.0.1:{port}/search' \
  --data-urlencode 'q=rust async traits 2026' \
  --data-urlencode 'n=8'
```

On Windows (PowerShell), `curl.exe` is built in:

```powershell
$q = [uri]::EscapeDataString('rust async traits 2026')
curl.exe -s "http://127.0.0.1:{port}/search?q=$q&n=8"
```

The response is a JSON array of `{{title, url, snippet}}` objects. Read it,
then write your answer to the user as normal prose with markdown links.
Do not paste the raw JSON back into the chat. If you need the body of a
specific page, fetch that page separately.

## When to use

- User asks to "search the web", "look up", "google", "find recent", etc.
- You need information that post-dates your training cutoff.
- You need a canonical URL for documentation, a release, a spec, or an issue.

Do not use this for fetching content from a URL the user already gave
you — just fetch that URL directly.
"#
    );

    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!(
            "Warning: failed to create {}: {}",
            dir.display(),
            e
        );
        return;
    }
    if let Err(e) = std::fs::write(&skill_path, body) {
        eprintln!(
            "Warning: failed to write {}: {}",
            skill_path.display(),
            e
        );
    }
}
