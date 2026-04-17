// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LuiConfig {
    pub server: ServerConfig,
    // Short aliases for long --hf strings. e.g. qwen = "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M"
    // When --hf receives a bare word (no '/'), it checks here first.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub aliases: BTreeMap<String, String>,
    // Per-model overrides keyed by the short model name produced by
    // `derive_model_name` (same name shown in `lui -l`). Anything present
    // here wins over the matching field in `[server]` when that model is
    // the active one. Missing keys just mean "no model-specific tweaks."
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub models: BTreeMap<String, ModelOverrides>,
}

/// Per-model overrides. Only fields that actually vary with the model live
/// here. Identity (`model`, `hf_repo`) and machine-shape (`port`, `host`,
/// websearch settings) deliberately stay global — overriding those per model
/// either doesn't make sense or would silently shadow settings the user
/// expects to be machine-wide.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelOverrides {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ctx_size: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_layers: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temp: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
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
    // Merged into the global chat_template_kwargs at resolve time (not
    // last-wins replace). See the note on ServerConfig::chat_template_kwargs.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub chat_template_kwargs: std::collections::BTreeMap<String, toml::Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chat_template_kwargs_drop: Vec<String>,
    // Model-specific extra args append to the global ones at resolve time.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra_args: Vec<String>,
}

impl ModelOverrides {
    /// True when no override fields are set. Used so we can prune empty
    /// entries from the TOML and keep the file diff-clean.
    pub fn is_empty(&self) -> bool {
        self.ctx_size.is_none()
            && self.gpu_layers.is_none()
            && self.temp.is_none()
            && self.top_p.is_none()
            && self.top_k.is_none()
            && self.min_p.is_none()
            && self.ubatch_size.is_none()
            && self.batch_size.is_none()
            && self.parallel.is_none()
            && self.threads_batch.is_none()
            && self.cache_type_k.is_none()
            && self.cache_type_v.is_none()
            && self.swa_full.is_none()
            && self.cache_ram.is_none()
            && self.prio_batch.is_none()
            && self.chat_template_kwargs.is_empty()
            && self.chat_template_kwargs_drop.is_empty()
            && self.extra_args.is_empty()
    }
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

    // Chat-template kwargs are merge-semantics, not last-wins: llama-server's
    // --chat-template-kwargs takes a single JSON object, so passing the flag
    // twice just replaces the earlier value. We store an explicit map here
    // (global) and on each ModelOverrides (per-model). resolve() seeds a
    // code-level default (`preserve_thinking: true`), then layers global on
    // top, then per-model, applying the matching `_drop` list at each layer.
    // Emitted as exactly one `--chat-template-kwargs <json>` from build_args.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub chat_template_kwargs: std::collections::BTreeMap<String, toml::Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chat_template_kwargs_drop: Vec<String>,

    #[serde(default)]
    pub extra_args: Vec<String>,

    // Local web-search HTTP endpoint (see src/websearch.rs). On by default;
    // disable with --no-websearch. Port defaults to llama-server port + 1
    // when `web_port` is None.
    #[serde(default)]
    pub websearch_disabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_port: Option<u16>,

    // When true, save_config strips per-model sections that have no
    // overrides set. Off by default: the empty sections at the bottom of
    // the file act as a history of every --hf the user has ever run, so
    // they can jump back to one and start tweaking without retyping.
    #[serde(default)]
    pub prune_unused_model_configs: bool,
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
            chat_template_kwargs: std::collections::BTreeMap::new(),
            chat_template_kwargs_drop: Vec::new(),
            extra_args: Vec::new(),
            websearch_disabled: false,
            web_port: None,
            prune_unused_model_configs: false,
        }
    }
}

/// Keys lui seeds into chat_template_kwargs before any user merging. Kept
/// in code (not toml) so the default survives a user-written global map —
/// the toml only sees keys the user explicitly set.
///
/// preserve_thinking=true: keeps prior assistant <think> blocks in the
/// rebuilt prompt on templates that honor the flag (Qwen3.6+); ignored by
/// templates that don't reference it.
fn default_chat_template_kwargs() -> std::collections::BTreeMap<String, toml::Value> {
    let mut m = std::collections::BTreeMap::new();
    m.insert("preserve_thinking".to_string(), toml::Value::Boolean(true));
    m
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
            aliases: BTreeMap::new(),
            models: BTreeMap::new(),
        }
    }
}

/// Merge the active model's overrides on top of the global server config,
/// producing the effective `ServerConfig` that actually gets handed to
/// llama-server. Model-shape fields win when Some; extra_args appends.
/// Call this AFTER the CLI has finished mutating the LuiConfig, not during
/// parse — the active key only stabilizes once -m / --hf have been applied.
pub fn resolve(config: &LuiConfig) -> ServerConfig {
    let mut effective = config.server.clone();

    // Build the final chat-template kwargs map up front, since it has its
    // own layered merge/drop semantics that don't match the rest of the
    // "per-model Some() wins" fields. Layer order: code default → global
    // map → global drops → per-model map → per-model drops. Write the
    // result back into effective.chat_template_kwargs; the drop list stays
    // on the stored config (so it round-trips on save) but is only consulted
    // here during resolution.
    let per_model = model_key(&effective).and_then(|k| config.models.get(&k));
    let mut kwargs = default_chat_template_kwargs();
    for (k, v) in &config.server.chat_template_kwargs {
        kwargs.insert(k.clone(), v.clone());
    }
    for k in &config.server.chat_template_kwargs_drop {
        kwargs.remove(k);
    }
    if let Some(ov) = per_model {
        for (k, v) in &ov.chat_template_kwargs {
            kwargs.insert(k.clone(), v.clone());
        }
        for k in &ov.chat_template_kwargs_drop {
            kwargs.remove(k);
        }
    }
    effective.chat_template_kwargs = kwargs;
    // Drop list only matters during resolve; clearing it on the effective
    // config keeps build_args from having to know about it.
    effective.chat_template_kwargs_drop.clear();

    let Some(ov) = per_model else {
        return effective;
    };
    if let Some(v) = ov.ctx_size {
        effective.ctx_size = v;
    }
    if let Some(v) = ov.gpu_layers {
        effective.gpu_layers = v;
    }
    if ov.temp.is_some() {
        effective.temp = ov.temp;
    }
    if ov.top_p.is_some() {
        effective.top_p = ov.top_p;
    }
    if ov.top_k.is_some() {
        effective.top_k = ov.top_k;
    }
    if ov.min_p.is_some() {
        effective.min_p = ov.min_p;
    }
    if ov.ubatch_size.is_some() {
        effective.ubatch_size = ov.ubatch_size;
    }
    if ov.batch_size.is_some() {
        effective.batch_size = ov.batch_size;
    }
    if ov.parallel.is_some() {
        effective.parallel = ov.parallel;
    }
    if ov.threads_batch.is_some() {
        effective.threads_batch = ov.threads_batch;
    }
    if ov.cache_type_k.is_some() {
        effective.cache_type_k = ov.cache_type_k.clone();
    }
    if ov.cache_type_v.is_some() {
        effective.cache_type_v = ov.cache_type_v.clone();
    }
    if ov.swa_full.is_some() {
        effective.swa_full = ov.swa_full;
    }
    if ov.cache_ram.is_some() {
        effective.cache_ram = ov.cache_ram;
    }
    if ov.prio_batch.is_some() {
        effective.prio_batch = ov.prio_batch;
    }
    // Append, not replace: globals tend to be machine-tuning (thread pins,
    // mlock) and per-model entries model-tuning; both should reach llama-server.
    effective.extra_args.extend(ov.extra_args.iter().cloned());

    effective
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

    let mut to_write = config.clone();

    // Make sure the active model has *some* entry in [models.*], even if
    // it has no overrides. This turns the tail of the toml into a history
    // of every --hf the user has tried; users can delete stale ones by
    // hand, or flip prune_unused_model_configs to have save_config do it.
    if let Some(k) = model_key(&to_write.server) {
        to_write.models.entry(k).or_default();
    }
    if to_write.server.prune_unused_model_configs {
        to_write.models.retain(|_, ov| !ov.is_empty());
    }

    // BTreeMap iterates in sorted-key order; partition into "has overrides"
    // and "empty" while preserving that order.
    let mut non_empty: Vec<(&String, &ModelOverrides)> = Vec::new();
    let mut empty: Vec<&String> = Vec::new();
    for (k, ov) in &to_write.models {
        if ov.is_empty() {
            empty.push(k);
        } else {
            non_empty.push((k, ov));
        }
    }

    // Serialize [server] on its own so we fully control what follows.
    #[derive(Serialize)]
    struct GlobalOnly<'a> {
        server: &'a ServerConfig,
    }
    let mut out = match toml::to_string_pretty(&GlobalOnly {
        server: &to_write.server,
    }) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Warning: failed to serialize config: {}", e);
            return;
        }
    };
    if !out.ends_with('\n') {
        out.push('\n');
    }

    // Aliases section, if any.
    if !to_write.aliases.is_empty() {
        #[derive(Serialize)]
        struct AliasesOnly<'a> {
            aliases: &'a BTreeMap<String, String>,
        }
        match toml::to_string_pretty(&AliasesOnly {
            aliases: &to_write.aliases,
        }) {
            Ok(s) => {
                out.push('\n');
                out.push_str(s.trim_end());
                out.push('\n');
            }
            Err(e) => eprintln!("Warning: failed to serialize aliases: {}", e),
        }
    }

    // Non-empty model sections next, alphabetical. Serializing each one
    // through a single-entry map gives us toml's own key-quoting for free.
    #[derive(Serialize)]
    struct ModelsOnly<'a> {
        models: &'a BTreeMap<String, ModelOverrides>,
    }
    for (k, ov) in &non_empty {
        let mut single: BTreeMap<String, ModelOverrides> = BTreeMap::new();
        single.insert((*k).clone(), (*ov).clone());
        match toml::to_string_pretty(&ModelsOnly { models: &single }) {
            Ok(s) => {
                out.push('\n');
                out.push_str(s.trim_end());
                out.push('\n');
            }
            Err(e) => eprintln!("Warning: failed to serialize model section: {}", e),
        }
    }

    // Empty model sections last, packed tight — just header lines with no
    // blank line between them, so the "history" block stays compact.
    if !empty.is_empty() {
        out.push('\n');
        for k in &empty {
            out.push_str(&format!("[models.{}]\n", toml_quote_key(k)));
        }
    }

    if let Err(e) = std::fs::write(&path, out) {
        eprintln!("Warning: failed to write {}: {}", path.display(), e);
    }
}

/// Quote a string for use as a TOML key. Bare keys (`[A-Za-z0-9_-]+`) are
/// emitted as-is; anything else gets basic-string quoting. Matches what
/// the `toml` crate produces so round-tripping stays stable.
fn toml_quote_key(s: &str) -> String {
    let bare = !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if bare {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04X}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// If `name` doesn't contain '/' it might be an alias. Returns the
/// resolved full string, or the original name unchanged.
pub fn resolve_alias(config: &LuiConfig, name: &str) -> String {
    if !name.contains('/') {
        if let Some(full) = config.aliases.get(name) {
            return full.clone();
        }
    }
    name.to_string()
}

/// The identity string used to key per-model overrides. Deliberately the
/// EXACT text the user typed after `--hf` (or `-m`) — including org prefix
/// and quantization — so e.g. `unsloth/Foo:Q4_K_M` and `unsloth/Foo:Q8_0`
/// get separate override entries, and `unsloth/Bar` vs `zai-org/Bar` don't
/// collide. Distinct from `derive_model_name`, which produces the short
/// cosmetic name used for opencode's provider/model ID.
pub fn model_key(config: &ServerConfig) -> Option<String> {
    if !config.hf_repo.is_empty() {
        Some(config.hf_repo.clone())
    } else if !config.model.is_empty() {
        Some(config.model.clone())
    } else {
        None
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
description: Search the web via lui's browser-mediated search endpoint. Use whenever the user asks to search the web, look up current information, or find documentation online. Returns JSON results with title, url, and snippet.
license: BSD-2-Clause
---

# lui-web-search

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
