// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::opencode_config;
use crate::pi_config;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LuiConfig {
    pub server: ServerConfig,
    // Two typed alias pools. `hf` maps short names to HuggingFace repo
    // strings (as passed to --hf); `model` maps short names to local
    // GGUF paths (as passed to -m). Positional arguments (bare words
    // before `--`) check both pools; the typed flags check only their
    // own pool.
    #[serde(default, skip_serializing_if = "AliasPools::is_empty")]
    pub aliases: AliasPools,
    // Per-model overrides keyed by the short model name produced by
    // `derive_model_name` (same name shown in `lui -l`). Anything present
    // here wins over the matching field in `[server]` when that model is
    // the active one. Missing keys just mean "no model-specific tweaks."
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub models: BTreeMap<String, ModelOverrides>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AliasPools {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub hf: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub model: BTreeMap<String, String>,
}

impl AliasPools {
    pub fn is_empty(&self) -> bool {
        self.hf.is_empty() && self.model.is_empty()
    }
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
    // Target free-memory margin (MiB) for llama-server's --fit. Stored as
    // a string so the per-device comma form ("2048,256,256") round-trips.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit_target: Option<String>,
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
            && self.fit_target.is_none()
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
    // Free-memory margin (MiB) to reserve per GPU device for llama-server's
    // `--fit`. Stored as a string so the comma-separated per-device form
    // ("2048,256,256") round-trips untouched to llama-server.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit_target: Option<String>,

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

    // When true, lui writes `compaction.prune: false` into opencode's config,
    // telling opencode to stop its tool-output pruning. Preserves
    // llama-server's prompt cache at the cost of unbounded context growth,
    // which matters for huge (160k+) local contexts. Off by default — users
    // of cloud providers don't want a global compaction override they didn't
    // ask for.
    #[serde(default)]
    pub opencode_disable_prune: bool,

    // When true, save_config strips per-model sections that have no
    // overrides set. Off by default: the empty sections at the bottom of
    // the file act as a history of every --hf the user has ever run, so
    // they can jump back to one and start tweaking without retyping.
    #[serde(default)]
    pub prune_unused_model_configs: bool,

    // When true, lui will not abort even if llama-server's memory-breakdown
    // log indicates the GPU is over budget at load time. Leave off unless
    // you know llama-server can still cope (e.g. relying on driver spill to
    // system RAM) and you'd rather take that risk than have lui exit.
    #[serde(default)]
    pub allow_vram_oversubscription: bool,

    // When true, lui writes/updates opencode.jsonc and the lui-web-search
    // SKILL.md. Set to false when running under external harnesses that
    // manage their own opencode config. Always global; rejected with --this.
    #[serde(default = "default_harness_opencode")]
    pub harness_opencode: bool,

    // When true, lui writes/updates ~/.pi/agent/models.json and the
    // lui-web-search SKILL.md for pi. Set to false when running under
    // external harnesses that manage their own pi config. Always global;
    // rejected with --this.
    #[serde(default = "default_harness_pi")]
    pub harness_pi: bool,
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

// KV cache element types. llama.cpp defaults both to f16; q8_0 halves VRAM
// usage with essentially no quality impact (~0.002-0.05 ppl on modern GQA
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

fn default_harness_opencode() -> bool {
    true
}

fn default_harness_pi() -> bool {
    false
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
            fit_target: None,
            chat_template_kwargs: std::collections::BTreeMap::new(),
            chat_template_kwargs_drop: Vec::new(),
            extra_args: Vec::new(),
            websearch_disabled: false,
            web_port: None,
            opencode_disable_prune: false,
            prune_unused_model_configs: false,
            allow_vram_oversubscription: false,
            harness_opencode: true,
            harness_pi: false,
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
    config
        .web_port
        .unwrap_or_else(|| config.port.saturating_add(1))
}

/// Render the "source" line shown under the Model KV: `--hf ORG/REPO`,
/// `-m /path/to.gguf`, or `none`. Lives here (not in display.rs) so the
/// server can pre-format it into the UiSnapshot and any renderer — local or
/// client — shows an identical line.
pub fn format_source(cfg: &ServerConfig) -> String {
    if !cfg.hf_repo.is_empty() {
        format!("--hf {}", cfg.hf_repo)
    } else if !cfg.model.is_empty() {
        format!("-m {}", cfg.model)
    } else {
        "none".to_string()
    }
}

/// Sampler overrides line under the Model KV. Returns `None` when every
/// sampler is at default, so the renderer can skip the row entirely rather
/// than showing an empty "sampling: " label.
pub fn format_sampling(cfg: &ServerConfig) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    if let Some(v) = cfg.temp {
        parts.push(format!("temp={}", v));
    }
    if let Some(v) = cfg.top_p {
        parts.push(format!("top-p={}", v));
    }
    if let Some(v) = cfg.top_k {
        parts.push(format!("top-k={}", v));
    }
    if let Some(v) = cfg.min_p {
        parts.push(format!("min-p={}", v));
    }
    if parts.is_empty() {
        None
    } else {
        Some(format!("sampling: {}", parts.join(" · ")))
    }
}

/// Effective-tuning line under the llamacpp KV. Always returns a string
/// (at minimum `np=N`) — the tuning section is never entirely absent.
pub fn format_tuning(cfg: &ServerConfig) -> String {
    let np = cfg.parallel.unwrap_or(DEFAULT_PARALLEL);
    let mut parts = vec![format!("np={}", np)];
    if let Some(ub) = cfg.ubatch_size {
        parts.push(format!("ubatch={}", ub));
    }

    // Only include KV types when the user explicitly set them.
    if let (Some(ctk), Some(ctv)) = (&cfg.cache_type_k, &cfg.cache_type_v) {
        parts.push(format!("KV={}/{}", ctk, ctv));
    }
    let default_b = cfg
        .ubatch_size
        .map(|ub| ub.max(DEFAULT_BATCH_SIZE))
        .unwrap_or(DEFAULT_BATCH_SIZE);
    parts.push(format!("batch={}", cfg.batch_size.unwrap_or(default_b)));
    if let Some(v) = cfg.threads_batch {
        parts.push(format!("tb={}", v));
    }
    if let Some(v) = cfg.cache_ram {
        parts.push(format!("cache-ram={}MiB", v));
    }
    if let Some(v) = cfg.prio_batch {
        parts.push(format!("prio-batch={}", v));
    }
    if let Some(ref v) = cfg.fit_target {
        parts.push(format!("fit-target={}", v));
    }
    match cfg.swa_full {
        Some(true) => parts.push("swa-full".to_string()),
        Some(false) => parts.push("swa-full=off".to_string()),
        None => {}
    }
    if !cfg.extra_args.is_empty() {
        parts.push(format!("+{} extra", cfg.extra_args.len()));
    }
    parts.join(" · ")
}

impl Default for LuiConfig {
    fn default() -> Self {
        LuiConfig {
            server: ServerConfig::default(),
            aliases: AliasPools::default(),
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
    if ov.fit_target.is_some() {
        effective.fit_target = ov.fit_target.clone();
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

    // Aliases section, if any. Serializing the AliasPools struct directly
    // makes toml emit [aliases.hf] and [aliases.model] sub-tables.
    if !to_write.aliases.is_empty() {
        #[derive(Serialize)]
        struct AliasesOnly<'a> {
            aliases: &'a AliasPools,
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

/// Resolve `name` via `[aliases.hf]`. If `name` looks like a real HF
/// repo (contains '/') we never consult the pool — users can't shadow
/// real repos. Returns the original string if no alias matches.
pub fn resolve_hf_alias(config: &LuiConfig, name: &str) -> String {
    if !name.contains('/') {
        if let Some(full) = config.aliases.hf.get(name) {
            return full.clone();
        }
    }
    name.to_string()
}

/// Resolve `name` via `[aliases.model]`. We treat anything path-shaped
/// (contains '/', '\', or starts with '.'/'~') as a literal path and
/// don't consult the pool, so users can't shadow real files.
pub fn resolve_model_alias(config: &LuiConfig, name: &str) -> String {
    let path_shaped =
        name.contains('/') || name.contains('\\') || name.starts_with('.') || name.starts_with('~');
    if !path_shaped {
        if let Some(full) = config.aliases.model.get(name) {
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

/// Pick which opencode config file to read/write:
///   - `opencode.jsonc` exists → edit it
///   - only `opencode.json` exists → edit it (never orphan it with a sibling
///     `.jsonc` — opencode would merge both and produce duplicated keys)
///   - neither exists → create `opencode.jsonc`
///
/// Fresh-machine default is `.jsonc` because opencode's own docs lead with
/// it and hand-maintained team configs almost universally use it. JSON is a
/// strict subset of JSONC, so a user who never wants comments loses nothing.
pub fn opencode_config_path() -> PathBuf {
    // opencode uses ~/.config/opencode/ (XDG-style), not ~/Library/Application Support/
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let dir = home.join(".config").join("opencode");
    let jsonc = dir.join("opencode.jsonc");
    if jsonc.exists() {
        return jsonc;
    }
    let json = dir.join("opencode.json");
    if json.exists() {
        return json;
    }
    jsonc
}

/// Return the `.luibackup` sibling for an opencode config file. Preserves the
/// full filename so `opencode.jsonc` → `opencode.jsonc.luibackup` (not
/// `opencode.luibackup`).
fn luibackup_path(path: &Path) -> PathBuf {
    let mut os = path.as_os_str().to_os_string();
    os.push(".luibackup");
    PathBuf::from(os)
}

/// Before lui first edits an opencode config that predates lui touching it
/// (no `provider.lui` entry), copy the original to a `.luibackup` sibling.
/// Written exactly once — never overwrites an existing backup. Failure is
/// non-fatal; the backup is a belt-and-suspenders, not a hard dependency.
fn maybe_write_luibackup(path: &Path, existing_text: &str) {
    if !opencode_config::needs_backup(existing_text) {
        return;
    }
    let backup = luibackup_path(path);
    if backup.exists() {
        return;
    }
    if let Err(e) = std::fs::write(&backup, existing_text) {
        eprintln!("Warning: failed to write {}: {}", backup.display(), e);
    }
}

pub fn update_opencode_config(config: &ServerConfig) {
    let path = opencode_config_path();

    let existing = std::fs::read_to_string(&path).unwrap_or_default();

    maybe_write_luibackup(&path, &existing);

    let base_url = format!("http://localhost:{}/v1", config.port);
    let model_name = derive_model_name(config);
    let new_text = match opencode_config::update_opencode_config_text(
        &existing,
        &model_name,
        &base_url,
        websearch_port(config),
        config.ctx_size,
        config.websearch_disabled,
        config.opencode_disable_prune,
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Warning: failed to update {}: {}", path.display(), e);
            return;
        }
    };

    // Skip the write (and atomic-rename churn) if nothing changed. Preserves
    // mtime so file-watchers don't fire for no reason.
    if new_text == existing {
        return;
    }

    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    // Atomic write: temp file + rename. We're now promising to preserve the
    // user's comments and formatting, so a power-loss mid-write would destroy
    // their hand-maintained config. Rename is atomic on POSIX when source and
    // dest are on the same filesystem, which they are (same directory).
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(".luitmp");
    let tmp_path = PathBuf::from(tmp);
    if let Err(e) = std::fs::write(&tmp_path, &new_text) {
        eprintln!("Warning: failed to write {}: {}", tmp_path.display(), e);
        return;
    }
    if let Err(e) = std::fs::rename(&tmp_path, &path) {
        eprintln!(
            "Warning: failed to rename {} -> {}: {}",
            tmp_path.display(),
            path.display(),
            e
        );
        let _ = std::fs::remove_file(&tmp_path);
    }
}

pub fn websearch_skill_dir() -> PathBuf {
    // Skills live under ~/.config/opencode/skills/<name>/SKILL.md — see
    // https://opencode.ai/docs/skills/. The directory name must match the
    // `name:` field in frontmatter.
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".config")
        .join("opencode")
        .join("skills")
        .join("lui-web-search")
}

/// Render the SKILL.md body that teaches opencode how to call lui's
/// `/bsearch` endpoint. The port is baked in so the file is self-contained —
/// called by both the local `update_websearch_skill` (using the local
/// web-search port) and the `--ssh` path (using the remote-side tunnel port).
pub fn render_websearch_skill(port: u16) -> String {
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

    let body = render_websearch_skill(websearch_port(config));

    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("Warning: failed to create {}: {}", dir.display(), e);
        return;
    }
    if let Err(e) = std::fs::write(&skill_path, body) {
        eprintln!("Warning: failed to write {}: {}", skill_path.display(), e);
    }
}

/// Write/update ~/.pi/agent/models.json with the lui provider entry.
pub fn update_pi_models_config(config: &ServerConfig) {
    let path = pi_config::pi_models_json_path();

    let existing = std::fs::read_to_string(&path).unwrap_or_default();

    // Mirror the opencode backup behavior: first-ever touch gets a backup.
    if pi_config::pi_needs_backup(&existing) {
        let mut backup = path.as_os_str().to_os_string();
        backup.push(".luibackup");
        let backup = PathBuf::from(backup);
        if !backup.exists() {
            let _ = std::fs::write(&backup, &existing);
        }
    }

    let base_url = format!("http://localhost:{}", config.port);
    let model_name = derive_model_name(config);
    let new_text = match pi_config::update_pi_models_json(
        &existing,
        &model_name,
        &base_url,
        config.ctx_size,
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Warning: failed to update {}: {}", path.display(), e);
            return;
        }
    };

    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    if let Err(e) = std::fs::write(&path, &new_text) {
        eprintln!("Warning: failed to write {}: {}", path.display(), e);
    }
}

/// Write/update the lui-web-search SKILL.md for pi.
pub fn update_pi_websearch_skill(config: &ServerConfig) {
    pi_config::update_pi_websearch_skill(websearch_port(config), config.websearch_disabled)
}
