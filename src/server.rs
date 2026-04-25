// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::config::{format_source, websearch_port};
use crate::settings::setting::PassthroughMode;
use crate::settings::store::Effective;
use crate::settings::value::Value;

const LOG_RING_SIZE: usize = 200;
const MAX_RECENT_REQUESTS: usize = 3;
/// How long a warning stays in the list before it's pruned. A recurring
/// condition just re-pushes the warning and it sticks around; a transient
/// one drops off quietly after the TTL.
const WARNING_TTL: Duration = Duration::from_secs(5 * 60);

/// Wire-format version for `/data`. Bump on breaking changes; additive
/// fields (with `#[serde(default)]` on the reader side) don't require a
/// bump. Kept so a client renderer can refuse an incompatible server.
pub const UI_SNAPSHOT_VERSION: u32 = 5;

#[derive(Debug, Clone)]
pub struct SlotInfo {
    pub slot_id: u32,
    pub n_tokens: u32,
    pub prompt_tps: f64,
    pub gen_tps: f64,
    pub gen_tokens: u32,
    pub total_time_ms: f64,
    // Prefill progress 0.0..1.0, from "prompt processing progress" lines.
    // Reaches ~1.0 when prefill finishes and generation begins.
    pub progress: f32,

    // When the slot started processing. Used with the quadratic prefill model
    // (elapsed ∝ progress²) to estimate remaining time. Attention cost per
    // token scales with prompt position, so a linear elapsed/progress
    // extrapolation is badly optimistic at long context - the last 30% of
    // progress can take as long as the first 70%.
    pub processing_started: Option<Instant>,
}

#[derive(Debug, Clone, Default)]
pub struct ServerState {
    // Startup info (parsed from logs)
    pub model_name: String,
    pub size_label: String,
    pub quantization: String,
    pub file_size_n: String,
    pub file_size_unit: String,
    pub file_bpw: String,
    pub model_params_n: String,
    pub model_params_unit: String,
    pub gpu_layers_loaded: u32,
    pub total_layers: u32,
    // Layers where llama.cpp counts the layer as "offloaded" but has spilled
    // its MoE experts back to host RAM (attn-only on GPU). Summed across all
    // GPU devices from the final `llama_params_fit_impl: ... (N overflowing)`
    // summary. Zero means no MoE expert spill — either fully GPU or
    // plain partial offload.
    pub overflow_layers: u32,
    pub cpu_mem_mib: f64,
    // Metal's MoE expert spill lives in a `CPU_REPACK` buffer (separate from
    // `CPU_Mapped`, which is typically just the token embedding). Non-zero
    // here is a reliable signal that layers actually overflowed to host RAM,
    // independent of the `(N overflowing)` summary line.
    pub cpu_repack_mib: f64,
    pub cpu_compute_mib: f64,
    pub gpu_mem_mib: f64,
    pub kv_cache_mib: f64,
    pub compute_buf_mib: f64,
    // Tensors llama.cpp forced to plain CPU because the preferred backend
    // couldn't accept them (e.g. q8_0 embedding with a CPU_REPACK backend).
    // Parsed from the `done_getting_tensors: ... (and N others) ... using CPU
    // instead` line; count is N + 1, primary is the first-named tensor. Lets
    // the display distinguish "fully GPU with embedding on CPU" from a real
    // weight spill.
    pub cpu_forced_count: u32,
    pub cpu_forced_primary: String,
    pub ctx_size: u32,
    pub max_ctx_size: u32,

    // llama.cpp version
    pub llama_version: String,
    pub update_available: bool,

    // Runtime state
    pub ready: bool,
    pub listen_url: String,
    pub request_count: u64,
    pub active_requests: u32,

    // Slot tracking
    pub active_slots: HashMap<u32, SlotInfo>,
    pub recent_completed: VecDeque<SlotInfo>,

    // Tokens/sec tracking
    pub last_prompt_tps: f64,
    pub last_gen_tps: f64,
    pub avg_prompt_tps: f64,
    pub avg_gen_tps: f64,
    pub prompt_tps_samples: u64,
    pub gen_tps_samples: u64,

    // Cache-health diagnostics. High counts = prompt cache not being reused
    // turn-to-turn (the usual cause of "it got slow at long context").
    pub full_reprocess_count: u64,
    pub invalidated_checkpoint_count: u64,

    // Download progress: filename -> percentage
    pub downloads: HashMap<String, u32>,

    // Log ring buffer
    pub log_lines: VecDeque<String>,

    // Process exited?
    pub exited: bool,
    pub exit_message: String,

    // Set when lui wants to abort with a specific, user-actionable explanation
    // (e.g. GPU VRAM oversubscribed at load time). Causes main() to exit 1
    // after print_summary renders the reason.
    pub fatal_reason: Option<String>,

    // Seeded from the `allow_vram_oversubscription` registry setting at
    // spawn time so parse_line can consult it without plumbing an
    // `Effective` reference down. When true, the VRAM-oversubscribed
    // detection below is skipped.
    pub allow_vram_oversubscription: bool,

    // True while llama-server's `-fit` logic is probing memory at candidate
    // context sizes. Those probes emit llama_memory_breakdown_print lines
    // showing self>total for the over-budget candidates — which is expected,
    // not a failure. Ignore the oversubscription check while probing.
    pub fit_probing: bool,

    // Web search tracking (local /bsearch endpoint; see src/websearch.rs).
    // `active_searches` holds one entry per in-flight search, keyed by the
    // request id; the display iterates values() to render them as sub-lines
    // under the WebSearch KV. Active count is just `.len()`.
    pub websearch_total: u64,
    pub active_searches: HashMap<String, String>,

    // Warnings surfaced in the UI's Warnings section. Stored with the push
    // Instant so entries can age out after WARNING_TTL; a recurring condition
    // just re-pushes and the warning sticks, a transient one drops off.
    pub warnings: Vec<(Instant, String)>,
}

impl ServerState {
    /// Pre-filtered log ingress: drops CUDA Graph reuse noise before it
    /// reaches the ring buffer.
    pub fn push_log(&mut self, line: String) {
        // Suppress "CUDA Graph id N reused" lines — they're informational
        // reuse notifications that clutter the log with no diagnostic value.
        static CUDA_GRAPH_RE: OnceLock<Regex> = OnceLock::new();
        let re =
            CUDA_GRAPH_RE.get_or_init(|| Regex::new(r"^\s*CUDA Graph id \d+ reused\s*$").unwrap());
        if re.is_match(&line) {
            return;
        }
        // Suppress CUDA graph warmup messages — transient progress noise.
        static CUDA_WARMUP_RE: OnceLock<Regex> = OnceLock::new();
        let re = CUDA_WARMUP_RE.get_or_init(|| {
            Regex::new(
                r"^\s*ggml_backend_cuda_graph_compute: CUDA graph warmup (reset|complete)\s*$",
            )
            .unwrap()
        });
        if re.is_match(&line) {
            return;
        }
        if self.log_lines.len() >= LOG_RING_SIZE {
            self.log_lines.pop_front();
        }
        self.log_lines.push_back(line);
    }

    /// Append a warning to be shown in the UI's Warnings section. Prunes
    /// expired entries first, then dedupes on exact text so a recurring
    /// warning doesn't pile up duplicates while still live. Intentionally
    /// kept with no current callers — the plumbing (state, TTL, UI) is
    /// ready for future warnings to wire in.
    #[allow(dead_code)]
    pub fn push_warning(&mut self, msg: String) {
        self.prune_warnings();
        if !self.warnings.iter().any(|(_, w)| w == &msg) {
            self.warnings.push((Instant::now(), msg));
        }
    }

    /// Drop any warning older than WARNING_TTL. Called from push_warning and
    /// build_snapshot so the list is always fresh at both ingress and egress.
    pub fn prune_warnings(&mut self) {
        let now = Instant::now();
        self.warnings
            .retain(|(t, _)| now.duration_since(*t) < WARNING_TTL);
    }
}

/// Wire-format snapshot of the live state the UI needs to render.
/// Built from `ServerState` on each `/data` request.
///
/// A client renderer pointed at a server's `/data` gets the full UI from
/// this struct — that's the motivation for replacing `Instant` with
/// `processing_elapsed_ms` and carrying `uptime_seconds` explicitly instead
/// of computing it on the renderer side.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiSnapshot {
    pub version: u32,

    pub ready: bool,
    pub exited: bool,
    pub exit_message: String,
    pub fatal_reason: Option<String>,

    pub uptime_seconds: u64,

    pub model_name: String,
    pub quantization: String,
    pub file_size_n: String,
    pub file_size_unit: String,
    pub file_bpw: String,
    pub model_params_n: String,
    pub model_params_unit: String,
    pub ctx_size: u32,
    pub max_ctx_size: u32,

    pub cpu_mem_mib: f64,
    #[serde(default)]
    pub cpu_repack_mib: f64,
    #[serde(default)]
    pub cpu_compute_mib: f64,
    pub gpu_mem_mib: f64,
    pub kv_cache_mib: f64,
    pub compute_buf_mib: f64,
    pub gpu_layers_loaded: u32,
    pub total_layers: u32,
    #[serde(default)]
    pub overflow_layers: u32,
    #[serde(default)]
    pub cpu_forced_count: u32,
    #[serde(default)]
    pub cpu_forced_primary: String,

    pub llama_version: String,
    pub update_available: bool,

    pub request_count: u64,
    pub active_requests: u32,
    pub full_reprocess_count: u64,
    pub invalidated_checkpoint_count: u64,

    pub last_prompt_tps: f64,
    pub last_gen_tps: f64,
    pub avg_prompt_tps: f64,
    pub avg_gen_tps: f64,
    pub prompt_tps_samples: u64,
    pub gen_tps_samples: u64,

    pub downloads: Vec<DownloadSnapshot>,
    pub active_slots: Vec<SlotSnapshot>,
    pub recent_completed: Vec<SlotSnapshot>,

    pub websearch_total: u64,
    pub active_searches: Vec<String>,

    /// Warnings surfaced in the UI's Warnings section. Usually empty.
    pub warnings: Vec<String>,

    /// Pre-formatted config pieces the renderer needs (bind, model source,
    /// websearch toggle). Static for the lifetime of a server, but we
    /// include it in every snapshot so the renderer has a single source of
    /// truth per frame.
    pub config: ConfigSummary,

    /// Flat list of every registry-declared setting the UI wants to render,
    /// grouped by `group` label (`"sampling"`, `"tuning"`, ...). The
    /// renderer filters by group and concatenates — it never names an
    /// individual setting. See `build_setting_entries` for the production
    /// logic.
    #[serde(default)]
    pub settings: Vec<SettingEntry>,

    /// Tail of the server log ring (up to 40 lines, newest first, each
    /// truncated to 150 chars). Used by the Server Log panel in both local
    /// and remote mode.
    pub log_lines: Vec<String>,
}

/// One entry per registry-declared setting the UI wants to surface.
/// Pre-formatted server-side so a local renderer and a remote `--remote`
/// client render identical text. `display_label` pairs with `value` via
/// `"label=value"` — or just `label` when `value` is empty (bare flags
/// like `swa-full`, or aggregate rows like `"+3 extra"`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SettingEntry {
    /// Canonical registry name. Synthetic for aggregate rows (e.g. `"KV"`
    /// combining `cache_type_k` + `cache_type_v`).
    pub name: String,
    /// Human label shown in the UI (`"temp"`, `"KV"`, `"np"`, ...).
    pub display_label: String,
    /// Formatted value. Empty for bare-flag entries whose label is the
    /// whole rendered form (`"swa-full"`, `"+3 extra"`).
    pub value: String,
    /// Visual grouping: `"sampling"`, `"tuning"`, or None.
    pub group: Option<String>,
    /// True when no layer explicitly set this — the value shown is the
    /// registry default. The renderer uses this to dim-style defaults or
    /// skip them entirely for groups where "unset" means "don't show".
    pub is_default: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotSnapshot {
    pub slot_id: u32,
    pub n_tokens: u32,
    pub prompt_tps: f64,
    pub gen_tps: f64,
    pub gen_tokens: u32,
    pub total_time_ms: f64,
    pub progress: f32,
    /// Milliseconds since the slot started processing. `None` means it
    /// hasn't started yet (brand-new slot with `n_tokens == 0`).
    pub processing_elapsed_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadSnapshot {
    pub filename: String,
    pub pct: u32,
}

/// Pre-formatted configuration strings the renderer needs. We format on
/// the server side (once at spawn time) so a client renderer gets exactly
/// the same lines a local renderer would — no duplicate format functions
/// on both ends, and no need to ship the full Config over the wire.
///
/// Sampler / tuning detail lives in `UiSnapshot.settings` (built from
/// the registry); this struct carries only the static machine-shape
/// strings that don't change for the life of the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSummary {
    /// Exactly the text shown on the "Bind" sub-line, e.g. "0.0.0.0:8080".
    pub bind_addr: String,
    /// Port of the lui HTTP server (the one serving this snapshot). The
    /// renderer uses this to know where `/setup` lives on the *server* — a
    /// local renderer treats it as the bookmarklet URL; a client renderer
    /// ignores it in favor of its own local bsearch URL.
    pub web_port: u16,
    pub websearch: bool,
    /// "--hf org/repo" or "-m /path" or "none".
    pub model_source: String,
    /// Comma-separated alias names that resolve to the active model, e.g.
    /// "foo,bar". Empty when the active model has no aliases. Rendered
    /// in the UI on the Model line after the model name.
    #[serde(default)]
    pub model_aliases: String,
}

impl ConfigSummary {
    pub fn from_effective(
        eff: &Effective,
        aliases: &std::collections::BTreeMap<String, String>,
    ) -> Self {
        let host = eff.get_string("host").unwrap_or("127.0.0.1").to_string();
        let port = eff.get_i64("port").unwrap_or(8080) as u16;

        // Collect alias names that resolve to the active model key.
        let model_aliases = if let Some(active) = eff.get_string("active_model") {
            aliases
                .iter()
                .filter(|(_, target)| target.as_str() == active)
                .map(|(name, _)| name.as_str())
                .collect::<Vec<&str>>()
                .join(",")
        } else {
            String::new()
        };

        ConfigSummary {
            bind_addr: format!("{}:{}", host, port),
            web_port: websearch_port(eff),
            websearch: eff.get_bool("websearch").unwrap_or(true),
            model_source: format_source(eff),
            model_aliases,
        }
    }
}

/// Build the per-snapshot `settings` payload as a single registry walk.
/// Iteration order is the registry declaration order, which doubles as
/// the UI order for sampling and tuning rows; each entry's display
/// behavior is pulled entirely from `Setting::ui_label` +
/// `Setting::ui_format` with a generic fallback.
///
/// No setting names appear here: adding a new `group`-tagged setting to
/// the registry surfaces automatically.
pub fn build_setting_entries(eff: &Effective) -> Vec<SettingEntry> {
    let mut out: Vec<SettingEntry> = Vec::new();
    for s in eff.registry.settings() {
        let Some(group) = s.group else { continue };
        let raw = eff.get(s.name);
        let Some(value_str) = render_value(s, raw, eff) else {
            continue;
        };
        out.push(SettingEntry {
            name: s.name.to_string(),
            display_label: s.derived_ui_label(),
            value: value_str,
            group: Some(group.to_string()),
            is_default: !eff.is_explicitly_set(s.name),
        });
    }
    out
}

/// Run the setting's UI formatter (if any) or the generic value
/// renderer. Returns `None` when the row should be skipped — no value
/// set, no registry default, and no formatter output.
fn render_value(
    s: &crate::settings::setting::Setting,
    value: Option<&crate::settings::value::Value>,
    eff: &Effective,
) -> Option<String> {
    if let Some(fmt) = s.ui_format {
        return fmt(value, eff, s);
    }
    generic_value_display(value)
}

/// Format an f64 to 3 significant figures, trimming trailing zeros.
pub fn format_float(f: f64) -> String {
    if f == 0.0 {
        return "0".to_string();
    }

    let abs = f.abs();
    let decimal_places = if abs >= 100.0 {
        0
    } else if abs >= 10.0 {
        1
    } else if abs >= 1.0 {
        2
    } else if abs >= 0.1 {
        3
    } else if abs >= 0.01 {
        4
    } else if abs >= 0.001 {
        5
    } else {
        return format!("{:.3e}", f);
    };

    format!("{:.*}", decimal_places, f)
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_string()
}

/// Default renderer for scalar values. Composite kinds (`Map`,
/// `StringArray`) don't have a sensible single-row form here — settings
/// that need one provide a `ui_format`.
fn generic_value_display(value: Option<&crate::settings::value::Value>) -> Option<String> {
    use crate::settings::value::Value;
    match value? {
        Value::Bool(b) => Some(b.to_string()),
        Value::Integer(n) => Some(n.to_string()),
        Value::Float(f) => Some(format_float(*f)),
        Value::String(s) => Some(s.clone()),
        Value::StringArray(_) | Value::Map(_) => None,
    }
}

impl SlotInfo {
    fn to_snapshot(&self) -> SlotSnapshot {
        SlotSnapshot {
            slot_id: self.slot_id,
            n_tokens: self.n_tokens,
            prompt_tps: self.prompt_tps,
            gen_tps: self.gen_tps,
            gen_tokens: self.gen_tokens,
            total_time_ms: self.total_time_ms,
            progress: self.progress,
            processing_elapsed_ms: self
                .processing_started
                .map(|t| t.elapsed().as_millis() as u64),
        }
    }
}

impl ServerState {
    /// Materialize a wire-format snapshot. `uptime` is the elapsed time
    /// since the server process started — supplied by the caller because the
    /// lui HTTP server (not `ServerState`) owns that clock. `config` is
    /// the pre-formatted `ConfigSummary`, built once at spawn time.
    pub fn build_snapshot(
        &mut self,
        uptime: Duration,
        config: &ConfigSummary,
        setting_entries: &[SettingEntry],
    ) -> UiSnapshot {
        // Age out stale warnings before the renderer sees them.
        self.prune_warnings();
        let mut slots: Vec<SlotSnapshot> = self
            .active_slots
            .values()
            .map(SlotInfo::to_snapshot)
            .collect();
        // HashMap iteration is unordered; sort so the renderer sees a
        // stable slot order (avoids slots appearing to swap places between
        // ticks when the hasher reorders them).
        slots.sort_by_key(|s| s.slot_id);

        let recent: Vec<SlotSnapshot> = self
            .recent_completed
            .iter()
            .map(SlotInfo::to_snapshot)
            .collect();

        let mut downloads: Vec<DownloadSnapshot> = self
            .downloads
            .iter()
            .map(|(filename, pct)| DownloadSnapshot {
                filename: filename.clone(),
                pct: *pct,
            })
            .collect();
        downloads.sort_by(|a, b| a.filename.cmp(&b.filename));

        let mut active_searches: Vec<String> = self.active_searches.values().cloned().collect();
        active_searches.sort();

        // Tail of the log ring: newest first, up to 100 lines, each
        // truncated to 300 chars to keep the wire payload reasonable.
        let log_lines: Vec<String> = self
            .log_lines
            .iter()
            .rev()
            .take(100)
            .map(|l| {
                if l.len() > 300 {
                    let end = l
                        .char_indices()
                        .find(|(i, _)| *i >= 300)
                        .map(|(i, _)| i)
                        .unwrap_or(l.len());
                    l[..end].to_string()
                } else {
                    l.clone()
                }
            })
            .collect();

        UiSnapshot {
            version: UI_SNAPSHOT_VERSION,

            ready: self.ready,
            exited: self.exited,
            exit_message: self.exit_message.clone(),
            fatal_reason: self.fatal_reason.clone(),

            uptime_seconds: uptime.as_secs(),

            model_name: self.model_name.clone(),
            quantization: self.quantization.clone(),
            file_size_n: self.file_size_n.clone(),
            file_size_unit: self.file_size_unit.clone(),
            file_bpw: self.file_bpw.clone(),
            model_params_n: self.model_params_n.clone(),
            model_params_unit: self.model_params_unit.clone(),
            ctx_size: self.ctx_size,
            max_ctx_size: self.max_ctx_size,

            cpu_mem_mib: self.cpu_mem_mib,
            cpu_repack_mib: self.cpu_repack_mib,
            cpu_compute_mib: self.cpu_compute_mib,
            gpu_mem_mib: self.gpu_mem_mib,
            kv_cache_mib: self.kv_cache_mib,
            compute_buf_mib: self.compute_buf_mib,
            gpu_layers_loaded: self.gpu_layers_loaded,
            total_layers: self.total_layers,
            overflow_layers: self.overflow_layers,
            cpu_forced_count: self.cpu_forced_count,
            cpu_forced_primary: self.cpu_forced_primary.clone(),

            llama_version: self.llama_version.clone(),
            update_available: self.update_available,

            request_count: self.request_count,
            active_requests: self.active_requests,
            full_reprocess_count: self.full_reprocess_count,
            invalidated_checkpoint_count: self.invalidated_checkpoint_count,

            last_prompt_tps: self.last_prompt_tps,
            last_gen_tps: self.last_gen_tps,
            avg_prompt_tps: self.avg_prompt_tps,
            avg_gen_tps: self.avg_gen_tps,
            prompt_tps_samples: self.prompt_tps_samples,
            gen_tps_samples: self.gen_tps_samples,

            downloads,
            active_slots: slots,
            recent_completed: recent,

            websearch_total: self.websearch_total,
            active_searches,

            warnings: self.warnings.iter().map(|(_, w)| w.clone()).collect(),

            config: config.clone(),
            settings: setting_entries.to_vec(),

            // Tail of the log ring: newest first, up to 40 lines, each
            // truncated to 150 chars to keep the wire payload reasonable.
            log_lines,
        }
    }
}

pub struct ServerProcess {
    pub child: Box<dyn portable_pty::Child + Send>,
    pub state: Arc<Mutex<ServerState>>,
    // On Windows, dropping the slave kills the ConPTY and the child process,
    // so we keep it alive for the lifetime of the server.
    #[cfg(windows)]
    _slave: Box<dyn portable_pty::SlavePty + Send>,
}

/// Convert a toml-valued map into a flat JSON object suitable for
/// llama-server's --chat-template-kwargs. We only support the scalar +
/// array + nested-table cases (which is everything jinja templates care
/// about); datetimes — which JSON can't represent — get serialized as
/// their RFC 3339 string form rather than erroring, since a user who
/// types a datetime here almost certainly wants the string value in the
/// template's render context anyway.
pub fn toml_map_to_json_object(
    map: &std::collections::BTreeMap<String, toml::Value>,
) -> serde_json::Value {
    let mut obj = serde_json::Map::with_capacity(map.len());
    for (k, v) in map {
        obj.insert(k.clone(), toml_value_to_json(v));
    }
    serde_json::Value::Object(obj)
}

fn toml_value_to_json(v: &toml::Value) -> serde_json::Value {
    match v {
        toml::Value::String(s) => serde_json::Value::String(s.clone()),
        toml::Value::Integer(i) => serde_json::Value::Number((*i).into()),
        toml::Value::Float(f) => serde_json::Number::from_f64(*f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        toml::Value::Boolean(b) => serde_json::Value::Bool(*b),
        toml::Value::Datetime(d) => serde_json::Value::String(d.to_string()),
        toml::Value::Array(a) => {
            serde_json::Value::Array(a.iter().map(toml_value_to_json).collect())
        }
        toml::Value::Table(t) => {
            let mut obj = serde_json::Map::with_capacity(t.len());
            for (k, v) in t {
                obj.insert(k.clone(), toml_value_to_json(v));
            }
            serde_json::Value::Object(obj)
        }
    }
}

pub fn build_args(eff: &Effective) -> Vec<String> {
    let mut args = Vec::new();

    // Always-on lui policy. These are lui's opinions (unified KV, -fa on,
    // verbose logging, progress-friendly cache-reuse), not user settings,
    // so they don't live in the registry.
    let host = eff.get_string("host").unwrap_or("127.0.0.1").to_string();
    let port = eff.get_i64("port").unwrap_or(8080) as u16;
    args.push("--host".to_string());
    args.push(host);
    args.push("--port".to_string());
    args.push(port.to_string());
    args.push("--metrics".to_string());
    args.push("--jinja".to_string());
    args.push("--log-colors".to_string());
    args.push("off".to_string());
    args.push("-v".to_string());
    args.push("-fa".to_string());
    args.push("on".to_string());
    args.push("--cache-reuse".to_string());
    args.push("256".to_string());
    args.push("-kvu".to_string());

    // Model identity: `[server].active_model` names a per-model entry
    // whose `type` field decides between `-hf` and `-m`. Fallback path
    // for old-shape stores (no active_model, just hf_repo/model) still
    // synthesizes a flag from whichever is set.
    let active = eff
        .get_string("active_model")
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| {
            eff.get_string("hf_repo")
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        })
        .or_else(|| {
            eff.get_string("model")
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        });
    if let Some(active) = active {
        let ty = eff
            .per_model
            .and_then(|s| match s.get("type") {
                Some(crate::settings::value::Value::String(s)) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_else(|| {
                if eff.get_string("hf_repo").is_some_and(|s| !s.is_empty()) {
                    "huggingface".to_string()
                } else if eff.get_string("model").is_some_and(|s| !s.is_empty()) {
                    "local".to_string()
                } else {
                    crate::config::infer_model_type(&active).to_string()
                }
            });
        match ty.as_str() {
            "local" => {
                args.push("-m".to_string());
                args.push(active);
            }
            _ => {
                args.push("-hf".to_string());
                args.push(active);
            }
        }
    }

    // Everything else — sampler knobs, KV cache types, batch sizes, swa,
    // chat-template-kwargs, post-`--` extras — flows from walking the
    // SettingsRegistry. Adding a new passthrough flag is one declaration
    // in `src/settings/registry.rs`; this loop picks it up for free.
    for s in eff.registry.settings() {
        match s.passthrough {
            PassthroughMode::None => {}
            PassthroughMode::FlagValue => {
                let Some(val) = eff.get(s.name) else { continue };
                let Some(flag) = s.llama_flag else { continue };
                args.push(flag.to_string());
                args.push(format_passthrough_value(val));
            }
            PassthroughMode::BoolFlagIfTrue => {
                if matches!(eff.get(s.name), Some(Value::Bool(true))) {
                    if let Some(flag) = s.llama_flag {
                        args.push(flag.to_string());
                    }
                }
            }
            PassthroughMode::LiteralTokens => {
                // Append-semantics: global entries, then active-model entries.
                // `extra_args` is the sole consumer today; any future
                // StringArray setting with this mode gets the same treatment.
                args.extend(eff.merged_string_array(s.name));
            }
        }
    }

    args
}

/// Stringify a `Value` as llama-server would expect it on the argv.
/// Integers and floats get their natural display; strings forward
/// verbatim. Maps are serialised as a JSON object (llama-server's
/// --chat-template-kwargs takes this shape). Bool and StringArray aren't
/// `FlagValue`-valid — the registry should never route them here.
fn format_passthrough_value(v: &Value) -> String {
    match v {
        Value::Integer(n) => n.to_string(),
        Value::Float(f) => format_float(*f),
        Value::String(s) => s.clone(),
        Value::Map(m) => toml_map_to_json_object(m).to_string(),
        Value::Bool(_) | Value::StringArray(_) => String::new(),
    }
}

pub fn spawn_server(eff: &Effective, debug_log: Option<&str>) -> Result<ServerProcess, String> {
    let args = build_args(eff);

    let pty_system = native_pty_system();
    let pty_pair = pty_system
        .openpty(PtySize {
            rows: 24,
            cols: 200,
            pixel_width: 0,
            pixel_height: 0,
        })
        .map_err(|e| format!("Failed to open pty: {}", e))?;

    let mut cmd = CommandBuilder::new("llama-server");
    cmd.args(&args);

    let child = pty_pair
        .slave
        .spawn_command(cmd)
        .map_err(|e| format!("Failed to spawn llama-server: {}", e))?;

    // Read from the pty master (gets both stdout and stderr, and \r progress lines)
    let reader = pty_pair
        .master
        .try_clone_reader()
        .map_err(|e| format!("Failed to clone pty reader: {}", e))?;

    let state = Arc::new(Mutex::new(ServerState {
        allow_vram_oversubscription: eff.get_bool("allow_vram_oversubscription").unwrap_or(false),
        ..ServerState::default()
    }));

    let debug_file =
        debug_log.map(|path| std::fs::File::create(path).expect("Failed to create debug log file"));

    // Spawn blocking reader task (pty reader is sync)
    let state_clone = state.clone();
    let reader_handle = tokio::task::spawn_blocking(move || {
        read_output_sync(reader, state_clone, debug_file);
    });

    // Monitor for exit
    let state_clone = state.clone();
    tokio::spawn(async move {
        let _ = reader_handle.await;
        let mut st = state_clone.lock().unwrap();
        if !st.exited {
            st.exited = true;
            st.exit_message = "llama-server process exited".to_string();
            st.push_log("llama-server process exited".to_string());
        }
    });

    // On Unix, drop slave so reads on master will EOF when child exits.
    // On Windows, dropping the slave kills the ConPTY (and the child), so keep it alive.
    #[cfg(not(windows))]
    drop(pty_pair.slave);

    Ok(ServerProcess {
        child,
        state,
        #[cfg(windows)]
        _slave: pty_pair.slave,
    })
}

/// Strip ANSI escape sequences from a string
fn strip_ansi(s: &str) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        // CSI sequences (includes ? for DEC private modes like ?25l),
        // OSC sequences, and single-character escapes (e.g. \x1b=)
        Regex::new(r"\x1b\[[\x20-\x3f]*[\x40-\x7e]|\x1b\][\x20-\x7e]*(?:\x07|\x1b\\)|\x1b[\x20-\x2f]*[\x30-\x7e]").unwrap()
    });
    re.replace_all(s, "").to_string()
}

fn read_output_sync(
    mut reader: impl std::io::Read,
    state: Arc<Mutex<ServerState>>,
    mut debug_file: Option<std::fs::File>,
) {
    use std::io::Write;
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        match std::io::Read::read(&mut reader, &mut byte) {
            Ok(0) => {
                if !buf.is_empty() {
                    let raw = String::from_utf8_lossy(&buf).to_string();
                    let line = strip_ansi(&raw);
                    if let Some(f) = debug_file.as_mut() {
                        let _ = writeln!(f, "{}", line);
                    }
                    if !line.is_empty() {
                        let mut st = state.lock().unwrap();
                        parse_line(&line, &mut st);
                        st.push_log(line);
                    }
                }
                return;
            }
            Ok(_) => {
                if byte[0] == b'\n' || byte[0] == b'\r' {
                    if !buf.is_empty() {
                        let raw = String::from_utf8_lossy(&buf).to_string();
                        let line = strip_ansi(&raw);
                        if let Some(f) = debug_file.as_mut() {
                            let _ = writeln!(f, "{}", line);
                        }
                        if !line.is_empty() {
                            let mut st = state.lock().unwrap();
                            let is_progress = parse_line(&line, &mut st);
                            if !is_progress {
                                st.push_log(line);
                            }
                        }
                    }
                    buf.clear();
                } else {
                    buf.push(byte[0]);
                }
            }
            Err(_) => return,
        }
    }
}

/// Parse a log line. Returns true if this is a transient progress line (don't add to log ring).
fn parse_line(line: &str, state: &mut ServerState) -> bool {
    // llama-server dumps the full request and response bodies to stdout via
    // `srv operator(): converted request: {...}` and `srv log_server_r:
    // request:` / `response:`. Those dumps echo arbitrary prompt/tool-output
    // JSON — a prompt containing lui's own source code has been observed to
    // inject fake "model buffer size = 7948 MiB" and "general.name str = ..."
    // fragments per pty-wrapped chunk, driving gpu_mem_mib to ~55 GiB and
    // overwriting model_name. Filter them here before any parser runs. Keep
    // `log_server_r: done request:` — that one is the real completion event
    // that feeds request_count.
    if line.contains("converted request:")
        || (line.contains("log_server_r:") && !line.contains("done request:"))
    {
        return false;
    }
    // Download progress: "Downloading Qwen3.6-35B-A3B-UD-Q4_K_M.gguf ──       9%"
    // ConPTY on Windows can concatenate multiple progress lines into one,
    // so we use captures_iter with a non-greedy match to extract all entries.
    if line.contains("Downloading ") {
        static RE_DL: OnceLock<Regex> = OnceLock::new();
        let re =
            RE_DL.get_or_init(|| Regex::new(r"Downloading (\S+\.\S+)\s.*?(\d{1,3})%").unwrap());
        let mut found = false;
        for caps in re.captures_iter(line) {
            let name = caps[1].to_string();
            let pct: u32 = caps[2].parse().unwrap_or(0);
            // Only update if progress moved forward (avoid glitchy backward jumps)
            let entry = state.downloads.entry(name).or_insert(0);
            if pct >= *entry {
                *entry = pct;
            }
            found = true;
        }
        return found;
    }
    // Everything below the download check is split into load-phase parsers
    // (fields set once during model load) and runtime parsers (request-level
    // events). Gating the load parsers on !state.ready is belt-and-braces on
    // top of the echo filter above: even if some future llama-server log
    // prefix slips past the filter, the accumulators (gpu_mem_mib, etc.) and
    // one-shot fields (model_name, quantization, ...) can no longer be
    // clobbered after startup.
    if !state.ready {
        parse_load_line(line, state);
    } else {
        parse_runtime_line(line, state);
    }
    false
}

/// Parsers for fields that only appear during model load. Called only while
/// `state.ready == false`; the `server is listening on` branch flips that
/// bit and ends this phase.
fn parse_load_line(line: &str, state: &mut ServerState) {
    // general.name
    if is_kv_line(line) && line.contains("general.name") && line.contains("str") {
        if let Some(val) = extract_kv_str(line) {
            state.model_name = val;
        }
    }
    // general.size_label
    else if is_kv_line(line) && line.contains("general.size_label") && line.contains("str") {
        if let Some(val) = extract_kv_str(line) {
            state.size_label = val;
        }
    }
    // context_length from model metadata (e.g. "llama.context_length u32 = 131072")
    else if line.contains(".context_length") && line.contains("u32") {
        if let Some(v) = extract_after_eq(line).and_then(|s| s.parse::<u32>().ok()) {
            state.max_ctx_size = v;
        }
    }
    // file type
    else if line.contains("print_info: file type") {
        if let Some(v) = extract_after_eq(line) {
            state.quantization = v.to_string();
        }
    }
    // file size: "4.36 GiB (4.91 BPW)"
    else if line.contains("print_info: file size") {
        if let Some(val) = extract_after_eq(line) {
            let parts: Vec<&str> = val.splitn(3, ' ').collect();
            if parts.len() >= 2 {
                state.file_size_n = parts[0].to_string();
                state.file_size_unit = parts[1].to_string();
            }
            // Extract BPW from parenthesized portion if present
            if let Some(bp) = val.find('(') {
                let inner = &val[bp + 1..];
                if let Some(ep) = inner.find(')') {
                    let bpw_parts: Vec<&str> = inner[..ep].splitn(2, ' ').collect();
                    if bpw_parts.len() == 2 {
                        state.file_bpw = bpw_parts[0].to_string();
                    }
                }
            }
        }
    }
    // model params: "7.62 B"
    else if line.contains("print_info: model params") {
        if let Some(val) = extract_after_eq(line) {
            let parts: Vec<&str> = val.splitn(2, ' ').collect();
            if parts.len() == 2 {
                state.model_params_n = parts[0].to_string();
                state.model_params_unit = parts[1].to_string();
            } else {
                state.model_params_n = val.to_string();
            }
        }
    }
    // offloaded layers: "offloaded 29/29 layers to GPU"
    else if line.contains("offloaded") && line.contains("layers to GPU") {
        static RE: OnceLock<Regex> = OnceLock::new();
        let re = RE.get_or_init(|| Regex::new(r"offloaded (\d+)/(\d+) layers to GPU").unwrap());
        if let Some(caps) = re.captures(line) {
            state.gpu_layers_loaded = caps[1].parse().unwrap_or(0);
            state.total_layers = caps[2].parse().unwrap_or(0);
        }
    }
    // CPU_Mapped model buffer size (embedding/output tensors forced to plain
    // CPU; on CUDA overflow this also carries the spilled expert weights).
    else if line.contains("CPU_Mapped model buffer size") {
        if let Some(mib) = extract_mib(line) {
            state.cpu_mem_mib += mib;
        }
    }
    // CPU_REPACK model buffer size (Metal's path for MoE experts that spilled
    // to host RAM — CUDA uses CPU_Mapped for the same thing).
    else if line.contains("CPU_REPACK model buffer size") {
        if let Some(mib) = extract_mib(line) {
            state.cpu_repack_mib += mib;
        }
    }
    // Plain "CPU model buffer size" (probe allocations print 0 here; kept so
    // it doesn't leak into the GPU branch below).
    else if line.contains("CPU model buffer size") {
        if let Some(mib) = extract_mib(line) {
            state.cpu_mem_mib += mib;
        }
    }
    // GPU model buffer size (MTL0 on macOS, CUDA0 on NVIDIA, Vulkan0, etc.)
    else if line.contains("model buffer size") && !line.contains("CPU") {
        if let Some(mib) = extract_mib(line) {
            state.gpu_mem_mib += mib;
        }
    }
    // KV buffer size (any GPU backend)
    else if line.contains("KV buffer size") && !line.contains("CPU") {
        if let Some(mib) = extract_mib(line) {
            state.kv_cache_mib = mib;
        }
    }
    // CPU compute buffer (small on GPU-backed runs; non-trivial once there's
    // real spill). Multiple emissions per load — last one (alloc_compute_meta)
    // wins, matching how the GPU compute line is handled.
    else if line.contains("CPU compute buffer size") {
        if let Some(mib) = extract_mib(line) {
            state.cpu_compute_mib = mib;
        }
    }
    // Compute buffer size (any GPU backend)
    else if line.contains("compute buffer size") && !line.contains("CPU") {
        if let Some(mib) = extract_mib(line) {
            state.compute_buf_mib = mib;
        }
    }
    // `done_getting_tensors: tensor 'X' (q8_0) (and N others) cannot be used
    // with preferred buffer type Y, using CPU instead`. Names the tensors that
    // fell back to plain CPU; count = N + 1. When the primary is `token_embd`
    // with N == 0, the CPU side is just the embedding and the load is
    // effectively full-GPU — distinct from a real weight spill.
    else if line.contains("done_getting_tensors:") && line.contains("using CPU instead") {
        static RE: OnceLock<Regex> = OnceLock::new();
        let re = RE
            .get_or_init(|| Regex::new(r"tensor\s+'([^']+)'.*?\(and\s+(\d+)\s+others\)").unwrap());
        if let Some(caps) = re.captures(line) {
            let others: u32 = caps[2].parse().unwrap_or(0);
            state.cpu_forced_count = others.saturating_add(1);
            state.cpu_forced_primary = caps[1].to_string();
        }
    }
    // llama-server's `-fit` logic probes memory at candidate context sizes,
    // starting at the model's full trained ctx and binary-searching down.
    // Each probe emits a breakdown line that may show self>total — that's
    // the signal fit uses to reject the candidate, NOT an allocation failure.
    // Suppress the oversubscription check during this window.
    //
    // The same block is where we harvest "N layers (M overflowing)" — the
    // summary llama.cpp prints right before `successfully fit`. "Overflowing"
    // means the layer's MoE experts were spilled to host RAM even though the
    // layer is nominally offloaded, so the plain `offloaded X/Y layers to
    // GPU` count alone is not truth-telling on MoE models.
    else if line.contains("llama_params_fit_impl:") {
        state.fit_probing = true;
        // Start of a new fit run — reset so the next successful fit sees a
        // fresh total across its per-device summary lines.
        if line.contains("memory for test allocation") {
            state.overflow_layers = 0;
        }
        // Per-device summary line: "  - CUDA0 (...): 41 layers (28 overflowing), ..."
        // Only the "(N overflowing)" fragment is stable enough to match on —
        // the device label and layer count formatting have both shifted
        // between llama.cpp versions. The outer `llama_params_fit_impl:`
        // guard above already limits this regex to the right line family.
        static OVERFLOW_RE: OnceLock<Regex> = OnceLock::new();
        let re = OVERFLOW_RE.get_or_init(|| Regex::new(r"\(\s*(\d+)\s+overflowing\)").unwrap());
        if let Some(caps) = re.captures(line) {
            let n: u32 = caps[1].parse().unwrap_or(0);
            state.overflow_layers = state.overflow_layers.saturating_add(n);
        }
    } else if line.contains("llama_params_fit:")
        && (line.contains("successfully fit") || line.contains("cannot fit"))
    {
        state.fit_probing = false;
    }
    // GPU memory breakdown. Format (whitespace varies):
    //   "llama_memory_breakdown_print: |   - MTL0 (Apple M4 Max) |
    //    28753 = 28690 + (28872 = 20583 + 5182 + 3106) + 17592186015607 |"
    //      total   free     self   model   context   compute   unaccounted
    // If self > total the GPU is already over budget at load time and the
    // server will almost certainly crash later. Kill lui now with a clear
    // explanation instead of letting the user discover it 60k tokens in.
    else if line.contains("llama_memory_breakdown_print:") && !line.contains("CPU") {
        static RE: OnceLock<Regex> = OnceLock::new();
        let re = RE.get_or_init(|| Regex::new(r"(\d+)\s*=\s*\d+\s*\+\s*\(\s*(\d+)\s*=").unwrap());
        if let Some(caps) = re.captures(line) {
            let total: u64 = caps[1].parse().unwrap_or(0);
            let selfsz: u64 = caps[2].parse().unwrap_or(0);
            if total > 0
                && selfsz > total
                && !state.allow_vram_oversubscription
                && !state.fit_probing
            {
                let over = selfsz - total;
                let msg = format!(
                    "GPU VRAM oversubscribed: model + KV + compute = {} MiB but device has only {} MiB ({} MiB over).\n  Try one or more of:\n    --ctk q8_0 --ctv q8_0   (halves KV cache)\n    --ubatch-size 512       (smaller compute buffer)\n    -c <smaller>            (reduce context window)\n\n  This check can be a false positive when the driver is willing to page\n  GPU memory to host RAM instead of failing the allocation:\n    - NVIDIA on Windows with \"CUDA Sysmem Fallback Policy\" enabled\n      (NVIDIA Control Panel > Manage 3D Settings; on by default)\n    - macOS Metal, which treats its working-set limit as a soft cap\n\n  If that applies to you, load will succeed but long-context inference\n  will pay a PCIe/paging cost. Pass --avo (or set allow_vram_oversub-\n  scription = true in [server]) to skip this check and accept that\n  trade-off.",
                    selfsz, total, over
                );
                state.fatal_reason = Some(msg.clone());
                state.exit_message = msg;
                state.exited = true;
            }
        }
    }
    // Context size: "n_ctx         = 131072"
    else if line.contains("llama_context: n_ctx") {
        static RE: OnceLock<Regex> = OnceLock::new();
        let re = RE.get_or_init(|| Regex::new(r"n_ctx\s+=\s+(\d+)").unwrap());
        if let Some(caps) = re.captures(line) {
            state.ctx_size = caps[1].parse().unwrap_or(0);
        }
    }
    // Server ready: "server is listening on http://127.0.0.1:8080".
    else if line.contains("server is listening on") {
        if let Some(pos) = line.find("on ") {
            state.listen_url = line[pos + 3..].trim().to_string();
        }
        state.ready = true;
    }
}

/// Parsers for request-level events. Called only after `server is listening
/// on` has fired; load-phase fields are frozen by this point.
fn parse_runtime_line(line: &str, state: &mut ServerState) {
    // HTTP request completed.
    if line.contains("done request: POST") {
        state.request_count += 1;
    }
    // Prompt cache reuse failed: llama-server is redoing the whole prefill.
    // Message shape: "forcing full prompt re-processing due to lack of cache data
    // (likely due to SWA or hybrid/recurrent memory)"
    else if line.contains("forcing full prompt re-processing") {
        state.full_reprocess_count += 1;
    }
    // SWA checkpoint got discarded (hybrid-model reprocessing bug smell).
    else if line.contains("invalidated context checkpoint")
        || line.contains("invalidated checkpoint")
    {
        state.invalidated_checkpoint_count += 1;
    }
    // All slots idle: "srv  update_slots: all slots are idle"
    else if line.starts_with("srv") && line.contains("all slots are idle") {
        state.active_requests = 0;
        state.active_slots.clear();
    }
    // New slot processing a task: "slot launch_slot_: id  3 | task 0 | processing task, is_child = 0"
    // Require the line to start with "slot " so we don't match against the
    // verbose request dump (srv log_server_r:) that echoes prompt text -
    // a prompt containing this codebase as context will otherwise create
    // phantom "slot 3" entries in the UI.
    else if line.starts_with("slot launch_slot_") && line.contains("processing task") {
        if let Some((slot_id, _)) = extract_slot_task(line) {
            state.active_requests += 1;
            state.active_slots.insert(
                slot_id,
                SlotInfo {
                    slot_id,
                    n_tokens: 0,
                    prompt_tps: 0.0,
                    gen_tps: 0.0,
                    gen_tokens: 0,
                    total_time_ms: 0.0,
                    progress: 0.0,
                    processing_started: Some(Instant::now()),
                },
            );
        }
    }
    // Token count: "slot update_slots: id  3 | task 0 | new prompt, ..., task.n_tokens = 536"
    else if line.starts_with("slot update_slots:") && line.contains("new prompt") {
        if let Some((slot_id, _)) = extract_slot_task(line) {
            static RE: OnceLock<Regex> = OnceLock::new();
            let re = RE.get_or_init(|| Regex::new(r"task\.n_tokens\s*=\s*(\d+)").unwrap());
            if let Some(caps) = re.captures(line) {
                let n_tokens: u32 = caps[1].parse().unwrap_or(0);
                if let Some(slot) = state.active_slots.get_mut(&slot_id) {
                    slot.n_tokens = n_tokens;
                }
            }
        }
    }
    // Prefill progress: "slot update_slots: id  0 | task 0 | prompt processing progress,
    //   n_tokens = 4096, batch.n_tokens = 2048, progress = 0.024940"
    else if line.starts_with("slot update_slots:") && line.contains("prompt processing progress")
    {
        if let Some((slot_id, _)) = extract_slot_task(line) {
            static RE: OnceLock<Regex> = OnceLock::new();
            let re = RE.get_or_init(|| Regex::new(r"progress\s*=\s*([0-9.]+)").unwrap());
            if let Some(caps) = re.captures(line) {
                if let Ok(p) = caps[1].parse::<f32>() {
                    if let Some(slot) = state.active_slots.get_mut(&slot_id) {
                        slot.progress = p.clamp(0.0, 1.0);
                    }
                }
            }
        }
    }
    // Slot released: "slot release: id  3 | task 0 | stop processing: n_tokens = 538"
    else if line.starts_with("slot release:") && line.contains("stop processing") {
        if let Some((slot_id, _)) = extract_slot_task(line) {
            if let Some(mut slot) = state.active_slots.remove(&slot_id) {
                // Extract final n_tokens
                static RE: OnceLock<Regex> = OnceLock::new();
                let re = RE.get_or_init(|| Regex::new(r"n_tokens\s*=\s*(\d+)").unwrap());
                if let Some(caps) = re.captures(line) {
                    slot.n_tokens = caps[1].parse().unwrap_or(slot.n_tokens);
                }
                // Add to recent completed
                if state.recent_completed.len() >= MAX_RECENT_REQUESTS {
                    state.recent_completed.pop_front();
                }
                state.recent_completed.push_back(slot);
            }
            state.active_requests = state.active_requests.saturating_sub(1);
        }
    }
    // Prompt eval timing: "prompt eval time =     728.45 ms /   536 tokens (    1.36 ms per token,   735.81 tokens per second)"
    else if line.contains("prompt eval time =") {
        static RE: OnceLock<Regex> = OnceLock::new();
        let re = RE.get_or_init(|| Regex::new(r"(\d+\.?\d*)\s+tokens per second").unwrap());
        if let Some(caps) = re.captures(line) {
            let tps: f64 = caps[1].parse().unwrap_or(0.0);
            state.last_prompt_tps = tps;
            state.prompt_tps_samples += 1;
            let n = state.prompt_tps_samples as f64;
            state.avg_prompt_tps = state.avg_prompt_tps * ((n - 1.0) / n) + tps / n;
            // Also update the most recent active slot's prompt_tps
            // The timing line includes "id  N | task M |" before the timing
            if let Some((slot_id, _)) = extract_slot_task(line) {
                if let Some(slot) = state.active_slots.get_mut(&slot_id) {
                    slot.prompt_tps = tps;
                }
            }
        }
    }
    // Generation eval timing. llama.cpp pads the label with a run of spaces
    // (or, historically, a tab); match on the trimmed prefix so a future
    // indent tweak doesn't silently drop the sample.
    else if line.trim_start().starts_with("eval time =") {
        static RE: OnceLock<Regex> = OnceLock::new();
        let re = RE.get_or_init(|| Regex::new(r"(\d+\.?\d*)\s+tokens per second").unwrap());
        if let Some(caps) = re.captures(line) {
            let tps: f64 = caps[1].parse().unwrap_or(0.0);
            state.last_gen_tps = tps;
            state.gen_tps_samples += 1;
            let n = state.gen_tps_samples as f64;
            state.avg_gen_tps = state.avg_gen_tps * ((n - 1.0) / n) + tps / n;
        }
        // Gen timing doesn't have slot id in the line, update via last_timing_slot
    }
    // Total timing: "      total time =   25690.22 ms / 14289 tokens".
    // Trim-then-prefix avoids depending on llama.cpp's exact padding.
    else if line.trim_start().starts_with("total time =") {
        static RE_TIME: OnceLock<Regex> = OnceLock::new();
        let re =
            RE_TIME.get_or_init(|| Regex::new(r"(\d+\.?\d*)\s+ms\s*/\s*(\d+)\s+tokens").unwrap());
        if let Some(caps) = re.captures(line) {
            let time_ms: f64 = caps[1].parse().unwrap_or(0.0);
            let tokens: u32 = caps[2].parse().unwrap_or(0);
            // Update the most recently active slot (timing block follows print_timing with slot id)
            if let Some(slot) = state.active_slots.values_mut().last() {
                slot.total_time_ms = time_ms;
                slot.gen_tokens = tokens;
            }
        }
    }
    // Timing header: "slot print_timing: id  3 | task 0 |"
    // Sets which slot the next timing lines belong to
    else if line.starts_with("slot print_timing:") {
        if let Some((slot_id, _)) = extract_slot_task(line) {
            // Store last gen tps into this slot when we next see eval time
            if let Some(slot) = state.active_slots.get_mut(&slot_id) {
                slot.gen_tps = state.last_gen_tps;
            }
        }
    }
}

/// Extract "id  N | task M" from a slot log line
fn extract_slot_task(line: &str) -> Option<(u32, u32)> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"id\s+(\d+)\s*\|\s*task\s+(-?\d+)").unwrap());
    let caps = re.captures(line)?;
    let slot_id: u32 = caps[1].parse().ok()?;
    let task_id: u32 = caps[2].parse().unwrap_or(0);
    Some((slot_id, task_id))
}

fn is_kv_line(line: &str) -> bool {
    // Pattern: "llama_model_loader: - kv  N:  ..."
    line.contains("llama_model_loader:") && line.contains("kv")
}

fn extract_kv_str(line: &str) -> Option<String> {
    extract_after_eq(line).map(|s| s.to_string())
}

/// Return the substring after the first `" = "` divider, trimmed. Shared
/// by the KV-metadata and `print_info:` parsers whose lines all have the
/// "key <type> = value" shape. Using the space-bracketed form avoids
/// matching `!= ` / `>= ` / `<= ` and the `" = "` variant is what llama.cpp
/// emits consistently across both llama_model_loader and print_info.
fn extract_after_eq(line: &str) -> Option<&str> {
    line.find(" = ").map(|p| line[p + 3..].trim())
}

fn extract_mib(line: &str) -> Option<f64> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"(\d+\.?\d*)\s+MiB").unwrap());
    re.captures(line).and_then(|c| c[1].parse::<f64>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_graph_reuse_lines_are_filtered() {
        // "CUDA Graph id N reused" lines are noise; push_log should drop
        // them (leading/trailing whitespace too — llama.cpp pads some lines).
        let mut s = ServerState::default();
        s.push_log("CUDA Graph id 0 reused".to_string());
        s.push_log("CUDA Graph id 42 reused".to_string());
        s.push_log("CUDA Graph id 999999 reused".to_string());
        s.push_log("   CUDA Graph id 1 reused".to_string());
        s.push_log("CUDA Graph id 2 reused   ".to_string());
        s.push_log("\tCUDA Graph id 3 reused\t".to_string());
        assert_eq!(s.log_lines.len(), 0);

        // Real log lines still pass through.
        s.push_log("prompt eval time =   100.00 ms /  50 tokens".to_string());
        assert_eq!(s.log_lines.len(), 1);
    }
}
