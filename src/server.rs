// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::config::{ServerConfig, DEFAULT_BATCH_SIZE, DEFAULT_PARALLEL};

const LOG_RING_SIZE: usize = 200;
const MAX_RECENT_REQUESTS: usize = 3;

/// Wire-format version for `/data`. Bump on breaking changes; additive
/// fields (with `#[serde(default)]` on the reader side) don't require a
/// bump. Kept so a Remote renderer can refuse an incompatible Lui.
pub const UI_SNAPSHOT_VERSION: u32 = 1;

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
    pub cpu_mem_mib: f64,
    pub gpu_mem_mib: f64,
    pub kv_cache_mib: f64,
    pub compute_buf_mib: f64,
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

    // Seeded from ServerConfig::allow_vram_oversubscription at spawn time so
    // parse_line can consult it without plumbing a &ServerConfig reference
    // down. When true, the VRAM-oversubscribed detection below is skipped.
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
}

impl ServerState {
    pub fn push_log(&mut self, line: String) {
        if self.log_lines.len() >= LOG_RING_SIZE {
            self.log_lines.pop_front();
        }
        self.log_lines.push_back(line);
    }
}

/// Wire-format snapshot of the live state the UI needs to render its upper
/// sections (everything above the Server Log). Built from `ServerState` on
/// each `/data` request; the server-side log ring is deliberately not
/// included because the Display renders logs from its local `ServerState`
/// directly (they're large and update far faster than the 4 Hz render tick).
///
/// A Remote renderer pointed at a Lui's `/data` gets the full upper UI from
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
    pub gpu_mem_mib: f64,
    pub kv_cache_mib: f64,
    pub compute_buf_mib: f64,
    pub gpu_layers_loaded: u32,
    pub total_layers: u32,

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
    /// since the Lui process started — supplied by the caller because the
    /// lui HTTP server (not `ServerState`) owns that clock.
    pub fn to_snapshot(&self, uptime: Duration) -> UiSnapshot {
        let mut slots: Vec<SlotSnapshot> =
            self.active_slots.values().map(SlotInfo::to_snapshot).collect();
        // HashMap iteration is unordered; sort so the renderer sees a
        // stable slot order (avoids slots appearing to swap places between
        // ticks when the hasher reorders them).
        slots.sort_by_key(|s| s.slot_id);

        let recent: Vec<SlotSnapshot> =
            self.recent_completed.iter().map(SlotInfo::to_snapshot).collect();

        let mut downloads: Vec<DownloadSnapshot> = self
            .downloads
            .iter()
            .map(|(filename, pct)| DownloadSnapshot {
                filename: filename.clone(),
                pct: *pct,
            })
            .collect();
        downloads.sort_by(|a, b| a.filename.cmp(&b.filename));

        let mut active_searches: Vec<String> =
            self.active_searches.values().cloned().collect();
        active_searches.sort();

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
            gpu_mem_mib: self.gpu_mem_mib,
            kv_cache_mib: self.kv_cache_mib,
            compute_buf_mib: self.compute_buf_mib,
            gpu_layers_loaded: self.gpu_layers_loaded,
            total_layers: self.total_layers,

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
fn toml_map_to_json_object(
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

pub fn build_args(config: &ServerConfig) -> Vec<String> {
    let mut args = Vec::new();

    args.push("--host".to_string());
    args.push(config.host.clone());
    args.push("--port".to_string());
    args.push(config.port.to_string());
    args.push("--metrics".to_string());
    args.push("--jinja".to_string());
    // chat_template_kwargs is a merged map (see resolve() in config.rs for
    // the layering). llama-server's --chat-template-kwargs is last-wins on
    // the whole JSON object, so we emit it exactly once with the merged
    // result. Omit entirely if the map is empty (e.g. user dropped the
    // code default and didn't add anything), since passing `{}` is noise.
    if !config.chat_template_kwargs.is_empty() {
        let json = toml_map_to_json_object(&config.chat_template_kwargs);
        args.push("--chat-template-kwargs".to_string());
        args.push(json.to_string());
    }
    args.push("--log-colors".to_string());
    args.push("off".to_string());
    args.push("-v".to_string());
    args.push("-fa".to_string());
    args.push("on".to_string());
    args.push("--cache-reuse".to_string());
    args.push("256".to_string());
    // Unified KV so slot contexts aren't silently partitioned at long ctx.
    args.push("-kvu".to_string());

    // Physical batch size is opt-in. llama-server's default (512) is safe for
    // any context size; raising it speeds up prefill but inflates the compute
    // buffer linearly and can OOM at long ctx on memory-tight GPUs.
    if let Some(v) = config.ubatch_size {
        args.push("-ub".to_string());
        args.push(v.to_string());
    }

    // Single slot by default: a TUI has one conversation and wants the full
    // context window held in one slot (no fragmentation, no slot-thrash).
    args.push("-np".to_string());
    args.push(config.parallel.unwrap_or(DEFAULT_PARALLEL).to_string());

    if !config.hf_repo.is_empty() {
        args.push("-hf".to_string());
        args.push(config.hf_repo.clone());
    } else if !config.model.is_empty() {
        args.push("-m".to_string());
        args.push(config.model.clone());
    }

    if config.ctx_size > 0 {
        args.push("-c".to_string());
        args.push(config.ctx_size.to_string());
    }

    if config.gpu_layers != 0 {
        args.push("-ngl".to_string());
        args.push(config.gpu_layers.to_string());
    }

    if let Some(temp) = config.temp {
        args.push("--temp".to_string());
        args.push(temp.to_string());
    }
    if let Some(top_p) = config.top_p {
        args.push("--top-p".to_string());
        args.push(top_p.to_string());
    }
    if let Some(top_k) = config.top_k {
        args.push("--top-k".to_string());
        args.push(top_k.to_string());
    }
    if let Some(min_p) = config.min_p {
        args.push("--min-p".to_string());
        args.push(min_p.to_string());
    }

    // Logical batch: default to DEFAULT_BATCH_SIZE so prefill progress updates
    // are granular. Floor at -ub so llama.cpp's n_batch >= n_ubatch check passes
    // when the user has explicitly raised ubatch_size.
    let default_b = match config.ubatch_size {
        Some(ub) => ub.max(DEFAULT_BATCH_SIZE),
        None => DEFAULT_BATCH_SIZE,
    };
    args.push("-b".to_string());
    args.push(config.batch_size.unwrap_or(default_b).to_string());
    if let Some(v) = config.threads_batch {
        args.push("-tb".to_string());
        args.push(v.to_string());
    }
    if let Some(ref v) = config.cache_type_k {
        args.push("-ctk".to_string());
        args.push(v.clone());
    }
    if let Some(ref v) = config.cache_type_v {
        args.push("-ctv".to_string());
        args.push(v.clone());
    }
    if config.swa_full == Some(true) {
        args.push("--swa-full".to_string());
    }
    if let Some(v) = config.cache_ram {
        args.push("--cache-ram".to_string());
        args.push(v.to_string());
    }
    if let Some(v) = config.prio_batch {
        args.push("--prio-batch".to_string());
        args.push(v.to_string());
    }

    args.extend(config.extra_args.iter().cloned());

    args
}

pub fn spawn_server(config: &ServerConfig, debug_log: Option<&str>) -> Result<ServerProcess, String> {
    let args = build_args(config);

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
        allow_vram_oversubscription: config.allow_vram_oversubscription,
        ..ServerState::default()
    }));

    let debug_file = debug_log.map(|path| {
        std::fs::File::create(path).expect("Failed to create debug log file")
    });

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
    // Download progress: "Downloading Qwen3.6-35B-A3B-UD-Q4_K_M.gguf ──       9%"
    // ConPTY on Windows can concatenate multiple progress lines into one,
    // so we use captures_iter with a non-greedy match to extract all entries.
    if line.contains("Downloading ") {
        static RE_DL: OnceLock<Regex> = OnceLock::new();
        let re = RE_DL.get_or_init(|| {
            Regex::new(r"Downloading (\S+\.\S+)\s.*?(\d{1,3})%").unwrap()
        });
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
        if let Some(pos) = line.rfind("= ") {
            if let Ok(v) = line[pos + 2..].trim().parse::<u32>() {
                state.max_ctx_size = v;
            }
        }
    }
    // file type
    else if line.contains("print_info: file type") {
        if let Some(pos) = line.find("= ") {
            state.quantization = line[pos + 2..].trim().to_string();
        }
    }
    // file size: "4.36 GiB (4.91 BPW)"
    else if line.contains("print_info: file size") {
        if let Some(pos) = line.find("= ") {
            let val = line[pos + 2..].trim();
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
        if let Some(pos) = line.find("= ") {
            let val = line[pos + 2..].trim();
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
    // CPU_Mapped model buffer size
    else if line.contains("CPU_Mapped model buffer size") {
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
    // Compute buffer size (any GPU backend)
    else if line.contains("compute buffer size") && !line.contains("CPU") {
        if let Some(mib) = extract_mib(line) {
            state.compute_buf_mib = mib;
        }
    }
    // llama-server's `-fit` logic probes memory at candidate context sizes,
    // starting at the model's full trained ctx and binary-searching down.
    // Each probe emits a breakdown line that may show self>total — that's
    // the signal fit uses to reject the candidate, NOT an allocation failure.
    // Suppress the oversubscription check during this window.
    else if line.contains("llama_params_fit_impl:") {
        state.fit_probing = true;
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
        let re = RE.get_or_init(|| {
            Regex::new(r"(\d+)\s*=\s*\d+\s*\+\s*\(\s*(\d+)\s*=").unwrap()
        });
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
    // Server ready: "server is listening on http://127.0.0.1:8080"
    else if line.contains("server is listening on") {
        if let Some(pos) = line.find("on ") {
            state.listen_url = line[pos + 3..].trim().to_string();
        }
        state.ready = true;
    }
    // HTTP request completed
    else if line.contains("done request: POST") {
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
    else if line.starts_with("slot update_slots:") && line.contains("prompt processing progress") {
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
    // Generation eval timing
    else if line.starts_with("       eval time =") || line.contains("\teval time =") {
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
    // Total timing: "      total time =   25690.22 ms / 14289 tokens"
    else if line.starts_with("      total time =") {
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
    false
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
    // Pattern: "kv  N:   key str    = value"
    // Use the first occurrence of " = " on a known-safe KV line to avoid
    // pulling junk from tail content that happens to contain "= ".
    let eq_pos = line.find(" = ")?;
    Some(line[eq_pos + 3..].trim().to_string())
}

fn extract_mib(line: &str) -> Option<f64> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"(\d+\.?\d*)\s+MiB").unwrap());
    re.captures(line).and_then(|c| c[1].parse::<f64>().ok())
}
