use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use regex::Regex;

use crate::config::ServerConfig;

const LOG_RING_SIZE: usize = 200;
const MAX_RECENT_REQUESTS: usize = 3;

#[derive(Debug, Clone)]
pub struct SlotInfo {
    pub slot_id: u32,
    pub n_tokens: u32,
    pub prompt_tps: f64,
    pub gen_tps: f64,
    pub gen_tokens: u32,
    pub total_time_ms: f64,
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

    // Download progress: filename -> percentage
    pub downloads: HashMap<String, u32>,

    // Log ring buffer
    pub log_lines: VecDeque<String>,

    // Process exited?
    pub exited: bool,
    pub exit_message: String,
}

impl ServerState {
    pub fn push_log(&mut self, line: String) {
        if self.log_lines.len() >= LOG_RING_SIZE {
            self.log_lines.pop_front();
        }
        self.log_lines.push_back(line);
    }
}

pub struct ServerProcess {
    pub child: Box<dyn portable_pty::Child + Send>,
    pub state: Arc<Mutex<ServerState>>,
}

pub fn build_args(config: &ServerConfig) -> Vec<String> {
    let mut args = Vec::new();

    args.push("--host".to_string());
    args.push(config.host.clone());
    args.push("--port".to_string());
    args.push(config.port.to_string());
    args.push("--metrics".to_string());
    args.push("--jinja".to_string());
    args.push("--log-colors".to_string());
    args.push("off".to_string());
    args.push("-v".to_string());

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

    args.extend(config.extra_args.iter().cloned());

    args
}

pub fn spawn_server(config: &ServerConfig) -> Result<ServerProcess, String> {
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

    let state = Arc::new(Mutex::new(ServerState::default()));

    // Spawn blocking reader task (pty reader is sync)
    let state_clone = state.clone();
    let reader_handle = tokio::task::spawn_blocking(move || {
        read_output_sync(reader, state_clone);
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

    // Drop slave so reads on master will EOF when child exits
    drop(pty_pair.slave);

    Ok(ServerProcess { child, state })
}

/// Strip ANSI escape sequences from a string
fn strip_ansi(s: &str) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\][\x20-\x7e]*(?:\x07|\x1b\\)").unwrap()
    });
    re.replace_all(s, "").to_string()
}

fn read_output_sync(mut reader: Box<dyn std::io::Read + Send>, state: Arc<Mutex<ServerState>>) {
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        match std::io::Read::read(&mut reader, &mut byte) {
            Ok(0) => {
                if !buf.is_empty() {
                    let raw = String::from_utf8_lossy(&buf).to_string();
                    let line = strip_ansi(&raw);
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
    // Download progress: "Downloading qwen2.5-coder-32b-instruct-q4_k_m-00003…               16%"
    if line.starts_with("Downloading ") {
        if let Some(pct_pos) = line.rfind('%') {
            let before_pct = line[..pct_pos].trim_end();
            if let Some(space_pos) = before_pct.rfind(|c: char| !c.is_ascii_digit()) {
                if let Ok(pct) = before_pct[space_pos + 1..].parse::<u32>() {
                    // Extract filename: first word after "Downloading " (GGUF filenames have no spaces)
                    let rest = &line["Downloading ".len()..];
                    let name = rest.split_whitespace().next().unwrap_or("");
                    if !name.is_empty() {
                        state.downloads.insert(name.to_string(), pct);
                    }
                    return true;
                }
            }
        }
        return false;
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
    // MTL0_Mapped model buffer size (may appear multiple times, sum them)
    else if line.contains("MTL0_Mapped model buffer size") {
        if let Some(mib) = extract_mib(line) {
            state.gpu_mem_mib += mib;
        }
    }
    // KV buffer size
    else if line.contains("MTL0 KV buffer size") {
        if let Some(mib) = extract_mib(line) {
            state.kv_cache_mib = mib;
        }
    }
    // Compute buffer size
    else if line.contains("MTL0 compute buffer size") {
        if let Some(mib) = extract_mib(line) {
            state.compute_buf_mib = mib;
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
    // All slots idle
    else if line.contains("all slots are idle") {
        state.active_requests = 0;
        state.active_slots.clear();
    }
    // New slot processing a task: "slot launch_slot_: id  3 | task 0 | processing task, is_child = 0"
    else if line.contains("launch_slot_") && line.contains("processing task") {
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
                },
            );
        }
    }
    // Token count: "slot update_slots: id  3 | task 0 | new prompt, ..., task.n_tokens = 536"
    else if line.contains("update_slots:") && line.contains("new prompt") {
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
    // Slot released: "slot release: id  3 | task 0 | stop processing: n_tokens = 538"
    else if line.contains("release:") && line.contains("stop processing") {
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
    else if line.contains("print_timing:") {
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
