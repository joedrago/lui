mod config;
mod display;
mod gguf;
mod server;

use std::path::PathBuf;

use clap::Parser;

use config::{load_config, save_config, update_opencode_config};
use display::Display;
use server::spawn_server;

#[derive(Parser, Debug)]
#[command(
    name = "lui",
    about = "A friendly TUI wrapper for llama.cpp's llama-server"
)]
struct Cli {
    /// Model file path
    #[arg(short = 'm', long = "model")]
    model: Option<String>,

    /// HuggingFace repo (e.g. Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M)
    #[arg(long = "hf")]
    hf: Option<String>,

    /// Context window size (0 = model default)
    #[arg(short = 'c', long = "ctx-size")]
    ctx_size: Option<u32>,

    /// GPU layers (-1 = all)
    #[arg(long = "ngl", long = "gpu-layers")]
    gpu_layers: Option<i32>,

    /// Server port (default: 8080)
    #[arg(long = "port")]
    port: Option<u16>,

    /// Bind to 0.0.0.0 instead of 127.0.0.1
    #[arg(long = "public")]
    public: bool,

    /// List locally cached models
    #[arg(short = 'l', long = "list")]
    list: bool,

    /// Extra args passed through to llama-server
    #[arg(last = true)]
    extra_args: Vec<String>,
}

struct CachedModel {
    repo: String,
    quants: Vec<String>,
    total_size: u64,
    complete: bool,
    context_length: Option<u32>,
    model_name: Option<String>,
}

fn scan_cached_models() -> Vec<CachedModel> {
    let cache_dir = match dirs::home_dir().map(|d| d.join(".cache").join("huggingface").join("hub"))
    {
        Some(d) if d.exists() => d,
        _ => return Vec::new(),
    };

    let mut models = Vec::new();

    let Ok(entries) = std::fs::read_dir(&cache_dir) else {
        return models;
    };

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with("models--") {
            continue;
        }

        let repo = name
            .strip_prefix("models--")
            .unwrap_or(&name)
            .replacen("--", "/", 1);

        let path = entry.path();
        let blobs_dir = path.join("blobs");
        let snapshots_dir = path.join("snapshots");

        // Sum blob sizes for total on-disk size
        let mut total_size: u64 = 0;
        let mut has_blobs = false;
        if let Ok(blobs) = std::fs::read_dir(&blobs_dir) {
            for blob in blobs.flatten() {
                if let Ok(meta) = blob.metadata() {
                    total_size += meta.len();
                    has_blobs = true;
                }
            }
        }

        // Scan snapshots for GGUF filenames to extract quants
        let mut quants: Vec<String> = Vec::new();
        let mut has_gguf_files = false;
        if let Ok(snap_entries) = std::fs::read_dir(&snapshots_dir) {
            for snap in snap_entries.flatten() {
                if let Ok(files) = std::fs::read_dir(snap.path()) {
                    for file in files.flatten() {
                        let fname = file.file_name().to_string_lossy().to_string();
                        if !fname.ends_with(".gguf") {
                            continue;
                        }
                        has_gguf_files = true;
                        let lower = fname.to_lowercase();
                        for q in KNOWN_QUANTS {
                            if lower.contains(q) && !quants.contains(&q.to_uppercase()) {
                                quants.push(q.to_uppercase());
                            }
                        }
                    }
                }
            }
        }

        quants.sort();

        // A model is complete if it has blobs and GGUF files in snapshots
        let complete = has_blobs && has_gguf_files;

        // Read GGUF metadata from the first GGUF shard to get context_length.
        // For split models, metadata is in shard 1 (e.g. *-00001-of-*.gguf).
        let mut context_length = None;
        let mut model_name = None;
        if complete {
            let mut gguf_files: Vec<PathBuf> = Vec::new();
            if let Ok(snap_entries) = std::fs::read_dir(&snapshots_dir) {
                for snap in snap_entries.flatten() {
                    if let Ok(files) = std::fs::read_dir(snap.path()) {
                        for file in files.flatten() {
                            let fname = file.file_name().to_string_lossy().to_string();
                            if fname.ends_with(".gguf") {
                                gguf_files.push(file.path());
                            }
                        }
                    }
                }
            }
            // Sort so shard 00001 comes first
            gguf_files.sort();
            if let Some(first) = gguf_files.first() {
                let real_path: PathBuf =
                    std::fs::canonicalize(first).unwrap_or_else(|_| first.clone());
                if let Ok(meta) = gguf::read_gguf_metadata(&real_path) {
                    for (k, v) in &meta {
                        if k.ends_with(".context_length") {
                            context_length = v.parse::<u32>().ok();
                        }
                        if k == "general.name" {
                            model_name = Some(v.clone());
                        }
                    }
                }
            }
        }

        models.push(CachedModel {
            repo,
            quants,
            total_size,
            complete,
            context_length,
            model_name,
        });
    }

    models.sort_by(|a, b| a.repo.cmp(&b.repo));
    models
}

const KNOWN_QUANTS: &[&str] = &[
    "q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_0", "q4_k_s", "q4_k_m", "q4_k_l", "q5_0", "q5_k_s",
    "q5_k_m", "q5_k_l", "q6_k", "q8_0", "f16", "f32",
];

fn format_size(bytes: u64) -> String {
    if bytes == 0 {
        return "—".to_string();
    }
    let gib = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    if gib >= 1.0 {
        format!("{:.1} GiB", gib)
    } else {
        let mib = bytes as f64 / (1024.0 * 1024.0);
        format!("{:.0} MiB", mib)
    }
}

fn print_current_config() {
    use crossterm::style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor};
    let mut stdout = std::io::stdout();

    let lavender = Color::Rgb {
        r: 180,
        g: 150,
        b: 255,
    };
    let muted = Color::Rgb {
        r: 120,
        g: 100,
        b: 180,
    };

    let config = load_config();
    let s = &config.server;

    let _ = crossterm::execute!(
        stdout,
        Print("\n"),
        SetForegroundColor(lavender),
        SetAttribute(Attribute::Bold),
        Print("  Current config"),
        SetAttribute(Attribute::Reset),
        ResetColor,
    );

    // Show config file path
    let config_path = dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("lui.toml");
    if config_path.exists() {
        let _ = crossterm::execute!(
            stdout,
            SetForegroundColor(Color::DarkGrey),
            Print(format!(" ({})", config_path.display())),
            ResetColor,
        );
    } else {
        let _ = crossterm::execute!(
            stdout,
            SetForegroundColor(Color::DarkGrey),
            Print(" (no config file yet)"),
            ResetColor,
        );
    }
    let _ = crossterm::execute!(stdout, Print("\n\n"));

    // Helper closure for config lines
    let mut print_setting = |label: &str, value: &str, flag: &str, is_default: bool| {
        let label_color = if is_default {
            Color::DarkGrey
        } else {
            lavender
        };
        let value_color = if is_default {
            Color::DarkGrey
        } else {
            Color::Cyan
        };
        let _ = crossterm::execute!(
            stdout,
            Print("  "),
            SetForegroundColor(muted),
            Print("· "),
            SetForegroundColor(label_color),
            SetAttribute(Attribute::Bold),
            Print(label),
            SetAttribute(Attribute::Reset),
            Print("  "),
            SetForegroundColor(value_color),
            Print(value),
            ResetColor,
            Print("\n"),
            Print("      "),
            SetForegroundColor(Color::DarkGrey),
            Print(flag),
            ResetColor,
            Print("\n"),
        );
    };

    // Model
    if !s.hf_repo.is_empty() {
        print_setting("Model", &s.hf_repo, "--hf <repo>", false);
    } else if !s.model.is_empty() {
        print_setting("Model", &s.model, "-m <path>", false);
    } else {
        print_setting("Model", "(none)", "--hf <repo> or -m <path>", true);
    }

    // Context
    let ctx_str = if s.ctx_size == 0 {
        "model default".to_string()
    } else {
        format!("{}", s.ctx_size)
    };
    print_setting("Context", &ctx_str, "-c <size>", s.ctx_size == 0);

    // GPU layers
    let ngl_str = if s.gpu_layers == -1 {
        "all".to_string()
    } else {
        format!("{}", s.gpu_layers)
    };
    print_setting("GPU layers", &ngl_str, "--ngl <n>", s.gpu_layers == -1);

    // Port
    print_setting("Port", &s.port.to_string(), "--port <n>", s.port == 8080);

    // Host
    let host_flag = if s.host == "127.0.0.1" {
        "--public"
    } else {
        "--public (set)"
    };
    print_setting("Bind", &s.host, host_flag, s.host == "127.0.0.1");

    let _ = crossterm::execute!(stdout, Print("\n"));
}

fn list_cached_models() {
    print_current_config();

    let models = scan_cached_models();

    if models.is_empty() {
        println!("No cached models found. Download models with: llama-server -hf <repo>");
        return;
    }

    use crossterm::style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor};
    let mut stdout = std::io::stdout();

    let lavender = Color::Rgb {
        r: 180,
        g: 150,
        b: 255,
    };

    // Filter to only complete models
    let models: Vec<&CachedModel> = models.iter().filter(|m| m.complete).collect();

    if models.is_empty() {
        println!("No cached models found. Download models with: llama-server -hf <repo>");
        return;
    }

    let _ = crossterm::execute!(
        stdout,
        Print("\n"),
        SetForegroundColor(lavender),
        SetAttribute(Attribute::Bold),
        Print("  Cached models"),
        SetAttribute(Attribute::Reset),
        ResetColor,
        Print("\n\n"),
    );

    for model in &models {
        // Model name (use friendly name from GGUF if available, repo as fallback)
        let display_name = model.model_name.as_deref().unwrap_or(&model.repo);

        let _ = crossterm::execute!(
            stdout,
            Print("  "),
            SetForegroundColor(Color::White),
            SetAttribute(Attribute::Bold),
            Print(display_name),
            SetAttribute(Attribute::Reset),
            ResetColor,
            Print("\n"),
        );

        // Details line: quant, size, context
        let _ = crossterm::execute!(stdout, Print("    "));

        if !model.quants.is_empty() {
            let _ = crossterm::execute!(
                stdout,
                SetForegroundColor(Color::Cyan),
                Print(model.quants.join(", ")),
                ResetColor,
            );
        }

        let _ = crossterm::execute!(
            stdout,
            SetForegroundColor(Color::DarkGrey),
            Print(" · "),
            SetForegroundColor(Color::Yellow),
            Print(format_size(model.total_size)),
            ResetColor,
        );

        if let Some(ctx) = model.context_length {
            let _ = crossterm::execute!(
                stdout,
                SetForegroundColor(Color::DarkGrey),
                Print(" · "),
                SetForegroundColor(Color::Green),
                Print(format!("{}k ctx", ctx / 1024)),
                ResetColor,
            );
        }

        let _ = crossterm::execute!(stdout, Print("\n"));

        // Example command
        let quant_suffix = if model.quants.len() == 1 {
            format!(":{}", model.quants[0])
        } else if model.quants.is_empty() {
            String::new()
        } else {
            format!(":{}", model.quants[0])
        };
        let _ = crossterm::execute!(
            stdout,
            Print("    "),
            SetForegroundColor(Color::DarkGrey),
            Print(format!("lui --hf {}{}\n", model.repo, quant_suffix)),
            ResetColor,
        );
    }
    println!();
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Handle --list
    if cli.list {
        list_cached_models();
        return;
    }

    // Load config
    let mut config = load_config();

    // Apply CLI overrides
    if let Some(model) = cli.model {
        config.server.model = model;
        config.server.hf_repo.clear();
    }
    if let Some(hf) = cli.hf {
        config.server.hf_repo = hf;
        config.server.model.clear();
    }
    if let Some(ctx) = cli.ctx_size {
        config.server.ctx_size = ctx;
    }
    if let Some(ngl) = cli.gpu_layers {
        config.server.gpu_layers = ngl;
    }
    if let Some(port) = cli.port {
        config.server.port = port;
    }
    if cli.public {
        config.server.host = "0.0.0.0".to_string();
    }
    if !cli.extra_args.is_empty() {
        config.server.extra_args = cli.extra_args;
    }

    // Validate we have a model
    if config.server.model.is_empty() && config.server.hf_repo.is_empty() {
        eprintln!("Error: no model specified. Use --hf <repo> or -m <path>, or run 'lui --list' to see cached models.");
        std::process::exit(1);
    }

    // Save config
    save_config(&config);

    // Update opencode config
    update_opencode_config(&config.server);

    // Get llama-server version
    let llama_version = match std::process::Command::new("llama-server")
        .arg("--version")
        .output()
    {
        Ok(output) => {
            let text = String::from_utf8_lossy(&output.stderr);
            text.lines()
                .find(|l| l.starts_with("version:"))
                .map(|l| l.trim_start_matches("version:").trim().to_string())
                .unwrap_or_default()
        }
        Err(_) => String::new(),
    };

    // Spawn llama-server
    let mut proc = match spawn_server(&config.server) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    // Store version and kick off brew update check in background
    {
        let mut st = proc.state.lock().unwrap();
        st.llama_version = llama_version;
    }
    let state_for_brew = proc.state.clone();
    tokio::spawn(async move {
        let output = tokio::process::Command::new("brew")
            .args(["info", "--json=v2", "llama.cpp"])
            .output()
            .await;
        if let Ok(output) = output {
            if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&output.stdout) {
                let stable = json["formulae"][0]["versions"]["stable"].as_str();
                let installed = json["formulae"][0]["installed"][0]["version"].as_str();
                if let (Some(stable), Some(installed)) = (stable, installed) {
                    if stable != installed {
                        let mut st = state_for_brew.lock().unwrap();
                        st.update_available = true;
                    }
                }
            }
        }
    });

    // Update opencode config once server is ready and we know actual ctx_size
    let state_for_opencode = proc.state.clone();
    let config_for_opencode = config.server.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            let st = state_for_opencode.lock().unwrap();
            if st.ready && st.ctx_size > 0 {
                drop(st);
                update_opencode_config(&config_for_opencode);
                break;
            }
            if st.exited {
                break;
            }
        }
    });

    // Set up shutdown signal
    let (shutdown_tx, _shutdown_rx) = tokio::sync::watch::channel(false);

    // Create display
    let display = Display::new(proc.state.clone(), config.server.clone());

    // Monitor child process exit in background
    let state_for_monitor = proc.state.clone();
    let shutdown_tx_child = shutdown_tx.clone();
    tokio::spawn(async move {
        // We can't move proc.child here since we need it later,
        // so we just monitor via the state's exited flag
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            let st = state_for_monitor.lock().unwrap();
            if st.exited {
                let _ = shutdown_tx_child.send(true);
                break;
            }
        }
    });

    // Run display (blocks until Ctrl+C / 'q' / server exit)
    display.run(shutdown_tx).await;
    display.print_summary();

    // Kill child if still running
    let _ = proc.child.kill();
}
