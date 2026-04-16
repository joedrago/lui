use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossterm::{
    cursor::{Hide, MoveTo, Show},
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute, queue,
    style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor},
    terminal::{self, Clear, ClearType, DisableLineWrap, EnableLineWrap},
};

use crate::config::ServerConfig;
use crate::server::ServerState;

const RENDER_INTERVAL_MS: u64 = 250;
const LAVENDER: Color = Color::Rgb {
    r: 180,
    g: 150,
    b: 255,
};
const MUTED_PURPLE: Color = Color::Rgb {
    r: 120,
    g: 100,
    b: 180,
};
const COLOR_NUMBER: Color = Color::Rgb {
    r: 210,
    g: 150,
    b: 255,
};

pub struct Display {
    state: Arc<Mutex<ServerState>>,
    config: ServerConfig,
    start_time: Instant,
}

/// A buffered line writer that tracks the current row and pads/clears each line.
/// All output goes through this to avoid flicker and scrollback pollution.
struct TermBuf<'a> {
    stdout: &'a mut io::Stdout,
    row: u16,
    width: usize,
    height: u16,
}

impl<'a> TermBuf<'a> {
    fn new(stdout: &'a mut io::Stdout, width: usize, height: u16) -> Self {
        let _ = queue!(stdout, MoveTo(0, 0));
        TermBuf {
            stdout,
            row: 0,
            width,
            height,
        }
    }

    /// Write a line (already formatted with crossterm queue! commands), then clear to EOL and advance.
    fn newline(&mut self) {
        if self.row < self.height {
            let _ = queue!(
                self.stdout,
                Clear(ClearType::UntilNewLine),
                MoveTo(0, self.row + 1)
            );
            self.row += 1;
        }
    }

    /// Clear all remaining rows below the cursor.
    fn clear_rest(&mut self) {
        while self.row < self.height {
            let _ = queue!(
                self.stdout,
                MoveTo(0, self.row),
                Clear(ClearType::UntilNewLine)
            );
            self.row += 1;
        }
    }

    fn remaining(&self) -> usize {
        self.height.saturating_sub(self.row) as usize
    }

    fn flush(&mut self) {
        let _ = self.stdout.flush();
    }
}

impl Display {
    pub fn new(state: Arc<Mutex<ServerState>>, config: ServerConfig) -> Self {
        Display {
            state,
            config,
            start_time: Instant::now(),
        }
    }

    pub async fn run(&self, shutdown_tx: tokio::sync::watch::Sender<bool>) {
        let mut stdout = io::stdout();
        // Clear screen once at startup, hide cursor, disable line wrap
        let _ = execute!(
            stdout,
            Hide,
            DisableLineWrap,
            Clear(ClearType::All),
            Clear(ClearType::Purge),
            MoveTo(0, 0)
        );
        let _ = terminal::enable_raw_mode();

        loop {
            self.render();

            if Self::check_quit() {
                let _ = shutdown_tx.send(true);
                break;
            }

            {
                let st = self.state.lock().unwrap();
                if st.exited {
                    break;
                }
            }

            tokio::time::sleep(Duration::from_millis(RENDER_INTERVAL_MS)).await;
        }

        let _ = terminal::disable_raw_mode();
        let _ = execute!(
            stdout,
            Clear(ClearType::All),
            MoveTo(0, 0),
            EnableLineWrap,
            Show
        );
    }

    fn check_quit() -> bool {
        if event::poll(Duration::ZERO).unwrap_or(false) {
            if let Ok(Event::Key(KeyEvent {
                code, modifiers, ..
            })) = event::read()
            {
                if code == KeyCode::Char('c') && modifiers.contains(KeyModifiers::CONTROL) {
                    return true;
                }
                if code == KeyCode::Char('q') {
                    return true;
                }
            }
        }
        false
    }

    fn render(&self) {
        let mut stdout = io::stdout();
        let st = self.state.lock().unwrap();
        let (term_width, term_height) = terminal::size().unwrap_or((80, 24));
        let width = term_width as usize;

        let mut t = TermBuf::new(&mut stdout, width, term_height);

        // Header
        let left = "  ── ";
        let mid_text = "lui";
        let right_text = " ── llama.cpp ui ";
        let right_fill = width.saturating_sub(left.len() + mid_text.len() + right_text.len());
        let _ = queue!(
            t.stdout,
            SetForegroundColor(MUTED_PURPLE),
            Print(left),
            SetForegroundColor(LAVENDER),
            SetAttribute(Attribute::Bold),
            Print(mid_text),
            SetAttribute(Attribute::Reset),
            SetForegroundColor(MUTED_PURPLE),
            Print(right_text),
            Print("─".repeat(right_fill)),
            ResetColor
        );
        t.newline();

        if !st.ready {
            t.newline();

            if !st.downloads.is_empty() {
                // Show download progress
                let bar_width = width.saturating_sub(30);
                let mut downloads: Vec<_> = st.downloads.iter().collect();
                downloads.sort_by_key(|(name, _)| (*name).clone());
                for (name, pct) in &downloads {
                    let filled = ((**pct as usize) * bar_width) / 100;
                    let empty = bar_width.saturating_sub(filled);
                    let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));
                    let _ = queue!(
                        t.stdout,
                        Print("  "),
                        SetForegroundColor(Color::White),
                        Print(truncate(name, 20)),
                    );
                    // Pad name to 20 chars
                    let name_len = name.chars().count().min(20);
                    let _ = queue!(
                        t.stdout,
                        Print(" ".repeat(21 - name_len)),
                        SetForegroundColor(LAVENDER),
                        Print(&bar),
                        Print(" "),
                        SetForegroundColor(COLOR_NUMBER),
                        Print(format!("{:>3}%", pct)),
                        ResetColor
                    );
                    t.newline();
                }
            } else {
                let _ = queue!(
                    t.stdout,
                    Print("  "),
                    SetForegroundColor(COLOR_NUMBER),
                    Print("Loading model..."),
                    ResetColor
                );
                t.newline();
            }

            t.newline();
            self.render_source(&mut t);
            t.newline();

            self.render_log(&st, &mut t);
            t.clear_rest();
            t.flush();
            return;
        }

        t.newline();

        // Memory
        if st.gpu_mem_mib > 0.0 || st.kv_cache_mib > 0.0 {
            let gpu_total = st.gpu_mem_mib + st.kv_cache_mib + st.compute_buf_mib;
            let _ = queue!(
                t.stdout,
                Print("  "),
                SetForegroundColor(MUTED_PURPLE),
                Print("Memory   : "),
                SetForegroundColor(COLOR_NUMBER),
                Print(format!("{:.1}", gpu_total / 1024.0)),
                SetForegroundColor(Color::White),
                Print(" GiB VRAM"),
            );
            if st.cpu_mem_mib > 0.0 {
                let _ = queue!(
                    t.stdout,
                    SetForegroundColor(Color::White),
                    Print(" · "),
                    SetForegroundColor(COLOR_NUMBER),
                    Print(format!("{:.0}", st.cpu_mem_mib)),
                    SetForegroundColor(Color::White),
                    Print(" MiB RAM"),
                );
            }
            let _ = queue!(t.stdout, ResetColor);
            t.newline();

            // Breakdown on next line in grey
            let mut parts = Vec::new();
            if st.gpu_mem_mib > 0.0 {
                parts.push(format!("{:.0} model", st.gpu_mem_mib));
            }
            if st.kv_cache_mib > 0.0 {
                parts.push(format!("{:.0} KV", st.kv_cache_mib));
            }
            if st.compute_buf_mib > 0.0 {
                parts.push(format!("{:.0} compute", st.compute_buf_mib));
            }
            let breakdown = format!("{} MiB", parts.join(" + "));
            self.print_sub(&mut t, &breakdown);

            // GPU offload as second grey line under Memory
            if st.total_layers > 0 {
                let status = if st.gpu_layers_loaded == st.total_layers {
                    "fully GPU"
                } else if st.gpu_layers_loaded == 0 {
                    "CPU only"
                } else {
                    "partial"
                };
                let gpu_line = format!(
                    "{}/{} layers offloaded ({})",
                    st.gpu_layers_loaded, st.total_layers, status
                );
                self.print_sub(&mut t, &gpu_line);
            }
        }

        // Model
        let model_display = if st.quantization.is_empty() {
            st.model_name.clone()
        } else {
            format!("{} ({})", st.model_name, st.quantization)
        };
        self.print_kv(&mut t, "Model   ", &model_display, Color::White);

        // Source (grey sub of Model)
        self.render_source(&mut t);

        // Params (grey sub of Model)
        if !st.model_params_n.is_empty() {
            let params_display = if st.file_size_n.is_empty() {
                format!("{} {}", st.model_params_n, st.model_params_unit)
            } else if st.file_bpw.is_empty() {
                format!(
                    "{} {} · {} {} on disk",
                    st.model_params_n, st.model_params_unit, st.file_size_n, st.file_size_unit
                )
            } else {
                format!(
                    "{} {} · {} {} on disk ({} BPW)",
                    st.model_params_n,
                    st.model_params_unit,
                    st.file_size_n,
                    st.file_size_unit,
                    st.file_bpw
                )
            };
            self.print_sub(&mut t, &params_display);
        }

        // Context (grey sub of Model)
        let ctx_display = if st.max_ctx_size > 0 && st.max_ctx_size != st.ctx_size {
            format!(
                "{} token context window ({} max)",
                format_number(st.ctx_size as u64),
                format_number(st.max_ctx_size as u64)
            )
        } else {
            format!("{} token context window", format_number(st.ctx_size as u64))
        };
        self.print_sub(&mut t, &ctx_display);

        // llamacpp + status
        let uptime = self.start_time.elapsed();
        let uptime_str = format_duration(uptime);
        let _ = queue!(
            t.stdout,
            Print("  "),
            SetForegroundColor(MUTED_PURPLE),
            Print("llamacpp : "),
            SetForegroundColor(Color::Green),
            Print("Ready"),
        );
        if !st.llama_version.is_empty() {
            let _ = queue!(
                t.stdout,
                SetForegroundColor(Color::White),
                Print(format!(" ({}, uptime: {})", st.llama_version, uptime_str)),
            );
        } else {
            let _ = queue!(
                t.stdout,
                SetForegroundColor(Color::White),
                Print(format!(" (uptime: {})", uptime_str)),
            );
        }
        if st.update_available {
            let _ = queue!(
                t.stdout,
                Print("  "),
                SetForegroundColor(Color::Green),
                Print("(update available)"),
            );
        }
        let _ = queue!(t.stdout, ResetColor);
        t.newline();

        // Bind (grey sub-line under llamacpp)
        let bind_display = format!("{}:{}", self.config.host, self.config.port);
        self.print_sub(&mut t, &bind_display);

        // Performance section
        t.newline();
        let perf_header = format!("  ── Performance {}", "─".repeat(width.saturating_sub(18)));
        let _ = queue!(
            t.stdout,
            SetForegroundColor(MUTED_PURPLE),
            Print(truncate(&perf_header, width)),
            ResetColor
        );
        t.newline();

        if st.prompt_tps_samples > 0 {
            self.print_tps(&mut t, "Prompt  ", st.last_prompt_tps, st.avg_prompt_tps);
        }
        if st.gen_tps_samples > 0 {
            self.print_tps(&mut t, "Generate", st.last_gen_tps, st.avg_gen_tps);
        }

        let _ = queue!(
            t.stdout,
            Print("  "),
            SetForegroundColor(MUTED_PURPLE),
            Print("Requests : "),
            SetForegroundColor(COLOR_NUMBER),
            Print(format!("{}", st.request_count)),
            SetForegroundColor(Color::White),
            Print(" total · "),
            SetForegroundColor(COLOR_NUMBER),
            Print(format!("{}", st.active_requests)),
            SetForegroundColor(Color::White),
            Print(" active"),
            ResetColor
        );
        t.newline();

        if st.prompt_tps_samples == 0 && st.gen_tps_samples == 0 {
            let _ = queue!(
                t.stdout,
                Print("    "),
                SetForegroundColor(Color::DarkGrey),
                Print("No requests yet"),
                ResetColor
            );
            t.newline();
        }

        // Active slots
        for slot in st.active_slots.values() {
            let desc = if slot.n_tokens > 0 {
                format!(
                    "● slot {} processing {} tokens",
                    slot.slot_id, slot.n_tokens
                )
            } else {
                format!("● slot {} starting...", slot.slot_id)
            };
            let _ = queue!(
                t.stdout,
                Print("    "),
                SetForegroundColor(COLOR_NUMBER),
                Print(truncate(&desc, t.width.saturating_sub(4))),
                ResetColor
            );
            t.newline();
        }

        // Recent completed
        for slot in st.recent_completed.iter().rev() {
            let time_str = if slot.total_time_ms > 0.0 {
                format!(" in {:.1}s", slot.total_time_ms / 1000.0)
            } else {
                String::new()
            };
            let tps_str = if slot.gen_tps > 0.0 {
                format!(" ({:.1} tok/s)", slot.gen_tps)
            } else {
                String::new()
            };
            let desc = format!(
                "✓ slot {} done {} tokens{}{}",
                slot.slot_id, slot.n_tokens, time_str, tps_str
            );
            let _ = queue!(
                t.stdout,
                Print("    "),
                SetForegroundColor(Color::DarkGrey),
                Print(truncate(&desc, t.width.saturating_sub(4))),
                ResetColor
            );
            t.newline();
        }

        // Server log -- fills remaining space
        t.newline();
        self.render_log(&st, &mut t);
        t.clear_rest();
        t.flush();
    }

    fn render_source(&self, t: &mut TermBuf) {
        let source = if !self.config.hf_repo.is_empty() {
            format!("--hf {}", self.config.hf_repo)
        } else if !self.config.model.is_empty() {
            format!("-m {}", self.config.model)
        } else {
            "none".to_string()
        };
        self.print_sub(t, &source);
    }

    fn print_tps(&self, t: &mut TermBuf, label: &str, last: f64, avg: f64) {
        let _ = queue!(
            t.stdout,
            Print("  "),
            SetForegroundColor(MUTED_PURPLE),
            Print(label),
            Print(" : "),
            SetForegroundColor(COLOR_NUMBER),
            Print(format!("{:>8.1}", last)),
            SetForegroundColor(Color::White),
            Print(" tokens/s (last)  "),
            SetForegroundColor(COLOR_NUMBER),
            Print(format!("{:>8.1}", avg)),
            SetForegroundColor(Color::White),
            Print(" tokens/s (avg)"),
            ResetColor
        );
        t.newline();
    }

    fn print_sub(&self, t: &mut TermBuf, value: &str) {
        let prefix_len = 17;
        let max_val = t.width.saturating_sub(prefix_len);
        let _ = queue!(
            t.stdout,
            Print("                 "),
            SetForegroundColor(Color::DarkGrey),
            Print(truncate(value, max_val)),
            ResetColor
        );
        t.newline();
    }

    fn print_kv(&self, t: &mut TermBuf, label: &str, value: &str, color: Color) {
        let prefix_len = 2 + label.len() + 3;
        let max_val = t.width.saturating_sub(prefix_len);
        let _ = queue!(
            t.stdout,
            Print("  "),
            SetForegroundColor(MUTED_PURPLE),
            Print(label),
            Print(" : "),
            SetForegroundColor(color),
            Print(truncate(value, max_val)),
            ResetColor
        );
        t.newline();
    }

    fn render_log(&self, st: &ServerState, t: &mut TermBuf) {
        let log_header = format!("  ── Server Log {}", "─".repeat(t.width.saturating_sub(17)));
        let _ = queue!(
            t.stdout,
            SetForegroundColor(MUTED_PURPLE),
            Print(truncate(&log_header, t.width)),
            ResetColor
        );
        t.newline();

        let show = t.remaining().saturating_sub(1).max(1);
        let lines: Vec<&String> = st.log_lines.iter().collect();
        let start = lines.len().saturating_sub(show);
        for line in &lines[start..] {
            let display = truncate(line, t.width.saturating_sub(4));
            let _ = queue!(
                t.stdout,
                Print("  "),
                SetForegroundColor(Color::DarkGrey),
                Print(&display),
                ResetColor
            );
            t.newline();
        }
    }

    pub fn print_summary(&self) {
        let st = self.state.lock().unwrap();
        let mut stdout = io::stdout();
        let _ = execute!(stdout, Show);

        let uptime = self.start_time.elapsed();

        println!();
        let _ = execute!(
            stdout,
            SetForegroundColor(LAVENDER),
            SetAttribute(Attribute::Bold),
            Print("lui"),
            SetAttribute(Attribute::Reset),
            ResetColor,
            Print(" shutting down\n")
        );

        if !st.model_name.is_empty() {
            let _ = execute!(
                stdout,
                Print("  Model   : "),
                SetForegroundColor(Color::White),
                Print(&st.model_name),
                ResetColor,
                Print("\n")
            );
        }
        let _ = execute!(
            stdout,
            Print("  Uptime  : "),
            SetForegroundColor(Color::White),
            Print(format_duration(uptime)),
            ResetColor,
            Print("\n")
        );
        let _ = execute!(
            stdout,
            Print("  Requests: "),
            SetForegroundColor(Color::White),
            Print(format!("{}", st.request_count)),
            ResetColor,
            Print("\n\n")
        );

        let _ = stdout.flush();
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else if max_len <= 3 {
        s.chars().take(max_len).collect()
    } else {
        let truncated: String = s.chars().take(max_len - 3).collect();
        format!("{}...", truncated)
    }
}

fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    let hours = secs / 3600;
    let mins = (secs % 3600) / 60;
    if hours > 0 {
        format!("{}h {:02}m", hours, mins)
    } else if mins > 0 {
        format!("{}m", mins)
    } else {
        "<1m".to_string()
    }
}
