// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

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

use crate::config::{websearch_port, ServerConfig, DEFAULT_BATCH_SIZE, DEFAULT_PARALLEL};
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

        // Sampling (grey sub of Model) — only if any sampler was overridden.
        if let Some(sampling) = format_sampling(&self.config) {
            self.print_sub(&mut t, &sampling);
        }

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

        // Tuning (grey sub-line under llamacpp) — effective performance knobs.
        self.print_sub(&mut t, &format_tuning(&self.config));

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

        // Web search panel — always shown when websearch is enabled, so the
        // setup URL is discoverable before the first search (the bookmarklet
        // must be installed before searches will work).
        if !self.config.websearch_disabled {
            let _ = queue!(
                t.stdout,
                Print("  "),
                SetForegroundColor(MUTED_PURPLE),
                Print("Search   : "),
                SetForegroundColor(COLOR_NUMBER),
                Print(format!("{}", st.websearch_total)),
                SetForegroundColor(Color::White),
                Print(" total · "),
                SetForegroundColor(COLOR_NUMBER),
                Print(format!("{}", st.websearch_active)),
                SetForegroundColor(Color::White),
                Print(" active"),
                ResetColor
            );
            t.newline();
            let setup_url = format!("setup: http://127.0.0.1:{}/setup", websearch_port(&self.config));
            self.print_sub(&mut t, &setup_url);
            if !st.websearch_last_query.is_empty() {
                let q = format!("last: \"{}\"", st.websearch_last_query);
                self.print_sub(&mut t, &q);
            }
        }

        // Cache health: only render when something has gone wrong. Any non-zero
        // count means llama-server redid the whole prefill at least once -
        // the smoking gun for "why did it get slow at long context".
        if st.full_reprocess_count > 0 || st.invalidated_checkpoint_count > 0 {
            let warn_color = Color::Rgb {
                r: 255,
                g: 170,
                b: 80,
            };
            let _ = queue!(
                t.stdout,
                Print("  "),
                SetForegroundColor(MUTED_PURPLE),
                Print("Cache    : "),
                SetForegroundColor(warn_color),
                Print(format!("{}", st.full_reprocess_count)),
                SetForegroundColor(Color::White),
                Print(" full reprocess · "),
                SetForegroundColor(warn_color),
                Print(format!("{}", st.invalidated_checkpoint_count)),
                SetForegroundColor(Color::White),
                Print(" invalidated checkpoints"),
                ResetColor
            );
            t.newline();
        }

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
            let _ = queue!(t.stdout, Print("    "), SetForegroundColor(COLOR_NUMBER));
            if slot.n_tokens == 0 {
                let desc = format!("● slot {} starting...", slot.slot_id);
                let _ = queue!(
                    t.stdout,
                    Print(truncate(&desc, t.width.saturating_sub(4))),
                    ResetColor
                );
            } else if slot.progress > 0.0 && slot.progress < 1.0 {
                // Still prefilling: show a little progress bar.
                let head = format!(
                    "● slot {} prefilling {} tokens  ",
                    slot.slot_id, slot.n_tokens
                );
                let pct = (slot.progress * 100.0).round() as u32;
                // ETA uses a quadratic model: elapsed ∝ progress². Attention
                // cost per prefilled token grows with prompt position, so the
                // last 30% of progress takes roughly as long as the first 70%.
                // A linear extrapolation pins at a stale value; the windowed-
                // rate approach also underestimates because it doesn't account
                // for future batches being slower than current. Quadratic is a
                // decent first-principles fit for attention-dominated prefill
                // on long prompts, and it monotonically decreases.
                let eta_str = slot
                    .processing_started
                    .and_then(|started| {
                        let p = slot.progress as f64;
                        // Wait until 20% progress - the quadratic model is
                        // noisy early on, but stabilizes well past this point.
                        if p > 0.20 {
                            let elapsed = started.elapsed().as_secs_f64();
                            let remaining = elapsed * (1.0 / (p * p) - 1.0);
                            Some(format!(" · {} left", format_eta(remaining)))
                        } else {
                            None
                        }
                    })
                    .unwrap_or_default();
                let pct_str = format!(" {:>3}%{}", pct, eta_str);
                let bar_space = t
                    .width
                    .saturating_sub(4 + head.chars().count() + pct_str.chars().count());
                let bar_width = bar_space.min(30);
                if bar_width >= 4 {
                    let filled = ((slot.progress as f64) * bar_width as f64).round() as usize;
                    let filled = filled.min(bar_width);
                    let empty = bar_width - filled;
                    let _ = queue!(
                        t.stdout,
                        Print(&head),
                        SetForegroundColor(LAVENDER),
                        Print("█".repeat(filled)),
                        SetForegroundColor(MUTED_PURPLE),
                        Print("░".repeat(empty)),
                        SetForegroundColor(COLOR_NUMBER),
                        Print(pct_str),
                        ResetColor
                    );
                } else {
                    // Not enough room for a bar; fall back to plain line.
                    let desc = format!(
                        "● slot {} prefilling {} tokens ({}%{})",
                        slot.slot_id, slot.n_tokens, pct, eta_str
                    );
                    let _ = queue!(
                        t.stdout,
                        Print(truncate(&desc, t.width.saturating_sub(4))),
                        ResetColor
                    );
                }
            } else {
                // Prefill done (or unknown): generating tokens.
                let desc = format!(
                    "● slot {} generating ({} tokens prompt)",
                    slot.slot_id, slot.n_tokens
                );
                let _ = queue!(
                    t.stdout,
                    Print(truncate(&desc, t.width.saturating_sub(4))),
                    ResetColor
                );
            }
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
            Print("\n")
        );

        // Fatal reason: lui aborted with a specific, actionable explanation
        // (e.g. GPU VRAM oversubscribed at load time). Rendered prominently in
        // red so it stands out from the generic "server exited" path below.
        if let Some(ref reason) = st.fatal_reason {
            let _ = execute!(
                stdout,
                Print("\n"),
                SetForegroundColor(Color::Red),
                SetAttribute(Attribute::Bold),
                Print("  lui aborted: "),
                SetAttribute(Attribute::Reset),
                ResetColor,
            );
            for (i, line) in reason.lines().enumerate() {
                if i == 0 {
                    let _ = execute!(
                        stdout,
                        SetForegroundColor(Color::Red),
                        Print(line),
                        ResetColor,
                        Print("\n"),
                    );
                } else {
                    let _ = execute!(
                        stdout,
                        SetForegroundColor(Color::White),
                        Print(line),
                        ResetColor,
                        Print("\n"),
                    );
                }
            }
            println!();
            let _ = stdout.flush();
            return;
        }
        // Show exit message if the server exited unexpectedly (never became ready)
        if st.exited && !st.ready {
            let _ = execute!(
                stdout,
                Print("\n"),
                SetForegroundColor(Color::Red),
                SetAttribute(Attribute::Bold),
                Print("  Server exited before becoming ready."),
                SetAttribute(Attribute::Reset),
                ResetColor,
                Print("\n")
            );
            if !st.exit_message.is_empty() {
                let _ = execute!(
                    stdout,
                    Print("  "),
                    SetForegroundColor(Color::DarkGrey),
                    Print(&st.exit_message),
                    ResetColor,
                    Print("\n")
                );
            }

            // Show last log lines so the user can see the error
            let lines: Vec<&String> = st.log_lines.iter().collect();
            let show = lines.len().min(20);
            if show > 0 {
                let _ = execute!(
                    stdout,
                    Print("\n"),
                    SetForegroundColor(LAVENDER),
                    Print("  Last server output:\n"),
                    ResetColor,
                );
                for line in &lines[lines.len() - show..] {
                    let _ = execute!(
                        stdout,
                        Print("  "),
                        SetForegroundColor(Color::DarkGrey),
                        Print(line),
                        ResetColor,
                        Print("\n")
                    );
                }
            }
        }

        println!();
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

fn format_sampling(cfg: &ServerConfig) -> Option<String> {
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

fn format_tuning(cfg: &ServerConfig) -> String {
    let np = cfg.parallel.unwrap_or(DEFAULT_PARALLEL);
    let mut parts = vec![format!("np={}", np)];
    if let Some(ub) = cfg.ubatch_size {
        parts.push(format!("ubatch={}", ub));
    }

    let ctk = cfg.cache_type_k.as_deref().unwrap_or("f16");
    let ctv = cfg.cache_type_v.as_deref().unwrap_or("f16");
    if ctk != "f16" || ctv != "f16" {
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

fn format_eta(seconds: f64) -> String {
    let s = seconds.round() as u64;
    if s < 60 {
        format!("{}s", s)
    } else if s < 3600 {
        let m = s / 60;
        let sec = s % 60;
        if m < 10 {
            format!("{}m {:02}s", m, sec)
        } else {
            format!("{}m", m)
        }
    } else {
        let h = s / 3600;
        let m = (s % 3600) / 60;
        format!("{}h {:02}m", h, m)
    }
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
