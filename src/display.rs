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
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::timeout;

use crate::server::{ServerState, SlotSnapshot, UiSnapshot, UI_SNAPSHOT_VERSION};

const RENDER_INTERVAL_MS: u64 = 250;
/// Timeout for a single `/data` poll. Well under the render interval so a
/// stuck localhost fetch can't wedge the UI — we fall back to the last
/// good snapshot and try again on the next tick.
const FETCH_TIMEOUT_MS: u64 = 200;
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
const COLOR_ALIASES: Color = Color::Rgb {
    r: 220,
    g: 215,
    b: 230,
};
const WARNING_AMBER: Color = Color::Rgb {
    r: 230,
    g: 180,
    b: 80,
};

/// How long the setup URL stays bright cyan before switching to dark grey.
const SETUP_URL_START_MS: u64 = 1_500;
// Reuse COLOR_NUMBER (lavender-purple) for the remote indicator so it
// visually groups with the other numeric/key-value elements on screen.

pub struct Display {
    /// Host to poll `/data` on. `"127.0.0.1"` locally; a server's hostname
    /// when this Display is driving a client view.
    snapshot_host: String,
    /// Port of the lui HTTP server (`web_port`), not llama-server's port.
    snapshot_port: u16,
    /// The ServerState backing this renderer's *local* HTTP server. In
    /// local mode it's the server's own state (log ring, websearch counts,
    /// the lot). In `--remote` mode it's the client's in-process bsearch
    /// state — websearch counts reflect opencode's searches through our
    /// local bsearch (which is what the user cares about; the remote
    /// server's counts are meaningless when opencode runs on this machine).
    local_state: Option<Arc<Mutex<ServerState>>>,
    /// URL of the bookmarklet `/setup` page the user should open in their
    /// *local* browser. Always a 127.0.0.1 URL — browser-mediated search
    /// is a this-machine-has-a-user concern, not a server-side concern. `None`
    /// when this machine isn't running a bsearch server (e.g. local mode
    /// with `--no-websearch`). The renderer shows it in the top-right
    /// corner when `Some`.
    local_setup_url: Option<String>,
    /// When this Display was created. Only consulted by `print_summary`
    /// (which runs after the poll loop stops and therefore has no
    /// `/data` response in hand). The main render loop uses
    /// `UiSnapshot::uptime_seconds` instead, so a client renderer agrees
    /// with the server on actual lifetime.
    start_time: Instant,
    /// The remote server host when running in `--remote` mode. Rendered as
    /// `[Remote: HOST]` in the header so the user always knows whether they
    /// are looking at a local server or a remote one.
    remote_host: Option<String>,
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

/// Result of a `/data` poll.
///
/// `Retry(reason)` means no usable snapshot was produced — connect timeout,
/// non-2xx HTTP status, parse failure, etc. The caller should keep polling.
/// The reason string is surfaced on the "Starting..." screen so a stuck
/// client tells the user *why* it's stuck instead of spinning silently.
///
/// `Ok(snap)` means a valid snapshot arrived.
///
/// `VersionMismatch(msg)` means we got a response but the schema is
/// incompatible. This is a permanent error — the caller should bail
/// out with the message rather than retrying forever.
enum FetchResult {
    Retry(String),
    Ok(Box<UiSnapshot>),
    VersionMismatch(String),
}

impl Display {
    pub fn new(
        snapshot_host: String,
        snapshot_port: u16,
        local_state: Option<Arc<Mutex<ServerState>>>,
        local_setup_url: Option<String>,
        remote_host: Option<String>,
    ) -> Self {
        Display {
            snapshot_host,
            snapshot_port,
            local_state,
            local_setup_url,
            start_time: Instant::now(),
            remote_host,
        }
    }

    pub async fn run(&self, shutdown_tx: tokio::sync::watch::Sender<bool>) {
        let mut stdout = io::stdout();
        // Clear the visible area once at startup; deliberately do NOT
        // Clear(ClearType::Purge) so any banner printed before we started
        // (notably `--remote`'s setup-info banner) survives in scrollback
        // and the user can scroll up to recover the URLs / opencode
        // status after the Display takes over the screen.
        let _ = execute!(
            stdout,
            Hide,
            DisableLineWrap,
            Clear(ClearType::All),
            MoveTo(0, 0)
        );
        let _ = terminal::enable_raw_mode();

        // Last successful snapshot. Polls that fail (server not up yet,
        // transient error) leave this unchanged so the UI stays on the
        // last-good frame instead of flickering to empty.
        let mut last: Option<UiSnapshot> = None;
        // Error message from a fatal condition (e.g. version mismatch).
        // Rendered on-screen before exiting so the user doesn't have to
        // scroll up to find it.
        let mut fatal_error: Option<String> = None;
        // Why the most recent /data poll failed, plus how many failures
        // we've seen so far. Surfaced on the "Starting..." screen so a
        // stuck client tells the user what's going wrong instead of
        // spinning silently. Both are reset implicitly once `last` is
        // Some — the success path renders `render(snap)`, never
        // `render_starting`, so stale diagnostics never leak through.
        let mut last_retry_reason: Option<String> = None;
        let mut retry_count: u64 = 0;

        loop {
            match self.fetch_snapshot().await {
                FetchResult::Retry(reason) => {
                    retry_count += 1;
                    last_retry_reason = Some(reason);
                }
                FetchResult::Ok(snap) => last = Some(*snap),
                FetchResult::VersionMismatch(msg) => {
                    // Permanent incompatibility — print a clear error and
                    // exit rather than spinning on "Starting..." forever.
                    fatal_error = Some(msg);
                    break;
                }
            }

            match &last {
                Some(snap) => self.render(snap),
                None => self.render_starting(last_retry_reason.as_deref(), retry_count),
            }

            if Self::check_quit() {
                let _ = shutdown_tx.send(true);
                break;
            }

            if last.as_ref().map(|s| s.exited).unwrap_or(false) {
                break;
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

        // Render any fatal error after giving back the terminal so the
        // message is visible without scrolling.
        if let Some(ref msg) = fatal_error {
            eprintln!("lui: {}", msg);
        }
    }

    async fn fetch_snapshot(&self) -> FetchResult {
        let t = Duration::from_millis(FETCH_TIMEOUT_MS);
        let addr = format!("{}:{}", self.snapshot_host, self.snapshot_port);

        let mut stream = match timeout(t, TcpStream::connect(&addr)).await {
            Err(_) => {
                return FetchResult::Retry(format!(
                    "connect to {} timed out after {}ms",
                    addr, FETCH_TIMEOUT_MS
                ))
            }
            Ok(Err(e)) => {
                return FetchResult::Retry(format!("connect to {} failed: {}", addr, e))
            }
            Ok(Ok(s)) => s,
        };
        let req = "GET /data HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nAccept: application/json\r\n\r\n";
        match timeout(t, stream.write_all(req.as_bytes())).await {
            Err(_) => {
                return FetchResult::Retry(format!(
                    "write to {} timed out after {}ms",
                    addr, FETCH_TIMEOUT_MS
                ))
            }
            Ok(Err(e)) => {
                return FetchResult::Retry(format!("write to {} failed: {}", addr, e))
            }
            Ok(Ok(())) => {}
        }

        let mut buf = Vec::new();
        match timeout(t, stream.read_to_end(&mut buf)).await {
            Err(_) => {
                return FetchResult::Retry(format!(
                    "read from {} timed out after {}ms",
                    addr, FETCH_TIMEOUT_MS
                ))
            }
            Ok(Err(e)) => {
                return FetchResult::Retry(format!("read from {} failed: {}", addr, e))
            }
            Ok(Ok(_)) => {}
        }

        let text = match String::from_utf8(buf) {
            Ok(s) => s,
            Err(_) => {
                return FetchResult::Retry(format!("non-UTF8 response from {}", addr))
            }
        };

        // Status line: "HTTP/1.1 NNN ...\r\n". Surface non-2xx statuses as
        // themselves so a 404 / 503 from a misrouted or still-warming server
        // doesn't get swallowed into a downstream JSON parse error.
        let status_line = match text.lines().next() {
            Some(s) => s,
            None => return FetchResult::Retry(format!("empty response from {}", addr)),
        };
        match status_line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<u16>().ok())
        {
            Some(c) if (200..300).contains(&c) => {}
            Some(c) => {
                return FetchResult::Retry(format!("HTTP {} from {}/data", c, addr))
            }
            None => {
                return FetchResult::Retry(format!(
                    "malformed status line from {}: {}",
                    addr,
                    status_line.chars().take(80).collect::<String>()
                ))
            }
        }

        let body_start = match text.find("\r\n\r\n") {
            Some(p) => p + 4,
            None => {
                return FetchResult::Retry(format!(
                    "no header/body separator in response from {}",
                    addr
                ))
            }
        };
        let body_section = &text[body_start..];
        // Axum emits Content-Length (no chunked) for Json<T>, and we set
        // Connection: close, so the body is a plain byte slice with no
        // dechunking needed. But in case a proxy ever intervenes, tolerate
        // stray trailing whitespace by trimming.
        let snap: UiSnapshot = match serde_json::from_str(body_section.trim()) {
            Ok(s) => s,
            Err(e) => {
                return FetchResult::Retry(format!(
                    "JSON parse from {}/data: {}",
                    addr, e
                ))
            }
        };
        if snap.version != UI_SNAPSHOT_VERSION {
            // Version mismatch means the server's /data schema is newer
            // (or older) than what we can parse. This is a permanent
            // incompatibility — bail out instead of retrying forever.
            return FetchResult::VersionMismatch(format!(
                "server /data version is incompatible (expected {}, got {}). Upgrade the older side.",
                UI_SNAPSHOT_VERSION, snap.version
            ));
        }
        FetchResult::Ok(Box::new(snap))
    }

    /// Frame shown until the first successful `/data` poll. In local mode
    /// this is typically one tick; in `--remote` mode it can persist if
    /// the client can't reach the server, so we surface the polled URL,
    /// attempt count, elapsed time, and the most recent failure reason
    /// underneath the "Starting..." line. That turns a silent hang into a
    /// debuggable one — the user can see whether they're getting connect
    /// timeouts, non-2xx HTTP, or schema/parse errors.
    fn render_starting(&self, last_error: Option<&str>, attempts: u64) {
        let mut stdout = io::stdout();
        let (term_width, term_height) = terminal::size().unwrap_or((80, 24));
        let width = (term_width as usize).saturating_sub(2);
        let mut t = TermBuf::new(&mut stdout, width, term_height);
        let _ = queue!(
            t.stdout,
            SetForegroundColor(MUTED_PURPLE),
            Print("  ── lui ── llama.cpp ui "),
            ResetColor
        );
        if let Some(ref host) = self.remote_host {
            let _ = queue!(
                t.stdout,
                SetForegroundColor(COLOR_NUMBER),
                SetAttribute(Attribute::Bold),
                Print("  ["),
                Print(host),
                Print("]"),
                ResetColor
            );
        }
        t.newline();
        t.newline();
        let _ = queue!(
            t.stdout,
            Print("  "),
            SetForegroundColor(COLOR_NUMBER),
            Print("Starting..."),
            ResetColor
        );
        t.newline();
        t.newline();

        let url = format!(
            "http://{}:{}/data",
            self.snapshot_host, self.snapshot_port
        );
        let _ = queue!(
            t.stdout,
            Print("  Polling "),
            SetForegroundColor(COLOR_NUMBER),
            Print(&url),
            ResetColor
        );
        t.newline();

        if let Some(err) = last_error {
            let elapsed = self.start_time.elapsed().as_secs();
            let _ = queue!(
                t.stdout,
                Print("  Attempts: "),
                SetForegroundColor(COLOR_NUMBER),
                Print(format!("{}", attempts)),
                ResetColor,
                Print("  Elapsed: "),
                SetForegroundColor(COLOR_NUMBER),
                Print(format!("{}s", elapsed)),
                ResetColor
            );
            t.newline();

            // Truncate to the visible width minus indent + "Last error: "
            // prefix. crossterm/Print trusts us to fit the line; long
            // serde errors otherwise wrap and shove the rest of the
            // screen down.
            let prefix = "  Last error: ";
            let avail = width.saturating_sub(prefix.len());
            let truncated: String = if err.chars().count() > avail {
                let head: String = err.chars().take(avail.saturating_sub(1)).collect();
                format!("{}…", head)
            } else {
                err.to_string()
            };
            let _ = queue!(
                t.stdout,
                Print(prefix),
                SetForegroundColor(WARNING_AMBER),
                Print(&truncated),
                ResetColor
            );
            t.newline();
        }

        t.clear_rest();
        t.flush();
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

    fn render(&self, snap: &UiSnapshot) {
        let mut stdout = io::stdout();
        let st = snap;
        let (term_width, term_height) = terminal::size().unwrap_or((80, 24));
        // Windows Terminal truncates characters written to the last column(s)
        // (e.g. the trailing "p" of "/setup" disappears). Treat the usable
        // width as 2 columns shy of reported width so rules, the right-
        // aligned setup URL, and header fills all stay inside the safe zone.
        let width = (term_width as usize).saturating_sub(2);

        let mut t = TermBuf::new(&mut stdout, width, term_height);

        // Header: "  ── lui ── llama.cpp ui ─────────"
        // Width math is in display *columns*, not bytes — `─` is 3 bytes but
        // 1 column, so using .len() on strings containing it under-counts.
        let left = "  ── ";
        let mid_text = "lui";
        let right_text = " ── llama.cpp ui ";
        let prefix_cols =
            left.chars().count() + mid_text.chars().count() + right_text.chars().count();
        let remote_cols = self
            .remote_host
            .as_deref()
            .map_or(0, |h| 1 + 2 + h.chars().count());
        let right_fill = width.saturating_sub(prefix_cols + remote_cols);
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
        if let Some(ref host) = self.remote_host {
            let _ = queue!(
                t.stdout,
                SetForegroundColor(MUTED_PURPLE),
                Print(" ["),
                Print(host),
                Print("]"),
                ResetColor
            );
        }
        t.newline();

        if !st.ready {
            t.newline();

            if !st.downloads.is_empty() {
                // Show download progress (already sorted by filename server-side).
                let bar_width = width.saturating_sub(30);
                for dl in &st.downloads {
                    let filled = ((dl.pct as usize) * bar_width) / 100;
                    let empty = bar_width.saturating_sub(filled);
                    let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));
                    let _ = queue!(
                        t.stdout,
                        Print("  "),
                        SetForegroundColor(Color::White),
                        Print(truncate(&dl.filename, 20)),
                    );
                    // Pad name to 20 chars
                    let name_len = dl.filename.chars().count().min(20);
                    let _ = queue!(
                        t.stdout,
                        Print(" ".repeat(21 - name_len)),
                        SetForegroundColor(LAVENDER),
                        Print(&bar),
                        Print(" "),
                        SetForegroundColor(COLOR_NUMBER),
                        Print(format!("{:>3}%", dl.pct)),
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
            self.render_source(&mut t, snap);
            t.newline();

            Display::render_log(snap, &mut t);
            t.clear_rest();
            t.flush();
            return;
        }

        // Blank line below the header doubles as the always-visible home of
        // the websearch setup URL, right-aligned. Starts bold lavender and
        // fades to dark grey over the first 7 seconds.
        if let Some(ref url) = self.local_setup_url {
            let pad = width.saturating_sub(url.chars().count());
            let url_color = self.setup_url_color();
            let bold = url_color == Color::Cyan;
            let _ = queue!(
                t.stdout,
                Print(" ".repeat(pad)),
                SetForegroundColor(url_color),
            );
            if bold {
                let _ = queue!(t.stdout, SetAttribute(Attribute::Bold));
            }
            let _ = queue!(t.stdout, Print(url));
            if bold {
                let _ = queue!(t.stdout, SetAttribute(Attribute::Reset));
            }
            let _ = queue!(t.stdout, ResetColor);
        }
        t.newline();

        // Memory
        if st.gpu_mem_mib > 0.0 || st.kv_cache_mib > 0.0 {
            // On Metal, `MTL0_Mapped model buffer size` reports the size of
            // the whole-file mmap, which does NOT shrink when MoE experts
            // spill to CPU_REPACK — the spilled tensors are duplicated into
            // host RAM, but the Metal buffer still covers them. Subtract the
            // CPU_REPACK total to recover the GPU's actual working set.
            // `cpu_repack_mib` is 0 on CUDA (which routes spill through
            // CPU_Mapped and accurately shrinks its own GPU buffer), so this
            // is a no-op there.
            let gpu_model_active = (st.gpu_mem_mib - st.cpu_repack_mib).max(0.0);
            let gpu_total = gpu_model_active + st.kv_cache_mib + st.compute_buf_mib;
            let cpu_total = st.cpu_mem_mib + st.cpu_repack_mib + st.cpu_compute_mib;
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
            if cpu_total > 0.0 {
                let _ = queue!(
                    t.stdout,
                    SetForegroundColor(Color::White),
                    Print(" · "),
                    SetForegroundColor(COLOR_NUMBER),
                    Print(format!("{:.0}", cpu_total)),
                    SetForegroundColor(Color::White),
                    Print(" MiB RAM"),
                );
            }
            let _ = queue!(t.stdout, ResetColor);
            t.newline();

            // Breakdown on next line in grey. When any CPU-side buffers exist,
            // split the line into GPU and CPU halves so the user can see where
            // every MiB went; otherwise the single-side form is less cluttered.
            let mut gpu_parts = Vec::new();
            if gpu_model_active > 0.0 {
                gpu_parts.push(format!("{:.0} model", gpu_model_active));
            }
            if st.kv_cache_mib > 0.0 {
                gpu_parts.push(format!("{:.0} KV", st.kv_cache_mib));
            }
            if st.compute_buf_mib > 0.0 {
                gpu_parts.push(format!("{:.0} compute", st.compute_buf_mib));
            }
            let mut cpu_parts = Vec::new();
            if st.cpu_mem_mib > 0.0 {
                cpu_parts.push(format!("{:.0} model", st.cpu_mem_mib));
            }
            if st.cpu_repack_mib > 0.0 {
                cpu_parts.push(format!("{:.0} expert", st.cpu_repack_mib));
            }
            if st.cpu_compute_mib > 0.0 {
                cpu_parts.push(format!("{:.0} compute", st.cpu_compute_mib));
            }
            let breakdown = if cpu_parts.is_empty() {
                format!("{} MiB", gpu_parts.join(" + "))
            } else {
                format!(
                    "GPU: {} MiB · CPU: {} MiB",
                    gpu_parts.join(" + "),
                    cpu_parts.join(" + ")
                )
            };
            self.print_wrapped(&mut t, &breakdown);

            // GPU offload as second grey line under Memory.
            //
            // llama.cpp's "offloaded X/Y layers to GPU" count lies on modern
            // MoE models: when experts spill back to host RAM via the MoE fit
            // path, every layer is still reported as "offloaded" even though
            // its weights mostly live in CPU memory. Trust the `(N overflowing)`
            // summary, the CPU_REPACK buffer (Metal's spill), and the
            // done_getting_tensors fallback list over the raw offload number.
            if st.total_layers > 0 {
                // Metal MoE spill always lands in CPU_REPACK; non-zero means
                // real layer-level overflow even if the summary line was
                // missing or its digit pattern didn't match the parser.
                let metal_spill = st.cpu_repack_mib > 0.0;
                // `token_embd.weight` (and rarely `output.weight`) falls back
                // to plain CPU on Metal with a q8_0 embedding because the
                // CPU_REPACK backend won't accept it. That alone doesn't make
                // the load "partial" — it's a normal full-GPU offload.
                let embed_only = st.cpu_forced_count <= 1
                    && (st.cpu_forced_primary.starts_with("token_embd")
                        || st.cpu_forced_primary.starts_with("output"));
                let gpu_line = if st.gpu_layers_loaded == 0 {
                    format!(
                        "{}/{} layers offloaded (CPU only)",
                        st.gpu_layers_loaded, st.total_layers
                    )
                } else if st.overflow_layers > 0 {
                    let full = st.gpu_layers_loaded.saturating_sub(st.overflow_layers);
                    format!(
                        "{}/{} layers offloaded ({} fully GPU, {} with experts on CPU)",
                        st.gpu_layers_loaded, st.total_layers, full, st.overflow_layers
                    )
                } else if st.gpu_layers_loaded == st.total_layers && metal_spill {
                    // Safety net: CPU_REPACK > 0 but `(N overflowing)` didn't
                    // populate. Report the spill even if we can't attribute
                    // it to a specific layer count.
                    format!(
                        "{}/{} layers offloaded (partial, experts on CPU)",
                        st.gpu_layers_loaded, st.total_layers
                    )
                } else if st.gpu_layers_loaded == st.total_layers && embed_only {
                    format!(
                        "{}/{} layers offloaded (fully GPU, embedding on CPU)",
                        st.gpu_layers_loaded, st.total_layers
                    )
                } else if st.gpu_layers_loaded == st.total_layers && st.cpu_forced_count > 0 {
                    format!(
                        "{}/{} layers offloaded (fully GPU, {} tensors on CPU)",
                        st.gpu_layers_loaded, st.total_layers, st.cpu_forced_count
                    )
                } else if st.gpu_layers_loaded == st.total_layers {
                    format!(
                        "{}/{} layers offloaded (fully GPU)",
                        st.gpu_layers_loaded, st.total_layers
                    )
                } else {
                    format!(
                        "{}/{} layers offloaded (partial)",
                        st.gpu_layers_loaded, st.total_layers
                    )
                };
                self.print_wrapped(&mut t, &gpu_line);
            }
        }

        // Model
        let model_display = if st.quantization.is_empty() {
            st.model_name.clone()
        } else {
            format!("{} ({})", st.model_name, st.quantization)
        };
        let has_aliases = !st.config.model_aliases.is_empty();
        let prefix_len = 2 + "Model   ".len() + 3;
        let max_val = t.width.saturating_sub(prefix_len);
        let _ = queue!(
            t.stdout,
            Print("  "),
            SetForegroundColor(MUTED_PURPLE),
            Print("Model    : "),
            SetForegroundColor(Color::White),
            Print(truncate(&model_display, max_val)),
        );
        if has_aliases {
            let _ = queue!(
                t.stdout,
                SetForegroundColor(MUTED_PURPLE),
                Print(" — "),
                SetForegroundColor(COLOR_ALIASES),
                Print(&st.config.model_aliases),
                SetAttribute(Attribute::Reset),
                ResetColor
            );
        }
        t.newline();

        // Source (grey sub of Model)
        self.render_source(&mut t, snap);

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
            self.print_wrapped(&mut t, &params_display);
        }

        // Context (grey sub of Model). `st.ctx_size` is per-slot (see the
        // v6 `UI_SNAPSHOT_VERSION` note); when there's more than one slot
        // we surface the total too so the user can see what llama-server
        // actually allocated.
        let n_par = st.n_parallel.max(1);
        let ctx_display = if n_par > 1 {
            let total = (st.ctx_size as u64) * (n_par as u64);
            format!(
                "{} token context per slot · {} total ({} slots)",
                format_number(st.ctx_size as u64),
                format_number(total),
                n_par
            )
        } else if st.max_ctx_size > 0 && st.max_ctx_size != st.ctx_size {
            format!(
                "{} token context window ({} max)",
                format_number(st.ctx_size as u64),
                format_number(st.max_ctx_size as u64)
            )
        } else {
            format!("{} token context window", format_number(st.ctx_size as u64))
        };
        self.print_wrapped(&mut t, &ctx_display);

        // Sampling (grey sub of Model) — only if any sampler was explicitly
        // set. The renderer doesn't know which setting names belong to this
        // group: it filters on the `group` tag the server-side walker
        // embedded into each `SettingEntry`.
        let sampling_parts: Vec<String> = snap
            .settings
            .iter()
            .filter(|s| s.group.as_deref() == Some("sampling") && !s.is_default)
            .map(format_setting_entry)
            .collect();
        if !sampling_parts.is_empty() {
            self.print_wrapped(&mut t, &format!("sampling: {}", sampling_parts.join(" · ")));
        }

        // llamacpp + status
        let uptime = Duration::from_secs(snap.uptime_seconds);
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
        self.print_wrapped(&mut t, &snap.config.bind_addr);

        // Tuning (grey sub-line under llamacpp) — effective performance
        // knobs. Every `tuning`-group entry lands here regardless of
        // is_default; that matches legacy behavior where lui always showed
        // `np`, `KV`, and `batch` even at their defaults.
        let tuning_parts: Vec<String> = snap
            .settings
            .iter()
            .filter(|s| s.group.as_deref() == Some("tuning"))
            .map(format_setting_entry)
            .collect();
        if !tuning_parts.is_empty() {
            self.print_wrapped(&mut t, &tuning_parts.join(" · "));
        }

        // Performance section
        t.newline();
        let perf_prefix = "  ── Performance ";
        let perf_header = format!(
            "{}{}",
            perf_prefix,
            "─".repeat(width.saturating_sub(perf_prefix.chars().count()))
        );
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

        // Blank line separates the static perf stats above from the
        // potentially-in-flight sections (WebSearch, Requests) below.
        t.newline();

        // WebSearch counts: in local mode these come from the server's own
        // state (via /data). In `--remote` mode, opencode runs on *this*
        // machine and hits *our* local bsearch, not the remote server's — so
        // we read counts from `local_state` when available. The gate is
        // `local_setup_url.is_some()`: the URL is this machine's bsearch
        // URL, so its presence is the definitive signal that this machine
        // is actually serving websearch traffic.
        if self.local_setup_url.is_some() {
            let (total, active_count) = if let Some(ref state) = self.local_state {
                let ls = state.lock().unwrap();
                (ls.websearch_total, ls.active_searches.len())
            } else {
                (st.websearch_total, st.active_searches.len())
            };
            let _ = queue!(
                t.stdout,
                Print("  "),
                SetForegroundColor(MUTED_PURPLE),
                Print("WebSearch: "),
                SetForegroundColor(COLOR_NUMBER),
                Print(format!("{:>4}", total)),
                SetForegroundColor(Color::White),
                Print(" total · "),
                SetForegroundColor(COLOR_NUMBER),
                Print(format!("{:>4}", active_count)),
                SetForegroundColor(Color::White),
                Print(" active"),
                ResetColor
            );
            t.newline();
        }

        // Requests: one-line summary with cache-health counts as further
        // dot-separated stats. Non-zero `reproc` or `invalidated` mean the
        // prompt cache isn't being reused turn-to-turn — the smoking gun for
        // "why did it get slow at long context".
        let _ = queue!(
            t.stdout,
            Print("  "),
            SetForegroundColor(MUTED_PURPLE),
            Print("Requests : "),
            SetForegroundColor(COLOR_NUMBER),
            Print(format!("{:>4}", st.request_count)),
            SetForegroundColor(Color::White),
            Print(" total · "),
            SetForegroundColor(COLOR_NUMBER),
            Print(format!("{:>4}", st.active_requests)),
            SetForegroundColor(Color::White),
            Print(" active · "),
            SetForegroundColor(COLOR_NUMBER),
            Print(format!("{:>4}", st.full_reprocess_count)),
            SetForegroundColor(Color::White),
            Print(" reproc · "),
            SetForegroundColor(COLOR_NUMBER),
            Print(format!("{:>4}", st.invalidated_checkpoint_count)),
            SetForegroundColor(Color::White),
            Print(" invalidated"),
            ResetColor
        );
        t.newline();

        // Mirror the WebSearch count policy: in `--remote` mode the
        // interesting active searches are the ones opencode triggered
        // through our local bsearch, not the remote server's (which has no
        // opencode pointed at it). Fall back to snapshot when there's no
        // local state at all.
        let local_active_searches: Option<Vec<String>> = self.local_state.as_ref().map(|s| {
            let ls = s.lock().unwrap();
            let mut v: Vec<String> = ls.active_searches.values().cloned().collect();
            v.sort();
            v
        });
        let active_searches: &[String] = local_active_searches
            .as_deref()
            .unwrap_or(&st.active_searches);

        if !st.active_slots.is_empty() || !active_searches.is_empty() {
            t.newline();
        }

        for slot in st.active_slots.iter() {
            self.render_active_slot(&mut t, slot);
        }

        for query in active_searches.iter() {
            let desc = format!("● websearch: {}", query);
            let _ = queue!(
                t.stdout,
                Print("             "),
                SetForegroundColor(COLOR_NUMBER),
                Print(truncate(&desc, t.width.saturating_sub(13))),
                ResetColor
            );
            t.newline();
        }

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
                Print("             "),
                SetForegroundColor(Color::DarkGrey),
                Print(truncate(&desc, t.width.saturating_sub(13))),
                ResetColor
            );
            t.newline();
        }

        // Warnings section — only rendered when the snapshot carries at
        // least one warning. Sits between Performance and Server Log so
        // drift notices don't scroll off with the log ring.
        self.render_warnings(&mut t, snap);

        // Server log -- fills remaining space
        t.newline();
        Display::render_log(snap, &mut t);
        t.clear_rest();
        t.flush();
    }

    fn render_warnings(&self, t: &mut TermBuf, snap: &UiSnapshot) {
        if snap.warnings.is_empty() {
            return;
        }
        t.newline();
        let prefix = "  ── Warnings ";
        let header = format!(
            "{}{}",
            prefix,
            "─".repeat(t.width.saturating_sub(prefix.chars().count()))
        );
        let _ = queue!(
            t.stdout,
            SetForegroundColor(MUTED_PURPLE),
            Print(truncate(&header, t.width)),
            ResetColor
        );
        t.newline();
        for w in &snap.warnings {
            let _ = queue!(
                t.stdout,
                Print("  "),
                SetForegroundColor(WARNING_AMBER),
                Print(truncate(w, t.width.saturating_sub(2))),
                ResetColor
            );
            t.newline();
        }
    }

    fn render_source(&self, t: &mut TermBuf, snap: &UiSnapshot) {
        self.print_wrapped(t, &snap.config.model_source);
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

    fn render_active_slot(&self, t: &mut TermBuf, slot: &SlotSnapshot) {
        // Indent 13 to align with the KV value column under "Requests : ".
        let _ = queue!(
            t.stdout,
            Print("             "),
            SetForegroundColor(COLOR_NUMBER)
        );
        if slot.n_tokens == 0 {
            let desc = format!("● slot {} starting...", slot.slot_id);
            let _ = queue!(
                t.stdout,
                Print(truncate(&desc, t.width.saturating_sub(13))),
                ResetColor
            );
        } else if slot.progress > 0.0 && slot.progress < 1.0 {
            let head = format!(
                "● slot {} prefilling {} tokens  ",
                slot.slot_id, slot.n_tokens
            );
            let pct = (slot.progress * 100.0).round() as u32;
            // ETA uses a quadratic model: elapsed ∝ progress². Attention
            // cost per prefilled token grows with prompt position, so the
            // last 30% of progress takes roughly as long as the first 70%.
            let eta_str = slot
                .processing_elapsed_ms
                .and_then(|ms| {
                    let p = slot.progress as f64;
                    if p > 0.20 {
                        let elapsed = (ms as f64) / 1000.0;
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
                .saturating_sub(13 + head.chars().count() + pct_str.chars().count());
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
                let desc = format!(
                    "● slot {} prefilling {} tokens ({}%{})",
                    slot.slot_id, slot.n_tokens, pct, eta_str
                );
                let _ = queue!(
                    t.stdout,
                    Print(truncate(&desc, t.width.saturating_sub(13))),
                    ResetColor
                );
            }
        } else {
            let desc = format!(
                "● slot {} generating ({} tokens prompt)",
                slot.slot_id, slot.n_tokens
            );
            let _ = queue!(
                t.stdout,
                Print(truncate(&desc, t.width.saturating_sub(13))),
                ResetColor
            );
        }
        t.newline();
    }

    fn print_wrapped(&self, t: &mut TermBuf, value: &str) {
        let prefix = "                 ";
        let prefix_len = prefix.chars().count();
        let max_val = t.width.saturating_sub(prefix_len);

        if max_val == 0 {
            return;
        }

        let wrapped = wrap_text(value, max_val);
        let lines: Vec<&str> = wrapped.split('\n').collect();

        for line in lines {
            let _ = queue!(
                t.stdout,
                Print(prefix),
                SetForegroundColor(Color::DarkGrey),
                Print(truncate(line, max_val)),
                ResetColor
            );
            t.newline();
        }
    }

    fn render_log(snap: &UiSnapshot, t: &mut TermBuf) {
        let log_prefix = "  ── Server Log ";
        let log_header = format!(
            "{}{}",
            log_prefix,
            "─".repeat(t.width.saturating_sub(log_prefix.chars().count()))
        );
        let _ = queue!(
            t.stdout,
            SetForegroundColor(MUTED_PURPLE),
            Print(truncate(&log_header, t.width)),
            ResetColor
        );
        t.newline();

        let show = t.remaining().saturating_sub(1).max(1);
        if snap.log_lines.is_empty() {
            let _ = queue!(
                t.stdout,
                Print("  "),
                SetForegroundColor(Color::DarkGrey),
                Print("(no log lines yet)"),
                ResetColor
            );
            t.newline();
            return;
        }

        let lines: Vec<&String> = snap.log_lines.iter().collect();
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

    /// Compute the setup URL color based on how long the UI has been running.
    /// Bright cyan for the first 3 seconds, then dark grey.
    fn setup_url_color(&self) -> Color {
        if (self.start_time.elapsed().as_millis() as u64) < SETUP_URL_START_MS {
            Color::Cyan
        } else {
            Color::DarkGrey
        }
    }

    pub fn print_summary(&self) {
        // print_summary is only meaningful on the server side — it's where
        // llama-server actually ran. A client Display (no local_state) just
        // restores the cursor and bails.
        let Some(ref state) = self.local_state else {
            let _ = execute!(io::stdout(), Show);
            return;
        };
        let st = state.lock().unwrap();
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
                    let _ = execute!(stdout, SetForegroundColor(Color::Yellow));
                    let mut rest = line;
                    while let Some(pos) = rest.find("--avo") {
                        let (before, after) = rest.split_at(pos);
                        let _ = execute!(
                            stdout,
                            Print(before),
                            SetForegroundColor(Color::Cyan),
                            SetAttribute(Attribute::Bold),
                            Print("--avo"),
                            SetAttribute(Attribute::Reset),
                            SetForegroundColor(Color::Yellow),
                        );
                        rest = &after["--avo".len()..];
                    }
                    let _ = execute!(stdout, Print(rest), ResetColor, Print("\n"));
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

/// Render one `SettingEntry` as the `label=value` (or bare `label`) form
/// used on the sampling and tuning sub-lines. The server-side walker has
/// already done the hard work of resolving defaults, collapsing
/// cache_type_k/v into a single KV row, formatting swa_full's bare / =off
/// variants, and so on — the renderer just concatenates.
fn format_setting_entry(s: &crate::server::SettingEntry) -> String {
    if s.value.is_empty() {
        s.display_label.clone()
    } else {
        format!("{}={}", s.display_label, s.value)
    }
}

/// Word-wrap text to the given max width. Splits on whitespace and
/// returns a single string with `\n` between wrapped lines.
fn wrap_text(text: &str, max_width: usize) -> String {
    if text.is_empty() || max_width == 0 {
        return String::new();
    }

    let mut lines = Vec::new();
    let mut current_line = String::new();

    for word in text.split_whitespace() {
        if current_line.is_empty() {
            current_line = word.to_string();
        } else if current_line.chars().count() + 1 + word.chars().count() <= max_width {
            current_line.push(' ');
            current_line.push_str(word);
        } else {
            lines.push(current_line);
            current_line = word.to_string();
        }
    }

    if !current_line.is_empty() {
        lines.push(current_line);
    }

    lines.join("\n")
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
