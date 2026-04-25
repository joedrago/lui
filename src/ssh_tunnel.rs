// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

//! One-shot peer opencode configuration for server/client pairs.
//!
//! # Terminology
//!
//! These words appear throughout this module and the rest of the
//! codebase. They describe the two roles machines play in lui's
//! multi-machine layouts:
//!
//! - **server**: a machine running `lui` (and thus `llama-server`). It hosts
//!   the model. In every flow, exactly one machine is the server.
//! - **client**: a user's workstation that wants to drive a server's model via
//!   opencode, but doesn't run llama-server itself. A client *initiates*
//!   the connection to a server (via `lui --remote HOST`).
//!
//! # The two flows
//!
//! They're asymmetric on purpose — the two directions have different
//! constraints:
//!
//! `--ssh user@host` runs on the server and prepares a *client* to
//! use this server's llama-server over a reverse tunnel. It SSHes into the
//! client, picks a fresh pair of high ports there (to dodge 8080/8081
//! collisions), writes its `~/.config/opencode/opencode.json` and (unless
//! websearch is disabled) its `lui-web-search` SKILL.md baked with those
//! client ports, and prints the `ssh -R … user@host` command the server user
//! runs to establish the tunnel. SSH is load-bearing here: the client
//! usually *can't* reach back to the server any other way (NAT, no public
//! address), so a reverse tunnel is the whole point.
//!
//! `--remote host[:port]` runs on a *client* and points this machine at
//! an already-running `--public` server. No SSH involved: if `/config` (port
//! defaults to 8081) is reachable from here, so is llama-server (the server
//! exposes both on the same interface). We fetch `/config` over plain HTTP,
//! write this client's own `~/.config/opencode/opencode.json` with `baseURL`
//! pointing *directly* at `http://<host>:<server_llama>/v1`, and write a
//! `lui-web-search` SKILL.md pointed at a bsearch server we spin up
//! in-process here. Then we block on Ctrl-C so that in-process bsearch
//! stays alive for as long as the user is running opencode. The bsearch
//! server lives on the client on purpose: browser-mediated search needs a
//! real human at a real browser, which is wherever the user actually is —
//! not on the server.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crossterm::style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor};

use crate::config::{derive_model_name, websearch_port};
use crate::display::Display;
use crate::harness;
use crate::server::{ConfigSummary, ServerState};
use crate::websearch::{self, LuiConfigResponse, CONFIG_VERSION};

#[derive(Debug, Clone)]
pub struct SshTarget {
    pub user: String,
    pub host: String,
}

impl SshTarget {
    pub fn spec(&self) -> String {
        format!("{}@{}", self.user, self.host)
    }
}

/// Parse `user@host` for `--ssh`. A bare `host` (with no `@`) is
/// rejected — we want the printed `ssh -R …` command to match exactly what
/// the user typed, and silently filling in `$USER` would surprise them.
pub fn parse_share_target(s: &str) -> Result<SshTarget, String> {
    let (user, host) = s
        .split_once('@')
        .ok_or_else(|| format!("--ssh expects USER@HOST, got {:?}", s))?;
    if user.is_empty() || host.is_empty() {
        return Err(format!("--ssh expects USER@HOST, got {:?}", s));
    }
    Ok(SshTarget {
        user: user.to_string(),
        host: host.to_string(),
    })
}

/// Target of `--remote`: the server's hostname plus the HTTP port where
/// its `/config` endpoint is listening. Same port also hosts `/bsearch`
/// et al., but we don't need that here — the client runs its own bsearch.
#[derive(Debug, Clone)]
pub struct UseTarget {
    pub host: String,
    pub http_port: u16,
}

/// Default HTTP port lui serves on a server that hasn't customized `--web-port`:
/// mirrors the `llama_port + 1` convention with the default 8080 llama port.
pub const DEFAULT_REMOTE_HTTP_PORT: u16 = 8081;

impl UseTarget {
    pub fn http_url(&self, path: &str) -> String {
        format!("http://{}:{}{}", self.host, self.http_port, path)
    }
}

/// Parse `HOST` or `HOST:PORT` for `--remote`. Bare-host form uses
/// `DEFAULT_REMOTE_HTTP_PORT` (8081), which is what a default-config server
/// serves `/config` on. IPv6 literals aren't supported here (bracketed form
/// would be ambiguous with our split logic); users who need IPv6 can set up
/// a hostname alias in `/etc/hosts`.
pub fn parse_use_target(s: &str) -> Result<UseTarget, String> {
    if s.is_empty() {
        return Err("--remote expects HOST or HOST:PORT".into());
    }
    let (host, http_port) = match s.rsplit_once(':') {
        Some((h, p)) => {
            if h.is_empty() || p.is_empty() {
                return Err(format!("--remote expects HOST or HOST:PORT, got {:?}", s));
            }
            let port: u16 = p
                .parse()
                .map_err(|_| format!("--remote port {:?} is not a valid u16", p))?;
            (h.to_string(), port)
        }
        None => (s.to_string(), DEFAULT_REMOTE_HTTP_PORT),
    };
    Ok(UseTarget { host, http_port })
}

/// Pick the base port for the remote side of the tunnel. We use a pseudo-
/// random value in [18000, 29000) seeded from the wall clock so re-running
/// `--ssh` on a machine that already has a tunnel active is likely to pick
/// a different pair. The websearch port is always base+1, to match the
/// local-side convention (`web_port = llama_port + 1` by default).
fn pick_remote_port() -> u16 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    // base+1 must also stay below 30000, so cap base at 28999.
    18000 + (ns as u64 % 11000) as u16
}

/// Run `ssh <spec> <args...>` with the given stdin (if any) and return
/// stdout on success. On non-zero exit we return a message composed from
/// stderr (falling back to stdout) so the user sees the actual ssh/remote
/// error verbatim.
pub(crate) fn ssh_run(
    target: &SshTarget,
    args: &[&str],
    stdin: Option<&[u8]>,
) -> Result<String, String> {
    let mut cmd = Command::new("ssh");
    cmd.arg(target.spec());
    for a in args {
        cmd.arg(a);
    }
    cmd.stdin(if stdin.is_some() {
        Stdio::piped()
    } else {
        Stdio::null()
    });
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed to spawn ssh: {}", e))?;
    if let Some(bytes) = stdin {
        let mut si = child
            .stdin
            .take()
            .ok_or_else(|| "failed to open ssh stdin".to_string())?;
        si.write_all(bytes)
            .map_err(|e| format!("failed to write ssh stdin: {}", e))?;
        drop(si);
    }
    let output = child
        .wait_with_output()
        .map_err(|e| format!("failed to wait on ssh: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let msg = if !stderr.is_empty() {
            stderr
        } else if !stdout.is_empty() {
            stdout
        } else {
            format!("ssh exited with status {}", output.status)
        };
        return Err(msg);
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn print_share_success(
    target: &SshTarget,
    effective: &crate::settings::store::Effective,
    remote_llama: u16,
    remote_web: u16,
) {
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

    let local_llama = effective.get_i64("port").unwrap_or(8080) as u16;
    let local_web = websearch_port(effective);
    let websearch = effective.get_bool("websearch").unwrap_or(true);

    let mut cmd = format!(
        "ssh -R {}:localhost:{} {}",
        remote_llama,
        local_llama,
        target.spec()
    );
    if websearch {
        cmd = format!(
            "ssh -R {}:localhost:{} -R {}:localhost:{} {}",
            remote_llama,
            local_llama,
            remote_web,
            local_web,
            target.spec()
        );
    }

    let _ = crossterm::execute!(
        stdout,
        Print("\n"),
        SetForegroundColor(lavender),
        SetAttribute(Attribute::Bold),
        Print("  opencode configured on "),
        SetForegroundColor(Color::Cyan),
        Print(target.spec()),
        SetForegroundColor(lavender),
        Print("\n"),
        SetAttribute(Attribute::Reset),
        ResetColor,
        Print("\n"),
        SetForegroundColor(muted),
        Print("  To connect from this machine, run in another terminal:\n\n"),
        ResetColor,
        Print("    "),
        SetForegroundColor(lavender),
        SetAttribute(Attribute::Bold),
        Print(&cmd),
        SetAttribute(Attribute::Reset),
        ResetColor,
        Print("\n\n"),
    );
}

/// `--ssh`: configure the given client's enabled harnesses to point back
/// at this server over a reverse tunnel. Any error is fatal — the caller
/// prints and exits.
pub fn setup_share(
    target: &SshTarget,
    effective: &crate::settings::store::Effective,
) -> Result<(), String> {
    let remote_llama = pick_remote_port();
    let remote_web = remote_llama + 1;

    let inputs = harness::HarnessInputs {
        model_name: derive_model_name(effective),
        base_url: format!("http://localhost:{}/v1", remote_llama),
        ctx_size: effective.get_i64("ctx_size").unwrap_or(0) as u32,
        web_port: remote_web,
        websearch: effective.get_bool("websearch").unwrap_or(true),
    };

    for h in harness::HARNESSES {
        if !effective.get_bool(h.setting_name).unwrap_or(h.default_on) {
            continue;
        }
        harness::apply_remote(h, target, remote_web, &inputs, effective)?;
    }

    print_share_success(target, effective, remote_llama, remote_web);
    Ok(())
}

// ----------------------------------------------------------------------------
// --remote flow (this machine is a client pointing at a running server)
// ----------------------------------------------------------------------------

/// Minimal blocking HTTP/1.1 GET. We use `Connection: close` so the server
/// closes after the body, then read-to-EOF — that sidesteps having to parse
/// `Content-Length` or dechunk transfer-encoding. Response is expected to
/// be small (a tiny JSON blob from `/config`).
///
/// Returns (status_code, body). 5-second timeouts on connect/read/write; a
/// misconfigured `--public` on the server is a much more common failure than
/// a slow network, so we'd rather fail fast with a clear hint.
fn http_get(host: &str, port: u16, path: &str) -> Result<(u16, String), String> {
    let sockaddrs: Vec<SocketAddr> = (host, port)
        .to_socket_addrs()
        .map_err(|e| format!("resolve {}: {}", host, e))?
        .collect();
    let first = sockaddrs
        .first()
        .ok_or_else(|| format!("{} resolved to no addresses", host))?;

    let timeout = Duration::from_secs(5);
    let mut stream = TcpStream::connect_timeout(first, timeout)
        .map_err(|e| format!("connect {}:{}: {}", host, port, e))?;
    stream.set_read_timeout(Some(timeout)).ok();
    stream.set_write_timeout(Some(timeout)).ok();

    let req = format!(
        "GET {} HTTP/1.1\r\nHost: {}:{}\r\nConnection: close\r\nAccept: application/json\r\n\r\n",
        path, host, port
    );
    stream
        .write_all(req.as_bytes())
        .map_err(|e| format!("write: {}", e))?;

    let mut buf = Vec::new();
    stream
        .read_to_end(&mut buf)
        .map_err(|e| format!("read: {}", e))?;

    let text = String::from_utf8_lossy(&buf).into_owned();
    let split = text
        .find("\r\n\r\n")
        .ok_or_else(|| "malformed HTTP response (no header/body split)".to_string())?;
    let (headers, body_with_sep) = text.split_at(split);
    let body = &body_with_sep[4..];

    let status_line = headers
        .lines()
        .next()
        .ok_or_else(|| "empty HTTP response".to_string())?;
    let mut parts = status_line.splitn(3, ' ');
    let _proto = parts.next();
    let code: u16 = parts
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("bad HTTP status line: {:?}", status_line))?;

    Ok((code, body.to_string()))
}

/// Hit `/config` and decode it. Error messages drop a hint about `--public`
/// on connection-refused / timeout, since that's the overwhelmingly common
/// cause: a server bound to 127.0.0.1 is invisible on the network.
fn fetch_lui_config(target: &UseTarget) -> Result<LuiConfigResponse, String> {
    let (code, body) = http_get(&target.host, target.http_port, "/config").map_err(|e| {
        format!(
            "could not reach {} — {}\n\nIs the server running with `--public`? \
             Without it, the HTTP server binds to 127.0.0.1 only and a client can't see it.",
            target.http_url("/config"),
            e
        )
    })?;
    if !(200..300).contains(&code) {
        return Err(format!(
            "{} returned HTTP {} (expected 200)",
            target.http_url("/config"),
            code
        ));
    }
    let resp: LuiConfigResponse = serde_json::from_str(&body).map_err(|e| {
        format!(
            "{} returned unparseable JSON: {}",
            target.http_url("/config"),
            e
        )
    })?;
    if resp.version != CONFIG_VERSION {
        return Err(format!(
            "server reported /config version {}, this lui understands {}. Upgrade the older side.",
            resp.version, CONFIG_VERSION
        ));
    }
    Ok(resp)
}

fn print_use_banner(
    target: &UseTarget,
    lui_cfg: &LuiConfigResponse,
    llama_base_url: &str,
    local_web: u16,
) {
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

    let _ = crossterm::execute!(
        stdout,
        Print("\n"),
        SetForegroundColor(lavender),
        SetAttribute(Attribute::Bold),
        Print("  Using server at "),
        SetForegroundColor(Color::Cyan),
        Print(format!("{}:{}", target.host, target.http_port)),
        SetForegroundColor(lavender),
        Print("\n"),
        SetAttribute(Attribute::Reset),
        ResetColor,
        Print("\n"),
        SetForegroundColor(muted),
        Print(format!(
            "    model:            {}\n    llama (direct):   {}\n    bsearch (local):  http://127.0.0.1:{}/bsearch\n    bookmarklet:      http://127.0.0.1:{}/setup\n",
            lui_cfg.model_name, llama_base_url, local_web, local_web
        )),
        ResetColor,
        Print("\n"),
        SetForegroundColor(muted),
        Print("  opencode config written. Run `opencode` in another terminal.\n"),
        Print("  Open the bookmarklet URL once to install/update the search bookmark;\n"),
        Print("  /bsearch stays broken until that bookmarklet fires in a browser.\n"),
        Print("  Ctrl-C here to shut down the local bsearch server.\n\n"),
        ResetColor,
    );
}

/// `--remote`: entry point on a client. Fetches the server's `/config`
/// over plain HTTP, writes each enabled harness's config locally pointing
/// at the server, spawns an in-process bsearch HTTP server, and blocks on
/// Ctrl-C so bsearch stays alive for the life of the opencode session. No
/// SSH anywhere — `--public` on the server is the only prerequisite, and
/// if `/config` is reachable so is llama-server on the same interface.
pub async fn setup_use(
    target: &UseTarget,
    effective: &crate::settings::store::Effective<'_>,
) -> Result<(), String> {
    let lui_cfg = fetch_lui_config(target)?;

    // Use the conventional 8081 port here, not a randomized high port. A
    // client doesn't run llama-server (or anything else on 8081), so
    // collision risk is minimal, and a stable port means the user only has
    // to drag the /setup bookmarklet into their bookmarks bar once — not
    // once per `lui --remote` invocation.
    let local_web = DEFAULT_REMOTE_HTTP_PORT;

    // opencode points straight at the server's exposed llama-server over the
    // network. We use the host the user typed (not a reverse-resolved name)
    // so the URL matches what they'd see in `lui --public`'s banner.
    let llama_base_url = format!("http://{}:{}/v1", target.host, lui_cfg.llama_port);

    // Apply every enabled harness locally, pointed at the remote server.
    // Websearch is always on here — the client always spins up the local
    // bsearch server that the skill talks to.
    let inputs = harness::HarnessInputs {
        model_name: lui_cfg.model_name.clone(),
        base_url: llama_base_url.clone(),
        ctx_size: lui_cfg.ctx_size,
        web_port: local_web,
        websearch: true,
    };
    for h in harness::HARNESSES {
        if !effective.get_bool(h.setting_name).unwrap_or(h.default_on) {
            continue;
        }
        harness::apply_local(h, effective, &inputs);
    }

    // In-process bsearch server. We synthesize a minimal ServerState just
    // to satisfy the API; only `websearch_total` / `active_searches` get
    // touched by bsearch handlers, so defaults are fine. The `/config`
    // payload we hand it reflects this client's view — mostly there so a
    // future tool introspecting *this* instance sees something coherent.
    let state = Arc::new(Mutex::new(ServerState::default()));
    let config_info = LuiConfigResponse {
        version: CONFIG_VERSION,
        llama_port: lui_cfg.llama_port,
        web_port: local_web,
        websearch: true,
        model_name: lui_cfg.model_name.clone(),
        ctx_size: lui_cfg.ctx_size,
    };
    // Minimal ConfigSummary for the client's in-process HTTP server. Nothing
    // ever polls this client's /data (the renderer points at the *server's*
    // /data, not ours), so fields here are placeholders — but we keep the
    // shape coherent in case something future introspects it.
    let dummy_summary = ConfigSummary {
        bind_addr: format!("127.0.0.1:{}", local_web),
        web_port: local_web,
        websearch: true,
        model_source: "none".to_string(),
        model_aliases: String::new(),
    };
    websearch::spawn(
        "127.0.0.1",
        local_web,
        state.clone(),
        config_info,
        std::time::Instant::now(),
        dummy_summary,
        Vec::new(),
    );

    // Print the setup banner before the Display starts. Display doesn't
    // Purge scrollback, so this stays scrollable during the session —
    // useful for recovering the bookmarklet URL or confirming what
    // opencode was pointed at without having to exit.
    print_use_banner(target, &lui_cfg, &llama_base_url, local_web);

    // Render the server's UI by polling its `/data`. We pass `Some(state)`
    // for the client's in-process bsearch ServerState — `websearch_total`
    // and `active_searches` reflect the user's opencode-driven searches
    // through *our* bsearch, which is what they care about. The server's
    // log lines now come via `/data` so the Server Log panel works in
    // remote mode too. The bookmarklet URL is
    // this client's own bsearch (8081 by convention), not the server's —
    // bookmarklets live in the browser the user is sitting at.
    let local_setup_url = Some(format!("http://127.0.0.1:{}/setup", local_web));
    let display = Display::new(
        target.host.clone(),
        target.http_port,
        Some(state.clone()),
        local_setup_url,
    );

    // Display.run owns the screen and exits on Ctrl-C / 'q' / the server
    // reporting `exited: true`. shutdown_tx is unused here (no child
    // process to tear down) so we just hand it a throwaway channel.
    let (shutdown_tx, _rx) = tokio::sync::watch::channel(false);
    display.run(shutdown_tx).await;
    Ok(())
}
