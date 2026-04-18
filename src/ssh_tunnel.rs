// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

//! One-shot peer opencode configuration for Lui/Remote pairs.
//!
//! # Terminology
//!
//! These three words appear throughout this module and the rest of the
//! codebase. They're not OpenSSH terms — they're our own names for the
//! three roles a machine can play in lui's multi-machine layouts:
//!
//! - **Lui**: a machine running `lui` (and thus `llama-server`). It hosts
//!   the model. In every flow, exactly one machine is the Lui.
//! - **Remote**: a user's workstation that wants to drive a Lui's model via
//!   opencode, but doesn't run llama-server itself. A Remote *initiates*
//!   the connection to a Lui (via `lui --remote HOST`).
//! - **ReverseRemote**: same idea as a Remote, but the Lui side initiates
//!   configuration. The Lui user runs `lui --ssh user@host` to push an
//!   opencode config into that host and get back an `ssh -R …` command
//!   that forwards this Lui's llama-server through the tunnel. Used when
//!   the ReverseRemote can't reach the Lui directly (NAT, no public IP).
//!
//! "Remote" on its own always means the forward-initiated kind; the word
//! "ReverseRemote" is only used when the distinction matters.
//!
//! # The two flows
//!
//! They're asymmetric on purpose — the two directions have different
//! constraints:
//!
//! `--ssh user@host` runs on a Lui and prepares a *ReverseRemote* to
//! use this Lui's llama-server over a reverse tunnel. It SSHes into the
//! ReverseRemote, picks a fresh pair of high ports there (to dodge 8080/8081
//! collisions), writes its `~/.config/opencode/opencode.json` and (unless
//! websearch is disabled) its `lui-web-search` SKILL.md baked with those
//! remote ports, and prints the `ssh -R … user@host` command the Lui user
//! runs to establish the tunnel. SSH is load-bearing here: the ReverseRemote
//! usually *can't* reach back to the Lui any other way (NAT, no public
//! address), so a reverse tunnel is the whole point.
//!
//! `--remote host[:port]` runs on a *Remote* and points this machine at
//! an already-running `--public` Lui. No SSH involved: if `/config` (port
//! defaults to 8081) is reachable from here, so is llama-server (the Lui
//! exposes both on the same interface). We fetch `/config` over plain HTTP,
//! write this Remote's own `~/.config/opencode/opencode.json` with `baseURL`
//! pointing *directly* at `http://<host>:<lui_llama>/v1`, and write a
//! `lui-web-search` SKILL.md pointed at a bsearch server we spin up
//! in-process here. Then we block on Ctrl-C so that in-process bsearch
//! stays alive for as long as the user is running opencode. The bsearch
//! server lives on the Remote on purpose: browser-mediated search needs a
//! real human at a real browser, which is wherever the user actually is —
//! not on the Lui.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crossterm::style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor};

use crate::config::{
    build_opencode_json, derive_model_name, opencode_config_path, render_websearch_skill,
    websearch_port, websearch_skill_dir, ServerConfig,
};
use crate::server::ServerState;
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

/// Target of `--remote`: the Lui's hostname plus the HTTP port where
/// its `/config` endpoint is listening. Same port also hosts `/bsearch`
/// et al., but we don't need that here — the Remote runs its own bsearch.
#[derive(Debug, Clone)]
pub struct UseTarget {
    pub host: String,
    pub http_port: u16,
}

/// Default HTTP port lui serves on a Lui that hasn't customized `--web-port`:
/// mirrors the `llama_port + 1` convention with the default 8080 llama port.
pub const DEFAULT_REMOTE_HTTP_PORT: u16 = 8081;

impl UseTarget {
    pub fn http_url(&self, path: &str) -> String {
        format!("http://{}:{}{}", self.host, self.http_port, path)
    }
}

/// Parse `HOST` or `HOST:PORT` for `--remote`. Bare-host form uses
/// `DEFAULT_REMOTE_HTTP_PORT` (8081), which is what a default-config Lui
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
fn ssh_run(target: &SshTarget, args: &[&str], stdin: Option<&[u8]>) -> Result<String, String> {
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

    let mut child = cmd.spawn().map_err(|e| format!("failed to spawn ssh: {}", e))?;
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

/// Verify `opencode` is installed on the remote.
///
/// Non-interactive SSH runs a non-login shell, so `~/.bashrc`/`~/.zshrc`
/// aren't sourced and PATH edits the opencode installer makes there don't
/// show up. We check three things, OR'd together:
///   1. `command -v opencode` (the happy path: it's on the default PATH)
///   2. `bash -lc 'command -v opencode'` (login shell — picks up
///      `~/.bash_profile`/`~/.profile` PATH edits)
///   3. the opencode installer's canonical location, `~/.opencode/bin/opencode`
/// Any one succeeding means opencode is usable once the user actually SSHes
/// in (their interactive shell will source the rc file the installer touched).
fn check_opencode(target: &SshTarget) -> Result<(), String> {
    let probe = "command -v opencode \
        || bash -lc 'command -v opencode' \
        || { [ -x \"$HOME/.opencode/bin/opencode\" ] && echo \"$HOME/.opencode/bin/opencode\"; }";
    match ssh_run(target, &[probe], None) {
        Ok(out) if !out.trim().is_empty() => Ok(()),
        Ok(_) | Err(_) => Err(format!(
            "opencode not found on {}. Install it there first.",
            target.spec()
        )),
    }
}

/// Fetch the existing remote opencode.json, if any. A missing file (ssh
/// exits non-zero from `cat`) is not an error here — we just layer onto an
/// empty object in that case.
fn fetch_remote_opencode_json(target: &SshTarget) -> Option<String> {
    ssh_run(target, &["cat", "~/.config/opencode/opencode.json"], None).ok()
}

/// Write the opencode.json on the remote, creating ~/.config/opencode if
/// needed. We shell out to bash with a heredoc-free pipe — the JSON arrives
/// on stdin and `cat > file` writes it, which avoids any quoting concerns.
fn write_remote_opencode_json(target: &SshTarget, contents: &str) -> Result<(), String> {
    ssh_run(
        target,
        &[
            "mkdir -p ~/.config/opencode && cat > ~/.config/opencode/opencode.json",
        ],
        Some(contents.as_bytes()),
    )?;
    Ok(())
}

/// Write (or remove) the lui-web-search SKILL.md on the remote. Baked with
/// `remote_web_port` so the curl examples in the skill match what the tunnel
/// will actually expose on the remote side.
fn write_remote_websearch_skill(
    target: &SshTarget,
    remote_web_port: u16,
    disabled: bool,
) -> Result<(), String> {
    let skill_dir = "~/.config/opencode/skills/lui-web-search";
    let skill_path = format!("{}/SKILL.md", skill_dir);

    if disabled {
        // Best-effort cleanup: if either step errors (e.g. file didn't
        // exist) we just ignore. rmdir only succeeds if empty, which is
        // what we want.
        let _ = ssh_run(
            target,
            &[&format!("rm -f {} && rmdir {} 2>/dev/null || true", skill_path, skill_dir)],
            None,
        );
        return Ok(());
    }

    let body = render_websearch_skill(remote_web_port);
    ssh_run(
        target,
        &[&format!("mkdir -p {} && cat > {}", skill_dir, skill_path)],
        Some(body.as_bytes()),
    )?;
    Ok(())
}

fn print_share_success(
    target: &SshTarget,
    effective: &ServerConfig,
    remote_llama: u16,
    remote_web: u16,
) {
    let mut stdout = std::io::stdout();
    let lavender = Color::Rgb { r: 180, g: 150, b: 255 };
    let muted = Color::Rgb { r: 120, g: 100, b: 180 };

    let local_llama = effective.port;
    let local_web = websearch_port(effective);

    let mut cmd = format!(
        "ssh -R {}:localhost:{} {}",
        remote_llama,
        local_llama,
        target.spec()
    );
    if !effective.websearch_disabled {
        cmd = format!(
            "ssh -R {}:localhost:{} -R {}:localhost:{} {}",
            remote_llama, local_llama, remote_web, local_web, target.spec()
        );
    }

    let _ = crossterm::execute!(
        stdout,
        Print("\n"),
        SetForegroundColor(lavender),
        SetAttribute(Attribute::Bold),
        Print("  Remote opencode configured on "),
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

/// `--ssh`: configure the given ReverseRemote's opencode to point back
/// at this Lui over a reverse tunnel. Any error here is fatal — the caller
/// prints it and exits.
pub fn setup_share(target: &SshTarget, effective: &ServerConfig) -> Result<(), String> {
    check_opencode(target)?;

    let remote_llama = pick_remote_port();
    let remote_web = remote_llama + 1;

    let existing = fetch_remote_opencode_json(target);
    let base_url = format!("http://localhost:{}/v1", remote_llama);
    let model_name = derive_model_name(effective);
    let json = build_opencode_json(
        &model_name,
        effective.websearch_disabled,
        &base_url,
        remote_web,
        existing.as_deref(),
    );
    let contents = serde_json::to_string_pretty(&json)
        .map_err(|e| format!("failed to serialize opencode.json: {}", e))?;

    write_remote_opencode_json(target, &contents)?;
    write_remote_websearch_skill(target, remote_web, effective.websearch_disabled)?;

    print_share_success(target, effective, remote_llama, remote_web);
    Ok(())
}

// ----------------------------------------------------------------------------
// --remote flow (this machine is a Remote pointing at a running Lui)
// ----------------------------------------------------------------------------

/// Minimal blocking HTTP/1.1 GET. We use `Connection: close` so the server
/// closes after the body, then read-to-EOF — that sidesteps having to parse
/// `Content-Length` or dechunk transfer-encoding. Response is expected to
/// be small (a tiny JSON blob from `/config`).
///
/// Returns (status_code, body). 5-second timeouts on connect/read/write; a
/// misconfigured `--public` on the Lui is a much more common failure than
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
/// cause: a Lui bound to 127.0.0.1 is invisible on the network.
fn fetch_lui_config(target: &UseTarget) -> Result<LuiConfigResponse, String> {
    let (code, body) = http_get(&target.host, target.http_port, "/config").map_err(|e| {
        format!(
            "could not reach {} — {}\n\nIs the Lui running with `--public`? \
             Without it, the HTTP server binds to 127.0.0.1 only and a Remote can't see it.",
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
    let resp: LuiConfigResponse = serde_json::from_str(&body)
        .map_err(|e| format!("{} returned unparseable JSON: {}", target.http_url("/config"), e))?;
    if resp.version != CONFIG_VERSION {
        return Err(format!(
            "Lui reported /config version {}, this lui understands {}. Upgrade the older side.",
            resp.version, CONFIG_VERSION
        ));
    }
    Ok(resp)
}

/// Write `opencode.json` into the local `~/.config/opencode/` directory,
/// layered on any existing file so hand-added keys survive. `llama_base_url`
/// points directly at the Lui's exposed llama-server over HTTP — no tunnel
/// involved — while `local_web` is the port of the bsearch server we're
/// running in-process on this Remote.
fn write_local_opencode_json(
    model_name: &str,
    llama_base_url: &str,
    local_web: u16,
) -> Result<(), String> {
    let path = opencode_config_path();
    let existing = std::fs::read_to_string(&path).ok();
    let json = build_opencode_json(
        model_name,
        /* websearch_disabled */ false,
        llama_base_url,
        local_web,
        existing.as_deref(),
    );
    let contents = serde_json::to_string_pretty(&json)
        .map_err(|e| format!("failed to serialize opencode.json: {}", e))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create {}: {}", parent.display(), e))?;
    }
    std::fs::write(&path, contents).map_err(|e| format!("write {}: {}", path.display(), e))?;
    Ok(())
}

/// Write the `lui-web-search` SKILL.md locally, pointed at the in-process
/// bsearch server we're about to spawn. Same file layout as the local
/// `update_websearch_skill` path uses, just bypassing the ServerConfig
/// plumbing since we don't have a full config on this side.
fn write_local_websearch_skill(local_web: u16) -> Result<(), String> {
    let dir = websearch_skill_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("create {}: {}", dir.display(), e))?;
    let path = dir.join("SKILL.md");
    std::fs::write(&path, render_websearch_skill(local_web))
        .map_err(|e| format!("write {}: {}", path.display(), e))
}

fn print_use_banner(
    target: &UseTarget,
    lui_cfg: &LuiConfigResponse,
    llama_base_url: &str,
    local_web: u16,
) {
    let mut stdout = std::io::stdout();
    let lavender = Color::Rgb { r: 180, g: 150, b: 255 };
    let muted = Color::Rgb { r: 120, g: 100, b: 180 };

    let _ = crossterm::execute!(
        stdout,
        Print("\n"),
        SetForegroundColor(lavender),
        SetAttribute(Attribute::Bold),
        Print("  Using Lui at "),
        SetForegroundColor(Color::Cyan),
        Print(format!("{}:{}", target.host, target.http_port)),
        SetForegroundColor(lavender),
        Print("\n"),
        SetAttribute(Attribute::Reset),
        ResetColor,
        Print("\n"),
        SetForegroundColor(muted),
        Print(format!(
            "    model:            {}\n    llama (direct):   {}\n    bsearch (local):  http://127.0.0.1:{}/bsearch\n",
            lui_cfg.model_name, llama_base_url, local_web
        )),
        ResetColor,
        Print("\n"),
        SetForegroundColor(muted),
        Print("  opencode config written. Run `opencode` in another terminal.\n"),
        Print("  Ctrl-C here to shut down the local bsearch server.\n\n"),
        ResetColor,
    );
}

/// `--remote`: entry point on a Remote. Fetches the Lui's `/config`
/// over plain HTTP, writes local opencode.json + skill, spawns an in-process
/// bsearch HTTP server, and blocks on Ctrl-C so bsearch stays alive for the
/// life of the opencode session. No SSH anywhere — `--public` on the Lui is
/// the only prerequisite, and if `/config` is reachable so is llama-server
/// on the same interface.
pub async fn setup_use(target: &UseTarget) -> Result<(), String> {
    let lui_cfg = fetch_lui_config(target)?;

    // Pick a fresh high port for the local bsearch server so it doesn't
    // collide with anything already bound on 8081 here. `pick_remote_port`
    // seeds from wall-clock ns, which is good enough for two separate
    // invocations to pick different values most of the time.
    let local_web = pick_remote_port();

    // opencode points straight at the Lui's exposed llama-server over the
    // network. We use the host the user typed (not a reverse-resolved name)
    // so the URL matches what they'd see in `lui --public`'s banner.
    let llama_base_url = format!("http://{}:{}/v1", target.host, lui_cfg.llama_port);

    write_local_opencode_json(&lui_cfg.model_name, &llama_base_url, local_web)?;
    write_local_websearch_skill(local_web)?;

    // In-process bsearch server. We synthesize a minimal ServerState just
    // to satisfy the API; only `websearch_total` / `active_searches` get
    // touched by bsearch handlers, so defaults are fine. The `/config`
    // payload we hand it reflects this Remote's view — mostly there so a
    // future tool introspecting *this* instance sees something coherent.
    let state = Arc::new(Mutex::new(ServerState::default()));
    let config_info = LuiConfigResponse {
        version: CONFIG_VERSION,
        llama_port: lui_cfg.llama_port,
        web_port: local_web,
        websearch_disabled: false,
        model_name: lui_cfg.model_name.clone(),
    };
    websearch::spawn("127.0.0.1", local_web, state, config_info);

    print_use_banner(target, &lui_cfg, &llama_base_url, local_web);

    // Block on Ctrl-C. bsearch lives on a tokio task spawned by
    // `websearch::spawn`; returning from here would drop the runtime and
    // take it with us. ctrl_c's Err path means the signal handler itself
    // failed to install — rare, but treat it as fatal so we don't silently
    // hang unreacheable to a keypress.
    tokio::signal::ctrl_c()
        .await
        .map_err(|e| format!("failed to install Ctrl-C handler: {}", e))?;
    Ok(())
}
