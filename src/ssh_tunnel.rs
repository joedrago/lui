// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

//! One-shot remote opencode configuration via SSH.
//!
//! When the user passes `--ssh user@host`, lui doesn't start llama-server at
//! all. Instead it:
//!   1. verifies `opencode` exists on the remote,
//!   2. picks a fresh pair of high ports for the remote side of the tunnel
//!      (so we don't collide with a remote service on 8080/8081),
//!   3. writes remote `~/.config/opencode/opencode.json` and (unless websearch
//!      is disabled) the remote `lui-web-search` SKILL.md, both baked with
//!      those remote ports,
//!   4. prints the `ssh -R ... -R ... user@host` command the user should run
//!      from another terminal to actually tunnel back to this machine's
//!      llama-server + websearch endpoint.

use std::io::Write;
use std::process::{Command, Stdio};

use crossterm::style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor};

use crate::config::{build_opencode_json, render_websearch_skill, websearch_port, ServerConfig};

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

/// Parse `user@host`. A bare `host` (with no `@`) is rejected — we want the
/// printed `ssh -R ...` command to match exactly what the user typed, and
/// silently filling in `$USER` would surprise them.
pub fn parse_target(s: &str) -> Result<SshTarget, String> {
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

fn print_success(target: &SshTarget, effective: &ServerConfig, remote_llama: u16, remote_web: u16) {
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

/// The whole flow. Any error here is fatal — the caller prints it and exits.
pub fn setup_remote(target: &SshTarget, effective: &ServerConfig) -> Result<(), String> {
    check_opencode(target)?;

    let remote_llama = pick_remote_port();
    let remote_web = remote_llama + 1;

    let existing = fetch_remote_opencode_json(target);
    let base_url = format!("http://localhost:{}/v1", remote_llama);
    let json = build_opencode_json(effective, &base_url, remote_web, existing.as_deref());
    let contents = serde_json::to_string_pretty(&json)
        .map_err(|e| format!("failed to serialize opencode.json: {}", e))?;

    write_remote_opencode_json(target, &contents)?;
    write_remote_websearch_skill(target, remote_web, effective.websearch_disabled)?;

    print_success(target, effective, remote_llama, remote_web);
    Ok(())
}
