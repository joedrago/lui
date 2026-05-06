// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

//! `lui --sandbox HARNESSNAME [args...]` — wrap a harness launch in
//! [nono](https://nono.sh).
//!
//! Unlike `--ssh` and `--remote`, this mode short-circuits the entire
//! lui pipeline: no llama-server, no TUI, no websearch HTTP. lui is a
//! thin launcher that resolves persisted `[sandbox]` settings into a
//! `nono run …` invocation, execs the chosen harness underneath it with
//! full stdio inherited, and exits with the child's status.
//!
//! Argv slicing is intentionally permissive: every token after the
//! harness name is forwarded verbatim, so `lui --sandbox opencode -c`
//! runs `opencode -c` inside nono, and the harness can use any of its
//! own flags (including `--`) without lui interpreting them.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::harness::HARNESSES;
use crate::settings::store::Effective;

/// Literal stand-in printed in the `--cmd` preview wherever the live
/// launcher would substitute the harness name. Single source so the
/// renderer's color match and the build function can never disagree.
pub const HARNESS_PLACEHOLDER: &str = "HARNESSNAME";

/// Profile name used when the harness has no nono-shipped profile (and
/// the user hasn't picked one explicitly). nono's `default` is the
/// conservative base every shipped profile extends, so falling back to
/// it gives /tmp, /usr/bin, system reads, deny-rules for credentials,
/// etc. — usable for arbitrary CLIs.
const FALLBACK_PROFILE: &str = "default";

/// Value the user passes to `--sandbox-profile` to opt out of any
/// `-p NAME` flag entirely. Distinguishes "no profile" from "auto".
const PROFILE_OPT_OUT: &str = "none";

/// Context that decides how `build_nono_args` resolves the profile slot.
#[derive(Debug, Clone)]
pub enum ProfileContext<'a> {
    /// Live launch: use the harness name to probe nono and pick a
    /// concrete profile (auto-detect → harness name if shipped, else
    /// fallback). Triggers a subprocess call.
    Launch(&'a str),
    /// `--cmd` preview: never spawns nono. Renders the auto slot as the
    /// literal `HARNESSNAME` placeholder so the printed line still
    /// substitutes magenta-rendered placeholders cleanly.
    Display,
}

/// Captured shape of `--sandbox HARNESSNAME [args...]`.
#[derive(Debug, Clone)]
pub struct SandboxRequest {
    /// Harness name. Validated against `HARNESSES` at parse time so the
    /// launcher only ever execs known external tools (also catches typos
    /// before nono gets involved).
    pub harness: String,
    /// Every token after the harness name, in order. Forwarded literally
    /// to the harness command line — no `--` splitting, no registry
    /// parsing, no shell interpretation.
    pub harness_args: Vec<OsString>,
}

/// Pre-scan an argv slice for `--sandbox`. If found, splits the argv
/// into the prefix that the registry-driven lexopt loop should still see
/// and a `SandboxRequest` carrying the harness name plus everything
/// after it. Otherwise returns the original argv unchanged.
///
/// The harness name must immediately follow `--sandbox`; absence is a
/// fatal user error (matches the contract for `--ssh USER@HOST`).
/// Unknown harnesses are also fatal here so the failure message lists
/// the legal names instead of letting nono fail later with a confusing
/// "no such binary" error.
pub fn extract_request(argv: Vec<OsString>) -> (Vec<OsString>, Option<SandboxRequest>) {
    let needle = OsString::from("--sandbox");
    let pos = match argv.iter().position(|a| *a == needle) {
        Some(p) => p,
        None => return (argv, None),
    };

    let mut iter = argv.into_iter();
    let pre: Vec<OsString> = iter.by_ref().take(pos).collect();
    let _flag = iter.next();
    let harness_os = iter.next().unwrap_or_else(|| {
        die("--sandbox requires a HARNESSNAME (e.g. --sandbox opencode)")
    });
    let harness = harness_os
        .into_string()
        .unwrap_or_else(|_| die("--sandbox HARNESSNAME must be UTF-8"));

    if !HARNESSES.iter().any(|h| h.name == harness) {
        let known: Vec<&str> = HARNESSES.iter().map(|h| h.name).collect();
        die(&format!(
            "--sandbox: unknown harness {:?}. Known harnesses: {}",
            harness,
            known.join(", ")
        ));
    }

    let harness_args: Vec<OsString> = iter.collect();
    (
        pre,
        Some(SandboxRequest {
            harness,
            harness_args,
        }),
    )
}

/// Resolve the persisted `[sandbox]` settings into the nono argv prefix
/// that ends just before the harness binary. Layout:
///
/// ```text
/// [<bin>, "run", -p PROFILE, <flags...>, "--"]
/// ```
///
/// The trailing `--` separates nono's own flags from the program nono
/// will exec; everything past it is the harness argv.
///
/// Profile resolution:
/// - If the user set `[sandbox].profile` to a non-empty value, that
///   wins verbatim. The literal `none` opts out (no `-p` emitted).
/// - In `Launch` context, the harness name is probed against nono's
///   shipped profiles; matched → that profile, unmatched → fallback.
/// - In `Display` context, the auto slot becomes the literal
///   `HARNESSNAME` placeholder so the renderer can highlight it.
///
/// Used by both the `--cmd` renderer (Display) and the live launcher.
pub fn build_nono_args(eff: &Effective, ctx: ProfileContext<'_>) -> Vec<String> {
    let bin = eff.get_string("sandbox_bin").unwrap_or("nono").to_string();
    let mut out: Vec<String> = vec![bin.clone(), "run".to_string()];

    let chosen_profile = resolve_profile(eff, &ctx, Path::new(&bin));
    if let Some(p) = chosen_profile {
        out.push("-p".to_string());
        out.push(p);
    }

    if eff.get_bool("sandbox_silent").unwrap_or(false) {
        out.push("-s".to_string());
    }
    if eff.get_bool("sandbox_allow_cwd").unwrap_or(true) {
        // `--allow .` grants R+W to the cwd unconditionally (some
        // profiles default the cwd to read-only); `--allow-cwd` skips
        // nono's first-run prompt for the same path. Pair them so the
        // agent can both read and write the project tree without
        // interactive friction.
        out.push("--allow".to_string());
        out.push(".".to_string());
        out.push("--allow-cwd".to_string());
    }
    if eff.get_bool("sandbox_allow_gpu").unwrap_or(true) {
        out.push("--allow-gpu".to_string());
    }
    if eff.get_bool("sandbox_block_net").unwrap_or(false) {
        out.push("--block-net".to_string());
    }
    if eff.get_bool("sandbox_rollback").unwrap_or(false) {
        out.push("--rollback".to_string());
    }
    if eff.get_bool("sandbox_dev_tools").unwrap_or(true) {
        for dir in dev_tool_dirs() {
            out.push("--allow".to_string());
            out.push(dir);
        }
    }
    for dir in eff.merged_string_array("sandbox_allow") {
        out.push("--allow".to_string());
        out.push(dir);
    }
    for dir in eff.merged_string_array("sandbox_read") {
        out.push("--read".to_string());
        out.push(dir);
    }
    for dir in eff.merged_string_array("sandbox_write") {
        out.push("--write".to_string());
        out.push(dir);
    }
    for dom in eff.merged_string_array("sandbox_allow_domain") {
        out.push("--allow-domain".to_string());
        out.push(dom);
    }
    for arg in eff.merged_string_array("sandbox_extra") {
        out.push(arg);
    }

    out.push("--".to_string());
    out
}

fn resolve_profile(eff: &Effective, ctx: &ProfileContext<'_>, bin: &Path) -> Option<String> {
    let explicit = eff
        .get_string("sandbox_profile")
        .map(str::trim)
        .filter(|s| !s.is_empty());
    match (explicit, ctx) {
        (Some(s), _) if s.eq_ignore_ascii_case(PROFILE_OPT_OUT) => None,
        (Some(s), _) => Some(s.to_string()),
        (None, ProfileContext::Display) => Some(HARNESS_PLACEHOLDER.to_string()),
        (None, ProfileContext::Launch(harness)) => {
            if nono_profile_exists(bin, harness) {
                Some((*harness).to_string())
            } else {
                Some(FALLBACK_PROFILE.to_string())
            }
        }
    }
}

/// Canonical toolchain directories an AI agent typically needs to
/// invoke `cargo`, `rustc`, `go`, `python`, `pip`, `npm`, `bun`, etc.
/// Order is stable so `--cmd` previews match across runs. Only existing
/// paths are returned; nono would warn-and-skip non-existent ones, but
/// listing them in the preview would be noisy and misleading.
///
/// Standard-location env vars (`CARGO_HOME`, `RUSTUP_HOME`, `GOPATH`,
/// `PYENV_ROOT`) override the home-relative defaults when set, so users
/// with relocated toolchains get the right paths automatically.
///
/// Granted as `--allow` (R+W) rather than `--read`: cargo writes to
/// `~/.cargo/registry/cache`, go writes to `~/go/pkg/mod`, npm to
/// `~/.npm/_cacache`, etc. Read-only would silently break first-run
/// fetches. The user has already trusted the agent enough to grant
/// `--allow .` on the project tree; toolchain caches are in the same
/// trust class.
fn dev_tool_dirs() -> Vec<String> {
    let home = dirs::home_dir().unwrap_or_default();
    let env_or_home = |env: &str, rel: &str| -> PathBuf {
        match std::env::var_os(env) {
            Some(v) if !v.is_empty() => PathBuf::from(v),
            _ => home.join(rel),
        }
    };

    let candidates: Vec<PathBuf> = vec![
        // Rust
        env_or_home("CARGO_HOME", ".cargo"),
        env_or_home("RUSTUP_HOME", ".rustup"),
        // Go
        env_or_home("GOPATH", "go"),
        PathBuf::from("/usr/local/go"),
        // Python
        env_or_home("PYENV_ROOT", ".pyenv"),
        home.join(".local/lib"),
        home.join(".local/share/uv"),
        home.join(".conda"),
        // Node / Bun / Deno
        home.join(".nvm"),
        home.join(".fnm"),
        home.join(".npm"),
        home.join(".bun"),
        home.join(".deno"),
        home.join("Library/pnpm"),
        home.join(".local/share/pnpm"),
        PathBuf::from("/usr/local/lib/node_modules"),
        // Nix
        home.join(".nix-profile"),
        home.join(".nix-defexpr"),
        home.join(".local/state/nix"),
        PathBuf::from("/nix/store"),
        PathBuf::from("/nix/var/nix/profiles"),
    ];

    let mut out: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
    for p in candidates {
        if !p.exists() {
            continue;
        }
        let canon = std::fs::canonicalize(&p).unwrap_or(p);
        if seen.insert(canon.clone()) {
            out.push(canon.to_string_lossy().to_string());
        }
    }
    out
}

/// Probe whether nono ships (or the user has installed) a profile with
/// the given name. Spawns `nono profile show NAME --silent` with all
/// stdio nulled and checks the exit code — exit 0 means the profile
/// resolves, anything else means it doesn't. The probe is only invoked
/// from launch context, never from `--cmd`, so the subprocess cost only
/// applies when the user is actually running the sandbox.
fn nono_profile_exists(bin: &Path, name: &str) -> bool {
    Command::new(bin)
        .args(["profile", "show", name, "--silent"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Resolve the configured nono binary on PATH (or as an absolute path).
/// Fatal error with an actionable message if it can't be found, so the
/// user knows exactly what to install or set.
fn which_nono(eff: &Effective) -> PathBuf {
    let bin = eff.get_string("sandbox_bin").unwrap_or("nono");
    match which::which(bin) {
        Ok(p) => p,
        Err(_) => die(&format!(
            "--sandbox requires {:?} on PATH (install from https://nono.sh, or override with --sandbox-bin)",
            bin
        )),
    }
}

/// Spawn nono with the resolved argv plus the harness binary and its
/// arguments. Inherits stdio so the harness gets a real terminal, blocks
/// until the child exits, and propagates the exit code. Never returns.
pub fn launch(req: &SandboxRequest, eff: &Effective) -> ! {
    let bin_path = which_nono(eff);

    let mut argv = build_nono_args(eff, ProfileContext::Launch(&req.harness));
    argv.push(req.harness.clone());

    let mut cmd = Command::new(&bin_path);
    // The first element of `argv` here is the bin name we baked in
    // earlier; Command::new already covers that, so skip element 0 and
    // pass the rest as args.
    cmd.args(&argv[1..]);
    for a in &req.harness_args {
        cmd.arg(a);
    }

    match cmd.status() {
        Ok(status) => std::process::exit(status.code().unwrap_or(1)),
        Err(e) => {
            eprintln!(
                "lui: --sandbox: failed to spawn {}: {}",
                bin_path.display(),
                e
            );
            std::process::exit(2);
        }
    }
}

fn die(msg: &str) -> ! {
    eprintln!("lui: {}", msg);
    std::process::exit(2);
}
