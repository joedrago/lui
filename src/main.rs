// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

mod config;
mod display;
mod gguf;
mod server;
mod ssh_tunnel;
mod websearch;

use std::ffi::OsString;
use std::path::PathBuf;

use config::{
    config_path, derive_model_name, load_config, model_key, resolve, resolve_hf_alias,
    resolve_model_alias, save_config, update_opencode_config, update_websearch_skill,
    websearch_port, LuiConfig, DEFAULT_BATCH_SIZE, DEFAULT_PARALLEL,
};
use display::Display;
use server::{spawn_server, ConfigSummary};

/// Scope tracks which section of the toml gets written by a setting flag.
/// Sticky: once `--this` is seen, subsequent value flags update the active
/// model's overrides; `--global` flips back. Default is `Global` so plain
/// `lui --temp 0.6` keeps behaving as it did before.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scope {
    Global,
    This,
}

struct RunOpts {
    list: bool,
    debug: Option<String>,
    // Dry-run: print the fully-resolved llama-server command line and exit
    // without spawning. Useful for "what would this actually run?" debugging
    // after a chain of --global / --this edits plus SWA auto-detection.
    cmd: bool,
    // One-shot peer-configuration modes. Neither persists to lui.toml.
    //
    //   ssh_share: run on a server. Configures a client's opencode over
    //       SSH to reach this machine's llama-server via a reverse tunnel.
    //       Prints `ssh -R ...` for this machine to run. SSH is load-
    //       bearing: the client can't always reach back otherwise.
    //
    //   use_lui:   run on a client. Fetches /config from an already-running
    //       --public server over plain HTTP, writes *local* opencode.json +
    //       skill pointing opencode directly at that server's llama-server URL,
    //       spawns a local bsearch server so the browser opens here, and
    //       blocks on Ctrl-C. No SSH.
    ssh_share: Option<ssh_tunnel::SshTarget>,
    use_lui: Option<ssh_tunnel::UseTarget>,
}

// Descriptions all start at column 36 so the right-hand column is aligned
// across every row. The widest entry is `--tb, --threads-batch <N>` at 33
// chars incl. its 8-space indent; column 36 leaves 2 spaces of breathing
// room after that longest row, and more on every shorter row.
const HELP: &str = "\
lui — a friendly TUI wrapper for llama.cpp's llama-server

USAGE:
    lui [OPTIONS] [-- <EXTRA_LLAMA_ARGS>...]

MODEL (identity — always global):
    -m, --model <PATH>             Local GGUF path (or [aliases.model] name)
        --hf <REPO[:QUANT]>        HuggingFace repo (or [aliases.hf] name)
        <NAME>                     Bare positional: looks up either pool
        --alias <NAME>             Alias current --hf or -m as NAME

SCOPE (sticky; defaults to --global):
        --global                   Subsequent settings update [server] (the global defaults)
        --this, --local            Subsequent settings update [models.\"<active-model>\"] only

    Scope toggles may appear multiple times. Example:
        lui --temp 0.6 --this --temp 0.2       # global=0.6, this model=0.2

SETTINGS (scoped; pass `default` as the value to clear a per-scope override):
    -c, --ctx-size <N>             Context window (0 = model default)
        --ngl, --gpu-layers <N>    GPU layers (-1 = all)
        --temp <F>                 Sampling temperature
        --top-p <F>                Top-p (nucleus)
        --top-k <N>                Top-k
        --min-p <F>                Min-p
        --ub, --ubatch-size <N>    Physical batch size (llama-server -ub)
        --batch-size <N>           Logical batch size (llama-server -b)
        --np, --parallel <N>       Server slots (llama-server -np)
        --tb, --threads-batch <N>  Prompt/batch threads (llama-server -tb)
        --ctk, --cache-type-k <T>  KV cache key type (f16, q8_0, ...)
        --ctv, --cache-type-v <T>  KV cache value type
        --swa-full                 Force --swa-full on
        --no-swa-full              Force --swa-full off (disables auto-detect)
        --cache-ram <MIB>          Host-memory prompt cache (llama-server --cache-ram)
        --prio-batch <0-3>         Batch thread priority

MACHINE SETTINGS (always global; rejected with --this):
        --port <N>                 Server port (default 8080)
        --public                   Bind 0.0.0.0 instead of 127.0.0.1
        --websearch                Enable lui's local web-search endpoint
        --no-websearch             Disable and remove its opencode skill
        --web-port <N>             Port for the local web-search endpoint (default: llama port + 1)
        --avo                      Allow VRAM oversubscription (skip lui's abort on GPU over-budget)
        --no-avo                   Abort on VRAM oversubscription (default)

REMOTE (one-shot; not persisted):
        --ssh <USER@HOST>          Run on a server. Configures that
                                   remote's opencode to reach this machine's
                                   llama-server over a reverse tunnel, prints
                                   the matching `ssh -R ...` command, and exits.
        --remote <HOST[:PORT]>     Run on a client. Fetches /config from a
                                   --public server over plain HTTP, writes
                                   local opencode.json pointed directly at that
                                   server's llama-server, and stands up a local
                                   bsearch server so the browser opens on this
                                   machine. Blocks until Ctrl-C. PORT is the
                                   server's HTTP port; defaults to 8081.

OTHER:
    -l, --list                     List cached models and show current config
        --cmd                      Print the resolved llama-server command and exit
        --debug <PATH>             Dump raw llama-server output to a file
    -h, --help                     Show this help

Pass-through (`--`):
    Everything after `--` is appended to llama-server's argv. The pass-through
    list is scoped too: `lui --this -- --some-llama-flag` appends to the active
    model's extra_args; without --this, it appends to the global list.
";

/// Parse argv, mutating `config` in place and returning per-run options.
/// We walk the args in order and track a scope cursor — that's the whole
/// point of not using clap. Anything that can fail the user's intent
/// (e.g. `--this --port 9`, or `--this` with no model) is reported
/// and exits cleanly so we don't silently write a bogus toml.
fn parse_args(config: &mut LuiConfig) -> RunOpts {
    use lexopt::prelude::*;

    // Split argv at `--` so the lexopt loop below only ever sees pre-`--`
    // args. That turns a `Value(v)` match into an unambiguous "bare
    // positional", which we treat as an alias lookup. Everything after
    // `--` bypasses lexopt entirely and lands in extras at the end.
    let all: Vec<OsString> = std::env::args_os().collect();
    let (pre_argv, post_argv): (Vec<OsString>, Vec<OsString>) = {
        let mut iter = all.into_iter();
        let prog = iter.next(); // program name
        let mut pre: Vec<OsString> = prog.into_iter().collect();
        let mut post: Vec<OsString> = Vec::new();
        let mut in_post = false;
        for a in iter {
            if !in_post && a == "--" {
                in_post = true;
                continue;
            }
            if in_post {
                post.push(a);
            } else {
                pre.push(a);
            }
        }
        (pre, post)
    };

    let mut parser = lexopt::Parser::from_args(pre_argv.into_iter().skip(1));
    let mut scope = Scope::Global;
    let mut list = false;
    let mut debug: Option<String> = None;
    let mut cmd = false;
    let mut ssh_share: Option<ssh_tunnel::SshTarget> = None;
    let mut use_lui: Option<ssh_tunnel::UseTarget> = None;

    // Active model key: the one that per-model scoped settings write into.
    // Seeded from the loaded config so `lui --this --temp 0.3` works with
    // no `-m`/`--hf` on the command line, and updated whenever `-m` or `--hf`
    // is seen so chains like `--hf X --this --temp 0.3` write to X.
    let mut active_key: Option<String> = model_key(&config.server);

    // Extra-args replacement tracking: if the user passes ANY extra args for
    // a scope in this invocation, those replace the stored list for that
    // scope. We can't just push-into config.server.extra_args because that
    // would grow the list across runs. None = untouched (keep stored); Some
    // = replace with these.
    let mut new_global_extras: Option<Vec<String>> = None;
    let mut new_model_extras: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();

    loop {
        let arg = match parser.next() {
            Ok(Some(a)) => a,
            Ok(None) => break,
            Err(e) => die(&format!("{}", e)),
        };
        match arg {
            Short('h') | Long("help") => {
                print!("{}", HELP);
                std::process::exit(0);
            }
            Long("global") => scope = Scope::Global,
            Long("this") | Long("local") => scope = Scope::This,

            Short('m') | Long("model") => {
                let v = take_string(&mut parser, "--model");
                let v = resolve_model_alias(config, &v);
                config.server.model = v;
                config.server.hf_repo.clear();
                active_key = model_key(&config.server);
            }
            Long("hf") => {
                let v = take_string(&mut parser, "--hf");
                let v = resolve_hf_alias(config, &v);
                config.server.hf_repo = v;
                config.server.model.clear();
                active_key = model_key(&config.server);
            }

            Long("alias") => {
                let name = take_string(&mut parser, "--alias");
                if name.is_empty() || name.contains('/') || name.contains('=') {
                    die("--alias NAME must be a bare word (no '/' or '=')");
                }
                // Pool inferred from what's currently set:
                //   hf_repo active → [aliases.hf]
                //   model active   → [aliases.model]
                if !config.server.hf_repo.is_empty() {
                    config.aliases.hf.insert(name, config.server.hf_repo.clone());
                } else if !config.server.model.is_empty() {
                    config.aliases.model.insert(name, config.server.model.clone());
                } else {
                    die("--alias requires an active model; pass --hf <repo> or -m <path> first");
                }
            }

            Short('c') | Long("ctx-size") => {
                match take_scalar::<u32>(&mut parser, "--ctx-size") {
                    Some(v) => apply_u32(config, scope, &active_key, "ctx-size", Some(v)),
                    None => apply_u32(config, scope, &active_key, "ctx-size", None),
                }
            }
            Long("ngl") | Long("gpu-layers") => {
                match take_scalar::<i32>(&mut parser, "--gpu-layers") {
                    Some(v) => apply_i32(config, scope, &active_key, "gpu-layers", Some(v)),
                    None => apply_i32(config, scope, &active_key, "gpu-layers", None),
                }
            }
            Long("temp") => set_temp(config, scope, &active_key, take_scalar::<f32>(&mut parser, "--temp")),
            Long("top-p") => set_top_p(config, scope, &active_key, take_scalar::<f32>(&mut parser, "--top-p")),
            Long("top-k") => set_top_k(config, scope, &active_key, take_scalar::<i32>(&mut parser, "--top-k")),
            Long("min-p") => set_min_p(config, scope, &active_key, take_scalar::<f32>(&mut parser, "--min-p")),

            Long("ubatch-size") | Long("ub") => {
                let v = take_scalar::<u32>(&mut parser, "--ubatch-size");
                set_ubatch(config, scope, &active_key, v);
            }
            Long("batch-size") => {
                let v = take_scalar::<u32>(&mut parser, "--batch-size");
                set_batch(config, scope, &active_key, v);
            }
            Long("parallel") | Long("np") => {
                let v = take_scalar::<i32>(&mut parser, "--parallel");
                set_parallel(config, scope, &active_key, v);
            }
            Long("threads-batch") | Long("tb") => {
                let v = take_scalar::<i32>(&mut parser, "--threads-batch");
                set_tb(config, scope, &active_key, v);
            }
            Long("cache-type-k") | Long("ctk") => {
                let v = take_string_or_default(&mut parser, "--cache-type-k");
                set_ctk(config, scope, &active_key, v);
            }
            Long("cache-type-v") | Long("ctv") => {
                let v = take_string_or_default(&mut parser, "--cache-type-v");
                set_ctv(config, scope, &active_key, v);
            }
            Long("swa-full") => set_swa(config, scope, &active_key, Some(true)),
            Long("no-swa-full") => set_swa(config, scope, &active_key, Some(false)),
            Long("cache-ram") => {
                let v = take_scalar::<u32>(&mut parser, "--cache-ram");
                set_cache_ram(config, scope, &active_key, v);
            }
            Long("prio-batch") => {
                let v = take_scalar::<i32>(&mut parser, "--prio-batch");
                set_prio(config, scope, &active_key, v);
            }

            // Machine-only settings: reject --this so the user doesn't
            // silently write a key that resolve() will ignore.
            Long("port") => {
                require_global(scope, "--port");
                config.server.port = take_scalar::<u16>(&mut parser, "--port")
                    .unwrap_or_else(|| die("--port requires a number"));
            }
            Long("public") => {
                require_global(scope, "--public");
                config.server.host = "0.0.0.0".to_string();
            }
            Long("no-websearch") => {
                require_global(scope, "--no-websearch");
                config.server.websearch_disabled = true;
            }
            Long("websearch") => {
                require_global(scope, "--websearch");
                config.server.websearch_disabled = false;
            }
            Long("web-port") => {
                require_global(scope, "--web-port");
                config.server.web_port = take_scalar::<u16>(&mut parser, "--web-port");
            }
            Long("avo") => {
                require_global(scope, "--avo");
                config.server.allow_vram_oversubscription = true;
            }
            Long("no-avo") => {
                require_global(scope, "--no-avo");
                config.server.allow_vram_oversubscription = false;
            }

            Short('l') | Long("list") => list = true,
            Long("cmd") => cmd = true,
            Long("debug") => {
                debug = Some(take_string(&mut parser, "--debug"));
            }
            Long("ssh") => {
                require_global(scope, "--ssh");
                let v = take_string(&mut parser, "--ssh");
                ssh_share =
                    Some(ssh_tunnel::parse_share_target(&v).unwrap_or_else(|e| die(&e)));
            }
            Long("remote") => {
                require_global(scope, "--remote");
                let v = take_string(&mut parser, "--remote");
                use_lui = Some(ssh_tunnel::parse_use_target(&v).unwrap_or_else(|e| die(&e)));
            }

            // Bare positional before `--`: must be an alias in either pool.
            // We check both; an unknown name is a hard error so typos don't
            // silently become llama-server passthrough args.
            Value(v) => {
                let s = os_into_string(v, "positional");
                let hf_hit = config.aliases.hf.get(&s).cloned();
                let model_hit = config.aliases.model.get(&s).cloned();
                match (hf_hit, model_hit) {
                    (Some(_), Some(_)) => {
                        die(&format!("alias {:?} is defined in both [aliases.hf] and [aliases.model]; disambiguate with --hf {} or -m {}", s, s, s));
                    }
                    (Some(target), None) => {
                        config.server.hf_repo = target;
                        config.server.model.clear();
                        active_key = model_key(&config.server);
                    }
                    (None, Some(target)) => {
                        config.server.model = target;
                        config.server.hf_repo.clear();
                        active_key = model_key(&config.server);
                    }
                    (None, None) => {
                        die(&format!("unknown alias: {}", s));
                    }
                }
            }

            other => die(&format!("unknown argument: {}", other.unexpected())),
        }
    }

    // Anything after `--` is pure llama-server passthrough. Scope at the
    // time the loop ended determines whether it appends to global or the
    // active model's extras.
    for a in post_argv {
        let s = os_into_string(a, "extra arg");
        match scope {
            Scope::Global => {
                new_global_extras.get_or_insert_with(Vec::new).push(s);
            }
            Scope::This => {
                let key = active_key.clone().unwrap_or_else(|| {
                    die("--this (for pass-through args) requires an active model; pass --hf or -m first")
                });
                new_model_extras.entry(key).or_default().push(s);
            }
        }
    }

    if let Some(v) = new_global_extras {
        config.server.extra_args = v;
    }
    for (k, v) in new_model_extras {
        config.models.entry(k).or_default().extra_args = v;
    }

    if ssh_share.is_some() && use_lui.is_some() {
        die("--ssh and --remote are mutually exclusive");
    }

    RunOpts {
        list,
        debug,
        cmd,
        ssh_share,
        use_lui,
    }
}

/// Minimal POSIX-style shell quoting for `--cmd` output, so the printed
/// line is directly copy-pasteable. Bare if the arg is strictly
/// alphanum/`-._/:=,+`; otherwise single-quoted with embedded single quotes
/// escaped as `'\''`. Good enough for the llama-server args we emit
/// (notably the JSON kwargs blob, which contains braces and quotes).
fn shell_quote(s: &str) -> String {
    let safe = !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/' | ':' | '=' | ',' | '+'));
    if safe {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for ch in s.chars() {
        if ch == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(ch);
        }
    }
    out.push('\'');
    out
}

fn die(msg: &str) -> ! {
    eprintln!("lui: {}", msg);
    std::process::exit(2);
}

fn require_global(scope: Scope, flag: &str) {
    if scope == Scope::This {
        die(&format!("{} is a machine-wide setting and can't be scoped to --this", flag));
    }
}

fn os_into_string(v: OsString, what: &str) -> String {
    v.into_string().unwrap_or_else(|_| die(&format!("non-UTF8 {}", what)))
}

fn take_string(parser: &mut lexopt::Parser, flag: &str) -> String {
    let v = parser.value().unwrap_or_else(|_| die(&format!("{} requires a value", flag)));
    os_into_string(v, flag)
}

/// Accept either a concrete value or the literal word `default`, which
/// clears the override at the current scope. Returns None for "default".
/// Used by scalar-typed settings (numbers) — the scope handler decides
/// what None means for the global (usually: clear to llama-server default).
fn take_scalar<T: std::str::FromStr>(parser: &mut lexopt::Parser, flag: &str) -> Option<T>
where
    T::Err: std::fmt::Display,
{
    let raw = take_string(parser, flag);
    if raw == "default" {
        return None;
    }
    match raw.parse::<T>() {
        Ok(v) => Some(v),
        Err(e) => die(&format!("{} value {:?} isn't valid: {}", flag, raw, e)),
    }
}

/// String-valued settings: "default" clears, anything else is the value.
fn take_string_or_default(parser: &mut lexopt::Parser, flag: &str) -> Option<String> {
    let raw = take_string(parser, flag);
    if raw == "default" { None } else { Some(raw) }
}

// --- Scope-aware setters. Each one handles a single field for both scopes.
// Pattern: for per-model overrides, None means "clear" and we prune with
// save_config; for global, None means "revert to llama-server default"
// (i.e. set Option<T> fields to None, or the sentinel 0 / -1 for the two
// fields that still use plain integers).

fn require_active_model(key: &Option<String>, flag: &str) -> String {
    key.clone().unwrap_or_else(|| {
        die(&format!("--this {} requires an active model; pass --hf or -m first", flag))
    })
}

fn apply_u32(cfg: &mut LuiConfig, scope: Scope, key: &Option<String>, flag: &str, v: Option<u32>) {
    match (scope, flag) {
        (Scope::Global, "ctx-size") => cfg.server.ctx_size = v.unwrap_or(0),
        (Scope::This, "ctx-size") => {
            let k = require_active_model(key, flag);
            cfg.models.entry(k).or_default().ctx_size = v;
        }
        _ => unreachable!(),
    }
}

fn apply_i32(cfg: &mut LuiConfig, scope: Scope, key: &Option<String>, flag: &str, v: Option<i32>) {
    match (scope, flag) {
        (Scope::Global, "gpu-layers") => cfg.server.gpu_layers = v.unwrap_or(-1),
        (Scope::This, "gpu-layers") => {
            let k = require_active_model(key, flag);
            cfg.models.entry(k).or_default().gpu_layers = v;
        }
        _ => unreachable!(),
    }
}

// The rest are mechanical — one per Option field. Kept as individual
// setters (rather than a macro) because there are only 12 of them and
// readable code is nicer to grep than macro-expanded code.
fn set_temp(cfg: &mut LuiConfig, scope: Scope, key: &Option<String>, v: Option<f32>) {
    match scope {
        Scope::Global => cfg.server.temp = v,
        Scope::This => cfg.models.entry(require_active_model(key, "--temp")).or_default().temp = v,
    }
}
fn set_top_p(cfg: &mut LuiConfig, scope: Scope, key: &Option<String>, v: Option<f32>) {
    match scope {
        Scope::Global => cfg.server.top_p = v,
        Scope::This => cfg.models.entry(require_active_model(key, "--top-p")).or_default().top_p = v,
    }
}
fn set_top_k(cfg: &mut LuiConfig, scope: Scope, key: &Option<String>, v: Option<i32>) {
    match scope {
        Scope::Global => cfg.server.top_k = v,
        Scope::This => cfg.models.entry(require_active_model(key, "--top-k")).or_default().top_k = v,
    }
}
fn set_min_p(cfg: &mut LuiConfig, scope: Scope, key: &Option<String>, v: Option<f32>) {
    match scope {
        Scope::Global => cfg.server.min_p = v,
        Scope::This => cfg.models.entry(require_active_model(key, "--min-p")).or_default().min_p = v,
    }
}
fn set_ubatch(cfg: &mut LuiConfig, scope: Scope, key: &Option<String>, v: Option<u32>) {
    match scope {
        Scope::Global => cfg.server.ubatch_size = v,
        Scope::This => cfg.models.entry(require_active_model(key, "--ubatch-size")).or_default().ubatch_size = v,
    }
}
fn set_batch(cfg: &mut LuiConfig, scope: Scope, key: &Option<String>, v: Option<u32>) {
    match scope {
        Scope::Global => cfg.server.batch_size = v,
        Scope::This => cfg.models.entry(require_active_model(key, "--batch-size")).or_default().batch_size = v,
    }
}
fn set_parallel(cfg: &mut LuiConfig, scope: Scope, key: &Option<String>, v: Option<i32>) {
    match scope {
        Scope::Global => cfg.server.parallel = v,
        Scope::This => cfg.models.entry(require_active_model(key, "--parallel")).or_default().parallel = v,
    }
}
fn set_tb(cfg: &mut LuiConfig, scope: Scope, key: &Option<String>, v: Option<i32>) {
    match scope {
        Scope::Global => cfg.server.threads_batch = v,
        Scope::This => cfg.models.entry(require_active_model(key, "--threads-batch")).or_default().threads_batch = v,
    }
}
fn set_ctk(cfg: &mut LuiConfig, scope: Scope, key: &Option<String>, v: Option<String>) {
    match scope {
        Scope::Global => cfg.server.cache_type_k = v,
        Scope::This => cfg.models.entry(require_active_model(key, "--cache-type-k")).or_default().cache_type_k = v,
    }
}
fn set_ctv(cfg: &mut LuiConfig, scope: Scope, key: &Option<String>, v: Option<String>) {
    match scope {
        Scope::Global => cfg.server.cache_type_v = v,
        Scope::This => cfg.models.entry(require_active_model(key, "--cache-type-v")).or_default().cache_type_v = v,
    }
}
fn set_swa(cfg: &mut LuiConfig, scope: Scope, key: &Option<String>, v: Option<bool>) {
    match scope {
        Scope::Global => cfg.server.swa_full = v,
        Scope::This => cfg.models.entry(require_active_model(key, "--swa-full")).or_default().swa_full = v,
    }
}
fn set_cache_ram(cfg: &mut LuiConfig, scope: Scope, key: &Option<String>, v: Option<u32>) {
    match scope {
        Scope::Global => cfg.server.cache_ram = v,
        Scope::This => cfg.models.entry(require_active_model(key, "--cache-ram")).or_default().cache_ram = v,
    }
}
fn set_prio(cfg: &mut LuiConfig, scope: Scope, key: &Option<String>, v: Option<i32>) {
    match scope {
        Scope::Global => cfg.server.prio_batch = v,
        Scope::This => cfg.models.entry(require_active_model(key, "--prio-batch")).or_default().prio_batch = v,
    }
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
                        // mmproj-*.gguf is the vision projector sidecar, not
                        // a quantization of the model itself — its own F16/BF16
                        // tag would otherwise pollute the quants list.
                        if lower.starts_with("mmproj") {
                            continue;
                        }
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

/// Find the on-disk GGUF for this config, if we can. Handles both -m <path>
/// and --hf <repo>[:quant] (via the HuggingFace hub cache). For split models,
/// prefers shard 00001 (metadata lives there).
fn locate_gguf(s: &config::ServerConfig) -> Option<PathBuf> {
    if !s.model.is_empty() {
        let p = PathBuf::from(&s.model);
        if p.exists() {
            return std::fs::canonicalize(&p).ok().or(Some(p));
        }
        return None;
    }

    if s.hf_repo.is_empty() {
        return None;
    }

    // hf_repo = "Org/Name[-GGUF][:QUANT]"
    let repo = s.hf_repo.split(':').next().unwrap_or(&s.hf_repo);
    let quant = s.hf_repo.split(':').nth(1).map(|q| q.to_lowercase());

    let cache_dir = dirs::home_dir()?
        .join(".cache")
        .join("huggingface")
        .join("hub");
    let folder = format!("models--{}", repo.replace('/', "--"));
    let snapshots = cache_dir.join(folder).join("snapshots");
    if !snapshots.exists() {
        return None;
    }

    let mut candidates: Vec<PathBuf> = Vec::new();
    for snap in std::fs::read_dir(&snapshots).ok()?.flatten() {
        for file in std::fs::read_dir(snap.path()).ok()?.flatten() {
            let fname = file.file_name().to_string_lossy().to_string();
            if !fname.ends_with(".gguf") {
                continue;
            }
            if let Some(ref q) = quant {
                if !fname.to_lowercase().contains(q) {
                    continue;
                }
            }
            candidates.push(file.path());
        }
    }
    candidates.sort();
    let pick = candidates.into_iter().next()?;
    std::fs::canonicalize(&pick).ok().or(Some(pick))
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
    let config_path = config_path();
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

    // Sampling (only shown when set; dim "model default" otherwise)
    let temp_str = s.temp.map(|v| v.to_string()).unwrap_or_else(|| "model default".to_string());
    print_setting("Temperature", &temp_str, "--temp <f>", s.temp.is_none());
    let top_p_str = s.top_p.map(|v| v.to_string()).unwrap_or_else(|| "model default".to_string());
    print_setting("Top-p", &top_p_str, "--top-p <f>", s.top_p.is_none());
    let top_k_str = s.top_k.map(|v| v.to_string()).unwrap_or_else(|| "model default".to_string());
    print_setting("Top-k", &top_k_str, "--top-k <n>", s.top_k.is_none());
    let min_p_str = s.min_p.map(|v| v.to_string()).unwrap_or_else(|| "model default".to_string());
    print_setting("Min-p", &min_p_str, "--min-p <f>", s.min_p.is_none());

    // Performance knobs. lui supplies its own default for parallel; everything
    // else is "server default" until the user sets it.
    let ub_str = s
        .ubatch_size
        .map(|v| v.to_string())
        .unwrap_or_else(|| "server default".to_string());
    print_setting(
        "Ubatch",
        &ub_str,
        "--ubatch-size <n>",
        s.ubatch_size.is_none(),
    );
    let np = s.parallel.unwrap_or(DEFAULT_PARALLEL);
    print_setting(
        "Parallel slots",
        &np.to_string(),
        "--parallel <n>",
        s.parallel.is_none(),
    );
    let b = s.batch_size.unwrap_or(DEFAULT_BATCH_SIZE);
    print_setting("Batch", &b.to_string(), "--batch-size <n>", s.batch_size.is_none());
    let tb_str = s.threads_batch.map(|v| v.to_string()).unwrap_or_else(|| "auto".to_string());
    print_setting("Threads-batch", &tb_str, "--threads-batch <n>", s.threads_batch.is_none());
    let ctk = s.cache_type_k.clone().unwrap_or_else(|| "f16".to_string());
    print_setting("KV type (K)", &ctk, "--cache-type-k <t>", s.cache_type_k.is_none());
    let ctv = s.cache_type_v.clone().unwrap_or_else(|| "f16".to_string());
    print_setting("KV type (V)", &ctv, "--cache-type-v <t>", s.cache_type_v.is_none());
    let (swa_str, swa_is_default) = match s.swa_full {
        Some(true) => ("on (forced)", false),
        Some(false) => ("off (forced)", false),
        None => ("auto (SWA/hybrid detection at launch)", true),
    };
    print_setting("SWA full", swa_str, "--swa-full / --no-swa-full", swa_is_default);
    let cram_str = s.cache_ram.map(|v| format!("{} MiB", v)).unwrap_or_else(|| "server default".to_string());
    print_setting("Cache RAM", &cram_str, "--cache-ram <MiB>", s.cache_ram.is_none());
    let pb_str = s.prio_batch.map(|v| v.to_string()).unwrap_or_else(|| "normal".to_string());
    print_setting("Prio batch", &pb_str, "--prio-batch <0-3>", s.prio_batch.is_none());

    let _ = crossterm::execute!(stdout, Print("\n"));

    // Per-model overrides. Grouped by model so the relationship between
    // [models."Foo"] in the toml and the values here is obvious. The
    // currently-active model's header gets highlighted.
    if !config.models.is_empty() {
        let active = model_key(s);
        let _ = crossterm::execute!(
            stdout,
            SetForegroundColor(lavender),
            SetAttribute(Attribute::Bold),
            Print("  Per-model overrides\n\n"),
            SetAttribute(Attribute::Reset),
            ResetColor,
        );
        for (name, ov) in &config.models {
            let is_active = active.as_deref() == Some(name.as_str());
            let _ = crossterm::execute!(
                stdout,
                Print("  "),
                SetForegroundColor(if is_active { Color::Cyan } else { Color::DarkGrey }),
                SetAttribute(Attribute::Bold),
                Print(if is_active {
                    format!("{} (active)", name)
                } else {
                    name.clone()
                }),
                SetAttribute(Attribute::Reset),
                ResetColor,
                Print("\n"),
            );
            let mut print_kv = |label: &str, val: String| {
                let _ = crossterm::execute!(
                    stdout,
                    Print("      "),
                    SetForegroundColor(muted),
                    Print("· "),
                    SetForegroundColor(lavender),
                    Print(label),
                    Print("  "),
                    SetForegroundColor(Color::Cyan),
                    Print(val),
                    ResetColor,
                    Print("\n"),
                );
            };
            if let Some(v) = ov.ctx_size { print_kv("Context", v.to_string()); }
            if let Some(v) = ov.gpu_layers { print_kv("GPU layers", v.to_string()); }
            if let Some(v) = ov.temp { print_kv("Temperature", v.to_string()); }
            if let Some(v) = ov.top_p { print_kv("Top-p", v.to_string()); }
            if let Some(v) = ov.top_k { print_kv("Top-k", v.to_string()); }
            if let Some(v) = ov.min_p { print_kv("Min-p", v.to_string()); }
            if let Some(v) = ov.ubatch_size { print_kv("Ubatch", v.to_string()); }
            if let Some(v) = ov.batch_size { print_kv("Batch", v.to_string()); }
            if let Some(v) = ov.parallel { print_kv("Parallel slots", v.to_string()); }
            if let Some(v) = ov.threads_batch { print_kv("Threads-batch", v.to_string()); }
            if let Some(v) = &ov.cache_type_k { print_kv("KV type (K)", v.clone()); }
            if let Some(v) = &ov.cache_type_v { print_kv("KV type (V)", v.clone()); }
            if let Some(v) = ov.swa_full { print_kv("SWA full", if v { "on".into() } else { "off".into() }); }
            if let Some(v) = ov.cache_ram { print_kv("Cache RAM", format!("{} MiB", v)); }
            if let Some(v) = ov.prio_batch { print_kv("Prio batch", v.to_string()); }
            if !ov.extra_args.is_empty() {
                print_kv("Extra args (append)", ov.extra_args.join(" "));
            }
        }
        let _ = crossterm::execute!(stdout, Print("\n"));
    }

    // Aliases — split into the two pools so users can see which flag each
    // alias resolves under. Positional `lui NAME` checks both pools.
    if !config.aliases.is_empty() {
        let _ = crossterm::execute!(
            stdout,
            SetForegroundColor(lavender),
            SetAttribute(Attribute::Bold),
            Print("  Aliases\n\n"),
            SetAttribute(Attribute::Reset),
            ResetColor,
        );
        let mut print_pool = |label: &str, pool: &std::collections::BTreeMap<String, String>| {
            if pool.is_empty() {
                return;
            }
            let _ = crossterm::execute!(
                stdout,
                Print("    "),
                SetForegroundColor(Color::DarkGrey),
                Print(label),
                ResetColor,
                Print("\n"),
            );
            for (name, target) in pool {
                let _ = crossterm::execute!(
                    stdout,
                    Print("      "),
                    SetForegroundColor(Color::Cyan),
                    SetAttribute(Attribute::Bold),
                    Print(name),
                    SetAttribute(Attribute::Reset),
                    SetForegroundColor(Color::DarkGrey),
                    Print(" → "),
                    SetForegroundColor(lavender),
                    Print(target),
                    ResetColor,
                    Print("\n"),
                );
            }
        };
        print_pool("[aliases.hf]", &config.aliases.hf);
        print_pool("[aliases.model]", &config.aliases.model);
        let _ = crossterm::execute!(stdout, Print("\n"));
    }
}

fn list_cached_models() {
    print_current_config();

    let config = load_config();
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

        // Any [aliases.hf] entries that target this repo — show each as
        // the terser positional form `lui <alias>`.
        for (alias_name, alias_target) in &config.aliases.hf {
            let target_repo = alias_target.split(':').next().unwrap_or(alias_target);
            if target_repo == model.repo {
                let _ = crossterm::execute!(
                    stdout,
                    Print("    "),
                    SetForegroundColor(Color::DarkGrey),
                    Print("lui "),
                    SetForegroundColor(Color::Cyan),
                    Print(alias_name),
                    ResetColor,
                    Print("\n"),
                );
            }
        }
    }
    println!();
}

#[tokio::main]
async fn main() {
    // Load stored config, then walk argv in order to apply CLI edits
    // (including scope toggles). parse_args mutates `config` directly; at
    // this point `config` reflects the user's persisted intent.
    let mut config = load_config();
    let opts = parse_args(&mut config);

    // Handle --list after the CLI has been applied but before any model
    // validation, so `lui -l` works even with an empty config.
    if opts.list {
        list_cached_models();
        return;
    }

    // Validate we have a model
    if config.server.model.is_empty() && config.server.hf_repo.is_empty() {
        eprintln!("Error: no model specified. Use --hf <repo> or -m <path>, or run 'lui --list' to see cached models.");
        std::process::exit(1);
    }

    // Save config (stores user intent only; auto-detected values are resolved below).
    save_config(&config);

    // Fold per-model overrides on top of the global server config to get
    // the flattened config that actually drives llama-server.
    let mut effective = resolve(&config);

    // Resolve --swa-full when the user hasn't made an explicit choice.
    // Stored as None in TOML so we re-detect next run if the model changes.
    if effective.swa_full.is_none() {
        if let Some(path) = locate_gguf(&effective) {
            if let Ok(meta) = gguf::read_gguf_metadata(&path) {
                if gguf::uses_sliding_window(&meta) {
                    effective.swa_full = Some(true);
                }
            }
        }
    }

    // --ssh: one-shot client configuration. Doesn't spawn
    // llama-server, doesn't touch local opencode. Placed after SWA auto-
    // detect so the remote config reflects the same effective server config
    // we would have launched with. We deliberately ran save_config above:
    // it's fine for a `lui --hf X --ssh ...` invocation to also record
    // that --hf intent in lui.toml, since that isn't --ssh-specific.
    if let Some(target) = &opts.ssh_share {
        if let Err(e) = ssh_tunnel::setup_share(target, &effective) {
            eprintln!("lui: {}", e);
            std::process::exit(1);
        }
        return;
    }

    // --use: this machine is a client; point ourselves at an already-
    // running --public server by fetching its /config over HTTP, writing our
    // own ~/.config/opencode/opencode.json + skill pointed directly at that
    // server, and standing up a local bsearch server so the browser opens on
    // this machine. Blocks until Ctrl-C. Doesn't spawn llama-server and
    // doesn't touch lui.toml beyond whatever was already saved above.
    if let Some(target) = &opts.use_lui {
        if let Err(e) = ssh_tunnel::setup_use(target).await {
            eprintln!("lui: {}", e);
            std::process::exit(1);
        }
        return;
    }

    // --cmd: print the fully-resolved llama-server invocation and exit.
    // Placed after SWA auto-detect so the printed line matches what we'd
    // actually launch, but BEFORE any side effects (opencode config,
    // websearch skill, brew check, spawning) — a dry run shouldn't mutate
    // anything outside lui.toml (which save_config already handled above).
    if opts.cmd {
        let args = server::build_args(&effective);
        let mut line = String::from("llama-server");
        for a in &args {
            line.push(' ');
            line.push_str(&shell_quote(a));
        }
        println!("{}", line);
        return;
    }

    // Update opencode config
    update_opencode_config(&effective);

    // Write (or remove) the lui-web-search opencode skill.
    update_websearch_skill(&effective);

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
    let mut proc = match spawn_server(&effective, opts.debug.as_deref()) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    // Spawn the local lui HTTP server. Always mounts /health and /config so
    // a client's `lui --ssh` can discover us regardless of whether
    // browser-mediated web search is on; the /bsearch side of the server
    // only mounts when websearch is enabled. Binds to config.host so
    // --public also opens up /config to the LAN.
    let web_port = websearch_port(&effective);
    let config_info = websearch::LuiConfigResponse {
        version: websearch::CONFIG_VERSION,
        llama_port: effective.port,
        web_port,
        websearch_disabled: effective.websearch_disabled,
        model_name: derive_model_name(&effective),
    };
    // Shared start time: the lui HTTP server reports uptime off this clock
    // via `/data`, and the Display uses the same `start_time` so the local
    // renderer and any future client renderer agree on server lifetime.
    let start_time = std::time::Instant::now();
    let config_summary = ConfigSummary::from_config(&effective);
    websearch::spawn(
        &effective.host,
        web_port,
        proc.state.clone(),
        config_info,
        start_time,
        config_summary,
    );

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
    let config_for_opencode = effective.clone();
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

    // Create display. Polls `/data` on 127.0.0.1 — whether the server bound
    // to 127.0.0.1 or 0.0.0.0 (under --public), loopback reaches it either
    // way. The local `ServerState` is handed through only so the Server Log
    // panel can keep its cheap direct access to the ring buffer. The
    // bookmarklet URL is always local (it's a browser-on-this-machine
    // thing); we pass `None` when websearch is disabled so the renderer
    // omits that header row.
    let local_setup_url = (!effective.websearch_disabled)
        .then(|| format!("http://127.0.0.1:{}/setup", web_port));
    let display = Display::new(
        "127.0.0.1".to_string(),
        web_port,
        Some(proc.state.clone()),
        local_setup_url,
        false,
    );

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

    // Propagate fatal errors (e.g. GPU VRAM oversubscribed) as a non-zero
    // exit code so shell scripts and launchers notice.
    let had_fatal = proc.state.lock().unwrap().fatal_reason.is_some();
    if had_fatal {
        std::process::exit(1);
    }
}
