// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

mod config;
mod display;
mod gguf;
mod harness;
mod server;
mod settings;
mod ssh_tunnel;
mod websearch;

use std::ffi::OsString;
use std::path::PathBuf;

use config::{config_path, derive_model_name, load_config, model_key, save_config, websearch_port};
use display::Display;
use server::{spawn_server, ConfigSummary};
use settings::registry::Registry;
use settings::setting::{Scope as RegScope, Setting};
use settings::store::{validate_integer, Config};
use settings::value::{Value as SettingValue, ValueKind};

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

/// Parse argv, mutating `config` in place and returning per-run options.
/// The parse loop writes exclusively into `Config`. Adding a new setting
/// is a matter of declaring it in `settings::registry`; the dispatch below
/// picks it up automatically.
fn parse_args(config: &mut Config) -> RunOpts {
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

    let reg = Registry::build();
    let mut parser = lexopt::Parser::from_args(pre_argv.into_iter().skip(1));

    // Sticky scope cursor: flipped by `--global` / `--this`, consulted by
    // `handle_flag` to decide where to write scoped settings. Starts global
    // so plain `lui --temp 0.6` keeps its previous meaning.
    let mut scope_is_this = false;

    let mut list = false;
    let mut debug: Option<String> = None;
    let mut cmd = false;
    let mut ssh_share: Option<ssh_tunnel::SshTarget> = None;
    let mut use_lui: Option<ssh_tunnel::UseTarget> = None;

    // Active model key: the one that per-model scoped settings write into.
    // Seeded from the loaded `[server].active_model`, so `lui --this --temp
    // 0.3` works with no positional. Updated whenever a bare positional
    // or alias target resolves.
    let mut active_key: Option<String> = model_key(config);

    // Tracks whether the user passed an explicit type flag (-m or --hf)
    // this invocation. Relevant for the "new model requires type" check at
    // the end of parse — if the bare positional names a model that has no
    // [models."X"] entry, we refuse to run without a type hint.
    let mut type_flag_this_run: Option<&'static str> = None;
    let mut positional_was_new_model: bool = false;

    // Extra-args replacement tracking: if the user passes ANY extra args for
    // a scope in this invocation, those replace the stored list for that
    // scope. We can't just push-into the store because that would grow the
    // list across runs. None = untouched (keep stored); Some = replace.
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
                // Registry-driven help: one declarative place produces the
                // flag table, so adding a setting no longer requires
                // retuning a hand-aligned HELP string.
                print!("{}", settings::help::emit_help(&reg));
                std::process::exit(0);
            }
            Long("global") => scope_is_this = false,
            Long("this") | Long("local") => scope_is_this = true,

            // Zero-arg type setters: flag the active model's per-model
            // `type` as `local` or `huggingface`. We accumulate here and
            // apply at end-of-parse so the flag can appear in any order
            // relative to the positional — both `lui X --hf` and
            // `lui --hf X` work.
            Short('m') | Long("model") => {
                type_flag_this_run = Some("local");
            }
            Long("hf") => {
                type_flag_this_run = Some("huggingface");
            }

            // Ephemeral transformers (side-effects, not stored settings).
            Long("alias") => {
                let name = take_string(&mut parser, "--alias");
                if name.is_empty() || name.contains('/') || name.contains('=') {
                    die("--alias NAME must be a bare word (no '/' or '=')");
                }
                let target = active_key.clone().unwrap_or_else(|| {
                    die("--alias requires an active model; pass a positional name first")
                });
                config.aliases.insert(name, target);
            }
            Long("public") => {
                require_global(scope_is_this, "--public");
                config
                    .global
                    .set("host", SettingValue::String("0.0.0.0".to_string()));
            }

            // Ephemeral mode flags — they leave the parse loop as RunOpts
            // rather than landing in any store.
            Short('l') | Long("list") => list = true,
            Long("cmd") => cmd = true,
            Long("debug") => {
                debug = Some(take_string(&mut parser, "--debug"));
            }
            Long("ssh") => {
                require_global(scope_is_this, "--ssh");
                let v = take_string(&mut parser, "--ssh");
                ssh_share = Some(ssh_tunnel::parse_share_target(&v).unwrap_or_else(|e| die(&e)));
            }
            Long("remote") => {
                require_global(scope_is_this, "--remote");
                let v = take_string(&mut parser, "--remote");
                use_lui = Some(ssh_tunnel::parse_use_target(&v).unwrap_or_else(|e| die(&e)));
            }

            // Bare positional: free-form model key. Resolved against the
            // unified alias pool first; unresolved names pass through as
            // literals. Updates active_model + active_key.
            Value(v) => {
                let raw = os_into_string(v, "positional");
                let target = config
                    .aliases
                    .get(&raw)
                    .cloned()
                    .unwrap_or_else(|| raw.clone());
                let is_new = !config.per_model.contains_key(&target);
                config
                    .global
                    .set("active_model", SettingValue::String(target.clone()));
                active_key = Some(target);
                if is_new {
                    positional_was_new_model = true;
                }
            }

            // Every other flag: look it up in the registry and dispatch
            // based on kind + scope + passthrough. This is where the 16
            // bespoke setters used to live.
            Short(c) => match reg.lookup_short(c) {
                Some(setting) => handle_flag(
                    &reg,
                    setting,
                    false,
                    &mut parser,
                    config,
                    scope_is_this,
                    &active_key,
                ),
                None => die(&format!("unknown -{}", c)),
            },
            Long(name) => match reg.lookup_long(name) {
                Some((lookup, setting)) => handle_flag(
                    &reg,
                    setting,
                    lookup.negated,
                    &mut parser,
                    config,
                    scope_is_this,
                    &active_key,
                ),
                None => die(&format!("unknown --{}", name)),
            },
        }
    }

    // Anything after `--` is pure llama-server passthrough. Scope at the
    // time the loop ended determines whether it appends to global or the
    // active model's extras.
    for a in post_argv {
        let s = os_into_string(a, "extra arg");
        if scope_is_this {
            let key = active_key.clone().unwrap_or_else(|| {
                die("--this (for pass-through args) requires an active model; pass --hf or -m first")
            });
            new_model_extras.entry(key).or_default().push(s);
        } else {
            new_global_extras.get_or_insert_with(Vec::new).push(s);
        }
    }
    if let Some(v) = new_global_extras {
        if v.is_empty() {
            config.global.unset("extra_args");
        } else {
            config
                .global
                .set("extra_args", SettingValue::StringArray(v));
        }
    }
    for (k, v) in new_model_extras {
        let store = config.per_model.entry(k).or_default();
        if v.is_empty() {
            store.unset("extra_args");
        } else {
            store.set("extra_args", SettingValue::StringArray(v));
        }
    }

    if ssh_share.is_some() && use_lui.is_some() {
        die("--ssh and --remote are mutually exclusive");
    }

    // Apply the accumulated type flag (`--hf` / `-m`) to the active model's
    // per-model store. Deferred to end-of-parse so the flag can appear in
    // any order relative to the positional.
    if let Some(ty) = type_flag_this_run {
        match active_key {
            Some(ref key) => {
                config
                    .per_model
                    .entry(key.clone())
                    .or_default()
                    .set("type", SettingValue::String(ty.to_string()));
            }
            None => {
                let flag = if ty == "local" { "-m" } else { "--hf" };
                die(&format!(
                    "{} requires an active model; pass a positional name or rely on [server].active_model",
                    flag
                ));
            }
        }
    }

    // New-model-without-type check: a brand-new positional (not present in
    // [models."X"]) needs an explicit type hint. Without it the registry
    // doesn't know whether to synthesize `-hf <key>` or `-m <key>` for
    // llama-server, and silently guessing is worse than a clear error.
    if positional_was_new_model && type_flag_this_run.is_none() {
        if let Some(ref k) = active_key {
            die(&format!(
                "Model {} is new — pass --hf or -m to set its type",
                k
            ));
        }
    }

    RunOpts {
        list,
        debug,
        cmd,
        ssh_share,
        use_lui,
    }
}

/// Dispatch a registry-declared flag. Consults the setting's scope /
/// kind / passthrough mode to decide how to read the next argv token (if
/// any) and where the resulting value lands in `config`. Ephemeral
/// settings aren't routed here — the main match arm handles their side
/// effects explicitly.
fn handle_flag(
    reg: &Registry,
    setting: &Setting,
    negated: bool,
    parser: &mut lexopt::Parser,
    config: &mut Config,
    scope_is_this: bool,
    active_key: &Option<String>,
) {
    let flag_display = primary_flag_display(setting, negated);

    // Scope guard. Ephemerals and `Both`-scope settings don't gate; the
    // two pinned-scope cases (Global, PerModel) fire early.
    match setting.scope {
        RegScope::Global if scope_is_this => die(&format!(
            "{} is a machine-wide setting and can't be scoped to --this",
            flag_display
        )),
        RegScope::PerModel if !scope_is_this => die(&format!(
            "{} is a per-model setting; prefix with --this",
            flag_display
        )),
        _ => {}
    }

    // Read the value. `None` means "clear" — matches the legacy
    // take-scalar-or-default behavior, so `--temp default` unsets.
    let value: Option<SettingValue> = match setting.kind {
        ValueKind::Bool => Some(SettingValue::Bool(!negated)),
        ValueKind::Integer => {
            let raw = take_string(parser, &flag_display);
            if raw == "default" {
                None
            } else {
                match raw.parse::<i64>() {
                    Ok(n) => match validate_integer(reg, setting.name, n) {
                        Ok(n) => Some(SettingValue::Integer(n)),
                        Err(reason) => die(&format!("{}: {}", flag_display, reason)),
                    },
                    Err(e) => die(&format!(
                        "{} value {:?} isn't valid: {}",
                        flag_display, raw, e
                    )),
                }
            }
        }
        ValueKind::Float => {
            let raw = take_string(parser, &flag_display);
            if raw == "default" {
                None
            } else {
                match raw.parse::<f64>() {
                    Ok(f) => Some(SettingValue::Float(f)),
                    Err(e) => die(&format!(
                        "{} value {:?} isn't valid: {}",
                        flag_display, raw, e
                    )),
                }
            }
        }
        ValueKind::String => {
            let raw = take_string(parser, &flag_display);
            if raw == "default" {
                None
            } else {
                Some(SettingValue::String(raw))
            }
        }
        ValueKind::StringArray | ValueKind::Map => {
            // Composites aren't CLI-reachable for registry-declared flags.
            // `extra_args` is handled by the post-`--` tail; `chat_template_kwargs`
            // is TOML-only.
            die(&format!(
                "{} can't be set on the command line",
                flag_display
            ));
        }
    };

    // Target layer: Global settings always land in the global store;
    // Both / PerModel settings honor the sticky scope.
    let target_is_this = match setting.scope {
        RegScope::Global => false,
        RegScope::PerModel => true,
        RegScope::Both => scope_is_this,
        // Ephemerals aren't routed here; if one somehow slips through,
        // default to global so we can't silently shadow a per-model store.
        RegScope::Ephemeral => false,
    };

    if target_is_this {
        let key = active_key.clone().unwrap_or_else(|| {
            die(&format!(
                "--this {} requires an active model; pass --hf or -m first",
                flag_display
            ))
        });
        let store = config.per_model.entry(key).or_default();
        match value {
            Some(v) => store.set(setting.name, v),
            None => store.unset(setting.name),
        }
    } else {
        match value {
            Some(v) => config.global.set(setting.name, v),
            None => config.global.unset(setting.name),
        }
    }
}

/// Produce the flag form that's displayed in error messages. Picks the
/// short flag when one exists and we're not in the negated form; falls
/// back to the primary long form (or the `--no-<long>` inverse when
/// `negated` is true for a bool).
fn primary_flag_display(setting: &Setting, negated: bool) -> String {
    if negated {
        if let Some(long) = setting.long {
            return format!("--no-{}", long);
        }
    }
    if let Some(c) = setting.short {
        return format!("-{}", c);
    }
    if let Some(long) = setting.long {
        return format!("--{}", long);
    }
    format!("<{}>", setting.name)
}

/// Minimal POSIX-style shell quoting for `--cmd` output, so the printed
/// line is directly copy-pasteable. Bare if the arg is strictly
/// alphanum/`-._/:=,+`; otherwise single-quoted with embedded single quotes
/// escaped as `'\''`. Good enough for the llama-server args we emit
/// (notably the JSON kwargs blob, which contains braces and quotes).
fn shell_quote(s: &str) -> String {
    let safe = !s.is_empty()
        && s.chars().all(|c| {
            c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/' | ':' | '=' | ',' | '+')
        });
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

fn require_global(scope_is_this: bool, flag: &str) {
    if scope_is_this {
        die(&format!(
            "{} is a machine-wide setting and can't be scoped to --this",
            flag
        ));
    }
}

fn os_into_string(v: OsString, what: &str) -> String {
    v.into_string()
        .unwrap_or_else(|_| die(&format!("non-UTF8 {}", what)))
}

fn take_string(parser: &mut lexopt::Parser, flag: &str) -> String {
    let v = parser
        .value()
        .unwrap_or_else(|_| die(&format!("{} requires a value", flag)));
    os_into_string(v, flag)
}

/// Assemble the universal inputs every harness apply fn takes. `ctx_size`
/// is explicit because the post-ready task passes llama-server's reported
/// value (which may differ from what the user configured).
fn build_harness_inputs(eff: &settings::store::Effective, ctx_size: u32) -> harness::HarnessInputs {
    let port = eff.get_i64("port").unwrap_or(8080) as u16;
    harness::HarnessInputs {
        model_name: derive_model_name(eff),
        base_url: format!("http://localhost:{}/v1", port),
        ctx_size,
        web_port: websearch_port(eff),
        websearch: eff.get_bool("websearch").unwrap_or(true),
    }
}

/// Find the on-disk GGUF for this config, if we can. Handles both -m <path>
/// and --hf <repo>[:quant] (via the HuggingFace hub cache). For split models,
/// prefers shard 00001 (metadata lives there).
fn locate_gguf(eff: &settings::store::Effective) -> Option<PathBuf> {
    if let Some(m) = eff.get_string("model") {
        if !m.is_empty() {
            let p = PathBuf::from(m);
            if p.exists() {
                return std::fs::canonicalize(&p).ok().or(Some(p));
            }
            return None;
        }
    }

    let hf = eff.get_string("hf_repo").unwrap_or("");
    if hf.is_empty() {
        return None;
    }

    // hf_repo = "Org/Name[-GGUF][:QUANT]"
    let repo = hf.split(':').next().unwrap_or(hf);
    let quant = hf.split(':').nth(1).map(|q| q.to_lowercase());

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

/// Registry-driven pretty-printer for `lui --list`. The body is a walk
/// over every settings-section flag the registry declares; adding a new
/// setting to the registry surfaces here automatically. Four display
/// quirks stay hand-coded so the output still reads as a summary and not
/// a flat key/value dump: the Model row (synthesizes the flag from the
/// active model's `type`), Bind (flag doubles as a directive toggle),
/// SWA full ("on (forced)" / "off (forced)" / "auto"), and Cache RAM (MiB
/// unit). Everything else is a uniform registry loop.
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
    let reg = Registry::build();
    let active_key_opt = model_key(&config);
    let eff = config.effective(&reg, active_key_opt.as_deref());

    // ---------- Header ----------
    let _ = crossterm::execute!(
        stdout,
        Print("\n"),
        SetForegroundColor(lavender),
        SetAttribute(Attribute::Bold),
        Print("  Current config"),
        SetAttribute(Attribute::Reset),
        ResetColor,
    );
    let cp = config_path();
    let _ = crossterm::execute!(
        stdout,
        SetForegroundColor(Color::DarkGrey),
        Print(if cp.exists() {
            format!(" ({})", cp.display())
        } else {
            " (no config file yet)".to_string()
        }),
        ResetColor,
        Print("\n\n"),
    );

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

    // ---------- Model (special: flag synthesized from active model's type) ----------
    let active_key_for_display = active_key_opt.clone();
    let active_ty: Option<String> = active_key_for_display.as_ref().and_then(|k| {
        config
            .per_model
            .get(k)
            .and_then(|store| match store.get("type") {
                Some(SettingValue::String(t)) => Some(t.clone()),
                _ => None,
            })
    });
    match (active_key_for_display.as_deref(), active_ty.as_deref()) {
        (Some(key), Some("huggingface")) => print_setting("Model", key, "--hf <NAME>", false),
        (Some(key), Some("local")) => print_setting("Model", key, "-m <NAME>", false),
        (Some(key), _) => print_setting("Model", key, "<NAME>", false),
        (None, _) => print_setting("Model", "(none)", "<NAME> (with --hf or -m)", true),
    }

    // ---------- Bind (special: `--public` toggles between two host values) ----------
    let host = eff.get_string("host").unwrap_or("127.0.0.1").to_string();
    let host_flag = if host == "127.0.0.1" {
        "--public"
    } else {
        "--public (set)"
    };
    print_setting("Bind", &host, host_flag, host == "127.0.0.1");

    // ---------- Everything else: one loop over the registry ----------
    //
    // Every declared setting that has a long flag and isn't an ephemeral
    // mode flag appears in order. Label, value, and unset-phrase all come
    // from the setting's registry metadata; adding a new setting to the
    // registry surfaces here automatically.
    for setting in reg.iter_by_section() {
        if !shows_in_current_config(setting) {
            continue;
        }
        let Some(value) = render_config_row_value(&eff, setting) else {
            continue;
        };
        let label = setting.derived_ui_label();
        let flag = setting_flag_hint(setting);
        let is_default = !eff.is_explicitly_set(setting.name);
        print_setting(&label, &value, &flag, is_default);
    }

    let _ = crossterm::execute!(stdout, Print("\n"));

    // ---------- Per-model overrides ----------
    if !config.per_model.is_empty() {
        let _ = crossterm::execute!(
            stdout,
            SetForegroundColor(lavender),
            SetAttribute(Attribute::Bold),
            Print("  Per-model overrides\n\n"),
            SetAttribute(Attribute::Reset),
            ResetColor,
        );
        for (name, store) in &config.per_model {
            let is_active = active_key_opt.as_deref() == Some(name.as_str());
            let _ = crossterm::execute!(
                stdout,
                Print("  "),
                SetForegroundColor(if is_active {
                    Color::Cyan
                } else {
                    Color::DarkGrey
                }),
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
            // Copy-pasteable command line. The per_model entry already
            // exists, so the positional form works — no -m / --hf
            // needed; those are only for net-new models.
            let _ = crossterm::execute!(
                stdout,
                Print("      "),
                SetForegroundColor(Color::DarkGrey),
                Print(format!("lui {}\n", name)),
                ResetColor,
            );
            // Registry-order walk over fields the user actually set on
            // this model. The `Effective` view here layers per-model
            // over-empty-global so `ui_format` sees the usual read view.
            // `type` is filtered out — it's an identity tag surfaced by
            // the command line above, not an override row.
            let per_model_eff = settings::store::Effective {
                registry: &reg,
                global: &settings::store::Store::new(),
                per_model: Some(store),
            };
            for setting in reg.iter_by_section() {
                if setting.name == "type" || !store.contains(setting.name) {
                    continue;
                }
                let Some(display) = render_config_row_value(&per_model_eff, setting) else {
                    continue;
                };
                let label = setting.derived_ui_label();
                let _ = crossterm::execute!(
                    stdout,
                    Print("      "),
                    SetForegroundColor(muted),
                    Print("· "),
                    SetForegroundColor(lavender),
                    Print(&label),
                    Print("  "),
                    SetForegroundColor(Color::Cyan),
                    Print(display),
                    ResetColor,
                    Print("\n"),
                );
            }
        }
        let _ = crossterm::execute!(stdout, Print("\n"));
    }

    // ---------- Aliases (unified pool) ----------
    if !config.aliases.is_empty() {
        let _ = crossterm::execute!(
            stdout,
            SetForegroundColor(lavender),
            SetAttribute(Attribute::Bold),
            Print("  Aliases\n\n"),
            SetAttribute(Attribute::Reset),
            ResetColor,
        );
        for (name, target) in &config.aliases {
            let _ = crossterm::execute!(
                stdout,
                Print("    "),
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
        let _ = crossterm::execute!(stdout, Print("\n"));
    }
}

/// Whether a setting earns a row on the "Current config" view. Identity
/// carriers (active_model / model / hf_repo / host) don't — they're
/// surfaced by the hand-written Model and Bind rows above. Ephemeral
/// mode flags don't have a stored value worth showing. Everything else
/// with a long flag appears.
fn shows_in_current_config(s: &settings::setting::Setting) -> bool {
    if s.long.is_none() {
        return false;
    }
    if s.scope == settings::setting::Scope::Ephemeral {
        return false;
    }
    if s.section == "MODEL" {
        return false;
    }
    true
}

/// Render the value cell via the setting's `ui_format` if present, else
/// fall back to `Value::display()`. Returns `None` when nothing useful
/// can be shown (no value, no formatter output, no `ui_unset` phrase) —
/// the caller skips the row entirely, which is how "uninteresting
/// settings stay hidden until explicitly set" is implemented.
fn render_config_row_value(
    eff: &settings::store::Effective,
    setting: &settings::setting::Setting,
) -> Option<String> {
    let raw = eff.get(setting.name);
    if let Some(fmt) = setting.ui_format {
        if let Some(s) = fmt(raw, eff, setting) {
            return Some(s);
        }
    } else if let Some(v) = raw {
        return Some(v.display());
    }
    setting.ui_unset.map(str::to_string)
}

/// Render the `--long <PLACEHOLDER>` hint for the second line under a
/// config row. Bools with a negated `--no-<long>` show both forms;
/// value-taking flags show their placeholder when one is declared.
fn setting_flag_hint(s: &settings::setting::Setting) -> String {
    let long = s.long.unwrap_or("");
    if s.kind == settings::value::ValueKind::Bool && s.has_no_form() {
        return format!("--{} / --no-{}", long, long);
    }
    if s.placeholder.is_empty() {
        format!("--{}", long)
    } else {
        format!("--{} <{}>", long, s.placeholder)
    }
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
        print_current_config();
        return;
    }

    // Validate we have a model.
    if config::model_key(&config).is_none() {
        eprintln!(
            "Error: no model specified. Pass a positional model name (with --hf or -m for new models), or run 'lui --list' to see cached models."
        );
        std::process::exit(1);
    }

    // Save config (config user intent only; auto-detected values are resolved below).
    save_config(&config);

    // Build an Effective view once; everything below reads through it.
    let reg = Registry::build();
    let active_key: Option<String> = config::model_key(&config);

    // Resolve --swa-full when the user hasn't made an explicit choice.
    // Stored as None in TOML so we re-detect next run if the model changes.
    let swa_explicit = {
        let eff = config.effective(&reg, active_key.as_deref());
        eff.get_bool("swa_full").is_some()
    };
    if !swa_explicit {
        let gguf_path = {
            let eff = config.effective(&reg, active_key.as_deref());
            locate_gguf(&eff)
        };
        if let Some(path) = gguf_path {
            if let Ok(meta) = gguf::read_gguf_metadata(&path) {
                if gguf::uses_sliding_window(&meta) {
                    config.global.set("swa_full", SettingValue::Bool(true));
                }
            }
        }
    }

    let effective = config.effective(&reg, active_key.as_deref());

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
        if let Err(e) = ssh_tunnel::setup_use(target, &effective).await {
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

    // First harness pass: every enabled harness writes its config + skill
    // with whatever ctx_size the user has configured (may be 0 for
    // "model default" — llama-server will tell us the real value once it
    // starts; the post-ready task below rewrites with that).
    let initial_ctx_size = effective.get_i64("ctx_size").unwrap_or(0) as u32;
    let inputs_pre = build_harness_inputs(&effective, initial_ctx_size);
    for h in harness::HARNESSES {
        if effective.get_bool(h.setting_name).unwrap_or(h.default_on) {
            harness::apply_local(h, &effective, &inputs_pre);
        }
    }

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
    let host_for_spawn = effective
        .get_string("host")
        .unwrap_or("127.0.0.1")
        .to_string();
    let llama_port = effective.get_i64("port").unwrap_or(8080) as u16;
    let web_port = websearch_port(&effective);
    let config_info = websearch::LuiConfigResponse {
        version: websearch::CONFIG_VERSION,
        llama_port,
        web_port,
        websearch: effective.get_bool("websearch").unwrap_or(true),
        model_name: derive_model_name(&effective),
        ctx_size: effective.get_i64("ctx_size").unwrap_or(0) as u32,
    };
    // Shared start time: the lui HTTP server reports uptime off this clock
    // via `/data`, and the Display uses the same `start_time` so the local
    // renderer and any future client renderer agree on server lifetime.
    let start_time = std::time::Instant::now();
    let config_summary = ConfigSummary::from_effective(&effective);
    let setting_entries = server::build_setting_entries(&effective);
    websearch::spawn(
        &host_for_spawn,
        web_port,
        proc.state.clone(),
        config_info,
        start_time,
        config_summary,
        setting_entries,
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

    // Re-apply harnesses once the server is ready and we know the actual
    // ctx_size. We clone the Config so the task owns its own data and
    // rebuild `Effective` inside; no borrow crosses `.await`.
    let state_for_harness = proc.state.clone();
    let config_for_harness = config.clone();
    let active_key_for_harness = active_key.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            let st = state_for_harness.lock().unwrap();
            if st.ready && st.ctx_size > 0 {
                let actual_ctx_size = st.ctx_size;
                drop(st);
                let reg = Registry::build();
                let eff = config_for_harness.effective(&reg, active_key_for_harness.as_deref());
                let inputs = build_harness_inputs(&eff, actual_ctx_size);
                for h in harness::HARNESSES {
                    if eff.get_bool(h.setting_name).unwrap_or(h.default_on) {
                        harness::apply_local(h, &eff, &inputs);
                    }
                }
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
    let websearch = effective.get_bool("websearch").unwrap_or(true);
    let local_setup_url = websearch.then(|| format!("http://127.0.0.1:{}/setup", web_port));
    let display = Display::new(
        "127.0.0.1".to_string(),
        web_port,
        Some(proc.state.clone()),
        local_setup_url,
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
