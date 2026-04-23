// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

//! The single declarative registry. `declare_all_settings` is the one
//! place a contributor edits when they want to add, remove, or tweak a
//! lui option. Every downstream concern — help output, argv parsing,
//! TOML IO, llama-server passthrough, UI payload — walks this registry
//! rather than referring to settings by name.

use std::collections::HashMap;

use super::setting::{PassthroughMode, Scope, Setting};
use super::value::{Value, ValueKind};

/// Ordered list of help sections and their narrative content. A
/// `Setting::section` string must match one of these `name`s to appear in
/// the right block; anything unmatched gets bucketed at the end of
/// `--help`. Adding a new section is one entry here; the help printer
/// walks this list and auto-renders every field.
///
/// `title` is the one-line header. `preamble` (rare) appears between the
/// title and the first flag row. `extra_rows` render as synthetic aligned
/// rows after real flags (used for the positional `<NAME>` in MODEL, which
/// isn't a registry flag but belongs in that block visually). `postamble`
/// is freeform multi-line text appended after the rows.
pub struct Section {
    pub name: &'static str,
    pub title: &'static str,
    pub preamble: &'static str,
    pub extra_rows: &'static [ExtraRow],
    pub postamble: &'static str,
}

pub struct ExtraRow {
    pub signature: &'static str,
    pub description: &'static str,
}

pub const SECTIONS: &[Section] = &[
    Section {
        name: "MODEL",
        title: "MODEL (identity — always global):",
        preamble: "",
        extra_rows: &[ExtraRow {
            signature: "<NAME>",
            description:
                "Active model (alias or literal key). New keys need -m or --hf.",
        }],
        postamble: "",
    },
    Section {
        name: "SCOPE",
        title: "SCOPE (sticky; defaults to --global):",
        preamble: "",
        extra_rows: &[],
        postamble: "\n    Scope toggles may appear multiple times. Example:\n        \
                    lui --temp 0.6 --this --temp 0.2       # global=0.6, this model=0.2\n",
    },
    Section {
        name: "SETTINGS",
        title: "SETTINGS (scoped; pass `default` as the value to clear a per-scope override):",
        preamble: "",
        extra_rows: &[],
        postamble: "",
    },
    Section {
        name: "MACHINE",
        title: "MACHINE SETTINGS (always global; rejected with --this):",
        preamble: "",
        extra_rows: &[],
        postamble: "",
    },
    Section {
        name: "HARNESS",
        title: "HARNESS (external tools lui configures alongside llama-server):",
        preamble: "",
        extra_rows: &[],
        postamble: "",
    },
    Section {
        name: "REMOTE",
        title: "REMOTE (one-shot; not persisted):",
        preamble: "",
        extra_rows: &[],
        postamble: "",
    },
    Section {
        name: "OTHER",
        title: "OTHER:",
        preamble: "",
        extra_rows: &[],
        postamble: "",
    },
];

/// Immutable-after-build registry. Constructed once at startup via
/// `Registry::build()`. Holds every declared setting plus lookup tables
/// that the parser hits on each argv token.
pub struct Registry {
    settings: Vec<Setting>,
    by_name: HashMap<&'static str, usize>,
    by_long: HashMap<String, LongLookup>,
    by_short: HashMap<char, usize>,
}

/// A hit on a long flag: the index of the matching setting plus whether
/// the form was negated (`--no-<long>`). Negation is only produced for
/// settings where `has_no_form()` is true.
#[derive(Debug, Clone, Copy)]
pub struct LongLookup {
    pub index: usize,
    pub negated: bool,
}

impl Registry {
    /// Build the canonical registry. Calling this twice is cheap; production
    /// code should call it exactly once and share the result.
    pub fn build() -> Self {
        let mut reg = Registry {
            settings: Vec::new(),
            by_name: HashMap::new(),
            by_long: HashMap::new(),
            by_short: HashMap::new(),
        };
        declare_all_settings(&mut reg);
        reg.finalize();
        reg
    }

    /// Register a single setting. Called from `declare_all_settings`.
    /// Panics on duplicate name / flag to catch authoring mistakes at
    /// startup, since those would produce silent-and-subtle parse bugs.
    pub fn push(&mut self, s: Setting) {
        let idx = self.settings.len();
        if self.by_name.contains_key(s.name) {
            panic!("SettingsRegistry: duplicate setting name {:?}", s.name);
        }
        self.by_name.insert(s.name, idx);
        self.settings.push(s);
    }

    fn finalize(&mut self) {
        for (idx, s) in self.settings.iter().enumerate() {
            if let Some(c) = s.short {
                if let Some(&other) = self.by_short.get(&c) {
                    panic!(
                        "SettingsRegistry: duplicate short flag -{} ({} and {})",
                        c, self.settings[other].name, s.name
                    );
                }
                self.by_short.insert(c, idx);
            }
            let mut longs: Vec<&'static str> = Vec::new();
            if let Some(l) = s.long {
                longs.push(l);
            }
            longs.extend(s.long_aliases.iter().copied());
            for l in &longs {
                if self.by_long.contains_key(*l) {
                    panic!("SettingsRegistry: duplicate long flag --{}", l);
                }
                self.by_long.insert(
                    (*l).to_string(),
                    LongLookup {
                        index: idx,
                        negated: false,
                    },
                );
                if s.has_no_form() {
                    let neg = format!("no-{}", l);
                    if self.by_long.contains_key(&neg) {
                        panic!("SettingsRegistry: duplicate long flag --{}", neg);
                    }
                    self.by_long.insert(
                        neg,
                        LongLookup {
                            index: idx,
                            negated: true,
                        },
                    );
                }
            }
        }
    }

    pub fn settings(&self) -> &[Setting] {
        &self.settings
    }

    pub fn get(&self, name: &str) -> Option<&Setting> {
        self.by_name.get(name).map(|&i| &self.settings[i])
    }

    pub fn lookup_long(&self, flag: &str) -> Option<(LongLookup, &Setting)> {
        self.by_long
            .get(flag)
            .copied()
            .map(|l| (l, &self.settings[l.index]))
    }

    pub fn lookup_short(&self, c: char) -> Option<&Setting> {
        self.by_short.get(&c).map(|&i| &self.settings[i])
    }

    /// Iterate settings in the order they'll show up in `--help`.
    /// Within a section, the order matches declaration order.
    pub fn iter_by_section(&self) -> impl Iterator<Item = &Setting> {
        // Group into known sections, then trailing unknowns.
        let mut buckets: Vec<Vec<&Setting>> =
            (0..SECTIONS.len() + 1).map(|_| Vec::new()).collect();
        for s in &self.settings {
            let idx = SECTIONS
                .iter()
                .position(|sec| sec.name == s.section)
                .unwrap_or(SECTIONS.len());
            buckets[idx].push(s);
        }
        buckets.into_iter().flatten()
    }
}

/// The one place to declare lui's settings. Every item gets a single
/// adjacent block, ordered the way it should appear in `--help`.
/// Adding a new setting is a single block here; the parser, help
/// printer, TOML IO, llama-server passthrough, and UI payload all pick
/// it up by walking this registry.
pub fn declare_all_settings(reg: &mut Registry) {
    use PassthroughMode::*;
    use Scope::*;
    use ValueKind::*;

    // ---------- MODEL (identity) ----------
    //
    // CLI semantics:
    //   - Positional <NAME> sets the active model (alias-resolved first,
    //     then literal).
    //   - `-m` / `--hf` are zero-arg per-model `type` setters.
    //
    // The two type-flag settings here (`model_local_flag` /
    // `model_huggingface_flag`) carry only the help-text + flag shape; the
    // parser special-cases their side effect before the generic dispatch
    // runs, so their `passthrough`/`scope` fields are effectively dead.
    //
    // `active_model` is the stored last-used key. `model` / `hf_repo` are
    // load-only entries — they survive on disk for configs written by
    // older lui versions; the migrator renames them into `active_model`
    // on first load.
    reg.push(
        Setting::new("model_local_flag")
            .short('m')
            .long("model")
            .kind(Bool)
            .no_form(false)
            .scope(Ephemeral)
            .section("MODEL")
            .tag("type_flag_local")
            .help(&["Mark the active model's type = local (GGUF path)"]),
    );
    reg.push(
        Setting::new("model_huggingface_flag")
            .long("hf")
            .kind(Bool)
            .no_form(false)
            .scope(Ephemeral)
            .section("MODEL")
            .tag("type_flag_huggingface")
            .help(&["Mark the active model's type = huggingface"]),
    );
    reg.push(
        Setting::new("alias")
            .long("alias")
            .placeholder("NAME")
            .kind(String)
            .scope(Ephemeral)
            .section("MODEL")
            .help(&["Alias the active model as NAME"]),
    );
    reg.push(
        Setting::new("active_model")
            .kind(String)
            .scope(Global)
            .section("MODEL"),
    );

    // Per-model identity tag: `"huggingface"` or `"local"`. Populated by
    // the migrator on load and by the zero-arg `--hf` / `-m` flags.
    // Scope is `PerModel` so `[server].type = "..."` gets flagged at
    // load time; there's no CLI flag of its own.
    reg.push(
        Setting::new("type")
            .kind(String)
            .scope(PerModel)
            .passthrough(PassthroughMode::None)
            .section("MODEL"),
    );

    // ---------- SCOPE cursor ----------
    //
    // Not persisted; mutates the sticky scope cursor during parse. Kept
    // in the registry so `--help` stays generated from one source. The
    // parser special-cases their effect before the generic dispatch
    // runs.
    reg.push(
        Setting::new("global")
            .long("global")
            .kind(Bool)
            .scope(Ephemeral)
            .section("SCOPE")
            .no_form(false)
            .help(&["Subsequent settings update [server] (the global defaults)"]),
    );
    reg.push(
        Setting::new("this")
            .long("this")
            .long_aliases(&["local"])
            .kind(Bool)
            .scope(Ephemeral)
            .section("SCOPE")
            .no_form(false)
            .help(&[
                "Subsequent settings update [models.\"<active-model>\"] only",
            ]),
    );

    // ---------- SETTINGS ----------
    //
    // Declaration order doubles as UI order: `--help` lists these in this
    // order, and the UI's sampling / tuning rows iterate in the same
    // order. Adding a new setting is one `reg.push(...)` block in the
    // right spot — every downstream renderer picks it up.
    reg.push(
        Setting::new("ctx_size")
            .short('c')
            .long("ctx-size")
            .placeholder("N")
            .kind(Integer)
            .min(0)
            .scope(Both)
            .passthrough(FlagValue)
            .llama_flag("-c")
            .section("SETTINGS")
            // No `group` — ctx_size renders in its own "N token context
            // window" row, not the aggregated tuning line. Same for
            // gpu_layers.
            .ui_label("Context")
            .ui_format(super::setting::format_nonzero_int)
            .ui_unset("model default")
            .help(&["Context window (0 = model default)"]),
    );
    reg.push(
        Setting::new("gpu_layers")
            .long("ngl")
            .long_aliases(&["gpu-layers"])
            .placeholder("N")
            .kind(Integer)
            .min(-1)
            .scope(Both)
            .default(Value::Integer(-1))
            .passthrough(FlagValue)
            .llama_flag("-ngl")
            .section("SETTINGS")
            .ui_label("GPU layers")
            .ui_format(super::setting::format_negative_as_all)
            .help(&["GPU layers (-1 = all)"]),
    );
    reg.push(
        Setting::new("temp")
            .long("temp")
            .placeholder("F")
            .kind(Float)
            .scope(Both)
            .passthrough(FlagValue)
            .llama_flag("--temp")
            .section("SETTINGS")
            .group("sampling")
            .ui_label("Temperature")
            .ui_unset("model default")
            .help(&["Sampling temperature"]),
    );
    reg.push(
        Setting::new("top_p")
            .long("top-p")
            .placeholder("F")
            .kind(Float)
            .scope(Both)
            .passthrough(FlagValue)
            .llama_flag("--top-p")
            .section("SETTINGS")
            .group("sampling")
            .ui_label("Top-p")
            .ui_unset("model default")
            .help(&["Top-p (nucleus)"]),
    );
    reg.push(
        Setting::new("top_k")
            .long("top-k")
            .placeholder("N")
            .kind(Integer)
            .scope(Both)
            .passthrough(FlagValue)
            .llama_flag("--top-k")
            .section("SETTINGS")
            .group("sampling")
            .ui_label("Top-k")
            .ui_unset("model default")
            .help(&["Top-k"]),
    );
    reg.push(
        Setting::new("min_p")
            .long("min-p")
            .placeholder("F")
            .kind(Float)
            .scope(Both)
            .passthrough(FlagValue)
            .llama_flag("--min-p")
            .section("SETTINGS")
            .group("sampling")
            .ui_label("Min-p")
            .ui_unset("model default")
            .help(&["Min-p"]),
    );
    reg.push(
        Setting::new("parallel")
            .long("np")
            .long_aliases(&["parallel"])
            .placeholder("N")
            .kind(Integer)
            .min(1)
            .scope(Both)
            .default(Value::Integer(1))
            .passthrough(FlagValue)
            .llama_flag("-np")
            .section("SETTINGS")
            .group("tuning")
            .ui_label("np")
            .help(&["Server slots (llama-server -np)"]),
    );
    reg.push(
        Setting::new("ubatch_size")
            .long("ub")
            .long_aliases(&["ubatch-size"])
            .placeholder("N")
            .kind(Integer)
            .min(1)
            .scope(Both)
            .passthrough(FlagValue)
            .llama_flag("-ub")
            .section("SETTINGS")
            .group("tuning")
            .ui_label("ubatch")
            .ui_unset("server default")
            .help(&["Physical batch size (llama-server -ub)"]),
    );
    reg.push(
        Setting::new("cache_type_k")
            .long("ctk")
            .long_aliases(&["cache-type-k"])
            .placeholder("T")
            .kind(String)
            .scope(Both)
            .passthrough(FlagValue)
            .llama_flag("-ctk")
            .section("SETTINGS")
            .group("tuning")
            .ui_label("ctk")
            .ui_unset("server default")
            .help(&["KV cache key type (f16, q8_0, ...)"]),
    );
    reg.push(
        Setting::new("cache_type_v")
            .long("ctv")
            .long_aliases(&["cache-type-v"])
            .placeholder("T")
            .kind(String)
            .scope(Both)
            .passthrough(FlagValue)
            .llama_flag("-ctv")
            .section("SETTINGS")
            .group("tuning")
            .ui_label("ctv")
            .ui_unset("server default")
            .help(&["KV cache value type"]),
    );
    reg.push(
        Setting::new("batch_size")
            .long("batch-size")
            .placeholder("N")
            .kind(Integer)
            .min(1)
            .scope(Both)
            .passthrough(FlagValue)
            .llama_flag("-b")
            .section("SETTINGS")
            .group("tuning")
            .ui_label("batch")
            .ui_unset("server default")
            .help(&["Logical batch size (llama-server -b)"]),
    );
    reg.push(
        Setting::new("threads_batch")
            .long("tb")
            .long_aliases(&["threads-batch"])
            .placeholder("N")
            .kind(Integer)
            .min(1)
            .scope(Both)
            .passthrough(FlagValue)
            .llama_flag("-tb")
            .section("SETTINGS")
            .group("tuning")
            .ui_label("tb")
            .ui_unset("auto")
            .help(&["Prompt/batch threads (llama-server -tb)"]),
    );
    reg.push(
        Setting::new("cache_ram")
            .long("cache-ram")
            .placeholder("MIB")
            .kind(Integer)
            .min(0)
            .scope(Both)
            .passthrough(FlagValue)
            .llama_flag("--cache-ram")
            .section("SETTINGS")
            .group("tuning")
            .ui_label("cache-ram")
            .ui_format(super::setting::format_mib)
            .ui_unset("server default")
            .help(&["Host-memory prompt cache (llama-server --cache-ram)"]),
    );
    reg.push(
        Setting::new("prio_batch")
            .long("prio-batch")
            .placeholder("0-3")
            .kind(Integer)
            .min(0)
            .max(3)
            .scope(Both)
            .passthrough(FlagValue)
            .llama_flag("--prio-batch")
            .section("SETTINGS")
            .group("tuning")
            .ui_label("prio-batch")
            .ui_unset("normal")
            .help(&["Batch thread priority"]),
    );
    reg.push(
        Setting::new("fit_target")
            .long("fit-target")
            .long_aliases(&["fitt"])
            .placeholder("MiB")
            .kind(String) // comma-separated per-device list — keep as String
            .scope(Both)
            .passthrough(FlagValue)
            .llama_flag("--fit-target")
            .section("SETTINGS")
            .group("tuning")
            .ui_label("fit-target")
            .help(&[
                "Free-VRAM headroom reserved by llama-server --fit",
                "(default 1024). Accepts a single value or comma-",
                "separated per-device list (e.g. 2048,512,512).",
            ]),
    );
    reg.push(
        Setting::new("swa_full")
            .long("swa-full")
            .kind(Bool)
            .scope(Both)
            .passthrough(BoolFlagIfTrue)
            .llama_flag("--swa-full")
            .section("SETTINGS")
            .group("tuning")
            .ui_label("swa-full")
            .ui_format(super::setting::format_bare_or_off)
            .ui_unset("auto (SWA/hybrid detection at launch)")
            .help(&[
                "Force --swa-full on",
                "Force --swa-full off (disables auto-detect)",
            ]),
    );
    // Chat template kwargs: no CLI flag, loaded from TOML only. Emitted
    // as a JSON object to llama-server. Last-wins across global/per-model
    // (Effective's standard precedence) — users write the full map at
    // whichever scope they want to override.
    reg.push(
        Setting::new("chat_template_kwargs")
            .kind(Map)
            .scope(Both)
            .passthrough(FlagValue)
            .llama_flag("--chat-template-kwargs")
            .section("SETTINGS"),
    );
    // Extra args: post-`--` passthrough. Append semantics at resolve
    // time (global entries + active model's entries, concatenated). The
    // `ui_label` doubles as the "+N _label_" phrase on the tuning line.
    reg.push(
        Setting::new("extra_args")
            .kind(StringArray)
            .scope(Both)
            .passthrough(LiteralTokens)
            .section("SETTINGS")
            .group("tuning")
            .ui_label("extra")
            .ui_format(super::setting::format_count_aggregate),
    );

    // ---------- MACHINE (global-only; rejected under --this) ----------
    reg.push(
        Setting::new("port")
            .long("port")
            .placeholder("N")
            .kind(Integer)
            .min(0)
            .max(65535)
            .scope(Global)
            .default(Value::Integer(8080))
            .section("MACHINE")
            .help(&["Server port (default 8080)"]),
    );
    reg.push(
        Setting::new("host")
            .kind(String)
            .scope(Global)
            .default(Value::String("127.0.0.1".to_string()))
            .section("MACHINE"),
    );
    reg.push(
        Setting::new("public")
            .long("public")
            .kind(Bool)
            .scope(Ephemeral)
            .no_form(false)
            .section("MACHINE")
            .help(&["Bind 0.0.0.0 instead of 127.0.0.1"]),
    );
    reg.push(
        // Positive-form bool: `websearch = true` (the default) enables
        // the endpoint; `websearch = false` disables it. `--websearch` /
        // `--no-websearch` flip it on the CLI in the usual way.
        Setting::new("websearch")
            .long("websearch")
            .kind(Bool)
            .scope(Global)
            .default(Value::Bool(true))
            .section("MACHINE")
            .help(&[
                "Enable lui's local web-search endpoint",
                "Disable and remove its opencode skill",
            ]),
    );
    reg.push(
        Setting::new("web_port")
            .long("web-port")
            .placeholder("N")
            .kind(Integer)
            .min(0)
            .max(65535)
            .scope(Global)
            .section("MACHINE")
            .help(&[
                "Port for the local web-search endpoint (default: llama port + 1)",
            ]),
    );
    reg.push(
        Setting::new("allow_vram_oversubscription")
            .long("avo")
            .kind(Bool)
            .scope(Global)
            .default(Value::Bool(false))
            .section("MACHINE")
            .ui_label("Allow VRAM Oversubscription")
            .help(&[
                "Allow VRAM oversubscription (skip lui's abort on GPU over-budget)",
                "Abort on VRAM oversubscription (default)",
            ]),
    );
    // ---------- HARNESS (one bool per declared harness) ----------
    //
    // Walks `crate::harness::HARNESSES` so adding a new harness module
    // automatically adds its `--harness-<name>` / `--no-harness-<name>`
    // flag, persists a TOML entry on override, and surfaces a row in the
    // HARNESS help section.
    for h in crate::harness::HARNESSES {
        reg.push(
            Setting::new(h.setting_name)
                .long(h.flag_long)
                .kind(Bool)
                .scope(Global)
                .default(Value::Bool(h.default_on))
                .section("HARNESS")
                .help(h.help),
        );
    }
    reg.push(
        Setting::new("harness_opencode_disable_prune")
            .long("harness-opencode-disable-prune")
            .kind(Bool)
            .scope(Global)
            .default(Value::Bool(false))
            .no_form(false)
            .section("HARNESS")
            .ui_label("Opencode: Disable Compaction Pruning")
            .help(&[
                "Tell opencode not to prune tool outputs",
                "(preserves llama-server prompt cache on",
                " large contexts; off by default so cloud",
                " providers retain opencode's normal pruning)",
            ]),
    );
    // ---------- REMOTE (one-shot; not persisted) ----------
    reg.push(
        Setting::new("ssh")
            .long("ssh")
            .placeholder("USER@HOST")
            .kind(String)
            .scope(Ephemeral)
            .section("REMOTE")
            .help(&[
                "Run on a server. Configures that",
                "remote's opencode to reach this machine's",
                "llama-server over a reverse tunnel, prints",
                "the matching `ssh -R ...` command, and exits.",
            ]),
    );
    reg.push(
        Setting::new("remote")
            .long("remote")
            .placeholder("HOST[:PORT]")
            .kind(String)
            .scope(Ephemeral)
            .section("REMOTE")
            .help(&[
                "Run on a client. Fetches /config from a",
                "--public server over plain HTTP, writes",
                "local opencode.json pointed directly at that",
                "server's llama-server, and stands up a local",
                "bsearch server so the browser opens on this",
                "machine. Blocks until Ctrl-C. PORT is the",
                "server's HTTP port; defaults to 8081.",
            ]),
    );

    // ---------- OTHER (mode flags) ----------
    reg.push(
        Setting::new("list")
            .short('l')
            .long("list")
            .kind(Bool)
            .scope(Ephemeral)
            .no_form(false)
            .section("OTHER")
            .help(&["List cached models and show current config"]),
    );
    reg.push(
        Setting::new("cmd")
            .long("cmd")
            .kind(Bool)
            .scope(Ephemeral)
            .no_form(false)
            .section("OTHER")
            .help(&["Print the resolved llama-server command and exit"]),
    );
    reg.push(
        Setting::new("debug")
            .long("debug")
            .placeholder("PATH")
            .kind(String)
            .scope(Ephemeral)
            .section("OTHER")
            .help(&["Dump raw llama-server output to a file"]),
    );
    reg.push(
        Setting::new("help")
            .short('h')
            .long("help")
            .kind(Bool)
            .scope(Ephemeral)
            .no_form(false)
            .section("OTHER")
            .help(&["Show this help"]),
    );
}
