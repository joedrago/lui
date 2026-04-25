# AGENTS.md

A field guide for agents landing in this repo cold. Covers the high-level
architecture, the settings registry (the one file you most often edit),
the networked modes, the harness system, and the handful of
commands that make "did my change plumb through?" a 2-second check.

## What lui is

`lui` is a TUI wrapper around llama.cpp's `llama-server`. It owns one
child `llama-server` process, renders a live status panel by parsing
that process's stdout, and configures external tools (opencode, pi)
via a **harness** layer so they know where to find the model. Runtime
config lives in `~/.config/lui.toml` (XDG-style; the same path on
macOS). Every CLI flag is persisted, so plain `lui` on subsequent runs
resumes the last session.

## Source map

```
src/
  main.rs              argv parse, top-level orchestration, --list / --cmd / --ssh / --remote dispatch
  config.rs            load_config / save_config, ~/.config/lui.toml IO, model_key / derive_model_name
  server.rs            build_args (llama-server argv), spawn_server, log parser, ServerState, UiSnapshot wire format
  display.rs           TUI renderer; polls /data and draws
  websearch.rs         axum HTTP server: /bsearch (browser-mediated search), /config, /data, /setup (bookmarklet)
  ssh_tunnel.rs        --ssh and --remote flows; HTTP GET to /config; setup_share / setup_use
  gguf.rs              minimal GGUF metadata reader (for SWA auto-detect)
  harness/
    mod.rs             Harness trait-table, HARNESSES slice, apply_local / apply_remote, render_websearch_skill
    opencode.rs        writes ~/.config/opencode/opencode.{jsonc,json}
    pi.rs              writes ~/.pi/agent/models.json
  settings/
    mod.rs             re-exports
    setting.rs         Setting struct + fluent builders + Scope / PassthroughMode / ValueKind; UI formatters
    value.rs           Value enum (Bool/Integer/Float/String/StringArray/Map) + toml <-> value bridging
    store.rs           Store, Config (the runtime config type), Effective (3-layer resolver)
    registry.rs        *** declare_all_settings — the one place all settings live ***
    help.rs            auto-generated --help (walks the registry)
    migrate.rs         one-shot TOML migrators run at load time
```

## The settings registry (you will edit this a lot)

Every CLI flag and persisted config key is declared exactly once in
`src/settings/registry.rs::declare_all_settings`. All downstream
code — `--help`, argv parser, TOML load/save, `llama-server` argv
builder, UI payload for the TUI — walks the registry by name. There
are no typed config structs; the registry *is* the schema.

A setting is constructed via `Setting::new("name")` and fluent
builders. Adjacent-call style is the house rule (see memory note on
builder style) — ten short lines per entry that you can scan and
grep:

```rust
reg.push(
    Setting::new("ctx_size")                          // canonical name; matches TOML key + Store HashMap key
        .short('c')                                   // -c on the CLI
        .long("ctx-size")                             // --ctx-size
        .placeholder("N")                             // shown in --help as `--ctx-size <N>`
        .kind(Integer)                                // ValueKind — how the token is parsed
        .min(0)                                       // parse-time clamp
        .scope(Both)                                  // Global / PerModel / Both / Ephemeral
        .passthrough(FlagValue)                       // how it reaches llama-server
        .llama_flag("-c")                             // the flag llama-server actually wants
        .section("SETTINGS")                          // --help grouping
        .ui_label("Context")                          // TUI label
        .ui_format(format_nonzero_int)                // custom value renderer (optional)
        .ui_unset("model default")                    // phrase when unset
        .help(&["Context window (0 = model default)"])
);
```

### What every field means

All fields live on `Setting` in `src/settings/setting.rs`.

**Identity — what the setting is called:**
- `name` *(required, `Setting::new`)*: canonical snake_case key. Same
  string is used for the TOML key, the HashMap key in `Store`, the
  registry lookup, and error messages. Must be unique.
- `short(c)`: single-char flag (`-c`). Omit for long-only flags.
- `long(s)`: primary long flag body, without the leading `--`
  (`"ctx-size"`, not `"--ctx-size"`). Omit for purely storage-only
  settings (e.g. `active_model`).
- `long_aliases(&[...])`: extra long forms resolved to this same
  setting (`"np"` primary + `["parallel"]` alias). Registry panics
  at build time on duplicate flags.
- `placeholder(s)`: the bit that shows up in `--help` after the flag
  (`--temp <F>`). Empty for zero-arg bool flags.

**How it's parsed:**
- `kind(ValueKind)`: `Bool` / `Integer` / `Float` / `String` /
  `StringArray` / `Map`. Drives argv parsing and TOML load/save. The
  setter flips `no_form` off for non-bools automatically.
- `min(n)`, `max(n)`: parse-time clamp for `Integer`. Ignored for
  other kinds. Enforced on CLI input *and* TOML load.
- `no_form(bool)`: for `Bool`, whether `--no-<long>` is accepted as
  the inverse. Defaults to true; flip off for mode flags like
  `--list` where "no" makes no sense.
- `tag(s)`: free-form discriminator string. Today it only carries
  `"type_flag_local"` / `"type_flag_huggingface"` so the parser can
  spot the zero-arg `-m` / `--hf` flags. Prefer real struct fields
  over new tags when you can.

**Where it's stored:**
- `scope(Scope)`:
  - `Global` — lives in `[server]` only. `--this --port 8080` is a
    hard error.
  - `PerModel` — lives in `[models."<key>"]` only. No CLI flag
    lands here today except the per-model `type` tag.
  - `Both` — honors the sticky `--this` / `--global` cursor. Writes
    to the chosen scope; effective value is per-model over global.
  - `Ephemeral` — not persisted. Mode flags (`--list`, `--cmd`,
    `--ssh`, `--alias`, `--public`) use this. Parse loop reads side
    effects directly in `main.rs`, not via `handle_flag`.
- `default(Value)`: registry-level default, used when neither
  store has a value. **Absence of a default means "truly unset"** —
  no flag reaches `llama-server` and the UI falls back to
  `ui_unset`. Settings that must always reach llama-server
  (e.g. `gpu_layers = -1`, `parallel = 1`, `chat_template_kwargs`)
  declare one.

**How it reaches llama-server:**
- `llama_flag(s)`: the flag llama-server itself takes (`"-c"`,
  `"--temp"`, `"--fit-target"`). Our name and llama-server's name
  often differ.
- `passthrough(PassthroughMode)`:
  - `None` — lui handles this itself (mode flags, identity keys).
  - `FlagValue` — emit `<llama_flag> <value>` if the effective
    value is present. The common case.
  - `BoolFlagIfTrue` — emit the bare `<llama_flag>` iff value is
    `true`. Used for `--swa-full` where mere presence flips it on.
  - `LiteralTokens` — append the `StringArray` contents literally,
    no flag prefix. Used by post-`--` extra args.

**How it shows up in `--help`:**
- `section(s)`: one of `"MODEL" | "SCOPE" | "SETTINGS" | "MACHINE" |
  "HARNESS" | "REMOTE" | "OTHER"` (see `SECTIONS` at the top of
  `registry.rs`). Drives ordering and headers. Anything unmatched
  bucketed to the end.
- `help(&[...])`: one `&str` per line. For bool settings with a
  `--no-` pair, `help[0]` describes the positive form and `help[1]`
  the negated form — they get rendered as two rows. For everything
  else `help[0]` is the summary and `help[1..]` is continuation.

**How it shows up in the TUI / `--list`:**
- `group(s)`: `"sampling"` / `"tuning"` / `None`. The TUI filters
  `UiSnapshot.settings` by this and concatenates the label=value
  pairs into one line per group. `None` means "don't surface in the
  grouped sections" (either hidden or rendered in its own typed
  block).
- `ui_label(s)`: human label. If omitted, derived from the long
  flag with `-` → space and first-letter uppercased.
- `ui_format(fn)`: custom formatter with signature
  `fn(Option<&Value>, &Effective, &Setting) -> Option<String>`.
  Return values:
  - `None` → skip this row in grouped contexts; fall back to
    `ui_unset` in single-row contexts.
  - `Some("")` → render the label only, no `=value` suffix
    (bare-flag rendering, used by `swa-full`).
  - `Some(s)` → render `label=s`.
  The stock formatters `format_mib`, `format_negative_as_all`,
  `format_nonzero_int`, `format_bare_or_off`,
  `format_count_aggregate` cover the typical cases.
- `ui_unset(s)`: phrase shown in the "Current config" view when
  nothing is explicitly set *and* the formatter didn't override.
  Typical values: `"model default"`, `"server default"`, `"auto"`,
  `"normal"`.

### Adding a new setting — the five-minute checklist

1. Add one `reg.push(Setting::new(...)...)` block in
   `declare_all_settings` under the section where it belongs (the
   section header comment in `registry.rs` groups them visually).
2. Pick a `kind`. Integer/Float gets `min`/`max` if there's a real
   clamp. Bool gets a thoughtful `no_form` (almost always keep
   default true).
3. Pick a `scope`. If you're in doubt, `Both` is the right default
   for tuning knobs.
4. If llama-server should receive it, set `llama_flag` and
   `passthrough`. If the default behavior of llama-server is fine
   when the user hasn't touched the flag, **don't** set a `default`
   — that's what keeps `--temp` off the argv when unset.
5. Fill in `section`, `help`, and (if grouped in the TUI)
   `group` + `ui_label` + (often) `ui_unset`.
6. `cargo build` / `cargo clippy --all-targets`. Registry build is a
   startup check, so duplicate names / flags panic immediately.
7. Smoke-test (see **Testing settings plumbing** below).

No other file needs editing: `--help` autogenerates, argv parser
dispatches through `Registry::lookup_long` / `lookup_short`, TOML load
goes through `Store::from_toml_table` which reads the registry for
kinds, TOML save goes through `Store::to_toml_table`, and llama-server
args come from the `build_args` loop that walks every setting's
`passthrough` mode.

### Config migrations

`src/settings/migrate.rs` runs one-shot steps against a raw
`toml::Value::Table` at load time (before any typed parsing).
Idempotent; each step flips `did_migrate` to true only if it mutated
something. On first migration, `load_config` drops
`<path>.pre-migration.bak` next to the TOML and rewrites in the new
canonical shape. Add a new migrator by writing a function that takes
`&mut toml::value::Table` and ORing it into `migrate()`. Use when you
rename a stored key or change its sense (see `flip_websearch_sense`,
`rename_model_identity_to_active_model`).

## The three "sharing" modes — websearch, ssh, remote

These are *three different things*. Easy to confuse on a fresh read.

### 1. Websearch (`src/websearch.rs`)

Always-on local HTTP server mounted by every `lui` run (unless
`--no-websearch`). It serves `/data`, `/config`, `/setup`, and
`/bsearch`:

- `/data` — the `UiSnapshot` JSON the TUI polls on 250ms ticks. A
  `--remote` client polls this on the server.
- `/config` — a tiny `LuiConfigResponse` blob describing the server
  (llama port, web port, model name, ctx size, version). Used by
  `--remote` to discover the server's shape.
- `/setup` — HTML page hosting the draggable `lui-grab` bookmarklet.
- `/bsearch?q=...` — the interesting one. Browser-mediated web
  search. When opencode's skill calls this endpoint, lui opens a
  Google tab in the user's *real* browser (with their real cookies
  and rendered JS). The user clicks the bookmarklet on that tab,
  which POSTs the DOM-scraped results to `/results?id=<id>`, which
  unblocks the original `/bsearch` request. Every keyless Google
  scraper now captchas; driving a real browser session is the only
  thing that still works reliably. Up to 120s wait; 504 on timeout.

The companion `lui-web-search` SKILL.md (rendered by
`harness::render_websearch_skill`) tells opencode how to call
`/bsearch`. The port defaults to `llama_port + 1` (so 8081 with an
8080 llama); override with `--web-port`.

### 2. `--ssh USER@HOST` (`src/ssh_tunnel.rs::setup_share`)

Run **on the server** (the machine with `llama-server`). Configures a
*client* over SSH to reach your llama-server via a reverse tunnel, and
prints the `ssh -R ...` command you then run in another terminal to
establish the tunnel. The client's opencode talks to `localhost:<port>`
which is the tunnel back to your machine.

Flow:
1. Pick two random client-side ports in `[18000, 29000)` — one for
   llama-server, one for websearch (`base+1`). Random to avoid
   collisions with whatever else the client is running.
2. Preflight the client over SSH — check `opencode` is installed
   (tries `command -v opencode`, `bash -lc 'command -v opencode'`,
   and the installer's default `~/.opencode/bin/opencode`).
3. For every enabled harness, apply remotely via SSH: probe for an
   existing config file, drop a `.luibackup` if it's a first touch,
   rewrite the config via `cat >` over SSH, write the `lui-web-search`
   SKILL.md (unless `--no-websearch`).
4. Print the `ssh -R <remote_llama>:localhost:<local_llama>
   -R <remote_web>:localhost:<local_web> USER@HOST` command and exit.

SSH is load-bearing: clients typically can't reach back to the server
over the network (NAT, no public address), so a reverse tunnel is the
whole point. Not persisted to `lui.toml`.

### 3. `--remote HOST[:PORT]` (`src/ssh_tunnel.rs::setup_use`)

Run **on a client** (your laptop). The *server* must already be
running with `--public` (binds `0.0.0.0` instead of `127.0.0.1`). This
machine doesn't run llama-server — it configures itself to point
directly at the server over HTTP, but spins up a *local* bsearch
server so browser-mediated search still works here (where the real
browser is).

Flow:
1. Plain HTTP `GET /config` against the server (default port 8081).
   Version check — refuses a mismatched config version.
2. Write local opencode config with `baseURL` pointing at the
   server's llama-server directly (`http://<host>:<llama_port>/v1`).
3. Write local `lui-web-search` SKILL.md pointed at the local
   bsearch server (port 8081 by convention; stable so the user only
   drags the bookmarklet once).
4. Stand up the in-process bsearch HTTP server on the client.
5. Run the TUI, which polls the server's `/data` to render.
6. Block on Ctrl-C.

No SSH involved. Not persisted.

`--ssh` and `--remote` are mutually exclusive and guarded at parse
time.

## The harness system (`src/harness/`)

A *harness* is any external tool lui configures so it knows how to
reach llama-server. Each harness declares one `pub const HARNESS:
Harness` and is listed in `HARNESSES`. Adding one is a new file under
`src/harness/` plus one entry in that slice — the registry
auto-declares a `harness_<name>` bool with default-on / default-off
from the harness's `default_on` field, and both local-apply and
SSH-apply loops iterate `HARNESSES`.

Each harness ships:
- `ConfigFile { dir, candidates }` — home-relative directory and
  ordered filename candidates (first existing wins; first declared
  is the fresh-install default).
- `apply(root_obj, eff, inputs)` — CST-surgical edits against the
  parsed JSONC root. Use the builder helpers `s`, `b`, `n`, `arr`,
  `obj` from `harness/mod.rs`.
- `needs_backup(existing)` — returns true iff the file has content
  we're about to change but no prior lui block (drops a
  `.luibackup`).
- Optional `preflight_ssh(target)` — verify the tool is installed on
  a remote before we start writing config over SSH. Opencode uses
  this to check for its own binary in three plausible PATH locations.

Shared machinery in `harness/mod.rs`:
- `apply_local(harness, eff, inputs)` — parse existing JSONC, call
  `apply`, atomic rename, write the SKILL.md (or delete it when
  websearch is off).
- `apply_remote(harness, target, port, inputs, eff)` — same shape
  over SSH via `ssh_run`.
- `render_websearch_skill(port)` — the lui-web-search SKILL.md body,
  shared across harnesses since the skill itself doesn't care which
  tool loads it.

`jsonc_parser::cst` is used instead of raw JSON so user comments and
formatting round-trip through our edits.

Current harnesses: `opencode` (on by default), `pi` (off by default).

## The "always-on" llama-server args

Not every flag lui sends to llama-server is registry-driven. `lui`
has opinions it enforces on every run, regardless of user config.
These live as hard-coded `args.push(...)` lines at the top of
`server::build_args` (`src/server.rs`, around L619–635):

```rust
args.push("--host".to_string());  args.push(host);
args.push("--port".to_string());  args.push(port.to_string());
args.push("--metrics".to_string());
args.push("--jinja".to_string());
args.push("--log-colors".to_string());  args.push("off".to_string());
args.push("-v".to_string());
args.push("-fa".to_string());  args.push("on".to_string());
args.push("--cache-reuse".to_string());  args.push("256".to_string());
args.push("-kvu".to_string());
```

These are lui policy, not user settings:
- `--host` / `--port` come from the registry (bound from
  `host`/`port`), but the flags themselves are emitted unconditionally
  so llama-server never falls back to its own defaults.
- `--metrics` — Prometheus-style endpoint on.
- `--jinja` — chat templates in jinja form (required for
  `--chat-template-kwargs` to do anything).
- `--log-colors off` + `-v` — verbose + plain ANSI so our stdout
  parser isn't fighting color codes. Matches `strip_ansi` in the log
  reader as belt-and-suspenders.
- `-fa on` — flash attention always on.
- `--cache-reuse 256` + `-kvu` — prompt-cache reuse tuned for
  agentic workloads (opencode re-sends similar prompts turn-to-turn).

**When to add to this list vs. the registry:** if the flag is a
universal lui opinion that no user should ever disable, hard-code it
here. If there's any scenario where someone would want to flip it,
declare a registry setting instead — then the user can scope it per
model, clear it with `default`, or toggle it with `--no-`. The `-fa`
flag in particular has been discussed and is currently always-on by
design.

Everything after this block in `build_args` is registry-driven: it
walks `eff.registry.settings()` once and dispatches on `passthrough`.
So a new setting with `passthrough(FlagValue)` + `llama_flag("--foo")`
gets emitted automatically — no edit to `build_args` needed.

## Testing settings plumbing — the two-minute loop

Most "did my change work?" questions are answered by `--cmd` (prints
the fully-resolved `llama-server` command line and exits) and
`--list` / `-l` (prints the current config with everything resolved).
Neither of these actually launches anything.

### Build and format

```
cargo build                      # dev build, ~1s incremental
cargo build --release            # release, ~30s cold
cargo clippy --all-targets       # 6 pre-existing warnings, none in src/settings
cargo fmt                        # standard rustfmt
```

There are **no tests in this codebase** (there's a tiny inline
`#[cfg(test)] mod tests` in `server.rs` but no broader harness).
Verification is `cargo build` + `cargo clippy` + manual runs of the
two commands below.

### Quick-check loop

```
# Print the llama-server command that *would* be launched. No side effects beyond
# possibly saving config, so back up first (see below).
./target/debug/lui --cmd

# Pretty-print current config with scope overrides, aliases, and per-model blocks.
./target/debug/lui --list
./target/debug/lui -l
```

Both commands run the full parse → load → Effective-resolve pipeline,
so they exercise registry declarations, TOML IO, scope merging, UI
formatters, and (for `--cmd`) `build_args`. If a new setting doesn't
show up in `--cmd`'s output, its `passthrough` is probably `None` or
its `llama_flag` is missing. If it doesn't show up in `--list`,
it's probably missing `ui_label` / `ui_format` / `ui_unset`, or its
`group` is `None` and you're expecting it in a grouped section.

### ⚠ Restore `~/.config/lui.toml` after weird test arguments

**Every CLI flag is persisted.** That's the feature — plain `lui`
resumes last time. It's also the footgun for testing: running
`lui --hf fake/model --temp 999 --this -c 1` writes that garbage
straight into `~/.config/lui.toml`, clobbering whatever real
configuration the user had.

The standard dance when experimenting:

```
# 1. Back up (the config path is ~/.config/lui.toml on all platforms —
#    not "tui.toml", that's a misremembering).
cp ~/.config/lui.toml ~/.config/lui.toml.bak

# 2. Run any weird test invocations.
./target/debug/lui --hf fake/test:Q4 --alias fakealias --this -c 1234 --cmd
./target/debug/lui --ngl 99 --temp 0.1 --this --temp default -l

# 3. Restore before you forget.
mv ~/.config/lui.toml.bak ~/.config/lui.toml
```

`--cmd` and `--list` themselves only *read* config, but the
preceding parse has already written to it by the time they fire —
mode flags don't skip the save path. Keep the backup tight.

For truly hermetic testing you can point lui at a fake home:

```
HOME=/tmp/luihome mkdir -p /tmp/luihome/.config
HOME=/tmp/luihome ./target/debug/lui --hf fake/test:Q4 --cmd
```

Everything — `~/.config/lui.toml`, `~/.config/opencode/opencode.json`,
`~/.pi/agent/models.json`, HuggingFace cache probe — re-roots under
`$HOME`, so nothing in your real home gets touched.

### Running with --cmd against a specific config

`--cmd` emits a one-line `llama-server ...` command with every flag
fully resolved (SWA auto-detect, scope merging, defaults, the
always-on args, post-`--` extras). It's shell-quoted for direct
copy-paste. Handy for comparing two config states:

```
./target/debug/lui --cmd > /tmp/before.txt
./target/debug/lui --this --temp 0.2 --cmd > /tmp/after.txt
diff /tmp/before.txt /tmp/after.txt   # shows just `--temp 0.2` added
```

(Remember to restore `~/.config/lui.toml` — those two runs modified it.)

## Quirks worth knowing

- **`Config` the type vs `config` the module.** `settings::store::Config`
  is the runtime container (global store + per-model map + aliases).
  `config.rs` is a module of free functions. Rust disambiguates by
  syntax position; use `config` as a variable name everywhere.
- **No typed config structs.** All former per-field structs were
  deleted in the final phase of the SettingsRegistry refactor. Values
  are `Value` variants keyed by setting name; access is
  `eff.get_i64("port")` / `eff.get_string("active_model")` /
  `eff.get_bool("websearch")`.
- **Precedence.** `Effective::get` walks per-model store → global
  store → registry `default`. `None` at all three means "truly unset"
  — downstream code must treat it as "skip the flag / row".
- **StringArray append semantics.** `extra_args` (post-`--`
  passthrough) concatenates global + per-model at resolve time, not
  "last layer wins". Use `eff.merged_string_array(name)` for these.
- **Bare `Value`.** Only `Map` round-trips as a raw
  `toml::Value` (for `chat_template_kwargs`), so nested numeric /
  bool entries don't get lossily stringified. Everything else is a
  plain scalar / `Vec<String>`.
- **Alias pool is unified.** A single `[aliases]` flat table; HF wins
  on name collisions (with a warning once per load).
- **`allow_vram_oversubscription` (`--avo`).** Read in
  `spawn_server` and stored on `ServerState` so the VRAM-budget check
  in the log parser can consult it without plumbing an `Effective`
  into the parser thread.
- **`--chat-template-kwargs`.** Not CLI-reachable; TOML-only. Default
  `{ preserve_thinking = true }`. Serialized as a JSON object on
  llama-server's argv.
