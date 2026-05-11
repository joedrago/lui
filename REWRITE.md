# lui rewrite — plan

A clean-slate ESModule Node rewrite of lui. No compile step. Multi-engine.
Opaque arg passthrough. Same wire format and TUI feel as today's Rust impl,
just lighter and easier to extend.

The existing Rust source is preserved under `old/` for reference.

---

## 1. Goals

- **Simpler to read and extend** than the Rust codebase. New engines and
  harnesses should be a single small file with no central registry to teach.
- **Multi-engine from day one**, with `llama-server` as the only one shipped.
- **Opaque per-model `args[]`**: lui stores the raw argv tokens and never
  parses them. Engines opt to inspect for their own purposes (reject
  conflicts, inject required prefix, color-code segments).
- **Stable wire format**: `/data` payload is engine-agnostic "panels of lines
  and bars", so we can add engines forever without touching the renderer or
  the `--remote` contract.
- **No compile step**: pure ESM JavaScript on Node. Tiny dep tree.
- **Subcommand CLI** with one ergonomic exception: bare `lui` resumes.
- **Keep what's already great**: live stdout parsing into progress bars,
  snapshot/render split, websearch via real browser, harness autoconfig,
  `--ssh` reverse-tunnel client setup, `--remote` client, `--debug`/`--cmd`
  debugging.

---

## 2. What we keep from old lui

| feature                              | notes                                                  |
| ------------------------------------ | ------------------------------------------------------ |
| llama-server child process lifecycle | spawn, stdout parse, panel emit                        |
| TUI rendering separated from state   | TUI polls `/data` whether local or remote              |
| `/data` HTTP endpoint                | same idea, generalized to panels                       |
| websearch with bookmarklet           | `/bsearch`, `/results`, `/setup`                       |
| `lui ssh USER@HOST`                  | configure a remote client over ssh, reverse tunnel     |
| `lui remote HOST[:PORT]`             | connect TUI to a remote lui server                     |
| Harness autoconfig                   | opencode (on by default), pi (off by default)          |
| `lui-web-search` SKILL.md            | rendered by lui, dropped into each harness's directory |
| `--cmd` / `--list` semantics         | become `lui cmd` and `lui show` / `lui ls`             |
| `--debug PATH` raw engine log        | stays as a runtime-only flag on `lui run`              |
| TOML config at `~/.config/lui.toml`  | header renamed `[server]` → `[global]`                 |

---

## 3. What we cut or defer

These are all easy to add later without touching the wire format or the
engine contract:

- **GGUF SWA auto-detect.** llama-server handles its own defaults; if you
  want SWA you add `--swa-full` to your args. When/if we put SWA
  auto-detect back, it lives entirely inside the llama-server engine's
  `buildArgv` — it's an engine-internal concern, not a lui-level
  feature, so no top-level surface area changes.
- **`--hf` cache resolution / probing.** Pass the HF URL straight to the
  engine and let it resolve. lui no longer reads `~/.cache/huggingface`.
- **`--sandbox` / nono integration.** Run nono yourself, or we add it back
  later as `lui sandbox HARNESS`.
- **Separate alias table.** The model's _name_ in `[models.NAME]` is its
  alias. No second mapping.
- **Typed setting registry.** Replaced by a tiny `[global]` table for global
  knobs (engine_port, web_port, websearch, per-harness sub-tables) and
  opaque `args[]` per model.

---

## 4. Repo layout

```
package.json            // "type": "module"; deps: smol-toml, jsonc-parser
eslint.config.mjs
bin/lui                 // #!/usr/bin/env node → ../src/main.js
src/
  main.js               // argv parse, build a Lui, dispatch the subcommand
  lui.js                // the Lui class: owns engine + web + display, holds lui-state
  config.js             // load/save ~/.config/lui.toml (smol-toml)
  ansi.js               // colors, cursor, wrap, bar render (~80 lines)
  display.js            // TUI loop, polls /data, paints panels
  web.js                // raw node:http; /data /config /setup /bsearch /results
  ssh.js                // `lui ssh` (setup_share) and `lui remote` (setup_use)
  wire.js               // View builder for engines & lui to emit panels
  engine.js             // child-process runner + engines registry + STYLE_SEGMENT_* constants
  bookmarklet.html
  engine/
    llama-server.js     // llama.cpp engine; binary is `llama-server`
  harness/
    index.js            // re-exports concrete harnesses
    opencode.js
    pi.js
old/                    // original Rust, kept for reference
```

### Porting pointers

Two files do most of the genuinely-tricky behavior in the Rust impl —
the implementer should port from there rather than reinventing:

- `engine/llama-server.js` ⟵ `old/src/server.rs`
  Realtime stdout parser, phase detection, ANSI stripping, progress
  bars (download / prefill / generate), VRAM/RAM budget, version + git
  hash, update-available banner, crash detection, request counters.
  Every `ServerState` field becomes a key on `lui.state.*`; every
  rendered row in `old/src/display.rs` becomes a Line emitted by
  `appendPanels`.
- `web.js` ⟵ `old/src/websearch.rs`
  Endpoint semantics for `/data`, `/config`, `/setup`, `/bsearch`,
  `/results`; the bookmarklet-driven browser-mediated search flow
  (open Google tab → bookmarklet POSTs scraped DOM → unblock the
  waiting `/bsearch` request); the 120s wait, 504 on timeout. Plus
  the `lui-web-search` SKILL.md text harnesses drop into their config
  dirs.

Other files (`config.js`, `ansi.js`, `display.js`, `ssh.js`,
harness modules) are smaller and the wire-format / builder API spec
in this doc is sufficient.

**How to use the Rust as a reference, not a template.** Port the
_behavior_ — what gets parsed, what state gets updated, what shows
up on screen, what the HTTP endpoints do — not the structure. The
Rust code is shaped by Rust's constraints (lifetimes, channels,
typed sum types, the absence of a settings registry until late in
its life, etc.). The JS version should look like idiomatic modern
ESM Node: `async`/`await` for IO, promises over manual polling,
plain objects and arrays over enum + impl ceremonies, `for...of`
loops over iterator chains where iterator chains aren't clearer,
top-level `await` where it reads well, and short modules that
export a few functions rather than struct-with-methods objects
unless statefulness genuinely demands one. If a piece of Rust
machinery exists only to satisfy the borrow checker or to give a
state machine a typed name, drop it — the JS won't need it.

---

## 5. Dependencies

Minimal. Just the two we genuinely need:

- **`smol-toml`** — ESM-first TOML parser/serializer. Round-trips arrays of
  strings cleanly. No comment preservation, but our TOML is small and
  regenerated frequently.
- **`jsonc-parser`** — Microsoft's CST-aware JSONC parser. Required to edit
  opencode/pi configs while preserving user comments and formatting.

Dev deps: `prettier` and `eslint` (per the formatting skill), plus
`@eslint/js` and `globals`.

Everything else is `node:*` builtins (`http`, `child_process`, `fs`, `path`,
`os`, `crypto`).

---

## 6. Subcommands

| command                            | purpose                                                               |
| ---------------------------------- | --------------------------------------------------------------------- |
| `lui` (no args, or flags only)     | resume: equivalent to `lui run` — start the most recently used model  |
| `lui run [NAME]`                   | run a model by name; with no name, run the most recent                |
| `lui new NAME ENGINE [-- ARGS...]` | create a model entry; tokens after `--` are the model's `args[]`      |
| `lui set NAME -- ARGS...`          | replace `args[]` for an existing model (full replace; no append form) |
| `lui ls`                           | list models, compact, args color-coded by segment                     |
| `lui rm NAME`                      | delete a model entry                                                  |
| `lui cmd [NAME]`                   | print the spawn command lui would use, color-coded by segment         |
| `lui show [NAME]`                  | dump resolved config: `[global]` table + per-model block(s)           |
| `lui ssh USER@HOST`                | configure a client over ssh, print reverse-tunnel command             |
| `lui remote HOST[:PORT]`           | connect TUI to a remote lui server                                    |
| `lui websearch`                    | run only the websearch HTTP server + bookmarklet UI                   |
| `lui -h` / `lui --help`            | help                                                                  |

**Subcommand dispatch rule.** If the first _positional_ argument is a
known subcommand verb, that subcommand handles the rest. If there are
no positional arguments (flags only, or nothing at all), lui acts as
if you ran `lui run` with the same flags — i.e. it runs the most
recently used model.

So `lui` resumes; `lui --debug log.txt` resumes with a debug log;
`lui run NAME` runs `NAME`; `lui new phi llama-server -- --hf foo/bar`
creates an entry. `lui --help` and `lui -h` are special-cased to print
help instead of running.

There is no bare-positional shortcut: `lui phi` does **not** run the
`phi` model — `phi` is parsed as a (failed) subcommand. Use `lui run
phi`.

When lui falls through to the implicit `run`, every flag accepted by
`lui run` is accepted in the bare form too: `lui --debug ./log.txt`,
`lui --engine-port 9001`, `lui --public --web-port 9999` all behave
exactly like `lui run` with those flags. The parser doesn't care which
form the user typed.

Run-time flags (accepted on `lui run` — and therefore on bare `lui`):

- `--debug PATH` — write raw engine stdout to PATH for this run.
  Runtime-only; there is no persistent equivalent. If you always want
  a debug log you can shell-alias it.
- `--engine-port N`, `--web-port N` — override `[global]` defaults for
  this run only.
- `--public` — bind 0.0.0.0 instead of 127.0.0.1 for this run only.

`lui websearch` accepts `--web-port` and `--public`; the others don't
apply since there's no engine. Other subcommands (`new`, `set`, `ls`,
`rm`, `cmd`, `show`, `ssh`, `remote`) take no run-time flags — they
either touch the config or perform one-shot RPCs.

---

## 7. Config file format

TOML at `~/.config/lui.toml`. Top-level header is `[global]`, not `[server]`.

```toml
[global]
engine_port  = 8080
web_port     = 8081
websearch    = true
active_model = "phi"

[global.harness.opencode]
enabled = true

[global.harness.pi]
enabled = false

[global.engines.llama-server]
binary = "llama-server"      # optional; falls back to PATH lookup

[models.phi]
engine = "llama-server"
args   = [
    "--hf",
    "unsloth/phi-4-GGUF:Q4_K_M",
    "-c",
    "32768",
    "-ngl",
    "-1"
]

[models.qwen]
engine = "llama-server"
args   = [
    "-m",
    "/models/qwen2.5-coder-32b-q4.gguf",
    "-c",
    "65536",
    "-ngl",
    "-1",
    "--swa-full"
]
```

`[global.harness.<name>]` is a sub-table per harness so each one owns
its own namespace for whatever config it needs — cleaner than the flat
`[global.harness]` boolean map. See the consistency note below.

Notes:

- `args` is opaque. lui never reads it. The engine reads it when it wants to.
- Model names double as aliases. No second alias table.
- `active_model` is the implicit "most recent". Every `lui run NAME` updates
  it; bare `lui` reads it.
- `[global.engines.<name>]` is an optional per-engine sub-table for things
  like binary path override. Engines that don't need it just don't read
  it. Future hyphenated engine names (`mlx-lm`, etc.) work fine as bare
  TOML keys; names with dots would need quoting.
- Per-harness sub-tables: `enabled` is a convention we'll keep across
  every shipped harness, but lui doesn't enforce field names — each
  harness owns the body of its sub-table and we stay consistent by
  policy, not by schema.

When the file is loaded via `Config.load()`, it maps directly to JS:
`lui.config.global.engine_port`, `lui.config.global.harness.opencode.enabled`,
`lui.config.models.phi.args`, etc. No reshape, no rename — TOML keys
become JS property names verbatim.

---

## 8. Engine contract

Each engine lives at `src/engine/<name>.js` and exports one object:

```js
import {
    STYLE_SEGMENT_BINDING, // "the host/port lui owns"
    STYLE_SEGMENT_POLICY, // "engine-injected policy flags"
    STYLE_SEGMENT_USER // "user-supplied tokens" (i.e. default)
} from "../engine.js"

const DIM = { dim: true }
const BOLD = { bold: true }
const ORANGE = { fg: [255, 165, 0] }

export const engine = {
    name: "llama-server", // identifier in TOML: `engine = "llama-server"`
    defaultBinary: "llama-server", // what we look up on $PATH

    // Build the final argv as named, styled segments. Engines pick the
    // segment names that make sense to them — there is no fixed
    // taxonomy. `style` is a PaletteEntry; `lui cmd` and `lui show`
    // colorize using the same View tech that engines use for panels.
    // The STYLE_SEGMENT_* constants exported from `src/engine.js` give
    // engines a shared convention so all of them paint similar pieces
    // with similar colors; use them when applicable but plain inline
    // PaletteEntry objects are also fine.
    //
    // `model.args` is the raw user-supplied argv; `lui` is the live
    // Lui instance (its `lui.config.global` table provides engine_port,
    // public, etc.; `lui.addWarning(...)` can surface advisories).
    buildArgv(model, lui) {
        const host = lui.config.global.public ? "0.0.0.0" : "127.0.0.1"
        const port = lui.config.global.engine_port

        return {
            binary: this.defaultBinary,
            segments: [
                { name: "binding", style: STYLE_SEGMENT_BINDING, args: ["--host", host, "--port", String(port)] },
                { name: "policy", style: STYLE_SEGMENT_POLICY, args: ["-fa", "on", "--jinja", "-v", "--log-colors", "off"] },
                { name: "user", style: STYLE_SEGMENT_USER, args: [...model.args] }
            ],
            warnings: [], // string[] — surfaced in the Warnings panel
            errors: [] // string[] — fatal; lui aborts before spawn
        }
    },

    // Optional. Called once by `runEngine` (in src/engine.js) right after the child is
    // spawned, before any parseLine call. Used to seed fields that come
    // from spawn-time info (the model record itself) rather than from
    // stdout — e.g. `lui.state.modelName = lui.activeModel.name`.
    initState(lui) {
        lui.state.modelName = lui.activeModel.name
        lui.state.startedAt = Date.now()
    },

    // Mutate engine state from one stdout line. The engine reads and
    // writes its fields under `lui.state.*` — that namespace is the
    // engine's bag, isolated from lui's own fields on `lui` directly.
    //
    // For the llama-server engine specifically, the canonical behavior
    // — which regexes catch which lines, how phase transitions work,
    // ANSI stripping, download/prefill/generate progress bars,
    // model/VRAM/RAM/uptime fields, version + git hash + update banner,
    // request/websearch counters, crash detection — lives in the Rust
    // implementation at `old/src/server.rs`. Port behavior from there;
    // every `ServerState` field becomes a key on `lui.state`, and every
    // rendered row in `old/src/display.rs` becomes a `p.line(...)` call
    // in `appendPanels`.
    parseLine(line, lui) {
        /* read/write lui.state.* */
    },

    // Append engine panels onto a shared View. There's only ever one
    // View per /data request — lui creates it, appends its own panels,
    // then hands it to the engine, which appends more. One palette,
    // populated cooperatively, no merge step. The engine never creates
    // a View itself.
    appendPanels(v, lui) {
        const p = v.panel("Model")

        p.line()
            .text("Model    : ")
            .style(ORANGE)
            .text(lui.state.modelName)
            .style()
            .text(" — ")
            .style(BOLD)
            .text(lui.activeModel.name)

        p.line({ indent: 15 }).style(DIM).text(`--hf ${lui.state.hfUrl}`)

        p.bar({ label: "KV cache", value: lui.state.kvFrac, text: lui.state.kvText })

        // ...add Performance panel, Server Log panel, etc. in order...
    }
}
```

### Rules

- The engine **owns its required prefix**: host/port binding, opinionated
  policy flags (e.g. llama-server's `-fa on`, `--jinja`, `-v`,
  `--log-colors off`, `--cache-reuse 256`, `-kvu`).
- The engine **may reject conflicting user args**: if the user passed
  `--port` in `model.args`, the engine returns an `errors[]` entry and
  `lui run` refuses to spawn. lui surfaces the errors and exits.
  Same mechanism is what makes `lui new NAME ENGINE` work without a
  `starterArgs` concept: we just feed an empty `args[]` into
  `buildArgv`, and if the engine doesn't error, the new entry is valid
  even with no user-supplied args.
- The engine **does not parse semantics** of user args. It treats them as
  opaque tokens. Coloring of segments comes from the engine's own
  `style` choice per segment, not a fixed lui taxonomy.
- The engine **renders panels** into a shared View, not raw status
  fields. The TUI doesn't know what "KV cache" means; it just paints
  what the engine declared. The View is created by web.js and passed
  to the engine after lui has added its own panels — engines never
  create a View themselves, so there's a single palette across the
  whole payload.
- The engine **stores its running state under `lui.state.*`**. That
  namespace is the engine's bag — lui's own fields (`lui.warnings`,
  `lui.config`, `lui.web`, `lui.activeModel`, etc.) live on `lui`
  directly. The two-namespace split means engines can't accidentally
  clobber lui internals, and lui can clear `lui.state = {}` on engine
  swap without worrying about its own fields. There's no schema on
  `lui.state`; engines own its shape end-to-end.

### Argv segments

```
Segment = { name: string, style: PaletteEntry, args: string[] }
```

- `name`: free-form string the engine picks. Common names will
  emerge by convention ("binding", "policy", "user") but lui places
  no constraint on them.
- `style`: a `PaletteEntry` (same shape as palette entries everywhere
  else). `lui cmd` and `lui show` paint each segment with this style.
  `style: {}` means "render in default text style".
- `args`: the actual argv tokens in order. Concatenating every
  segment's `args` in order gives the spawn command.

`lui cmd` and `lui show` consume segments by building a `View` and
calling `.style(seg.style).text(seg.args.join(" "))` for each one in
turn — the same builder tech panels use. No bespoke colorizer.

---

## 9. Harness contract

Each harness lives at `src/harness/<name>.js`:

```js
export const harness = {
    name: "opencode",
    defaultEnabled: true, // seed for [global.harness.opencode].enabled on first config write
    configDir: "~/.config/opencode",
    configCandidates: ["opencode.jsonc", "opencode.json"], // first found wins; first listed is the fresh-install default

    // Optional. Called by `lui ssh USER@HOST` to verify the harness is
    // actually installed on the remote box before we start writing
    // config files over SSH. Implementation typically runs `command -v
    // <tool>` (or similar) over the SSH connection and returns
    // `{ ok: true }` if found, `{ ok: false, error: "..." }` otherwise.
    // If preflight fails, `lui ssh` aborts with the harness's error
    // message — silent misconfiguration is worse than a loud failure.
    // Local-only operation (`lui run`) never calls preflight.
    async preflight(target) {
        /* return { ok, error? } */
    },

    // CST-surgical edit of a jsonc-parser root object. Reads whatever
    // it needs off `lui` — typically `lui.config.global.engine_port`
    // (to point opencode's baseURL at our llama-server) and
    // `lui.config.global.web_port` (for the lui-web-search SKILL.md).
    apply(rootObj, lui) {
        /* ... */
    },

    // True iff the existing config file has user content but no prior
    // lui-managed block (in which case `lui` drops a `.luibackup` next
    // to it before applying changes).
    needsBackup(existing) {
        /* ... */
    }
}
```

JSONC editing uses `jsonc-parser` so comments and formatting round-trip
through our writes. The `lui-web-search` SKILL.md is generated by a
shared function in `src/harness/index.js` and dropped into
`harness.configDir`.

---

## 10. Wire format (locked)

The format leans on two ideas: styled text is encoded **inline** in strings
using single-byte palette-index escapes, and the wire stays small by sharing
one palette per payload. Everything is just panels of lines and bars; no
top-level status or engine block, no `log_tail` slot, no panel `id`. The
header bar (the `── lui ── llm ui ─────────…─`-looking divider) is drawn by
the renderer from a panel's `title` — it's never on the wire.

### Top-level shape

```
View = {
    version: 1,
    palette: PaletteEntry[],
    panels:  Panel[]
}
```

- `version`: bumped only for breaking changes. Adding fields, palette
  entries, new fg/bg color forms — all backwards-compatible.
- `palette`: one shared array of `PaletteEntry` objects for the whole
  payload. The server stitches lui's palette with each engine's palette
  before serving (re-indexing engine references as needed).
- `panels`: rendered top-to-bottom. The last panel is what becomes
  "Server Log" / scrollback in practice — there's no special slot for
  it, just put it last.

### PaletteEntry

```
PaletteEntry = {
    fg?:   ColorRef,
    bg?:   ColorRef,
    dim?:  boolean,
    bold?: boolean
}

ColorRef = "default" | NamedColor | [r, g, b]
```

Named colors are the standard 8 + bright 8 (`"black"`, `"red"`, …,
`"bright_red"`, …). RGB triplets allow full truecolor. The renderer
compiles each `PaletteEntry` to an ANSI SGR sequence once at parse time
and caches it; downgrade behavior for non-truecolor terminals lives
entirely in the renderer.

**`palette[0]` is always `{}`** — the default style, reserved. The
builder populates it automatically at the start of each render pass,
so the palette is never empty. Renderers may rely on this.

### Inline palette switches

Text fields (`Line.text`, `Bar.label`, `Bar.text`) carry style by
embedding single Unicode code points `\uE000..\uE0FF` (the Private Use
Area) that switch the current style to `palette[0..255]`. The mapping
is direct — `\uE003` switches to `palette[3]`. Since `palette[0]` is
always `{}`, emitting `\uE000` returns to the default style; there is
no separate "reset" sentinel.

PUA code points were chosen over the C0 control range (`\x01..\xFF`)
because they survive cleanly through JSON, markdown viewers, file
diffing tools, copy-paste, and our own editing harness — control bytes
were tripping every one of those.

A full payload sample (matches the screenshot you posted, with the
`── lui ── llm ui ──…─` banner rendered from `panels[0].title`):

```jsonc
{
    "version": 1,
    "palette": [
        {}, // 0 default
        { "dim": true }, // 1 dim
        { "bold": true }, // 2 bold
        { "fg": "cyan" }, // 3 cyan
        { "fg": [255, 165, 0] }, // 4 orange numbers
        { "fg": "green" }, // 5 ready/ok
        { "fg": "yellow", "dim": true } // 6 update-available hint
    ],
    "panels": [
        {
            "title": "lui — llm ui",
            "lines": [
                { "text": "\uE003http://127.0.0.1:8081/setup", "align": "right" },
                { "text": "Memory   : \uE00423.5\uE000 GiB VRAM · \uE004540\uE000 MiB RAM (\uE00424.0\uE000 GiB total)" },
                { "indent": 15, "text": "\uE001GPU: 21099 model + 2720 KV + 248 compute MiB · CPU: 515 model + 25 compute MiB" },
                { "indent": 15, "text": "\uE00141/41 layers offloaded (fully GPU, embedding on CPU)" },
                { "text": "Model    : \uE003Qwen3.6-35B-A3B\uE000 (Q4_K - Medium) \u2014 \uE002qwen\uE000" },
                { "indent": 15, "text": "\uE001--hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M" },
                { "indent": 15, "text": "\uE00134.66 B \u00B7 20.60 GiB on disk (5.11 BPW)" },
                { "indent": 15, "text": "\uE001262,144 token context window" },
                { "text": "llamacpp : \uE005Ready\uE000 (8680 (15f786e65), uptime: <1m)  \uE006(update available)\uE000" },
                { "indent": 15, "text": "\uE001127.0.0.1:8080" },
                {
                    "indent": 15,
                    "text": "\uE001-c 262144 -ngl -1 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 -np 1 -ctk q8_0 -ctv q8_0 -t 12 --fit-target 2048"
                }
            ],
            "bars": []
        },
        {
            "title": "Performance",
            "lines": [
                { "text": "WebSearch:    0 total \u00B7    0 active" },
                { "text": "Requests :    0 total \u00B7    0 active \u00B7    0 reproc \u00B7    0 invalidated" }
            ],
            "bars": []
        },
        {
            "title": "Server Log",
            "lines": [{ "text": "\uE001parameter_order: ''" }],
            "bars": []
        }
    ]
}
```

### Panel

```
Panel = { title: string, lines: Line[], bars: Bar[] }
```

`title` is a plain string. The renderer draws the `── TITLE ─────…─`
divider; the title string itself may contain palette switches if an
engine wants colored title pieces (rare).

**Implicit left gutter.** The renderer applies a small constant left
gutter (default `2` columns) to every panel — the title divider, body
lines (both left-aligned and the right-justified column they share),
and bars all shift right by this amount. There is no right gutter:
right-aligned content extends to the terminal's right edge and the
divider dashes do the same.

The gutter is a cosmetic renderer concern, not part of the wire format.
Engines and lui's own panel-builder never specify it; they emit
`indent: 0` to mean "flush with the panel's content area," not "column
zero of the terminal." So a Line with `indent: 0` paints at absolute
column `gutter`, and a Line with `indent: 15` paints at `gutter + 15`.

### Line

```
Line = {
    text:    string,                  // body; may contain palette switches; "" is a blank row
    align?:  "left" | "right",        // default "left"
    indent?: number                   // column count for body start + wrap continuations
}
```

- `text`: always present (use `""` for a blank row).
- `align`: omitted or `"left"` means left-aligned, starting at the
  panel content area's left edge plus `indent`. `"right"` means
  right-justified to the terminal's right edge; `indent` is ignored
  when right-aligned (right-aligned lines never wrap — if they
  overflow, the body is truncated on the _left_ with a leading `…`).
- `indent`: integer column count _relative to the panel content area_.
  Left-aligned text begins at `gutter + indent` and any wrap
  continuations resume at the same column. Default `0`. No
  leading-space-padding needed in the string — the renderer inserts
  the indent itself.

Renderer rules:

- **Width**: terminal width is known only at the renderer.
- **Style carries across wraps**: when a left-aligned `text` wraps, the
  renderer remembers the current palette index from the previous row
  and re-emits the corresponding SGR sequence at the start of the
  continuation row.
- **Right-aligned lines don't wrap**: they're meant for short pinned
  content (URLs, status badges). If they overflow, leading content is
  truncated with `…`.
- **Lines are independent**: style does not carry between Line objects.
  The renderer emits a hard reset at end-of-line before painting the
  next line.

### Bar

```
Bar = { label: string, value: number, max?: number, text?: string }
```

- `value` is in `[0, 1]` when `max` is omitted; otherwise `value/max`
  is the fraction.
- `label` and `text` may contain palette switches.
- No wrap, no alignment knob, no right-content. If you need anything
  more complex, render the bar as a Line yourself.

### What's on the renderer side, not the wire

- The `── A ── B ─────…─` divider style for panel titles.
- The bottom-pinned scrollback area is just the last `panels[]` entry.
  Renderer chooses how much terminal real estate it takes.
- Resize re-paint.
- Truecolor → 256-color → 16-color degradation (renderer compiles the
  palette).

### Forwards compatibility

Renderers must ignore unknown keys at every level. The `version` field
bumps only when we change the **semantics** of existing fields. New
optional fields, new palette entry forms (e.g. underline, italic), new
color refs — all non-breaking.

### Producing views: builder helpers in `src/wire.js`

Nobody — neither lui itself nor any engine — should be hand-typing
`\uE003…\uE000…` escapes into raw strings. `src/wire.js` exports a tiny
fluent builder API on top of the wire format. Two design rules drive
the shape:

1. **No named-style registry.** `.style()` accepts the `PaletteEntry`
   object directly (`{ fg: "cyan", bold: true }`). Callers can use
   object literals inline, or pull from module-level constants
   (`const NUMBER = { fg: [255, 165, 0] }`), or import a shared theme
   module. The builder doesn't care.
2. **Lazy, deduplicated palette population.** Each `View` is built
   fresh per render pass. Its palette starts at `[{}]` and grows on
   demand. The first time the builder sees a given entry it appends
   it and remembers the index; subsequent identical entries (compared
   by deep equality, keyed internally by a canonical JSON string)
   reuse the existing slot. Engines never think about indices.

The whole API:

```js
import { View } from "../wire.js"

const NUMBER = { fg: [255, 165, 0] }
const DIM = { dim: true }
const CYAN = { fg: "cyan" }

const v = View()
const lui = v.panel("lui — llm ui")

// right-justified URL, cyan
lui.line({ align: "right" }).style(CYAN).text("http://127.0.0.1:8081/setup")

// "Memory   : 23.5 GiB VRAM · 540 MiB RAM (24.0 GiB total)" with orange numbers
lui.line()
    .text("Memory   : ")
    .style(NUMBER)
    .text("23.5")
    .style()
    .text(" GiB VRAM · ")
    .style(NUMBER)
    .text("540")
    .style()
    .text(" MiB RAM (")
    .style(NUMBER)
    .text("24.0")
    .style()
    .text(" GiB total)")

// indented continuation; dim for the whole line
lui.line({ indent: 15 }).style(DIM).text("GPU: 21099 model + 2720 KV + 248 compute MiB · CPU: 515 model + 25 compute MiB")

// set once, write a run, then drop back to default
lui.line().style({ fg: "cyan", bold: true }).text("This is bold cyan text").style().text(" — back to default")

// blank row
lui.line()

// bar
lui.bar({ label: "KV cache", value: 0.42, text: "13.8/32 GiB" })

const view = v.build()
// → { version: 1, palette: [{}, { fg: "cyan" }, { fg: [255,165,0] }, ...], panels: [...] }
```

Methods on `View`:

- `.panel(title)` — start a new panel; returns a Panel builder.
- `.build()` → wire-format object ready to send to `/data`.

There is no `.merge()` — there's only ever **one** View per /data
request. lui creates it, lui appends its own panels, the engine
appends its panels into the same View, then `.build()` runs. One
palette, populated cooperatively, no cross-View index rewriting.

Methods on a Panel:

- `.line(opts?)` — push a new Line (`opts` = `{ align?, indent? }`).
  Called with no chain, produces a blank row; chain text/style to add
  content.
- `.bar(spec)` — push a bar (`spec` = `{ label, value, max?, text? }`).
  `label` and `text` accept palette-encoded strings.

Methods on a Line:

- `.text(s)` — append `s` under the current style. The default style
  applies until the first `.style(entry)` call.
- `.style(entry)` — switch the current style to `entry`. Subsequent
  `.text(...)` calls inherit this style until the next `.style(...)`.
- `.style()` (no argument) — shorthand for `.style({})`: switch back
  to the default style.

There are only two methods. Modal style + text. No span, no reset,
no overloads pretending to be different things.

Internally the View carries a `Map<canonicalKey, index>` so lookup is
O(1). The canonical key is `JSON.stringify` with sorted object keys,
so logically equivalent entries dedup regardless of declaration order
or source.

Same builder shape is used by **both** lui (in `appendLuiPanel(v)` and
`appendWarningsPanel(v)`) and every engine (in `appendPanels(v, lui)`).
web.js creates a single fresh View per /data request and calls those
appenders in order; everyone writes into the same View and its single
palette. No engine ever sees the wire-level escape chars and no merge
is ever needed.

Tag-template helpers stack nicely on top if engines want them; e.g.
``lui.line().push(t`Memory   : ${num("23.5")} GiB`)`` where `t` and
`num` are thin wrappers around the builder. We'll ship one set of
basics in `wire.js` and let engines build their own taste on top.

---

## 11. TUI rendering

Two files: `src/ansi.js` (primitives) and `src/display.js` (the loop).

### `src/ansi.js`

Roughly:

- `compilePalette(palette)` → `string[]` where index `i` is the SGR
  prefix string for `palette[i]`. Truecolor RGB compiles to
  `\x1b[38;2;R;G;B...m`; named colors to standard SGR codes; missing
  attributes resolve to "reset that attribute".
- `paint(text, compiled)` → emits text with palette switches expanded.
  Walks the string char-by-char; when it hits `\uE001`..`\uE0FF`,
  emits `\x1b[0m` + `compiled[code-1]`; `\uE000` emits `\x1b[0m`.
  Other chars are passed through.
- `vwidth(text)` — measures visible width, skipping palette-switch
  chars (which are zero-width on screen).
- Cursor and clear helpers (`cursorTo`, `clearLine`, `hideCursor`,
  `showCursor`).
- `renderBar(width, value, max?, text?, compiled?)` — single-row bar.

### `src/display.js`

Polls `http://127.0.0.1:<web_port>/data` every 250ms. On payload
receive:

1. Compile the payload's `palette` once.
2. Compute layout: terminal `[rows, cols]`. Last panel gets any leftover
   rows after fixed panels are sized.
3. Paint panels in order. Resize triggers full repaint.
4. Hard reset SGR at end of every visible row to keep style from
   bleeding.

### Painting a Line

Given a Line, terminal width `W`, the panel left gutter `G`
(`gutter = 2` by default), and the compiled palette:

**Right-aligned (`align === "right"`):**

1. Measure `bodyW = vwidth(line.text)`.
2. If `bodyW <= W`: emit `W - bodyW` spaces, then the body. Done.
   Right-aligned content extends to the terminal's right edge — no
   right gutter.
3. Else: truncate the leading content of the body with `…` so it fits
   in `W` columns. Right-aligned lines do not wrap.

**Left-aligned (default):**

1. Compute `startCol = G + (line.indent || 0)` and
   `bodyW = vwidth(line.text)`.
2. Available body columns per row are `W - startCol`.
3. If `bodyW <= W - startCol`: paint `startCol` spaces then the body.
   Done.
4. Else: wrap. Word-break preferred; mid-token if a single token is
   wider than the available columns. Continuation rows begin with
   `startCol` spaces, then resume the body. The renderer remembers the
   most-recently-seen palette index from the previous row and re-emits
   `compiled[index-1]` before resuming, so style carries cleanly across
   the break.

Engines never see `W` or `G` and never need to insert their own ANSI
sequences; they just emit text with `..\uE0FF` switches.

### Painting a Panel

Each panel becomes:

1. A divider title row: `G` spaces, then `── TITLE ─────…─` filling the
   remaining `W - G` columns out to the right edge.
2. Each `lines[]` entry, painted as above (left gutter applied to
   left-aligned content; right-aligned content goes flush to the right
   edge).
3. Each `bars[]` entry as a single row from `renderBar`, also offset by
   `G`.
4. A trailing blank row separates panels.

The last panel in the array is treated specially only in that it
absorbs leftover terminal height — its content scrolls within the
remaining rows. There's no separate "log tail" concept; if an engine
wants a log section, it emits a panel and puts it last.

---

## 12. Runtime behavior

### The `Lui` class

`src/lui.js` exports a `Lui` class. `main.js` parses argv, instantiates
one `Lui`, and dispatches the chosen subcommand by calling a method on
it. The class is the orchestrator and the bag of state that engines /
web.js / display.js / harnesses all read from. There is no separate
"engine state" parameter — engines read and write `lui.state.*`,
isolated from lui's own fields (`lui.config`, `lui.warnings`,
`lui.web`, `lui.activeModel`, …) on `lui` directly.

Skeleton:

```js
import { View } from "./wire.js"
import { Config } from "./config.js"
import { startWebServer } from "./web.js"
import { startTui } from "./display.js"
import { runEngine, engines } from "./engine.js"

export class Lui {
    constructor() {
        this.config = Config.load() // ~/.config/lui.toml
        this.startedAt = Date.now()
        this.warnings = [] // [{ text, addedAt }] — strings, dated when added
        this.requestCount = 0
        this.websearchCount = 0

        // populated when an engine is spawned
        this.engineModule = null // import * from src/engine/<name>.js
        this.engineChild = null // ChildProcess
        this.activeModel = null // { name, engine, args }
        this.state = {} // engine's bag — engine reads/writes lui.state.*

        // populated when services start
        this.web = null // { server, port, bookmarkletUrl, close() }
        this.tui = null // { stop() }
    }

    /* ---------- subcommands (one method each) ---------- */

    async run(name) {
        /* resolve model, spawnEngine, startWebServer, startTui, await shutdown */
    }
    async new(name, engineName, args) {
        /* validate via buildArgv, save config */
    }
    set(name, args) {
        /* full-replace this.config.models[name].args */
    }
    rm(name) {
        /* delete this.config.models[name] */
    }
    ls() {
        /* one line per model + reminder of `lui show NAME` */
    }
    show(name) {
        /* dump resolved TOML; non-TTY-aware */
    }
    cmd(name) {
        /* engine.buildArgv → paint segments via View; non-TTY-aware */
    }
    async ssh(target) {
        /* harness preflight + remote config write, print tunnel cmd */
    }
    async remote(host) {
        /* GET /config from remote, start local display + bsearch */
    }
    async websearch() {
        /* start web.js + display.js, no engine spawn */
    }

    /* ---------- engine lifecycle ---------- */

    async spawnEngine(model) {
        this.activeModel = model
        this.engineModule = engines[model.engine]
        this.state = {}

        const { binary, segments, errors, warnings } = this.engineModule.buildArgv(model, this)
        if (errors?.length) {
            /* print + process.exit(1) */
        }
        for (const w of warnings ?? []) this.addWarning(w)

        this.engineModule.initState?.(this)
        this.engineChild = await runEngine(this, binary, segments) // runEngine() lives in src/engine.js
    }

    async shutdown(code = 0) {
        this.tui?.stop()
        if (this.engineChild) {
            this.engineChild.kill("SIGTERM")
            // wait up to 5s, then SIGKILL
        }
        this.web?.close()
        this.config.save()
        process.exit(code)
    }

    /* ---------- View composition ---------- */

    // Append the "lui — llm ui" status panel onto the shared View.
    // Always emits exactly one panel.
    appendLuiPanel(v) {
        const p = v.panel("lui — llm ui")
        if (this.web?.bookmarkletUrl) {
            p.line({ align: "right" }).style({ fg: "cyan" }).text(this.web.bookmarkletUrl)
        }
        // ...port, web_port, websearch state, harnesses configured...
    }

    // Append the "Warnings" panel onto the shared View, after filtering
    // out aged-out entries. No-op when nothing is live. TTL and style
    // are lui-internal — callers just supply the warning text.
    appendWarningsPanel(v) {
        const now = Date.now()
        const live = this.warnings.filter((w) => now - w.addedAt < Lui.WARNING_TTL_MS)
        this.warnings = live
        if (!live.length) return
        const p = v.panel("Warnings")
        for (const w of live) p.line().style({ fg: "yellow" }).text(w.text)
    }

    addWarning(text) {
        this.warnings.push({ text, addedAt: Date.now() })
    }
}

Lui.WARNING_TTL_MS = 60_000 // single TTL for all warnings; tweak in one place
```

### Runtime topology for `lui run`

Single Node process. `main.js` builds a `Lui`, calls `lui.run()`. That
method:

1. Resolves the model from `this.config.models[name]`.
2. Calls `this.spawnEngine(model)`. That invokes
   `engine.buildArgv(model, this)`; errors abort with stderr, warnings
   flow into `this.warnings`. `initState(this)` runs once, then the
   child is spawned via `runEngine()` (exported from `src/engine.js`),
   which wires its stdout to `engine.parseLine(line, this)` per line
   and tees raw output to the `--debug` file if one was provided.
3. `this.web = await startWebServer(this)` — raw `node:http` on
   `config.global.web_port`. The `/data` route creates one fresh
   `View`, calls `lui.appendLuiPanel(v)`,
   `lui.appendWarningsPanel(v)`, then
   `lui.engineModule?.appendPanels(v, lui)`, then serializes
   `v.build()`.
4. `this.tui = startTui(this)` — polls `localhost:<web_port>/data`
   every 250ms and paints.
5. `await` a shutdown promise resolved by SIGINT or engine exit.

For `lui remote` the same display.js runs locally but polls the
_remote_ lui's `/data`. The local `Lui` runs a stripped-down web.js
serving only `/bsearch` + `/results` + `/setup` (so the bookmarklet
still works against the real browser on the client side).

For `lui websearch` the same web.js + display.js run, but no engine
is spawned. The "lui" panel still renders; the engine panel is absent.

### Composing the View

Inside `web.js`, the `/data` handler is roughly:

```js
function handleData(req, res, lui) {
    const v = View()
    lui.appendLuiPanel(v) // always: the "lui — llm ui" status panel
    lui.appendWarningsPanel(v) // conditional: the "Warnings" panel, if any
    lui.engineModule?.appendPanels(v, lui) // engine panels (skipped in `lui websearch` mode)
    res.setHeader("content-type", "application/json")
    res.end(JSON.stringify(v.build()))
}
```

The order is explicit and caller-controlled: lui status first, then
warnings, then whatever panels the engine wants to emit. Each appender
is named after the panel(s) it adds, so reading the handler tells you
the final panel order without chasing definitions.

One View, one palette. lui and the engine each call `v.panel(...)`
and add lines/bars; every `.style(entry)` call dedups against the
same growing palette. `.build()` returns a wire-format object ready
to JSON-serialize. No merge step, no index rewriting — the View is
the shared workspace.

### Shutdown

SIGINT or SIGTERM to lui triggers `lui.shutdown()`:

1. Tear down the TUI (`display.stop()`) — restore cursor, reset SGR,
   clear screen if appropriate. Done first so any subsequent error
   messages aren't painted over a stale render.
2. SIGTERM the engine child. Wait up to 5s for clean exit.
3. SIGKILL if it didn't exit.
4. Close the HTTP server.
5. Save config (if any persistent state changed during the run, e.g.
   `active_model`).
6. Exit with code 0 (normal) or whatever the engine's exit code was
   if it died unexpectedly.

### Engine crash teardown

When the engine child exits unexpectedly mid-session, the same
shutdown order runs:

1. Tear down the TUI first.
2. Print lui's best understanding of the failure to stderr — the
   engine's parsed exit reason from its last stdout lines, plus the
   exit code.
3. Close the HTTP server, save config.
4. Exit nonzero (the engine's exit code, or `1` if it died from a
   signal).

Matches the current Rust behavior: lui doesn't sit there showing a
crashed badge or attempt to restart. The user gets an explanation and
a fresh prompt.

### Warnings panel

A warning is just a string. lui owns the `warnings` array
(`[{ text, addedAt }]`), the TTL (a single class-level constant), and
the rendering choice (amber on the "Warnings" panel) — those are all
implementation details, not part of any caller's interface. Anyone
that wants to surface an advisory calls `lui.addWarning(text)`.

Sources:

- `engine.buildArgv(...).warnings` is a `string[]`; each entry becomes
  one warning.
- Harnesses, ssh setup, websearch, etc. can call `lui.addWarning(...)`
  directly for things like "config file looked hand-edited, backed up
  before overwrite."

When `lui.appendWarningsPanel(v)` runs it drops aged-out entries,
then if anything is left emits one Line per warning in the "Warnings"
panel, amber. The panel slots in between the `lui` status panel and
the engine's panels because that's the order web.js calls the
appenders in. Subsequent `/data` polls re-run the filter, so the
panel silently disappears once everything has aged out.

If we later decide warnings need styles or per-entry TTLs, the
internal shape can grow without changing the `addWarning(text)`
public surface — but we're not paying that complexity yet.

### Atomic config writes

`config.save()` writes to `~/.config/lui.toml.tmp` first, then renames
to `~/.config/lui.toml`. Power failure mid-write at worst leaves the
old file intact plus a `.tmp` file lying around (and lui can clean
those up at startup if it sees one).

### TTY detection for `lui cmd` and `lui show`

When stdout isn't a TTY (`!process.stdout.isTTY`), strip palette
switches and emit the text raw. So `lui cmd phi > spawn.sh` and `lui
show | less` both work without ANSI gunk. The `display.js` TUI loop
is unaffected — it always paints to a TTY by definition.

### No TOML migrations

lui doesn't migrate `~/.config/lui.toml`. The format is intentionally
small and stable. If we ever rename a key, users edit their TOML by
hand. Keeps the load path simple: parse → use, no migration ladder.

---

## 13. Decisions (settled)

The first round of open questions has been resolved:

1. **Engine binary discovery.** Lookup by `engine.defaultBinary` on
   PATH, overridable via `[global.engines.<engine>].binary = "/abs/path"`.

2. **`lui new` semantics.** Required args are both positional: `NAME`
   and `ENGINE`. Optional `-- ARGS...` after that. No `starterArgs`
   concept. lui hands the new model (with whatever args the user did
   or didn't supply via `--`) to `engine.buildArgv` for validation;
   if it returns `errors[]`, lui refuses to add the entry. If it
   returns no errors, the entry is valid — even with zero
   user-supplied args. This naturally accommodates future engines
   that don't need any args.

3. **Run-time flag form.** Order-insensitive — flags can appear before
   or after the positional. Not worth fussing over; the parser handles
   both.

4. **`lui show` vs `lui ls`.** Split confirmed: `lui ls` prints one
   compact line per model (name, engine, args summary); `lui show`
   prints the full resolved config (`[global]` table, `[global.engines.*]`,
   all `[models.*]` blocks rendered TOML-style). `lui show NAME` shows
   just that model. `lui ls` ends with a punctuating reminder line
   like `(use \`lui show NAME\` for full details)`.

5. **CLI parser.** Hand-rolled in `src/main.js`. No minimist or
   commander.

6. **TOML formatting.** Keep it simple. smol-toml's native output is
   fine; the one custom touch is that string arrays get one entry per
   line so diffs stay readable. No fancy column alignment, no flag /
   value pairing logic — just one string per line.

7. **HTTP server.** Raw `node:http` with a small route dispatcher in
   `web.js`. No framework.

8. **Engine crash behavior.** lui prints its best understanding of the
   cause (the engine's parsed exit reason / last stderr line) and
   exits with a nonzero status. Same as the current Rust impl —
   matches user expectation. No auto-restart, no zombie TUI showing a
   "crashed" badge forever.

9. **`lui websearch` mode.** Uses the same TUI + panels as `lui run`,
   just without the engine panel. lui still emits its own panel
   (port, bookmarklet URL, queries served counter, etc.) and the log
   panel for HTTP request lines.

10. **Web port configurability.** The web port is just another `[global]`
    setting — `[global].web_port`. Default `8081`. If the user changes it
    they re-drag the bookmarklet. We don't pin the bookmarklet port
    artificially. (The persistent setting is `web_port`; the persistent
    engine port is `engine_port` — both clearer than the old generic
    `port`.)

---

## 14. Style notes for the implementation

(These bind whoever writes the code, including future me.)

- ESM throughout. No CommonJS, no `require`.
- 4-space indent, no semis, 130 col, `trailingComma: "none"` (per the
  formatting skill's prettier config).
- ESLint config per the formatting skill, with `globals.node` swapped in
  for `globals.browser`.
- `npm run format` and `npm run lint` are the only invocations — never
  call prettier or eslint directly.
- **Comments describe the code in front of the reader.** Never reference
  plan-state ("v1", "phase 6 adds this", "for now we...", "will be removed
  later"). The plan lives in markdown; source files only describe what
  they currently are.
- Prefer inline code over single-call-site helper functions. Extract only
  when there are real second and third callers.
- Top-of-file comments are welcome: a few lines saying what the file
  offers. Function-level comments only when the _why_ isn't obvious from
  the code.

---

## 15. Out of scope until the above is signed off

- Actually writing any code.
- Building `package.json`.
- Filling in implementation details for `parseLine` on llama-server
  (will need a quick survey of `old/src/server.rs`'s log parser to
  port the regexes).
- Designing the per-engine "starter args" curated lists.
- Validating that smol-toml's array-of-strings round-trip is pretty
  enough or whether we need a small post-format pass.
