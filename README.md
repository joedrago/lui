# lui

A friendly TUI wrapper for [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`. Pronounced **"Louie"** - short for *llama.cpp ui*.

## Setup

1. **Install opencode** - <https://opencode.ai>. No config needed; lui will wire it up for you.
2. **Put `llama-server` on your PATH.**
   - **macOS:** `brew install llama.cpp`
   - **Windows:** grab the matching `llama-bin-win-*` zip **and** the `cudart-llama-bin-win-cuda-*` zip (for NVIDIA) from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases), extract them into the same folder, and add that folder to your `PATH`.
    - **Linux:** grab the matching `llama-bin-ubuntu-*` tarball from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) - `*-rocm-*` for AMD GPUs, `*-vulkan-*` for any other GPU - extract it, and add the folder to your `PATH`.
    - Verify the install - run `llama-server --version` and make sure it detects your GPU with no serious errors.
3. **Build lui** - `cargo build --release`, then put `target/release/lui` on your PATH (symlink, copy, or add the directory to your `PATH`).

## Run

Example first runs (use `lui -h` to see what these settings mean and tune them for your machine):

```
# This runs on *anything* and is pretty good. the UD-Q6_K might be even better if you turn down -c a bit
lui --hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M
lui --hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M --this -c 262144 --presence-penalty 1.5
lui --hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M --alias qwenmoe --this -c 262144 --ctk q8_0 --ctv q8_0 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00 --presence-penalty 1.5

# This runs on fewer things but seems great so far
lui --hf unsloth/Qwen3.6-27B-GGUF:Q4_K_M --alias qwendense --this -c 131072 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --reasoning on --presence-penalty 1.5

# GLM 4.7 is fun
lui --hf unsloth/GLM-4.7-Flash-GGUF:Q4_K_M
lui --hf unsloth/GLM-4.7-Flash-GGUF:Q4_K_M --alias glm --this -c 131072 --ctk q8_0 --ctv q8_0

# Gemma is also neat
lui --hf "unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M" --alias gemma --this --ctk q8_0 --ctv q8_0 --temp 1.0 --top-p 0.95 --top-k 64
```

lui will show download progress bars, start `llama-server`, configure opencode, and print **Ready** once the model is loaded.

Every flag you pass is saved to a toml config, so next time just run:

```
lui
```

Other useful flags: `-m <path>` for a local GGUF, `--ngl <n>` for GPU layers, `--port <n>`, `--public` to bind `0.0.0.0`, `-l` to list cached models.

## Connecting to a Shared Server

lui supports two machines connecting to share a model. The two flags are mutually exclusive (`--ssh` and `--remote` can't be used together) and neither is persisted to `lui.toml`.

### `--ssh` - share your local LLM with a remote machine

`lui --ssh` is a turnkey way to let someone access the LLM running on your workstation over SSH. It configures their machine to point back at your llama-server through a reverse tunnel, so they can use opencode as if the model were running locally on their end.

Run `lui --ssh USER@HOST` **on the server** (the machine where `llama-server` is running). It SSHes into the client machine, installs an opencode config there, and prints an `ssh -R ...` command for you to run.

**What it does:**

1. Checks that `opencode` is installed on the client machine.
2. Picks a random high port on the client (18000–28999) for llama-server and the next port for websearch.
3. Writes `~/.config/opencode/opencode.json` on the client with `baseURL` pointing to `http://localhost:<client_port>/v1`.
4. Writes the `lui-web-search` SKILL.md on the client (unless websearch is disabled), baked with the correct client ports.
5. Prints an `ssh -R ...` command. You run that command in another terminal to establish the reverse tunnel.
6. Exits.

**Requirements:**

- `opencode` must already be installed on the client (lui probes the default PATH, login shell PATH, and the installer canonical location `~/.opencode/bin/opencode`).
- SSH access to the client (`USER@HOST` format required - no bare hostnames).
- The client should be able to reach this machine's network interface for the reverse tunnel to work.

**Example:**

```
# On the server machine:
lui --ssh user@workstation

# Then in another terminal on the server:
ssh -R 23847:localhost:8080 -R 23848:localhost:8081 user@workstation
```

### `--remote` - use a remote LLM while keeping web search local

`lui --remote` is a special mode that doesn't run `llama-server` locally. Instead, it connects to a lui instance on another machine (running with `--public`) while still providing local web search on your workstation. The TUI renders by polling the remote server, and the in-process web search server runs here so browser-mediated search works from your browser.

Run `lui --remote HOST[:PORT]` **on your client** (your laptop or workstation that wants to use the server's model). It fetches the server's config over plain HTTP, writes your local opencode config, starts a local websearch server, and runs the TUI.

**What it does:**

1. Fetches `/config` from the server's HTTP endpoint (default port 8081).
2. Validates the config version matches.
3. Writes local `~/.config/opencode/opencode.json` with `baseURL` pointing directly at the server's llama-server (e.g. `http://server:8080/v1`).
4. Writes the local `lui-web-search` SKILL.md, pointed at a bsearch server spawned on this client (port 8081).
5. Spawns an in-process bsearch HTTP server so browser-mediated web search works on your machine.
6. Renders the TUI by polling the server's `/data` endpoint.
7. Blocks until `Ctrl+C`.

**Requirements:**

- The server must be running with `--public` (so the HTTP server binds to `0.0.0.0` and is reachable over the network).
- Network access from the client to the server's HTTP port (default 8081).
- The server's llama-server port will be read from its `/config` - it doesn't have to match the HTTP port.

**Example:**

```
# On your client machine (server must be running with --public):
lui --remote server.local
lui --remote server.local:9000    # custom HTTP port
```

## Sandboxing the Harness

`lui --sandbox HARNESSNAME` launches a configured harness (currently
`opencode`, `pi`, or `late`) wrapped in [nono](https://nono.sh) - a
capability-based sandbox that uses Seatbelt on macOS and Landlock on
Linux. The harness can read+write your project, hit the network, and
invoke local toolchains, but **cannot** touch credentials, browser
data, shell history, or other dirs outside the allow-list. lui itself
never enters the TUI in this mode - it's a thin launcher that blocks
on the sandboxed child and propagates its exit code.

If you never type `--sandbox` (or any `--sandbox-*` flag), lui behaves
exactly as it does today.

### Setup

1. Install nono - <https://nono.sh>. Single binary; the installer
   drops it on your `PATH`.
2. That's it. Defaults are tuned for "drop into a project and run an
   AI agent."

### Defaults at a glance

```
lui --sandbox opencode                    # this works out of the box
```

Resolves to roughly:

```
nono run -p opencode --allow . --allow-cwd \
    --allow ~/.cargo --allow ~/.rustup --allow ~/go --allow ~/.pyenv \
    --allow ~/.npm --allow ~/.bun --allow /usr/local/go ...           \
    -- opencode
```

What that gets you:

- **Profile auto-detected.** If nono ships a profile by the harness's
  name (`opencode`, `claude-code`, `codex`, `swival`, ...), lui uses it.
  Otherwise it falls back to nono's `default` profile (gives `/tmp`,
  `/usr/bin`, homebrew, system reads, plus deny-rules for credentials,
  keychains, browser data, shell history).
- **Project tree r+w** via `--allow .` (the cwd). Skips nono's
  first-run prompt with `--allow-cwd`.
- **Toolchains r+w** for any of `~/.cargo`, `~/.rustup`, `~/go`,
  `/usr/local/go`, `~/.pyenv`, `~/.local/share/uv`, `~/.conda`,
  `~/.nvm`, `~/.fnm`, `~/.npm`, `~/.bun`, `~/.deno`, `~/Library/pnpm`,
  `/usr/local/lib/node_modules`, `~/.nix-profile`, `/nix/store`, etc.
  that exist on your machine. `cargo build` / `go test` / `npm install`
  / `pip install --user` work first try. `$CARGO_HOME` / `$RUSTUP_HOME`
  / `$GOPATH` / `$PYENV_ROOT` override the defaults if set.
- **GPU off**, **network on**. The agent (Node-based) doesn't need
  GPU; lui's llama-server runs *outside* the sandbox.

### Make it the default

If you always want opencode (or any harness) sandboxed, alias it in
your shell rc:

```sh
# ~/.zshrc, ~/.bashrc, etc.
alias opencode='lui --sandbox opencode'
```

Now every plain `opencode` invocation runs through nono with lui's
configured grants. lui itself stays a thin launcher (no TUI, no
llama-server) so the alias adds essentially zero overhead beyond
nono's sandbox setup.

To bypass the sandbox temporarily, prefix with `command`:
`command opencode` (or escape with `\opencode`).

### Preview the resolved sandbox commandline

Plain `lui --cmd` now prints three commandlines: the `lui` invocation
that would reproduce your config, the resolved `llama-server` argv,
and a third **sandbox commandline** showing exactly what
`lui --sandbox HARNESSNAME` would run, with `HARNESSNAME` highlighted
in magenta wherever it'd be substituted. Use it to inspect what's
being granted before you launch.

### Adding access to extra directories

Every `--sandbox-*` flag is persisted under `[sandbox]` in `lui.toml`,
so you only configure once.

* **The harness can't access `~/.<thing>`.**
  * Fix: `lui --sandbox-allow ~/.<thing>` (repeatable; r+w). Read-only?
    Use `lui --sandbox-read ~/.<thing>` instead.
  * Tip: when nono blocks something, it prints
    `Fix: --read /path --read /path …` at the bottom of its denial
    output. Translate each `--read` to `--sandbox-read` and each
    `--allow` to `--sandbox-allow` and re-run.
  * Example: `lui --sandbox-allow ~/.pi --sandbox pi` gives the `pi`
    harness r+w on its state dir. Persists - subsequent
    `lui --sandbox pi` invocations include the grant automatically.

* **The harness needs to reach an extra domain (e.g. an internal
  registry).**
  * Fix: `lui --sandbox-allow-domain api.example.com` (repeatable).
    Maps to nono's `--allow-domain`.

* **I want a tighter sandbox - block all network.**
  * Fix: `lui --sandbox-block-net`. Mirrors nono's `--block-net`.

* **I want to skip the toolchain auto-allow-list and configure paths
  manually.**
  * Fix: `lui --no-sandbox-dev-tools`. Then add what you actually
    need with `--sandbox-allow` / `--sandbox-read`.

### Picking a different profile

* **I have a custom nono profile and want lui to use it.**
  * Fix: `lui --sandbox-profile mycustom`. Skips lui's auto-detect
    and passes `-p mycustom` verbatim to nono.

* **I want no profile at all - just the explicit allows lui produces.**
  * Fix: `lui --sandbox-profile none`. The literal `none` opts out
    of the `-p` flag entirely.

### Authoring a custom nono profile (optional)

If a harness needs a recurring set of grants that don't fit lui's
flags cleanly (e.g. a complex set of allow-rules for a private
toolchain), nono's own `nono profile init` builds a starter:

```
nono profile init my-team --extends opencode --groups rust_runtime,python_runtime
```

That writes a JSON profile under `~/.config/nono/profiles/` (path may
vary). Then point lui at it:

```
lui --sandbox-profile my-team --sandbox opencode
```

You don't *need* to do this for typical use - lui's defaults plus
`--sandbox-allow` / `--sandbox-read` cover most workflows.

### Troubleshooting

* **`--allow-gpu requires the selected profile to opt into allow_gpu`.**
  * Cause: nono's shipped profiles for AI agents (opencode,
    claude-code, etc.) intentionally refuse GPU access since the agent
    is a Node process that doesn't need it.
  * Fix: lui's default is `--no-sandbox-allow-gpu`, so this shouldn't
    trigger unless you flipped it on. To run a GPU-using tool inside
    the sandbox, switch to a profile that opts into GPU
    (`--sandbox-profile none` or a custom one) and pass
    `--sandbox-allow-gpu`.

* **`'nono' not on PATH`.**
  * Fix: install nono from <https://nono.sh>, or override the binary
    location with `--sandbox-bin /absolute/path/to/nono`.

## Quit

`Ctrl+C` or `q`.

## FAQ / Cheat Sheet

`lui` tries to have sensible defaults to make things as turnkey as possible, but every machine+model combo is different, so here's some solutions to problems you might have.

### Scope & aliases

* **I only want to change this setting for _this_ model, not all models globally.**
  * Fix: Put `--this` (or `--local`) before the setting. It writes into `[models."<active>"]` instead of the global `[server]` block.
  * Example: `lui --this --temp 0.2` tweaks only the active model; other models keep whatever global `--temp` you set.
  * Notes: `--this` is sticky until you flip back with `--global`. You can mix them in one command: `lui --temp 0.6 --this --temp 0.2` sets global 0.6 and per-model 0.2 in a single invocation.

* **I want to wipe a per-model override and fall back to the global value.**
  * Fix: Pass the literal word `default` as the value under `--this`. Example: `lui --this --temp default` clears the per-model temperature override.

* **I never remember these model names, help!**
  * Fix: Use `--alias NAME` right after selecting a model. `lui --hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M --alias qwen` lets you later run just `lui qwen`.
  * Fix: The alias pool is inferred - `--hf` aliases go under `[aliases.hf]`, local-path aliases under `[aliases.model]`. Bare positionals (`lui qwen`) look up both pools; a collision is a hard error.
  * Tip: Aliases also work as the argument to `--hf` / `-m` (e.g. `lui --hf qwen --this -c 131072`).

* **How do I switch between cached models without typing the full repo every time?**
  * Fix: After the first run, just `lui` starts the last-used model. To switch, pass the alias (or repo) on its own: `lui gemma`.

### Model identity

* **I have a GGUF sitting on disk, not a HuggingFace repo.**
  * Fix: `-m /path/to/model.gguf` (or `--model`). Mutually exclusive with `--hf`; setting one clears the other.

* **I want to download a specific quant from HuggingFace.**
  * Fix: `--hf ORG/REPO:QUANT`, e.g. `--hf unsloth/GLM-4.7-Flash-GGUF:Q4_K_M`. lui will show download progress and cache the file.

* **What models do I have cached locally?**
  * Fix: `lui -l` (or `--list`) prints cached models along with the current resolved config.

### VRAM & fit

* **This model is using too much VRAM!**
  * Fix: `--fit-target N` - reserves N MiB of free VRAM headroom; llama-server's `--fit` logic will offload more to CPU to honor it. Default is 1024.
  * Fix: Lower `-c` / `--ctx-size`. A huge context is usually the real VRAM hog; halving it often solves the problem alone.
  * Impact: llama-server will offload more of the model to CPU, which typically makes generation slower.

* **lui aborts because it thinks the model won't fit in VRAM - but I know better.**
  * Fix: `--avo` ("allow VRAM oversubscription") skips lui's pre-flight abort. Flip back with `--no-avo` (the default).
  * Impact: You're now trusting llama-server's own fit logic; if it's wrong you'll get an OOM instead of a clean lui error.

* **I bumped context and now loading crashes or fails.**
  * Fix: Check `--fit-target` (maybe lower it), drop context, or set `--ngl` to force more CPU offload. For MoE-friendly setups, `--cache-ram <MiB>` enables llama-server's host-memory prompt cache.

### Context, samplers, batching

* **The context window is too small / too big.**
  * Fix: `-c N` or `--ctx-size N`. `0` means "whatever the model ships with". Large contexts multiply KV-cache VRAM usage - pair with `--ctk/--ctv` if needed.

* **The model feels too random / too deterministic.**
  * Fix: `--temp F` (sampling temperature).
  * Fix: `--top-p F` (nucleus), `--top-k N`, `--min-p F`. Omitting a sampler uses llama.cpp's default; set `--this --top-p default` to clear a per-model override.

* **Prefill is slow / prompt processing is CPU-bound.**
  * Fix: `--ubatch-size N` (aka `--ub`) - physical batch size llama.cpp processes at once. Larger = faster prefill, more VRAM.
  * Fix: `--batch-size N` - logical batch size (lui defaults this to 512 instead of llama.cpp's 2048, which is kinder to VRAM).
  * Fix: `--threads-batch N` (aka `--tb`) - threads llama-server uses for prompt/batch work. Tune to your CPU's physical core count.
  * Fix: `--prio-batch 0..3` - OS-level thread priority for batch work. Higher values are more aggressive.

* **I want multiple concurrent sessions on one server.**
  * Fix: `--parallel N` (aka `--np`) - number of llama-server slots. Default is 1.
  * Impact: Each extra slot roughly multiplies context VRAM usage by N.

### KV cache

* **My model uses sliding-window attention and I want to force it on or off.**
  * Fix: `--swa-full` to force-enable, `--no-swa-full` to force-disable (this also disables lui's auto-detection). Without either, lui auto-detects.

* **Prompt caching across requests would be great.**
  * Fix: `--cache-ram MIB` - llama-server's host-memory prompt cache. Useful for agentic workloads that re-run similar prompts.

### Server / networking

* **Port 8080 is taken / I want a different port.**
  * Fix: `--port N`. Machine-wide setting; can't be scoped to `--this`.

* **I want another machine on my LAN to talk to this server.**
  * Fix: `--public` binds to `0.0.0.0` instead of `127.0.0.1`.
  * Impact: Anyone on the network can hit the endpoint. No auth. Don't do this on hostile networks.

* **Web search is getting in my way / I don't want lui's skill in opencode.**
  * Fix: `--no-websearch` disables lui's in-process websearch server and removes its opencode skill. Re-enable with `--websearch`.
  * Fix: `--web-port N` chooses a specific port for the websearch endpoint (default: llama port + 1).

* **I want to SSH into another machine and let that machine use my local lui instance!**
  * Fix: `lui --ssh USER@HOST` on the server - writes an opencode config on the client and prints the matching `ssh -R ...` command. See the "Connecting to a Shared Server" section above.

* **I want to use a another `--public` lui server from my machine, but keep web search running locally.**
  * Fix: `lui --remote HOST[:PORT]` on the client. It fetches `/config`, writes a local opencode config pointing at the server, and runs a local websearch server.

### Debug / introspection

* **How do I know if the right settings are being passed to `llama-server`?**
  * Fix: `--cmd` - prints the fully-resolved llama-server command line (with all scope merging, defaults, and auto-detected SWA applied) and exits. Copy-pasteable.

* **llama-server is failing at startup and I can't see why.**
  * Fix: `--debug PATH` - dumps raw llama-server stdout/stderr to a file for inspection.

* **I want to pass a llama-server flag lui doesn't expose.**
  * Fix: Everything after `--` is appended to llama-server's argv. `lui -- --flash-attn 1 --something-else`.
  * Fix: Scope applies here too. `lui --this -- --some-flag` appends to the active model's `extra_args` instead of the global list.
  * Note: Passing _any_ extras for a given scope replaces the stored list for that scope. To clear pass-through args, re-run with an empty `--` tail under the right scope.

* **Where is my config stored?**
  * Fix: lui writes a toml at its standard config path (see `lui -l` or inspect the path logic in `src/config.rs`). Everything you pass on the command line is persisted there - next run, plain `lui` picks up where you left off.

### Sandbox

* **I want to run my agent isolated from the rest of my machine.**
  * Fix: `lui --sandbox opencode` (or `pi`, `late`). Wraps the harness
    in [nono](https://nono.sh) with sensible defaults; project tree
    and common toolchains (cargo, rustc, go, npm, bun, pyenv...) are
    r+w, credentials and shell history are blocked. See
    "Sandboxing the Harness" above for the full story.

* **The sandboxed harness can't access `~/.foo`.**
  * Fix: `lui --sandbox-allow ~/.foo` (or `--sandbox-read` for
    read-only). Repeatable, persists into `[sandbox] allow` in
    `lui.toml`.
  * Tip: nono prints a `Fix: --read … --read …` hint when something
    is denied. Translate to `--sandbox-read DIR` / `--sandbox-allow DIR`
    and re-run.

* **The harness needs network access to a private domain.**
  * Fix: `lui --sandbox-allow-domain api.internal.tld` (repeatable).

### Quick recipes

* **Fresh start on a new model, tuned and aliased in one line:**
  `lui --hf org/SomeModel-GGUF:Q4_K_M --alias foo --this -c 131072 --temp 0.6 --top-p 0.95`

* **Shrink VRAM without rewriting everything:**
  `lui --this --fit-target 2048 --ctk q4_0 --ctv q4_0`

* **See exactly what will run, without running it:**
  `lui --cmd`

* **Clear a per-model temperature override:**
  `lui --this --temp default`

* **Launch opencode in a sandbox over the project:**
  `lui --sandbox opencode`

* **Always sandbox opencode (alias trick):**
  `alias opencode='lui --sandbox opencode'` in your shell rc. Plain
  `opencode` then runs sandboxed; `command opencode` bypasses.

## License

BSD-2-Clause. See [LICENSE](LICENSE).
