# lui

A friendly TUI wrapper for [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`. Pronounced **"Louie"** — short for *llm ui*.

## Setup

1. **Install opencode** — <https://opencode.ai>. No config needed; lui will wire it up for you.
2. **Put `llama-server` on your PATH.**
   - **macOS:** `brew install llama.cpp`
   - **Windows:** grab the matching `llama-bin-win-*` zip **and** the `cudart-llama-bin-win-cuda-*` zip (for NVIDIA) from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases), extract them into the same folder, and add that folder to your `PATH`.
   - **Linux:** grab the matching `llama-bin-ubuntu-*` tarball from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) — `*-rocm-*` for AMD GPUs, `*-vulkan-*` for any other GPU — extract it, and add the folder to your `PATH`.
   - Verify: run `llama-server --version` and make sure it detects your GPU with no serious errors.
3. **Install lui** — requires Node 20+.

   ```
   npm install -g github:joedrago/lui
   ```

   That fetches the repo, installs deps, and drops `lui` on your `PATH`.
   Re-run the same command to update.

   Or, working from a checkout:

   ```
   git clone https://github.com/joedrago/lui && cd lui && npm install
   # then put `bin/lui` on your PATH (symlink, copy, or add <repo>/bin to PATH)
   ```

## Run

Register a model once with `lui add NAME ENGINE -- ARGS...`, then run it with `lui run NAME` (or just `lui` to resume the last one). Everything after `--` is the opaque argv lui passes to `llama-server`.

```
lui add qwenmoe llama-server -- -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M -c 262144 -ctk q8_0 -ctv q8_0 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00 --presence-penalty 1.5
lui run qwenmoe
```

lui shows download progress bars, starts `llama-server`, configures opencode, and prints **Ready** once the model is loaded. Subsequent runs are just `lui`.

To list, inspect, or remove models:

```
lui ls                  # one-line summary per model
lui show qwenmoe        # full resolved config block
lui cmd qwenmoe         # exact llama-server command lui would spawn
lui set qwenmoe -- ...  # replace this model's args (creates it if absent)
lui rm qwenmoe          # delete the entry
```

The TUI quits on `q` or `Ctrl+C`.

## Persistent config

Everything except a few run-time-only flags lives in `~/.config/lui.toml`, tunable with `lui config`:

```
lui config set engine_port 8080
lui config set websearch false
lui config set harness.opencode.enabled true
lui config set engine.llama-server.binary /usr/local/bin/llama-server
lui config set sandbox.allow_gpu true
lui config add sandbox.allow ./project
lui config clear sandbox.profile
```

Paths are dot-separated. A path that doesn't name a top-level table (`global`, `model`, `harness`, `engine`, `sandbox`) is automatically rooted under `global.`. See `lui --help` for a comprehensive list of every setting × verb.

The handful of flags that *aren't* persisted (override for one run only):

| flag | effect |
| --- | --- |
| `--debug PATH` | tee raw `llama-server` stdout to `PATH` |
| `--engine-port N` | override `engine_port` for this run |
| `--web-port N` | override `web_port` for this run |
| `--public` | bind `0.0.0.0` instead of `127.0.0.1` |

## Connecting to a shared server

lui supports two machines sharing one model. The two modes are mutually exclusive; neither is persisted.

### `lui ssh USER@HOST` — share your local LLM with a remote machine

Run **on the server** (the machine where `llama-server` is running). It SSHes into the client, writes an opencode config there pointing back at your llama-server through a reverse tunnel, and prints the `ssh -R ...` command for you to run in another terminal.

**What it does:**

1. Verifies `opencode` is installed on the client.
2. Picks a random high port on the client (18000–28999) for llama-server and the next port for websearch.
3. Writes `~/.config/opencode/opencode.json` on the client with `baseURL` pointing to `http://localhost:<client_port>/v1`.
4. Writes the `lui-web-search` SKILL.md on the client (unless websearch is disabled), baked with the correct client ports.
5. Prints the `ssh -R …` command. Run that in another terminal to establish the tunnel.

**Requirements:**

- `opencode` must already be installed on the client (lui probes the default PATH, login-shell PATH, and `~/.opencode/bin/opencode`).
- SSH access in `USER@HOST` form — no bare hostnames.

**Example:**

```
# On the server machine:
lui ssh user@workstation

# Then, in another terminal on the server:
ssh -R 23847:localhost:8080 -R 23848:localhost:8081 user@workstation
```

### `lui remote HOST[:PORT]` — use a remote LLM while keeping web search local

Run **on your client** (your laptop). It connects to a `--public` lui server, writes a local opencode config pointing at it, spawns an in-process websearch server here so browser-mediated search works against your real browser, and renders the TUI by polling the remote `/data`.

**What it does:**

1. Fetches `/config` from the server's HTTP endpoint (default port 8081).
2. Validates the config version matches.
3. Writes local `~/.config/opencode/opencode.json` with `baseURL` pointing directly at the server's llama-server (e.g. `http://server:8080/v1`).
4. Writes the local `lui-web-search` SKILL.md pointed at a bsearch server spawned on this client.
5. Renders the TUI; blocks until `Ctrl+C`.

**Requirements:**

- The server must be running with `lui --public` (or `lui config set public true`) so its HTTP server binds to `0.0.0.0`.
- Network access from client to server.

**Example:**

```
# On your client machine (server already running with --public):
lui remote server.local
lui remote server.local:9000    # custom HTTP port
```

## Sandboxing the harness

`lui sandbox HARNESS [ARGS...]` launches the harness wrapped in [nono](https://nono.sh) — a capability-based sandbox using Seatbelt on macOS and Landlock on Linux. The harness can read+write your project, hit the network, and invoke local toolchains, but **cannot** touch credentials, browser data, shell history, or anything outside the allow-list. lui is just a thin launcher in this mode — no TUI, no llama-server — and it propagates the child's exit code.

**Everything after `HARNESS` is passed verbatim to the harness**, including `--` and `--help`. That's the trick that makes shell aliases work cleanly:

```sh
# ~/.zshrc, ~/.bashrc, etc.
alias opencode='lui sandbox opencode'
```

Then plain `opencode --foo bar` runs sandboxed; `command opencode` bypasses.

### Setup

1. Install nono — <https://nono.sh>. Single binary; the installer drops it on your `PATH`.
2. That's it. Defaults are tuned for "drop into a project and run an AI agent."

### Defaults at a glance

```
lui sandbox opencode
```

resolves to roughly:

```
nono run -p opencode --allow . --allow-cwd \
    --allow ~/.cargo --allow ~/.rustup --allow ~/go --allow ~/.pyenv \
    --allow ~/.npm --allow ~/.bun --allow /usr/local/go ... \
    -- opencode
```

What that gets you:

- **Profile auto-detected.** If nono ships a profile by the harness's name (`opencode`, `claude-code`, `codex`, …), lui uses it. Otherwise it falls back to nono's `default` profile (gives `/tmp`, `/usr/bin`, homebrew, plus deny-rules for credentials, keychains, browser data, shell history).
- **Project tree r+w** via `--allow .` and `--allow-cwd` (skips nono's first-run prompt).
- **Toolchains r+w** for any of `~/.cargo`, `~/.rustup`, `~/go`, `/usr/local/go`, `~/.pyenv`, `~/.local/share/uv`, `~/.conda`, `~/.nvm`, `~/.fnm`, `~/.npm`, `~/.bun`, `~/.deno`, `~/Library/pnpm`, `/usr/local/lib/node_modules`, `~/.nix-profile`, `/nix/store`, etc. that exist on your machine. `$CARGO_HOME` / `$RUSTUP_HOME` / `$GOPATH` / `$PYENV_ROOT` override the defaults.
- **GPU off**, **network on**. Node-based agents don't need GPU; lui's llama-server runs *outside* the sandbox.

### Tuning the sandbox

Everything lives under `[sandbox]` in `lui.toml`:

```
lui config set sandbox.allow_cwd true        # default; r+w on cwd
lui config set sandbox.allow_gpu false       # default; flip on for GPU tools
lui config set sandbox.block_net true        # tighter sandbox: no network
lui config set sandbox.dev_tools true        # default; auto-allow toolchains
lui config set sandbox.profile mycustom      # override profile auto-detect
lui config set sandbox.profile none          # opt out of -p entirely
lui config set sandbox.bin /opt/nono/bin/nono

# Repeatable string arrays:
lui config add sandbox.allow ~/.foo          # extra r+w directory
lui config add sandbox.read /etc             # read-only directory
lui config add sandbox.write /tmp/out        # write-only directory
lui config add sandbox.allow_domain api.example.com
lui config add sandbox.extra --some-other-nono-flag
lui config clear sandbox.allow               # wipe the list
```

When nono blocks something, its denial output prints a `Fix: --read … --read …` hint. Translate each `--read` to `lui config add sandbox.read …` (and `--allow` to `sandbox.allow`) and re-run.

### Authoring a custom nono profile (optional)

If a harness needs a recurring set of grants that don't fit lui's flags cleanly, nono's own `nono profile init` builds a starter:

```
nono profile init my-team --extends opencode --groups rust_runtime,python_runtime
```

Then point lui at it:

```
lui config set sandbox.profile my-team
lui sandbox opencode
```

## License

BSD-2-Clause. See [LICENSE](LICENSE).
