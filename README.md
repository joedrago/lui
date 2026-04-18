# lui

A friendly TUI wrapper for [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`. Pronounced **"Louie"** — short for *llama.cpp ui*.

## Setup

1. **Install opencode** — <https://opencode.ai>. No config needed; lui will wire it up for you.
2. **Put `llama-server` on your PATH.**
   - **macOS:** `brew install llama.cpp`
   - **Windows:** grab the matching `llama-bin-win-*` zip **and** the `cudart-llama-bin-win-cuda-*` zip (for NVIDIA) from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases), extract them into the same folder, and add that folder to your `PATH`.
3. **Build lui** — `cargo build --release`, then put `target/release/lui` on your PATH (symlink, copy, or add the directory).

## Run

Example first runs:

```
lui --hf unsloth/GLM-4.7-Flash-GGUF:Q4_K_M
lui --hf unsloth/GLM-4.7-Flash-GGUF:Q4_K_M --alias glm --this -c 131072

lui --hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M
lui --hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M --this -c 262144
lui --hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M --alias qwen --this -c 262144 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00

lui --hf "unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M" --alias gemma --this --temp 1.0 --top-p 0.95 --top-k 64
```

lui will show download progress bars, start `llama-server`, configure opencode, and print **Ready** once the model is loaded.

Every flag you pass is saved to a toml config, so next time just run:

```
lui
```

Other useful flags: `-m <path>` for a local GGUF, `--ngl <n>` for GPU layers, `--port <n>`, `--public` to bind `0.0.0.0`, `-l` to list cached models.

## Connecting to a Shared Server

lui supports two machines connecting to share a model. The two flags are mutually exclusive (`--ssh` and `--remote` can't be used together) and neither is persisted to `lui.toml`.

### `--ssh` — share your local LLM with a remote machine

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
- SSH access to the client (`USER@HOST` format required — no bare hostnames).
- The client should be able to reach this machine's network interface for the reverse tunnel to work.

**Example:**

```
# On the server machine:
lui --ssh user@workstation

# Then in another terminal on the server:
ssh -R 23847:localhost:8080 -R 23848:localhost:8081 user@workstation
```

### `--remote` — use a remote LLM while keeping web search local

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
- The server's llama-server port will be read from its `/config` — it doesn't have to match the HTTP port.

**Example:**

```
# On your client machine (server must be running with --public):
lui --remote server.local
lui --remote server.local:9000    # custom HTTP port
```

## Quit

`Ctrl+C` or `q`.

## License

BSD-2-Clause. See [LICENSE](LICENSE).
