# lui

A friendly TUI wrapper for [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`. Pronounced **"Louie"** — short for *llama.cpp ui*.

## Setup

1. **Install opencode** — <https://opencode.ai>. No config needed; lui will wire it up for you.
2. **Put `llama-server` on your PATH.**
   - **macOS:** `brew install llama.cpp`
   - **Windows:** grab the matching `llama-bin-win-*` zip **and** the `cudart-llama-bin-win-cuda-*` zip (for NVIDIA) from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases), extract them into the same folder, and add that folder to your `PATH`.
3. **Build lui** — `cargo build --release`, then put `target/release/lui` on your PATH (symlink, copy, or add the directory).

## Run

```
lui --hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M -c 131072
```

lui will show download progress bars, start `llama-server`, configure opencode, and print **Ready** once the model is loaded.

Every flag you pass is saved to a toml config, so next time just run:

```
lui
```

Other useful flags: `-m <path>` for a local GGUF, `--ngl <n>` for GPU layers, `--port <n>`, `--public` to bind `0.0.0.0`, `-l` to list cached models.

## Quit

`Ctrl+C` or `q`.

## License

BSD-2-Clause. See [LICENSE](LICENSE).
