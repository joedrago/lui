# llama-server CPU RAM budget on hybrid SSM/attention models

How host memory is consumed by `llama-server` (build 9203+) when running a
hybrid-architecture model like Qwen3.6-27B-MTP, and how the relevant flags
trade against each other. All numbers below are measured against Qwen3.6-27B
at q5_K with q8_0 K / q4_0 V cache and `-c 256144`; the structure of each
component generalizes, the absolute sizes do not.

## Summary

For a config of `-np 4 -ctxcp 2 -cram 8192`:

| Item                          | Worst case  | What drives it                                |
| ----------------------------- | ----------- | --------------------------------------------- |
| Process + libraries           | ~1.2 GiB    | binary + ggml/Vulkan/ROCm shared libs (fixed) |
| `CPU_Mapped` model buffer     | ~0.83 GiB   | `token_embd.weight` (file-backed mmap)        |
| `Vulkan_Host` compute scratch | ~0.52 GiB   | pinned host staging for GPU transfers         |
| Context checkpoints           | ~4.8 GiB    | `-ctxcp × -np × ~600 MiB SSM snapshot`        |
| Prompt cache                  | ~8 GiB      | capped by `-cram`                             |
| Per-request scratch / noise   | ~0.5 GiB    | sampler, HTTP, glibc overhead                 |
| **Total ceiling**             | **~16 GiB** |                                               |

Typical operation lands at 4–10 GiB. The ~16 GiB ceiling is what you'd hit
only when every axis saturates at once (all 4 slots hold maxed-out
checkpoints, the prompt cache is full of long saved states).

## Components

### Fixed costs (~2.5 GiB)

Present from the moment llama-server is ready, regardless of usage.

- **Process and libraries** (~1.2 GiB): binary plus shared libraries
  (libLLVM, libggml-vulkan, libamd_comgr, librocsolver, libcrypto, …).
- **`CPU_Mapped` model buffer** (~830 MiB for q5_K Qwen3.6-27B):
  `token_embd.weight` fell back to plain CPU because the `Vulkan_Host`
  preferred buffer type rejected its quantization (search the load log
  for `cannot be used with preferred buffer type Vulkan_Host`). It's a
  shared file-backed mmap from the GGUF, so the kernel can drop pages
  under pressure — it shows in RSS but doesn't count toward the
  OOM-killer's `anon-rss` the way the rest of these do.
- **`Vulkan_Host` compute scratch** (~520 MiB at default `-ub 512`):
  pinned host buffers for staging tensor data on transfers. Scales with
  `-ub`; cutting `-ub` to 128 drops this to ~130 MiB.

### Context checkpoints (`-ctxcp N`)

Defined at `tools/server/server-context.cpp:1864` (`create_checkpoint`) and
`common/common.h:1052` (`common_prompt_checkpoint`). Each checkpoint
serializes the recurrent (SSM/Mamba) state of one slot into a host
`std::vector<uint8_t>` via
`llama_state_seq_get_data_ext(..., PARTIAL_ONLY)`. The `PARTIAL_ONLY` flag
(see `src/llama-memory-hybrid.cpp:181`) **skips the attention KV** — that
lives on the GPU and is rewindable by position alone.

- **Per-checkpoint size**: ~600 MiB for Qwen3.6-27B at this `-c`. Dominated
  by the SSM state for the slot's sequence; constant-size regardless of
  how many tokens have actually been processed (recurrent state per layer
  is fixed-size).
- **Per slot**: up to `N` checkpoints, evicted oldest-first when full.
- **Default upstream value is 32** — on a hybrid model with 256K context
  that produces a ~28 GiB host ceiling. That default is the footgun.
- **Why they exist**: for hybrid SSM/attention models, the recurrent
  state can't be reconstructed from KV alone (it's a function of every
  preceding token, and SSMs aren't randomly accessible). Checkpoints let
  a slot rewind to a known prior position without reprocessing from token 0. With `-ctxcp 2 -cpent 8192` (the default checkpoint interval) you
  can recover edits made up to ~16K tokens behind the conversation head;
  beyond that the slot falls back to full re-prefill (see
  `tools/server/server-context.cpp:2666`, the `do_reset` path).
- **Trade-off**: more checkpoints = wider rewind range, more host RAM.
  `-ctxcp 2` is enough for opencode-style "always append forward"
  workloads.

### Prompt cache (`-cram N` MiB)

Defined at `tools/server/server-task.cpp:1997`
(`server_prompt_cache::alloc`). Stores **full** slot states (attention KV

- SSM, via `FLAGS_NONE` — not `PARTIAL_ONLY` like checkpoints) so a slot
  can be swapped to a different conversation and swapped back without a
  re-prefill.

* **Per-entry size**: `actual_tokens × ~25 KiB` (q8_0 K + q4_0 V KV per
  token on this model) + ~600 MiB SSM. So a 50K-token saved state is
  ~1.85 GiB; a 256K save is ~7 GiB.
* **Cap**: `-cram` in MiB. Entries get evicted oldest-first when the
  cache total exceeds the cap, with a hard **"always keep at least one
  entry"** floor (`tools/server/server-task.cpp:2130`,
  `while (states.size() > 1 && size() > limit_size)`). So a single huge
  saved state can briefly exceed the cap if it's the only entry.
* **When it's used**: any time a slot is reassigned to a task whose
  prompt prefix differs significantly from the slot's current state.
  Heavy when a harness like opencode spawns subagents on a different
  context than the main conversation.
* **Trade-off**: small cap → more re-prefills on subagent rotation; large
  cap → more RSS. 8 GiB holds one long main conversation plus a few
  subagents.

### Per-request scratch (~0.5 GiB)

Sampler state, grammar machinery, token vectors, HTTP buffers, glibc
allocator pages held above the trim threshold. Doesn't grow unboundedly
with use, but can sit at 100–500 MiB during sustained activity.

## Interactions with `-c` (context size)

`-c` is the **per-slot maximum token capacity** when `-kvu` is set. It
sizes the unified KV buffer once at startup. Every memory item above
either scales directly with `-c` or scales with what slots actually
contain (which is itself bounded by `-c`).

- **`-c` inflates the unified KV buffer** (sized for `-c` tokens of KV
  cache at the configured quantization, on Vulkan0). This is VRAM, not
  host RAM — but it's why `-c` can't be arbitrary, VRAM stops you first.
- **`-c` does NOT inflate checkpoints directly.** Each checkpoint is
  SSM state, which is constant-size regardless of token count.
  `-c 256K` and `-c 32K` produce same-size checkpoints.
- **`-c` does inflate prompt-cache entries**, indirectly — the saved
  state includes the slot's full attention KV, which is proportional to
  the slot's actual token count (capped by `-c`). A 200K-token saved
  state is ~5.6 GiB; the same conversation under `-c 64K` is capped
  smaller.
- **`-c` mildly inflates per-request scratch peaks**, because some
  buffers are sized for worst-case prefill up to `-c`.

Dropping `-c` saves a lot of VRAM and slightly trims host scratch peaks;
it does **not** help with the checkpoint or cache axes that dominate
host RAM in practice. Reach for `-ctxcp` and `-cram` first.

## Interactions with `-kvu` (unified KV)

From `src/llama-context.cpp:202-205`:

```cpp
if (cparams.kv_unified) {
    cparams.n_ctx_seq = cparams.n_ctx;          // each slot can use full -c
} else {
    cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;  // split among slots
}
```

- **With `-kvu`** (recommended): one shared KV buffer sized for `-c`
  total tokens. Every slot can grow up to `-c` independently. Slots
  compete for the unified budget when active simultaneously. Lets you
  set `-np 4 -c 256144` without quartering each slot's effective
  context. No extra VRAM compared to `-np 1 -c 256144`.
- **Without `-kvu`**: each slot gets `-c / -np` context. To preserve
  per-slot context size when increasing `-np`, you'd have to multiply
  `-c` by `-np` — and pay the full `-np × -c` cost in VRAM.
- **Required for `cache-idle-slots`**: the prompt cache's save-on-idle
  path only enables when `kv_unified` is set, per
  `tools/server/server-context.cpp:1028`.

`-kvu` itself doesn't change host RAM, but it makes `-np > 1` cheap,
which then determines the multiplier on the checkpoint axis
(`-np × -ctxcp × ~600 MiB`).

## Flag quick reference

| Flag       | Default    | Effect on host RAM                                                                             |
| ---------- | ---------- | ---------------------------------------------------------------------------------------------- |
| `-c N`     | 4096       | mostly VRAM; mild host scratch growth                                                          |
| `-np N`    | auto (1)   | multiplies checkpoint cost; gates concurrent slot saves                                        |
| `-kvu`     | auto       | no direct host effect; makes `-np > 1` viable                                                  |
| `-ctxcp N` | **32**     | `N × -np × ~600 MiB` worst-case checkpoint cost                                                |
| `-cpent N` | 8192       | checkpoint-creation interval in tokens; higher = fewer checkpoints created during long prefill |
| `-cram N`  | 8192 (MiB) | hard cap on prompt-cache budget (with "keep ≥ 1" floor)                                        |
| `-b N`     | 2048       | logical batch; minor host effect                                                               |
| `-ub N`    | 512        | ubatch size; `Vulkan_Host` compute scratch scales linearly                                     |
| `-fa on`   | off        | flash attention; small host-side effect                                                        |

## Typical operating ranges

For `-c 256144 -np 4 -ctxcp 2 -cram 8192` on a 32 GiB host:

| Workload                               | Expected RSS |
| -------------------------------------- | ------------ |
| Just booted, idle                      | 2.5–3 GiB    |
| One active conversation, light editing | 4–6 GiB      |
| Long conversation (100K+ tokens)       | 6–9 GiB      |
| Heavy multi-subagent rotation          | 10–14 GiB    |
| Pathological worst case                | ~16 GiB      |

## See also

- `tools/server/server-context.cpp:1864` — `create_checkpoint`
- `tools/server/server-context.cpp:2666` — `do_reset` (full re-prefill fallback)
- `tools/server/server-task.cpp:1997` — `server_prompt_cache::alloc`
- `common/common.h:1052` — `common_prompt_checkpoint` struct
- `common/common.h:596` — `cache_ram_mib` default
- `common/arg.cpp:1328` — `--ctx-checkpoints` flag definition
- `src/llama-context.cpp:202` — `kv_unified` logic
- `src/llama-memory-hybrid.cpp:181` — `PARTIAL_ONLY` semantics
- llama.cpp PR #15293 — context checkpoints introduction
- llama.cpp PR #16391 — prompt cache budget mechanism
