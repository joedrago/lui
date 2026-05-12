// Engine registry. Each engine module exports `engine`. Per-segment
// palette conventions live in src/theme.js (STYLE.SEGMENT_*).
//
// Subprocess engines (llama-server, mlx_lm, future vllm) use the
// shared helper in src/spawn.js for PATH lookup, line buffering, and
// the SIGTERM dance. The framework itself doesn't spawn anything.

/** @import { Engine, SchemaEntry } from "./types.js" */

import { engine as llamaServer } from "./engine/llama-server.js"
import { engine as mlxLm } from "./engine/mlx_lm.js"
import { engine as remote } from "./engine/remote.js"

/** @type {Record<string, Engine>} */
export const engines = {
    [llamaServer.name]: llamaServer,
    [mlxLm.name]: mlxLm,
    [remote.name]: remote
}

// Each engine's own `schema` entries, prefixed with `engine.<name>.`
// and surfaced by `lui config` under "Available Settings".
/** @returns {SchemaEntry[]} */
export function engineSchemaDefaults() {
    const out = []
    for (const e of Object.values(engines)) {
        for (const s of e.schema ?? []) {
            out.push({ ...s, path: `engine.${e.name}.${s.path}` })
        }
    }
    return out
}
