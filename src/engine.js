// Engine registry. Each engine module exports `engine` (see the
// Engine type in types.js for the contract). Subprocess-backed
// engines share src/spawn.js for PATH lookup, line buffering, and
// the SIGTERM dance; the framework itself never spawns anything.
// Per-segment palette conventions live in src/theme.js (STYLE.SEGMENT_*).

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
// and surfaced in the config dump.
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
