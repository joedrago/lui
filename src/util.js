// Low-level catchall. Small, generic helpers with no upward deps.

import fs from "node:fs"
import os from "node:os"
import path from "node:path"
import process from "node:process"

export function expandTilde(p) {
    if (p.startsWith("~/")) return path.join(os.homedir(), p.slice(2))
    if (p === "~") return os.homedir()
    return p
}

// [engine.<name>].binary if set, else `binaryHint` (from buildArgv),
// else engine.defaultBinary — looked up on $PATH if it's a bare name.
//
// Engines reach for this when they need a path to spawn anything
// (version probes, sidecar tools). Engines that don't run a binary
// at all don't need to call this.
export function resolveBinary(lui, engineModule, binaryHint) {
    const override = lui.config.engine?.[engineModule.name]?.binary
    const candidate = override || binaryHint || engineModule.defaultBinary
    if (!candidate) return null

    if (candidate.includes(path.sep) || candidate.startsWith(".")) {
        return candidate
    }

    const pathDirs = (process.env.PATH || "").split(path.delimiter)
    for (const dir of pathDirs) {
        if (!dir) continue
        const full = path.join(dir, candidate)
        try {
            fs.accessSync(full, fs.constants.X_OK)
            return full
        } catch {
            // not here; try next
        }
    }
    return candidate
}
