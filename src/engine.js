// Engine runner + registry. Each engine module exports an `engine`
// object (see REWRITE.md §8); this file imports them, exposes the
// `engines` map keyed by `engine.name`, and provides `runEngine()`
// which spawns the child and pipes its stdout through the engine's
// `parseLine` line-by-line.
//
// STYLE.SEGMENT_* are shared PaletteEntry conventions so every engine
// paints its argv segments with a coherent look. Engines are free to
// use plain inline PaletteEntry objects instead.

import { spawn } from "node:child_process"
import fs from "node:fs"
import path from "node:path"

import { engine as llamaServer } from "./engine/llama-server.js"

export const STYLE = {
    SEGMENT_BINDING: { fg: "cyan" },
    SEGMENT_POLICY: { dim: true },
    SEGMENT_DEFAULTS: { fg: [100, 170, 200] },
    SEGMENT_USER: {}
}

export const engines = {
    [llamaServer.name]: llamaServer
}

// Resolve the engine binary path. Order:
//   1. [engine.<name>].binary, if set (absolute or relative)
//   2. engine.defaultBinary on $PATH
function resolveBinary(lui, engineModule, binaryHint) {
    const override = lui.config.engine?.[engineModule.name]?.binary
    const candidate = override || binaryHint || engineModule.defaultBinary

    if (candidate.includes(path.sep) || candidate.startsWith(".")) {
        return candidate
    }

    // PATH lookup
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

// Spawn the engine child, wire stdout/stderr → parseLine, tee raw output
// to lui.debugLogPath if set. Returns the ChildProcess.
export function runEngine(lui, binary, segments) {
    const argv = segments.flatMap((s) => s.args)
    const bin = resolveBinary(lui, lui.engineModule, binary)

    let debugFd = null
    if (lui.debugLogPath) {
        try {
            debugFd = fs.openSync(lui.debugLogPath, "w")
        } catch (e) {
            lui.addWarning(`--debug: cannot open ${lui.debugLogPath}: ${e.message}`)
        }
    }

    const child = spawn(bin, argv, { stdio: ["ignore", "pipe", "pipe"] })

    const buffers = { stdout: "", stderr: "" }
    function drain(name, chunk) {
        if (debugFd != null) {
            try {
                fs.writeSync(debugFd, chunk)
            } catch {
                // ignore
            }
        }
        buffers[name] += chunk
        let idx
        while ((idx = buffers[name].indexOf("\n")) !== -1) {
            const line = buffers[name].slice(0, idx).replace(/\r$/, "")
            buffers[name] = buffers[name].slice(idx + 1)
            try {
                lui.engineModule.parseLine?.(line, lui)
            } catch (e) {
                process.stderr.write(`lui: parseLine threw: ${e?.stack || e}\n`)
            }
        }
    }

    child.stdout.setEncoding("utf8")
    child.stderr.setEncoding("utf8")
    child.stdout.on("data", (c) => drain("stdout", c))
    child.stderr.on("data", (c) => drain("stderr", c))

    child.on("exit", (code, signal) => {
        if (debugFd != null) {
            try {
                fs.closeSync(debugFd)
            } catch {
                // ignore
            }
        }
        lui.onEngineExit?.(code, signal)
    })

    child.on("error", (err) => {
        process.stderr.write(`lui: failed to spawn ${bin}: ${err.message}\n`)
        lui.onEngineExit?.(1, null)
    })

    return child
}
