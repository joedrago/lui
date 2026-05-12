// Engine registry + runEngine(). Each engine module exports `engine`.
// Per-segment palette conventions live in src/theme.js (STYLE.SEGMENT_*).

import { spawn } from "node:child_process"
import fs from "node:fs"

import { engine as llamaServer } from "./engine/llama-server.js"
import { resolveBinary } from "./util.js"

export const engines = {
    [llamaServer.name]: llamaServer
}

// Each engine's own `schema` entries, prefixed with `engine.<name>.`
// and surfaced by `lui config` under "Available Settings".
export function engineSchemaDefaults() {
    const out = []
    for (const e of Object.values(engines)) {
        for (const s of e.schema ?? []) {
            out.push({ ...s, path: `engine.${e.name}.${s.path}` })
        }
    }
    return out
}

// Spawn the engine; line-pipe stdout/stderr to parseLine; tee raw to
// [global].debug_log if set.
export function runEngine(lui, binary, segments) {
    const argv = segments.flatMap((s) => s.args)
    const bin = resolveBinary(lui, lui.engineModule, binary)

    let debugFd = null
    const debugPath = lui.config.global?.debug_log
    if (debugPath) {
        try {
            debugFd = fs.openSync(debugPath, "w")
        } catch (e) {
            lui.addWarning(`debug_log: cannot open ${debugPath}: ${e.message}`)
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
