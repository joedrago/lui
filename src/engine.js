// Engine registry + runEngine(). Each engine module exports `engine`.
// Per-segment palette conventions live in src/theme.js (STYLE.SEGMENT_*).

import { spawn } from "node:child_process"
import fs from "node:fs"
import path from "node:path"

import { engine as llamaServer } from "./engine/llama-server.js"

export const engines = {
    [llamaServer.name]: llamaServer
}

// One default-binary path per engine, shown under "Available Settings".
// The default is "look up engine.defaultBinary on $PATH" — that's why
// the display is a hint, not a concrete path.
export function engineSchemaDefaults(engineModules) {
    return engineModules.map((e) => ({
        path: `engine.${e.name}.binary`,
        display: `(PATH: ${e.defaultBinary})`
    }))
}

// [engine.<name>].binary if set, else engine.defaultBinary on $PATH.
function resolveBinary(lui, engineModule, binaryHint) {
    const override = lui.config.engine?.[engineModule.name]?.binary
    const candidate = override || binaryHint || engineModule.defaultBinary

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
