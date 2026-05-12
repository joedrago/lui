// Subprocess plumbing shared by spawn-based engines (llama-server,
// mlx-lm.server, vllm, …). Owns PATH lookup, stdio piping, line
// buffering, debug-log tee, and the SIGTERM → grace → SIGKILL dance.
//
// Engine modules call spawnProcess() from inside their start(); they
// keep ownership of argv composition, parseLine logic, state schema,
// and panel rendering. Only the boilerplate lives here.

/** @import { SpawnSpec, SpawnHandle } from "./types.js" */

import { spawn } from "node:child_process"
import fs from "node:fs"
import path from "node:path"
import process from "node:process"

// PATH lookup for a bare command name. An override with a separator
// (absolute path or `./relative`) is returned as-is. Returns the
// candidate unchanged if it isn't found; spawn will surface the
// resulting ENOENT through onExit.
/** @param {string | null | undefined} candidate @returns {string | null} */
export function resolveBinary(candidate) {
    if (!candidate) return null
    if (candidate.includes(path.sep) || candidate.startsWith(".")) return candidate
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

const KILL_GRACE_MS = 5000

// Human-readable explanation of a spawn-time failure. ENOENT and
// EACCES are the two errors the user can fix; everything else falls
// through to the system message.
/** @param {string} binaryName @param {NodeJS.ErrnoException | null | undefined} err @returns {string} */
export function describeSpawnError(binaryName, err) {
    if (err?.code === "ENOENT") return `cannot find ${binaryName} on PATH`
    if (err?.code === "EACCES") return `${binaryName} is not executable`
    return `failed to spawn ${binaryName}: ${err?.message ?? err}`
}

// Spawn a subprocess; line-pipe stdout/stderr to parseLine; tee raw
// bytes to `debugLog` if set. Returns { child, stop } where stop()
// runs SIGTERM, waits up to KILL_GRACE_MS for exit, then SIGKILLs.
//
//   binary     — already PATH-resolved (engines call resolveBinary)
//   argv       — flat string array
//   parseLine  — (line: string) => void; called once per \n
//   debugLog   — optional path to tee raw bytes
//   onExit       — optional (code, signal) => void
//   onSpawnError — optional (err) => void, fired when the kernel can't
//                  start the child at all (ENOENT, EACCES…). Called
//                  before onExit so engines can stash a friendly
//                  message into state before the shutdown summary
//                  renders. With no callback, the error is written to
//                  stderr (which the TUI swallows).
//   addWarning   — optional (msg) => void for non-fatal issues
//                  (e.g. cannot open debug log)
//   env          — optional env mapping for the child. Default is
//                  process.env. Engines that need to override TTY-
//                  related vars (e.g. mlx_lm sets TQDM_POSITION=-1 to
//                  un-suppress huggingface_hub's bytes bar in pipe
//                  mode) pass an extended env here.
/** @param {SpawnSpec} spec @returns {SpawnHandle} */
export function spawnProcess({ binary, argv, parseLine, debugLog, onExit, onSpawnError, addWarning, env }) {
    /** @type {number | null} */
    let debugFd = null
    if (debugLog) {
        try {
            debugFd = fs.openSync(debugLog, "w")
        } catch (e) {
            addWarning?.(`debug_log: cannot open ${debugLog}: ${/** @type {Error} */ (e).message}`)
        }
    }

    const child = spawn(binary, argv, { stdio: ["ignore", "pipe", "pipe"], env: env ?? process.env })

    /** @type {{ stdout: string, stderr: string }} */
    const buffers = { stdout: "", stderr: "" }
    /** @param {"stdout" | "stderr"} name @param {string} chunk */
    function drain(name, chunk) {
        if (debugFd != null) {
            try {
                fs.writeSync(debugFd, chunk)
            } catch {
                // ignore
            }
        }
        buffers[name] += chunk
        // Split on both \n and \r so tqdm-style progress bars (which
        // overwrite the same line with \r-only updates between
        // newlines) become a stream of distinct line events instead
        // of one giant concatenation at the final \n. \r\n collapses
        // to a single break so DOS line endings don't double-fire.
        while (true) {
            const nIdx = buffers[name].indexOf("\n")
            const rIdx = buffers[name].indexOf("\r")
            const idx = nIdx === -1 ? rIdx : rIdx === -1 ? nIdx : Math.min(nIdx, rIdx)
            if (idx === -1) break
            const next = buffers[name][idx] === "\r" && buffers[name][idx + 1] === "\n" ? idx + 2 : idx + 1
            const line = buffers[name].slice(0, idx)
            buffers[name] = buffers[name].slice(next)
            try {
                parseLine(line)
            } catch (e) {
                process.stderr.write(`lui: parseLine threw: ${/** @type {any} */ (e)?.stack || e}\n`)
            }
        }
    }

    child.stdout?.setEncoding("utf8")
    child.stderr?.setEncoding("utf8")
    child.stdout?.on("data", (c) => drain("stdout", c))
    child.stderr?.on("data", (c) => drain("stderr", c))

    child.on("exit", (code, signal) => {
        if (debugFd != null) {
            try {
                fs.closeSync(debugFd)
            } catch {
                // ignore
            }
        }
        onExit?.(code, signal)
    })

    child.on("error", (err) => {
        if (onSpawnError) onSpawnError(err)
        else process.stderr.write(`lui: failed to spawn ${binary}: ${err.message}\n`)
        onExit?.(1, null)
    })

    async function stop() {
        if (child.exitCode != null) return
        try {
            child.kill("SIGTERM")
        } catch {
            return
        }
        await /** @type {Promise<void>} */ (
            new Promise((res) => {
                const t = setTimeout(() => {
                    try {
                        child.kill("SIGKILL")
                    } catch {
                        // ignore
                    }
                    res()
                }, KILL_GRACE_MS)
                child.once("exit", () => {
                    clearTimeout(t)
                    res()
                })
            })
        )
    }

    return { child, stop }
}
