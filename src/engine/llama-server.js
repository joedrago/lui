// llama.cpp engine. parseLine + appendPanels ported from
// old/src/server.rs and old/src/display.rs.

/** @import { Engine, Model, ViewBuilder } from "../types.js" */
/** @import { Lui } from "../lui.js" */

import { spawn } from "node:child_process"
import os from "node:os"

import { STYLE } from "../theme.js"
import { stripAnsi } from "../ansi.js"
import { resolveBinary, spawnProcess, describeSpawnError } from "../spawn.js"
import { formatDurationSeconds, formatNumber, formatBytes } from "../util.js"
import { createDownloadTracker } from "../downloads.js"

const BINARY_NAME = "llama-server"

const DIM = { dim: true }
const BOLD = { bold: true }
const TEXT = {}

const POLICY_ARGS = ["--metrics", "--jinja", "--log-colors", "off", "-v", "-fa", "on", "--cache-reuse", "256", "-kvu"]

// Per-flag defaults injected when the user didn't supply any of these
// aliases — so `--gpu-layers 20` suppresses `-ngl -1`.
const DEFAULT_HINTS = [
    { flags: ["-ngl", "--gpu-layers", "--n-gpu-layers"], emit: () => ["-ngl", "-1"] },
    { flags: ["-np", "--parallel"], emit: () => ["-np", "1"] },
    { flags: ["-t", "--threads"], emit: () => ["-t", String(autoThreads())] },
    { flags: ["--chat-template-kwargs"], emit: () => ["--chat-template-kwargs", '{"preserve_thinking":true}'] }
]

/** @returns {number} */
function autoThreads() {
    const n = (os.availableParallelism?.() ?? os.cpus().length) - 2
    return Math.max(1, n)
}

/** @param {string[]} args @param {string[]} flags @returns {boolean} */
function userSuppliedAny(args, flags) {
    return args.some((a) => flags.includes(a))
}

const LOG_RING_SIZE = 200
const MAX_RECENT_REQUESTS = 3

// Flags the engine won't let the user override.
const RESERVED_FLAGS = new Set(["--host", "--port"])

/** @type {Engine} */
export const engine = {
    name: "llama-server",

    // Knobs that show up in the config dump, prefixed by the
    // framework with `engine.<name>.`.
    schema: [{ path: "binary", default: BINARY_NAME }],

    // Curated "lui setup" entries. The wizard offers each as a toggle so
    // a fresh install can leave with a working model. `args` is the
    // argv passed straight to `lui add NAME llama-server ...`. Tune as
    // upstreams / quantizations change — this is a living list, not a
    // policy. Models the user already registered get skipped.
    setupDefaults: [
        {
            name: "qwen",
            // Qwen3.6-35B-A3B UD-Q4_K_M is ~22 GB on disk.
            sizeGiB: 22,
            args: [
                "-hf",
                "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M",
                "-c",
                "262144",
                "-ctk",
                "q8_0",
                "-ctv",
                "q8_0",
                "--temp",
                "0.6",
                "--top-p",
                "0.95",
                "--top-k",
                "20",
                "--min-p",
                "0.0"
            ]
        }
    ],

    // Best known context size. After Ready, state.ctxSize is the
    // authoritative value (lifted from `llama_context: n_ctx = ...`).
    // Before then — or in offline callers like `lui ssh` that have no
    // running engine — fall back to parsing -c / --ctx-size from the
    // model's argv. Returns null if neither is available.
    // Extra lines for `lui` shutdown summary (rendered between Uptime
    // and Reason) plus an optional bright-red abort message.
    shutdownSummary(state) {
        const lines = []
        const logLines = state?.logLines ?? []
        // Include last 5 log lines when the engine exited badly (non-zero
        // code or killed by a signal).
        const exitCode = state?.exitCode
        const exitSignal = state?.exitSignal
        if ((exitCode != null && exitCode !== 0) || exitSignal) {
            const tail = logLines.slice(-5)
            if (tail.length > 0) {
                lines.push({ label: "Last log lines", value: tail.join("\n") })
            }
        }
        return { lines, fatal: state?.fatalReason || null }
    },

    // Human-readable detail for the shutdown's Reason line.
    exitReason(state, code, signal) {
        if (state?.exitMessage) return state.exitMessage
        return signal ? `killed by ${signal}` : `exited with code ${code}`
    },

    contextSize(state, model) {
        if (state && state.ctxSize > 0) return state.ctxSize
        const args = model?.args || []
        for (let i = 0; i < args.length; i++) {
            if ((args[i] === "-c" || args[i] === "--ctx-size") && i + 1 < args.length) {
                const n = parseInt(args[i + 1], 10)
                if (Number.isFinite(n) && n > 0) return n
            }
        }
        return null
    },

    // llama-server's OpenAI endpoint accepts any string as the model
    // id, so this is purely cosmetic — harnesses just show a nicer
    // name to the user. We surface the -hf argument verbatim (org/repo
    // plus :quant tag) so the harness UI displays the real source.
    // Returns null when no -hf is present, letting the framework fall
    // back to the lui alias.
    servedModelName(_state, model) {
        const args = model?.args || []
        for (let i = 0; i < args.length; i++) {
            if (args[i] === "-hf" && i + 1 < args.length) return args[i + 1]
        }
        return null
    },

    // {host, port} naming where to actually reach the model. host=null
    // means "use the caller's context-appropriate fallback" — the HTTP
    // server fills it in from the request hostname; `lui ssh` fills it
    // in with "localhost" (the tunnel's server end). For llama-server
    // the engine binary listens on lui's engine_port, on whichever
    // address the user reached us with, so we always return null host.
    endpoint(lui) {
        return { host: null, port: lui.config.global.engine_port }
    },

    // describe() is how the framework asks an engine "what would your
    // commandline look like for this model?" — used for `lui add`
    // validation, the config dump's commandline rendering, and
    // (internally) by start() to build the spawn argv. Segment 0 is
    // the binary; the rest follow.
    describe(model, lui) {
        const binaryName = binaryNameFromConfig(lui)
        const host = lui.config.global.public ? "0.0.0.0" : "127.0.0.1"
        const port = lui.config.global.engine_port
        const userArgs = Array.isArray(model.args) ? [...model.args] : []

        const errors = []
        for (let i = 0; i < userArgs.length; i++) {
            const tok = userArgs[i]
            if (RESERVED_FLAGS.has(tok)) {
                errors.push(`${tok} is reserved by lui — drop it from this model's args`)
            }
        }

        const defaults = []
        for (const hint of DEFAULT_HINTS) {
            if (!userSuppliedAny(userArgs, hint.flags)) defaults.push(...hint.emit())
        }

        return {
            segments: [
                { name: "binary", args: [binaryName] },
                { name: "binding", style: STYLE.SEGMENT_BINDING, args: ["--host", host, "--port", String(port)] },
                { name: "policy", style: STYLE.SEGMENT_POLICY, args: [...POLICY_ARGS] },
                { name: "defaults", style: STYLE.SEGMENT_DEFAULTS, args: defaults },
                { name: "user", style: STYLE.SEGMENT_USER, args: userArgs }
            ],
            warnings: [],
            errors
        }
    },

    async start(lui, model, desc) {
        // Caller (lui.spawnEngine) already surfaced errors/warnings and
        // hands the same desc back to us — no need to recompute.
        lui.state.argSegments = desc.segments
        const [binarySeg, ...rest] = desc.segments
        const binaryName = binarySeg.args[0]
        const binaryPath = resolveBinary(binaryName) ?? binaryName
        const argv = rest.flatMap((s) => s.args)
        lui.state.proc = spawnProcess({
            binary: binaryPath,
            argv,
            parseLine: (line) => engine.parseLine?.(line, lui),
            debugLog: lui.config.global.debug_log,
            onExit: (code, signal) => lui.onEngineExit?.(code, signal),
            onSpawnError: (err) => {
                const msg = describeSpawnError(binaryName, err)
                lui.state.exitMessage = msg
                lui.state.fatalReason = msg
            },
            addWarning: (m) => lui.addWarning(m)
        })
    },

    async stop(lui) {
        lui.state?.downloads?.stop?.()
        await lui.state?.proc?.stop?.()
    },

    initState(lui) {
        const s = lui.state
        s.startedAt = Date.now()
        s.argSegments = null
        s.proc = null
        s.activeModelName = lui.activeModel?.name ?? ""
        s.modelName = ""
        s.quantization = ""
        s.fileSizeN = ""
        s.fileSizeUnit = ""
        s.fileBpw = ""
        s.modelParamsN = ""
        s.modelParamsUnit = ""
        s.gpuLayersLoaded = 0
        s.totalLayers = 0
        s.overflowLayers = 0
        s.cpuMemMib = 0
        s.cpuRepackMib = 0
        s.cpuComputeMib = 0
        s.gpuMemMib = 0
        s.kvCacheMib = 0
        s.computeBufMib = 0
        s.unifiedMemory = false
        s.cpuForcedCount = 0
        s.cpuForcedPrimary = ""
        s.ctxSize = 0
        s.maxCtxSize = 0
        s.nParallel = 1
        s.llamaVersion = ""
        s.updateAvailable = false
        s.listenUrl = ""
        s.activeSlots = new Map()
        s.recentCompleted = []
        s.lastPromptTps = 0
        s.lastGenTps = 0
        s.avgPromptTps = 0
        s.avgGenTps = 0
        s.promptTpsSamples = 0
        s.genTpsSamples = 0
        s.downloads = createDownloadTracker()
        s.logLines = []
        s.exited = false
        s.exitMessage = ""
        s.fitProbing = false
        s.fatalReason = null

        probeVersion(lui).catch(() => {})
    },

    parseLine(rawLine, lui) {
        let line = stripAnsi(rawLine)
        if (!line) return

        // Newer llama-server builds prefix every log line with a
        // timestamp and a single-letter severity, e.g.
        // "0.01.383.150 D common_download_file_single_online: ...".
        // Strip it so the downstream matchers (anchored on ^ or using
        // startsWith) keep working against both old and new builds.
        const ts = /^\d+\.\d+\.\d+\.\d+ [A-Z] /.exec(line)
        if (ts) line = line.slice(ts[0].length)

        // Drop llama-server's own request/response body dumps — they echo
        // arbitrary prompt JSON that has historically clobbered our state
        // accumulators when a user pasted code into the prompt.
        if (line.includes("converted request:")) return
        if (line.includes("log_server_r:") && !line.includes("done request:")) return

        // CUDA graph noise: never reaches the log ring.
        if (/^\s*CUDA Graph id \d+ reused\s*$/.test(line)) return
        if (/^\s*ggml_backend_cuda_graph_compute: CUDA graph warmup (reset|complete)\s*$/.test(line)) return

        // llama-server only prints in-line download progress when stdout
        // is a TTY (common/download.cpp gates on isatty(1)); we pipe
        // stdout, so progress never reaches us. Instead we sniff the
        // "downloading from URL to PATH" log line and let the tracker
        // poll the destination file size and HEAD the URL for total.
        const dl = /^common_download_file_single_online: downloading from (\S+) to (.+?\.downloadInProgress) \(etag:/.exec(line)
        if (dl) {
            lui.state.downloads.add({ url: dl[1], path: dl[2] })
            pushLog(lui.state, line)
            return
        }

        if (!lui.engineReadyFired) parseLoadLine(line, lui)
        else parseRuntimeLine(line, lui)

        pushLog(lui.state, line)
    },

    appendPanels(v, lui) {
        appendEnginePanel(v, lui)
        appendPerformancePanel(v, lui)
        appendServerLogPanel(v, lui)
    }
}

/** @param {any} s @param {string} line */
function pushLog(s, line) {
    if (s.logLines.length >= LOG_RING_SIZE) s.logLines.shift()
    s.logLines.push(line)
}

/** @param {string} line @param {Lui} lui */
function parseLoadLine(line, lui) {
    const s = lui.state

    if (isKvLine(line) && line.includes("general.name") && line.includes("str")) {
        const v = afterEq(line)
        if (v) s.modelName = v
        return
    }
    if (isKvLine(line) && line.includes("general.size_label") && line.includes("str")) {
        const v = afterEq(line)
        if (v) s.sizeLabel = v
        return
    }
    if (line.includes(".context_length") && line.includes("u32")) {
        const v = parseInt(afterEq(line) ?? "", 10)
        if (Number.isFinite(v)) s.maxCtxSize = v
        return
    }
    if (line.includes("print_info: file type")) {
        const v = afterEq(line)
        if (v) s.quantization = v
        return
    }
    if (line.includes("print_info: file size")) {
        const val = afterEq(line)
        if (val) {
            const parts = val.split(" ")
            if (parts.length >= 2) {
                s.fileSizeN = parts[0]
                s.fileSizeUnit = parts[1]
            }
            const open = val.indexOf("(")
            const close = val.indexOf(")", open + 1)
            if (open >= 0 && close > open) {
                const inner = val.slice(open + 1, close)
                const bp = inner.split(" ")
                if (bp.length === 2) s.fileBpw = bp[0]
            }
        }
        return
    }
    if (line.includes("print_info: model params")) {
        const val = afterEq(line)
        if (val) {
            const parts = val.split(" ")
            if (parts.length === 2) {
                s.modelParamsN = parts[0]
                s.modelParamsUnit = parts[1]
            } else {
                s.modelParamsN = val
            }
        }
        return
    }
    if (line.includes("offloaded") && line.includes("layers to GPU")) {
        const m = /offloaded (\d+)\/(\d+) layers to GPU/.exec(line)
        if (m) {
            s.gpuLayersLoaded = parseInt(m[1], 10) || 0
            s.totalLayers = parseInt(m[2], 10) || 0
        }
        return
    }
    if (line.includes("has unified memory")) {
        const v = afterEq(line)
        if (v) s.unifiedMemory = v.trim().toLowerCase() === "true"
        return
    }
    if (line.includes("CPU_Mapped model buffer size") || line.includes("CPU model buffer size")) {
        const mib = extractMib(line)
        if (mib != null) s.cpuMemMib += mib
        return
    }
    if (line.includes("CPU_REPACK model buffer size")) {
        const mib = extractMib(line)
        if (mib != null) s.cpuRepackMib += mib
        return
    }
    if (line.includes("model buffer size") && !line.includes("CPU")) {
        const mib = extractMib(line)
        if (mib != null) s.gpuMemMib += mib
        return
    }
    if (line.includes("KV buffer size") && !line.includes("CPU")) {
        const mib = extractMib(line)
        if (mib != null) s.kvCacheMib = mib
        return
    }
    if (line.includes("CPU compute buffer size")) {
        const mib = extractMib(line)
        if (mib != null) s.cpuComputeMib = mib
        return
    }
    if (line.includes("compute buffer size") && !line.includes("CPU")) {
        const mib = extractMib(line)
        if (mib != null) s.computeBufMib = mib
        return
    }
    if (line.includes("done_getting_tensors:") && line.includes("using CPU instead")) {
        const m = /tensor\s+'([^']+)'.*?\(and\s+(\d+)\s+others\)/.exec(line)
        if (m) {
            s.cpuForcedCount = (parseInt(m[2], 10) || 0) + 1
            s.cpuForcedPrimary = m[1]
        }
        return
    }
    if (line.includes("llama_params_fit_impl:")) {
        s.fitProbing = true
        if (line.includes("memory for test allocation")) s.overflowLayers = 0
        const m = /\(\s*(\d+)\s+overflowing\)/.exec(line)
        if (m) s.overflowLayers += parseInt(m[1], 10) || 0
        return
    }
    if (line.includes("llama_params_fit:") && (line.includes("successfully fit") || line.includes("cannot fit"))) {
        s.fitProbing = false
        return
    }
    if (line.includes("llama_context: n_ctx")) {
        const m = /n_ctx\s+=\s+(\d+)/.exec(line)
        if (m) s.ctxSize = parseInt(m[1], 10) || 0
        return
    }
    if (line.includes("n_parallel")) {
        const m = /n_parallel\s*=\s*(\d+)/.exec(line)
        if (m) s.nParallel = Math.max(1, parseInt(m[1], 10) || 1)
        // fall through — no return; line may carry other parseable info
    }
    if (line.includes("server is listening on")) {
        const at = line.indexOf("on ")
        if (at >= 0) s.listenUrl = line.slice(at + 3).trim()
        lui.markEngineReady()
    }
}

/** @param {string} line @param {Lui} lui */
function parseRuntimeLine(line, lui) {
    const s = lui.state

    if (line.startsWith("srv") && line.includes("all slots are idle")) {
        s.activeSlots.clear()
        return
    }
    if (line.startsWith("slot launch_slot_") && line.includes("processing task")) {
        const idTask = extractSlotTask(line)
        if (idTask) {
            const [slotId] = idTask
            s.activeSlots.set(slotId, {
                slotId,
                nTokens: 0,
                promptTps: 0,
                genTps: 0,
                genTokens: 0,
                totalTimeMs: 0,
                progress: 0,
                processingStarted: Date.now()
            })
        }
        return
    }
    if (line.startsWith("slot update_slots:") && line.includes("new prompt")) {
        const idTask = extractSlotTask(line)
        if (idTask) {
            const m = /task\.n_tokens\s*=\s*(\d+)/.exec(line)
            if (m) {
                const slot = s.activeSlots.get(idTask[0])
                if (slot) slot.nTokens = parseInt(m[1], 10) || 0
            }
        }
        return
    }
    if (line.startsWith("slot update_slots:") && line.includes("prompt processing progress")) {
        const idTask = extractSlotTask(line)
        if (idTask) {
            const m = /progress\s*=\s*([0-9.]+)/.exec(line)
            if (m) {
                const slot = s.activeSlots.get(idTask[0])
                if (slot) slot.progress = Math.max(0, Math.min(1, parseFloat(m[1])))
            }
        }
        return
    }
    if (line.startsWith("slot release:") && line.includes("stop processing")) {
        const idTask = extractSlotTask(line)
        if (idTask) {
            const slot = s.activeSlots.get(idTask[0])
            if (slot) {
                const m = /n_tokens\s*=\s*(\d+)/.exec(line)
                if (m) slot.nTokens = parseInt(m[1], 10) || slot.nTokens
                s.activeSlots.delete(idTask[0])
                s.recentCompleted.push(slot)
                while (s.recentCompleted.length > MAX_RECENT_REQUESTS) s.recentCompleted.shift()
            }
        }
        return
    }
    if (line.includes("prompt eval time =")) {
        const m = /(\d+\.?\d*)\s+tokens per second/.exec(line)
        if (m) {
            const tps = parseFloat(m[1]) || 0
            s.lastPromptTps = tps
            s.promptTpsSamples += 1
            const n = s.promptTpsSamples
            s.avgPromptTps = s.avgPromptTps * ((n - 1) / n) + tps / n
            const idTask = extractSlotTask(line)
            if (idTask) {
                const slot = s.activeSlots.get(idTask[0])
                if (slot) slot.promptTps = tps
            }
        }
        return
    }
    const trimmed = line.trimStart()
    if (trimmed.startsWith("eval time =")) {
        const m = /(\d+\.?\d*)\s+tokens per second/.exec(line)
        if (m) {
            const tps = parseFloat(m[1]) || 0
            s.lastGenTps = tps
            s.genTpsSamples += 1
            const n = s.genTpsSamples
            s.avgGenTps = s.avgGenTps * ((n - 1) / n) + tps / n
        }
        return
    }
    if (trimmed.startsWith("total time =")) {
        const m = /(\d+\.?\d*)\s+ms\s*\/\s*(\d+)\s+tokens/.exec(line)
        if (m) {
            const timeMs = parseFloat(m[1]) || 0
            const tokens = parseInt(m[2], 10) || 0
            const slots = [...s.activeSlots.values()]
            const last = slots[slots.length - 1]
            if (last) {
                last.totalTimeMs = timeMs
                last.genTokens = tokens
            }
        }
        return
    }
    if (line.startsWith("slot print_timing:")) {
        const idTask = extractSlotTask(line)
        if (idTask) {
            const slot = s.activeSlots.get(idTask[0])
            if (slot) slot.genTps = s.lastGenTps
        }
        return
    }
}

/** @param {string} line @returns {[number, number] | null} */
function extractSlotTask(line) {
    const m = /id\s+(\d+)\s*\|\s*task\s+(-?\d+)/.exec(line)
    if (!m) return null
    return [parseInt(m[1], 10), parseInt(m[2], 10) || 0]
}

/** @param {string} line @returns {boolean} */
function isKvLine(line) {
    return line.includes("llama_model_loader:") && line.includes("kv")
}

/** @param {string} line @returns {string | null} */
function afterEq(line) {
    const i = line.indexOf(" = ")
    if (i < 0) return null
    return line.slice(i + 3).trim()
}

/** @param {string} line @returns {number | null} */
function extractMib(line) {
    const m = /(\d+\.?\d*)\s+MiB/.exec(line)
    return m ? parseFloat(m[1]) : null
}

/** @param {Lui} lui @returns {string} */
function binaryNameFromConfig(lui) {
    return lui.config.engine?.[engine.name]?.binary || BINARY_NAME
}

/** @param {Lui} lui @returns {Promise<void>} */
async function probeVersion(lui) {
    return new Promise((resolve) => {
        const bin = resolveBinary(binaryNameFromConfig(lui))
        if (!bin) return resolve(undefined)
        let stderr = ""
        const child = spawn(bin, ["--version"], { stdio: ["ignore", "ignore", "pipe"] })
        child.stderr?.setEncoding("utf8")
        child.stderr?.on("data", (c) => (stderr += c))
        child.on("error", () => resolve(undefined))
        child.on("exit", () => {
            const line = stderr.split(/\r?\n/).find((l) => l.startsWith("version:"))
            if (line) lui.state.llamaVersion = line.slice("version:".length).trim()
            resolve(undefined)
        })
    })
}

/** @param {ViewBuilder} v @param {Lui} lui */
function appendEnginePanel(v, lui) {
    const s = lui.state
    const p = v.panel("llama-server")

    const gpuTotal = s.gpuMemMib + s.kvCacheMib + s.computeBufMib
    const cpuTotal = s.cpuMemMib + s.cpuRepackMib + s.cpuComputeMib
    if (gpuTotal > 0 || cpuTotal > 0) {
        const ln = p.line().style(STYLE.LABEL).text("Memory   : ").style()
        ln.style(STYLE.VALUE)
            .text((gpuTotal / 1024).toFixed(1))
            .style()
            .text(" GiB VRAM")
        if (cpuTotal > 0) {
            ln.text(" · ").style(STYLE.VALUE).text(cpuTotal.toFixed(0)).style().text(" MiB RAM")
        }
        if (s.unifiedMemory && cpuTotal > 0) {
            ln.text(" ")
                .style(DIM)
                .text(`(${((gpuTotal + cpuTotal) / 1024).toFixed(1)} GiB total)`)
                .style()
        }

        const gpuParts = []
        if (s.gpuMemMib > 0) gpuParts.push(`${s.gpuMemMib.toFixed(0)} model`)
        if (s.kvCacheMib > 0) gpuParts.push(`${s.kvCacheMib.toFixed(0)} KV`)
        if (s.computeBufMib > 0) gpuParts.push(`${s.computeBufMib.toFixed(0)} compute`)
        const cpuParts = []
        if (s.cpuMemMib > 0) cpuParts.push(`${s.cpuMemMib.toFixed(0)} model`)
        if (s.cpuRepackMib > 0) cpuParts.push(`${s.cpuRepackMib.toFixed(0)} expert`)
        if (s.cpuComputeMib > 0) cpuParts.push(`${s.cpuComputeMib.toFixed(0)} compute`)
        const breakdown = cpuParts.length
            ? `GPU: ${gpuParts.join(" + ")} MiB · CPU: ${cpuParts.join(" + ")} MiB`
            : `${gpuParts.join(" + ")} MiB`
        if (gpuParts.length || cpuParts.length) {
            p.line({ indent: 15 }).style(DIM).text(breakdown)
        }

        if (s.totalLayers > 0) {
            const offload =
                s.gpuLayersLoaded === 0
                    ? `${s.gpuLayersLoaded}/${s.totalLayers} layers offloaded (CPU only)`
                    : s.overflowLayers > 0
                      ? `${s.gpuLayersLoaded}/${s.totalLayers} layers offloaded (${s.gpuLayersLoaded - s.overflowLayers} fully GPU, ${s.overflowLayers} with experts on CPU)`
                      : s.gpuLayersLoaded === s.totalLayers && s.cpuRepackMib > 0
                        ? `${s.gpuLayersLoaded}/${s.totalLayers} layers offloaded (partial, experts on CPU)`
                        : s.gpuLayersLoaded === s.totalLayers
                          ? `${s.gpuLayersLoaded}/${s.totalLayers} layers offloaded (fully GPU)`
                          : `${s.gpuLayersLoaded}/${s.totalLayers} layers offloaded (partial)`
            p.line({ indent: 15 }).style(DIM).text(offload)
        }
    }

    p.line()

    const modelDisplay = s.quantization ? `${s.modelName} (${s.quantization})` : s.modelName
    const aliasName = lui.activeModel?.name ?? ""
    const ln = p
        .line()
        .style(STYLE.LABEL)
        .text("Model    : ")
        .style()
        .text(modelDisplay || "(loading...)")
    if (aliasName) {
        ln.style(STYLE.LABEL).text(" — ").style(STYLE.ALIAS).style(BOLD).text(aliasName).style()
    }

    const src = inferSource(lui.activeModel?.args || [])
    if (src) p.line({ indent: 15 }).style(DIM).text(src)

    if (s.modelParamsN) {
        let line = `${s.modelParamsN} ${s.modelParamsUnit}`
        if (s.fileSizeN) {
            line += ` · ${s.fileSizeN} ${s.fileSizeUnit} on disk`
            if (s.fileBpw) line += ` (${s.fileBpw} BPW)`
        }
        p.line({ indent: 15 }).style(DIM).text(line)
    }

    if (s.ctxSize > 0) {
        const perSlot = Math.floor(s.ctxSize / Math.max(1, s.nParallel))
        let line
        if (s.nParallel > 1) {
            line = `${formatNumber(perSlot)} token context per slot · ${formatNumber(s.ctxSize)} total (${s.nParallel} slots)`
        } else if (s.maxCtxSize > 0 && s.maxCtxSize !== perSlot) {
            line = `${formatNumber(perSlot)} token context window (${formatNumber(s.maxCtxSize)} max)`
        } else {
            line = `${formatNumber(perSlot)} token context window`
        }
        p.line({ indent: 15 }).style(DIM).text(line)
    }

    p.line()

    const uptimeSec = Math.floor((Date.now() - (s.startedAt || Date.now())) / 1000)
    const ln2 = p.line().style(STYLE.LABEL).text("llamacpp : ").style()
    if (s.exited) {
        ln2.style(STYLE.ERROR_INLINE).text("Exited").style()
        if (s.exitMessage) ln2.text(`  ${s.exitMessage}`)
    } else if (lui.engineReadyFired) {
        ln2.style(STYLE.READY).text("Ready").style()
        const tail = s.llamaVersion
            ? ` (${s.llamaVersion}, uptime: ${formatDurationSeconds(uptimeSec)})`
            : ` (uptime: ${formatDurationSeconds(uptimeSec)})`
        ln2.text(tail)
        if (s.updateAvailable) ln2.text("  ").style(STYLE.WARNING).text("(update available)").style()
    } else {
        ln2.style(DIM).text("Starting...").style()
    }

    if (s.listenUrl) p.line({ indent: 15 }).style(DIM).text(s.listenUrl)

    // Full resolved argv, all dim — colors compete with labels above.
    // Reads the segments stashed at spawn rather than recomputing.
    if (s.argSegments) {
        const parts = s.argSegments.flatMap(/** @param {import("../types.js").Segment} seg */ (seg) => seg.args)
        p.line({ indent: 15 }).style(DIM).text(parts.join(" "))
    }

    // Active downloads as bars. Total may be 0 until the HEAD response
    // lands (or stays 0 if HEAD failed) — show bytes-only in that case.
    const entries = s.downloads
        .entries()
        .sort(/** @param {[string, any]} x @param {[string, any]} y */ (x, y) => x[0].localeCompare(y[0]))
    for (const [name, e] of entries) {
        if (e.total > 0) {
            const frac = Math.max(0, Math.min(1, e.downloaded / e.total))
            const pct = Math.floor(frac * 100)
            const cur = formatBytes(e.downloaded).padStart(9)
            const tot = formatBytes(e.total).padStart(9)
            p.bar({
                label: `Downloading ${name}`,
                value: frac,
                text: `${cur} / ${tot} (${String(pct).padStart(3)}%)`,
                indent: 13
            })
        } else {
            p.bar({
                label: `Downloading ${name}`,
                value: 0,
                text: formatBytes(e.downloaded).padStart(9),
                indent: 13
            })
        }
    }
}

/** @param {ViewBuilder} v @param {Lui} lui */
function appendPerformancePanel(v, lui) {
    const s = lui.state
    const p = v.panel("Performance")

    const promptLn = p.line().style(STYLE.LABEL).text("Prompt   : ").style(STYLE.VALUE)
    if (s.promptTpsSamples > 0) {
        promptLn
            .text(s.lastPromptTps.toFixed(1).padStart(6))
            .style(TEXT)
            .text(" tok/s ")
            .style(DIM)
            .text(`(avg ${s.avgPromptTps.toFixed(1).padStart(6)})`)
    } else {
        promptLn.style(DIM).text("--")
    }

    const genLn = p.line().style(STYLE.LABEL).text("Generate : ").style(STYLE.VALUE)
    if (s.genTpsSamples > 0) {
        genLn
            .text(s.lastGenTps.toFixed(1).padStart(6))
            .style(TEXT)
            .text(" tok/s ")
            .style(DIM)
            .text(`(avg ${s.avgGenTps.toFixed(1).padStart(6)})`)
    } else {
        genLn.style(DIM).text("--")
    }

    for (const slot of [...s.recentCompleted].reverse()) {
        const time = slot.totalTimeMs > 0 ? ` in ${(slot.totalTimeMs / 1000).toFixed(1)}s` : ""
        const tps = slot.genTps > 0 ? ` (${slot.genTps.toFixed(1)} tok/s)` : ""
        p.line({ indent: 13 }).style(DIM).text(`✓ slot ${slot.slotId} done ${slot.nTokens} tokens${time}${tps}`)
    }

    // The blank goes into the lines stream — bars[] always renders
    // after lines[], so this lands directly above the first slot bar
    // and is omitted when there are no active slots.
    if (s.activeSlots.size > 0) p.line()
    for (const slot of [...s.activeSlots.values()].sort((a, b) => a.slotId - b.slotId)) {
        p.bar({
            label: `● slot ${slot.slotId}: ${String(slot.nTokens).padStart(7)} tokens`,
            value: slot.progress,
            text: `${String(Math.round(slot.progress * 100)).padStart(3)}%`,
            indent: 13
        })
    }
}

/** @param {ViewBuilder} v @param {Lui} lui */
function appendServerLogPanel(v, lui) {
    const s = lui.state
    const p = v.panel("Server Log")
    const tail = s.logLines.slice(-100)
    for (const line of tail) {
        p.line()
            .style(DIM)
            .text(line.length > 300 ? line.slice(0, 300) : line)
    }
}

/** @param {string[]} args @returns {string | null} */
function inferSource(args) {
    for (let i = 0; i < args.length; i++) {
        if (args[i] === "--hf" && i + 1 < args.length) return `--hf ${args[i + 1]}`
        if (args[i] === "-m" && i + 1 < args.length) return `-m ${args[i + 1]}`
    }
    return null
}
