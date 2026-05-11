// llama.cpp engine. Builds an argv (binding + policy + user), parses
// llama-server's stdout for status fields, emits Model / Performance /
// Server Log panels.
//
// Behavior ported from old/src/server.rs (parseLine, ServerState) and
// old/src/display.rs (Model and Performance panel rendering).

import { spawn } from "node:child_process"
import os from "node:os"

import { STYLE } from "../engine.js"

// Palette ported from old/src/display.rs:
//   MUTED_PURPLE   (120,100,180)  – labels (Model:, llamacpp:, …)
//   COLOR_NUMBER   (210,150,255)  – numeric/value spans + active markers
//   COLOR_ALIASES  (220,215,230)  – soft off-white for alias names
//   WARNING_AMBER  (230,180,80)   – the "(update available)" hint
const MUTED_PURPLE = { fg: [120, 100, 180] }
const NUMBER = { fg: [210, 150, 255] }
const ALIAS = { fg: [220, 215, 230] }
const WARNING_AMBER = { fg: [230, 180, 80] }
const DIM = { dim: true }
const BOLD = { bold: true }
const READY = { fg: "green" }
const TEXT = {}

const POLICY_ARGS = ["--metrics", "--jinja", "--log-colors", "off", "-v", "-fa", "on", "--cache-reuse", "256", "-kvu"]

// Per-model defaults the engine fills in when the user didn't supply
// the equivalent flag in `model.args`. Each entry lists every flag form
// that should suppress injection — both lui's own form (the first one
// passed to llama-server) and any other spelling llama-server itself
// accepts, so a user-typed `--gpu-layers 20` still wins over `-ngl -1`.
const DEFAULT_HINTS = [
    { flags: ["-ngl", "--gpu-layers", "--n-gpu-layers"], emit: () => ["-ngl", "-1"] },
    { flags: ["-np", "--parallel"], emit: () => ["-np", "1"] },
    { flags: ["-t", "--threads"], emit: () => ["-t", String(autoThreads())] },
    { flags: ["--chat-template-kwargs"], emit: () => ["--chat-template-kwargs", '{"preserve_thinking":true}'] }
]

function autoThreads() {
    const n = (os.availableParallelism?.() ?? os.cpus().length) - 2
    return Math.max(1, n)
}

function userSuppliedAny(args, flags) {
    return args.some((a) => flags.includes(a))
}

const LOG_RING_SIZE = 200
const MAX_RECENT_REQUESTS = 3
const SETUP_URL_BRIGHT_MS = 5000

// Flags the engine will not let the user override — lui owns these.
const RESERVED_FLAGS = new Set(["--host", "--port"])

// eslint-disable-next-line no-control-regex
const ANSI_RE = /\x1b\[[\x20-\x3f]*[\x40-\x7e]|\x1b\][\x20-\x7e]*(?:\x07|\x1b\\)|\x1b[\x20-\x2f]*[\x30-\x7e]/g

function stripAnsi(s) {
    return s.replace(ANSI_RE, "")
}

export const engine = {
    name: "llama-server",
    defaultBinary: "llama-server",

    buildArgv(model, lui) {
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
            binary: this.defaultBinary,
            segments: [
                { name: "binding", style: STYLE.SEGMENT_BINDING, args: ["--host", host, "--port", String(port)] },
                { name: "policy", style: STYLE.SEGMENT_POLICY, args: [...POLICY_ARGS] },
                { name: "defaults", style: STYLE.SEGMENT_DEFAULTS, args: defaults },
                { name: "user", style: STYLE.SEGMENT_USER, args: userArgs }
            ],
            warnings: [],
            errors
        }
    },

    initState(lui) {
        const s = lui.state
        s.startedAt = Date.now()
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
        s.ready = false
        s.listenUrl = ""
        s.requestCount = 0
        s.activeRequests = 0
        s.activeSlots = new Map()
        s.recentCompleted = []
        s.lastPromptTps = 0
        s.lastGenTps = 0
        s.avgPromptTps = 0
        s.avgGenTps = 0
        s.promptTpsSamples = 0
        s.genTpsSamples = 0
        s.fullReprocessCount = 0
        s.invalidatedCheckpointCount = 0
        s.downloads = new Map()
        s.logLines = []
        s.exited = false
        s.exitMessage = ""
        s.fitProbing = false
        s.fatalReason = null

        probeVersion(lui).catch(() => {})
    },

    parseLine(rawLine, lui) {
        const line = stripAnsi(rawLine)
        if (!line) return

        // Drop llama-server's own request/response body dumps — they echo
        // arbitrary prompt JSON that has historically clobbered our state
        // accumulators when a user pasted code into the prompt.
        if (line.includes("converted request:")) return
        if (line.includes("log_server_r:") && !line.includes("done request:")) return

        // CUDA graph noise: never reaches the log ring.
        if (/^\s*CUDA Graph id \d+ reused\s*$/.test(line)) return
        if (/^\s*ggml_backend_cuda_graph_compute: CUDA graph warmup (reset|complete)\s*$/.test(line)) return

        // Download progress is its own bar; doesn't go to log ring either.
        if (line.includes("Downloading ")) {
            const re = /Downloading (\S+\.\S+)\s.*?(\d{1,3})%/g
            let m
            let matched = false
            while ((m = re.exec(line))) {
                const name = m[1]
                const pct = parseInt(m[2], 10) || 0
                const prev = lui.state.downloads.get(name) ?? 0
                if (pct >= prev) lui.state.downloads.set(name, pct)
                matched = true
            }
            if (matched) return
        }

        if (!lui.state.ready) parseLoadLine(line, lui)
        else parseRuntimeLine(line, lui)

        pushLog(lui.state, line)
    },

    appendPanels(v, lui) {
        appendModelPanel(v, lui)
        appendPerformancePanel(v, lui)
        appendServerLogPanel(v, lui)
    }
}

function pushLog(s, line) {
    if (s.logLines.length >= LOG_RING_SIZE) s.logLines.shift()
    s.logLines.push(line)
}

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
        s.ready = true
    }
}

function parseRuntimeLine(line, lui) {
    const s = lui.state

    if (line.includes("done request: POST")) {
        s.requestCount += 1
        return
    }
    if (line.includes("forcing full prompt re-processing")) {
        s.fullReprocessCount += 1
        return
    }
    if (line.includes("invalidated context checkpoint") || line.includes("invalidated checkpoint")) {
        s.invalidatedCheckpointCount += 1
        return
    }
    if (line.startsWith("srv") && line.includes("all slots are idle")) {
        s.activeRequests = 0
        s.activeSlots.clear()
        return
    }
    if (line.startsWith("slot launch_slot_") && line.includes("processing task")) {
        const idTask = extractSlotTask(line)
        if (idTask) {
            const [slotId] = idTask
            s.activeRequests += 1
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
            s.activeRequests = Math.max(0, s.activeRequests - 1)
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

function extractSlotTask(line) {
    const m = /id\s+(\d+)\s*\|\s*task\s+(-?\d+)/.exec(line)
    if (!m) return null
    return [parseInt(m[1], 10), parseInt(m[2], 10) || 0]
}

function isKvLine(line) {
    return line.includes("llama_model_loader:") && line.includes("kv")
}

function afterEq(line) {
    const i = line.indexOf(" = ")
    if (i < 0) return null
    return line.slice(i + 3).trim()
}

function extractMib(line) {
    const m = /(\d+\.?\d*)\s+MiB/.exec(line)
    return m ? parseFloat(m[1]) : null
}

// One-shot `llama-server --version`; writes the parsed version into
// lui.state.llamaVersion. Best-effort; non-fatal if it fails.
async function probeVersion(lui) {
    return new Promise((resolve) => {
        const bin = lui.config.engine?.[engine.name]?.binary || engine.defaultBinary
        let stderr = ""
        const child = spawn(bin, ["--version"], { stdio: ["ignore", "ignore", "pipe"] })
        child.stderr.setEncoding("utf8")
        child.stderr.on("data", (c) => (stderr += c))
        child.on("error", () => resolve())
        child.on("exit", () => {
            const line = stderr.split(/\r?\n/).find((l) => l.startsWith("version:"))
            if (line) lui.state.llamaVersion = line.slice("version:".length).trim()
            resolve()
        })
    })
}

function formatNumber(n) {
    return Number(n).toLocaleString("en-US")
}

function formatDurationSeconds(sec) {
    if (sec < 60) return `<1m`
    const m = Math.floor(sec / 60)
    if (m < 60) return `${m}m`
    const h = Math.floor(m / 60)
    const rm = m % 60
    return rm ? `${h}h${rm}m` : `${h}h`
}

function appendModelPanel(v, lui) {
    const s = lui.state
    const p = v.panel("lui — llm ui")

    const webPort = lui.config.global.web_port
    if (lui.config.global.websearch !== false && webPort) {
        const host = lui.config.global.public ? (lui.ownHost ?? "127.0.0.1") : "127.0.0.1"
        // Bright cyan + bold for the first 1.5s so the URL grabs attention
        // on startup; then fades to dim so it stops competing with the
        // live status fields below it. Matches the Rust display behavior.
        const fresh = Date.now() - lui.startedAt < SETUP_URL_BRIGHT_MS
        const urlStyle = fresh ? { fg: "cyan", bold: true } : DIM
        p.line({ align: "right" }).style(urlStyle).text(`http://${host}:${webPort}/setup`)
    }

    // Memory line
    const gpuTotal = s.gpuMemMib + s.kvCacheMib + s.computeBufMib
    const cpuTotal = s.cpuMemMib + s.cpuRepackMib + s.cpuComputeMib
    if (gpuTotal > 0 || cpuTotal > 0) {
        const ln = p.line().style(MUTED_PURPLE).text("Memory   : ").style()
        ln.style(NUMBER)
            .text((gpuTotal / 1024).toFixed(1))
            .style()
            .text(" GiB VRAM")
        if (cpuTotal > 0) {
            ln.text(" · ").style(NUMBER).text(cpuTotal.toFixed(0)).style().text(" MiB RAM")
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

    // Model line
    const modelDisplay = s.quantization ? `${s.modelName} (${s.quantization})` : s.modelName
    const aliasName = lui.activeModel?.name ?? ""
    const ln = p
        .line()
        .style(MUTED_PURPLE)
        .text("Model    : ")
        .style()
        .text(modelDisplay || "(loading…)")
    if (aliasName) {
        ln.style(MUTED_PURPLE).text(" — ").style(ALIAS).style(BOLD).text(aliasName).style()
    }

    // Source line (--hf X or -m PATH). Inferred from model.args.
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

    // llamacpp status
    const uptimeSec = Math.floor((Date.now() - (s.startedAt || Date.now())) / 1000)
    const ln2 = p.line().style(MUTED_PURPLE).text("llamacpp : ").style()
    if (s.exited) {
        ln2.style({ fg: "red" }).text("Exited").style()
        if (s.exitMessage) ln2.text(`  ${s.exitMessage}`)
    } else if (s.ready) {
        ln2.style(READY).text("Ready").style()
        const tail = s.llamaVersion
            ? ` (${s.llamaVersion}, uptime: ${formatDurationSeconds(uptimeSec)})`
            : ` (uptime: ${formatDurationSeconds(uptimeSec)})`
        ln2.text(tail)
        if (s.updateAvailable) ln2.text("  ").style(WARNING_AMBER).text("(update available)").style()
    } else {
        ln2.style(DIM).text("Starting…").style()
    }

    if (s.listenUrl) p.line({ indent: 15 }).style(DIM).text(s.listenUrl)

    const userArgs = lui.activeModel?.args || []
    if (userArgs.length) p.line({ indent: 15 }).style(DIM).text(userArgs.join(" "))

    // Active downloads as bars.
    for (const [name, pct] of [...s.downloads.entries()].sort()) {
        p.bar({ label: `Downloading ${name}`, value: pct, max: 100, text: `${String(pct).padStart(3)}%`, indent: 13 })
    }

    // Active prefill bars (one per running slot).
    for (const slot of [...s.activeSlots.values()].sort((a, b) => a.slotId - b.slotId)) {
        if (slot.progress > 0 && slot.progress < 1) {
            p.bar({
                label: `● slot ${slot.slotId} prefilling`,
                value: slot.progress,
                text: `${String(Math.round(slot.progress * 100)).padStart(3)}%`,
                indent: 13
            })
        }
    }
}

function appendPerformancePanel(v, lui) {
    const s = lui.state
    const p = v.panel("Performance")

    if (s.promptTpsSamples > 0) {
        p.line()
            .style(MUTED_PURPLE)
            .text("Prompt   : ")
            .style(NUMBER)
            .text(s.lastPromptTps.toFixed(1))
            .style(TEXT)
            .text(" tok/s ")
            .style(DIM)
            .text(`(avg ${s.avgPromptTps.toFixed(1)})`)
    }
    if (s.genTpsSamples > 0) {
        p.line()
            .style(MUTED_PURPLE)
            .text("Generate : ")
            .style(NUMBER)
            .text(s.lastGenTps.toFixed(1))
            .style(TEXT)
            .text(" tok/s ")
            .style(DIM)
            .text(`(avg ${s.avgGenTps.toFixed(1)})`)
    }

    if (s.promptTpsSamples > 0 || s.genTpsSamples > 0) p.line()

    p.line()
        .style(MUTED_PURPLE)
        .text("WebSearch: ")
        .style(NUMBER)
        .text(`${lui.websearchCount ?? 0}`.padStart(4))
        .style(TEXT)
        .text(" total · ")
        .style(NUMBER)
        .text(`${lui.activeSearchCount ?? 0}`.padStart(4))
        .style(TEXT)
        .text(" active")

    p.line()
        .style(MUTED_PURPLE)
        .text("Requests : ")
        .style(NUMBER)
        .text(`${s.requestCount}`.padStart(4))
        .style(TEXT)
        .text(" total · ")
        .style(NUMBER)
        .text(`${s.activeRequests}`.padStart(4))
        .style(TEXT)
        .text(" active · ")
        .style(NUMBER)
        .text(`${s.fullReprocessCount}`.padStart(4))
        .style(TEXT)
        .text(" reproc · ")
        .style(NUMBER)
        .text(`${s.invalidatedCheckpointCount}`.padStart(4))
        .style(TEXT)
        .text(" invalidated")

    for (const slot of [...s.activeSlots.values()].sort((a, b) => a.slotId - b.slotId)) {
        const elapsed = slot.processingStarted ? ((Date.now() - slot.processingStarted) / 1000).toFixed(1) : "0"
        p.line({ indent: 13 }).style(NUMBER).text(`● slot ${slot.slotId}: ${slot.nTokens} tokens, ${elapsed}s`)
    }
    for (const slot of [...s.recentCompleted].reverse()) {
        const time = slot.totalTimeMs > 0 ? ` in ${(slot.totalTimeMs / 1000).toFixed(1)}s` : ""
        const tps = slot.genTps > 0 ? ` (${slot.genTps.toFixed(1)} tok/s)` : ""
        p.line({ indent: 13 }).style(DIM).text(`✓ slot ${slot.slotId} done ${slot.nTokens} tokens${time}${tps}`)
    }
}

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

function inferSource(args) {
    for (let i = 0; i < args.length; i++) {
        if (args[i] === "--hf" && i + 1 < args.length) return `--hf ${args[i + 1]}`
        if (args[i] === "-m" && i + 1 < args.length) return `-m ${args[i + 1]}`
    }
    return null
}
