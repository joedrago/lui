// strata engine — github.com/Niko1221/Strata's serve/server.py, via the
// `strata` wrapper an install's docs/AI_SETUP.md has you symlink onto PATH
// (bin/strata in the Strata folder; it always runs --engine strata, never
// --engine mock, from that install's own venv regardless of cwd).
//
// Unlike mlx_lm, no readiness probe is needed: serve/server.py doesn't print
// its `ready:` line until the model is actually loaded and serving, so
// parseLine's job is just to wait for that one line. Everything before it
// is narration about loading tens of GB of experts into RAM (and can take
// minutes); everything after is per-request progress (prompt read, decode
// phase, a final "done:" line with the request's own tok/s and expert cache
// hit rate).

/** @import { Engine, ViewBuilder } from "../types.js" */
/** @import { Lui } from "../lui.js" */

import fs from "node:fs"

import { STYLE } from "../theme.js"
import { stripAnsi } from "../ansi.js"
import { resolveBinary, spawnProcess, describeSpawnError } from "../spawn.js"
import { formatDurationSeconds } from "../util.js"

const BINARY_NAME = "strata"

const DIM = { dim: true }

const LOG_RING_SIZE = 200

// Flags lui owns. --engine is reserved too: the wrapper already passes
// --engine strata itself, and argparse's last-flag-wins would let a second
// one on the user's own args silently override it instead of failing loudly.
const RESERVED_FLAGS = new Set(["--host", "--port", "--engine"])

/** @param {string[]} args @returns {string | null} */
function configPathFromArgs(args) {
    for (let i = 0; i < args.length; i++) {
        if (args[i] === "--config" && i + 1 < args.length) return args[i + 1]
    }
    return null
}

// Best-effort read of the strata-<model>.json setup.py wrote — the source
// of truth for context size and the served model's API name while the
// engine isn't running (e.g. `lui ssh` previewing a model). Cached per path:
// config files don't change under a running model, and the panel renderer
// calls this several times a second. Only meaningful for an absolute
// --config path (the normal case) — a relative one resolves against lui's
// own cwd, not the wrapper's, and most likely just misses.
/** @type {Map<string, any>} */
const configCache = new Map()

/** @param {string[]} args @returns {any | null} */
function readStrataConfig(args) {
    const p = configPathFromArgs(args)
    if (!p) return null
    if (configCache.has(p)) return configCache.get(p)
    let cfg = null
    try {
        cfg = JSON.parse(fs.readFileSync(p, "utf8"))
    } catch {
        cfg = null
    }
    configCache.set(p, cfg)
    return cfg
}

// Shared by the engine's own servedModelName hook and appendEnginePanel,
// which needs the name before an engine is even running and so can't go
// through `lui.engineModule?.servedModelName?.(...)` (optional on the
// Engine type) the way other callers do.
/** @param {any} state @param {import("../types.js").Model | null | undefined} model @returns {string | null} */
function servedModelNameOf(state, model) {
    if (state?.servedModelName) return state.servedModelName
    const cfg = readStrataConfig(model?.args || [])
    return cfg ? cfg.model_name || "qwen3.8-flash-next" : null
}

/** @param {any} cfg @returns {number | null} */
function contextSizeFromConfig(cfg) {
    const args = cfg?.args
    if (!Array.isArray(args)) return null
    const i = args.indexOf("--max-context")
    if (i < 0 || i + 1 >= args.length) return null
    const n = parseInt(args[i + 1], 10)
    return Number.isFinite(n) && n > 0 ? n : null
}

/** @type {Engine} */
export const engine = {
    name: "strata",

    schema: [{ path: "binary", default: BINARY_NAME }],

    shutdownSummary(state) {
        const lines = []
        const logLines = state?.logLines ?? []
        const exitCode = state?.exitCode
        const exitSignal = state?.exitSignal
        if ((exitCode != null && exitCode !== 0) || exitSignal) {
            const tail = logLines.slice(-5)
            if (tail.length > 0) lines.push({ label: "Last log lines", value: tail.join("\n") })
        }
        return { lines, fatal: state?.fatalReason || null }
    },

    exitReason(state, code, signal) {
        if (state?.exitMessage) return state.exitMessage
        return signal ? `killed by ${signal}` : `exited with code ${code}`
    },

    // The live `ready:` line wins once parsed; otherwise fall back to the
    // --max-context baked into the model's strata-<model>.json.
    contextSize(state, model) {
        if (state && state.ctxSize > 0) return state.ctxSize
        return contextSizeFromConfig(readStrataConfig(model?.args || []))
    },

    // strata's OpenAI/Anthropic endpoints, like llama-server's, accept any
    // model id a request sends and answer regardless — this is cosmetic,
    // just what the harness UI shows. model_name mirrors serve/server.py's
    // own default ("qwen3.8-flash-next") when the config omits it.
    servedModelName(state, model) {
        return servedModelNameOf(state, model)
    },

    // strata listens on lui's engine_port, on whichever address the user
    // reached us with — same shape as llama-server/mlx_lm.
    endpoint(lui) {
        return { host: null, port: lui.config.global.engine_port }
    },

    describe(model, lui) {
        const binaryName = binaryNameFromConfig(lui)
        const host = lui.config.global.public ? "0.0.0.0" : "127.0.0.1"
        const port = lui.config.global.engine_port
        const userArgs = Array.isArray(model.args) ? [...model.args] : []

        const errors = []
        for (const tok of userArgs) {
            if (RESERVED_FLAGS.has(tok)) {
                errors.push(`${tok} is reserved by lui — drop it from this model's args`)
            }
        }
        if (!configPathFromArgs(userArgs)) {
            errors.push("strata needs --config <path to the strata-<model>.json setup.py wrote> (docs/AI_SETUP.md)")
        }

        return {
            segments: [
                { name: "binary", args: [binaryName] },
                { name: "binding", style: STYLE.SEGMENT_BINDING, args: ["--host", host, "--port", String(port)] },
                { name: "user", style: STYLE.SEGMENT_USER, args: userArgs }
            ],
            warnings: [],
            errors
        }
    },

    async start(lui, model, desc) {
        lui.state.argSegments = desc.segments
        const [binarySeg, ...rest] = desc.segments
        const binaryName = binarySeg.args[0]
        const binaryPath = resolveBinary(binaryName) ?? binaryName
        const argv = rest.flatMap((s) => s.args)
        const env = { ...process.env, ...(model.env || {}) }
        lui.state.proc = spawnProcess({
            binary: binaryPath,
            argv,
            env,
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
        await lui.state?.proc?.stop?.()
    },

    initState(lui) {
        const s = lui.state
        s.startedAt = Date.now()
        s.argSegments = null
        s.proc = null
        s.activeModelName = lui.activeModel?.name ?? ""
        s.exited = false
        s.exitMessage = ""
        s.fatalReason = null
        s.logLines = []

        // Startup narration (before the `ready:` line):
        // weights -> experts -> experts-loaded -> cache -> almost-ready -> ready.
        s.loadPhase = "weights"
        s.loadDetail = ""
        s.loadElapsedS = null

        // Set once the `ready:` line is parsed.
        s.ctxSize = 0
        s.listenUrl = ""
        s.servedModelName = null
        s.images = false
        s.apiKeyRequired = false

        // Per-request runtime state, from the narration serve/server.py
        // prints while a request is in flight (reading the prompt, then
        // thinking/answering/a tool call) and the "done:" line that closes
        // it out.
        s.promptStatus = null // { done, total, elapsedS }
        s.gen = null // { phase, generated, maxTokens, rate, elapsedS } while active
        s.lastGen = null // { tokens, elapsedS, rate, finish, cancelled, hitRate } once done
        s.totalGenTokens = 0
        s.totalGenMs = 0 // estimated from each request's own reported rate
    },

    parseLine(rawLine, lui) {
        const line = stripAnsi(rawLine)
        if (!line) return
        pushLog(lui.state, line)

        if (!lui.engineReadyFired) parseLoadLine(line, lui)
        else parseRuntimeLine(line, lui)
    },

    appendPanels(v, lui) {
        appendEnginePanel(v, lui)
        appendPerformancePanel(v, lui)
        appendServerLogPanel(v, lui)
    }
}

/** @param {string} line @param {Lui} lui */
function parseLoadLine(line, lui) {
    const s = lui.state

    const ready =
        /^ready: http:\/\/([^\s:/]+):(\d+)\/v1\s+\(OpenAI:[^,]*,\s*Anthropic:[^,]*,\s*context (\d+) tokens(, images on)?(, API key required)?\)$/.exec(
            line
        )
    if (ready) {
        s.listenUrl = `http://${ready[1]}:${ready[2]}/v1`
        s.ctxSize = parseInt(ready[3], 10) || 0
        s.images = !!ready[4]
        s.apiKeyRequired = !!ready[5]
        s.loadPhase = "ready"
        lui.markEngineReady()
        return
    }

    if (/^\[strata\] starting the engine: reading the model's weights \.\.\.$/.test(line)) {
        s.loadPhase = "weights"
        return
    }
    if (/^\[strata\] (loading|mapping) the( most-used)? experts/.test(line)) {
        s.loadPhase = "experts"
        return
    }

    const loaded = /^\[strata\] experts loaded: (.+) \((\d+) s so far\)$/.exec(line)
    if (loaded) {
        s.loadPhase = "experts-loaded"
        s.loadDetail = loaded[1]
        s.loadElapsedS = parseInt(loaded[2], 10) || 0
        return
    }

    const cache = /^\[strata\] filling the GPU's expert cache \((.+)\) \.\.\.$/.exec(line)
    if (cache) {
        s.loadPhase = "cache"
        s.loadDetail = cache[1]
        return
    }

    if (/^\[strata\] almost ready \.\.\.$/.test(line)) {
        s.loadPhase = "almost-ready"
        return
    }

    const still = /^\[strata\] still starting \((\d+) s\) - please wait \.\.\.$/.exec(line)
    if (still) s.loadElapsedS = parseInt(still[1], 10) || 0
}

/** @param {string} line @param {Lui} lui */
function parseRuntimeLine(line, lui) {
    const s = lui.state

    const prompt = /^\[strata\] reading the prompt: ([\d,]+)(?: of ([\d,]+))? tokens, (\d+) s so far$/.exec(line)
    if (prompt) {
        s.promptStatus = { done: prompt[1], total: prompt[2] || null, elapsedS: parseInt(prompt[3], 10) || 0 }
        s.gen = null
        return
    }

    const gen =
        /^\[strata\] (thinking|answering|tool call complete|writing a tool call: .+): (\d+) of max (\d+|None) tokens, ([\d.]+) tok\/s, (\d+) s$/.exec(
            line
        )
    if (gen) {
        s.promptStatus = null
        s.gen = {
            phase: gen[1],
            generated: parseInt(gen[2], 10) || 0,
            maxTokens: gen[3] === "None" ? null : parseInt(gen[3], 10),
            rate: parseFloat(gen[4]) || 0,
            elapsedS: parseInt(gen[5], 10) || 0
        }
        return
    }

    const done =
        /^\[strata\] done: (\d+) tokens in (\d+) s \(([\d.]+) tok\/s\) \((\w+), cancel=(True|False)\)(?:, expert cache ([\d.]+)% hit)?$/.exec(
            line
        )
    if (done) {
        const tokens = parseInt(done[1], 10) || 0
        const rate = parseFloat(done[3]) || 0
        s.gen = null
        s.promptStatus = null
        s.lastGen = {
            tokens,
            elapsedS: parseInt(done[2], 10) || 0,
            rate,
            finish: done[4],
            cancelled: done[5] === "True",
            hitRate: done[6] != null ? parseFloat(done[6]) : null
        }
        if (rate > 0) {
            s.totalGenTokens += tokens
            s.totalGenMs += (tokens / rate) * 1000
        }
        return
    }
}

/** @param {any} s @param {string} line */
function pushLog(s, line) {
    if (s.logLines.length >= LOG_RING_SIZE) s.logLines.shift()
    s.logLines.push(line)
}

/** @param {Lui} lui @returns {string} */
function binaryNameFromConfig(lui) {
    return lui.config.engine?.[engine.name]?.binary || BINARY_NAME
}

/** @type {Record<string, string>} */
const LOAD_PHASE_LABEL = {
    weights: "Reading the model's weights...",
    experts: "Loading experts into RAM...",
    "experts-loaded": "Experts loaded, filling the GPU's expert cache...",
    cache: "Filling the GPU's expert cache...",
    "almost-ready": "Almost ready..."
}

/** @param {ViewBuilder} v @param {Lui} lui */
function appendEnginePanel(v, lui) {
    const s = lui.state
    const p = v.panel("strata")

    const aliasName = lui.activeModel?.name ?? ""
    const modelLn = p.line().style(STYLE.LABEL).text("Model    : ").style()
    modelLn.text(servedModelNameOf(s, lui.activeModel) || "(unknown)")
    if (aliasName) {
        modelLn.style(STYLE.LABEL).text(" — ").style(STYLE.ALIAS).style({ bold: true }).text(aliasName).style()
    }

    p.line()

    const uptimeSec = Math.floor((Date.now() - (s.startedAt || Date.now())) / 1000)
    const statusLn = p.line().style(STYLE.LABEL).text("strata   : ").style()
    if (s.exited) {
        statusLn.style(STYLE.ERROR_INLINE).text("Exited").style()
        if (s.exitMessage) statusLn.text(`  ${s.exitMessage}`)
    } else if (lui.engineReadyFired) {
        statusLn
            .style(STYLE.READY)
            .text("Ready")
            .style()
            .text(` (uptime: ${formatDurationSeconds(uptimeSec)})`)
    } else {
        const label = LOAD_PHASE_LABEL[s.loadPhase] || "Starting..."
        statusLn.style(DIM).text(label).style()
        if (s.loadElapsedS != null) statusLn.text(` (${s.loadElapsedS}s)`)
    }

    if (s.loadDetail && !lui.engineReadyFired) p.line({ indent: 15 }).style(DIM).text(s.loadDetail)
    if (s.listenUrl) p.line({ indent: 15 }).style(DIM).text(s.listenUrl)
    if (s.ctxSize > 0)
        p.line({ indent: 15 })
            .style(DIM)
            .text(`context ${s.ctxSize.toLocaleString("en-US")} tokens`)

    if (s.argSegments) {
        const parts = s.argSegments.flatMap(/** @param {import("../types.js").Segment} seg */ (seg) => seg.args)
        p.line({ indent: 15 }).style(DIM).text(parts.join(" "))
    }
}

/** @param {ViewBuilder} v @param {Lui} lui */
function appendPerformancePanel(v, lui) {
    const s = lui.state
    if (!s.gen && !s.lastGen && !s.promptStatus) return

    const p = v.panel("Performance")

    if (s.promptStatus) {
        const { done, total, elapsedS } = s.promptStatus
        if (total) {
            const doneN = parseInt(done.replace(/,/g, ""), 10) || 0
            const totalN = parseInt(total.replace(/,/g, ""), 10) || 0
            p.bar({ label: "Reading the prompt", value: doneN, max: totalN, text: `${done}/${total}`, indent: 13 })
        } else {
            p.line()
                .style(STYLE.LABEL)
                .text("Prompt   : ")
                .style(STYLE.VALUE)
                .text(done)
                .style()
                .text(" tokens ")
                .style(DIM)
                .text(`(${elapsedS}s so far)`)
        }
    } else if (s.gen) {
        const g = s.gen
        const maxText = g.maxTokens != null ? g.maxTokens : "∞"
        p.line()
            .style(STYLE.LABEL)
            .text("Active   : ")
            .style(STYLE.VALUE)
            .text(g.rate.toFixed(1).padStart(6))
            .style()
            .text(" tok/s ")
            .style(DIM)
            .text(`(${g.phase}, ${g.generated} of max ${maxText} tokens, ${g.elapsedS}s)`)
    } else if (s.lastGen) {
        const g = s.lastGen
        p.line()
            .style(STYLE.LABEL)
            .text("Last gen : ")
            .style(STYLE.VALUE)
            .text(g.rate.toFixed(1).padStart(6))
            .style()
            .text(" tok/s ")
            .style(DIM)
            .text(`(${g.tokens} tokens, ${g.finish}${g.cancelled ? ", cancelled" : ""})`)
        if (g.hitRate != null) {
            p.line({ indent: 13 })
                .style(DIM)
                .text(`expert cache ${g.hitRate.toFixed(1)}% hit`)
        }
    }

    if (s.totalGenMs > 0) {
        const avg = (s.totalGenTokens / s.totalGenMs) * 1000
        p.line()
            .style(STYLE.LABEL)
            .text("Average  : ")
            .style(STYLE.VALUE)
            .text(avg.toFixed(1).padStart(6))
            .style()
            .text(" tok/s ")
            .style(DIM)
            .text(`(${s.totalGenTokens} tokens total)`)
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
