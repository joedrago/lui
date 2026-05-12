// mlx_lm.server engine. Bare-bones scaffold — parseLine just shoves
// lines into the log ring and flips to Ready when uvicorn/Flask
// announces the listen address. Once we have a debug log to study
// we can flesh out real parsing (model name, ctx, tps, slots…).

import { STYLE } from "../theme.js"
import { stripAnsi } from "../ansi.js"
import { resolveBinary, spawnProcess, describeSpawnError } from "../spawn.js"
import { formatDurationSeconds } from "../util.js"

const BINARY_NAME = "mlx_lm.server"

const DIM = { dim: true }

const LOG_RING_SIZE = 200

// Flags lui owns — drop them from user args with an error so the
// model record stays clean.
const RESERVED_FLAGS = new Set(["--host", "--port"])

// Per-flag defaults injected when the user didn't supply any of these
// aliases. INFO-level mlx_lm.server is very terse (no token counts,
// no tok/s) so we bump to DEBUG so parseLine has something to chew on.
const DEFAULT_HINTS = [{ flags: ["--log-level"], emit: () => ["--log-level", "DEBUG"] }]

function userSuppliedAny(args, flags) {
    return args.some((a) => flags.includes(a))
}

export const engine = {
    name: "mlx_lm",

    schema: [{ path: "binary", default: BINARY_NAME }],

    setupDefaults: [
        {
            name: "qwen-mlx",
            args: ["--model", "mlx-community/Qwen3-4B-4bit"]
        }
    ],

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

    // mlx_lm.server doesn't advertise the loaded context length on
    // stdout yet (TODO once we study a debug log). Until then fall
    // back to whatever the user passed via --max-tokens / similar; if
    // nothing matches we return null and the harness uses its own
    // default.
    contextSize(state, model) {
        if (state && state.ctxSize > 0) return state.ctxSize
        const args = model?.args || []
        for (let i = 0; i < args.length; i++) {
            if (args[i] === "--max-tokens" && i + 1 < args.length) {
                const n = parseInt(args[i + 1], 10)
                if (Number.isFinite(n) && n > 0) return n
            }
        }
        return null
    },

    // mlx_lm.server uses the --model arg value as its OpenAI API
    // model id and 404s any other id. Harnesses need this exact
    // string in their request bodies. If the server later learns a
    // different id (e.g. /v1/models reports an adapter override) we
    // promote state.servedModelName ahead of the parsed default.
    servedModelName(state, model) {
        if (state?.servedModelName) return state.servedModelName
        const args = model?.args || []
        for (let i = 0; i < args.length; i++) {
            if (args[i] === "--model" && i + 1 < args.length) return args[i + 1]
        }
        return null
    },

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

        const defaults = []
        for (const hint of DEFAULT_HINTS) {
            if (!userSuppliedAny(userArgs, hint.flags)) defaults.push(...hint.emit())
        }

        return {
            segments: [
                { name: "binary", args: [binaryName] },
                { name: "binding", style: STYLE.SEGMENT_BINDING, args: ["--host", host, "--port", String(port)] },
                { name: "defaults", style: STYLE.SEGMENT_DEFAULTS, args: defaults },
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
        const binaryPath = resolveBinary(binaryName)
        const argv = rest.flatMap((s) => s.args)
        lui.state.proc = spawnProcess({
            binary: binaryPath,
            argv,
            parseLine: (line) => engine.parseLine(line, lui),
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
        s.ctxSize = 0
        s.listenUrl = ""
        s.logLines = []
        s.exited = false
        s.exitMessage = ""
        s.fatalReason = null

        // Performance / runtime stats lifted from DEBUG output.
        s.gen = null // { startMs, tokens, lastTokenMs } while active
        s.lastGen = null // { tokens, durationMs, tps } after finalize
        // Token-weighted average across every finalized generation —
        // true throughput, so a 10-token blip doesn't anchor the
        // number the same way a 500-token completion does.
        s.totalGenTokens = 0
        s.totalGenMs = 0
        s.cacheSequences = 0
        s.cacheGiB = 0
        s.promptProcessed = 0
        s.promptTotal = 0

        // Multi-line body suppression: when set we drop continuation
        // lines until the next logger-prefixed line or access-log line.
        s.swallowBody = false
        s.swallowCacheRoles = false
    },

    parseLine(rawLine, lui) {
        const line = stripAnsi(rawLine)
        if (!line) return
        const s = lui.state
        const lineTs = parseLineTimestamp(line) ?? Date.now()

        // tqdm carriage-return progress bars from huggingface_hub.
        if (/Fetching \d+ files:/.test(line)) return

        // Mid-body continuation of a previously-suppressed JSON dump.
        // Break out the moment we see a real logger line or the
        // BaseHTTPServer access log.
        if (s.swallowBody) {
            if (isLoggerPrefixed(line) || isAccessLog(line)) s.swallowBody = false
            else return
        }

        const m = /^\d{4}-\d{2}-\d{2} \S+ - (DEBUG|INFO|WARNING|ERROR) - ([\s\S]*)$/.exec(line)
        if (!m) {
            // Lines without the python-logging prefix: most commonly
            // BaseHTTPServer access log lines, occasional UserWarning
            // banners on startup. Just log them.
            pushLog(s, line)
            return
        }

        const level = m[1]
        const content = m[2]

        // httpcore / urllib3 plumbing at DEBUG — pure transport noise.
        if (level === "DEBUG" && /^(connect_tcp|start_tls|send_request|receive_response|response_closed)\./.test(content)) {
            return
        }

        // HF API call that fires on every model resolve.
        if (level === "INFO" && /^HTTP Request: GET https:\/\/huggingface\.co\//.test(content)) return

        if (!lui.engineReadyFired && level === "INFO") {
            const httpd = /^Starting httpd at (\S+) on port (\d+)/.exec(content)
            if (httpd) {
                s.listenUrl = `http://${httpd[1]}:${httpd[2]}`
                lui.markEngineReady()
            }
        }

        // Prompt cache snapshot. Keep the header, drop the per-role
        // breakdown that immediately follows so the log stays compact.
        // The breakdown is variable-length (mlx_lm currently lists
        // assistant/user/system; tool may join later) so we use a
        // flag cleared by the first non-matching INFO line rather than
        // a hardcoded count.
        const cache = /^Prompt Cache: (\d+) sequences, ([\d.]+) GB/.exec(content)
        if (cache && level === "INFO") {
            s.cacheSequences = parseInt(cache[1], 10) || 0
            s.cacheGiB = parseFloat(cache[2]) || 0
            s.swallowCacheRoles = true
            pushLog(s, line)
            return
        }
        if (s.swallowCacheRoles) {
            if (level === "INFO" && /^- \S+: \d+ sequences/.test(content)) return
            s.swallowCacheRoles = false
        }

        const prog = /^Prompt processing progress: (\d+)\/(\d+)/.exec(content)
        if (prog && level === "INFO") {
            s.promptProcessed = parseInt(prog[1], 10) || 0
            s.promptTotal = parseInt(prog[2], 10) || 0
            pushLog(s, line)
            return
        }

        // Generation lifecycle. "Starting stream/completion" marks the
        // boundary; for non-streaming requests "Outgoing Response:" is
        // the explicit end. For streaming requests there's no end
        // marker — generation just stops emitting tokens — so we
        // finalize lazily on the next Start.
        //
        // startMs is deliberately left null here; we anchor it on the
        // first token timestamp below so the duration covers decode
        // only and excludes the prompt-processing prefill window.
        if (level === "DEBUG" && (content === "Starting stream:" || content === "Starting completion:")) {
            finalizeGen(s)
            s.gen = { startMs: null, tokens: 0, lastTokenMs: null }
            return
        }
        if (level === "DEBUG" && content.startsWith("Outgoing Response:")) {
            finalizeGen(s)
            s.swallowBody = true
            return
        }
        if (level === "DEBUG" && content.startsWith("Incoming Request Body:")) {
            s.swallowBody = true
            return
        }

        // Once we're inside a generation, every other DEBUG line is a
        // single generated token's text. Counting them powers tok/s.
        // First token anchors the decode-only window.
        if (level === "DEBUG" && s.gen) {
            if (s.gen.startMs == null) s.gen.startMs = lineTs
            s.gen.tokens += 1
            s.gen.lastTokenMs = lineTs
            return
        }

        // Catch-all: a DEBUG line that didn't match any known marker
        // and isn't inside a generation. Mostly stray library debug
        // chatter; drop to keep the panel readable.
        if (level === "DEBUG") return

        pushLog(s, line)
    },

    appendPanels(v, lui) {
        appendEnginePanel(v, lui)
        appendPerformancePanel(v, lui)
        appendServerLogPanel(v, lui)
    }
}

function finalizeGen(s) {
    if (!s.gen || s.gen.tokens === 0 || s.gen.startMs == null) {
        s.gen = null
        return
    }
    const durationMs = Math.max(1, s.gen.lastTokenMs - s.gen.startMs)
    const tps = (s.gen.tokens / durationMs) * 1000
    s.lastGen = { tokens: s.gen.tokens, durationMs, tps }
    s.totalGenTokens += s.gen.tokens
    s.totalGenMs += durationMs
    s.gen = null
}

function isLoggerPrefixed(line) {
    return /^\d{4}-\d{2}-\d{2} \S+ - (DEBUG|INFO|WARNING|ERROR) - /.test(line)
}

function isAccessLog(line) {
    return /^\S+ - - \[[^\]]+\] "(POST|GET|OPTIONS) /.test(line)
}

// Python's `%Y-%m-%d %H:%M:%S,%f` prefix. We treat the wall-clock
// value as UTC since deltas are all we care about — picking a
// timezone keeps offline replays of debug.log deterministic and
// avoids any local-time skew between mlx_lm and lui.
function parseLineTimestamp(line) {
    const m = /^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}),(\d{3})/.exec(line)
    if (!m) return null
    return Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6], +m[7])
}

function pushLog(s, line) {
    if (s.logLines.length >= LOG_RING_SIZE) s.logLines.shift()
    s.logLines.push(line)
}

function binaryNameFromConfig(lui) {
    return lui.config.engine?.[engine.name]?.binary || BINARY_NAME
}

function appendEnginePanel(v, lui) {
    const s = lui.state
    const p = v.panel("mlx_lm")

    const aliasName = lui.activeModel?.name ?? ""
    const src = inferSource(lui.activeModel?.args || [])
    const modelLn = p.line().style(STYLE.LABEL).text("Model    : ").style()
    modelLn.text(src || "(no --model)")
    if (aliasName) {
        modelLn.style(STYLE.LABEL).text(" — ").style(STYLE.ALIAS).style({ bold: true }).text(aliasName).style()
    }

    p.line()

    const uptimeSec = Math.floor((Date.now() - (s.startedAt || Date.now())) / 1000)
    const statusLn = p.line().style(STYLE.LABEL).text("mlx_lm   : ").style()
    if (s.exited) {
        statusLn.style(STYLE.ERROR_INLINE).text("Exited").style()
        if (s.exitMessage) statusLn.text(`  ${s.exitMessage}`)
    } else if (lui.engineReadyFired) {
        statusLn.style(STYLE.READY).text("Ready").style().text(` (uptime: ${formatDurationSeconds(uptimeSec)})`)
    } else {
        statusLn.style(DIM).text("Starting…").style()
    }

    if (s.listenUrl) p.line({ indent: 15 }).style(DIM).text(s.listenUrl)

    if (s.argSegments) {
        const parts = s.argSegments.flatMap((seg) => seg.args)
        p.line({ indent: 15 }).style(DIM).text(parts.join(" "))
    }
}

function appendPerformancePanel(v, lui) {
    const s = lui.state
    const hasGen = s.gen || s.lastGen
    const hasCache = s.cacheGiB > 0 || s.cacheSequences > 0
    const hasProgress = s.promptTotal > 0 && s.promptProcessed < s.promptTotal
    if (!hasGen && !hasCache && !hasProgress) return

    const p = v.panel("Performance")

    if (hasCache) {
        p.line()
            .style(STYLE.LABEL)
            .text("Cache    : ")
            .style(STYLE.VALUE)
            .text(s.cacheGiB.toFixed(2))
            .style()
            .text(" GB ")
            .style(DIM)
            .text(`(${s.cacheSequences} seq)`)
    }

    // Active gen has a startMs only once the first token has arrived;
    // during prompt-processing prefill we suppress this line (the
    // Prompt processing bar below covers that phase).
    if (s.gen && s.gen.startMs != null) {
        const now = Date.now()
        const elapsedMs = Math.max(1, (s.gen.lastTokenMs || now) - s.gen.startMs)
        const tps = (s.gen.tokens / elapsedMs) * 1000
        p.line()
            .style(STYLE.LABEL)
            .text("Active   : ")
            .style(STYLE.VALUE)
            .text(tps.toFixed(1).padStart(6))
            .style()
            .text(" tok/s ")
            .style(DIM)
            .text(`(${s.gen.tokens} tokens in ${(elapsedMs / 1000).toFixed(1)}s)`)
    } else if (s.lastGen) {
        p.line()
            .style(STYLE.LABEL)
            .text("Last gen : ")
            .style(STYLE.VALUE)
            .text(s.lastGen.tps.toFixed(1).padStart(6))
            .style()
            .text(" tok/s ")
            .style(DIM)
            .text(`(${s.lastGen.tokens} tokens in ${(s.lastGen.durationMs / 1000).toFixed(1)}s)`)
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

    if (hasProgress) {
        p.bar({
            label: "Prompt processing",
            value: s.promptProcessed,
            max: s.promptTotal,
            text: `${s.promptProcessed}/${s.promptTotal}`,
            indent: 13
        })
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
        if (args[i] === "--model" && i + 1 < args.length) return `--model ${args[i + 1]}`
        if (args[i] === "--adapter-path" && i + 1 < args.length) return `--adapter-path ${args[i + 1]}`
    }
    return null
}
