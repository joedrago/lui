// mlx_lm.server engine. parseLine sifts the python-logging stream
// into tok/s and cache stats; appendPanels paints them. Ready
// detection waits for an active completion probe (not the bare
// "Starting httpd" line) because mlx_lm's HTTP server starts
// accepting connections before its worker thread has finished
// loading the model.

import http from "node:http"

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
            // ~14 GB on disk at 4-bit. Sampling params are Qwen3's
            // recommended thinking-mode preset; the 12 GB cache lets
            // long-context conversations stay warm.
            sizeGiB: 14,
            args: [
                "--model",
                "mlx-community/Qwen3.6-27B-4bit",
                "--prompt-cache-bytes",
                "12GB",
                "--temp",
                "0.6",
                "--top-k",
                "20",
                "--top-p",
                "0.95",
                "--min-p",
                "0.0"
            ]
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
        // TQDM_POSITION=-1 forces huggingface_hub's tqdm subclass to
        // emit progress bars even when stderr isn't a TTY (see
        // huggingface_hub/utils/tqdm.py:is_tqdm_disabled). Without
        // this we only see the snapshot_download wrapper bar and miss
        // the aggregated bytes bar (which is the actually-useful one
        // for shard downloads).
        const env = { ...process.env, TQDM_POSITION: "-1" }
        lui.state.proc = spawnProcess({
            binary: binaryPath,
            argv,
            env,
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

        // serverListening flips when mlx_lm logs "Starting httpd…";
        // engineReadyFired waits for the probe completion below to
        // actually return (i.e. the worker thread has loaded the
        // model and is processing requests). probeInProgress
        // suppresses gen tracking during the probe so its single
        // throwaway token doesn't anchor lastGen / Average.
        s.serverListening = false
        s.probeFired = false
        s.probeInProgress = false

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

        // filename → percent (0-100) for huggingface_hub downloads.
        // The "Fetching N files" wrapper bar is stored under the
        // sentinel key OVERALL_KEY so the panel can render it first.
        s.downloads = new Map()

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

        // tqdm progress lines from huggingface_hub. These arrive as
        // \r-overwriting updates (now split into separate parseLine
        // events by spawn.js), so we extract the latest percentage
        // and route to s.downloads. Both the "Fetching N files"
        // wrapper and individual file bars are handled.
        if (parseDownloadProgress(s, line)) return

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

        if (!s.serverListening && level === "INFO") {
            const httpd = /^Starting httpd at (\S+) on port (\d+)/.exec(content)
            if (httpd) {
                s.serverListening = true
                s.listenUrl = `http://${httpd[1]}:${httpd[2]}`
                // The HTTP server is now accepting connections but
                // the worker thread may still be loading the model.
                // Fire a probe completion and only mark Ready when it
                // returns — mlx_lm queues requests during load, so a
                // response means decode is actually online.
                if (!s.probeFired) {
                    s.probeFired = true
                    probeReady(lui).catch(() => lui.markEngineReady())
                }
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
            if (!s.probeInProgress) {
                s.promptProcessed = parseInt(prog[1], 10) || 0
                s.promptTotal = parseInt(prog[2], 10) || 0
            }
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
        //
        // During the readiness probe (a 1-token throwaway completion)
        // we skip all gen tracking — otherwise lastGen / Average get
        // anchored at the probe's "1 token in 0.0s = 1000 tok/s".
        if (level === "DEBUG" && (content === "Starting stream:" || content === "Starting completion:")) {
            if (s.probeInProgress) return
            finalizeGen(s)
            s.gen = { startMs: null, tokens: 0, lastTokenMs: null }
            return
        }
        if (level === "DEBUG" && content.startsWith("Outgoing Response:")) {
            if (!s.probeInProgress) finalizeGen(s)
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
        if (level === "DEBUG" && s.gen && !s.probeInProgress) {
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

const OVERALL_KEY = "__overall__"

// tqdm shapes from huggingface_hub:
//
//   wrapper:  "Fetching 14 files:  35%|███       | 5/14 [..]"
//   per-file: "model.safetensors:  18%|█▊        | 2.25G/12.5G [..]"
//   truncated: "(…)f-00009.safetensors:  18%|█▊        | 2.25G/12.5G [..]"
//
// We try the wrapper first; anything else with the `NAME: NN%|…|`
// shape is treated as a per-file bar. The per-file regex
// deliberately accepts any name (including the "(…)" ellipsis tqdm
// inserts when the desc is too long for the terminal) so we don't
// silently miss progress for repos with long shard names. Either
// match suppresses the line from the log ring and the panel renders
// an active bar instead.
function parseDownloadProgress(s, line) {
    const wrapper = /Fetching (\d+) files:\s+(\d+)%\|[^|]*\|\s*(\d+)\/(\d+)/.exec(line)
    if (wrapper) {
        const pct = parseInt(wrapper[2], 10) || 0
        const cur = parseInt(wrapper[3], 10) || 0
        const total = parseInt(wrapper[4], 10) || 0
        s.downloads.set(OVERALL_KEY, { label: `Fetching ${total} files (${cur}/${total})`, pct })
        if (pct >= 100) s.downloads.delete(OVERALL_KEY)
        return true
    }
    // Reject "Fetching N files" first so the generic shape doesn't
    // pick it up; the wrapper case above handles it explicitly.
    const perFile = /^([^:\r\n]+?):\s+(\d+)%\|[^|]*\|\s*([\d.]+\s*[KMGT]?i?B(?:\/[\d.]+\s*[KMGT]?i?B)?)?/.exec(line)
    if (perFile && !/^Fetching \d+ files/.test(perFile[1])) {
        let name = perFile[1].replace(/^\(…\)|^\(\.\.\.\)/, "…")
        // huggingface_hub's aggregated bytes bar starts life with
        // desc="Downloading (incomplete total...)" and renames to
        // "Download complete" at the end. Collapse both to one key so
        // the rename doesn't double-list the bar, and shorten the
        // verbose initial desc for display.
        if (/^Downloading|^Download complete/.test(name)) name = "Downloading"
        const pct = parseInt(perFile[2], 10) || 0
        const sizes = perFile[3] || ""
        if (pct >= 100) s.downloads.delete(name)
        else {
            const prev = s.downloads.get(name)?.pct ?? 0
            if (pct >= prev) {
                const label = sizes ? `${name}  ${sizes}` : name
                s.downloads.set(name, { label, pct })
            }
        }
        return true
    }
    return false
}

// Probe the engine with the smallest possible chat completion so we
// can tell when the worker thread has finished loading. mlx_lm.server
// holds the request until model load completes; the response (any
// status code) confirms decode is online. On connection or transport
// failure we still fire markEngineReady so the harness gets
// configured — degrading to old behavior is better than wedging the
// UI on "Loading model…" forever.
//
// probeInProgress + the post-probe stat reset together keep the
// probe's one-token completion from anchoring lastGen / Average:
// the flag suppresses gen tracking inline, and the reset wipes any
// state that slipped through pipe-ordering races between the HTTP
// response and the final stderr log lines.
async function probeReady(lui) {
    const s = lui.state
    s.probeInProgress = true

    const port = lui.config.global.engine_port
    const modelName = engine.servedModelName(s, lui.activeModel) || "default"
    const body = JSON.stringify({
        model: modelName,
        messages: [{ role: "user", content: "." }],
        max_tokens: 1,
        temperature: 0,
        stream: false
    })

    function settle() {
        s.gen = null
        s.lastGen = null
        s.totalGenTokens = 0
        s.totalGenMs = 0
        s.promptProcessed = 0
        s.promptTotal = 0
        s.swallowBody = false
        s.probeInProgress = false
        lui.markEngineReady()
    }

    return new Promise((resolve) => {
        const req = http.request(
            {
                host: "127.0.0.1",
                port,
                path: "/v1/chat/completions",
                method: "POST",
                headers: {
                    "content-type": "application/json",
                    "content-length": Buffer.byteLength(body)
                }
            },
            (res) => {
                res.on("data", () => {})
                res.on("end", () => {
                    settle()
                    resolve()
                })
            }
        )
        req.on("error", () => {
            settle()
            resolve()
        })
        req.write(body)
        req.end()
    })
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
    } else if (s.serverListening) {
        statusLn.style(DIM).text("Loading model…").style()
    } else {
        statusLn.style(DIM).text("Starting…").style()
    }

    if (s.listenUrl) p.line({ indent: 15 }).style(DIM).text(s.listenUrl)

    if (s.argSegments) {
        const parts = s.argSegments.flatMap((seg) => seg.args)
        p.line({ indent: 15 }).style(DIM).text(parts.join(" "))
    }

    // Active huggingface_hub downloads. Wrapper bar first (when
    // present) then per-file bars in stable sort order so the panel
    // doesn't jitter as keys arrive.
    const overall = s.downloads.get(OVERALL_KEY)
    if (overall) {
        p.bar({
            label: overall.label,
            value: overall.pct,
            max: 100,
            text: `${String(overall.pct).padStart(3)}%`,
            indent: 13
        })
    }
    const files = [...s.downloads.entries()].filter(([k]) => k !== OVERALL_KEY).sort(([a], [b]) => a.localeCompare(b))
    for (const [, entry] of files) {
        p.bar({
            label: entry.label,
            value: entry.pct,
            max: 100,
            text: `${String(entry.pct).padStart(3)}%`,
            indent: 13
        })
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
