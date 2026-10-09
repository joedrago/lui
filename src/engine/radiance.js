// radiance engine — codeberg.org/StillDeadcode/radiance, a C++/HIP server
// built around RDNA4 cards (R9700). Symlink build/bin/radiance onto PATH
// like llama-server; a build tree finds its own plugins when configured
// with -DRAD_DEFAULT_HOME=<tree>/build/radiance_home, otherwise set
// RADIANCE_HOME with `lui env NAME RADIANCE_HOME=...`.
//
// Unlike llama-server, radiance's log says little per request — what it
// knows lives behind HTTP: /v1/models (served id + max_model_len) and
// /stats (the JSON its own dashboard draws from). So readiness is a probe:
// the listener only comes up once the model is loaded and the budgets are
// settled, and the first /v1/models answer marks Ready. After that /stats
// is polled for the panels. The log is still tailed for the Server Log
// panel and to surface the line that explains a failed start.

/** @import { Engine, ViewBuilder } from "../types.js" */
/** @import { Lui } from "../lui.js" */

import { STYLE } from "../theme.js"
import { stripAnsi } from "../ansi.js"
import { resolveBinary, spawnProcess, describeSpawnError } from "../spawn.js"
import { formatBytes, formatDurationSeconds, formatNumber } from "../util.js"

const BINARY_NAME = "radiance"

const DIM = { dim: true }
const TEXT = {}

const LOG_RING_SIZE = 200
const PROBE_MS = 1000
const PROBE_TIMEOUT_MS = 2000

// Flags lui owns: it binds the engine to engine_port and to public/private.
const RESERVED_FLAGS = new Set(["--host", "--port"])

/** @param {string[]} args @param {string} flag @returns {string | null} */
function argValue(args, flag) {
    for (let i = 0; i < args.length; i++) {
        if (args[i] === flag && i + 1 < args.length) return args[i + 1]
        if (args[i].startsWith(flag + "=")) return args[i].slice(flag.length + 1)
    }
    return null
}

// What --model names, shortened for the panel: a repo id stays as it is,
// a path keeps only its file name.
/** @param {string | null} m @returns {string | null} */
function modelLabel(m) {
    if (!m) return null
    if (m.startsWith("/") || m.startsWith("~") || m.startsWith(".")) return m.split("/").filter(Boolean).pop() || m
    return m
}

/** @type {Engine} */
export const engine = {
    name: "radiance",

    schema: [{ path: "binary", default: BINARY_NAME }],

    shutdownSummary(state) {
        const lines = []
        const logLines = state?.logLines ?? []
        const exitCode = state?.exitCode
        const exitSignal = state?.exitSignal
        if ((exitCode != null && exitCode !== 0) || exitSignal) {
            // radiance explains a refused start on its E lines (a budget
            // that does not fit, a kernel that fell back to libref, ...);
            // those say more than the last few lines of narration.
            const errs = logLines.filter(/** @param {string} l */ (l) => /^E /.test(l)).slice(-5)
            const tail = errs.length ? errs : logLines.slice(-5)
            if (tail.length > 0) lines.push({ label: "Last log lines", value: tail.join("\n") })
        }
        return { lines, fatal: state?.fatalReason || null }
    },

    exitReason(state, code, signal) {
        if (state?.exitMessage) return state.exitMessage
        return signal ? `killed by ${signal}` : `exited with code ${code}`
    },

    // The served max_model_len once /v1/models has answered, else whatever
    // --max-model-len says (0 there means "the model's training context",
    // which only the running server can name).
    contextSize(state, model) {
        if (state && state.ctxSize > 0) return state.ctxSize
        const n = parseInt(argValue(model?.args || [], "--max-model-len") ?? "", 10)
        return Number.isFinite(n) && n > 0 ? n : null
    },

    // radiance answers any model id a request sends; this is what
    // /v1/models reports, so a harness that lists models sees a match.
    servedModelName(state) {
        return state?.servedModelName || null
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
            const flag = tok.split("=")[0]
            if (RESERVED_FLAGS.has(flag)) errors.push(`${flag} is reserved by lui — drop it from this model's args`)
        }
        if (!argValue(userArgs, "--model")) {
            errors.push("radiance needs --model <a .rad container, a checkpoint directory, or an HF repo id>")
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
            onExit: (code, signal) => {
                lui.state.exited = true
                stopProbe(lui)
                lui.onEngineExit?.(code, signal)
            },
            onSpawnError: (err) => {
                const msg = describeSpawnError(binaryName, err)
                lui.state.exitMessage = msg
                lui.state.fatalReason = msg
            },
            addWarning: (m) => lui.addWarning(m)
        })
        startProbe(lui)
    },

    async stop(lui) {
        stopProbe(lui)
        await lui.state?.proc?.stop?.()
    },

    initState(lui) {
        const s = lui.state
        s.startedAt = Date.now()
        s.argSegments = null
        s.proc = null
        s.exited = false
        s.exitMessage = ""
        s.fatalReason = null
        s.logLines = []
        s.probeTimer = null
        s.probeBusy = false

        // From the log, before Ready: the last informative line (the hub
        // check and fetch when --model names a repo id, then the load).
        s.loadStatus = ""

        // From /v1/models and /version, at Ready.
        s.version = ""
        s.ctxSize = 0
        s.servedModelName = null

        // From /stats, after Ready.
        s.stats = null
        // Per-request rates from /stats' request table, kept once the
        // request is gone so the panel shows the last one's speed. The
        // engine's own prefill_tps/decode_tps are decaying averages that
        // fade toward zero once it goes idle, so they make a poor "last".
        s.livePrefillTps = 0
        s.liveDecodeTps = 0
        s.lastPrefillTps = 0
        s.lastDecodeTps = 0
    },

    parseLine(rawLine, lui) {
        const line = stripAnsi(rawLine)
        if (!line) return
        const s = lui.state
        pushLog(s, line)

        if (!lui.engineReadyFired && /^[IW] /.test(line)) s.loadStatus = line.slice(2)
        if (/^E /.test(line) && !lui.engineReadyFired) s.fatalReason = line.slice(2)
    },

    appendPanels(v, lui) {
        appendEnginePanel(v, lui)
        appendPerformancePanel(v, lui)
        appendServerLogPanel(v, lui)
    }
}

/** @param {Lui} lui @param {string} path @returns {Promise<any | null>} */
async function getJson(lui, path) {
    const url = `http://127.0.0.1:${lui.config.global.engine_port}${path}`
    try {
        const r = await fetch(url, { signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) })
        if (!r.ok) return null
        return await r.json()
    } catch {
        return null
    }
}

/** @param {Lui} lui */
function startProbe(lui) {
    const s = lui.state
    const tick = async () => {
        if (s.probeBusy || s.exited) return
        s.probeBusy = true
        try {
            if (!lui.engineReadyFired) {
                const models = await getJson(lui, "/v1/models")
                const m = models?.data?.[0]
                if (m) {
                    s.servedModelName = m.id || null
                    s.ctxSize = Number(m.max_model_len) || 0
                    s.loadStatus = ""
                    s.version = (await getJson(lui, "/version"))?.version || ""
                    lui.markEngineReady()
                }
            } else {
                const st = await getJson(lui, "/stats")
                if (st) {
                    s.stats = st
                    trackRates(s, Array.isArray(st.requests) ? st.requests : [])
                }
            }
        } finally {
            s.probeBusy = false
        }
    }
    s.probeTimer = setInterval(tick, PROBE_MS)
}

// A prefilling request's rate is what it has computed beyond its cache hit
// over its age; a decoding one reports its own average since first token.
/** @param {any} s @param {any[]} reqs */
function trackRates(s, reqs) {
    s.livePrefillTps = 0
    s.liveDecodeTps = 0
    for (const q of reqs) {
        if (q.computed < q.prompt) {
            const fresh = q.computed - (q.cached || 0)
            if (q.age_s > 0 && fresh > 0) s.livePrefillTps += fresh / q.age_s
        } else if (q.tps > 0) {
            s.liveDecodeTps += q.tps
        }
    }
    if (s.livePrefillTps > 0) s.lastPrefillTps = s.livePrefillTps
    if (s.liveDecodeTps > 0) s.lastDecodeTps = s.liveDecodeTps
}

/** @param {Lui} lui */
function stopProbe(lui) {
    if (lui.state?.probeTimer) {
        clearInterval(lui.state.probeTimer)
        lui.state.probeTimer = null
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

/** @param {ViewBuilder} v @param {Lui} lui */
function appendEnginePanel(v, lui) {
    const s = lui.state
    const st = s.stats
    const p = v.panel("radiance")

    const aliasName = lui.activeModel?.name ?? ""
    const modelLn = p.line().style(STYLE.LABEL).text("Model    : ").style()
    modelLn.text(s.servedModelName || modelLabel(argValue(lui.activeModel?.args || [], "--model")) || "(loading...)")
    if (aliasName) {
        modelLn.style(STYLE.LABEL).text(" — ").style(STYLE.ALIAS).style({ bold: true }).text(aliasName).style()
    }
    if (s.ctxSize > 0)
        p.line({ indent: 15 })
            .style(DIM)
            .text(`${formatNumber(s.ctxSize)} token context window`)

    p.line()

    const uptimeSec = Math.floor((Date.now() - (s.startedAt || Date.now())) / 1000)
    const statusLn = p.line().style(STYLE.LABEL).text("radiance : ").style()
    if (s.exited) {
        statusLn.style(STYLE.ERROR_INLINE).text("Exited").style()
        if (s.exitMessage) statusLn.text(`  ${s.exitMessage}`)
    } else if (lui.engineReadyFired) {
        statusLn.style(STYLE.READY).text("Ready").style()
        const ver = s.version ? `${s.version}, ` : ""
        statusLn.text(` (${ver}uptime: ${formatDurationSeconds(uptimeSec)})`)
    } else {
        statusLn.style(DIM).text(`Starting... (${uptimeSec}s)`).style()
        if (s.loadStatus) p.line({ indent: 15, nowrap: true }).style(DIM).text(s.loadStatus.slice(0, 200))
    }

    if (s.argSegments) {
        const parts = s.argSegments.flatMap(/** @param {import("../types.js").Segment} seg */ (seg) => seg.args)
        p.line({ indent: 15 }).style(DIM).text(parts.join(" "))
    }

    if (!st) return

    for (const c of st.cards ?? []) {
        if (!(c.vram_total > 0)) continue
        const extra = [c.temp_c > 0 ? `${Math.round(c.temp_c)}°C` : null, c.power_w > 0 ? `${Math.round(c.power_w)} W` : null]
            .filter(Boolean)
            .join(" · ")
        p.bar({
            label: `VRAM card ${c.index}`,
            value: c.vram_used / c.vram_total,
            text: `${formatBytes(c.vram_used)} / ${formatBytes(c.vram_total)}${extra ? ` · ${extra}` : ""}`,
            indent: 13
        })
    }
    if (st.kv_tokens_total > 0) {
        const held = st.kv_tokens_cached > 0 ? ` · ${formatNumber(st.kv_tokens_cached)} held for next turn` : ""
        p.bar({
            label: "KV cache",
            value: st.kv_tokens_used / st.kv_tokens_total,
            text: `${formatNumber(st.kv_tokens_used)} / ${formatNumber(st.kv_tokens_total)} tok${held}`,
            indent: 13
        })
    }
    const ex = st.experts
    if (ex && ex.bytes > 0 && Array.isArray(ex.tiers)) {
        const vram = ex.tiers.find(/** @param {any} t */ (t) => /vram/i.test(t.name))
        if (vram) {
            p.bar({
                label: "Experts in VRAM",
                value: vram.bytes / ex.bytes,
                text: `${formatBytes(vram.bytes)} / ${formatBytes(ex.bytes)}`,
                indent: 13
            })
        }
    }
}

/** @param {ViewBuilder} v @param {Lui} lui */
function appendPerformancePanel(v, lui) {
    const s = lui.state
    const st = s.stats
    if (!st) return
    const p = v.panel("Performance")

    /** @param {string} label @param {number} live @param {number} last */
    const rateLine = (label, live, last) => {
        const ln = p.line().style(STYLE.LABEL).text(label).style(STYLE.VALUE)
        if (last > 0) {
            ln.text((live > 0 ? live : last).toFixed(1).padStart(6))
                .style(TEXT)
                .text(" tok/s ")
                .style(DIM)
                .text(live > 0 ? "" : "(last)")
        } else {
            ln.style(DIM).text("--")
        }
    }
    rateLine("Prompt   : ", s.livePrefillTps, s.lastPrefillTps)
    rateLine("Generate : ", s.liveDecodeTps, s.lastDecodeTps)

    if (st.draft_accept > 0) {
        const pos = Array.isArray(st.draft_positions)
            ? st.draft_positions.map(/** @param {any} d */ (d) => `${Math.round(d.rate * 100)}%`).join(" / ")
            : ""
        p.line()
            .style(STYLE.LABEL)
            .text("Spec     : ")
            .style(STYLE.VALUE)
            .text((st.draft_accept * 100).toFixed(1).padStart(6))
            .style(TEXT)
            .text("%       ")
            .style(DIM)
            .text(pos ? `(by position ${pos})` : "")
    }
    if (st.total_requests > 0) {
        const hit = st.prefix_hit_rate > 0 ? ` · prefix hit ${(st.prefix_hit_rate * 100).toFixed(1)}%` : ""
        p.line({ indent: 11 })
            .style(DIM)
            .text(`${formatNumber(st.total_requests)} requests${hit}`)
    }

    const reqs = Array.isArray(st.requests) ? st.requests : []
    if (reqs.length > 0) p.line()
    for (const q of reqs) {
        if (q.computed < q.prompt) {
            p.bar({
                label: `● prompt ${String(q.prompt).padStart(7)} tokens`,
                value: q.prompt > 0 ? q.computed / q.prompt : 0,
                text: `${formatNumber(q.computed)}/${formatNumber(q.prompt)}${q.cached > 0 ? ` (${formatNumber(q.cached)} cached)` : ""}`,
                indent: 13
            })
        } else {
            const tps = q.tps > 0 ? ` · ${q.tps.toFixed(1)} tok/s` : ""
            p.bar({
                label: `● output ${String(q.output).padStart(7)} tokens`,
                value: q.max > 0 ? q.output / q.max : 0,
                text: `${formatNumber(q.ctx)} ctx${tps}`,
                indent: 13
            })
        }
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
