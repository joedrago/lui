// remote engine — points lui at another lui's HTTP server.
//
// Args: a single positional HOST[:PORT] naming the upstream lui's
// /config endpoint (default :8081).
//
// Lifecycle:
//   start()  fetches /config once (learns base_url + context_size),
//            does one /data fetch to seed the panel cache, signals
//            engine-ready, then polls /data on a 250ms tick.
//   stop()   clears the poll timer.
//
// The harness baseURL it surfaces is exactly what upstream's /config
// reported — for a llama-server upstream that's "this host I dialed"
// + the engine port; for a remote upstream it's whatever *that*
// learned from its own upstream. The URL propagates verbatim through
// any number of hops, so turtles → relay → llm writes a harness on
// turtles pointing straight at llm.

/** @import { Engine, HostSpec } from "../types.js" */

import http from "node:http"

import { STYLE } from "../theme.js"
import { CONFIG_VERSION } from "../wire.js"

const POLL_MS = 250
const CONFIG_TIMEOUT_MS = 5000
const DATA_TIMEOUT_MS = 1500
const DEFAULT_LUI_PORT = 8081

/** @type {Engine} */
export const engine = {
    name: "remote",
    schema: [],

    describe(model) {
        const args = Array.isArray(model.args) ? model.args : []
        const errors = []
        if (args.length !== 1) {
            errors.push(`remote takes exactly one arg (HOST or HOST:PORT), got ${args.length}`)
        } else if (!parseHostSpec(args[0])) {
            errors.push(`remote: invalid HOST[:PORT] ${JSON.stringify(args[0])}`)
        }
        return {
            segments: [{ name: "user", style: STYLE.SEGMENT_USER, args: args.slice(0, 1) }],
            warnings: [],
            errors
        }
    },

    initState(lui) {
        const s = lui.state
        s.startedAt = Date.now()
        s.target = null
        s.baseURL = null
        s.ctxSize = 0
        s.remoteActiveModel = null
        s.servedModelName = null
        s.cachedView = null
        s.pollTimer = null
        s.connectError = null
        s.fatalReason = null
    },

    async start(lui, model, _desc) {
        const target = parseHostSpec(model.args[0])
        if (!target) throw new Error("remote: missing HOST")
        lui.state.target = target

        let cfg
        try {
            cfg = await fetchConfig(target)
        } catch (e) {
            lui.state.fatalReason = `could not reach ${target.host}:${target.port}: ${/** @type {Error} */ (e).message}`
            throw new Error(lui.state.fatalReason)
        }

        if (cfg.version !== CONFIG_VERSION) {
            lui.state.fatalReason =
                `remote /config version ${cfg.version}, this lui understands ${CONFIG_VERSION} ` + `— upgrade the older side`
            throw new Error(lui.state.fatalReason)
        }
        if (!cfg.base_url) {
            lui.state.fatalReason = `remote /config did not include base_url (upstream may not be ready yet)`
            throw new Error(lui.state.fatalReason)
        }

        lui.state.baseURL = cfg.base_url
        lui.state.ctxSize = cfg.context_size ?? 0
        lui.state.remoteActiveModel = cfg.active_model ?? null
        lui.state.servedModelName = cfg.served_model ?? null

        // Prime the panel cache before signaling ready so the TUI's
        // first tick isn't empty.
        try {
            lui.state.cachedView = await fetchData(target)
        } catch (e) {
            lui.state.connectError = /** @type {Error} */ (e).message || String(e)
        }

        lui.markEngineReady()

        lui.state.pollTimer = setInterval(async () => {
            try {
                lui.state.cachedView = await fetchData(target)
                lui.state.connectError = null
            } catch (e) {
                lui.state.connectError = /** @type {Error} */ (e).message || String(e)
            }
        }, POLL_MS)
    },

    async stop(lui) {
        if (lui.state?.pollTimer) {
            clearInterval(lui.state.pollTimer)
            lui.state.pollTimer = null
        }
    },

    contextSize(state) {
        return state && state.ctxSize > 0 ? state.ctxSize : null
    },

    // Forward whatever the upstream's /config reported, so a chained
    // lui → remote → … → mlx_lm still hands the right API id to the
    // harness on this hop.
    servedModelName(state) {
        return state?.servedModelName || null
    },

    // Where to actually reach the model. Parsed from the upstream's
    // base_url so callers (this lui's /config response, an `lui ssh`
    // tunnel command) can route directly to the real host instead of
    // proxying through this process.
    endpoint(lui) {
        const u = lui.state?.baseURL
        if (!u) return null
        try {
            const parsed = new URL(u)
            const port = parsed.port ? parseInt(parsed.port, 10) : parsed.protocol === "https:" ? 443 : 80
            return { host: parsed.hostname, port }
        } catch {
            return null
        }
    },

    exitReason(state) {
        if (state?.fatalReason) return state.fatalReason
        if (state?.connectError) return `disconnected: ${state.connectError}`
        return "stopped"
    },

    shutdownSummary(state) {
        const lines = []
        if (state?.target) lines.push({ label: "Upstream", value: `${state.target.host}:${state.target.port}` })
        if (state?.remoteActiveModel) lines.push({ label: "Remote model", value: state.remoteActiveModel })
        if (state?.baseURL) lines.push({ label: "Base URL", value: state.baseURL })
        return { lines, fatal: state?.fatalReason || null }
    },

    appendPanels(v, lui) {
        const cached = lui.state.cachedView
        if (cached?.panels?.length) {
            for (const panel of cached.panels) v.adoptPanel(panel)
            return
        }
        // No usable poll yet — show a small placeholder so the UI isn't
        // blank while we connect (or while we're between failed polls).
        const t = lui.state.target
        const p = v.panel("remote")
        const ln = p.line().style(STYLE.LABEL).text("Status   : ").style()
        if (lui.state.fatalReason) ln.style(STYLE.FATAL).text(lui.state.fatalReason).style()
        else if (lui.state.connectError) ln.style(STYLE.WARNING).text(lui.state.connectError).style()
        else ln.style({ dim: true }).text("connecting…").style()
        if (t) p.line().style(STYLE.LABEL).text("Upstream : ").style().text(`${t.host}:${t.port}`)
        if (lui.state.baseURL) p.line().style(STYLE.LABEL).text("Base URL : ").style().text(lui.state.baseURL)
    }
}

/** @param {string | null | undefined} s @returns {HostSpec | null} */
function parseHostSpec(s) {
    if (!s || typeof s !== "string") return null
    const i = s.lastIndexOf(":")
    if (i < 0) return s ? { host: s, port: DEFAULT_LUI_PORT } : null
    const host = s.slice(0, i)
    const port = parseInt(s.slice(i + 1), 10)
    if (!host || !Number.isFinite(port) || port <= 0 || port > 65535) return null
    return { host, port }
}

/** @param {HostSpec} target */
function fetchConfig(target) {
    return httpGetJSON(target, "/config", CONFIG_TIMEOUT_MS)
}

/** @param {HostSpec} target */
function fetchData(target) {
    return httpGetJSON(target, "/data", DATA_TIMEOUT_MS)
}

/** @param {HostSpec} target @param {string} path @param {number} timeoutMs @returns {Promise<any>} */
function httpGetJSON(target, path, timeoutMs) {
    return new Promise((resolve, reject) => {
        const req = http.get({ host: target.host, port: target.port, path, timeout: timeoutMs }, (res) => {
            let body = ""
            res.setEncoding("utf8")
            res.on("data", (c) => (body += c))
            res.on("end", () => {
                const status = res.statusCode ?? 0
                if (status < 200 || status >= 300) {
                    return reject(new Error(`${target.host}:${target.port}${path} returned HTTP ${status}`))
                }
                try {
                    resolve(JSON.parse(body))
                } catch (e) {
                    reject(new Error(`unparseable JSON from ${path}: ${/** @type {Error} */ (e).message}`))
                }
            })
        })
        req.on("error", (e) => reject(new Error(`${target.host}:${target.port}${path}: ${e.message}`)))
        req.on("timeout", () => req.destroy(new Error(`timeout fetching ${path} from ${target.host}:${target.port}`)))
    })
}
