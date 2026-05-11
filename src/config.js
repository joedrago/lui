// Config IO. Loads ~/.config/lui.toml via smol-toml into the runtime
// shape (lui.config.global.*, lui.config.models.*); saves it back with
// an atomic tmp+rename. TOML emission is bespoke so `args` arrays
// render one-entry-per-line for diff-friendly storage.

import fs from "node:fs"
import path from "node:path"
import os from "node:os"
import { parse } from "smol-toml"

export const CONFIG_PATH = path.join(os.homedir(), ".config", "lui.toml")

export const DEFAULTS = {
    engine_port: 8080,
    web_port: 8081,
    websearch: true
}

export class Config {
    constructor(data = {}) {
        const g = data.global ?? {}
        this.global = {
            engine_port: g.engine_port ?? DEFAULTS.engine_port,
            web_port: g.web_port ?? DEFAULTS.web_port,
            websearch: g.websearch ?? DEFAULTS.websearch,
            ...(g.active_model != null ? { active_model: g.active_model } : {}),
            harness: g.harness ?? {},
            engines: g.engines ?? {}
        }
        this.models = data.models ?? {}
    }

    static load() {
        return Config.loadFrom(CONFIG_PATH)
    }

    static loadFrom(p) {
        if (!fs.existsSync(p)) return new Config()
        let text
        try {
            text = fs.readFileSync(p, "utf8")
        } catch (e) {
            process.stderr.write(`lui: failed to read ${p}: ${e.message}\n`)
            return new Config()
        }
        let data
        try {
            data = parse(text)
        } catch (e) {
            process.stderr.write(`lui: failed to parse ${p}: ${e.message}\n`)
            return new Config()
        }
        return new Config(data)
    }

    save() {
        Config.saveTo(this, CONFIG_PATH)
    }

    static saveTo(cfg, p) {
        fs.mkdirSync(path.dirname(p), { recursive: true })
        const tmp = p + ".tmp"
        fs.writeFileSync(tmp, serialize(cfg))
        fs.renameSync(tmp, p)
    }

    // Convenience accessors used elsewhere — keeping them as methods is
    // cleaner than reaching into nested objects from every call site.
    get activeModelName() {
        return this.global.active_model || null
    }
    setActiveModel(name) {
        this.global.active_model = name
    }
}

function serialize(cfg) {
    const out = []
    out.push(...emitTable("global", cfg.global, ["harness", "engines"]))

    const harness = cfg.global?.harness || {}
    for (const name of Object.keys(harness).sort()) {
        const sub = harness[name]
        if (sub && typeof sub === "object" && !Array.isArray(sub)) {
            out.push("")
            out.push(...emitTable(`global.harness.${tomlKey(name)}`, sub))
        }
    }

    const engines = cfg.global?.engines || {}
    for (const name of Object.keys(engines).sort()) {
        const sub = engines[name]
        if (sub && typeof sub === "object" && !Array.isArray(sub)) {
            out.push("")
            out.push(...emitTable(`global.engines.${tomlKey(name)}`, sub))
        }
    }

    for (const name of Object.keys(cfg.models).sort()) {
        out.push("")
        out.push(...emitTable(`models.${tomlKey(name)}`, cfg.models[name]))
    }

    return out.join("\n") + "\n"
}

function emitTable(header, obj, skipNested = []) {
    const lines = [`[${header}]`]
    if (!obj || typeof obj !== "object") return lines
    const scalarKeys = []
    const arrayKeys = []
    for (const [k, v] of Object.entries(obj)) {
        if (skipNested.includes(k)) continue
        if (v == null) continue
        if (typeof v === "object" && !Array.isArray(v)) continue
        if (Array.isArray(v)) arrayKeys.push(k)
        else scalarKeys.push(k)
    }
    for (const k of scalarKeys) lines.push(`${tomlKey(k)} = ${tomlScalar(obj[k])}`)
    for (const k of arrayKeys) lines.push(...emitArrayLine(k, obj[k]))
    return lines
}

function emitArrayLine(key, arr) {
    if (arr.length === 0) return [`${tomlKey(key)} = []`]
    const allStrings = arr.every((v) => typeof v === "string")
    if (allStrings && arr.length > 1) {
        const inner = arr.map((s) => `    ${tomlString(s)}`).join(",\n")
        return [`${tomlKey(key)} = [`, inner, `]`]
    }
    return [`${tomlKey(key)} = [${arr.map(tomlScalar).join(", ")}]`]
}

function tomlScalar(v) {
    if (typeof v === "string") return tomlString(v)
    if (typeof v === "boolean") return v ? "true" : "false"
    if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(v)
    if (Array.isArray(v)) return `[${v.map(tomlScalar).join(", ")}]`
    return tomlString(String(v))
}

function tomlString(s) {
    return (
        '"' +
        s.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t") +
        '"'
    )
}

function tomlKey(k) {
    if (/^[A-Za-z0-9_-]+$/.test(k)) return k
    return tomlString(k)
}
