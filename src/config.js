// Config IO at ~/.config/lui.toml. Atomic tmp+rename on save.
// Also: `lui set` / `lui unset` command handlers (bare forms dump).

/** @import { SchemaEntry } from "./types.js" */
/** @import { Lui } from "./lui.js" */

import fs from "node:fs"
import path from "node:path"
import os from "node:os"
import process from "node:process"
import { parse } from "smol-toml"

import { engineSchemaDefaults } from "./engine.js"
import { DEFAULT_MAX_OUTPUT_TOKENS } from "./wire.js"
import { harnessSchemaDefaults } from "./harness.js"
import { sandboxSchemaDefaults } from "./sandbox.js"
import { styled } from "./ansi.js"
import { STYLE } from "./theme.js"

export const CONFIG_PATH = path.join(os.homedir(), ".config", "lui.toml")

// Top-level TOML tables lui knows about. `global` settings are the bare
// keys at the file root; the rest are nested. `model` is user-data
// (added/removed via `lui add/rm`), not a schema-controlled setting,
// but it's still a valid path prefix for `lui set model.X.Y ...`.
export const TOP_LEVEL_TABLES = ["global", "model", "harness", "engine", "sandbox"]

// Schema entries surfaced by the config dump. Each is
// {path, default, isArray?}. `isArray` marks list-typed paths where
// `set` appends and `unset` drops the whole list. The default values
// here are the single source of truth — Config.global is seeded from
// them, and the dump renders unset entries with their defaults in
// place.
/** @returns {SchemaEntry[]} */
export function globalSchemaDefaults() {
    return [
        { path: "engine_port", default: 8080 },
        { path: "web_port", default: 8081 },
        { path: "websearch", default: true },
        { path: "public", default: false },
        { path: "debug_log", default: null },

        // Host-authoritative: the machine actually running the model
        // picks this, and it rides /config out to every attached lui.
        // Setting it on a client running the `remote` engine does
        // nothing — that hop honors whatever upstream reported.
        { path: "max_output_tokens", default: DEFAULT_MAX_OUTPUT_TOKENS }
    ]
}

/** @returns {Record<string, any>} */
function globalDefaults() {
    /** @type {Record<string, any>} */
    const out = {}
    for (const { path, default: def } of globalSchemaDefaults()) {
        if (def == null) continue
        if (path.includes(".")) continue
        out[path] = def
    }
    return out
}

export class Config {
    /** @param {Record<string, any>} [data] */
    constructor(data = {}) {
        // Drop anything not in the schema — retired settings, typos, or
        // values written by an older build with a different vocabulary
        // don't survive a load. `lui set` is gated against the same
        // schema, so the in-memory shape stays honest.
        const clean = filterToSchema(data)
        /** @type {Record<string, any>} */
        this.global = { ...globalDefaults(), ...clean.global }
        // Top-level catalogs / tuning tables. Each is its own root in
        // the TOML — [harness.opencode], [engine.llama-server],
        // [sandbox], [model.phi] — so they sit visually as peers.
        /** @type {Record<string, any>} */
        this.harness = clean.harness
        /** @type {Record<string, any>} */
        this.engine = clean.engine
        /** @type {Record<string, any>} */
        this.sandbox = clean.sandbox
        /** @type {Record<string, { engine: string, args: string[] }>} */
        this.model = clean.model
    }

    static load() {
        return Config.loadFrom(CONFIG_PATH)
    }

    /** @param {string} p */
    static loadFrom(p) {
        if (!fs.existsSync(p)) return new Config()
        let text
        try {
            text = fs.readFileSync(p, "utf8")
        } catch (e) {
            process.stderr.write(`lui: failed to read ${p}: ${/** @type {Error} */ (e).message}\n`)
            return new Config()
        }
        let data
        try {
            data = parse(text)
        } catch (e) {
            process.stderr.write(`lui: failed to parse ${p}: ${/** @type {Error} */ (e).message}\n`)
            return new Config()
        }
        return new Config(data)
    }

    save() {
        Config.saveTo(this, CONFIG_PATH)
    }

    /** @param {Config} cfg @param {string} p */
    static saveTo(cfg, p) {
        fs.mkdirSync(path.dirname(p), { recursive: true })
        const tmp = p + ".tmp"
        fs.writeFileSync(tmp, serialize(cfg))
        fs.renameSync(tmp, p)
    }
}

/** @param {Config} cfg @returns {string} */
function serialize(cfg) {
    const out = []

    // Belt-and-braces: catch any object-valued child of [global].
    const globalNested = []
    for (const [k, v] of Object.entries(cfg.global || {})) {
        if (v && typeof v === "object" && !Array.isArray(v)) globalNested.push(k)
    }
    out.push(...emitTable("global", cfg.global, globalNested))

    for (const groupKey of globalNested.sort()) {
        const group = cfg.global[groupKey]
        if (!group || Object.keys(group).length === 0) continue
        out.push("")
        out.push(...emitTable(`global.${groupKey}`, group))
    }

    out.push(...emitTopLevelTable(cfg, "harness"))
    out.push(...emitTopLevelTable(cfg, "engine"))
    out.push(...emitTopLevelTable(cfg, "sandbox"))
    out.push(...emitTopLevelTable(cfg, "model"))

    return out.join("\n") + "\n"
}

// `harness` / `engine` / `model` are map-of-tables; `sandbox` is a leaf
// table. Detect from the children's shape so new top-levels just work.
/** @param {Config} cfg @param {string} rootKey @returns {string[]} */
function emitTopLevelTable(cfg, rootKey) {
    const obj = /** @type {Record<string, any>} */ (/** @type {any} */ (cfg)[rootKey])
    if (!obj || typeof obj !== "object") return []
    const inner = Object.entries(obj)
    if (inner.length === 0) return []
    const allInnerAreTables = inner.every(([, v]) => v && typeof v === "object" && !Array.isArray(v))
    const out = []
    if (allInnerAreTables) {
        for (const name of Object.keys(obj).sort()) {
            out.push("")
            out.push(...emitTable(`${rootKey}.${tomlKey(name)}`, obj[name]))
        }
    } else {
        out.push("")
        out.push(...emitTable(rootKey, obj))
    }
    return out
}

/** @param {string} header @param {Record<string, any>} obj @param {string[]} [skipNested] @returns {string[]} */
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

/** @param {string} key @param {any[]} arr @returns {string[]} */
function emitArrayLine(key, arr) {
    if (arr.length === 0) return [`${tomlKey(key)} = []`]
    const allStrings = arr.every((v) => typeof v === "string")
    if (allStrings && arr.length > 1) {
        const inner = arr.map((s) => `    ${tomlString(s)}`).join(",\n")
        return [`${tomlKey(key)} = [`, inner, `]`]
    }
    return [`${tomlKey(key)} = [${arr.map(tomlScalar).join(", ")}]`]
}

/** @param {any} v @returns {string} */
function tomlScalar(v) {
    if (typeof v === "string") return tomlString(v)
    if (typeof v === "boolean") return v ? "true" : "false"
    if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(v)
    if (Array.isArray(v)) return `[${v.map(tomlScalar).join(", ")}]`
    return tomlString(String(v))
}

/** @param {string} s @returns {string} */
function tomlString(s) {
    return (
        '"' +
        s.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t") +
        '"'
    )
}

/** @param {string} k @returns {string} */
function tomlKey(k) {
    if (/^[A-Za-z0-9_-]+$/.test(k)) return k
    return tomlString(k)
}

// ─── `lui set` / `lui unset` CLI ─────────────────────────────────────

/** @returns {SchemaEntry[]} */
function allSchemaDefaults() {
    return [...globalSchemaDefaults(), ...harnessSchemaDefaults(), ...engineSchemaDefaults(), ...sandboxSchemaDefaults()]
}

const TOP_LEVEL_KEYS = new Set(TOP_LEVEL_TABLES)

/** @param {string[]} path @returns {boolean} */
function isArrayPath(path) {
    const joined = path.join(".")
    return allSchemaDefaults().some((s) => s.isArray && s.path === joined)
}

// Globals live at bare keys in the schema ([global] is implicit), so
// "global.engine_port" looks up as "engine_port"; everything else uses
// its fully dotted form. `model.*` is user-data managed by add/set/rm
// — it bypasses schema entirely.
/** @param {string[]} path @returns {boolean} */
function isSchemaAllowed(path) {
    if (path[0] === "model") return true
    const lookup = path[0] === "global" ? path.slice(1).join(".") : path.join(".")
    return allSchemaDefaults().some((s) => s.path === lookup)
}

// Strip parsed TOML data down to schema-known keys. `model` passes
// through (it has no schema). Used by the Config constructor so any
// retired or mistyped key disappears on the next save — the in-memory
// shape is always what the file *should* contain.
/** @param {Record<string, any>} data @returns {{ global: Record<string, any>, harness: Record<string, any>, engine: Record<string, any>, sandbox: Record<string, any>, model: Record<string, any> }} */
function filterToSchema(data) {
    const known = new Set(allSchemaDefaults().map((s) => s.path))
    /** @type {{ global: Record<string, any>, harness: Record<string, Record<string, any>>, engine: Record<string, Record<string, any>>, sandbox: Record<string, any>, model: Record<string, any> }} */
    const out = { global: {}, harness: {}, engine: {}, sandbox: {}, model: data.model ?? {} }
    for (const [k, v] of Object.entries(data.global ?? {})) {
        if (known.has(k)) out.global[k] = v
    }
    for (const [name, sub] of Object.entries(data.harness ?? {})) {
        if (!sub || typeof sub !== "object" || Array.isArray(sub)) continue
        for (const [k, v] of Object.entries(/** @type {Record<string, any>} */ (sub))) {
            if (known.has(`harness.${name}.${k}`)) (out.harness[name] ??= {})[k] = v
        }
    }
    for (const [name, sub] of Object.entries(data.engine ?? {})) {
        if (!sub || typeof sub !== "object" || Array.isArray(sub)) continue
        for (const [k, v] of Object.entries(/** @type {Record<string, any>} */ (sub))) {
            if (known.has(`engine.${name}.${k}`)) (out.engine[name] ??= {})[k] = v
        }
    }
    for (const [k, v] of Object.entries(data.sandbox ?? {})) {
        if (known.has(`sandbox.${k}`)) out.sandbox[k] = v
    }
    return out
}

/** @param {string} msg @param {number} [code] @returns {never} */
function fatal(msg, code = 2) {
    process.stderr.write(`lui: ${msg}\n`)
    process.exit(code)
}

/** @param {string} pathStr @returns {string[]} */
function resolveConfigPath(pathStr) {
    if (!pathStr || pathStr.includes("..") || pathStr.startsWith(".") || pathStr.endsWith(".")) {
        fatal(`invalid path ${JSON.stringify(pathStr)}`)
    }
    const parts = pathStr.split(".")
    if (TOP_LEVEL_KEYS.has(parts[0])) return parts
    return ["global", ...parts]
}

/** @param {string[]} path @returns {string} */
function displayPath(path) {
    return path[0] === "global" ? path.slice(1).join(".") : path.join(".")
}

/** @param {string} s @returns {any} */
function parseConfigValue(s) {
    if (s === "true") return true
    if (s === "false") return false
    if (/^-?\d+$/.test(s)) return parseInt(s, 10)
    if (/^-?\d+\.\d+$/.test(s)) return parseFloat(s)
    return s
}

/** @param {any} v @returns {string} */
function formatConfigValue(v) {
    if (typeof v === "string") return JSON.stringify(v)
    if (Array.isArray(v)) return JSON.stringify(v)
    return String(v)
}

/** @param {any} root @param {string[]} path @param {any} value */
function setNested(root, path, value) {
    let cur = root
    for (let i = 0; i < path.length - 1; i++) {
        const k = path[i]
        if (cur[k] == null || typeof cur[k] !== "object" || Array.isArray(cur[k])) cur[k] = {}
        cur = cur[k]
    }
    cur[path[path.length - 1]] = value
}

/** @param {any} root @param {string[]} path @returns {any} */
function getNested(root, path) {
    let cur = root
    for (const k of path) {
        if (cur == null) return undefined
        cur = cur[k]
    }
    return cur
}

/** @param {any} root @param {string[]} path @returns {boolean} */
function deleteNested(root, path) {
    let cur = root
    for (let i = 0; i < path.length - 1; i++) {
        cur = cur[path[i]]
        if (cur == null) return false
    }
    const last = path[path.length - 1]
    if (!(last in cur)) return false
    delete cur[last]
    return true
}

/** @param {Lui} lui @param {string[]} args */
export function runConfigSet(lui, args) {
    if (args.length !== 2) fatal("set PATH VALUE")
    const path = resolveConfigPath(args[0])
    if (!isSchemaAllowed(path)) {
        fatal(`unknown setting ${JSON.stringify(displayPath(path))} (run \`lui set\` to see available settings)`)
    }
    const value = parseConfigValue(args[1])
    if (isArrayPath(path)) {
        const current = getNested(lui.config, path)
        const arr = Array.isArray(current) ? current : []
        arr.push(value)
        setNested(lui.config, path, arr)
        lui.config.save()
        process.stdout.write(
            `Added ${formatConfigValue(value)} to ${displayPath(path)} (now ${arr.length} item${arr.length === 1 ? "" : "s"})\n`
        )
    } else {
        setNested(lui.config, path, value)
        lui.config.save()
        process.stdout.write(`Set ${displayPath(path)} = ${formatConfigValue(value)}\n`)
    }
}

/** @param {Lui} lui @param {string[]} args */
export function runConfigUnset(lui, args) {
    if (args.length !== 1) fatal("unset PATH")
    const path = resolveConfigPath(args[0])
    const removed = deleteNested(lui.config, path)
    lui.config.save()
    if (removed) process.stdout.write(`Unset ${displayPath(path)}\n`)
    else process.stdout.write(`${displayPath(path)} was already unset\n`)
}

/** @param {Lui} lui */
export function runConfigDump(lui) {
    writeAllSettings(lui, "  ")
}

// Sort by path, then by value — keeps multi-value arrays grouped and in
// stable alphabetical order.
/** @param {{ path: string, value: any }} a @param {{ path: string, value: any }} b */
function comparePairs(a, b) {
    if (a.path < b.path) return -1
    if (a.path > b.path) return 1
    const av = String(a.value)
    const bv = String(b.value)
    if (av < bv) return -1
    if (av > bv) return 1
    return 0
}

// Single sorted list of every known setting. Each row is marked `set`
// when the current value differs from the schema default and `def`
// otherwise (which is also where unseen schema entries land). Def
// rows render dim so explicitly-configured values stay scannable
// without splitting the table into two sections.
/** @param {Lui} lui @param {string} [indent] */
function writeAllSettings(lui, indent = "") {
    const tty = process.stdout.isTTY

    /** @type {Map<string, string>} */
    const schemaByPath = new Map()
    for (const { path, default: def } of allSchemaDefaults()) {
        schemaByPath.set(path, formatDefault(def))
    }

    /** @type {{ path: string, value: string }[]} */
    const present = []
    visit(present, "", lui.config.global)
    for (const table of TOP_LEVEL_TABLES) {
        // `global` was just walked above; `model` is user data,
        // rendered separately under "Models".
        if (table === "global" || table === "model") continue
        visit(present, table, /** @type {any} */ (lui.config)[table])
    }

    const presentPaths = new Set(present.map((p) => p.path))
    /** @type {{ path: string, value: string, isDef: boolean }[]} */
    const rows = present.map((p) => ({
        ...p,
        isDef: schemaByPath.has(p.path) && String(p.value) === schemaByPath.get(p.path)
    }))
    for (const [path, value] of schemaByPath) {
        if (presentPaths.has(path)) continue
        rows.push({ path, value, isDef: true })
    }
    rows.sort(comparePairs)

    const out = []
    const DIM = { dim: true }
    for (const { path, value, isDef } of rows) {
        const marker = isDef ? "def" : "set"
        if (tty) {
            const m = isDef ? styled(marker, DIM) : marker
            const k = styled(path, isDef ? DIM : STYLE.CONFIG_KEY)
            const v = styled(value, isDef ? DIM : STYLE.VALUE)
            out.push(`${indent}${m} ${k} ${v}\n`)
        } else {
            out.push(`${indent}${marker} ${path} ${value}\n`)
        }
    }
    process.stdout.write(out.join(""))
}

/** @param {{ path: string, value: string }[]} pairs @param {string} prefix @param {any} obj */
function visit(pairs, prefix, obj) {
    if (!obj || typeof obj !== "object") return
    for (const k of Object.keys(obj).sort()) {
        const v = obj[k]
        if (v == null) continue
        const path = prefix ? `${prefix}.${k}` : k
        if (Array.isArray(v)) {
            if (v.length === 0) pairs.push({ path, value: "[]" })
            else for (const item of v) pairs.push({ path, value: formatLeaf(item) })
        } else if (typeof v === "object") {
            visit(pairs, path, v)
        } else {
            pairs.push({ path, value: formatLeaf(v) })
        }
    }
}

/** @param {any} v @returns {string} */
function formatLeaf(v) {
    if (typeof v === "string") return v
    return String(v)
}

/** @param {any} v @returns {string} */
function formatDefault(v) {
    if (v == null) return "(unset)"
    if (Array.isArray(v)) return v.length === 0 ? "[]" : JSON.stringify(v)
    return String(v)
}
