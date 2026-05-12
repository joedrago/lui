// Config IO at ~/.config/lui.toml. Atomic tmp+rename on save.
// Also: `lui config` dump + `lui config set/clear` command handlers.

import fs from "node:fs"
import path from "node:path"
import os from "node:os"
import process from "node:process"
import { parse } from "smol-toml"

import { engineSchemaDefaults } from "./engine.js"
import { harnessSchemaDefaults } from "./harness.js"
import { sandboxSchemaDefaults } from "./sandbox.js"
import { styled } from "./ansi.js"
import { STYLE } from "./theme.js"

export const CONFIG_PATH = path.join(os.homedir(), ".config", "lui.toml")

// Top-level TOML tables lui knows about. `global` settings are the bare
// keys at the file root; the rest are nested. `model` is user-data
// (added/removed via `lui add/rm`), not a schema-controlled setting,
// but it's still a valid path prefix for `lui config set model.X.Y …`.
export const TOP_LEVEL_TABLES = ["global", "model", "harness", "engine", "sandbox"]

// Schema entries surfaced by `lui config` under "Available Settings".
// Each is {path, default, isArray?}. `isArray` marks list-typed paths
// where `set` appends and `clear` drops the whole list. The default
// values here are the single source of truth — Config.global is seeded
// from them, and the "Available Settings" dump renders them in place.
export function globalSchemaDefaults() {
    return [
        { path: "engine_port", default: 8080 },
        { path: "web_port", default: 8081 },
        { path: "websearch", default: true },
        { path: "public", default: false },
        { path: "debug_log", default: null }
    ]
}

function globalDefaults() {
    const out = {}
    for (const { path, default: def } of globalSchemaDefaults()) {
        if (def == null) continue
        if (path.includes(".")) continue
        out[path] = def
    }
    return out
}

export class Config {
    constructor(data = {}) {
        this.global = { ...globalDefaults(), ...(data.global ?? {}) }
        // Top-level catalogs / tuning tables. Each is its own root in
        // the TOML — [harness.opencode], [engine.llama-server],
        // [sandbox], [model.phi] — so they sit visually as peers.
        this.harness = data.harness ?? {}
        this.engine = data.engine ?? {}
        this.sandbox = data.sandbox ?? {}
        this.model = data.model ?? {}
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

    get activeModelName() {
        return this.global.active_model || null
    }
    setActiveModel(name) {
        this.global.active_model = name
    }
}

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
function emitTopLevelTable(cfg, rootKey) {
    const obj = cfg[rootKey]
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

// ─── `lui config` CLI ────────────────────────────────────────────────

function allSchemaDefaults() {
    return [...globalSchemaDefaults(), ...harnessSchemaDefaults(), ...engineSchemaDefaults(), ...sandboxSchemaDefaults()]
}

const TOP_LEVEL_KEYS = new Set(TOP_LEVEL_TABLES)

function isArrayPath(path) {
    const joined = path.join(".")
    return allSchemaDefaults().some((s) => s.isArray && s.path === joined)
}

function fatal(msg, code = 2) {
    process.stderr.write(`lui: ${msg}\n`)
    process.exit(code)
}

function resolveConfigPath(pathStr) {
    if (!pathStr || pathStr.includes("..") || pathStr.startsWith(".") || pathStr.endsWith(".")) {
        fatal(`config: invalid path ${JSON.stringify(pathStr)}`)
    }
    const parts = pathStr.split(".")
    if (TOP_LEVEL_KEYS.has(parts[0])) return parts
    return ["global", ...parts]
}

function displayPath(path) {
    return path[0] === "global" ? path.slice(1).join(".") : path.join(".")
}

function parseConfigValue(s) {
    if (s === "true") return true
    if (s === "false") return false
    if (/^-?\d+$/.test(s)) return parseInt(s, 10)
    if (/^-?\d+\.\d+$/.test(s)) return parseFloat(s)
    return s
}

function formatConfigValue(v) {
    if (typeof v === "string") return JSON.stringify(v)
    if (Array.isArray(v)) return JSON.stringify(v)
    return String(v)
}

function setNested(root, path, value) {
    let cur = root
    for (let i = 0; i < path.length - 1; i++) {
        const k = path[i]
        if (cur[k] == null || typeof cur[k] !== "object" || Array.isArray(cur[k])) cur[k] = {}
        cur = cur[k]
    }
    cur[path[path.length - 1]] = value
}

function getNested(root, path) {
    let cur = root
    for (const k of path) {
        if (cur == null) return undefined
        cur = cur[k]
    }
    return cur
}

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

export function runConfigCommand(lui, args) {
    const [op, ...rest] = args
    if (op === "set") {
        if (rest.length !== 2) fatal("config set PATH VALUE")
        const path = resolveConfigPath(rest[0])
        const value = parseConfigValue(rest[1])
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
        return
    }
    if (op === "clear") {
        if (rest.length !== 1) fatal("config clear PATH")
        const path = resolveConfigPath(rest[0])
        const removed = deleteNested(lui.config, path)
        lui.config.save()
        if (removed) process.stdout.write(`Cleared ${displayPath(path)}\n`)
        else process.stdout.write(`${displayPath(path)} was already unset\n`)
        return
    }
    fatal(`config: unknown operation ${JSON.stringify(op)} (try set, clear)`)
}

export function runConfigDump(lui) {
    const tty = process.stdout.isTTY
    const header = (label) => process.stdout.write((tty ? styled(label, STYLE.LABEL) : label) + "\n")

    header("Settings:")
    writeAllSettings(lui, "  ")

    process.stdout.write("\n")
    header("Models:")
    lui.printModels({ indent: "  " })

    process.stdout.write("\n")
    lui.printSandboxCommandline()
    process.stdout.write("\n")
}

// Sort by path, then by value — keeps multi-value arrays grouped and in
// stable alphabetical order.
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
function writeAllSettings(lui, indent = "") {
    const tty = process.stdout.isTTY

    const schemaByPath = new Map()
    for (const { path, default: def } of allSchemaDefaults()) {
        schemaByPath.set(path, formatDefault(def))
    }

    const present = []
    visit(present, "", lui.config.global)
    for (const table of TOP_LEVEL_TABLES) {
        // `global` was just walked above; `model` is user data,
        // rendered separately under "Models".
        if (table === "global" || table === "model") continue
        visit(present, table, lui.config[table])
    }

    const presentPaths = new Set(present.map((p) => p.path))
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

function formatLeaf(v) {
    if (typeof v === "string") return v
    return String(v)
}

function formatDefault(v) {
    if (v == null) return "(unset)"
    if (Array.isArray(v)) return v.length === 0 ? "[]" : JSON.stringify(v)
    return String(v)
}
