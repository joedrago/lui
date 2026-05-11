import process from "node:process"

import { Lui } from "./lui.js"
import { harnesses } from "./harness/index.js"
import { engines } from "./engine.js"
import { styled } from "./ansi.js"

const HEADER_STYLE = { fg: [120, 100, 180] }
const PATH_STYLE = { fg: [230, 200, 140] }
const VAL_STYLE = { fg: [210, 150, 255] }

const SUBCOMMANDS = new Set(["run", "add", "cp", "set", "rm", "ssh", "remote", "websearch", "sandbox", "config"])

function printHelp() {
    process.stdout.write(`lui — a friendly TUI wrapper for LLM engines.

USAGE
  lui                              print this help

  lui run [NAME]                   run a model by name (resumes last if absent)

  lui add NAME ENGINE ARGS...      create a model (ARGS go to the engine)
  lui set NAME ARGS...             replace ARGS for a model
  lui cp OLDNAME NEWNAME           copy a model under a new name
  lui rm NAME                      delete a model

  lui config                       settings + models + resolved commandlines
  lui config set PATH VALUE        set a config value (appends for array paths)
  lui config clear PATH            remove a config value (or whole array)

  lui remote HOST[:PORT]           connect TUI to a remote lui
  lui ssh USER@HOST                configure a remote client
  lui websearch                    run only the websearch server

  lui sandbox HARNESS [ARGS...]    launch HARNESS under nono — every
                                   token after HARNESS is passed
                                   verbatim

`)
}

function fatal(msg, code = 2) {
    process.stderr.write(`lui: ${msg}\n`)
    process.exit(code)
}

function runConfigDump(lui) {
    const tty = process.stdout.isTTY
    const header = (label) => process.stdout.write((tty ? styled(label, HEADER_STYLE) : label) + "\n")

    header("Active Settings:")
    const setPaths = writeFlatConfig(lui, "  ")

    process.stdout.write("\n")
    header("Available Settings:")
    writeDefaultConfig(setPaths, "  ")

    process.stdout.write("\n")
    header("Models:")
    lui.printModels({ indent: "  " })

    process.stdout.write("\n")
    lui.printSandboxCommandline()
    process.stdout.write("\n")
}

function emitPair(out, indent, path, value, tty) {
    if (tty) out.push(`${indent}${styled(path, PATH_STYLE)} ${styled(value, VAL_STYLE)}\n`)
    else out.push(`${indent}${path} ${value}\n`)
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

function writeFlatConfig(lui, indent = "") {
    const pairs = []
    visit(pairs, "", lui.config.global)
    visit(pairs, "harness", lui.config.harness)
    visit(pairs, "engine", lui.config.engine)
    visit(pairs, "sandbox", lui.config.sandbox)

    pairs.sort(comparePairs)

    const setPaths = new Set()
    const out = []
    const tty = process.stdout.isTTY
    for (const { path, value } of pairs) {
        setPaths.add(path)
        emitPair(out, indent, path, value, tty)
    }
    process.stdout.write(out.join(""))
    return setPaths
}

function writeDefaultConfig(setPaths, indent = "") {
    const out = []
    const tty = process.stdout.isTTY
    const sorted = schemaDefaults()
        .filter(({ path }) => !setPaths.has(path))
        .sort(comparePairs)
    for (const { path, value } of sorted) emitPair(out, indent, path, value, tty)
    process.stdout.write(out.join(""))
}

function schemaDefaults() {
    const out = []
    out.push({ path: "engine_port", value: "8080" })
    out.push({ path: "web_port", value: "8081" })
    out.push({ path: "websearch", value: "true" })
    out.push({ path: "public", value: "false" })
    out.push({ path: "debug_log", value: "(unset)" })

    for (const h of harnesses) out.push({ path: `harness.${h.name}.enabled`, value: String(h.defaultEnabled) })

    for (const name of Object.keys(engines)) {
        out.push({ path: `engine.${name}.binary`, value: `(PATH: ${engines[name].defaultBinary})` })
    }

    out.push({ path: "sandbox.allow_cwd", value: "true" })
    out.push({ path: "sandbox.block_net", value: "false" })
    out.push({ path: "sandbox.allow_gpu", value: "false" })
    out.push({ path: "sandbox.rollback", value: "false" })
    out.push({ path: "sandbox.silent", value: "false" })
    out.push({ path: "sandbox.dev_tools", value: "true" })
    out.push({ path: "sandbox.profile", value: "(auto-detected from harness name)" })
    out.push({ path: "sandbox.bin", value: "nono" })
    out.push({ path: "sandbox.allow", value: "[]" })
    out.push({ path: "sandbox.read", value: "[]" })
    out.push({ path: "sandbox.write", value: "[]" })
    out.push({ path: "sandbox.allow_domain", value: "[]" })
    out.push({ path: "sandbox.extra", value: "[]" })

    return out
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

const TOP_LEVEL_KEYS = new Set(["global", "model", "harness", "engine", "sandbox"])

// `set` on these paths appends; `clear` removes the whole list.
const ARRAY_PATHS = new Set(["sandbox.allow", "sandbox.read", "sandbox.write", "sandbox.allow_domain", "sandbox.extra"])

function isArrayPath(path) {
    return ARRAY_PATHS.has(path.join("."))
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

function runConfigCommand(lui, args) {
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

async function main() {
    const argv = process.argv.slice(2)

    if (argv.length === 0) {
        printHelp()
        return
    }

    const first = argv[0]
    if (!SUBCOMMANDS.has(first)) {
        process.stderr.write(`lui: unknown subcommand "${first}". Run \`lui\` for usage.\n`)
        process.exit(2)
    }

    const verb = first
    const rest = argv.slice(1)
    const lui = new Lui()

    if (verb === "run") {
        if (rest.length > 1) fatal(`run takes at most one NAME, got: ${rest.join(" ")}`)
        await lui.run(rest[0])
        return
    }

    if (verb === "config") {
        if (rest.length === 0) {
            runConfigDump(lui)
            return
        }
        runConfigCommand(lui, rest)
        return
    }

    if (verb === "add") {
        if (rest.length < 2) fatal("add requires NAME and ENGINE")
        const [name, engineName, ...tail] = rest
        const args = tail[0] === "--" ? tail.slice(1) : tail
        lui.add(name, engineName, args)
        return
    }

    if (verb === "set") {
        if (rest.length < 1) fatal("set requires NAME")
        const [name, ...tail] = rest
        if (tail.length === 0) fatal("set requires ARGS after NAME")
        const args = tail[0] === "--" ? tail.slice(1) : tail
        lui.set(name, args)
        return
    }

    if (verb === "cp") {
        if (rest.length !== 2) fatal("cp requires OLDNAME NEWNAME")
        lui.cp(rest[0], rest[1])
        return
    }

    if (verb === "rm") {
        if (rest.length !== 1) fatal("rm requires NAME")
        lui.rm(rest[0])
        return
    }

    if (verb === "ssh") {
        if (rest.length !== 1) fatal("ssh requires USER@HOST")
        await lui.ssh(rest[0])
        return
    }

    if (verb === "remote") {
        if (rest.length !== 1) fatal("remote requires HOST[:PORT]")
        await lui.remote(rest[0])
        return
    }

    if (verb === "sandbox") {
        if (rest.length < 1) fatal("sandbox requires HARNESS [args...]")
        const harnessName = rest[0]
        const harnessArgs = rest.slice(1)
        await lui.sandbox(harnessName, harnessArgs)
        return
    }

    if (verb === "websearch") {
        if (rest.length > 0) fatal("websearch takes no arguments")
        await lui.websearch()
        return
    }
}

main().catch((e) => {
    process.stderr.write(`lui: ${e?.stack || e}\n`)
    process.exit(1)
})
