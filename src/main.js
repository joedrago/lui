// argv parser + subcommand dispatcher. Hand-rolled — no commander, no
// minimist. The grammar is small:
//
//   lui                       → resume (implicit `run`)
//   lui --flag                → resume with run-time flags
//   lui run [NAME] [flags]    → run a model
//   lui add NAME ENGINE -- ARGS...
//   lui set NAME -- ARGS...
//   lui ls
//   lui rm NAME
//   lui cmd [NAME]
//   lui show [NAME]
//   lui ssh USER@HOST
//   lui remote HOST[:PORT]
//   lui websearch
//   lui -h | --help
//
// Subcommand dispatch rule: if the first positional is a known verb, that
// subcommand handles the rest. Otherwise fall through to `run`. Flags
// can appear before or after positionals.

import process from "node:process"

import { Lui } from "./lui.js"
import { harnesses } from "./harness/index.js"
import { engines } from "./engine.js"

const SUBCOMMANDS = new Set(["run", "add", "set", "rm", "ls", "cmd", "show", "ssh", "remote", "websearch", "sandbox", "config"])

// The only flags lui itself reads at runtime. Every other knob lives in
// the TOML and is tuned via `lui config set/clear/add`.
const VALUE_FLAGS = new Set(["--debug", "--engine-port", "--web-port"])
const BOOL_FLAGS = new Set(["--public"])

const RUN_FLAGS = new Set([...VALUE_FLAGS, ...BOOL_FLAGS])
const WEBSEARCH_FLAGS = new Set(["--web-port", "--public"])

function printHelp() {
    process.stdout.write(`lui — a friendly TUI wrapper for LLM engines.

USAGE
  lui                              resume the last model
  lui run [NAME]                   run a model by name
  lui add NAME ENGINE -- ARGS...   create a model
  lui set NAME -- ARGS...          replace ARGS for a model
  lui ls                           list models
  lui rm NAME                      delete a model
  lui cmd [NAME]                   print the spawn command
  lui show [NAME]                  dump resolved config
  lui ssh USER@HOST                configure a remote client
  lui remote HOST[:PORT]           connect TUI to a remote lui
  lui websearch                    run only the websearch server
  lui sandbox HARNESS [ARGS...]    launch HARNESS under nono — every
                                   token after HARNESS is passed
                                   verbatim, including --
  lui config set PATH VALUE        set a persistent config value
  lui config clear PATH            remove a persistent config value
  lui config add PATH VALUE        append VALUE to a config array

CONFIG EXAMPLES
${configExamples()}
  Paths are dot-separated. A path that doesn't start with one of
  (global, model, harness, engine, sandbox) is rooted under [global.…].
  Display always strips the implicit "global." prefix.

RUN-TIME FLAGS (lui run, or bare lui)
  --debug PATH                     tee raw engine stdout to PATH
  --engine-port N                  override engine_port for this run
  --web-port N                     override web_port for this run
  --public                         bind 0.0.0.0 instead of 127.0.0.1
`)
}

function fatal(msg, code = 2) {
    process.stderr.write(`lui: ${msg}\n`)
    process.exit(code)
}

// Build a comprehensive CONFIG EXAMPLES block from the live harness +
// engine registries and the spec'd sandbox keys. Adding a new harness
// or engine grows this block automatically.
function configExamples() {
    const lines = []
    const push = (s) => lines.push(`  ${s}`)
    const section = (label) => {
        if (lines.length) lines.push("")
        push(`# ${label}`)
    }

    section("Server ports + bind")
    push(`lui config set engine_port 8080`)
    push(`lui config set web_port 8081`)
    push(`lui config set public false`)

    section("Browser-mediated web search")
    push(`lui config set websearch true`)

    section("Harness toggles")
    for (const h of harnesses) push(`lui config set harness.${h.name}.enabled true`)

    section("Engine binary overrides (defaults to PATH lookup)")
    for (const name of Object.keys(engines)) {
        push(`lui config set engine.${name}.binary /path/to/${name}`)
        push(`lui config clear engine.${name}.binary`)
    }

    section("Sandbox tuning (used by `lui sandbox HARNESS`)")
    const sbBool = ["allow_cwd", "block_net", "allow_gpu", "rollback", "silent", "dev_tools"]
    for (const k of sbBool) push(`lui config set sandbox.${k} true`)
    push(`lui config set sandbox.profile opencode`)
    push(`lui config set sandbox.bin /usr/local/bin/nono`)
    const sbArr = ["allow", "read", "write", "allow_domain", "extra"]
    for (const k of sbArr) push(`lui config add sandbox.${k} ./value`)
    push(`lui config clear sandbox.allow`)

    return lines.join("\n") + "\n"
}

// Split argv at the first `--` separator. Everything before is parsed
// normally; everything after is captured verbatim as `passthrough`.
function splitAtDoubleDash(argv) {
    const i = argv.indexOf("--")
    if (i < 0) return { pre: argv.slice(), passthrough: null }
    return { pre: argv.slice(0, i), passthrough: argv.slice(i + 1) }
}

// Walk pre-argv, separating positionals from flags. Flags listed in
// `accepted` may take a value; others are bare booleans. Unknown flags
// are an error in this small grammar.
function parseFlags(pre, accepted) {
    const positionals = []
    const flags = {}
    for (let i = 0; i < pre.length; i++) {
        const tok = pre[i]
        if (tok === "-h" || tok === "--help") {
            flags.help = true
            continue
        }
        if (tok.startsWith("--")) {
            if (!accepted.has(tok)) fatal(`unknown flag ${tok}`)
            if (BOOL_FLAGS.has(tok)) {
                flags[tok.slice(2)] = true
                continue
            }
            const value = pre[i + 1]
            if (value == null || value.startsWith("-")) fatal(`${tok} expects a value`)
            flags[tok.slice(2)] = value
            i += 1
            continue
        }
        positionals.push(tok)
    }
    return { positionals, flags }
}

function applyRunFlags(lui, flags) {
    if (flags.debug) lui.debugLogPath = flags.debug
    if (flags["engine-port"]) lui.config.global.engine_port = parseInt(flags["engine-port"], 10) || lui.config.global.engine_port
    if (flags["web-port"]) lui.config.global.web_port = parseInt(flags["web-port"], 10) || lui.config.global.web_port
    if (flags.public) {
        lui.publicBind = true
        lui.config.global.public = true
    }
}

// `lui config` family. Three operations:
//   set   PATH VALUE   — write a scalar (or replace an array)
//   clear PATH         — remove the key
//   add   PATH VALUE   — append to an array (creates it if absent)
//
// PATH is dot-separated. If the first segment isn't a known top-level
// table, the path is rooted under "global." (so `engine_port` is the
// same as `global.engine_port` but `harness.opencode.enabled` lands at
// the top level).
const TOP_LEVEL_KEYS = new Set(["global", "model", "harness", "engine", "sandbox"])

function resolveConfigPath(pathStr) {
    if (!pathStr || pathStr.includes("..") || pathStr.startsWith(".") || pathStr.endsWith(".")) {
        fatal(`config: invalid path ${JSON.stringify(pathStr)}`)
    }
    const parts = pathStr.split(".")
    if (TOP_LEVEL_KEYS.has(parts[0])) return parts
    return ["global", ...parts]
}

// `global` is implicit when a path doesn't name another top-level table
// — strip it from messages so they read the way the user typed it.
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
    if (args.length === 0) fatal("config requires one of: set, clear, add")
    const [op, ...rest] = args
    if (op === "set") {
        if (rest.length !== 2) fatal("config set PATH VALUE")
        const path = resolveConfigPath(rest[0])
        const value = parseConfigValue(rest[1])
        setNested(lui.config, path, value)
        lui.config.save()
        process.stdout.write(`Set ${displayPath(path)} = ${formatConfigValue(value)}\n`)
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
    if (op === "add") {
        if (rest.length !== 2) fatal("config add PATH VALUE")
        const path = resolveConfigPath(rest[0])
        const value = parseConfigValue(rest[1])
        const current = getNested(lui.config, path)
        if (current != null && !Array.isArray(current)) {
            fatal(`config add: ${displayPath(path)} is not an array`)
        }
        const arr = Array.isArray(current) ? current : []
        arr.push(value)
        setNested(lui.config, path, arr)
        lui.config.save()
        process.stdout.write(`Added ${formatConfigValue(value)} to ${displayPath(path)} (now ${arr.length} item${arr.length === 1 ? "" : "s"})\n`)
        return
    }
    fatal(`config: unknown operation ${JSON.stringify(op)} (try set, clear, add)`)
}

async function main() {
    const argv = process.argv.slice(2)

    const first = argv[0]
    const isVerb = first && SUBCOMMANDS.has(first)
    const verb = isVerb ? first : "run"
    const rest = isVerb ? argv.slice(1) : argv.slice()

    // sandbox passes every remaining token through to the harness, so
    // `lui sandbox opencode --help` must NOT trigger lui's own help.
    if (verb !== "sandbox" && (argv.includes("-h") || argv.includes("--help"))) {
        printHelp()
        return
    }

    const lui = new Lui()

    if (verb === "run") {
        const { pre } = splitAtDoubleDash(rest)
        const { positionals, flags } = parseFlags(pre, RUN_FLAGS)
        if (positionals.length > 1) fatal(`run takes at most one NAME, got: ${positionals.join(" ")}`)
        applyRunFlags(lui, flags)
        await lui.run(positionals[0])
        return
    }

    if (verb === "config") {
        runConfigCommand(lui, rest)
        return
    }

    if (verb === "add") {
        const { pre, passthrough } = splitAtDoubleDash(rest)
        const { positionals } = parseFlags(pre, new Set())
        if (positionals.length !== 2) fatal(`add requires NAME and ENGINE`)
        const [name, engineName] = positionals
        lui.add(name, engineName, passthrough ?? [])
        return
    }

    if (verb === "set") {
        const { pre, passthrough } = splitAtDoubleDash(rest)
        const { positionals } = parseFlags(pre, new Set())
        if (positionals.length !== 1) fatal(`set requires NAME`)
        if (passthrough == null) fatal(`set requires \`-- ARGS...\``)
        lui.set(positionals[0], passthrough)
        return
    }

    if (verb === "rm") {
        const { pre } = splitAtDoubleDash(rest)
        const { positionals } = parseFlags(pre, new Set())
        if (positionals.length !== 1) fatal(`rm requires NAME`)
        lui.rm(positionals[0])
        return
    }

    if (verb === "ls") {
        lui.ls()
        return
    }

    if (verb === "cmd") {
        const { pre } = splitAtDoubleDash(rest)
        const { positionals } = parseFlags(pre, new Set())
        if (positionals.length > 1) fatal(`cmd takes at most one NAME`)
        lui.cmd(positionals[0])
        return
    }

    if (verb === "show") {
        const { pre } = splitAtDoubleDash(rest)
        const { positionals } = parseFlags(pre, new Set())
        if (positionals.length > 1) fatal(`show takes at most one NAME`)
        lui.show(positionals[0])
        return
    }

    if (verb === "ssh") {
        const { pre } = splitAtDoubleDash(rest)
        const { positionals } = parseFlags(pre, new Set())
        if (positionals.length !== 1) fatal(`ssh requires USER@HOST`)
        await lui.ssh(positionals[0])
        return
    }

    if (verb === "remote") {
        const { pre } = splitAtDoubleDash(rest)
        const { positionals } = parseFlags(pre, new Set())
        if (positionals.length !== 1) fatal(`remote requires HOST[:PORT]`)
        await lui.remote(positionals[0])
        return
    }

    if (verb === "sandbox") {
        // No flag parsing. Everything after `sandbox` is HARNESS plus
        // verbatim args for it — `alias opencode='lui sandbox opencode'`
        // relies on `--help` etc. reaching opencode, not lui.
        if (rest.length < 1) fatal("sandbox requires HARNESS [args...]")
        const harnessName = rest[0]
        const harnessArgs = rest.slice(1)
        await lui.sandbox(harnessName, harnessArgs)
        return
    }

    if (verb === "websearch") {
        const { pre } = splitAtDoubleDash(rest)
        const { flags } = parseFlags(pre, WEBSEARCH_FLAGS)
        applyRunFlags(lui, flags)
        await lui.websearch()
        return
    }
}

main().catch((e) => {
    process.stderr.write(`lui: ${e?.stack || e}\n`)
    process.exit(1)
})
