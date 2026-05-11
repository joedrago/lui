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

const SUBCOMMANDS = new Set(["run", "add", "set", "rm", "ls", "cmd", "show", "ssh", "remote", "websearch"])

// Flags that take a separate value token (e.g. `--debug log.txt`).
const VALUE_FLAGS = new Set(["--debug", "--engine-port", "--web-port"])

// Flags that stand alone as a boolean. Each shipped harness contributes
// a `--harness-NAME` / `--no-harness-NAME` pair so a user can flip the
// toggle from the command line; the override is written back to TOML so
// it sticks across runs.
const BOOL_FLAGS = new Set(["--public"])
const HARNESS_FLAGS = new Set()
for (const h of harnesses) {
    HARNESS_FLAGS.add(`--harness-${h.name}`)
    HARNESS_FLAGS.add(`--no-harness-${h.name}`)
}
for (const f of HARNESS_FLAGS) BOOL_FLAGS.add(f)

const RUN_FLAGS = new Set([...VALUE_FLAGS, ...BOOL_FLAGS])
const WEBSEARCH_FLAGS = new Set(["--web-port", "--public", ...HARNESS_FLAGS])
const SSH_FLAGS = new Set([...HARNESS_FLAGS])

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

RUN-TIME FLAGS (lui run, or bare lui)
  --debug PATH                     tee raw engine stdout to PATH
  --engine-port N                  override [global].engine_port
  --web-port N                     override [global].web_port
  --public                         bind 0.0.0.0 instead of 127.0.0.1

HARNESS FLAGS (lui run / lui ssh / lui websearch)
${harnesses.map((h) => `  --harness-${h.name.padEnd(20)} enable the ${h.name} harness (persists to TOML)\n  --no-harness-${h.name.padEnd(17)} disable it`).join("\n")}
`)
}

function fatal(msg, code = 2) {
    process.stderr.write(`lui: ${msg}\n`)
    process.exit(code)
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
    applyHarnessFlags(lui, flags)
}

// `--harness-NAME` / `--no-harness-NAME` set the enabled bit on the
// matching `[global.harness.NAME]` sub-table. Writes go through
// lui.config which is saved by every subcommand that touches it, so the
// override persists to TOML.
function applyHarnessFlags(lui, flags) {
    lui.config.global.harness ??= {}
    for (const h of harnesses) {
        const sub = (lui.config.global.harness[h.name] ??= {})
        if (flags[`harness-${h.name}`]) sub.enabled = true
        if (flags[`no-harness-${h.name}`]) sub.enabled = false
    }
}

async function main() {
    const argv = process.argv.slice(2)

    if (argv.includes("-h") || argv.includes("--help")) {
        printHelp()
        return
    }

    const first = argv[0]
    const isVerb = first && SUBCOMMANDS.has(first)
    const verb = isVerb ? first : "run"
    const rest = isVerb ? argv.slice(1) : argv.slice()

    const lui = new Lui()

    if (verb === "run") {
        const { pre } = splitAtDoubleDash(rest)
        const { positionals, flags } = parseFlags(pre, RUN_FLAGS)
        if (positionals.length > 1) fatal(`run takes at most one NAME, got: ${positionals.join(" ")}`)
        applyRunFlags(lui, flags)
        // Persist harness/port/public toggles before lui.run starts so a
        // subsequent failure (e.g. no active model) still saves the flag
        // the user asked for.
        lui.config.save()
        await lui.run(positionals[0])
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
        const { positionals, flags } = parseFlags(pre, SSH_FLAGS)
        if (positionals.length !== 1) fatal(`ssh requires USER@HOST`)
        applyHarnessFlags(lui, flags)
        lui.config.save()
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

    if (verb === "websearch") {
        const { pre } = splitAtDoubleDash(rest)
        const { flags } = parseFlags(pre, WEBSEARCH_FLAGS)
        applyRunFlags(lui, flags)
        lui.config.save()
        await lui.websearch()
        return
    }
}

main().catch((e) => {
    process.stderr.write(`lui: ${e?.stack || e}\n`)
    process.exit(1)
})
