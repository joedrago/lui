import process from "node:process"

import { Lui } from "./lui.js"
import { runConfigSet, runConfigUnset, runConfigDump } from "./config.js"
import { engines } from "./engine.js"
import { runSetup } from "./setup.js"

const SUBCOMMANDS = new Set(["run", "add", "cp", "args", "rm", "ssh", "websearch", "sandbox", "set", "unset", "setup"])

/** @returns {void} */
function printHelp() {
    process.stdout.write(`lui — a friendly TUI wrapper for LLM engines.

USAGE
  lui                              print this help

  lui run                          show all models
  lui run NAME                     run a model by name

  lui add NAME ENGINE ARGS...      create a model (${Object.keys(engines).join(", ")})
  lui cp OLDNAME NEWNAME           copy a model under a new name
  lui rm NAME                      delete a model

  lui args NAME                    show ARGS for a model
  lui args NAME ARGS...            replace ARGS for a model

  lui set                          show config settings
  lui set PATH VALUE               set a config value (appends for array paths)
  lui unset                        show config settings
  lui unset PATH                   unset a config value (or whole array)

  lui ssh USER@HOST                configure a remote client
  lui websearch                    run only the websearch server

  lui sandbox                      show sandbox commandline preview
  lui sandbox HARNESS [ARGS...]    launch HARNESS under nono — every
                                   token after HARNESS is passed
                                   verbatim

Tip: run \`lui setup\` for an interactive first-run wizard.

`)
}

/** @param {string} msg @param {number} [code] @returns {never} */
function fatal(msg, code = 2) {
    process.stderr.write(`lui: ${msg}\n`)
    process.exit(code)
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
        if (rest.length === 0) {
            lui.printModels({ indent: "  " })
            return
        }
        await lui.run(rest[0])
        return
    }

    if (verb === "set") {
        if (rest.length === 0) runConfigDump(lui)
        else runConfigSet(lui, rest)
        return
    }

    if (verb === "unset") {
        if (rest.length === 0) runConfigDump(lui)
        else runConfigUnset(lui, rest)
        return
    }

    if (verb === "add") {
        if (rest.length < 2) fatal("add requires NAME and ENGINE")
        const [name, engineName, ...args] = rest
        lui.add(name, engineName, args)
        return
    }

    if (verb === "args") {
        if (rest.length < 1) fatal("args requires NAME")
        const [name, ...args] = rest
        lui.setArgs(name, args)
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

    if (verb === "sandbox") {
        if (rest.length === 0) {
            lui.printSandboxCommandline()
            return
        }
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

    if (verb === "setup") {
        if (rest.length > 0) fatal("setup takes no arguments")
        await runSetup(lui)
        return
    }
}

main().catch((e) => {
    process.stderr.write(`lui: ${e?.stack || e}\n`)
    process.exit(1)
})
