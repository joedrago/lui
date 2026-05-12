import process from "node:process"

import { Lui } from "./lui.js"
import { runConfigCommand, runConfigDump } from "./config.js"

const SUBCOMMANDS = new Set(["run", "add", "cp", "set", "rm", "ssh", "websearch", "sandbox", "config"])

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
