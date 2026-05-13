/** @import { Engine, Model, ViewBuilder } from "./types.js" */

import process from "node:process"

import { Config } from "./config.js"
import { View } from "./wire.js"
import { stripStyle, compilePalette, paint, wrapStyled, vwidth, styled } from "./ansi.js"
import { STYLE } from "./theme.js"
import { engines } from "./engine.js"
import { startWebServer } from "./web.js"
import { startTui } from "./display.js"
import { sshSetupShare } from "./ssh.js"
import { runSandbox, previewSandboxArgs } from "./sandbox.js"
import { applyAllLocal } from "./harness.js"
import { formatDurationMilliseconds } from "./util.js"

export class Lui {
    static WARNING_TTL_MS = 60_000
    static SETUP_URL_BRIGHT_MS = 5000

    constructor() {
        this.config = Config.load()
        this.startedAt = Date.now()
        /** @type {{ text: string, addedAt: number }[]} */
        this.warnings = []

        /** @type {Engine | null} */
        this.engineModule = null
        /** @type {Model | null} */
        this.activeModel = null
        /** @type {Record<string, any>} */
        this.state = {}

        /** @type {{ server: any, port: number, bookmarkletUrl: string, close: () => Promise<void> } | null} */
        this.web = null
        /** @type {{ stop: () => void, repaint?: () => void } | null} */
        this.tui = null

        this.shuttingDown = false
        this.exitCode = 0
        /** @type {boolean} */
        this.engineReadyFired = false
        /** @type {(() => void) | null} */
        this.onEngineReady = null
        /** @type {(() => void) | null} */
        this.onShutdownResolve = null
        /** @type {string | null} */
        this.quitReason = null
    }

    /** @param {string} name */
    async run(name) {
        if (!name) {
            const all = Object.keys(this.config.model).sort()
            const hint = all.length ? `Available: ${all.join(", ")}.` : "Try `lui add NAME ENGINE ARGS...` first."
            process.stderr.write(`lui: \`lui run\` requires a model name. ${hint}\n`)
            process.exit(2)
        }
        const model = this.resolveModel(name)
        if (!model) {
            process.stderr.write(`lui: model "${name}" not found. Try \`lui add ${name} llama-server -- ARGS\`.\n`)
            process.exit(1)
        }

        this.activeModel = model

        // Harness configs only get written once the engine reports
        // Ready — otherwise an engine that has to download or otherwise
        // take its time before knowing the real context size would
        // hand the harness a wrong default. The engine signals via
        // `lui.markEngineReady()` from inside its parseLine.
        this.onEngineReady = () => {
            const ctxSize = this.engineModule?.contextSize?.(this.state, this.activeModel) ?? null
            applyAllLocal(this, { ctxSize }).catch((e) => {
                process.stderr.write(`lui: harness apply: ${e?.stack || e}\n`)
            })
        }

        await this.spawnEngine(model)

        this.web = await startWebServer(this)
        this.tui = startTui(this)

        await this.awaitShutdown()
    }

    markEngineReady() {
        if (this.engineReadyFired) return
        this.engineReadyFired = true
        try {
            this.onEngineReady?.()
        } catch (e) {
            process.stderr.write(`lui: onEngineReady threw: ${/** @type {any} */ (e)?.stack || e}\n`)
        }
    }

    /** @param {string} name @param {string} engineName @param {string[]} args */
    add(name, engineName, args) {
        if (!engines[engineName]) {
            process.stderr.write(`lui: unknown engine "${engineName}". Available: ${Object.keys(engines).join(", ")}\n`)
            process.exit(2)
        }
        if (this.config.model[name]) {
            process.stderr.write(`lui: model "${name}" already exists. Use \`lui args ${name} -- ...\` to replace its args.\n`)
            process.exit(2)
        }
        const model = { name, engine: engineName, args: [...args] }
        const probe = engines[engineName].describe(model, this)
        exitOnErrors(probe.errors)
        this.config.model[name] = { engine: engineName, args: [...args] }
        this.config.save()
        process.stdout.write(`Added model "${name}" (${engineName}).\n`)
    }

    /** @param {string} name @param {string[]} args */
    setArgs(name, args) {
        const existing = this.config.model[name]
        if (!existing) {
            process.stderr.write(`lui: model "${name}" not found. Use \`lui add ${name} ENGINE ARGS...\` to create it.\n`)
            process.exit(1)
        }

        // Zero-arg `lui args NAME` is a print-only shortcut: emit the
        // model's current argv in shell-copypaste form and exit
        // without touching the config. Lets you round-trip via
        // `lui args NAME $(lui args NAME)` or paste between hosts.
        if (args.length === 0) {
            process.stdout.write((existing.args || []).map(shellQuote).join(" ") + "\n")
            return
        }

        const engineName = existing.engine
        if (!engines[engineName]) {
            process.stderr.write(`lui: model "${name}" has unknown engine "${engineName}".\n`)
            process.exit(1)
        }
        const probe = engines[engineName].describe({ name, engine: engineName, args: [...args] }, this)
        exitOnErrors(probe.errors)
        existing.args = [...args]
        this.config.save()
        process.stdout.write(`Updated "${name}" args.\n`)
    }

    /** @param {string} name */
    rm(name) {
        if (!this.config.model[name]) {
            process.stderr.write(`lui: model "${name}" not found.\n`)
            process.exit(1)
        }
        delete this.config.model[name]
        this.config.save()
        process.stdout.write(`Removed "${name}".\n`)
    }

    /** @param {string} oldName @param {string} newName */
    cp(oldName, newName) {
        if (newName === oldName) {
            process.stderr.write(`lui: cp needs a different NEWNAME than OLDNAME.\n`)
            process.exit(1)
        }
        const src = this.config.model[oldName]
        if (!src) {
            process.stderr.write(`lui: model "${oldName}" not found.\n`)
            process.exit(1)
        }
        if (this.config.model[newName]) {
            process.stderr.write(`lui: model "${newName}" already exists. Use \`lui rm ${newName}\` first.\n`)
            process.exit(1)
        }
        this.config.model[newName] = { engine: src.engine, args: [...(src.args || [])] }
        this.config.save()
        process.stdout.write(`Copied "${oldName}" → "${newName}" (engine ${src.engine}).\n`)
    }

    /** @param {{ indent?: string }} [opts] */
    printModels({ indent = "" } = {}) {
        const names = Object.keys(this.config.model).sort()
        if (!names.length) {
            process.stdout.write(`${indent}(no models — try \`lui add NAME ENGINE ARGS...\`)\n`)
            return
        }

        const v = View()
        const p = v.panel("")

        for (let i = 0; i < names.length; i++) {
            const name = names[i]
            const m = this.config.model[name]

            const ln = p.line()
            ln.style({ bold: true }).text(name).style().text("  ").style(STYLE.ENGINE_NAME).text(m.engine).style()

            // Body: the model's full resolved commandline, with
            // per-segment colors. Segment 0 (for subprocess engines)
            // is the binary; non-subprocess engines like `remote` just
            // skip that one and emit user-args directly. Either way
            // this is the same payload `lui run NAME` would emit.
            const engineModule = engines[m.engine]
            if (engineModule) {
                const model = { name, engine: m.engine, args: m.args || [] }
                const { segments } = engineModule.describe(model, this)
                const cmdLine = p.line({ indent: 4 })
                let first = true
                for (const seg of segments) {
                    if (!seg.args.length) continue
                    if (!first) cmdLine.text(" ")
                    first = false
                    cmdLine
                        .style(seg.style ?? {})
                        .text(seg.args.join(" "))
                        .style()
                }
            }

            if (i < names.length - 1) p.line()
        }

        const built = v.build()
        emitPaintedLines(built, indent)
    }

    printSandboxCommandline() {
        const v = View()
        const p = v.panel("")

        p.line().style(STYLE.LABEL).text("Sandbox Commandline:")
        const sb = previewSandboxArgs(this)
        // Indent 6 so the body aligns with the Models section's wrapped
        // engine commandlines (which sit at outer indent 2 + line
        // indent 4). The View's `indent` property — not text-prefixed
        // spaces — so wrapStyled honors it on continuation rows.
        const sandboxLine = p.line({ indent: 6 }).text(sb.bin)
        for (const seg of sb.segments) {
            if (!seg.args.length) continue
            for (const tok of seg.args) {
                sandboxLine.text(" ")
                if (tok === "HARNESS") sandboxLine.style(STYLE.HARNESS_NAME).text(tok).style()
                else
                    sandboxLine
                        .style(seg.style ?? {})
                        .text(tok)
                        .style()
            }
        }

        const built = v.build()
        emitPaintedLines(built)
    }

    /** @param {string} target */
    async ssh(target) {
        await sshSetupShare(this, target)
    }
    /** @param {string} harnessName @param {string[]} harnessArgs */
    async sandbox(harnessName, harnessArgs) {
        await runSandbox(this, harnessName, harnessArgs)
    }

    async websearch() {
        this.web = await startWebServer(this)
        this.tui = startTui(this)
        await this.awaitShutdown({ recordReason: true })
    }

    // Install SIGINT/SIGTERM → shutdown(0) and return a promise that
    // resolves once `shutdown` runs. `recordReason: true` stamps
    // `this.quitReason` from the signal name so the shutdown summary can
    // show why we left (only modes without a richer reason source — like
    // a parsed engine exit — want this).
    /** @param {{ recordReason?: boolean }} [opts] @returns {Promise<void>} */
    awaitShutdown({ recordReason = false } = {}) {
        return new Promise((resolve) => {
            this.onShutdownResolve = resolve
            /** @param {string} sig */
            const onSignal = (sig) => {
                if (recordReason) this.quitReason ??= `received ${sig}`
                this.shutdown(0)
            }
            process.on("SIGINT", () => onSignal("SIGINT"))
            process.on("SIGTERM", () => onSignal("SIGTERM"))
        })
    }

    /** @param {string} name @returns {Model | null} */
    resolveModel(name) {
        const m = this.config.model[name]
        if (!m) return null
        return { name, engine: m.engine, args: m.args || [] }
    }

    /** @param {Model} model */
    async spawnEngine(model) {
        this.engineModule = engines[model.engine]
        if (!this.engineModule) {
            process.stderr.write(`lui: unknown engine "${model.engine}"\n`)
            process.exit(1)
        }
        this.state = {}
        const desc = this.engineModule.describe(model, this)
        exitOnErrors(desc.errors)
        for (const w of desc.warnings ?? []) this.addWarning(w)
        this.engineModule.initState?.(this)
        await this.engineModule.start(this, model, desc)
    }

    /** @param {number | null} code @param {string | null} signal */
    onEngineExit(code, signal) {
        if (this.shuttingDown) return
        const detail =
            this.engineModule?.exitReason?.(this.state, code, signal) ??
            (signal ? `killed by ${signal}` : `exited with code ${code}`)
        this.quitReason = `${this.engineModule?.name ?? "engine"} ${detail}`
        // Store exit details on state so shutdownSummary can include them
        this.state.exitCode = code
        this.state.exitSignal = signal
        this.shutdown(code || 1)
    }

    /** @param {number} [code] */
    async shutdown(code = 0) {
        if (this.shuttingDown) return
        this.shuttingDown = true
        this.exitCode = code
        this.tui?.stop?.()
        try {
            await this.engineModule?.stop?.(this)
        } catch (e) {
            process.stderr.write(`lui: engine.stop threw: ${/** @type {any} */ (e)?.stack || e}\n`)
        }
        await this.web?.close?.()
        try {
            this.config.save()
        } catch (e) {
            process.stderr.write(`lui: failed to save config: ${/** @type {Error} */ (e).message}\n`)
        }
        this.printShutdownSummary()
        this.onShutdownResolve?.()
        process.exit(this.exitCode)
    }

    printShutdownSummary() {
        const uptimeMs = Date.now() - this.startedAt
        const uptime = formatDurationMilliseconds(uptimeMs)
        const model = this.activeModel?.name
        const reason = this.quitReason || (this.exitCode === 0 ? "shutdown" : `exit code ${this.exitCode}`)
        const summary = this.engineModule?.shutdownSummary?.(this.state) ?? { lines: [], fatal: null }
        const labelW = labelWidth(["Model", "Uptime", "Reason", ...summary.lines.map((l) => l.label)])

        const out = []
        out.push("\n")
        out.push(`${styled("lui", STYLE.BRAND)} shutting down\n`)
        if (model) out.push(line("Model", model, labelW))
        out.push(line("Uptime", uptime, labelW))
        for (const l of summary.lines) out.push(line(l.label, l.value, labelW))
        out.push(line("Reason", reason, labelW))
        if (summary.fatal) {
            out.push(`\n  ${styled("lui aborted:", STYLE.FATAL_LABEL)} ${styled(summary.fatal, STYLE.FATAL)}\n`)
        }
        out.push("\n")
        process.stdout.write(out.join(""))
    }

    /** @param {ViewBuilder} v */
    appendLuiPanel(v) {
        const webPort = this.config.global.web_port
        const wantsUrl = this.config.global.websearch !== false && webPort
        const noEngine = !this.engineModule
        if (!wantsUrl && !noEngine) return

        const p = v.panel("lui — llm ui")
        if (wantsUrl) {
            // Bright for the first SETUP_URL_BRIGHT_MS so the URL grabs
            // attention on startup; fades to dim afterwards.
            const fresh = Date.now() - this.startedAt < Lui.SETUP_URL_BRIGHT_MS
            p.line({ align: "right" })
                .style(fresh ? STYLE.URL_FRESH : { dim: true })
                .text(`http://127.0.0.1:${webPort}/setup`)
        }
        if (noEngine) {
            p.line().style(STYLE.LABEL).text("Mode     : ").style().text("websearch only")
            p.line().style(STYLE.LABEL).text("Listening: ").style().text(`127.0.0.1:${webPort}`)
        }
    }

    /** @param {ViewBuilder} v */
    appendWarningsPanel(v) {
        const now = Date.now()
        const live = this.warnings.filter((w) => now - w.addedAt < Lui.WARNING_TTL_MS)
        this.warnings = live
        if (!live.length) return
        const p = v.panel("Warnings")
        for (const w of live) p.line().style(STYLE.WARNING).text(w.text)
    }

    /** @param {string} text */
    addWarning(text) {
        this.warnings.push({ text, addedAt: Date.now() })
    }
}

// Quote a single argv token for sh-compatible shells. Bare tokens
// composed entirely of safe characters pass through unquoted; anything
// else gets wrapped in single quotes (with embedded single quotes
// escaped via the standard '\'' dance). Matches what the user would
// type to reproduce the token at a shell prompt.
/** @param {string} arg @returns {string} */
function shellQuote(arg) {
    if (arg === "") return "''"
    if (/^[A-Za-z0-9_\-./:@+,=%]+$/.test(arg)) return arg
    return "'" + arg.replace(/'/g, "'\\''") + "'"
}

/** @param {string[] | undefined} errors */
function exitOnErrors(errors) {
    if (!errors?.length) return
    for (const e of errors) process.stderr.write(`lui: ${e}\n`)
    process.exit(1)
}

/** @param {string[]} labels @returns {number} */
function labelWidth(labels) {
    let w = 0
    for (const l of labels) if (l.length > w) w = l.length
    return w
}

/** @param {string} label @param {string} value @param {number} width @returns {string} */
function line(label, value, width) {
    return `  ${styled(`${label.padEnd(width)} :`, STYLE.LABEL)} ${value}\n`
}

// Per-Line `indent` plus an optional outer indent. Long lines wrap at
// the terminal width with continuation rows aligned at the same start
// column; non-TTY skips the wrap.
/** @param {import("./types.js").BuiltView} built @param {string} [outerIndent] */
function emitPaintedLines(built, outerIndent = "") {
    const panel = built.panels[0]
    if (!panel) return
    const lines = panel.lines ?? []
    const tty = process.stdout.isTTY
    const compiled = tty ? compilePalette(panel.palette || []) : null
    const cols = tty ? process.stdout.columns || 80 : Infinity
    const RIGHT_MARGIN = 2

    for (const l of lines) {
        const indent = outerIndent + (l.indent ? " ".repeat(l.indent) : "")
        const text = l.text || ""
        const available = cols - indent.length - RIGHT_MARGIN

        if (!tty || available <= 0 || vwidth(text) <= available) {
            const body = tty && compiled ? paint(text, compiled) : stripStyle(text)
            process.stdout.write(indent + body + "\n")
            continue
        }

        for (const row of wrapStyled(text, available)) {
            process.stdout.write(indent + paint(row, compiled ?? []) + "\n")
        }
    }
}
