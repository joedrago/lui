import process from "node:process"

import { Config } from "./config.js"
import { View } from "./wire.js"
import { stripStyle, compilePalette, paint, wrapStyled, vwidth, styled } from "./ansi.js"
import { STYLE } from "./theme.js"
import { engines, runEngine } from "./engine.js"
import { startWebServer } from "./web.js"
import { startTui } from "./display.js"
import { sshSetupShare, sshSetupUse } from "./ssh.js"
import { runSandbox, previewSandboxArgs } from "./sandbox.js"
import { applyAllLocal } from "./harness.js"

export class Lui {
    static WARNING_TTL_MS = 60_000

    constructor() {
        this.config = Config.load()
        this.startedAt = Date.now()
        this.warnings = []
        this.websearchCount = 0
        this.activeSearchCount = 0

        this.engineModule = null
        this.engineChild = null
        this.activeModel = null
        this.state = {}

        // Resolved spawn argv, populated by spawnEngine. Render paths
        // read these instead of re-invoking buildArgv, which has side
        // effects (PATH lookup, thread autodetect).
        this.spawnBinary = null
        this.spawnSegments = null

        this.web = null
        this.tui = null

        this.shuttingDown = false
        this.exitCode = 0
    }

    async run(name) {
        const model = this.resolveModel(name)
        if (!model) {
            const want = name ? `model "${name}"` : "any model"
            process.stderr.write(`lui: ${want} not found. Try \`lui add ${name || "NAME"} llama-server -- ARGS\`.\n`)
            process.exit(1)
        }
        this.config.setActiveModel(model.name)
        this.config.save()

        this.activeModel = model

        // Harness configs only get written once the engine reports
        // Ready — otherwise an engine that has to download or otherwise
        // take its time before knowing the real context size would
        // hand the harness a wrong default. The engine signals via
        // `lui.markEngineReady()` from inside its parseLine.
        this.onEngineReady = () => {
            const ctxSize = this.engineModule.contextSize?.(this.state, this.activeModel) ?? null
            applyAllLocal(this, { ctxSize })
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
            process.stderr.write(`lui: onEngineReady threw: ${e?.stack || e}\n`)
        }
    }

    add(name, engineName, args) {
        if (!engines[engineName]) {
            process.stderr.write(`lui: unknown engine "${engineName}". Available: ${Object.keys(engines).join(", ")}\n`)
            process.exit(2)
        }
        if (this.config.model[name]) {
            process.stderr.write(`lui: model "${name}" already exists. Use \`lui set ${name} -- ...\` to replace its args.\n`)
            process.exit(2)
        }
        const model = { name, engine: engineName, args: [...args] }
        const probe = engines[engineName].buildArgv(model, this)
        if (probe.errors?.length) {
            for (const e of probe.errors) process.stderr.write(`lui: ${e}\n`)
            process.exit(1)
        }
        this.config.model[name] = { engine: engineName, args: [...args] }
        this.config.save()
        process.stdout.write(`Added model "${name}" (${engineName}).\n`)
    }

    set(name, args) {
        const existing = this.config.model[name]
        const creating = !existing
        const engineName = existing?.engine ?? "llama-server"
        if (!engines[engineName]) {
            process.stderr.write(`lui: model "${name}" has unknown engine "${engineName}".\n`)
            process.exit(1)
        }
        const probe = engines[engineName].buildArgv({ name, engine: engineName, args: [...args] }, this)
        if (probe.errors?.length) {
            for (const e of probe.errors) process.stderr.write(`lui: ${e}\n`)
            process.exit(1)
        }
        if (creating) {
            this.config.model[name] = { engine: engineName, args: [...args] }
            process.stdout.write(`(model "${name}" didn't exist — created with engine "${engineName}".)\n`)
        } else {
            existing.args = [...args]
            process.stdout.write(`Updated "${name}" args.\n`)
        }
        this.config.save()
    }

    rm(name) {
        if (!this.config.model[name]) {
            process.stderr.write(`lui: model "${name}" not found.\n`)
            process.exit(1)
        }
        delete this.config.model[name]
        if (this.config.global.active_model === name) delete this.config.global.active_model
        this.config.save()
        process.stdout.write(`Removed "${name}".\n`)
    }

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

    printModels({ indent = "" } = {}) {
        const names = Object.keys(this.config.model).sort()
        if (!names.length) {
            process.stdout.write(`${indent}(no models — try \`lui add NAME ENGINE ARGS...\`)\n`)
            return
        }
        const active = this.config.activeModelName

        const v = View()
        const p = v.panel("")

        for (let i = 0; i < names.length; i++) {
            const name = names[i]
            const m = this.config.model[name]
            const isActive = name === active

            // Header: ● name  engine
            const ln = p.line()
            if (isActive) ln.style(STYLE.ACTIVE).text("● ").style()
            else ln.style({ dim: true }).text("○ ").style()
            ln.style({ bold: true }).text(name).style().text("  ").style(STYLE.ENGINE_NAME).text(m.engine).style()

            // Body: the model's full resolved engine commandline, with
            // per-segment colors (binding/policy/defaults/user). This
            // is the same payload `lui run NAME` would spawn.
            const engineModule = engines[m.engine]
            if (engineModule) {
                const model = { name, engine: m.engine, args: m.args || [] }
                const { binary, segments } = engineModule.buildArgv(model, this)
                const cmdLine = p.line({ indent: 4 }).text(binary)
                for (const seg of segments) {
                    if (!seg.args.length) continue
                    cmdLine
                        .text(" ")
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

    async ssh(target) {
        await sshSetupShare(this, target)
    }
    async remote(host) {
        await sshSetupUse(this, host)
    }
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
    awaitShutdown({ recordReason = false } = {}) {
        return new Promise((resolve) => {
            this.onShutdownResolve = resolve
            const onSignal = (sig) => {
                if (recordReason) this.quitReason ??= `received ${sig}`
                this.shutdown(0)
            }
            process.on("SIGINT", () => onSignal("SIGINT"))
            process.on("SIGTERM", () => onSignal("SIGTERM"))
        })
    }

    resolveModel(name) {
        const wanted = name || this.config.activeModelName
        if (!wanted) {
            const all = Object.keys(this.config.model)
            if (all.length === 0) return null
            return null
        }
        const m = this.config.model[wanted]
        if (!m) return null
        return { name: wanted, engine: m.engine, args: m.args || [] }
    }

    async spawnEngine(model) {
        this.activeModel = model
        this.engineModule = engines[model.engine]
        if (!this.engineModule) {
            process.stderr.write(`lui: unknown engine "${model.engine}"\n`)
            process.exit(1)
        }
        this.state = {}
        const { binary, segments, errors, warnings } = this.engineModule.buildArgv(model, this)
        if (errors?.length) {
            for (const e of errors) process.stderr.write(`lui: ${e}\n`)
            process.exit(1)
        }
        for (const w of warnings ?? []) this.addWarning(w)
        this.spawnBinary = binary
        this.spawnSegments = segments
        this.engineModule.initState?.(this)
        this.engineChild = runEngine(this, binary, segments)
    }

    onEngineExit(code, signal) {
        if (this.shuttingDown) return
        const detail =
            this.engineModule?.exitReason?.(this.state, code, signal) ??
            (signal ? `killed by ${signal}` : `exited with code ${code}`)
        this.quitReason = `${this.engineModule?.name ?? "engine"} ${detail}`
        this.shutdown(code || 1)
    }

    async shutdown(code = 0) {
        if (this.shuttingDown) return
        this.shuttingDown = true
        this.exitCode = code
        this.tui?.stop?.()
        if (this.engineChild && this.engineChild.exitCode == null) {
            try {
                this.engineChild.kill("SIGTERM")
                await new Promise((res) => {
                    const t = setTimeout(() => {
                        try {
                            this.engineChild.kill("SIGKILL")
                        } catch {
                            // ignore
                        }
                        res()
                    }, 5000)
                    this.engineChild.once("exit", () => {
                        clearTimeout(t)
                        res()
                    })
                })
            } catch {
                // ignore
            }
        }
        await this.web?.close?.()
        try {
            this.config.save()
        } catch (e) {
            process.stderr.write(`lui: failed to save config: ${e.message}\n`)
        }
        this.printShutdownSummary()
        this.onShutdownResolve?.()
        process.exit(this.exitCode)
    }

    printShutdownSummary() {
        const uptimeMs = Date.now() - this.startedAt
        const uptime = formatDuration(uptimeMs)
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

    appendLuiPanel(v) {
        // The engine's own appendPanels already emits the top "lui — llm ui"
        // panel since it carries the model + memory + version state. When no
        // engine is loaded (websearch-only mode), emit a minimal status panel
        // here so the screen isn't empty.
        if (this.engineModule) return
        const p = v.panel("lui — llm ui")
        const webPort = this.config.global.web_port
        if (webPort) p.line({ align: "right" }).style(STYLE.URL).text(`http://127.0.0.1:${webPort}/setup`)
        p.line().style(STYLE.LABEL).text("Mode     : ").style().text("websearch only")
        p.line().style(STYLE.LABEL).text("Listening: ").style().text(`127.0.0.1:${webPort}`)
    }

    appendWarningsPanel(v) {
        const now = Date.now()
        const live = this.warnings.filter((w) => now - w.addedAt < Lui.WARNING_TTL_MS)
        this.warnings = live
        if (!live.length) return
        const p = v.panel("Warnings")
        for (const w of live) p.line().style(STYLE.WARNING).text(w.text)
    }

    addWarning(text) {
        this.warnings.push({ text, addedAt: Date.now() })
    }
}

function formatDuration(ms) {
    const s = Math.floor(ms / 1000)
    if (s < 60) return `${s}s`
    const m = Math.floor(s / 60)
    const rs = s % 60
    if (m < 60) return rs ? `${m}m${rs}s` : `${m}m`
    const h = Math.floor(m / 60)
    const rm = m % 60
    return rm ? `${h}h${rm}m` : `${h}h`
}

function labelWidth(labels) {
    let w = 0
    for (const l of labels) if (l.length > w) w = l.length
    return w
}

function line(label, value, width) {
    return `  ${styled(`${label.padEnd(width)} :`, STYLE.LABEL)} ${value}\n`
}

// Per-Line `indent` plus an optional outer indent. Long lines wrap at
// the terminal width with continuation rows aligned at the same start
// column; non-TTY skips the wrap.
function emitPaintedLines(built, outerIndent = "") {
    const lines = built.panels[0]?.lines ?? []
    const tty = process.stdout.isTTY
    const compiled = tty ? compilePalette(built.palette) : null
    const cols = tty ? process.stdout.columns || 80 : Infinity
    const RIGHT_MARGIN = 2

    for (const l of lines) {
        const indent = outerIndent + (l.indent ? " ".repeat(l.indent) : "")
        const text = l.text || ""
        const available = cols - indent.length - RIGHT_MARGIN

        if (!tty || available <= 0 || vwidth(text) <= available) {
            const body = tty ? paint(text, compiled) : stripStyle(text)
            process.stdout.write(indent + body + "\n")
            continue
        }

        for (const row of wrapStyled(text, available)) {
            process.stdout.write(indent + paint(row, compiled) + "\n")
        }
    }
}
