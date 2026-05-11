// The Lui class: orchestrator and shared state bag. main.js parses argv,
// instantiates one Lui, and dispatches the chosen subcommand. Engines,
// web.js, display.js, and the harnesses all read from this object.
//
// Engines write their bag of running state under `lui.state.*`; lui's
// own fields (config, warnings, web, activeModel, …) live on `lui`
// directly so the two namespaces never collide.

import process from "node:process"

import { Config } from "./config.js"
import { View } from "./wire.js"
import { stripStyle, compilePalette, paint, wrapStyled, vwidth, styled } from "./ansi.js"
import { engines, runEngine } from "./engine.js"
import { startWebServer } from "./web.js"
import { startTui } from "./display.js"
import { sshSetupShare, sshSetupUse } from "./ssh.js"
import { runSandbox, previewSandboxArgs } from "./sandbox.js"
import { harnesses, applyAllLocal } from "./harness/index.js"

export class Lui {
    static WARNING_TTL_MS = 60_000

    constructor() {
        this.config = Config.load()
        this.startedAt = Date.now()
        this.warnings = []
        this.requestCount = 0
        this.websearchCount = 0
        this.activeSearchCount = 0

        this.engineModule = null
        this.engineChild = null
        this.activeModel = null
        this.state = {}

        this.web = null
        this.tui = null

        this.shuttingDown = false
        this.exitCode = 0
    }

    // ---------- subcommand methods ----------

    async run(name) {
        const model = this.resolveModel(name)
        if (!model) {
            const want = name ? `model "${name}"` : "any model"
            process.stderr.write(`lui: ${want} not found. Try \`lui add ${name || "NAME"} llama-server -- ARGS\`.\n`)
            process.exit(1)
        }
        this.config.setActiveModel(model.name)
        this.config.save()

        // Harnesses read lui.activeModel for the chosen model name and its
        // args (e.g. `-c 262144` for opencode's context-window field), so
        // populate it before applyAllLocal runs.
        this.activeModel = model
        applyAllLocal(this)

        await this.spawnEngine(model)

        this.web = await startWebServer(this)
        this.tui = startTui(this)

        await new Promise((resolve) => {
            this.onShutdownResolve = resolve
            process.on("SIGINT", () => this.shutdown(0))
            process.on("SIGTERM", () => this.shutdown(0))
        })
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

    clone(newName, oldName) {
        if (newName === oldName) {
            process.stderr.write(`lui: clone needs a different NEWNAME than OLDNAME.\n`)
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
        process.stdout.write(`Cloned "${oldName}" → "${newName}" (engine ${src.engine}).\n`)
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
        const ACTIVE_DOT = { fg: [230, 200, 140], bold: true }
        const NAME = { bold: true }
        const ENGINE_NAME = { fg: "cyan" }
        const DIM_TEXT = { dim: true }

        for (let i = 0; i < names.length; i++) {
            const name = names[i]
            const m = this.config.model[name]
            const isActive = name === active

            // Header: ● name  engine
            const ln = p.line()
            if (isActive) ln.style(ACTIVE_DOT).text("● ").style()
            else ln.style(DIM_TEXT).text("○ ").style()
            ln.style(NAME).text(name).style().text("  ").style(ENGINE_NAME).text(m.engine).style()

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
                    cmdLine.text(" ").style(seg.style ?? {}).text(seg.args.join(" ")).style()
                }
            }

            if (i < names.length - 1) p.line()
        }

        const built = v.build()
        emitPaintedLines(built, indent)
    }

    // Just the sandbox preview line, segmented like the engine
    // commandline and with magenta-bold overlays on every HARNESS
    // placeholder slot.
    printSandboxCommandline() {
        const v = View()
        const p = v.panel("")
        const HEADER = { fg: [120, 100, 180] }
        const HARNESS = { fg: "magenta", bold: true }

        p.line().style(HEADER).text("Sandbox Commandline:")
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
                if (tok === "HARNESS") sandboxLine.style(HARNESS).text(tok).style()
                else sandboxLine.style(seg.style ?? {}).text(tok).style()
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
        await new Promise((resolve) => {
            this.onShutdownResolve = resolve
            process.on("SIGINT", () => {
                this.quitReason ??= "received SIGINT"
                this.shutdown(0)
            })
            process.on("SIGTERM", () => {
                this.quitReason ??= "received SIGTERM"
                this.shutdown(0)
            })
        })
    }

    // ---------- engine lifecycle ----------

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
        this.engineModule.initState?.(this)
        this.engineChild = runEngine(this, binary, segments)
    }

    onEngineExit(code, signal) {
        if (this.shuttingDown) return
        const detail = this.state?.exitMessage || (signal ? `killed by ${signal}` : `exited with code ${code}`)
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

    // Print a short banner on the main screen after the TUI has handed
    // the terminal back. Always includes the reason — even for clean
    // quits, so the user sees what just happened ("user pressed q" vs.
    // "engine exited with code 1").
    printShutdownSummary() {
        const LAVENDER_BOLD = { fg: [180, 150, 255], bold: true }
        const LABEL = { fg: [120, 100, 180] }
        const FATAL_LABEL = { fg: [230, 100, 100], bold: true }
        const FATAL_BODY = { fg: [230, 100, 100] }

        const uptimeMs = Date.now() - this.startedAt
        const uptime = formatDuration(uptimeMs)
        const model = this.activeModel?.name
        const reason = this.quitReason || (this.exitCode === 0 ? "shutdown" : `exit code ${this.exitCode}`)
        const fatal = this.state?.fatalReason

        const out = []
        out.push("\n")
        out.push(`${styled("lui", LAVENDER_BOLD)} shutting down\n`)
        if (model) out.push(`  ${styled("Model   :", LABEL)} ${model}\n`)
        out.push(`  ${styled("Uptime  :", LABEL)} ${uptime}\n`)
        if (this.state?.requestCount != null) {
            out.push(`  ${styled("Requests:", LABEL)} ${this.state.requestCount}\n`)
        }
        out.push(`  ${styled("Reason  :", LABEL)} ${reason}\n`)
        if (fatal) {
            out.push(`\n  ${styled("lui aborted:", FATAL_LABEL)} ${styled(fatal, FATAL_BODY)}\n`)
        }
        out.push("\n")
        process.stdout.write(out.join(""))
    }

    // ---------- View composition ----------

    appendLuiPanel(v) {
        // The engine's own appendPanels already emits the top "lui — llm ui"
        // panel since it carries the model + memory + version state. When no
        // engine is loaded (websearch-only mode), emit a minimal status panel
        // here so the screen isn't empty.
        if (this.engineModule) return
        const p = v.panel("lui — llm ui")
        const webPort = this.config.global.web_port
        if (webPort) p.line({ align: "right" }).style({ fg: "cyan" }).text(`http://127.0.0.1:${webPort}/setup`)
        p.line()
            .style({ fg: [180, 130, 220] })
            .text("Mode     : ")
            .style()
            .text("websearch only")
        p.line()
            .style({ fg: [180, 130, 220] })
            .text("Listening: ")
            .style()
            .text(`127.0.0.1:${webPort}`)
    }

    appendWarningsPanel(v) {
        const now = Date.now()
        const live = this.warnings.filter((w) => now - w.addedAt < Lui.WARNING_TTL_MS)
        this.warnings = live
        if (!live.length) return
        const p = v.panel("Warnings")
        for (const w of live) p.line().style({ fg: "yellow" }).text(w.text)
    }

    addWarning(text) {
        this.warnings.push({ text, addedAt: Date.now() })
    }

    bumpRequest() {
        this.requestCount += 1
    }
    bumpWebsearch() {
        this.websearchCount += 1
    }

    // Returns the list of harnesses currently enabled for this lui's config.
    enabledHarnesses() {
        const cfg = this.config.harness || {}
        return harnesses.filter((h) => cfg[h.name]?.enabled ?? h.defaultEnabled)
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

// Render a built View to stdout, honoring per-Line `indent` (panel
// content area) plus an optional outer indent (used to nest a model
// listing or commandline block under a section header). On a TTY,
// long lines wrap at the available width with a hanging indent
// matching the line's own start column — so a 300-char engine
// commandline reads as a tidy paragraph under its header.
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

