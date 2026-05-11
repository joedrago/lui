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
import { stripStyle } from "./ansi.js"
import { engines, runEngine } from "./engine.js"
import { startWebServer } from "./web.js"
import { startTui } from "./display.js"
import { sshSetupShare, sshSetupUse } from "./ssh.js"
import { runSandbox } from "./sandbox.js"
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

        this.debugLogPath = null
        this.publicBind = false

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

    ls() {
        const names = Object.keys(this.config.model).sort()
        if (!names.length) {
            process.stdout.write("(no models — try `lui add NAME ENGINE -- ARGS`)\n")
            return
        }
        const active = this.config.activeModelName
        for (const name of names) {
            const m = this.config.model[name]
            const star = name === active ? "*" : " "
            const args = (m.args || []).join(" ")
            process.stdout.write(`${star} ${name}  [${m.engine}]  ${args}\n`)
        }
        process.stdout.write("\n(use `lui show NAME` for full details)\n")
    }

    show(name) {
        const out = []
        const tty = process.stdout.isTTY
        const emit = (s) => out.push(tty ? s : stripStyle(s))

        emit("[global]\n")
        for (const [k, v] of Object.entries(this.config.global)) {
            if (k === "harness" || k === "engine") continue
            emit(`${k} = ${tomlScalar(v)}\n`)
        }
        const harness = this.config.harness || {}
        for (const h of Object.keys(harness).sort()) {
            emit(`\n[harness.${h}]\n`)
            for (const [k, v] of Object.entries(harness[h] || {})) emit(`${k} = ${tomlScalar(v)}\n`)
        }
        const eng = this.config.engine || {}
        for (const e of Object.keys(eng).sort()) {
            emit(`\n[engine.${e}]\n`)
            for (const [k, v] of Object.entries(eng[e] || {})) emit(`${k} = ${tomlScalar(v)}\n`)
        }
        const sb = this.config.sandbox || {}
        if (Object.keys(sb).length) {
            emit(`\n[sandbox]\n`)
            for (const [k, v] of Object.entries(sb)) emit(`${k} = ${tomlScalar(v)}\n`)
        }

        const names = name ? [name].filter((n) => this.config.model[n]) : Object.keys(this.config.model).sort()
        for (const n of names) {
            const m = this.config.model[n]
            emit(`\n[model.${tomlKey(n)}]\n`)
            emit(`engine = ${tomlScalar(m.engine)}\n`)
            const a = m.args || []
            if (a.length === 0) emit(`args = []\n`)
            else emit(`args = [\n${a.map((x) => `    ${tomlScalar(x)}`).join(",\n")}\n]\n`)
        }
        if (name && !names.length) {
            process.stderr.write(`lui: model "${name}" not found.\n`)
            process.exit(1)
        }
        process.stdout.write(out.join(""))
    }

    cmd(name) {
        const model = this.resolveModel(name)
        if (!model) {
            process.stderr.write(`lui: no model to print. Add one with \`lui add NAME ENGINE -- ARGS\`.\n`)
            process.exit(1)
        }
        const engineModule = engines[model.engine]
        if (!engineModule) {
            process.stderr.write(`lui: model "${model.name}" has unknown engine "${model.engine}".\n`)
            process.exit(1)
        }
        const { binary, segments, errors } = engineModule.buildArgv(model, this)
        if (errors?.length) {
            for (const e of errors) process.stderr.write(`lui: ${e}\n`)
            process.exit(1)
        }

        const v = View()
        const p = v.panel("")
        const ln = p.line()
        ln.text(binary)
        for (const seg of segments) {
            if (!seg.args.length) continue
            ln.text(" ")
                .style(seg.style ?? {})
                .text(seg.args.join(" "))
                .style()
        }
        const built = v.build()
        const text = built.panels[0]?.lines?.[0]?.text ?? ""
        const tty = process.stdout.isTTY
        if (tty) {
            // The TUI uses ansi.paint with a compiled palette; for a one-shot
            // line we can just lean on the same compilation step here.
            import("./ansi.js").then(({ compilePalette, paint }) => {
                const compiled = compilePalette(built.palette)
                process.stdout.write(paint(text, compiled) + "\n")
            })
        } else {
            process.stdout.write(stripStyle(text) + "\n")
        }
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
        const LAVENDER = "\x1b[38;2;180;150;255m"
        const MUTED = "\x1b[38;2;120;100;180m"
        const RED = "\x1b[38;2;230;100;100m"
        const RESET = "\x1b[0m"
        const BOLD = "\x1b[1m"

        const uptimeMs = Date.now() - this.startedAt
        const uptime = formatDuration(uptimeMs)
        const model = this.activeModel?.name
        const reason = this.quitReason || (this.exitCode === 0 ? "shutdown" : `exit code ${this.exitCode}`)
        const fatal = this.state?.fatalReason

        const out = []
        out.push("\n")
        out.push(`${BOLD}${LAVENDER}lui${RESET} shutting down\n`)
        if (model) out.push(`  ${MUTED}Model   :${RESET} ${model}\n`)
        out.push(`  ${MUTED}Uptime  :${RESET} ${uptime}\n`)
        if (this.state?.requestCount != null) {
            out.push(`  ${MUTED}Requests:${RESET} ${this.state.requestCount}\n`)
        }
        out.push(`  ${MUTED}Reason  :${RESET} ${reason}\n`)
        if (fatal) {
            out.push(`\n  ${BOLD}${RED}lui aborted:${RESET} ${RED}${fatal}${RESET}\n`)
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

function tomlScalar(v) {
    if (typeof v === "string") return `"${v.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`
    if (typeof v === "boolean") return v ? "true" : "false"
    if (typeof v === "number") return String(v)
    if (Array.isArray(v)) return `[${v.map(tomlScalar).join(", ")}]`
    return JSON.stringify(v)
}

function tomlKey(k) {
    if (/^[A-Za-z0-9_-]+$/.test(k)) return k
    return tomlScalar(k)
}
