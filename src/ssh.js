// `lui ssh USER@HOST` configures harnesses on a remote client for a
// reverse-tunneled connection to this machine. `lui remote HOST[:PORT]`
// is the inverse — this client fetches /config from a --public server.

import http from "node:http"
import process from "node:process"
import { spawn } from "node:child_process"

import { CONFIG_VERSION } from "./web.js"
import { harnesses, renderWebsearchSkill, applyAllLocal, harnessContext } from "./harness.js"
import { startWebServer } from "./web.js"
import { startTui } from "./display.js"

export async function sshSetupShare(lui, spec) {
    const target = parseShareTarget(spec)
    if (!target) {
        process.stderr.write(`lui: ssh expects USER@HOST, got "${spec}"\n`)
        process.exit(2)
    }

    const remoteEnginePort = pickRemotePort()
    const remoteWebPort = remoteEnginePort + 1

    const localEnginePort = lui.config.global.engine_port
    const localWebPort = lui.config.global.web_port
    const websearch = lui.config.global.websearch !== false

    for (const h of harnesses) {
        const enabled = (lui.config.harness?.[h.name]?.enabled ?? h.defaultEnabled) === true
        if (!enabled) continue
        if (h.preflight) {
            const ok = await sshPreflight(target, h)
            if (!ok.ok) {
                process.stderr.write(`lui: ${ok.error}\n`)
                process.exit(1)
            }
        }
        await applyHarnessRemote(lui, target, h, remoteEnginePort, remoteWebPort)
    }

    printShareSuccess(target, { localEnginePort, localWebPort, remoteEnginePort, remoteWebPort, websearch })
}

export async function sshSetupUse(lui, hostSpec) {
    const target = parseUseTarget(hostSpec)
    if (!target) {
        process.stderr.write(`lui: remote expects HOST or HOST:PORT, got "${hostSpec}"\n`)
        process.exit(2)
    }

    const cfg = await fetchRemoteConfig(target).catch((e) => {
        process.stderr.write(
            `lui: ${e.message}\n\nIs the server running with \`--public\`? Without it, the HTTP server binds to 127.0.0.1 and a client can't see it.\n`
        )
        process.exit(1)
    })

    if (cfg.version !== CONFIG_VERSION) {
        process.stderr.write(
            `lui: server /config version ${cfg.version}, this lui understands ${CONFIG_VERSION}. Upgrade the older side.\n`
        )
        process.exit(1)
    }

    const llamaBaseURL = `http://${target.host}:${cfg.engine_port}/v1`
    lui.activeModel = { name: cfg.active_model || "lui", engine: "remote", args: [] }
    const enabled = lui.config.global.websearch !== false
    applyAllLocal(lui, { baseURL: llamaBaseURL })

    process.stdout.write(`\n  Using server at ${target.host}:${target.httpPort}\n`)
    process.stdout.write(`    model:           ${cfg.active_model ?? "(unknown)"}\n`)
    process.stdout.write(`    llama (direct):  ${llamaBaseURL}\n`)
    if (enabled) {
        process.stdout.write(`    bsearch (local): http://127.0.0.1:${lui.config.global.web_port}/bsearch\n`)
        process.stdout.write(`    bookmarklet:     http://127.0.0.1:${lui.config.global.web_port}/setup\n`)
    }
    process.stdout.write(`\n  opencode config written. Run \`opencode\` in another terminal.\n\n`)

    if (enabled) {
        lui.web = await startWebServer(lui)
    }
    lui.tui = startTui(lui)

    await lui.awaitShutdown()
}

function parseShareTarget(s) {
    const i = s.indexOf("@")
    if (i <= 0 || i >= s.length - 1) return null
    return { user: s.slice(0, i), host: s.slice(i + 1) }
}

function parseUseTarget(s) {
    if (!s) return null
    const i = s.lastIndexOf(":")
    if (i < 0) return { host: s, httpPort: 8081 }
    const host = s.slice(0, i)
    const port = parseInt(s.slice(i + 1), 10)
    if (!host || !Number.isFinite(port)) return null
    return { host, httpPort: port }
}

function pickRemotePort() {
    const base = 18000 + Math.floor(Math.random() * 11000)
    return base
}

function sshTargetSpec(target) {
    return `${target.user}@${target.host}`
}

async function sshRun(target, command, stdinText) {
    return new Promise((resolve, reject) => {
        const child = spawn("ssh", [sshTargetSpec(target), command], {
            stdio: ["pipe", "pipe", "pipe"]
        })
        let stdout = ""
        let stderr = ""
        child.stdout.setEncoding("utf8")
        child.stderr.setEncoding("utf8")
        child.stdout.on("data", (c) => (stdout += c))
        child.stderr.on("data", (c) => (stderr += c))
        child.on("error", (e) => reject(new Error(`failed to spawn ssh: ${e.message}`)))
        child.on("exit", (code) => {
            if (code === 0) return resolve(stdout)
            const msg = (stderr || stdout || `ssh exited with code ${code}`).trim()
            reject(new Error(msg))
        })
        if (stdinText != null) {
            child.stdin.end(stdinText)
        } else {
            child.stdin.end()
        }
    })
}

async function sshPreflight(target, harness) {
    if (harness.name !== "opencode") return { ok: true }
    const probe =
        'command -v opencode || bash -lc \'command -v opencode\' || { [ -x "$HOME/.opencode/bin/opencode" ] && echo "$HOME/.opencode/bin/opencode"; }'
    try {
        const out = await sshRun(target, probe)
        if (out.trim()) return { ok: true }
        return { ok: false, error: `opencode not found on ${sshTargetSpec(target)}. Install it there first.` }
    } catch (e) {
        return { ok: false, error: `opencode preflight on ${sshTargetSpec(target)} failed: ${e.message}` }
    }
}

async function applyHarnessRemote(lui, target, harness, remoteEnginePort, remoteWebPort) {
    const dir = harness.configDir.replace(/^~\//, "")
    const basename = harness.configCandidates[0]
    const remotePath = `~/${dir}/${basename}`

    const existing = await sshRun(target, `cat ${remotePath} 2>/dev/null || true`).catch(() => "")

    if (existing && harness.needsBackup?.(existing)) {
        const cp = `cp -n ${remotePath} ${remotePath}.luibackup 2>/dev/null || true`
        try {
            await sshRun(target, cp)
        } catch {
            // ignore
        }
    }

    const ctx = harnessContext({
        activeModel: lui.activeModel ?? { name: "lui", engine: "llama-server", args: [] },
        enginePort: remoteEnginePort,
        webPort: remoteWebPort,
        websearch: lui.config.global.websearch
    })
    const next = harness.apply(existing || "", ctx)
    await sshRun(target, `mkdir -p ~/${dir} && cat > ${remotePath}`, next)

    const skillDir = `~/${dir}/skills/lui-web-search`
    if (lui.config.global.websearch !== false) {
        const body = renderWebsearchSkill(remoteWebPort)
        await sshRun(target, `mkdir -p ${skillDir} && cat > ${skillDir}/SKILL.md`, body)
    } else {
        try {
            await sshRun(target, `rm -f ${skillDir}/SKILL.md && rmdir ${skillDir} 2>/dev/null || true`)
        } catch {
            // ignore
        }
    }
}

function fetchRemoteConfig(target) {
    return new Promise((resolve, reject) => {
        const req = http.get({ host: target.host, port: target.httpPort, path: "/config", timeout: 5000 }, (res) => {
            let body = ""
            res.setEncoding("utf8")
            res.on("data", (c) => (body += c))
            res.on("end", () => {
                if (res.statusCode < 200 || res.statusCode >= 300) {
                    return reject(new Error(`${target.host}:${target.httpPort}/config returned HTTP ${res.statusCode}`))
                }
                try {
                    resolve(JSON.parse(body))
                } catch (e) {
                    reject(new Error(`unparseable /config JSON: ${e.message}`))
                }
            })
        })
        req.on("error", (e) => reject(new Error(`could not reach ${target.host}:${target.httpPort}: ${e.message}`)))
        req.on("timeout", () => req.destroy(new Error(`timeout connecting to ${target.host}:${target.httpPort}`)))
    })
}

function printShareSuccess(target, ports) {
    const cmd = ports.websearch
        ? `ssh -R ${ports.remoteEnginePort}:localhost:${ports.localEnginePort} -R ${ports.remoteWebPort}:localhost:${ports.localWebPort} ${sshTargetSpec(target)}`
        : `ssh -R ${ports.remoteEnginePort}:localhost:${ports.localEnginePort} ${sshTargetSpec(target)}`

    process.stdout.write(`\n  opencode configured on ${sshTargetSpec(target)}\n\n`)
    process.stdout.write(`  To connect from this machine, run in another terminal:\n\n`)
    process.stdout.write(`    ${cmd}\n\n`)
}
