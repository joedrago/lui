// `lui ssh USER@HOST` configures harnesses on a remote client for a
// reverse-tunneled connection back to this machine. The inverse —
// running lui on a client that fetches its engine state from somewhere
// else — is just the `remote` engine: `lui add NAME remote HOST[:PORT]`.

import process from "node:process"
import { spawn } from "node:child_process"

import { harnesses, applyHarness, harnessContext, isHarnessEnabled } from "./harness.js"
import { engines } from "./engine.js"

export async function sshSetupShare(lui, spec) {
    const target = parseShareTarget(spec)
    if (!target) {
        process.stderr.write(`lui: ssh expects USER@HOST, got "${spec}"\n`)
        process.exit(2)
    }

    const enabled = harnesses.filter((h) => isHarnessEnabled(lui, h))
    if (enabled.length === 0) {
        process.stderr.write(luiNeedsHarnessError("ssh"))
        process.exit(1)
    }

    const remoteEnginePort = pickRemotePort()
    const remoteWebPort = remoteEnginePort + 1

    // Tunnel's server end: where on *this* machine to forward client
    // traffic. For a llama-server engine that's localhost:engine_port;
    // for a remote engine, the upstream's actual host:port — so an
    // `lui ssh` from a relay tunnels straight to llm without
    // proxying through this process.
    const engineEndpoint = lui.engineModule?.endpoint?.(lui) ?? {
        host: null,
        port: lui.config.global.engine_port
    }
    const localWebPort = lui.config.global.web_port
    const websearch = lui.config.global.websearch !== false

    for (const h of enabled) {
        if (h.sshPreflight) {
            const ok = await h.sshPreflight(target, sshRun)
            if (!ok.ok) {
                process.stderr.write(`lui: ${ok.error}\n`)
                process.exit(1)
            }
        }
        await applyHarnessRemote(lui, target, h, remoteEnginePort, remoteWebPort)
    }

    printShareSuccess(target, { engineEndpoint, localWebPort, remoteEnginePort, remoteWebPort, websearch })
}

function luiNeedsHarnessError(verb) {
    const all = harnesses.map((h) => h.name).join(", ")
    return (
        `lui ${verb}: no harnesses are enabled, which makes this subcommand a no-op.\n` +
        `Enable at least one before re-running, e.g. \`lui config set harness.${harnesses[0].name}.enabled true\`.\n` +
        `Available: ${all}.\n`
    )
}

function parseShareTarget(s) {
    const i = s.indexOf("@")
    if (i <= 0 || i >= s.length - 1) return null
    return { user: s.slice(0, i), host: s.slice(i + 1) }
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

// SSH transport: paths stay as "~/..." since the remote shell expands ~.
function sshTransport(target) {
    const q = (s) => `'${String(s).replace(/'/g, `'\\''`)}'`
    return {
        name: "ssh",
        resolve(p) {
            return p
        },
        async exists(p) {
            try {
                await sshRun(target, `test -e ${q(p)}`)
                return true
            } catch {
                return false
            }
        },
        async read(p) {
            return await sshRun(target, `cat ${q(p)} 2>/dev/null || true`).catch(() => "")
        },
        async write(p, body) {
            await sshRun(target, `cat > ${q(p)}`, body)
        },
        async remove(p) {
            try {
                await sshRun(target, `rm -f ${q(p)}`)
            } catch {
                // ignore
            }
        },
        async mkdirp(p) {
            await sshRun(target, `mkdir -p ${q(p)}`)
        },
        async tryRmDir(p) {
            try {
                await sshRun(target, `rmdir ${q(p)} 2>/dev/null || true`)
            } catch {
                // ignore
            }
        }
    }
}

async function applyHarnessRemote(lui, target, harness, remoteEnginePort, remoteWebPort) {
    const activeModel = lui.activeModel ?? resolveActiveModel(lui) ?? { name: "lui", engine: "llama-server", args: [] }
    const engineModule = engines[activeModel.engine]
    const ctxSize = engineModule?.contextSize?.(lui.state, activeModel) ?? null

    // The client's harness always points at localhost: the reverse
    // tunnel terminates on the client side, so the client's traffic
    // routes through localhost:<remote port> back to this machine.
    const ctx = harnessContext({
        activeModel,
        baseURL: `http://localhost:${remoteEnginePort}/v1`,
        webPort: remoteWebPort,
        websearch: lui.config.global.websearch,
        ctxSize
    })
    await applyHarness(sshTransport(target), harness, ctx, { enabled: true })
}

function resolveActiveModel(lui) {
    const name = lui.config.activeModelName
    if (!name) return null
    const m = lui.config.model[name]
    if (!m) return null
    return { name, engine: m.engine, args: m.args || [] }
}

function printShareSuccess(target, ports) {
    // engine.endpoint.host === null means "the engine lives on this
    // machine" — the tunnel's server end is localhost. A non-null
    // host means the engine is upstream of *us* (we're a remote
    // engine), and we want the tunnel to skip our process entirely
    // and land directly on the real model host.
    const engineHost = ports.engineEndpoint.host ?? "localhost"
    const enginePort = ports.engineEndpoint.port
    const engineFwd = `${ports.remoteEnginePort}:${engineHost}:${enginePort}`
    const webFwd = `${ports.remoteWebPort}:localhost:${ports.localWebPort}`
    const cmd = ports.websearch
        ? `ssh -R ${engineFwd} -R ${webFwd} ${sshTargetSpec(target)}`
        : `ssh -R ${engineFwd} ${sshTargetSpec(target)}`

    process.stdout.write(`\n  opencode configured on ${sshTargetSpec(target)}\n\n`)
    process.stdout.write(`  To connect from this machine, run in another terminal:\n\n`)
    process.stdout.write(`    ${cmd}\n\n`)
}
