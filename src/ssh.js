// `lui ssh USER@HOST` configures harnesses on a remote client for a
// reverse-tunneled connection back to this machine. The inverse —
// running lui on a client that fetches its engine state from somewhere
// else — is just the `remote` engine: `lui add NAME remote HOST[:PORT]`.

/** @import { Lui } from "./lui.js" */
/** @import { Harness, SshTarget, Transport, Endpoint } from "./types.js" */

import http from "node:http"
import process from "node:process"
import { spawn } from "node:child_process"

import { harnesses, applyHarness, harnessContext, isHarnessEnabled } from "./harness.js"
import { CONFIG_VERSION } from "./wire.js"

// applyHarness does ~8–10 sequential remote ops per harness (exists,
// read, mkdirp, write, ...). Without multiplexing each one pays a fresh
// TCP+auth handshake and the whole `lui ssh` run drags. ControlMaster
// reuses one connection for all of them. Windows OpenSSH does not
// implement ControlMaster (Win32-OpenSSH issue #405), so the gate
// falls back to a plain `ssh` invocation per call there.
//
// ControlPath lives in /tmp rather than os.tmpdir() because macOS's
// per-user tmpdir (/var/folders/.../T/) plus ssh's own atomic-create
// suffix overruns the 104-byte Unix-domain-socket path limit. The
// %C token is a per-target hash so paths don't collide across runs.
const SSH_MUX_ARGS =
    process.platform === "win32"
        ? []
        : ["-o", "ControlMaster=auto", "-o", "ControlPath=/tmp/lui-cm-%C", "-o", "ControlPersist=60"]

/** @param {Lui} lui @param {string} spec */
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

    const localWebPort = lui.config.global.web_port
    const websearch = lui.config.global.websearch !== false

    // Source of truth for the tunnel's server end + harness context: a
    // /config call against the lui that's already running on this
    // machine. That endpoint has the resolved base_url cached (after
    // its own start() completed), so for a remote-engine session it
    // already names the real upstream host:port — no second-guessing
    // here. If no lui is running we have nothing to share; fail loudly.
    const localConfig = await fetchLocalConfig(localWebPort)
    const engineEndpoint = parseEndpointFromBaseURL(localConfig.base_url)
    if (!engineEndpoint) {
        process.stderr.write(
            `lui ssh: the running lui at 127.0.0.1:${localWebPort} did not report a base_url ` +
                `(its engine may not be Ready yet — wait for it to finish loading and re-run).\n`
        )
        process.exit(1)
    }
    const sessionModel = { name: localConfig.active_model ?? null }
    const sessionCtxSize = typeof localConfig.context_size === "number" ? localConfig.context_size : null
    const sessionServedName = localConfig.served_model ?? null

    process.stdout.write("\n")
    for (const h of enabled) {
        if (h.sshPreflight) {
            const ok = await h.sshPreflight(target, sshRun)
            if (!ok.ok) {
                process.stderr.write(`lui: ${ok.error}\n`)
                process.exit(1)
            }
        }
        await applyHarnessRemote({
            lui,
            target,
            harness: h,
            remoteEnginePort,
            remoteWebPort,
            sessionModel,
            sessionCtxSize,
            sessionServedName
        })
        process.stdout.write(`  ${h.name} configured on ${sshTargetSpec(target)}\n`)
    }

    printShareSuccess(target, { engineEndpoint, localWebPort, remoteEnginePort, remoteWebPort, websearch })
}

/** @param {string} verb @returns {string} */
function luiNeedsHarnessError(verb) {
    const all = harnesses.map((h) => h.name).join(", ")
    return (
        `lui ${verb}: no harnesses are enabled, which makes this subcommand a no-op.\n` +
        `Enable at least one before re-running, e.g. \`lui config set harness.${harnesses[0].name}.enabled true\`.\n` +
        `Available: ${all}.\n`
    )
}

/** @param {string} s @returns {SshTarget | null} */
function parseShareTarget(s) {
    const i = s.indexOf("@")
    if (i <= 0 || i >= s.length - 1) return null
    return { user: s.slice(0, i), host: s.slice(i + 1) }
}

/** @returns {number} */
function pickRemotePort() {
    const base = 18000 + Math.floor(Math.random() * 11000)
    return base
}

const LOCAL_CONFIG_TIMEOUT_MS = 2000

// /config on the locally-running lui is the authoritative source for
// what `lui ssh` needs to share — base_url, active model name, context
// size. Anything else risks drifting from what the live session is
// actually serving.
/** @param {number} webPort @returns {Promise<any>} */
async function fetchLocalConfig(webPort) {
    let cfg
    try {
        cfg = await httpGetLocalJSON(webPort, "/config", LOCAL_CONFIG_TIMEOUT_MS)
    } catch (e) {
        process.stderr.write(
            `lui ssh: could not reach a running lui at 127.0.0.1:${webPort} (${/** @type {Error} */ (e).message}).\n` +
                `Start one in another terminal first (e.g. \`lui run NAME\`), then re-run \`lui ssh\`.\n`
        )
        process.exit(1)
    }
    if (cfg.version !== CONFIG_VERSION) {
        process.stderr.write(
            `lui ssh: the running lui reports /config version ${cfg.version}, this binary speaks ${CONFIG_VERSION}. ` +
                `Run matching builds on both ends.\n`
        )
        process.exit(1)
    }
    return cfg
}

/** @param {string | null | undefined} baseURL @returns {Endpoint | null} */
function parseEndpointFromBaseURL(baseURL) {
    if (!baseURL) return null
    try {
        const u = new URL(baseURL)
        const port = u.port ? parseInt(u.port, 10) : u.protocol === "https:" ? 443 : 80
        return { host: u.hostname, port }
    } catch {
        return null
    }
}

/** @param {number} port @param {string} path @param {number} timeoutMs @returns {Promise<any>} */
function httpGetLocalJSON(port, path, timeoutMs) {
    return new Promise((resolve, reject) => {
        const req = http.get({ host: "127.0.0.1", port, path, timeout: timeoutMs }, (res) => {
            let body = ""
            res.setEncoding("utf8")
            res.on("data", (c) => (body += c))
            res.on("end", () => {
                const status = res.statusCode ?? 0
                if (status < 200 || status >= 300) {
                    return reject(new Error(`HTTP ${status} from ${path}`))
                }
                try {
                    resolve(JSON.parse(body))
                } catch (e) {
                    reject(new Error(`unparseable JSON from ${path}: ${/** @type {Error} */ (e).message}`))
                }
            })
        })
        req.on("error", (e) => reject(e))
        req.on("timeout", () => req.destroy(new Error(`timeout after ${timeoutMs}ms`)))
    })
}

/** @param {SshTarget} target @returns {string} */
function sshTargetSpec(target) {
    return `${target.user}@${target.host}`
}

/** @param {SshTarget} target @param {string} command @param {string} [stdinText] @returns {Promise<string>} */
async function sshRun(target, command, stdinText) {
    return new Promise((resolve, reject) => {
        const child = spawn("ssh", [...SSH_MUX_ARGS, sshTargetSpec(target), command], {
            stdio: ["pipe", "pipe", "pipe"]
        })
        let stdout = ""
        let stderr = ""
        child.stdout?.setEncoding("utf8")
        child.stderr?.setEncoding("utf8")
        child.stdout?.on("data", (c) => (stdout += c))
        child.stderr?.on("data", (c) => (stderr += c))
        child.on("error", (e) => reject(new Error(`failed to spawn ssh: ${e.message}`)))
        child.on("exit", (code) => {
            if (code === 0) return resolve(stdout)
            const msg = (stderr || stdout || `ssh exited with code ${code}`).trim()
            reject(new Error(msg))
        })
        if (stdinText != null) {
            child.stdin?.end(stdinText)
        } else {
            child.stdin?.end()
        }
    })
}

// SSH transport: paths stay as "~/..." since the remote shell expands ~.
/** @param {SshTarget} target @returns {Transport} */
function sshTransport(target) {
    // Tilde expansion only fires on an *unquoted* leading `~`, so a
    // naive `'~/foo'` ends up as a literal `~` directory in CWD. Pull
    // the `~/` outside the quotes; single-quote the rest to neutralize
    // any other metacharacters in the path.
    /** @param {string} x */
    const sq = (x) => `'${String(x).replace(/'/g, `'\\''`)}'`
    /** @param {string} s */
    const q = (s) => {
        if (s === "~") return "~"
        if (s.startsWith("~/")) return `~/${sq(s.slice(2))}`
        return sq(s)
    }
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

/** @param {{ lui: Lui, target: SshTarget, harness: Harness, remoteEnginePort: number, remoteWebPort: number, sessionModel: { name: string | null }, sessionCtxSize: number | null, sessionServedName: string | null }} args */
async function applyHarnessRemote({
    lui,
    target,
    harness,
    remoteEnginePort,
    remoteWebPort,
    sessionModel,
    sessionCtxSize,
    sessionServedName
}) {
    // The client's harness always points at localhost: the reverse
    // tunnel terminates on the client side, so the client's traffic
    // routes through localhost:<remote port> back to this machine.
    const ctx = harnessContext({
        activeModel: /** @type {any} */ (sessionModel),
        baseURL: `http://localhost:${remoteEnginePort}/v1`,
        webPort: remoteWebPort,
        websearch: lui.config.global.websearch,
        ctxSize: sessionCtxSize,
        servedName: sessionServedName
    })
    await applyHarness({ transport: sshTransport(target), harness, ctx, enabled: true })
}

/** @param {SshTarget} target @param {{ engineEndpoint: Endpoint, localWebPort: number, remoteEnginePort: number, remoteWebPort: number, websearch: boolean }} ports */
function printShareSuccess(target, ports) {
    // engineEndpoint comes straight from the running lui's /config
    // base_url: for a llama-server session that's 127.0.0.1:engine_port;
    // for a remote-engine session it's the already-resolved upstream
    // host:port, so the tunnel skips our process entirely and lands
    // directly on the real model host.
    const engineFwd = `${ports.remoteEnginePort}:${ports.engineEndpoint.host}:${ports.engineEndpoint.port}`
    const webFwd = `${ports.remoteWebPort}:localhost:${ports.localWebPort}`
    const cmd = ports.websearch
        ? `ssh -R ${engineFwd} -R ${webFwd} ${sshTargetSpec(target)}`
        : `ssh -R ${engineFwd} ${sshTargetSpec(target)}`

    process.stdout.write(`\n  To connect from this machine, run in another terminal:\n\n`)
    process.stdout.write(`    ${cmd}\n\n`)
}
