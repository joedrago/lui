// `lui ssh USER@HOST` configures harnesses on a remote client for a
// reverse-tunneled connection back to this machine. The inverse —
// running lui on a client that fetches its engine state from somewhere
// else — is just the `remote` engine: `lui add NAME remote HOST[:PORT]`.

/** @import { Lui } from "./lui.js" */
/** @import { Harness, SshTarget, SshRemote, RemotePlatform, Transport, Endpoint } from "./types.js" */

import http from "node:http"
import process from "node:process"
import { Buffer } from "node:buffer"
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
    const sessionMaxOutputTokens = typeof localConfig.max_output_tokens === "number" ? localConfig.max_output_tokens : null
    const sessionServedName = localConfig.served_model ?? null

    const remote = await connectRemote(target)

    process.stdout.write("\n")
    if (remote.platform === "win32") {
        process.stdout.write(`  ${remote.spec} is a Windows client — configuring it through PowerShell\n`)
    }
    for (const h of enabled) {
        if (h.sshPreflight) {
            const ok = await h.sshPreflight(remote)
            if (!ok.ok) {
                process.stderr.write(`lui: ${ok.error}\n`)
                process.exit(1)
            }
        }
        await applyHarnessRemote({
            lui,
            remote,
            harness: h,
            remoteEnginePort,
            remoteWebPort,
            sessionModel,
            sessionCtxSize,
            sessionMaxOutputTokens,
            sessionServedName
        })
        process.stdout.write(`  ${h.name} configured on ${remote.spec}\n`)
    }

    printShareSuccess(target, { engineEndpoint, localWebPort, remoteEnginePort, remoteWebPort, websearch })
}

/** @param {string} verb @returns {string} */
function luiNeedsHarnessError(verb) {
    const all = harnesses.map((h) => h.name).join(", ")
    return (
        `lui ${verb}: no harnesses are enabled, which makes this subcommand a no-op.\n` +
        `Enable at least one before re-running, e.g. \`lui set harness.${harnesses[0].name}.enabled true\`.\n` +
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

// Everything below this line assumed a POSIX shell on the far end until
// Windows clients showed up. sshd on Windows hands the command line to
// cmd.exe by default, which answers `mkdir -p '~/.config/opencode'` with
// "The syntax of the command is incorrect." — so lui probes the remote
// once and then speaks whichever dialect actually lives there.

/** @param {SshTarget} target @returns {Promise<SshRemote>} */
async function connectRemote(target) {
    const spec = sshTargetSpec(target)
    const probe = await probeRemotePlatform(target)
    if (!probe) {
        process.stderr.write(
            `lui ssh: could not tell what kind of shell ${spec} runs — neither \`uname\` nor PowerShell answered.\n` +
                `Check that \`ssh ${spec}\` works and lands in a POSIX shell, cmd.exe, or PowerShell.\n`
        )
        process.exit(1)
    }
    if (probe.platform === "win32" && !probe.home) {
        process.stderr.write(
            `lui ssh: ${spec} answers as Windows but reports no home directory — both $HOME and ` +
                `%USERPROFILE% are empty in its ssh sessions, so there is nowhere to put the harness configs.\n`
        )
        process.exit(1)
    }

    const transport = probe.platform === "win32" ? windowsTransport(target, probe) : posixTransport(target, probe.platform)
    /** @param {string} command @param {string} [stdinText] */
    const run = (command, stdinText) => sshRun(target, command, stdinText)

    return {
        target,
        spec,
        platform: probe.platform,
        transport,
        run,
        async which(name) {
            const found = probe.platform === "win32" ? await windowsWhich(target, probe, name) : await posixWhich(target, name)
            return found || null
        },
        async exists(p) {
            return await transport.exists(transport.resolve(p))
        }
    }
}

/** @typedef {{ platform: RemotePlatform, psExe?: string, home?: string }} RemoteProbe */

// Windows PowerShell ships on every box that can run Win32-OpenSSH;
// `pwsh` covers the PowerShell-7-only installs that trimmed it.
const WINDOWS_SHELLS = ["powershell", "pwsh"]
const WIN_PROBE_MARKER = "lui-probe"

// `uname -s` settles the common case in a single round trip. On Windows
// it either isn't there at all (cmd.exe/PowerShell reject it) or it
// reports an emulation layer — MINGW64_NT under Git Bash — which still
// means native Windows tooling on the other side, so both answers fall
// through to the PowerShell probe.
/** @param {SshTarget} target @returns {Promise<RemoteProbe | null>} */
async function probeRemotePlatform(target) {
    let uname = ""
    try {
        uname = (await sshRun(target, "uname -s")).trim().toLowerCase()
    } catch {
        uname = ""
    }
    if (uname === "darwin") return { platform: "darwin" }
    if (uname && !/mingw|msys|cygwin|windows/.test(uname)) return { platform: "linux" }

    // Every script writes results through [Console]::Out rather than the
    // pipeline: pipeline output goes through PowerShell's formatter,
    // which hard-wraps at the host's buffer width and would chop a long
    // base64 payload into pieces.
    const script =
        `$h = $HOME\n` +
        `if (-not $h) { $h = $env:USERPROFILE }\n` +
        `[Console]::Out.Write("${WIN_PROBE_MARKER}|" + $env:OS + "|" + $h)`
    for (const psExe of WINDOWS_SHELLS) {
        let out = ""
        try {
            out = await sshRun(target, psCommand(psExe, script))
        } catch {
            continue
        }
        const m = new RegExp(`${WIN_PROBE_MARKER}\\|([^|]*)\\|(.*)$`).exec(out.trim())
        if (m && m[1] === "Windows_NT") return { platform: "win32", psExe, home: normalizeWinPath(m[2]) }
    }
    return null
}

// -EncodedCommand takes UTF-16LE base64, whose wire form is nothing but
// [A-Za-z0-9+/=]. That matters more than it looks: ssh concatenates the
// command into one string and lets the *remote* default shell parse it,
// so a payload with no metacharacters is the only thing that survives
// cmd.exe, PowerShell, and a Git-for-Windows bash identically. File
// bodies ride in on stdin instead, keeping the command line short.
/** @param {string} psExe @param {string} script @returns {string} */
function psCommand(psExe, script) {
    const full = `$ErrorActionPreference = 'Stop'\n$ProgressPreference = 'SilentlyContinue'\n${script}`
    return `${psExe} -NoProfile -NonInteractive -EncodedCommand ${Buffer.from(full, "utf16le").toString("base64")}`
}

/** @param {string} p @returns {string} */
function normalizeWinPath(p) {
    return p.trim().replace(/\\/g, "/").replace(/\/+$/, "")
}

// POSIX transport: paths stay as "~/..." since the remote shell expands ~.
/** @param {SshTarget} target @param {RemotePlatform} platform @returns {Transport} */
function posixTransport(target, platform) {
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
        platform,
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

// Windows transport: the probe already told us the remote home, so
// "~/..." is expanded here rather than left for a shell that wouldn't
// expand it. Paths keep forward slashes throughout — .NET accepts them
// on Windows, and it saves escaping backslashes into every script.
// File contents move as base64 in both directions so the remote console
// code page never gets a chance to mangle them.
/** @param {SshTarget} target @param {RemoteProbe} probe @returns {Transport} */
function windowsTransport(target, probe) {
    const home = probe.home ?? "~"
    const psExe = probe.psExe ?? WINDOWS_SHELLS[0]
    /** @param {string} script @param {string} [stdinText] */
    const ps = (script, stdinText) => sshRun(target, psCommand(psExe, script), stdinText)
    const q = psQuote

    return {
        name: "ssh",
        platform: "win32",
        resolve(p) {
            if (p === "~") return home
            if (p.startsWith("~/")) return `${home}/${p.slice(2)}`
            return p
        },
        async exists(p) {
            try {
                const out = await ps(`[Console]::Out.Write($(if (Test-Path -LiteralPath ${q(p)}) { '1' } else { '0' }))`)
                return out.trim() === "1"
            } catch {
                return false
            }
        },
        async read(p) {
            const out = await ps(
                `if (Test-Path -LiteralPath ${q(p)}) { [Console]::Out.Write([Convert]::ToBase64String([IO.File]::ReadAllBytes(${q(p)}))) }`
            ).catch(() => "")
            return decodeBase64Text(out)
        },
        async write(p, body) {
            await ps(
                `$b = [Convert]::FromBase64String([Console]::In.ReadToEnd())\n` +
                    `$d = [IO.Path]::GetDirectoryName(${q(p)})\n` +
                    `if ($d) { [IO.Directory]::CreateDirectory($d) | Out-Null }\n` +
                    `[IO.File]::WriteAllBytes(${q(p)}, $b)`,
                Buffer.from(body, "utf8").toString("base64")
            )
        },
        async remove(p) {
            try {
                await ps(`Remove-Item -LiteralPath ${q(p)} -Force -ErrorAction SilentlyContinue`)
            } catch {
                // ignore
            }
        },
        async mkdirp(p) {
            await ps(`[IO.Directory]::CreateDirectory(${q(p)}) | Out-Null`)
        },
        async tryRmDir(p) {
            // Non-recursive Delete throws on a non-empty directory,
            // which is exactly the "only if empty" semantics wanted.
            try {
                await ps(`try { [IO.Directory]::Delete(${q(p)}) } catch { }`)
            } catch {
                // ignore
            }
        }
    }
}

/** @param {string} s @returns {string} */
function psQuote(s) {
    return `'${String(s).replace(/'/g, "''")}'`
}

/** @param {string} b64 @returns {string} */
function decodeBase64Text(b64) {
    const trimmed = b64.trim()
    if (!trimmed) return ""
    // Windows editors happily leave a UTF-8 BOM on a config file; left
    // in place it would ride into every jsonc-parser call downstream.
    return Buffer.from(trimmed, "base64")
        .toString("utf8")
        .replace(/^\uFEFF/, "")
}

// Command names come from lui's own harness definitions, never from
// user input — an odd one is a bug in a harness, so say so loudly
// rather than quietly interpolating it into a remote command line.
/** @param {string} name @returns {string} */
function assertCommandName(name) {
    if (!/^[A-Za-z0-9._-]+$/.test(name)) throw new Error(`unsupported remote command name: ${name}`)
    return name
}

// A non-interactive ssh session skips the login shell, so a tool
// installed via ~/.profile or ~/.zshrc PATH edits is invisible to a
// plain `command -v`; the `bash -lc` retry covers that.
/** @param {SshTarget} target @param {string} name @returns {Promise<string>} */
async function posixWhich(target, name) {
    const n = assertCommandName(name)
    try {
        const out = await sshRun(target, `command -v ${n} || bash -lc 'command -v ${n}' || true`)
        return out.trim()
    } catch {
        return ""
    }
}

/** @param {SshTarget} target @param {RemoteProbe} probe @param {string} name @returns {Promise<string>} */
async function windowsWhich(target, probe, name) {
    const n = assertCommandName(name)
    const script =
        `$c = Get-Command -Name ${psQuote(n)} -ErrorAction SilentlyContinue | Select-Object -First 1\n` +
        `if ($c) { $s = [string]$c.Source; if (-not $s) { $s = [string]$c.Name }; [Console]::Out.Write($s) }`
    try {
        const out = await sshRun(target, psCommand(probe.psExe ?? WINDOWS_SHELLS[0], script))
        return out.trim()
    } catch {
        return ""
    }
}

/** @param {{ lui: Lui, remote: SshRemote, harness: Harness, remoteEnginePort: number, remoteWebPort: number, sessionModel: { name: string | null }, sessionCtxSize: number | null, sessionMaxOutputTokens: number | null, sessionServedName: string | null }} args */
async function applyHarnessRemote({
    lui,
    remote,
    harness,
    remoteEnginePort,
    remoteWebPort,
    sessionModel,
    sessionCtxSize,
    sessionMaxOutputTokens,
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
        maxOutputTokens: sessionMaxOutputTokens,
        servedName: sessionServedName
    })
    await applyHarness({
        transport: remote.transport,
        harness,
        ctx,
        enabled: true,
        config: lui.config.harness?.[harness.name] ?? {}
    })
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
