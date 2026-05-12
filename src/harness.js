// Harness registry + local apply flow.
//
// A harness is an external agent (opencode, pi, …) that lui teaches how
// to talk to llama-server. Each one lives in src/harness/<name>.js and
// exports a `harness` object. The minimum a new harness needs:
//
//   name              kebab-case identifier; doubles as the
//                     [harness.<name>] table key in lui.toml
//   configDir         where the agent reads its config (~-prefixed)
//   configCandidates  basenames lui will look for in configDir, in
//                     preference order; the first match is read, the
//                     first entry is the write target if none exist
//   schema            [{ path, default, isArray? }] of knobs under
//                     [harness.<name>]; surfaced by `lui config` as
//                     Available Settings. Must include an "enabled"
//                     entry with the default-enabled state.
//   apply(existing, ctx) → string
//                     given the current file contents (or "") and a
//                     HarnessContext, return the new file contents
//
// Optional fields:
//
//   skillsDir         relative path under configDir where the agent
//                     looks for SKILL.md files. When set, lui drops
//                     the lui-web-search skill in there (and sweeps
//                     it when disabled).
//   needsBackup(existing) → boolean
//                     return true on first write to a config lui didn't
//                     author, so the original gets stashed as .luibackup
//   sshPreflight(target, sshRun) → { ok, error? }
//                     async sanity check before `lui ssh` writes to a
//                     remote machine (e.g. is the agent installed there)
//
// The HarnessContext passed to apply():
//   modelName, baseURL, ctxSize, webPort, websearch

import fs from "node:fs"

import { renderWebsearchSkill } from "./web.js"
import { harness as opencode } from "./harness/opencode.js"
import { harness as pi } from "./harness/pi.js"
import { expandTilde } from "./util.js"

export const harnesses = [opencode, pi]

// Schema keys the framework reads generically from every harness. A
// harness that omits one of these is a bug — fail loudly at import
// rather than silently treating "missing" as "false" later. Add to
// this list whenever you introduce framework code that consults
// `[harness.<name>].<key>` for every harness.
const REQUIRED_HARNESS_SCHEMA_KEYS = ["enabled"]

for (const h of harnesses) {
    for (const key of REQUIRED_HARNESS_SCHEMA_KEYS) {
        if (!(h.schema ?? []).some((s) => s.path === key)) {
            process.stderr.write(
                `lui: harness "${h.name}" is missing required schema entry "${key}". ` +
                    `Add it to the harness's schema array.\n`
            )
            process.exit(1)
        }
    }
}

function harnessSchemaDefault(harness, key) {
    return (harness.schema ?? []).find((s) => s.path === key)?.default
}

export function isHarnessEnabled(lui, harness) {
    const sub = lui.config.harness?.[harness.name]
    if (sub && typeof sub.enabled === "boolean") return sub.enabled
    return harnessSchemaDefault(harness, "enabled") === true
}

// Each harness's own `schema` entries, prefixed with `harness.<name>.`
// and surfaced by `lui config` under "Available Settings".
export function harnessSchemaDefaults() {
    const out = []
    for (const h of harnesses) {
        for (const s of h.schema ?? []) {
            out.push({ ...s, path: `harness.${h.name}.${s.path}` })
        }
    }
    return out
}

// Caller assembles one of these and hands it to `harness.apply(existing, ctx)`.
// Pre-derives the values every harness wants. The caller computes the
// model's externally-reachable baseURL itself, since the right answer
// depends on context (req hostname for /config-driven flows, the
// SSH-tunnel localhost port for `lui ssh`, etc.). `ctxSize` likewise
// comes from the engine — `engine.contextSize(state, model)` only
// returns a real number once Ready, so callers pass a fallback for
// early/offline use.
export function harnessContext({ activeModel, baseURL, webPort, websearch, ctxSize }) {
    return {
        modelName: deriveModelName(activeModel?.name),
        baseURL,
        ctxSize: ctxSize ?? DEFAULT_CTX_SIZE,
        webPort,
        websearch: websearch !== false
    }
}

const DEFAULT_CTX_SIZE = 32768

export function deriveModelName(activeKey) {
    if (!activeKey) return "lui"
    const tail = activeKey.split("/").pop() || activeKey
    return tail.split(":")[0].replace(/-GGUF$/, "") || "lui"
}

// Minimal transport interface — methods applyHarness drives:
//   resolve(home-prefixed path) → path the other methods accept
//   exists(p), read(p), write(p, body), remove(p), mkdirp(p), tryRmDir(p)
// `localTransport` resolves "~/..." to a real fs path; `sshTransport`
// (in ssh.js) passes "~/..." through to the remote shell unchanged.
export const localTransport = {
    name: "local",
    resolve(p) {
        return expandTilde(p)
    },
    async exists(p) {
        return fs.existsSync(p)
    },
    async read(p) {
        return fs.existsSync(p) ? fs.readFileSync(p, "utf8") : ""
    },
    async write(p, body) {
        const tmp = p + ".tmp"
        fs.writeFileSync(tmp, body)
        fs.renameSync(tmp, p)
    },
    async remove(p) {
        if (fs.existsSync(p)) fs.unlinkSync(p)
    },
    async mkdirp(p) {
        fs.mkdirSync(p, { recursive: true })
    },
    async tryRmDir(p) {
        try {
            fs.rmdirSync(p)
        } catch {
            // ignore — non-empty or missing
        }
    }
}

// Shared apply-flow for a single harness over any transport. Returns
// the path written (or null if nothing changed) so callers can log.
//
// `opts.onBackup(file, backup)` fires when an existing config was
// stashed as .luibackup before lui's first write. Default noop.
export async function applyHarness(transport, harness, ctx, { enabled, onBackup } = {}) {
    const join = (...parts) => parts.join("/")
    const dir = transport.resolve(harness.configDir)

    // Skill add/remove runs regardless of `enabled` (sweep stale files).
    if (harness.skillsDir) {
        const skillDir = join(dir, harness.skillsDir, "lui-web-search")
        const skillPath = join(skillDir, "SKILL.md")
        const wantSkill = enabled && ctx.websearch
        if (wantSkill) {
            await transport.mkdirp(skillDir)
            const body = renderWebsearchSkill(ctx.webPort)
            const cur = await transport.read(skillPath)
            if (cur !== body) await transport.write(skillPath, body)
        } else if (await transport.exists(skillPath)) {
            await transport.remove(skillPath)
            await transport.tryRmDir(skillDir)
        }
    }

    if (!enabled) return null

    await transport.mkdirp(dir)
    const file = await pickConfigFile(transport, dir, harness.configCandidates)
    const existing = await transport.read(file)

    if (existing && harness.needsBackup?.(existing)) {
        const backup = file + ".luibackup"
        if (!(await transport.exists(backup))) {
            await transport.write(backup, existing)
            onBackup?.(file, backup)
        }
    }

    const next = harness.apply(existing, ctx)
    if (next !== existing) {
        await transport.write(file, next)
        return file
    }
    return null
}

async function pickConfigFile(transport, dir, candidates) {
    for (const name of candidates) {
        const p = `${dir}/${name}`
        if (await transport.exists(p)) return p
    }
    return `${dir}/${candidates[0]}`
}

// Walks every shipped harness so just-disabled ones get their stale
// SKILL.md swept; config edits stay gated on `enabled`. `baseURL`
// overrides where the harness should point — used by a future remote
// engine that knows the upstream URL. Defaults to the local engine's
// endpoint with a 127.0.0.1 host (harness is on this machine).
export async function applyAllLocal(lui, { baseURL, ctxSize } = {}) {
    const resolvedBaseURL = baseURL ?? localBaseURL(lui)
    const ctx = harnessContext({
        activeModel: lui.activeModel,
        baseURL: resolvedBaseURL,
        webPort: lui.config.global.web_port,
        websearch: lui.config.global.websearch,
        ctxSize
    })
    const onBackup = (file, backup) => lui.addWarning?.(`backed up ${file} → ${backup} before first lui write`)
    for (const h of harnesses) {
        try {
            await applyHarness(localTransport, h, ctx, { enabled: isHarnessEnabled(lui, h), onBackup })
        } catch (e) {
            process.stderr.write(`lui: harness "${h.name}" apply failed: ${e.message}\n`)
        }
    }
}

function localBaseURL(lui) {
    const ep = lui.engineModule?.endpoint?.(lui)
    if (!ep) return null
    return `http://${ep.host ?? "127.0.0.1"}:${ep.port}/v1`
}
