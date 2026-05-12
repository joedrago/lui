// Harness registry + local apply flow.
//
// A harness is an external agent (opencode, pi, …) that lui teaches how
// to talk to llama-server. Each one lives in src/harness/<name>.js and
// exports a `harness` object. The minimum a new harness needs:
//
//   name              kebab-case identifier; doubles as the
//                     [harness.<name>] table key in lui.toml
//   defaultEnabled    boolean — used when [harness.<name>].enabled is unset
//   configDir         where the agent reads its config (~-prefixed)
//   configCandidates  basenames lui will look for in configDir, in
//                     preference order; the first match is read, the
//                     first entry is the write target if none exist
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
//   schema            [{ path, display, isArray? }] of knobs under
//                     [harness.<name>]; surfaced by `lui config` as
//                     Available Settings
//   sshPreflight(target, sshRun) → { ok, error? }
//                     async sanity check before `lui ssh` writes to a
//                     remote machine (e.g. is the agent installed there)
//
// The HarnessContext passed to apply():
//   modelName, baseURL, ctxSize, webPort, websearch

import fs from "node:fs"
import path from "node:path"
import os from "node:os"

import { renderWebsearchSkill } from "./web.js"
import { harness as opencode } from "./harness/opencode.js"
import { harness as pi } from "./harness/pi.js"

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

export function isHarnessEnabled(lui, harness) {
    const sub = lui.config.harness?.[harness.name]
    if (sub && typeof sub.enabled === "boolean") return sub.enabled
    return harness.defaultEnabled
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
// Pre-derives the values every harness wants. `ctxSize` is the caller's
// responsibility — it comes from the engine module via
// `engine.contextSize(state, model)`, which only returns a real number
// once the engine has reported Ready. Callers pass a fallback default
// when the engine doesn't know yet.
export function harnessContext({ activeModel, baseURL, enginePort, webPort, websearch, ctxSize }) {
    return {
        modelName: deriveModelName(activeModel?.name),
        baseURL: baseURL ?? `http://127.0.0.1:${enginePort}/v1`,
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

// Walks every shipped harness so just-disabled ones get their stale
// SKILL.md swept; config edits stay gated on `enabled`. Pass
// `{ baseURL }` to point harness configs at a remote llama-server
// instead of the local one — used by `lui remote`. `ctxSize` is the
// engine's current best answer; caller computes it via the engine
// module (or, for `lui remote`, takes it from the server's /config).
export function applyAllLocal(lui, { baseURL, ctxSize } = {}) {
    const ctx = harnessContext({
        activeModel: lui.activeModel,
        baseURL,
        enginePort: lui.config.global.engine_port,
        webPort: lui.config.global.web_port,
        websearch: lui.config.global.websearch,
        ctxSize
    })
    for (const h of harnesses) {
        try {
            applyOneLocal(lui, h, ctx, isHarnessEnabled(lui, h))
        } catch (e) {
            process.stderr.write(`lui: harness "${h.name}" apply failed: ${e.message}\n`)
        }
    }
}

function applyOneLocal(lui, harness, ctx, enabled) {
    const dir = expandTilde(harness.configDir)

    // Skill add/remove runs regardless of `enabled` (sweep stale files).
    // Skipped entirely for harnesses that don't declare a skills dir.
    if (harness.skillsDir) {
        const skillDir = path.join(dir, harness.skillsDir, "lui-web-search")
        const skillPath = path.join(skillDir, "SKILL.md")
        const wantSkill = enabled && ctx.websearch
        if (wantSkill) {
            fs.mkdirSync(skillDir, { recursive: true })
            const body = renderWebsearchSkill(ctx.webPort)
            if (!fs.existsSync(skillPath) || fs.readFileSync(skillPath, "utf8") !== body) {
                fs.writeFileSync(skillPath, body)
            }
        } else if (fs.existsSync(skillPath)) {
            fs.unlinkSync(skillPath)
            try {
                fs.rmdirSync(skillDir)
            } catch {
                // ignore — directory not empty or already gone
            }
        }
    }

    if (!enabled) return

    fs.mkdirSync(dir, { recursive: true })
    const file = pickConfigFile(dir, harness.configCandidates)
    const existing = fs.existsSync(file) ? fs.readFileSync(file, "utf8") : ""

    if (existing && harness.needsBackup && harness.needsBackup(existing)) {
        const backup = file + ".luibackup"
        if (!fs.existsSync(backup)) {
            fs.writeFileSync(backup, existing)
            lui.addWarning?.(`backed up ${file} → ${backup} before first lui write`)
        }
    }

    const next = harness.apply(existing, ctx)
    if (next !== existing) {
        const tmp = file + ".tmp"
        fs.writeFileSync(tmp, next)
        fs.renameSync(tmp, file)
    }
}

function pickConfigFile(dir, candidates) {
    for (const name of candidates) {
        const p = path.join(dir, name)
        if (fs.existsSync(p)) return p
    }
    return path.join(dir, candidates[0])
}

export function expandTilde(p) {
    if (p.startsWith("~/")) return path.join(os.homedir(), p.slice(2))
    if (p === "~") return os.homedir()
    return p
}

