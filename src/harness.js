// Harness registry, local apply flow, lui-web-search SKILL.md, and
// shared helpers (deriveModelName, inferContextSize, harnessContext).
// Each concrete harness lives in src/harness/<name>.js and exports a
// `harness` object — see the contract referenced from those files.

import fs from "node:fs"
import path from "node:path"
import os from "node:os"

import { harness as opencode } from "./harness/opencode.js"
import { harness as pi } from "./harness/pi.js"

export const harnesses = [opencode, pi]

// One `enabled` toggle per harness, shown under "Available Settings".
export function harnessSchemaDefaults(harnessModules) {
    return harnessModules.map((h) => ({
        path: `harness.${h.name}.enabled`,
        display: String(h.defaultEnabled)
    }))
}

// Caller assembles one of these and hands it to `harness.apply(existing, ctx)`.
// Pre-derives the values every harness wants, so harness modules don't
// need to know how to walk an args array or build a baseURL.
export function harnessContext({ activeModel, baseURL, enginePort, webPort, websearch }) {
    return {
        modelName: deriveModelName(activeModel?.name),
        baseURL: baseURL ?? `http://127.0.0.1:${enginePort}/v1`,
        ctxSize: inferContextSize(activeModel?.args || []),
        webPort,
        websearch: websearch !== false
    }
}

export function deriveModelName(activeKey) {
    if (!activeKey) return "lui"
    const tail = activeKey.split("/").pop() || activeKey
    return tail.split(":")[0].replace(/-GGUF$/, "") || "lui"
}

export function inferContextSize(args) {
    for (let i = 0; i < args.length; i++) {
        if ((args[i] === "-c" || args[i] === "--ctx-size") && i + 1 < args.length) {
            const n = parseInt(args[i + 1], 10)
            if (Number.isFinite(n) && n > 0) return n
        }
    }
    return 32768
}

// Walks every shipped harness so just-disabled ones get their stale
// SKILL.md swept; config edits stay gated on `enabled`. Pass
// `{ baseURL }` to point harness configs at a remote llama-server
// instead of the local one — used by `lui remote`.
export function applyAllLocal(lui, { baseURL } = {}) {
    const ctx = harnessContext({
        activeModel: lui.activeModel,
        baseURL,
        enginePort: lui.config.global.engine_port,
        webPort: lui.config.global.web_port,
        websearch: lui.config.global.websearch
    })
    for (const h of harnesses) {
        try {
            applyOneLocal(lui, h, ctx, isEnabled(lui, h))
        } catch (e) {
            process.stderr.write(`lui: harness "${h.name}" apply failed: ${e.message}\n`)
        }
    }
}

function isEnabled(lui, harness) {
    const sub = lui.config.harness?.[harness.name]
    if (sub && typeof sub.enabled === "boolean") return sub.enabled
    return harness.defaultEnabled
}

function applyOneLocal(lui, harness, ctx, enabled) {
    const dir = expandTilde(harness.configDir)
    const wantSkill = enabled && ctx.websearch

    // Skill add/remove runs regardless of `enabled` (sweep stale files).
    const skillDir = path.join(dir, "skills", "lui-web-search")
    const skillPath = path.join(skillDir, "SKILL.md")
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

export function renderWebsearchSkill(port) {
    return `---
name: lui-web-search
description: Web search via browser bookmarklet. Extracts live search results from Google to answer questions requiring up-to-date information. Use when the user asks to search the web, look something up, find recent information, or you need data past your training cutoff. Returns JSON results with title, url, and snippet.
license: BSD-2-Clause
---

# lui-web-search

lui's search endpoint opens a Google search tab in the user's real
browser. The user clicks a one-time-installed \`lui-grab\` bookmarklet on
the resulting page; the bookmarklet POSTs the rendered results back to
lui, which returns them to you.

## Endpoint

\`\`\`
GET http://127.0.0.1:${port}/bsearch?q=<URL-ENCODED QUERY>
\`\`\`

- \`q\` (required): the search query. URL-encode it.

The request **blocks for up to 120 seconds** while waiting for the
user to click the bookmarklet. On timeout you'll get HTTP 504.

## Response

JSON object:

\`\`\`json
{
  "results": [
    {"title": "...", "url": "https://...", "snippet": "..."}
  ],
  "warnings": ["..."]
}
\`\`\`

\`results\` is always present. \`warnings\` is present only when the
bookmarklet had something to tell you — for example, when Google's
CSS class names rotated and the bookmarklet had to fall back to a
structural selector to find results. **If \`warnings\` is non-empty,
surface each warning verbatim to the user** at the end of your reply
(under a short heading like "Note from lui-grab:"), on top of your
normal answer. The user is the only one who can act on it (usually
by updating lui).

An HTTP 504 means the user did not click the bookmarklet in time
(probably they were AFK or the browser tab got buried). Other 4xx/5xx
or an empty \`results\` array means the search failed — say so plainly
rather than fabricating answers.

## How to invoke

\`\`\`sh
curl -sG 'http://127.0.0.1:${port}/bsearch' \\
  --data-urlencode 'q=rust async traits 2026'
\`\`\`

On Windows (PowerShell):

\`\`\`powershell
$q = [uri]::EscapeDataString('rust async traits 2026')
curl.exe -s "http://127.0.0.1:${port}/bsearch?q=$q"
\`\`\`

Read the JSON, then write your answer as normal prose with markdown
links. Do not paste the raw JSON back into the chat. If you need the
body of a specific page, fetch that page separately.

## When to use

- User asks to "search the web", "look up", "google", "find recent", etc.
- You need information that post-dates your training cutoff.
- You need a canonical URL for documentation, a release, a spec, or an issue.

Do not use this for fetching content from a URL the user already gave
you — just fetch that URL directly.

## Important: this requires user action

Each call pops a browser tab the user must click on. Before invoking
this for the first time in a conversation, **tell the user what's about
to happen** so they can be ready, e.g.:

> "I'm going to search the web for that. When I do, a Google tab will
> open in your browser — click the **lui-grab** bookmarklet on it. If
> you haven't installed lui-grab yet, visit
> \`http://127.0.0.1:${port}/setup\` and drag it to your bookmarks bar
> first. (This URL is also shown in the lui status panel.)"

Then call \`/bsearch\` and wait. If the call returns HTTP 504, the user
didn't click in time — most likely they don't have the bookmarklet
installed yet. Stop, point them at the setup page, wait for them to
say it's ready, then retry.

Be deliberate about when to search:
- One search at a time. Do not fire parallel searches.
- Pick the best query first instead of iterating with small variations.
- Don't search for things you already know.
`
}
