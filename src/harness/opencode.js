// opencode harness: writes ~/.config/opencode/opencode.{jsonc,json}.
// Inserts a `provider.lui` block pointing opencode at llama-server,
// and a `permission.bash` curl allowlist for lui's web-search port.

import { modify, applyEdits, parseTree, findNodeAtLocation } from "jsonc-parser"

const FORMAT = { tabSize: 2, insertSpaces: true, eol: "\n" }

export const harness = {
    name: "opencode",
    defaultEnabled: true,
    configDir: "~/.config/opencode",
    configCandidates: ["opencode.jsonc", "opencode.json"],
    skillsDir: "skills",
    schema: [{ path: "enabled", display: "true" }],

    apply(existing, ctx) {
        let text = existing.trim() ? existing : "{}\n"
        text = setProviderLui(text, ctx.modelName, ctx.baseURL, ctx.ctxSize)
        text = setPermissionBash(text, ctx.webPort, ctx.websearch)
        return text
    },

    needsBackup(existing) {
        if (!existing.trim()) return false
        const tree = parseTree(existing, [], { allowTrailingComma: true })
        if (!tree) return true
        const lui = findNodeAtLocation(tree, ["provider", "lui"])
        return !lui
    },

    async sshPreflight(target, sshRun) {
        const probe =
            'command -v opencode || bash -lc \'command -v opencode\' || { [ -x "$HOME/.opencode/bin/opencode" ] && echo "$HOME/.opencode/bin/opencode"; }'
        try {
            const out = await sshRun(target, probe)
            if (out.trim()) return { ok: true }
            return { ok: false, error: `opencode not found on ${target.user}@${target.host}. Install it there first.` }
        } catch (e) {
            return { ok: false, error: `opencode preflight on ${target.user}@${target.host} failed: ${e.message}` }
        }
    }
}

function setProviderLui(text, modelName, baseURL, ctxSize) {
    const luiValue = {
        name: "lui",
        npm: "@ai-sdk/openai-compatible",
        options: {
            baseURL,
            toolParser: [{ type: "raw-function-call" }, { type: "json" }]
        },
        models: {
            [modelName]: {
                name: modelName,
                supportsToolCalls: true,
                limit: { context: ctxSize, input: ctxSize, output: 8192 }
            }
        }
    }
    const edits = modify(text, ["provider", "lui"], luiValue, { formattingOptions: FORMAT })
    return applyEdits(text, edits)
}

function setPermissionBash(text, webPort, websearch) {
    const currentPattern = `curl*http://127.0.0.1:${webPort}/*`
    let cur = text

    // Sweep stale loopback patterns at any port, keeping the current one.
    const tree = parseTree(cur, [], { allowTrailingComma: true })
    const stale = /^curl\*http:\/\/127\.0\.0\.1:\d+\/\*$/
    const bashNode = tree && findNodeAtLocation(tree, ["permission", "bash"])
    if (bashNode?.type === "object") {
        for (const prop of bashNode.children ?? []) {
            const key = prop.children?.[0]
            if (!key || typeof key.value !== "string") continue
            if (key.value === currentPattern) continue
            if (!stale.test(key.value)) continue
            const edits = modify(cur, ["permission", "bash", key.value], undefined, { formattingOptions: FORMAT })
            cur = applyEdits(cur, edits)
        }
    }

    if (websearch) {
        const edits = modify(cur, ["permission", "bash", currentPattern], "allow", { formattingOptions: FORMAT })
        cur = applyEdits(cur, edits)
    } else {
        // jsonc-parser's modify(…, undefined) throws on a path that
        // doesn't already exist (e.g. fresh install with no permission
        // block). Only attempt the delete when the entry is actually
        // there.
        const post = parseTree(cur, [], { allowTrailingComma: true })
        if (post && findNodeAtLocation(post, ["permission", "bash", currentPattern])) {
            const edits = modify(cur, ["permission", "bash", currentPattern], undefined, { formattingOptions: FORMAT })
            cur = applyEdits(cur, edits)
        }
    }
    return cur
}
