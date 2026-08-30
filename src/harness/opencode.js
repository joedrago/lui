// opencode harness: writes ~/.config/opencode/opencode.{jsonc,json}.
// Inserts a `provider.lui` block pointing opencode at llama-server,
// and a `permission.bash` curl allowlist for lui's web-search port.
//
// `reasoning_effort` (unset by default) writes
// provider.lui.models.<model>.options.reasoningEffort. opencode feeds
// model options to the AI SDK, whose openai-compatible provider maps
// reasoningEffort to the request body's `reasoning_effort`, which
// llama-server hands to the chat template. Qwen3.8 accepts low,
// medium, xhigh; `none` disables thinking entirely.

/** @import { Harness } from "../types.js" */

import { modify, applyEdits, parseTree, findNodeAtLocation } from "jsonc-parser"

const FORMAT = { tabSize: 2, insertSpaces: true, eol: "\n" }

/** @type {Harness} */
export const harness = {
    name: "opencode",
    configDir: "~/.config/opencode",
    configCandidates: ["opencode.jsonc", "opencode.json"],
    skillsDir: "skills",
    schema: [
        { path: "enabled", default: false },
        { path: "reasoning_effort", default: null }
    ],

    apply(existing, ctx, config) {
        let text = existing.trim() ? existing : "{}\n"
        const effort = typeof config.reasoning_effort === "string" && config.reasoning_effort ? config.reasoning_effort : null
        text = setProviderLui({
            text,
            modelName: ctx.modelName,
            baseURL: ctx.baseURL,
            ctxSize: ctx.ctxSize,
            maxOutputTokens: ctx.maxOutputTokens,
            reasoningEffort: effort
        })
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

    async sshPreflight(remote) {
        try {
            if (await remote.which("opencode")) return { ok: true }
            // opencode's installer drops the binary here and edits the
            // shell profile to add it, which a non-interactive session
            // never reads — so check the install path directly.
            const installed = remote.platform === "win32" ? "~/.opencode/bin/opencode.exe" : "~/.opencode/bin/opencode"
            if (await remote.exists(installed)) return { ok: true }
            return { ok: false, error: `opencode not found on ${remote.spec}. Install it there first.` }
        } catch (e) {
            return {
                ok: false,
                error: `opencode preflight on ${remote.spec} failed: ${/** @type {Error} */ (e).message}`
            }
        }
    }
}

/** @param {{ text: string, modelName: string, baseURL: string, ctxSize: number, maxOutputTokens: number, reasoningEffort: string | null }} args @returns {string} */
function setProviderLui({ text, modelName, baseURL, ctxSize, maxOutputTokens, reasoningEffort }) {
    /** @type {{ name: string, supportsToolCalls: boolean, limit: { context: number, input: number, output: number }, options?: { reasoningEffort: string } }} */
    const modelValue = {
        name: modelName,
        supportsToolCalls: true,
        limit: { context: ctxSize, input: ctxSize, output: maxOutputTokens }
    }
    if (reasoningEffort) modelValue.options = { reasoningEffort }
    const luiValue = {
        name: "lui",
        npm: "@ai-sdk/openai-compatible",
        options: {
            baseURL,
            toolParser: [{ type: "raw-function-call" }, { type: "json" }]
        },
        models: {
            [modelName]: modelValue
        }
    }
    const edits = modify(text, ["provider", "lui"], luiValue, { formattingOptions: FORMAT })
    return applyEdits(text, edits)
}

/** @param {string} text @param {number} webPort @param {boolean} websearch @returns {string} */
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
        // jsonc-parser's modify(..., undefined) throws on a path that
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
