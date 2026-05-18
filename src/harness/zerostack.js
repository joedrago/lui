// zerostack harness: writes <config-dir>/zerostack/config.json.
// Inserts a `custom_providers.lui` entry pointing zerostack at
// llama-server, then sets `provider`, `model`, and `context_window`.
//
// Zerostack uses the `dirs` crate for config resolution: on macOS
// that's ~/Library/Application Support/zerostack, on Linux it's
// ~/.config/zerostack (or $XDG_CONFIG_HOME/zerostack).

/** @import { Harness } from "../types.js" */

import process from "node:process"

import { modify, applyEdits, parseTree, findNodeAtLocation } from "jsonc-parser"

const FORMAT = { tabSize: 2, insertSpaces: true, eol: "\n" }

/** @type {Harness} */
export const harness = {
    name: "zerostack",
    configDir: process.platform === "darwin" ? "~/Library/Application Support/zerostack" : "~/.config/zerostack",
    configCandidates: ["config.json"],
    schema: [{ path: "enabled", default: false }],

    apply(existing, ctx) {
        let text = existing.trim() ? existing : "{}\n"
        text = setCustomProviderLui({ text, baseURL: ctx.baseURL })
        text = setProvider(text, "lui")
        text = setModel(text, ctx.modelName)
        text = setContextWindow(text, ctx.ctxSize)
        return text
    },

    needsBackup(existing) {
        if (!existing.trim()) return false
        const tree = parseTree(existing, [], { allowTrailingComma: true })
        if (!tree) return true
        const lui = findNodeAtLocation(tree, ["custom_providers", "lui"])
        return !lui
    },

    async sshPreflight(target, sshRun) {
        const probe = "command -v zerostack || bash -lc 'command -v zerostack'"
        try {
            const out = await sshRun(target, probe)
            if (out.trim()) return { ok: true }
            return { ok: false, error: `zerostack not found on ${target.user}@${target.host}. Install it there first.` }
        } catch (e) {
            return {
                ok: false,
                error: `zerostack preflight on ${target.user}@${target.host} failed: ${/** @type {Error} */ (e).message}`
            }
        }
    }
}

/** @param {{ text: string, baseURL: string }} args @returns {string} */
function setCustomProviderLui({ text, baseURL }) {
    const luiValue = {
        provider_type: "openai",
        base_url: baseURL
    }
    const edits = modify(text, ["custom_providers", "lui"], luiValue, { formattingOptions: FORMAT })
    return applyEdits(text, edits)
}

/** @param {string} text @param {string} provider @returns {string} */
function setProvider(text, provider) {
    const edits = modify(text, ["provider"], provider, { formattingOptions: FORMAT })
    return applyEdits(text, edits)
}

/** @param {string} text @param {string} model @returns {string} */
function setModel(text, model) {
    const edits = modify(text, ["model"], model, { formattingOptions: FORMAT })
    return applyEdits(text, edits)
}

/** @param {string} text @param {number} ctxSize @returns {string} */
function setContextWindow(text, ctxSize) {
    const edits = modify(text, ["context_window"], ctxSize, { formattingOptions: FORMAT })
    return applyEdits(text, edits)
}
