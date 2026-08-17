// zerostack harness: writes <config-dir>/zerostack/config.json.
// Inserts a `custom_providers.lui` entry pointing zerostack at
// llama-server, then sets `provider`, `model`, and `context_window`.
//
// Zerostack uses the `dirs` crate for config resolution: on macOS
// that's ~/Library/Application Support/zerostack, on Windows it's
// %APPDATA%\zerostack, on Linux it's ~/.config/zerostack (or
// $XDG_CONFIG_HOME/zerostack). The platform is whichever machine is
// being configured — `lui ssh` passes the remote's, not this host's.

/** @import { Harness } from "../types.js" */

import { modify, applyEdits, parseTree, findNodeAtLocation } from "jsonc-parser"

const FORMAT = { tabSize: 2, insertSpaces: true, eol: "\n" }

/** @type {Harness} */
export const harness = {
    name: "zerostack",
    configDir: (platform) => {
        if (platform === "darwin") return "~/Library/Application Support/zerostack"
        if (platform === "win32") return "~/AppData/Roaming/zerostack"
        return "~/.config/zerostack"
    },
    configCandidates: ["config.json"],
    skillsDir: "prompts",
    skillsLayout: "flat",
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

    async sshPreflight(remote) {
        try {
            if (await remote.which("zerostack")) return { ok: true }
            return { ok: false, error: `zerostack not found on ${remote.spec}. Install it there first.` }
        } catch (e) {
            return {
                ok: false,
                error: `zerostack preflight on ${remote.spec} failed: ${/** @type {Error} */ (e).message}`
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
