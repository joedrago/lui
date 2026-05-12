// pi harness: writes ~/.pi/agent/models.json. pi's file is plain JSON
// but jsonc-parser handles it identically.

import { modify, applyEdits, parseTree, findNodeAtLocation } from "jsonc-parser"

const FORMAT = { tabSize: 2, insertSpaces: true, eol: "\n" }

export const harness = {
    name: "pi",
    configDir: "~/.pi/agent",
    configCandidates: ["models.json"],
    skillsDir: "skills",
    schema: [{ path: "enabled", default: false }],

    apply(existing, ctx) {
        let text = existing.trim() ? existing : "{}\n"
        const luiValue = {
            baseUrl: ctx.baseURL,
            api: "openai-completions",
            apiKey: "lui",
            models: [
                {
                    id: ctx.modelName,
                    name: ctx.modelName,
                    contextWindow: ctx.ctxSize,
                    maxTokens: 8192,
                    supportsToolCalls: true
                }
            ]
        }
        const edits = modify(text, ["providers", "lui"], luiValue, { formattingOptions: FORMAT })
        return applyEdits(text, edits)
    },

    needsBackup(existing) {
        if (!existing.trim()) return false
        const tree = parseTree(existing, [], { allowTrailingComma: true })
        if (!tree) return true
        const lui = findNodeAtLocation(tree, ["providers", "lui"])
        return !lui
    }
}
