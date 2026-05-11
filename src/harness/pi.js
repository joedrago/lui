// pi harness: writes ~/.pi/agent/models.json. pi's file is plain JSON
// but jsonc-parser handles it identically.

import { modify, applyEdits, parseTree, findNodeAtLocation } from "jsonc-parser"

const FORMAT = { tabSize: 2, insertSpaces: true, eol: "\n" }

export const harness = {
    name: "pi",
    defaultEnabled: false,
    configDir: "~/.pi/agent",
    configCandidates: ["models.json"],

    apply(existing, lui) {
        let text = existing.trim() ? existing : "{}\n"
        const modelName = deriveModelName(lui.activeModel?.name)
        const baseURL = lui.engineBaseURL ?? `http://127.0.0.1:${lui.config.global.engine_port}/v1`
        const ctxSize = inferContextSize(lui.activeModel?.args || [])

        const luiValue = {
            baseUrl: baseURL,
            api: "openai-completions",
            apiKey: "lui",
            models: [
                {
                    id: modelName,
                    name: modelName,
                    contextWindow: ctxSize,
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

function deriveModelName(activeKey) {
    if (!activeKey) return "lui"
    const tail = activeKey.split("/").pop() || activeKey
    return tail.split(":")[0].replace(/-GGUF$/, "") || "lui"
}

function inferContextSize(args) {
    for (let i = 0; i < args.length; i++) {
        if ((args[i] === "-c" || args[i] === "--ctx-size") && i + 1 < args.length) {
            const n = parseInt(args[i + 1], 10)
            if (Number.isFinite(n) && n > 0) return n
        }
    }
    return 32768
}
