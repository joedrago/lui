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

    apply(existing, lui) {
        let text = existing.trim() ? existing : "{}\n"
        const modelName = deriveModelName(lui.activeModel?.name)
        const baseURL = lui.engineBaseURL ?? `http://127.0.0.1:${lui.config.global.engine_port}/v1`
        const ctxSize = inferContextSize(lui.activeModel?.args || [])

        text = setProviderLui(text, modelName, baseURL, ctxSize)
        text = setPermissionBash(text, lui.config.global.web_port, lui.config.global.websearch !== false)
        return text
    },

    needsBackup(existing) {
        if (!existing.trim()) return false
        const tree = parseTree(existing, [], { allowTrailingComma: true })
        if (!tree) return true
        const lui = findNodeAtLocation(tree, ["provider", "lui"])
        return !lui
    },

    async preflight(_target) {
        // SSH preflight implemented in ssh.js so this stays purely declarative.
        return { ok: true }
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
        const edits = modify(cur, ["permission", "bash", currentPattern], undefined, { formattingOptions: FORMAT })
        cur = applyEdits(cur, edits)
    }
    return cur
}

function deriveModelName(activeKey) {
    if (!activeKey) return "lui"
    const tail = activeKey.split("/").pop() || activeKey
    const stripped = tail.split(":")[0].replace(/-GGUF$/, "")
    return stripped || "lui"
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
