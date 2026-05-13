// `lui setup` — interactive first-run wizard. Asks which harnesses to
// enable, offers each engine's curated `setupDefaults` as ready-to-add
// models, writes ~/.config/lui.toml, and prints a hand-tuned next-steps
// table. Detection of installed binaries is purely advisory: presence
// flips the preselect default, but the user can always toggle anything
// on or off. The next-steps table at the bottom of this file is
// intentionally hardcoded — treat it as a living document.

/** @import { Lui } from "./lui.js" */
/** @import { Engine } from "./types.js" */

import fs from "node:fs"
import path from "node:path"
import process from "node:process"

import { styled } from "./ansi.js"
import { STYLE } from "./theme.js"
import { harnesses } from "./harness.js"
import { engines } from "./engine.js"
import { multiselect, PromptAborted } from "./prompt.js"

/** @param {Lui} lui */
export async function runSetup(lui) {
    if (!process.stdin.isTTY || !process.stdout.isTTY) {
        process.stdout.write(
            "lui setup is interactive and needs a terminal.\n" +
                "Run it from a TTY, or configure directly with `lui set ...`.\n"
        )
        return
    }

    process.stdout.write("\n" + styled("lui setup", STYLE.BRAND) + " — let's get you wired up.\n\n")

    try {
        const enabledHarnesses = await pickHarnesses(lui)
        const addedModels = await pickModels(lui)
        writeChoices(lui, enabledHarnesses, addedModels)
        printInformation(lui, enabledHarnesses)
    } catch (e) {
        if (e instanceof PromptAborted) {
            process.stdout.write("\n" + styled("Cancelled.", { dim: true }) + " No changes written.\n")
            return
        }
        throw e
    }
}

/** @param {Lui} lui @returns {Promise<Set<string>>} */
async function pickHarnesses(lui) {
    const items = harnesses.map((h) => {
        const detected = commandOnPath(h.name)
        const currently = lui.config.harness?.[h.name]?.enabled
        const selected = currently != null ? !!currently : detected
        const hint = detected ? "(detected)" : "(not found on PATH)"
        return { label: h.name, value: h.name, hint, selected }
    })
    const picked = await multiselect("Harnesses to enable:", items)
    return new Set(picked)
}

/** @param {Lui} lui @returns {Promise<{ name: string, engine: string, args: string[] }[]>} */
async function pickModels(lui) {
    const items = []
    for (const [engineName, engine] of Object.entries(engines)) {
        if (!engine.setupDefaults?.length) continue
        // Engine name and binary name don't always match: mlx_lm's
        // binary is "mlx_lm.server", remote has none at all. Resolve
        // through schema/config so detection actually finds them.
        const binary = engineBinaryName(lui, engine, engineName)
        const detected = binary ? commandOnPath(binary) : false
        for (const def of engine.setupDefaults) {
            if (lui.config.model?.[def.name]) continue
            const detectedSuffix = detected ? "(detected)" : "(not found on PATH)"
            // sizeGiB is an approximate on-disk size for the model
            // weights — handy when users are picking which to download
            // on a constrained connection or drive.
            const sizeSuffix = typeof def.sizeGiB === "number" ? `, ~${def.sizeGiB} GB on disk ` : ""
            items.push({
                label: `${def.name}`,
                value: { name: def.name, engine: engineName, args: def.args },
                hint: `uses ${engineName}${sizeSuffix} ${detectedSuffix}`,
                selected: detected
            })
        }
    }
    if (items.length === 0) {
        process.stdout.write("\n" + styled("Models:", STYLE.LABEL) + " all curated picks already registered, skipping.\n")
        return []
    }
    process.stdout.write("\n")
    return multiselect("Curated models to add:", items)
}

/** @param {Lui} lui @param {Set<string>} enabledHarnesses @param {{ name: string, engine: string, args: string[] }[]} addedModels */
function writeChoices(lui, enabledHarnesses, addedModels) {
    for (const h of harnesses) {
        const sub = (lui.config.harness[h.name] ??= {})
        sub.enabled = enabledHarnesses.has(h.name)
    }
    for (const m of addedModels) {
        lui.config.model[m.name] = { engine: m.engine, args: [...m.args] }
    }
    lui.config.save()
}

// Hand-written. Tune labels, ordering, and which lines appear as the
// project grows — there's intentionally no abstraction here. Binary
// detection is purely informational upstream (it flips the multi-select
// preselect default); nothing in this table is filtered away based on
// what's installed. Everything is just helpful info.
/** @param {Lui} _lui @param {Set<string>} enabledHarnesses */
function printInformation(_lui, enabledHarnesses) {
    const w = process.stdout.write.bind(process.stdout)
    w("\n" + styled("Saved", STYLE.READY) + " " + dim(configPathForDisplay()) + "\n\n")

    w(styled("Information", STYLE.BRAND) + "\n\n")

    // One column width used for every block so every `:` lines up
    // top-to-bottom. Bump if any new row's label outgrows it.
    const W = 19

    if (enabledHarnesses.has("opencode")) {
        w("  " + styled("opencode", STYLE.HARNESS_NAME) + "\n")
        w(row("install", cmd("npm install -g opencode-ai"), W))
        w(row("docs", url("https://opencode.ai"), W))
        w(row("enable sandbox", dim("alias opencode='lui sandbox opencode'"), W))
        w("\n")
    }

    if (enabledHarnesses.has("pi")) {
        w("  " + styled("pi", STYLE.HARNESS_NAME) + "\n")
        w(row("install", cmd("npm install -g @earendil-works/pi-coding-agent"), W))
        w(row("docs", url("https://pi.dev"), W))
        w(row("enable sandbox", dim("alias pi='lui sandbox pi'"), W))
        w("\n")
    }

    w("  " + styled("llama-server", STYLE.ENGINE_NAME) + "\n")
    w(row("install (Win/Linux)", url("https://github.com/ggml-org/llama.cpp#quick-start"), W))
    w(row("install (macOS)", cmd("brew install llama.cpp"), W))
    w("\n")

    w("  " + styled("mlx_lm", STYLE.ENGINE_NAME) + "\n")
    w(row("install (macOS)", cmd("pip install mlx-lm"), W))
    w(row("docs", url("https://github.com/ml-explore/mlx-lm"), W))
    w("\n")

    w("  " + styled("lui", STYLE.BRAND) + "\n")
    w(row("inspect config", cmd("lui ls"), W))
    w(row("run a model", cmd("lui run"), W))
    w(row("disable websearch", cmd("lui set websearch false"), W))
    w(row("custom debug log", cmd("lui set debug_log /tmp/lui.log"), W))
    w(row("change engine port", cmd("lui set engine_port 9000"), W))
    w(row("change web port", cmd("lui set web_port 9001"), W))
    w(row("toggle a harness", cmd("lui set harness.opencode.enabled false"), W))
    w(row("llama-server binary", cmd("lui set engine.llama-server.binary /opt/llama/llama-server"), W))
    w(row("share publicly", cmd("lui set public true"), W))
    w(row("add sandbox r+w dir", cmd("lui set sandbox.allow ~/projects"), W))
    w(row("block sandbox net", cmd("lui set sandbox.block_net true"), W))
    w(row("sandbox a tool", cmd("lui sandbox opencode"), W))
    w("\n")

    w(
        styled("Tip:", STYLE.ACTIVE) +
            dim(" after `lui run`, wait for the model to report ") +
            styled("Ready", STYLE.READY) +
            dim(" before launching your harness.") +
            "\n\n"
    )
}

/** @param {string} label @param {string} bodyStyled @param {number} width @returns {string} */
function row(label, bodyStyled, width) {
    return "    " + styled(label.padEnd(width) + " :", STYLE.LABEL) + "   " + bodyStyled + "\n"
}

/** @param {string} s @returns {string} */
function cmd(s) {
    return styled(s, STYLE.CONFIG_KEY)
}
/** @param {string} s @returns {string} */
function url(s) {
    return styled(s, STYLE.URL)
}
/** @param {string} s @returns {string} */
function dim(s) {
    return styled(s, { dim: true })
}

// Resolve an engine's binary name: a user override under
// `engine.<name>.binary` wins, otherwise the schema default, otherwise
// the engine name as a last resort. Returns null when the engine
// doesn't declare a binary (e.g. the `remote` engine).
/** @param {Lui} lui @param {Engine} engine @param {string} engineName @returns {string | null} */
function engineBinaryName(lui, engine, engineName) {
    const override = lui.config.engine?.[engineName]?.binary
    if (override) return override
    const schemaEntry = (engine.schema ?? []).find((s) => s.path === "binary")
    if (schemaEntry?.default) return schemaEntry.default
    return null
}

// True when `name` resolves to an executable on the user's PATH. Pure
// fs walk — no subprocess — so this is safe at config-write time and
// matches what `command -v` would see for a non-shell-builtin.
/** @param {string} name @returns {boolean} */
function commandOnPath(name) {
    const PATH = process.env.PATH || ""
    const isWin = process.platform === "win32"
    const sep = isWin ? ";" : ":"
    const exts = isWin ? (process.env.PATHEXT || ".EXE;.CMD;.BAT;.COM").split(";") : [""]
    for (const dir of PATH.split(sep)) {
        if (!dir) continue
        for (const ext of exts) {
            const p = path.join(dir, name + ext)
            try {
                fs.accessSync(p, fs.constants.X_OK)
                return true
            } catch {
                // try next
            }
        }
    }
    return false
}

/** @returns {string} */
function configPathForDisplay() {
    return path.join("~", ".config", "lui.toml")
}
