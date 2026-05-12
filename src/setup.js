// `lui setup` — interactive first-run wizard. Asks which harnesses to
// enable, offers each engine's curated `setupDefaults` as ready-to-add
// models, writes ~/.config/lui.toml, and prints a hand-tuned next-steps
// table. Detection of installed binaries is purely advisory: presence
// flips the preselect default, but the user can always toggle anything
// on or off. The next-steps table at the bottom of this file is
// intentionally hardcoded — treat it as a living document.

import fs from "node:fs"
import path from "node:path"
import process from "node:process"

import { styled } from "./ansi.js"
import { STYLE } from "./theme.js"
import { harnesses } from "./harness.js"
import { engines } from "./engine.js"
import { multiselect, PromptAborted } from "./prompt.js"

export async function runSetup(lui) {
    if (!process.stdin.isTTY || !process.stdout.isTTY) {
        process.stdout.write(
            "lui setup is interactive and needs a terminal.\n" +
                "Run it from a TTY, or configure directly with `lui config set …`.\n"
        )
        return
    }

    process.stdout.write("\n" + styled("lui setup", STYLE.BRAND) + " — let's get you wired up.\n\n")

    try {
        const enabledHarnesses = await pickHarnesses(lui)
        const addedModels = await pickModels(lui)
        writeChoices(lui, enabledHarnesses, addedModels)
        printNextSteps(lui, enabledHarnesses, addedModels)
    } catch (e) {
        if (e instanceof PromptAborted) {
            process.stdout.write("\n" + styled("Cancelled.", { dim: true }) + " No changes written.\n")
            return
        }
        throw e
    }
}

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

async function pickModels(lui) {
    const items = []
    for (const [engineName, engine] of Object.entries(engines)) {
        if (!engine.setupDefaults?.length) continue
        const detected = commandOnPath(engineName)
        for (const def of engine.setupDefaults) {
            if (lui.config.model?.[def.name]) continue
            items.push({
                label: `${def.name}`,
                value: { name: def.name, engine: engineName, args: def.args },
                hint: `uses ${engineName} ${detected ? "(detected)" : "(not found on PATH)"}`,
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

function writeChoices(lui, enabledHarnesses, addedModels) {
    for (const h of harnesses) {
        const sub = (lui.config.harness[h.name] ??= {})
        sub.enabled = enabledHarnesses.has(h.name)
    }
    for (const m of addedModels) {
        lui.config.model[m.name] = { engine: m.engine, args: [...m.args] }
    }
    if (!lui.config.global.active_model && addedModels.length > 0) {
        lui.config.setActiveModel(addedModels[0].name)
    }
    lui.config.save()
}

// Hand-written. Adjust as the lineup of harnesses / engines / tips
// changes — there's intentionally no abstraction here.
function printNextSteps(lui, enabledHarnesses, addedModels) {
    const w = process.stdout.write.bind(process.stdout)
    w("\n" + styled("Saved", STYLE.READY) + " " + dimPath(configPathForDisplay()) + "\n\n")

    w(styled("Next steps", STYLE.BRAND) + "\n\n")

    if (enabledHarnesses.has("opencode")) {
        w("  " + styled("opencode", STYLE.HARNESS_NAME) + "\n")
        if (!commandOnPath("opencode")) {
            w("    " + label("install") + cmd("npm install -g opencode-ai") + "\n")
        }
        w("    " + label("docs   ") + url("https://opencode.ai") + "\n")
        w("    " + label("alias  ") + cmd("alias opencode='lui sandbox opencode'") + "\n\n")
    }

    if (enabledHarnesses.has("pi")) {
        w("  " + styled("pi", STYLE.HARNESS_NAME) + "\n")
        if (!commandOnPath("pi")) {
            w("    " + label("install") + cmd("npm install -g @earendil-works/pi-coding-agent") + "\n")
        }
        w("    " + label("docs   ") + url("https://github.com/block/pi") + "\n")
        w("    " + label("alias  ") + cmd("alias pi='lui sandbox pi'") + "\n\n")
    }

    const needsLlamaServer = addedModels.some((m) => m.engine === "llama-server") && !commandOnPath("llama-server")
    if (needsLlamaServer) {
        w("  " + styled("llama-server", STYLE.ENGINE_NAME) + "\n")
        w("    " + label("install") + url("https://github.com/ggml-org/llama.cpp#quick-start") + "\n")
        w("    " + dimText("macOS: brew install llama.cpp · Windows/Linux: grab a release binary and put it on PATH") + "\n\n")
    }

    w("  " + styled("lui", STYLE.BRAND) + "\n")
    if (lui.config.global.active_model) {
        w("    " + label("start  ") + cmd("lui run") + dimText("   # runs " + lui.config.global.active_model) + "\n")
    } else {
        w("    " + label("start  ") + cmd("lui run NAME") + "\n")
    }
    w("    " + label("inspect") + cmd("lui config") + "\n")
    w("    " + label("share  ") + cmd("lui config set public true") + dimText("   # expose this lui to other machines") + "\n")
    w("\n")
}

function label(s) {
    return styled(s + " ", STYLE.LABEL)
}
function cmd(s) {
    return styled(s, STYLE.CONFIG_KEY)
}
function url(s) {
    return styled(s, STYLE.URL)
}
function dimText(s) {
    return styled(s, { dim: true })
}
function dimPath(s) {
    return styled(s, { dim: true })
}

// True when `name` resolves to an executable on the user's PATH. Pure
// fs walk — no subprocess — so this is safe at config-write time and
// matches what `command -v` would see for a non-shell-builtin.
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

function configPathForDisplay() {
    return path.join("~", ".config", "lui.toml")
}
