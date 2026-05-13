// `lui sandbox HARNESS [args...]`: launch HARNESS under nono
// (https://nono.sh). Knobs live under [sandbox] in lui.toml.

/** @import { Lui } from "./lui.js" */
/** @import { SchemaEntry, Segment } from "./types.js" */

import { spawn, spawnSync } from "node:child_process"
import fs from "node:fs"
import path from "node:path"
import os from "node:os"
import process from "node:process"

import { STYLE } from "./theme.js"

// `lui config` displays these under "Available Settings". `isArray`
// marks list-typed paths where `set` appends and `clear` drops.
/** @returns {SchemaEntry[]} */
export function sandboxSchemaDefaults() {
    return [
        { path: "sandbox.allow_cwd", default: true },
        { path: "sandbox.block_net", default: false },
        { path: "sandbox.allow_gpu", default: false },
        { path: "sandbox.rollback", default: false },
        { path: "sandbox.silent", default: false },
        { path: "sandbox.dev_tools", default: true },
        { path: "sandbox.profile", default: null },
        { path: "sandbox.bin", default: "nono" },
        { path: "sandbox.allow", default: [], isArray: true },
        { path: "sandbox.read", default: [], isArray: true },
        { path: "sandbox.write", default: [], isArray: true },
        { path: "sandbox.allow_domain", default: [], isArray: true },
        { path: "sandbox.extra", default: [], isArray: true }
    ]
}

const PROFILE_OPT_OUT = "none"
const FALLBACK_PROFILE = "default"

const SEG_VERB = STYLE.SEGMENT_POLICY
const SEG_PROFILE = STYLE.SEGMENT_BINDING
const SEG_POLICY = STYLE.SEGMENT_POLICY
const SEG_DEFAULTS = STYLE.SEGMENT_DEFAULTS
const SEG_USER = STYLE.SEGMENT_USER
const SEG_SEP = STYLE.SEGMENT_POLICY

/** @param {Lui} lui @param {string} harnessName @param {string[]} harnessArgs */
export async function runSandbox(lui, harnessName, harnessArgs) {
    if (!harnessName) {
        process.stderr.write("lui: sandbox requires HARNESS\n")
        process.exit(2)
    }
    const cfg = lui.config.sandbox || {}
    const bin = cfg.bin || "nono"
    const profile = resolveProfile(cfg, bin, harnessName)
    const segments = buildNonoSegments(cfg, profile)
    const nonoArgs = segments.flatMap((s) => s.args)
    const argv = [...nonoArgs, "--", harnessName, ...harnessArgs]

    const child = spawn(bin, argv, { stdio: "inherit" })
    child.on("error", (e) => {
        if (/** @type {NodeJS.ErrnoException} */ (e).code === "ENOENT") {
            process.stderr.write(
                `lui: sandbox requires ${JSON.stringify(bin)} on PATH (install from https://nono.sh, ` +
                    `or set [sandbox].bin to its path).\n`
            )
        } else {
            process.stderr.write(`lui: failed to spawn ${bin}: ${e.message}\n`)
        }
        process.exit(2)
    })
    child.on("exit", (code, signal) => {
        if (signal) {
            try {
                process.kill(process.pid, signal)
            } catch {
                process.exit(128)
            }
        } else {
            process.exit(code ?? 0)
        }
    })
}

/** @param {any} cfg @param {string} bin @param {string} harness @returns {string | null} */
function resolveProfile(cfg, bin, harness) {
    const explicit = (cfg.profile ?? "").trim()
    if (explicit) {
        if (explicit.toLowerCase() === PROFILE_OPT_OUT) return null
        return explicit
    }
    return nonoProfileExists(bin, harness) ? harness : FALLBACK_PROFILE
}

/** @param {string} bin @param {string} name @returns {boolean} */
function nonoProfileExists(bin, name) {
    try {
        const r = spawnSync(bin, ["profile", "show", name, "--silent"], { stdio: "ignore" })
        return r.status === 0
    } catch {
        return false
    }
}

// Styled-segment preview of `lui sandbox HARNESS`; "HARNESS" stands in
// wherever the real harness name would land. No nono probe.
/** @param {Lui} lui @returns {{ bin: string, segments: Segment[] }} */
export function previewSandboxArgs(lui) {
    const cfg = lui.config.sandbox || {}
    const bin = cfg.bin || "nono"
    const explicit = (cfg.profile ?? "").trim()
    /** @type {string | null} */
    let profile
    if (explicit) {
        profile = explicit.toLowerCase() === PROFILE_OPT_OUT ? null : explicit
    } else {
        profile = "HARNESS"
    }
    const segments = buildNonoSegments(cfg, profile)
    segments.push({ name: "separator", style: SEG_SEP, args: ["--"] })
    segments.push({ name: "harness", style: SEG_USER, args: ["HARNESS"] })
    return { bin, segments }
}

/** @param {any} cfg @param {string | null} profile @returns {Segment[]} */
function buildNonoSegments(cfg, profile) {
    /** @type {Segment[]} */
    const segments = []
    segments.push({ name: "verb", style: SEG_VERB, args: ["run"] })

    if (profile) segments.push({ name: "profile", style: SEG_PROFILE, args: ["-p", profile] })

    const policy = []
    if (cfg.silent) policy.push("-s")
    // `--allow .` is R+W cwd; `--allow-cwd` skips nono's first-run prompt.
    if (cfg.allow_cwd !== false) policy.push("--allow", ".", "--allow-cwd")
    if (cfg.allow_gpu === true) policy.push("--allow-gpu")
    if (cfg.block_net) policy.push("--block-net")
    if (cfg.rollback) policy.push("--rollback")
    if (policy.length) segments.push({ name: "policy", style: SEG_POLICY, args: policy })

    if (cfg.dev_tools !== false) {
        const devArgs = []
        for (const dir of existingDevToolDirs()) devArgs.push("--allow", dir)
        if (devArgs.length) segments.push({ name: "defaults", style: SEG_DEFAULTS, args: devArgs })
    }

    const userArgs = []
    for (const dir of cfg.allow || []) userArgs.push("--allow", dir)
    for (const dir of cfg.read || []) userArgs.push("--read", dir)
    for (const dir of cfg.write || []) userArgs.push("--write", dir)
    for (const dom of cfg.allow_domain || []) userArgs.push("--allow-domain", dom)
    for (const tok of cfg.extra || []) userArgs.push(tok)
    if (userArgs.length) segments.push({ name: "user", style: SEG_USER, args: userArgs })

    return segments
}

// Toolchain dirs (cargo, go, npm, bun, pnpm, pip, ...) granted R+W so
// first-run fetches can populate caches. Non-existing paths skipped.
/** @returns {string[]} */
function existingDevToolDirs() {
    const home = os.homedir()
    /** @param {string} envKey @param {string} rel @returns {string} */
    const envOrHome = (envKey, rel) => {
        const v = process.env[envKey]
        return v && v.length ? v : path.join(home, rel)
    }
    const candidates = [
        envOrHome("CARGO_HOME", ".cargo"),
        envOrHome("RUSTUP_HOME", ".rustup"),
        envOrHome("GOPATH", "go"),
        "/usr/local/go",
        envOrHome("PYENV_ROOT", ".pyenv"),
        path.join(home, ".local/lib"),
        path.join(home, ".local/share/uv"),
        path.join(home, ".conda"),
        path.join(home, ".nvm"),
        path.join(home, ".fnm"),
        path.join(home, ".npm"),
        path.join(home, ".bun"),
        path.join(home, ".deno"),
        path.join(home, "Library/pnpm"),
        path.join(home, ".local/share/pnpm"),
        "/usr/local/lib/node_modules",
        path.join(home, ".nix-profile"),
        path.join(home, ".nix-defexpr"),
        path.join(home, ".local/state/nix"),
        "/nix/store",
        "/nix/var/nix/profiles"
    ]
    const out = []
    const seen = new Set()
    for (const p of candidates) {
        if (!p) continue
        try {
            if (!fs.statSync(p).isDirectory()) continue
        } catch {
            continue
        }
        let canon
        try {
            canon = fs.realpathSync(p)
        } catch {
            canon = p
        }
        if (seen.has(canon)) continue
        seen.add(canon)
        out.push(canon)
    }
    return out
}
