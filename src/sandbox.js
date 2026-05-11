// `lui sandbox HARNESS [args...]`: launch the chosen harness wrapped in
// nono (https://nono.sh). Everything after HARNESS reaches the harness
// verbatim — so `alias opencode='lui sandbox opencode'` works and
// `opencode --help` ends up at the right place.
//
// All knobs live under `[sandbox]` in lui.toml (allow_cwd,
// block_net, allow_gpu, rollback, silent, profile, bin, dev_tools, plus
// the repeatable string arrays allow/read/write/allow_domain/extra).
// Defaults chosen to make "lui sandbox HARNESS just works" the common
// case for agentic harnesses: cwd granted R+W, dev toolchains auto-
// allowed, GPU off, network on, no auto-rollback.

import { spawn, spawnSync } from "node:child_process"
import fs from "node:fs"
import path from "node:path"
import os from "node:os"
import process from "node:process"

import { STYLE } from "./engine.js"

const PROFILE_OPT_OUT = "none"
const FALLBACK_PROFILE = "default"

// Segment styles. The HARNESS placeholder is overlaid on top of these
// per-token by the printer, so the magenta only lands on the actual
// placeholder slots.
const SEG_VERB = STYLE.SEGMENT_POLICY
const SEG_PROFILE = STYLE.SEGMENT_BINDING
const SEG_POLICY = STYLE.SEGMENT_POLICY
const SEG_DEFAULTS = STYLE.SEGMENT_DEFAULTS
const SEG_USER = STYLE.SEGMENT_USER
const SEG_SEP = STYLE.SEGMENT_POLICY

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
        if (e.code === "ENOENT") {
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

function resolveProfile(cfg, bin, harness) {
    const explicit = (cfg.profile ?? "").trim()
    if (explicit) {
        if (explicit.toLowerCase() === PROFILE_OPT_OUT) return null
        return explicit
    }
    return nonoProfileExists(bin, harness) ? harness : FALLBACK_PROFILE
}

function nonoProfileExists(bin, name) {
    try {
        const r = spawnSync(bin, ["profile", "show", name, "--silent"], { stdio: "ignore" })
        return r.status === 0
    } catch {
        return false
    }
}

// Static preview of what `lui sandbox HARNESS` would run, with the
// literal string "HARNESS" standing in everywhere the actual harness
// name would land — both as the profile (when auto-detect is in play)
// and as the binary after `--`. Used by `lui cmd` so users can audit
// the sandbox invocation without launching anything. No nono probe.
//
// Returns styled segments (matching the engine commandline shape) so
// the renderer can color-code each role the same way it does the
// llama-server argv.
export function previewSandboxArgs(lui) {
    const cfg = lui.config.sandbox || {}
    const bin = cfg.bin || "nono"
    const explicit = (cfg.profile ?? "").trim()
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

// Group nono args by semantic role so the renderer can color them:
//   verb       → "run"                               (policy dim)
//   profile    → "-p PROFILE"                        (cyan; PROFILE may be HARNESS)
//   policy     → -s / --allow . / --allow-cwd / --allow-gpu / --block-net / --rollback
//   defaults   → --allow $CARGO_HOME … (auto-detected dev tools)
//   user       → --allow / --read / --write / --allow-domain / extra (user-configured)
function buildNonoSegments(cfg, profile) {
    const segments = []
    segments.push({ name: "verb", style: SEG_VERB, args: ["run"] })

    if (profile) segments.push({ name: "profile", style: SEG_PROFILE, args: ["-p", profile] })

    const policy = []
    if (cfg.silent) policy.push("-s")
    if (cfg.allow_cwd !== false) {
        // `--allow .` is R+W on the cwd; `--allow-cwd` skips nono's
        // first-run prompt for the same path. Pair them so an agent can
        // both read and write the project tree without friction.
        policy.push("--allow", ".", "--allow-cwd")
    }
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

// Canonical toolchain directories an agent typically needs to invoke
// cargo / go / npm / bun / pnpm / pip / etc. Granted as `--allow`
// (R+W) since these tools write caches on first use; read-only would
// silently break fetches. Skip non-existing paths so the preview stays
// tight.
function existingDevToolDirs() {
    const home = os.homedir()
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
