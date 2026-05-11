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

const PROFILE_OPT_OUT = "none"
const FALLBACK_PROFILE = "default"

export async function runSandbox(lui, harnessName, harnessArgs) {
    if (!harnessName) {
        process.stderr.write("lui: sandbox requires HARNESS\n")
        process.exit(2)
    }
    const cfg = lui.config.sandbox || {}
    const bin = cfg.bin || "nono"
    const profile = resolveProfile(cfg, bin, harnessName)
    const nonoArgs = buildNonoArgs(cfg, profile)
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
    const args = buildNonoArgs(cfg, profile)
    args.push("--", "HARNESS")
    return { bin, args }
}

function buildNonoArgs(cfg, profile) {
    const out = ["run"]
    if (profile) out.push("-p", profile)

    if (cfg.silent) out.push("-s")

    if (cfg.allow_cwd !== false) {
        // `--allow .` is R+W on the cwd; `--allow-cwd` skips nono's
        // first-run prompt for the same path. Pair them so an agent can
        // both read and write the project tree without friction.
        out.push("--allow", ".", "--allow-cwd")
    }

    if (cfg.allow_gpu === true) out.push("--allow-gpu")
    if (cfg.block_net) out.push("--block-net")
    if (cfg.rollback) out.push("--rollback")

    if (cfg.dev_tools !== false) {
        for (const dir of existingDevToolDirs()) out.push("--allow", dir)
    }

    for (const dir of cfg.allow || []) out.push("--allow", dir)
    for (const dir of cfg.read || []) out.push("--read", dir)
    for (const dir of cfg.write || []) out.push("--write", dir)
    for (const dom of cfg.allow_domain || []) out.push("--allow-domain", dom)
    for (const tok of cfg.extra || []) out.push(tok)

    return out
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
