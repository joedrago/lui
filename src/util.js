// Low-level catchall. Small, generic helpers with no upward deps.

import os from "node:os"
import path from "node:path"

/** @param {string} p @returns {string} */
export function expandTilde(p) {
    if (p.startsWith("~/")) return path.join(os.homedir(), p.slice(2))
    if (p === "~") return os.homedir()
    return p
}

/** @param {number} ms @returns {string} */
export function formatDurationMilliseconds(ms) {
    const s = Math.floor(ms / 1000)
    if (s < 60) return `${s}s`
    const m = Math.floor(s / 60)
    const rs = s % 60
    if (m < 60) return rs ? `${m}m${rs}s` : `${m}m`
    const h = Math.floor(m / 60)
    const rm = m % 60
    return rm ? `${h}h${rm}m` : `${h}h`
}

/** @param {number} sec @returns {string} */
export function formatDurationSeconds(sec) {
    if (sec < 60) return `<1m`
    const m = Math.floor(sec / 60)
    if (m < 60) return `${m}m`
    const h = Math.floor(m / 60)
    const rm = m % 60
    return rm ? `${h}h${rm}m` : `${h}h`
}

/** @param {number | string} n @returns {string} */
export function formatNumber(n) {
    return Number(n).toLocaleString("en-US")
}

/** @param {number} n @returns {string} */
export function formatBytes(n) {
    if (!Number.isFinite(n) || n <= 0) return "0 B"
    const units = ["B", "KiB", "MiB", "GiB", "TiB"]
    let v = n
    let i = 0
    while (v >= 1024 && i < units.length - 1) {
        v /= 1024
        i += 1
    }
    const digits = v >= 100 ? 0 : v >= 10 ? 1 : 2
    return `${v.toFixed(digits)} ${units[i]}`
}
