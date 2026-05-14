// Engine-agnostic download progress tracker. Inference servers tend
// to print download progress only when stdout is a TTY (llama-server),
// so we sidestep that by polling the destination file's size and
// learning the total via a HEAD on the source URL. Engines parse their
// own log lines to discover (url, path) pairs and hand them off here;
// rendering stays with the engine.

import fs from "node:fs"

const POLL_MS = 750

/**
 * @typedef {Object} DownloadEntry
 * @property {string} url
 * @property {string} path
 * @property {number} total       Bytes; 0 until HEAD resolves (or if it fails).
 * @property {number} downloaded  Bytes currently on disk.
 */

/**
 * @typedef {Object} DownloadTracker
 * @property {(spec: { url: string, path: string, label?: string }) => void} add
 * @property {() => [string, DownloadEntry][]} entries
 * @property {() => void} stop
 */

/** @returns {DownloadTracker} */
export function createDownloadTracker() {
    /** @type {Map<string, DownloadEntry & { seenOnce: boolean }>} */
    const map = new Map()
    /** @type {NodeJS.Timeout | null} */
    let timer = null

    /** @param {{ url: string, path: string, label?: string }} spec */
    function add(spec) {
        const key = spec.label || basename(spec.url)
        if (map.has(key)) return
        const entry = { url: spec.url, path: spec.path, total: 0, downloaded: 0, seenOnce: false }
        map.set(key, entry)
        probeTotal(entry)
        if (!timer) timer = setInterval(tick, POLL_MS)
    }

    /** @param {DownloadEntry & { seenOnce: boolean }} entry */
    function probeTotal(entry) {
        // Fire-and-forget; redirects (HF → CDN) are followed by default.
        fetch(entry.url, { method: "HEAD", redirect: "follow" })
            .then((r) => {
                const cl = parseInt(r.headers.get("content-length") ?? "", 10)
                if (Number.isFinite(cl) && cl > 0) entry.total = cl
            })
            .catch(() => {})
    }

    async function tick() {
        for (const [key, entry] of map) {
            try {
                const st = await fs.promises.stat(entry.path)
                entry.downloaded = st.size
                entry.seenOnce = true
            } catch {
                // ENOENT before we've ever seen the file: server hasn't
                // opened it yet. ENOENT after a successful stat: the
                // .downloadInProgress was renamed to its final blob path,
                // i.e. the download finished. Either way the entry retires.
                if (entry.seenOnce) map.delete(key)
            }
        }
        if (map.size === 0 && timer) {
            clearInterval(timer)
            timer = null
        }
    }

    function entries() {
        return [...map.entries()]
    }

    function stop() {
        if (timer) clearInterval(timer)
        timer = null
        map.clear()
    }

    return { add, entries, stop }
}

/** @param {string} url @returns {string} */
function basename(url) {
    try {
        const u = new URL(url)
        const parts = u.pathname.split("/").filter(Boolean)
        return parts[parts.length - 1] || url
    } catch {
        return url
    }
}
