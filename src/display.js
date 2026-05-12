// TUI loop. Polls /data every 250 ms in the alt-screen buffer; `q` or
// Ctrl-C quits via raw stdin.

/** @import { ViewBar, ViewLine, BuiltView } from "./types.js" */
/** @import { Lui } from "./lui.js" */

import http from "node:http"
import process from "node:process"

import {
    compilePalette,
    compileEntry,
    paint,
    styled,
    vwidth,
    cursorTo,
    clearLine,
    hideCursor,
    showCursor,
    reset,
    wrapStyled,
    truncateLeft,
    enterAltScreen,
    leaveAltScreen,
    disableLineWrap,
    enableLineWrap
} from "./ansi.js"
import { STYLE } from "./theme.js"
import { View } from "./wire.js"

const POLL_MS = 250
const GUTTER = 2
// Windows Terminal truncates the last column(s); shave 2 off for safety.
const RIGHT_MARGIN = 2

const ENTER_ALT = enterAltScreen() + disableLineWrap() + hideCursor() + cursorTo(1, 1)
const LEAVE_ALT = showCursor() + enableLineWrap() + leaveAltScreen()

/** @param {Lui} lui @returns {{ stop: () => void, repaint?: () => void }} */
export function startTui(lui) {
    if (!process.stdout.isTTY) return { stop() {} }

    let stopped = false
    /** @type {BuiltView | null} */
    let lastPayload = null

    function onResize() {
        if (lastPayload) paintScreen(lastPayload)
    }
    process.stdout.on("resize", onResize)

    process.stdout.write(ENTER_ALT)

    const rawWasOn = process.stdin.isRaw
    try {
        if (process.stdin.isTTY && process.stdin.setRawMode) {
            process.stdin.setRawMode(true)
        }
    } catch {
        // ignore
    }
    process.stdin.resume()
    process.stdin.on("data", onStdin)

    /** @param {Buffer | string} chunk */
    function onStdin(chunk) {
        for (const byte of Buffer.from(chunk)) {
            if (byte === 0x71 || byte === 0x51 /* q or Q */) {
                lui.quitReason = "user pressed q"
                lui.shutdown(0)
                return
            }
            if (byte === 0x03 /* Ctrl-C */) {
                lui.quitReason = "user pressed Ctrl-C"
                lui.shutdown(0)
                return
            }
        }
    }

    async function tick() {
        if (stopped) return
        // /data carries the server-side panels (warnings + engine).
        // The lui-panel is local to this machine — bookmarklet URL, etc.
        // — so it gets prepended here, not over the wire.
        /** @type {import("./types.js").ViewPanel[]} */
        let serverPanels = []
        try {
            const payload = await fetchData(lui.config.global.web_port)
            if (payload?.panels) serverPanels = payload.panels
        } catch {
            // Server not up yet, or transient — render local panel alone.
        }
        const v = View()
        lui.appendLuiPanel(v)
        for (const panel of serverPanels) v.adoptPanel(panel)
        const built = v.build()
        lastPayload = built
        paintScreen(built)
        setTimeout(tick, POLL_MS)
    }

    /** @param {BuiltView} payload */
    function paintScreen(payload) {
        const rows = process.stdout.rows || 24
        const rawCols = process.stdout.columns || 80
        const cols = Math.max(1, rawCols - RIGHT_MARGIN)

        let buf = cursorTo(1, 1)
        let row = 1

        const lastPanelIdx = payload.panels.length - 1
        for (let pi = 0; pi < payload.panels.length; pi++) {
            if (row > rows) break
            const panel = payload.panels[pi]
            const isLast = pi === lastPanelIdx
            const compiled = compilePalette(panel.palette || [])

            // Title divider: "  ── TITLE ─────…─" in muted purple, full
            // width to the right edge. paint() ends with a hard reset
            // so we have to re-establish the title style on each side
            // of the inline title.
            const titleSgr = compileEntry(STYLE.LABEL)
            buf += cursorTo(row, 1) + clearLine()
            buf += " ".repeat(GUTTER) + titleSgr + "── "
            const title = panel.title || ""
            buf += paint(title, compiled) + titleSgr + " "
            const titleVw = vwidth(title)
            const dashRoom = Math.max(0, cols - GUTTER - 4 - titleVw)
            buf += "─".repeat(dashRoom) + reset()
            row += 1

            for (const line of panel.lines || []) {
                if (row > rows) break
                const painted = paintLine(line, cols, compiled)
                for (const out of painted) {
                    if (row > rows) break
                    buf += cursorTo(row, 1) + clearLine() + out
                    row += 1
                }
            }

            for (const bar of panel.bars || []) {
                if (row > rows) break
                const indent = bar.indent || 0
                const startCol = GUTTER + indent
                buf += cursorTo(row, 1) + clearLine() + " ".repeat(startCol)
                buf += paintBar(bar, Math.max(1, cols - startCol), compiled)
                buf += reset()
                row += 1
            }

            if (!isLast && row <= rows) {
                buf += cursorTo(row, 1) + clearLine()
                row += 1
            }
        }

        while (row <= rows) {
            buf += cursorTo(row, 1) + clearLine()
            row += 1
        }

        process.stdout.write(buf)
    }

    function stop() {
        if (stopped) return
        stopped = true
        try {
            process.stdin.off("data", onStdin)
            if (process.stdin.isTTY && process.stdin.setRawMode) {
                process.stdin.setRawMode(rawWasOn === true)
            }
            process.stdin.pause()
        } catch {
            // ignore
        }
        process.stdout.off?.("resize", onResize)
        process.stdout.write(LEAVE_ALT + reset())
    }

    setTimeout(tick, 0)
    return { stop, repaint: () => (lastPayload ? paintScreen(lastPayload) : undefined) }
}

/** @param {ViewLine} line @param {number} cols @param {string[]} compiled @returns {string[]} */
function paintLine(line, cols, compiled) {
    const text = line.text || ""
    const align = line.align || "left"
    const indent = line.indent || 0
    const startCol = GUTTER + indent
    const available = Math.max(1, cols - startCol)

    if (align === "right") {
        const w = vwidth(text)
        if (w <= cols) {
            const pad = cols - w
            return [" ".repeat(pad) + paint(text, compiled) + reset()]
        }
        return [paint(truncateLeft(text, cols), compiled) + reset()]
    }

    if (vwidth(text) <= available) {
        return [" ".repeat(startCol) + paint(text, compiled) + reset()]
    }
    const wrapped = wrapStyled(text, available)
    return wrapped.map((r) => " ".repeat(startCol) + paint(r, compiled) + reset())
}

// Bars are right-justified into a fixed slot at the trailing half of
// the post-margin width, so every bar in the UI lines up regardless
// of how long its label is. The label fills the left half (truncated
// with "…" if necessary) and the bar+text combo occupies the right.
// A minimum bar width keeps narrow terminals from rendering a stub.
const BAR_RIGHT_FRACTION = 0.5
const BAR_MIN_RIGHT_WIDTH = 12

/** @param {ViewBar} bar @param {number} width @param {string[]} compiled @returns {string} */
function paintBar(bar, width, compiled) {
    const frac = bar.max ? Math.max(0, Math.min(1, bar.value / bar.max)) : Math.max(0, Math.min(1, bar.value || 0))
    const label = bar.label ?? ""
    const text = bar.text ?? ""
    const labelW = vwidth(label)
    const textW = vwidth(text)

    // Right slot holds `[bar] text` and is the same fixed width on
    // every row. Clamp so we don't disappear on tiny windows or
    // overflow on huge ones.
    const rightW = Math.max(BAR_MIN_RIGHT_WIDTH, Math.min(width, Math.floor(width * BAR_RIGHT_FRACTION)))
    const leftW = Math.max(0, width - rightW)

    const textGap = textW ? 1 : 0
    const inner = Math.max(1, rightW - 2 /* [] */ - textGap - textW)
    const filled = Math.round(frac * inner)
    const empty = Math.max(0, inner - filled)

    let left
    if (!label) {
        left = " ".repeat(leftW)
    } else if (labelW <= leftW - 1) {
        // Label fits with a one-column gutter before the bar slot.
        left = paint(label, compiled) + reset() + " ".repeat(leftW - labelW)
    } else if (labelW <= leftW) {
        left = paint(label, compiled) + reset()
    } else {
        // Label longer than its half — trim and ellipsize. vwidth("…") = 1.
        const cut = Math.max(0, leftW - 1)
        left = paint(label.slice(0, cut) + "…", compiled) + reset()
    }

    let right = "["
    right += styled("█".repeat(filled), STYLE.BAR_FILL)
    right += styled("░".repeat(empty), STYLE.BAR_EMPTY)
    right += "]"
    if (text) right += " " + styled(paint(text, compiled), STYLE.VALUE)

    return left + right
}

/** @param {number} port @returns {Promise<BuiltView>} */
function fetchData(port) {
    return new Promise((resolve, reject) => {
        const req = http.get({ host: "127.0.0.1", port, path: "/data", timeout: 1500 }, (res) => {
            let body = ""
            res.setEncoding("utf8")
            res.on("data", (c) => (body += c))
            res.on("end", () => {
                try {
                    resolve(JSON.parse(body))
                } catch (e) {
                    reject(e)
                }
            })
        })
        req.on("error", reject)
        req.on("timeout", () => req.destroy(new Error("timeout")))
    })
}
