// TUI loop. Polls /data every 250 ms in the alt-screen buffer; `q` or
// Ctrl-C quits via raw stdin.

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

const POLL_MS = 250
const GUTTER = 2
// Windows Terminal truncates the last column(s); shave 2 off for safety.
const RIGHT_MARGIN = 2

const ENTER_ALT = enterAltScreen() + disableLineWrap() + hideCursor() + cursorTo(1, 1)
const LEAVE_ALT = showCursor() + enableLineWrap() + leaveAltScreen()

export function startTui(lui) {
    if (!process.stdout.isTTY) return { stop() {} }

    let stopped = false
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

    function onStdin(chunk) {
        for (const byte of chunk) {
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
        try {
            const payload = await fetchData(lui.config.global.web_port)
            if (payload) {
                lastPayload = payload
                paintScreen(payload)
            }
        } catch {
            // Server not up yet, or transient — keep polling.
        }
        setTimeout(tick, POLL_MS)
    }

    function paintScreen(payload) {
        const compiled = compilePalette(payload.palette || [])
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
    return { stop, repaint: () => (lastPayload ? paintScreen(lastPayload) : null) }
}

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

function paintBar(bar, width, compiled) {
    const frac = bar.max ? Math.max(0, Math.min(1, bar.value / bar.max)) : Math.max(0, Math.min(1, bar.value || 0))
    const label = bar.label ?? ""
    const text = bar.text ?? ""
    const labelW = vwidth(label)
    const textW = vwidth(text)

    const labelGap = labelW ? 1 : 0
    const textGap = textW ? 1 : 0
    const fixed = labelW + labelGap + 2 /* [] */ + textGap + textW
    const inner = Math.max(1, width - fixed)
    const filled = Math.round(frac * inner)
    const empty = Math.max(0, inner - filled)

    let out = ""
    if (label) out += paint(label, compiled) + reset() + " "
    out += "["
    out += styled("█".repeat(filled), STYLE.BAR_FILL)
    out += styled("░".repeat(empty), STYLE.BAR_EMPTY)
    out += "]"
    if (text) out += " " + styled(paint(text, compiled), STYLE.VALUE)
    return out
}

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
