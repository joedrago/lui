// TUI loop. Polls http://127.0.0.1:<web_port>/data every 250 ms, paints
// the panels, repaints on terminal resize. Style is encoded inline in
// strings as Private Use Area switch chars; ansi.compilePalette turns
// the payload's palette into SGR sequences, and ansi.paint expands the
// switches at render time.
//
// The renderer takes over the screen with the alternate-screen buffer
// so the terminal's scrollback is preserved. Stdin is put into raw mode
// while the TUI runs so a bare `q` or Ctrl-C quits without echoing.

import http from "node:http"
import process from "node:process"

import {
    compilePalette,
    paint,
    vwidth,
    cursorTo,
    clearLine,
    hideCursor,
    showCursor,
    reset,
    wrapStyled,
    truncateLeft,
    renderBar
} from "./ansi.js"

const POLL_MS = 250
const GUTTER = 2
// Windows Terminal truncates characters written to the last column(s)
// (e.g. the trailing "p" of "/setup" disappears). Treat the usable width
// as 2 columns shy of reported width on every platform so rules, the
// right-aligned setup URL, and header fills all stay inside the safe
// zone. The same constant as the Rust version's saturating_sub(2).
const RIGHT_MARGIN = 2

// Renderer-owned colors for the chrome (panel dividers, bar fill/empty,
// bar inline text). Engines pick their own colors for the body text;
// these only style what the renderer itself emits.
const SGR_RESET = "\x1b[0m"
const TITLE_SGR = "\x1b[38;2;120;100;180m" // MUTED_PURPLE
const BAR_FILLED_SGR = "\x1b[38;2;180;150;255m" // LAVENDER
const BAR_EMPTY_SGR = "\x1b[38;2;120;100;180m" // MUTED_PURPLE
const BAR_TEXT_SGR = "\x1b[38;2;210;150;255m" // COLOR_NUMBER

// Alt-screen + bracketed-paste off + cursor off. The matching exit
// sequence appears in stop().
const ENTER_ALT = "\x1b[?1049h\x1b[?7l" + hideCursor() + cursorTo(1, 1)
const LEAVE_ALT = showCursor() + "\x1b[?7h\x1b[?1049l"

export function startTui(lui) {
    if (!process.stdout.isTTY) {
        // No TTY: paint nothing. Useful for `lui run > log` style tests.
        return { stop() {} }
    }

    let stopped = false
    let lastPayload = null

    function onResize() {
        if (lastPayload) paintScreen(lastPayload)
    }
    process.stdout.on("resize", onResize)

    process.stdout.write(ENTER_ALT)

    // Raw stdin so 'q' / Ctrl-C don't echo and we get them without Enter.
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
            // width to the right edge.
            buf += cursorTo(row, 1) + clearLine()
            buf += " ".repeat(GUTTER) + TITLE_SGR + "── "
            const title = panel.title || ""
            // Title text may carry inline palette switches; paint() expands
            // them and a hard reset at the end means our title-style SGR
            // gets clobbered, so we re-apply it before the trailing dashes.
            buf += paint(title, compiled) + TITLE_SGR + " "
            const titleVw = vwidth(title)
            const dashRoom = Math.max(0, cols - GUTTER - 4 - titleVw)
            buf += "─".repeat(dashRoom) + SGR_RESET
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
                buf += SGR_RESET
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
        process.stdout.write(LEAVE_ALT + SGR_RESET)
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

// Renderer-side bar. Bar.label and bar.text may carry inline palette
// switches and get expanded through `paint`; the filled and empty
// segments are colored uniformly with the renderer's own LAVENDER /
// MUTED_PURPLE so bars match the panel dividers visually.
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
    if (label) out += paint(label, compiled) + SGR_RESET + " "
    out += "["
    out += BAR_FILLED_SGR + "█".repeat(filled)
    out += BAR_EMPTY_SGR + "░".repeat(empty)
    out += SGR_RESET + "]"
    if (text) out += " " + BAR_TEXT_SGR + paint(text, compiled) + SGR_RESET
    return out
}

// renderBar from ansi.js is no longer the path the TUI takes for bars,
// but the import lives there for tests / programmatic callers.
void renderBar

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
