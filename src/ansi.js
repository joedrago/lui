// ANSI primitives: palette compile + paint, width / wrap / truncate,
// terminal mode controls, and one-shot styled strings.

/** @import { PaletteEntry } from "./types.js" */

const ESC = "\x1b["
const RESET = ESC + "0m"

/** @type {Record<string, number>} */
const NAMED_FG = {
    black: 30,
    red: 31,
    green: 32,
    yellow: 33,
    blue: 34,
    magenta: 35,
    cyan: 36,
    white: 37,
    bright_black: 90,
    bright_red: 91,
    bright_green: 92,
    bright_yellow: 93,
    bright_blue: 94,
    bright_magenta: 95,
    bright_cyan: 96,
    bright_white: 97
}

/** @type {Record<string, number>} */
const NAMED_BG = {
    black: 40,
    red: 41,
    green: 42,
    yellow: 43,
    blue: 44,
    magenta: 45,
    cyan: 46,
    white: 47,
    bright_black: 100,
    bright_red: 101,
    bright_green: 102,
    bright_yellow: 103,
    bright_blue: 104,
    bright_magenta: 105,
    bright_cyan: 106,
    bright_white: 107
}

// Truecolor when COLORTERM=truecolor/24bit, 256 when TERM has "256",
// else 16. Renderer downgrades automatically.
/** @returns {"truecolor" | "256" | "16"} */
function detectDepth() {
    const ct = process.env.COLORTERM || ""
    if (ct === "truecolor" || ct === "24bit") return "truecolor"
    const term = process.env.TERM || ""
    if (term.includes("256")) return "256"
    return "16"
}

let DEPTH = detectDepth()

/** @param {"truecolor" | "256" | "16"} d */
export function setColorDepth(d) {
    DEPTH = d
}

/** @param {number} r @param {number} g @param {number} b @returns {string} */
function rgbToFg(r, g, b) {
    if (DEPTH === "truecolor") return `38;2;${r};${g};${b}`
    if (DEPTH === "256") return `38;5;${rgbTo256(r, g, b)}`
    return String(rgbToBasic16(r, g, b))
}

/** @param {number} r @param {number} g @param {number} b @returns {string} */
function rgbToBg(r, g, b) {
    if (DEPTH === "truecolor") return `48;2;${r};${g};${b}`
    if (DEPTH === "256") return `48;5;${rgbTo256(r, g, b)}`
    return String(rgbToBasic16(r, g, b) + 10)
}

/** @param {number} r @param {number} g @param {number} b @returns {number} */
function rgbTo256(r, g, b) {
    if (r === g && g === b) {
        if (r < 8) return 16
        if (r > 248) return 231
        return 232 + Math.round(((r - 8) / 247) * 24)
    }
    /** @param {number} v */
    const conv = (v) => Math.round((v / 255) * 5)
    return 16 + 36 * conv(r) + 6 * conv(g) + conv(b)
}

/** @param {number} r @param {number} g @param {number} b @returns {number} */
function rgbToBasic16(r, g, b) {
    const bright = r > 170 || g > 170 || b > 170
    let code = 30
    if (r > 100) code += 1
    if (g > 100) code += 2
    if (b > 100) code += 4
    return bright ? code + 60 : code
}

/** @param {string | [number, number, number] | null | undefined} ref @param {boolean} isBg @returns {string | null} */
function colorRefToSgr(ref, isBg) {
    if (ref == null || ref === "default") return null
    if (typeof ref === "string") return String(isBg ? NAMED_BG[ref] : (NAMED_FG[ref] ?? ""))
    if (Array.isArray(ref) && ref.length === 3) {
        const [r, g, b] = ref
        return isBg ? rgbToBg(r, g, b) : rgbToFg(r, g, b)
    }
    return null
}

// SGR prefix per palette entry. Renderer emits a hard reset before each
// switch so the SGR is self-contained.
/** @param {PaletteEntry[]} palette @returns {string[]} */
export function compilePalette(palette) {
    return palette.map((entry) => compileEntry(entry || {}))
}

/** @param {PaletteEntry} entry @returns {string} */
export function compileEntry(entry) {
    const parts = []
    if (entry.bold) parts.push("1")
    if (entry.dim) parts.push("2")
    if (entry.italic) parts.push("3")
    if (entry.underline) parts.push("4")
    const fg = colorRefToSgr(entry.fg, false)
    const bg = colorRefToSgr(entry.bg, true)
    if (fg) parts.push(fg)
    if (bg) parts.push(bg)
    if (parts.length === 0) return ""
    return ESC + parts.join(";") + "m"
}

// One-shot styled string. Returns text untouched for an empty entry.
/** @param {string} text @param {PaletteEntry} [entry] @returns {string} */
export function styled(text, entry) {
    if (!entry) return text
    const sgr = compileEntry(entry)
    if (!sgr) return text
    return sgr + text + RESET
}

// Terminal mode controls. Alt-screen preserves the user's scrollback;
// line-wrap-off lets the renderer own wrapping.
export function enterAltScreen() {
    return ESC + "?1049h"
}
export function leaveAltScreen() {
    return ESC + "?1049l"
}
export function disableLineWrap() {
    return ESC + "?7l"
}
export function enableLineWrap() {
    return ESC + "?7h"
}

// Strip SGR / OSC / single-char escape sequences. Complements
// `stripStyle` which strips lui's own PUA palette switches.
const ANSI_STREAM_RE = (() => {
    // eslint-disable-next-line no-control-regex
    return /\x1b\[[\x20-\x3f]*[\x40-\x7e]|\x1b\][\x20-\x7e]*(?:\x07|\x1b\\)|\x1b[\x20-\x2f]*[\x30-\x7e]/g
})()
/** @param {string} s @returns {string} */
export function stripAnsi(s) {
    return s.replace(ANSI_STREAM_RE, "")
}

/** @param {number} code @returns {boolean} */
function isSwitchChar(code) {
    return code >= 0xe000 && code <= 0xe0ff
}

// Expand inline PUA switch chars into SGR sequences.
/** @param {string} text @param {string[]} compiled @returns {string} */
export function paint(text, compiled) {
    let out = ""
    for (let i = 0; i < text.length; i++) {
        const ch = text.charCodeAt(i)
        if (isSwitchChar(ch)) {
            const idx = ch - 0xe000
            out += RESET
            if (idx < compiled.length) out += compiled[idx]
            continue
        }
        out += text[i]
    }
    return out + RESET
}

// Visible width approximation: count code units, skip PUA switch chars.
// Doesn't handle wide CJK / emoji / combining marks.
/** @param {string} text @returns {number} */
export function vwidth(text) {
    let n = 0
    for (let i = 0; i < text.length; i++) {
        const ch = text.charCodeAt(i)
        if (isSwitchChar(ch)) continue
        n += 1
    }
    return n
}

// Strip PUA palette switches for plain-text output.
/** @param {string} text @returns {string} */
export function stripStyle(text) {
    let out = ""
    for (let i = 0; i < text.length; i++) {
        if (isSwitchChar(text.charCodeAt(i))) continue
        out += text[i]
    }
    return out
}

// Yields { style, text } runs so wrapStyled can re-emit the active
// palette switch at the start of each continuation row.
/** @param {string} text @returns {Generator<{ style: number, text: string }>} */
export function* visibleRuns(text) {
    let style = 0
    let buf = ""
    for (let i = 0; i < text.length; i++) {
        const ch = text.charCodeAt(i)
        if (isSwitchChar(ch)) {
            if (buf) {
                yield { style, text: buf }
                buf = ""
            }
            style = ch - 0xe000
            continue
        }
        buf += text[i]
    }
    if (buf) yield { style, text: buf }
}

// Word-wrap a palette-encoded string at `width` columns. Each returned
// row carries a leading switch char to re-assert the active style.
/** @param {string} text @param {number} width @returns {string[]} */
export function wrapStyled(text, width) {
    if (width <= 0) return [text]
    const rows = []
    let current = ""
    let currentWidth = 0
    let style = 0
    let rowStyle = 0

    /** @type {{ style: number, text: string, space: boolean }[]} */
    const tokens = []
    let active = ""
    /** @type {boolean | null} */
    let activeIsSpace = null

    for (const run of visibleRuns(text)) {
        let i = 0
        while (i < run.text.length) {
            const ch = run.text[i]
            const isSpace = ch === " " || ch === "\t"
            if (activeIsSpace === null) activeIsSpace = isSpace
            if (isSpace !== activeIsSpace || active.length === 0) {
                if (active.length > 0) tokens.push({ style: run.style, text: active, space: !!activeIsSpace })
                active = ch
                activeIsSpace = isSpace
            } else {
                active += ch
            }
            i++
        }
        if (active.length > 0) {
            tokens.push({ style: run.style, text: active, space: !!activeIsSpace })
            active = ""
            activeIsSpace = null
        }
    }

    function flushRow() {
        rows.push(switchCharSafe(rowStyle) + current)
        current = ""
        currentWidth = 0
        rowStyle = style
    }

    for (const tok of tokens) {
        if (tok.space) {
            if (currentWidth === 0) continue
            if (currentWidth + tok.text.length > width) {
                flushRow()
                continue
            }
            current += tok.text
            currentWidth += tok.text.length
            style = tok.style
            continue
        }

        let t = tok.text
        if (t.length > width) {
            while (t.length > 0) {
                if (currentWidth === width) flushRow()
                const room = width - currentWidth
                if (currentWidth === 0) rowStyle = tok.style
                const take = t.slice(0, room)
                current += switchCharSafe(tok.style) + take
                currentWidth += take.length
                style = tok.style
                t = t.slice(room)
            }
            continue
        }

        if (currentWidth + t.length > width) flushRow()
        if (currentWidth === 0) rowStyle = tok.style
        current += switchCharSafe(tok.style) + t
        currentWidth += t.length
        style = tok.style
    }

    if (current.length > 0 || rows.length === 0) rows.push(switchCharSafe(rowStyle) + current)
    return rows
}

/** @param {number} idx @returns {string} */
function switchCharSafe(idx) {
    return String.fromCharCode(0xe000 + idx)
}

// Left-truncate with a leading "…". Re-emits style switches for the
// retained tail.
/** @param {string} text @param {number} width @returns {string} */
export function truncateLeft(text, width) {
    if (width <= 0) return ""
    if (vwidth(text) <= width) return text
    const runs = [...visibleRuns(text)]
    let kept = ""
    let keptWidth = 0
    const budget = width - 1
    for (let i = runs.length - 1; i >= 0; i--) {
        const r = runs[i]
        if (keptWidth + r.text.length <= budget) {
            kept = switchCharSafe(r.style) + r.text + kept
            keptWidth += r.text.length
            continue
        }
        const room = budget - keptWidth
        if (room > 0) {
            kept = switchCharSafe(r.style) + r.text.slice(-room) + kept
        }
        break
    }
    return "…" + kept
}

/** @param {number} row @param {number} col @returns {string} */
export function cursorTo(row, col) {
    return `${ESC}${row};${col}H`
}

export function clearScreen() {
    return `${ESC}2J${ESC}H`
}

export function clearLine() {
    return `${ESC}2K`
}

export function hideCursor() {
    return `${ESC}?25l`
}

export function showCursor() {
    return `${ESC}?25h`
}

export function reset() {
    return RESET
}
