// ANSI rendering primitives. Compiles a wire-format palette into SGR
// prefix strings, paints palette-encoded text, measures visible widths,
// and renders progress bars.
//
// The palette switch chars are Private Use Area code points
// U+E000..U+E0FF; the renderer treats them as zero-width and translates
// each one to an SGR sequence at paint time.

const ESC = "\x1b["
const RESET = ESC + "0m"

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

// Detect what the terminal can do. Truecolor when COLORTERM is
// truecolor/24bit; 256 colors when TERM contains "256"; otherwise the
// basic 16. The renderer downgrades on its own — engines never see this.
function detectDepth() {
    const ct = process.env.COLORTERM || ""
    if (ct === "truecolor" || ct === "24bit") return "truecolor"
    const term = process.env.TERM || ""
    if (term.includes("256")) return "256"
    return "16"
}

let DEPTH = detectDepth()

export function setColorDepth(d) {
    DEPTH = d
}

function rgbToFg(r, g, b) {
    if (DEPTH === "truecolor") return `38;2;${r};${g};${b}`
    if (DEPTH === "256") return `38;5;${rgbTo256(r, g, b)}`
    return String(rgbToBasic16(r, g, b))
}

function rgbToBg(r, g, b) {
    if (DEPTH === "truecolor") return `48;2;${r};${g};${b}`
    if (DEPTH === "256") return `48;5;${rgbTo256(r, g, b)}`
    return String(rgbToBasic16(r, g, b) + 10)
}

function rgbTo256(r, g, b) {
    if (r === g && g === b) {
        if (r < 8) return 16
        if (r > 248) return 231
        return 232 + Math.round(((r - 8) / 247) * 24)
    }
    const conv = (v) => Math.round((v / 255) * 5)
    return 16 + 36 * conv(r) + 6 * conv(g) + conv(b)
}

function rgbToBasic16(r, g, b) {
    const bright = r > 170 || g > 170 || b > 170
    let code = 30
    if (r > 100) code += 1
    if (g > 100) code += 2
    if (b > 100) code += 4
    return bright ? code + 60 : code
}

function colorRefToSgr(ref, isBg) {
    if (ref == null || ref === "default") return null
    if (typeof ref === "string") return String(isBg ? NAMED_BG[ref] : (NAMED_FG[ref] ?? ""))
    if (Array.isArray(ref) && ref.length === 3) {
        const [r, g, b] = ref
        return isBg ? rgbToBg(r, g, b) : rgbToFg(r, g, b)
    }
    return null
}

// Compile each palette entry to a complete SGR sequence that establishes
// the absolute state for that entry. Renderer prepends a hard reset
// before painting any switch so the SGR is always self-contained.
export function compilePalette(palette) {
    return palette.map((entry) => compileEntry(entry || {}))
}

// Public so callers outside this module can produce one-off styled text
// without standing up a whole palette. `styled(text, entry)` wraps the
// text in the entry's SGR + RESET.
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

// One-shot styled string. Caller passes the same PaletteEntry shape
// engines/views use everywhere else; ansi.js handles the SGR
// translation. Returns the text untouched when the entry is empty or
// null.
export function styled(text, entry) {
    if (!entry) return text
    const sgr = compileEntry(entry)
    if (!sgr) return text
    return sgr + text + RESET
}

// Terminal mode controls (not styles). Switching to the alternate
// screen buffer preserves the user's main scrollback; line-wrap-off
// keeps the renderer in charge of when content wraps.
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

// Strip terminal SGR / OSC / single-char escape sequences out of an
// arbitrary string. Used by engines to clean up llama-server's stdout
// before parsing it; complements `stripStyle` which strips lui's own
// inline PUA palette switches.
const ANSI_STREAM_RE = (() => {
    // eslint-disable-next-line no-control-regex
    return /\x1b\[[\x20-\x3f]*[\x40-\x7e]|\x1b\][\x20-\x7e]*(?:\x07|\x1b\\)|\x1b[\x20-\x2f]*[\x30-\x7e]/g
})()
export function stripAnsi(s) {
    return s.replace(ANSI_STREAM_RE, "")
}

function isSwitchChar(code) {
    return code >= 0xe000 && code <= 0xe0ff
}

// Walk a text string and emit ANSI'd output. Each switch char becomes
// a hard reset followed by the compiled SGR for that palette entry; the
// default entry (index 0) is `""`, so a reset alone clears style.
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

// Visible width: count printable code units; skip switch chars (they're
// zero-width). This is a glyph approximation — full Unicode width
// handling (wide CJK, emoji ZWJ, combining marks) would need a heavier
// implementation; the panels are mostly ASCII so the approximation is
// fine.
export function vwidth(text) {
    let n = 0
    for (let i = 0; i < text.length; i++) {
        const ch = text.charCodeAt(i)
        if (isSwitchChar(ch)) continue
        n += 1
    }
    return n
}

// Strip palette switches and produce plain text (used for non-TTY
// output of `lui cmd` / `lui show`).
export function stripStyle(text) {
    let out = ""
    for (let i = 0; i < text.length; i++) {
        if (isSwitchChar(text.charCodeAt(i))) continue
        out += text[i]
    }
    return out
}

// Iterate visible chars and collect their (codepoint, currentStyleIdx).
// Used by the wrapping path to know which palette index is active when
// a wrap point is chosen — so the continuation row can re-emit the SGR.
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

// Wrap a palette-encoded string to `width` columns. Returns an array of
// strings, each already palette-encoded with a leading switch char that
// reasserts the active style at the start of the row. Word-break
// preferred; mid-token break if a single token exceeds width.
export function wrapStyled(text, width) {
    if (width <= 0) return [text]
    const rows = []
    let current = ""
    let currentWidth = 0
    let style = 0
    let rowStyle = 0

    const tokens = []
    let active = ""
    let activeIsSpace = null

    for (const run of visibleRuns(text)) {
        let i = 0
        while (i < run.text.length) {
            const ch = run.text[i]
            const isSpace = ch === " " || ch === "\t"
            if (activeIsSpace === null) activeIsSpace = isSpace
            if (isSpace !== activeIsSpace || active.length === 0) {
                if (active.length > 0) tokens.push({ style: run.style, text: active, space: activeIsSpace })
                active = ch
                activeIsSpace = isSpace
            } else {
                active += ch
            }
            i++
        }
        if (active.length > 0) {
            tokens.push({ style: run.style, text: active, space: activeIsSpace })
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

function switchCharSafe(idx) {
    return String.fromCharCode(0xe000 + idx)
}

// Right-truncate from the LEFT with a leading "…" when content exceeds
// width. Style switch chars are zero-width — we just walk the visible
// runs in reverse to find the cut point, then re-emit the chunks we
// kept with their style switches.
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

// Render a single-row progress bar. Returns a palette-encoded string of
// exactly `width` visible columns:
//   "label [█████░░░░░] text"
// Label and text already carry their own palette switches; the bar
// itself is uncolored — keep it simple.
export function renderBar(width, label, value, max, text) {
    const frac = max ? Math.max(0, Math.min(1, value / max)) : Math.max(0, Math.min(1, value))
    const labelW = vwidth(label || "")
    const textStr = text == null ? "" : String(text)
    const textW = vwidth(textStr)
    // structure: label + " [" + bar + "] " + text
    const overhead = labelW + 4 + textW + (textW ? 1 : 0)
    const inner = Math.max(1, width - overhead)
    const filled = Math.round(frac * inner)
    const bar = "█".repeat(filled) + "░".repeat(Math.max(0, inner - filled))
    const trailing = textStr ? " " + textStr : ""
    return `${label || ""} [${bar}]${trailing}`
}
