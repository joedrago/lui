// Tiny TTY input helpers. No deps — node's stdin in raw mode plus
// a hand-rolled escape-sequence decoder. Just what `lui setup` needs:
// `multiselect` and `confirm`. Both throw `PromptAborted` on Ctrl+C /
// Esc so callers can clean up.

import process from "node:process"

import { styled } from "./ansi.js"
import { STYLE } from "./theme.js"

export class PromptAborted extends Error {
    constructor() {
        super("aborted")
        this.name = "PromptAborted"
    }
}

const ESC = "\x1b"
const HIDE_CURSOR = ESC + "[?25l"
const SHOW_CURSOR = ESC + "[?25h"
const CLEAR_LINE = ESC + "[2K"
const CURSOR_UP = (n) => ESC + `[${n}A`
const CURSOR_LEFT = "\r"

// Multi-select prompt: `items` is [{label, value, hint?, selected?}].
// Returns the array of `value`s the user confirmed (Enter). Space toggles
// the current row. Esc / Ctrl+C throw `PromptAborted`.
export async function multiselect(question, items) {
    if (!process.stdin.isTTY || !process.stdout.isTTY) {
        throw new Error("multiselect requires a TTY")
    }
    if (items.length === 0) return []

    const state = items.map((it) => ({
        label: it.label,
        value: it.value,
        hint: it.hint ?? "",
        selected: !!it.selected
    }))
    let cursor = 0
    let painted = 0

    const render = () => {
        const lines = []
        lines.push(styled(question, STYLE.LABEL) + styled("  (space toggles · enter confirms)", { dim: true }))
        for (let i = 0; i < state.length; i++) {
            const row = state[i]
            const isCursor = i === cursor
            const pointer = isCursor ? styled("›", STYLE.ACTIVE) : " "
            const box = row.selected ? styled("[x]", STYLE.READY) : styled("[ ]", { dim: true })
            const label = isCursor ? styled(row.label, STYLE.VALUE) : row.label
            const hint = row.hint ? "  " + styled(row.hint, { dim: true }) : ""
            lines.push(`${pointer} ${box} ${label}${hint}`)
        }
        const out = lines.map((l) => CLEAR_LINE + l).join("\n")
        if (painted > 0) process.stdout.write(CURSOR_UP(painted) + CURSOR_LEFT)
        process.stdout.write(out + "\n")
        painted = lines.length
    }

    return runRawInput(render, (key, resolve, reject) => {
        if (key === "up" || key === "k") {
            cursor = (cursor - 1 + state.length) % state.length
            render()
        } else if (key === "down" || key === "j") {
            cursor = (cursor + 1) % state.length
            render()
        } else if (key === "space") {
            state[cursor].selected = !state[cursor].selected
            render()
        } else if (key === "enter") {
            resolve(state.filter((r) => r.selected).map((r) => r.value))
        } else if (key === "abort") {
            reject(new PromptAborted())
        }
    })
}

// Y/n confirm. `default` chooses the answer on bare Enter. Esc / Ctrl+C
// throw `PromptAborted`.
export async function confirm(question, defaultYes = true) {
    if (!process.stdin.isTTY || !process.stdout.isTTY) {
        throw new Error("confirm requires a TTY")
    }
    const hint = defaultYes ? "[Y/n]" : "[y/N]"
    let painted = 0
    const render = (answer) => {
        const line =
            styled(question, STYLE.LABEL) +
            " " +
            styled(hint, { dim: true }) +
            (answer != null ? " " + styled(answer ? "yes" : "no", STYLE.VALUE) : "")
        if (painted > 0) process.stdout.write(CURSOR_UP(painted) + CURSOR_LEFT)
        process.stdout.write(CLEAR_LINE + line + "\n")
        painted = 1
    }

    return runRawInput(
        () => render(null),
        (key, resolve, reject) => {
            if (key === "y") {
                render(true)
                resolve(true)
            } else if (key === "n") {
                render(false)
                resolve(false)
            } else if (key === "enter") {
                render(defaultYes)
                resolve(defaultYes)
            } else if (key === "abort") {
                reject(new PromptAborted())
            }
        }
    )
}

// Shared raw-mode runner. `paint()` draws the initial frame; `onKey` is
// called with a decoded key name plus the promise resolvers.
function runRawInput(paint, onKey) {
    return new Promise((resolve, reject) => {
        const stdin = process.stdin
        const wasRaw = stdin.isRaw
        process.stdout.write(HIDE_CURSOR)
        stdin.setRawMode(true)
        stdin.resume()
        stdin.setEncoding("utf8")
        paint()

        const cleanup = () => {
            stdin.off("data", onData)
            stdin.setRawMode(wasRaw)
            stdin.pause()
            process.stdout.write(SHOW_CURSOR)
        }
        const onData = (data) => {
            const key = decodeKey(data)
            onKey(
                key,
                (v) => {
                    cleanup()
                    resolve(v)
                },
                (e) => {
                    cleanup()
                    reject(e)
                }
            )
        }
        stdin.on("data", onData)
    })
}

function decodeKey(data) {
    if (data === "\x03" || data === "\x1b") return "abort"
    if (data === "\r" || data === "\n") return "enter"
    if (data === " ") return "space"
    if (data === "\x1b[A") return "up"
    if (data === "\x1b[B") return "down"
    const lower = data.toLowerCase()
    if (lower === "y") return "y"
    if (lower === "n") return "n"
    if (lower === "j" || lower === "k") return lower
    return ""
}
