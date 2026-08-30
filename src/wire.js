// View builder for the /data wire format. Each panel carries its own
// palette so panels can be moved between Views without rewriting their
// inline style markers. Inline style switches are PUA code points
// U+E000..U+E0FF indexing into that panel's palette; palette[0] is
// always {} (the default style), so emitting  is "back to default".
//
// The per-panel palette dedups entries via canonical-JSON keys, so
// callers never think about indices — they pass plain PaletteEntry
// objects to .style().

/** @import { PaletteEntry, ViewBuilder, PanelApi, LineApi, ViewPanel, ViewBar, ViewLine, BuiltView } from "./types.js" */

// Protocol version reported by /config and required by the `remote`
// engine when it dials another lui. Bump together with any breaking
// change to the /config response shape.
export const CONFIG_VERSION = 5

// Generation cap used when nothing else supplies one: the
// `max_output_tokens` schema default, and the floor a `remote` hop
// lands on if an upstream somehow announces no value.
export const DEFAULT_MAX_OUTPUT_TOKENS = 8192

const DEFAULT_KEY = "{}"

/** @param {PaletteEntry | null | undefined} entry @returns {string} */
function canonicalKey(entry) {
    if (!entry) return DEFAULT_KEY
    const keys = Object.keys(entry).sort()
    if (keys.length === 0) return DEFAULT_KEY
    /** @type {Record<string, any>} */
    const sorted = {}
    for (const k of keys) sorted[k] = /** @type {any} */ (entry)[k]
    return JSON.stringify(sorted)
}

/** @param {number} idx @returns {string} */
function switchChar(idx) {
    return String.fromCharCode(0xe000 + idx)
}

/** @returns {ViewBuilder} */
export function View() {
    /** @type {ViewPanel[]} */
    const panels = []

    /** @param {string} [title] @returns {PanelApi} */
    function panel(title) {
        /** @type {PaletteEntry[]} */
        const palette = [{}]
        /** @type {Map<string, number>} */
        const paletteIndex = new Map()
        paletteIndex.set(DEFAULT_KEY, 0)

        /** @param {PaletteEntry} entry @returns {number} */
        function styleIdx(entry) {
            const key = canonicalKey(entry)
            let i = paletteIndex.get(key)
            if (i === undefined) {
                i = palette.length
                palette.push({ ...entry })
                paletteIndex.set(key, i)
            }
            return i
        }

        /** @type {ViewPanel} */
        const p = { title: title ?? "", palette, lines: [], bars: [] }
        panels.push(p)

        /** @param {{ align?: "left" | "right", indent?: number, nowrap?: boolean }} [opts] @returns {LineApi} */
        function line(opts) {
            /** @type {ViewLine} */
            const obj = { text: "" }
            if (opts?.align) obj.align = opts.align
            if (opts?.indent) obj.indent = opts.indent
            if (opts?.nowrap) obj.nowrap = true
            p.lines.push(obj)

            /** @type {LineApi} */
            const api = {
                text(s) {
                    if (s != null && s !== "") obj.text += String(s)
                    return api
                },
                style(entry) {
                    const idx = styleIdx(entry ?? {})
                    obj.text += switchChar(idx)
                    return api
                }
            }
            return api
        }

        /** @param {ViewBar} spec @returns {PanelApi} */
        function bar(spec) {
            /** @type {ViewBar} */
            const out = { label: spec.label ?? "", value: spec.value ?? 0 }
            if (spec.max != null) out.max = spec.max
            if (spec.text != null) out.text = spec.text
            if (spec.indent) out.indent = spec.indent
            p.bars.push(out)
            return panelApi
        }

        /** @type {PanelApi} */
        const panelApi = { line, bar }
        return panelApi
    }

    // Append a panel object {title, palette, lines, bars} from another
    // View as-is. Because each panel owns its palette, the inline PUA
    // indices already point at the right entries — no rewrite needed.
    /** @param {ViewPanel | null | undefined} remotePanel */
    function adoptPanel(remotePanel) {
        if (!remotePanel) return
        panels.push({
            title: remotePanel.title ?? "",
            palette: remotePanel.palette ?? [{}],
            lines: remotePanel.lines ?? [],
            bars: remotePanel.bars ?? []
        })
    }

    /** @returns {BuiltView} */
    function build() {
        return { version: 2, panels }
    }

    return { panel, adoptPanel, build }
}
