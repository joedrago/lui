// View builder for the /data wire format. Each panel carries its own
// palette so panels can be moved between Views without rewriting their
// inline style markers. Inline style switches are PUA code points
// U+E000..U+E0FF indexing into that panel's palette; palette[0] is
// always {} (the default style), so emitting  is "back to default".
//
// The per-panel palette dedups entries via canonical-JSON keys, so
// callers never think about indices — they pass plain PaletteEntry
// objects to .style().

// Protocol version reported by /config and required by the `remote`
// engine when it dials another lui. Bump together with any breaking
// change to the /config response shape.
export const CONFIG_VERSION = 4

const DEFAULT_KEY = "{}"

function canonicalKey(entry) {
    if (!entry) return DEFAULT_KEY
    const keys = Object.keys(entry).sort()
    if (keys.length === 0) return DEFAULT_KEY
    const sorted = {}
    for (const k of keys) sorted[k] = entry[k]
    return JSON.stringify(sorted)
}

function switchChar(idx) {
    return String.fromCharCode(0xe000 + idx)
}

export function View() {
    const panels = []

    function panel(title) {
        const palette = [{}]
        const paletteIndex = new Map()
        paletteIndex.set(DEFAULT_KEY, 0)

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

        const p = { title: title ?? "", palette, lines: [], bars: [] }
        panels.push(p)

        function line(opts) {
            const obj = { text: "" }
            if (opts?.align) obj.align = opts.align
            if (opts?.indent) obj.indent = opts.indent
            p.lines.push(obj)

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

        function bar(spec) {
            const out = { label: spec.label ?? "", value: spec.value ?? 0 }
            if (spec.max != null) out.max = spec.max
            if (spec.text != null) out.text = spec.text
            if (spec.indent) out.indent = spec.indent
            p.bars.push(out)
            return panelApi
        }

        const panelApi = { line, bar }
        return panelApi
    }

    // Append a panel object {title, palette, lines, bars} from another
    // View as-is. Because each panel owns its palette, the inline PUA
    // indices already point at the right entries — no rewrite needed.
    function adoptPanel(remotePanel) {
        if (!remotePanel) return
        panels.push({
            title: remotePanel.title ?? "",
            palette: remotePanel.palette ?? [{}],
            lines: remotePanel.lines ?? [],
            bars: remotePanel.bars ?? []
        })
    }

    function build() {
        return { version: 2, panels }
    }

    return { panel, adoptPanel, build }
}
