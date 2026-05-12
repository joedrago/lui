// View builder for the /data wire format. Style is encoded inline as
// PUA code points U+E000..U+E0FF indexing into a per-View palette.
// palette[0] is always {} (the default style), so emitting  is
// the implicit "back to default" reset.
//
// The View dedups palette entries via canonical-JSON keys, so engines
// never think about indices — they pass plain PaletteEntry objects to
// .style().

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
    const palette = [{}]
    const paletteIndex = new Map()
    paletteIndex.set(DEFAULT_KEY, 0)
    const panels = []

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

    function panel(title) {
        const p = { title: title ?? "", lines: [], bars: [] }
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

    function build() {
        return { version: 1, palette, panels }
    }

    return { panel, build }
}
