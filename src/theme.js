// Single source of truth for palette entries used by lui's UI. Names
// describe role, not color — so a future theme change happens in one
// place. Each entry is a plain `PaletteEntry` (see wire.js).

/** @import { PaletteEntry } from "./types.js" */

/** @type {Record<string, PaletteEntry>} */
export const STYLE = {
    // Inline field labels ("Memory   :") and panel title dividers.
    LABEL: { fg: [120, 100, 180] },

    // Highlighted values: numbers in status lines, bar trailing text,
    // config-dump values.
    VALUE: { fg: [210, 150, 255] },

    // The "lui" brand mark in the shutdown summary.
    BRAND: { fg: [180, 150, 255], bold: true },

    // The bold alias name that follows the long model name.
    ALIAS: { fg: [220, 215, 230] },

    // Progress-bar glyphs.
    BAR_FILL: { fg: [180, 150, 255] },
    BAR_EMPTY: { fg: [120, 100, 180] },

    // Status states.
    READY: { fg: "green" },
    WARNING: { fg: [230, 180, 80] },
    FATAL: { fg: [230, 100, 100] },
    FATAL_LABEL: { fg: [230, 100, 100], bold: true },
    ERROR_INLINE: { fg: "red" },

    // Emphasis accent — multiselect cursor, "Tip:" labels.
    ACTIVE: { fg: [230, 200, 140], bold: true },

    // Config-key path in dumps ("global.engine_port", etc.).
    CONFIG_KEY: { fg: [230, 200, 140] },

    // Engine name shown next to a model in the config dump.
    ENGINE_NAME: { fg: "cyan" },

    // Clickable URL (bookmarklet setup, listen address).
    URL: { fg: "cyan" },

    // URL highlighted briefly at startup so the user can't miss it.
    URL_FRESH: { fg: "cyan", bold: true },

    // Harness name standin in the sandbox preview.
    HARNESS_NAME: { fg: "magenta", bold: true },

    // Argv segment categories. Engines tag each segment with one so
    // the config dump colors the spawn command consistently. The binary
    // segment (segment 0 of describe(), if present) is intentionally
    // unstyled — matches how the sandbox command renders `nono`, and
    // keeps a long resolved path from competing visually with the
    // colored flag groups that follow.
    SEGMENT_BINDING: { fg: "cyan" },
    SEGMENT_POLICY: { dim: true },
    SEGMENT_DEFAULTS: { fg: [100, 170, 200] },
    SEGMENT_USER: { fg: [230, 200, 140] }
}
