/**
 * @typedef {Object} PaletteEntry
 * @property {boolean} [bold]
 * @property {boolean} [dim]
 * @property {boolean} [italic]
 * @property {boolean} [underline]
 * @property {string | [number, number, number]} [fg]
 * @property {string | [number, number, number]} [bg]
 */

/** @typedef {Object} Model
 * @property {string} name
 * @property {string} engine
 * @property {string[]} args
 */

/** @typedef {{ name: string, style?: PaletteEntry, args: string[] }} Segment */

/** @typedef {{ segments: Segment[], warnings: string[], errors: string[] }} EngineDesc */

/** @typedef {{ host: string | null, port: number }} Endpoint */

/** @typedef {{ host: string, port: number }} HostSpec */

/** @typedef {{ user: string, host: string }} SshTarget */

/** @typedef {{ lines: { label: string, value: string }[], fatal: string | null }} ShutdownSummary */

/** @typedef {{ path: string, default: any, isArray?: boolean }} SchemaEntry */

/** @typedef {{ modelName: string, baseURL: string, ctxSize: number, webPort: number, websearch: boolean }} HarnessContext */

/**
 * @typedef {Object} Harness
 * @property {string} name
 * @property {string} configDir
 * @property {string[]} configCandidates
 * @property {string} [skillsDir]
 * @property {"nested" | "flat"} [skillsLayout]
 * @property {SchemaEntry[]} [schema]
 * @property {(existing: string, ctx: HarnessContext) => string} apply
 * @property {(existing: string) => boolean} [needsBackup]
 * @property {(target: SshTarget, sshRun: (t: SshTarget, cmd: string, stdin?: string) => Promise<string>) => Promise<{ ok: boolean, error?: string }>} [sshPreflight]
 */

/**
 * @typedef {Object} Engine
 * @property {string} name
 * @property {SchemaEntry[]} [schema]
 * @property {Array<{ name: string, sizeGiB?: number, args: string[] }>} [setupDefaults]
 * @property {(model: Model, lui: import("./lui.js").Lui) => EngineDesc} describe
 * @property {(lui: import("./lui.js").Lui, model: Model, desc: EngineDesc) => Promise<void>} start
 * @property {(lui: import("./lui.js").Lui) => Promise<void>} [stop]
 * @property {(lui: import("./lui.js").Lui) => void} [initState]
 * @property {(line: string, lui: import("./lui.js").Lui) => void} [parseLine]
 * @property {(v: ViewBuilder, lui: import("./lui.js").Lui) => void} [appendPanels]
 * @property {(state: any, code: number | null, signal: string | null) => string} [exitReason]
 * @property {(state: any) => ShutdownSummary} [shutdownSummary]
 * @property {(state: any, model: Model | null) => number | null} [contextSize]
 * @property {(state: any, model: Model | null) => string | null} [servedModelName]
 * @property {(lui: import("./lui.js").Lui) => Endpoint | null} [endpoint]
 */

/**
 * @typedef {Object} Transport
 * @property {string} name
 * @property {(p: string) => string} resolve
 * @property {(p: string) => Promise<boolean>} exists
 * @property {(p: string) => Promise<string>} read
 * @property {(p: string, body: string) => Promise<void>} write
 * @property {(p: string) => Promise<void>} remove
 * @property {(p: string) => Promise<void>} mkdirp
 * @property {(p: string) => Promise<void>} tryRmDir
 */

/** @typedef {{ text: string, align?: "left" | "right", indent?: number }} ViewLine */

/** @typedef {{ label: string, value: number, max?: number, text?: string, indent?: number }} ViewBar */

/** @typedef {{ title: string, palette: PaletteEntry[], lines: ViewLine[], bars: ViewBar[] }} ViewPanel */

/** @typedef {{ version: number, panels: ViewPanel[] }} BuiltView */

/**
 * @typedef {Object} LineApi
 * @property {(s: string | number | null | undefined) => LineApi} text
 * @property {(entry?: PaletteEntry) => LineApi} style
 */

/**
 * @typedef {Object} PanelApi
 * @property {(opts?: { align?: "left" | "right", indent?: number }) => LineApi} line
 * @property {(spec: ViewBar) => PanelApi} bar
 */

/**
 * @typedef {Object} ViewBuilder
 * @property {(title?: string) => PanelApi} panel
 * @property {(remotePanel: ViewPanel | null | undefined) => void} adoptPanel
 * @property {() => BuiltView} build
 */

/** @typedef {{ label: string, value: any, hint?: string, selected?: boolean }} MultiselectItem */

/** @typedef {{ binary: string, argv: string[], parseLine: (line: string) => void, debugLog?: string | null, onExit?: (code: number | null, signal: NodeJS.Signals | null) => void, onSpawnError?: (err: NodeJS.ErrnoException) => void, addWarning?: (msg: string) => void, env?: NodeJS.ProcessEnv }} SpawnSpec */

/** @typedef {{ child: import("node:child_process").ChildProcess, stop: () => Promise<void> }} SpawnHandle */

export {}
