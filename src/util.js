// Low-level catchall. Small, generic helpers with no upward deps.

import os from "node:os"
import path from "node:path"

export function expandTilde(p) {
    if (p.startsWith("~/")) return path.join(os.homedir(), p.slice(2))
    if (p === "~") return os.homedir()
    return p
}
