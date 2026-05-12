// HTTP server: /data /config /setup /bsearch /results. Also owns the
// lui-web-search SKILL.md text — harnesses that opt in (via skillsDir)
// drop a copy under their skills directory so the agent can find it.

import http from "node:http"
import os from "node:os"
import { spawn } from "node:child_process"

import { View, CONFIG_VERSION } from "./wire.js"

const BSEARCH_TIMEOUT_MS = 120_000

const CORS = {
    "access-control-allow-origin": "*",
    "access-control-allow-methods": "GET, POST, OPTIONS",
    "access-control-allow-headers": "Content-Type"
}

let bsearchSeq = 0
function nextBsearchId() {
    bsearchSeq += 1
    return `${Date.now().toString(36)}-${bsearchSeq.toString(36)}`
}

export async function startWebServer(lui) {
    const port = lui.config.global.web_port
    const host = lui.config.global.public ? "0.0.0.0" : "127.0.0.1"

    const pending = new Map() // id → { resolve, reject, query }

    const server = http.createServer((req, res) => handleRequest(req, res, lui, pending))

    await new Promise((resolve, reject) => {
        server.once("error", reject)
        server.listen(port, host, () => resolve())
    })

    const externalHost = host === "0.0.0.0" ? (bestLanIp() ?? "127.0.0.1") : "127.0.0.1"
    const bookmarkletUrl = `http://${externalHost}:${port}/setup`

    return {
        server,
        port,
        bookmarkletUrl,
        async close() {
            await new Promise((res) => server.close(() => res()))
        }
    }
}

function handleRequest(req, res, lui, pending) {
    if (req.method === "OPTIONS") {
        res.writeHead(204, CORS)
        res.end()
        return
    }

    const url = new URL(req.url, "http://localhost")
    const path = url.pathname

    try {
        if (path === "/data") return handleData(req, res, lui)
        if (path === "/config") return handleConfig(req, res, lui)
        if (path === "/setup") return handleSetup(req, res, lui)
        if (path === "/bsearch") return handleBsearch(req, res, lui, pending, url)
        if (path === "/results") return handleResults(req, res, lui, pending, url)
    } catch (e) {
        res.writeHead(500, { ...CORS, "content-type": "text/plain" })
        res.end(`internal error: ${e.message}`)
        return
    }
    res.writeHead(404, { ...CORS, "content-type": "text/plain" })
    res.end("not found")
}

// /data carries the engine's panels + warnings — precisely the subset
// a `remote` engine on another machine would replay. Warnings are
// produced server-side (engine spawn, harness apply, etc.) so they
// travel with the engine state. The local TUI composes its own
// lui-panel around the response.
function handleData(_req, res, lui) {
    const v = View()
    lui.appendWarningsPanel(v)
    lui.engineModule?.appendPanels?.(v, lui)
    const body = JSON.stringify(v.build())
    res.writeHead(200, { ...CORS, "content-type": "application/json" })
    res.end(body)
}

// /config exposes the engine's externally-reachable URL + its critical
// state (context size, model name). Each remote-engine hop just
// re-emits the base_url it learned from upstream, so a turtles→relay
// →llm chain propagates llm's real URL all the way through. Web-port
// and websearch are local concerns at every hop and don't travel.
function handleConfig(req, res, lui) {
    const body = JSON.stringify({
        version: CONFIG_VERSION,
        base_url: resolveBaseURL(req, lui),
        active_model: lui.activeModel?.name ?? lui.config.activeModelName ?? null,
        context_size: lui.engineModule?.contextSize?.(lui.state, lui.activeModel) ?? null
    })
    res.writeHead(200, { ...CORS, "content-type": "application/json" })
    res.end(body)
}

// Build an externally-reachable base_url for the active engine. The
// engine returns {host, port}; host=null means "use the hostname the
// client used to reach us" — which we read from req.headers.host so
// the URL is meaningful to whoever just called /config. Remote
// engines return their already-resolved upstream host, ignoring the
// request hostname entirely.
export function resolveBaseURL(req, lui) {
    const ep = lui.engineModule?.endpoint?.(lui)
    if (!ep) return null
    const reqHost = (req?.headers?.host || "").split(":")[0]
    const host = ep.host ?? reqHost ?? "127.0.0.1"
    return `http://${host}:${ep.port}/v1`
}

function handleSetup(_req, res, lui) {
    const port = lui.config.global.web_port
    res.writeHead(200, { ...CORS, "content-type": "text/html; charset=utf-8" })
    res.end(setupPageHtml(port))
}

function handleBsearch(_req, res, lui, pending, url) {
    const query = (url.searchParams.get("q") || "").trim()
    if (!query) {
        res.writeHead(400, { ...CORS, "content-type": "text/plain" })
        res.end("missing q")
        return
    }
    const id = nextBsearchId()

    const googleUrl = `https://www.google.com/search?q=${encodeURIComponent(query)}&lui=${encodeURIComponent(id)}`
    try {
        openInBrowser(googleUrl)
    } catch (e) {
        res.writeHead(500, { ...CORS, "content-type": "text/plain" })
        res.end(`failed to open browser: ${e.message}`)
        return
    }

    let settled = false
    const timer = setTimeout(() => {
        if (settled) return
        settled = true
        pending.delete(id)
        res.writeHead(504, { ...CORS, "content-type": "text/plain" })
        res.end(`user did not click the lui-grab bookmarklet within ${Math.floor(BSEARCH_TIMEOUT_MS / 1000)}s`)
    }, BSEARCH_TIMEOUT_MS)

    pending.set(id, {
        query,
        resolve(payload) {
            if (settled) return
            settled = true
            clearTimeout(timer)
            res.writeHead(200, { ...CORS, "content-type": "application/json" })
            res.end(JSON.stringify(payload))
        }
    })
}

function handleResults(req, res, _lui, pending, url) {
    const id = url.searchParams.get("id")
    if (!id || !pending.has(id)) {
        res.writeHead(404, { ...CORS, "content-type": "text/plain" })
        res.end("no pending search with that id (timed out or already received)")
        return
    }
    let body = ""
    req.setEncoding("utf8")
    req.on("data", (c) => (body += c))
    req.on("end", () => {
        let payload
        try {
            payload = JSON.parse(body)
        } catch {
            payload = { results: [], warnings: ["lui: failed to parse bookmarklet payload"] }
        }
        const entry = pending.get(id)
        pending.delete(id)
        entry?.resolve(payload)
        res.writeHead(200, CORS)
        res.end("ok")
    })
}

function openInBrowser(url) {
    if (process.platform === "darwin") return spawn("open", [url], { stdio: "ignore", detached: true }).unref()
    if (process.platform === "win32") return spawn("cmd", ["/c", "start", "", url], { stdio: "ignore", detached: true }).unref()
    return spawn("xdg-open", [url], { stdio: "ignore", detached: true }).unref()
}

function bestLanIp() {
    const ifaces = os.networkInterfaces()
    for (const list of Object.values(ifaces)) {
        for (const x of list || []) {
            if (x.family === "IPv4" && !x.internal) return x.address
        }
    }
    return null
}

function bookmarkletJs(port) {
    return `(function(){
  try {
    var params = new URL(location.href).searchParams;
    var id = params.get('lui') || '';
    var q = params.get('q') || '';
    var results = [];
    var warnings = [];
    var seen = {};
    function pushResult(h3, a, container) {
      if (!h3 || !a) return;
      var url = a.href;
      if (!url || seen[url]) return;
      seen[url] = 1;
      var scope = container || a.parentElement || document;
      var snipEl = scope.querySelector('div.VwiC3b, span.aCOpRe, div[data-sncf]');
      results.push({
        title: (h3.innerText || '').trim(),
        url: url,
        snippet: snipEl && !snipEl.contains(h3) ? (snipEl.innerText || '').trim() : ''
      });
    }
    document.querySelectorAll('div.g, div.tF2Cxc, div.MjjYud').forEach(function(node) {
      pushResult(node.querySelector('h3'), node.querySelector('a[href^="http"]'), node);
    });
    var fastCount = results.length;
    if (fastCount < 3) {
      document.querySelectorAll('h3').forEach(function(h3) {
        var a = h3.closest('a[href^="http"]');
        if (!a) return;
        var container = null;
        var n = a.parentElement;
        for (var i = 0; i < 5 && n; i++) {
          if (n.querySelector('div.VwiC3b, span.aCOpRe, div[data-sncf]')) { container = n; break; }
          n = n.parentElement;
        }
        pushResult(h3, a, container);
      });
      var added = results.length - fastCount;
      if (fastCount === 0 && added > 0) {
        warnings.push('lui-grab: the fast-path Google selectors (div.g, div.tF2Cxc, div.MjjYud) matched no results; the structural fallback recovered ' + added + '. The class names in lui\\'s web.js likely need updating.');
      } else if (added > 0) {
        warnings.push('lui-grab: the fast-path Google selectors matched only ' + fastCount + ' of ' + results.length + ' results; the structural fallback filled in the rest.');
      }
    }
    if (results.length === 0) {
      alert('lui-grab: found no results on this page. Are you on a Google search results page?');
      return;
    }
    var payload = { results: results, warnings: warnings };
    fetch('http://127.0.0.1:${port}/results?id=' + encodeURIComponent(id) + '&q=' + encodeURIComponent(q), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }).then(function(r) {
      if (r.ok) {
        document.title = '\\u2713 lui-grab: ' + results.length + ' results sent' + (warnings.length ? ' (' + warnings.length + ' warning)' : '');
        try { window.close(); } catch (e) {}
      } else {
        alert('lui-grab: server returned ' + r.status + ' (search may have timed out)');
      }
    }).catch(function(err) {
      navigator.clipboard.writeText(JSON.stringify(payload, null, 2))
        .then(function() { alert('lui-grab: server unreachable, ' + results.length + ' results copied to clipboard'); })
        .catch(function() { alert('lui-grab failed: ' + err); });
    });
  } catch (e) {
    alert('lui-grab error: ' + e);
  }
})();`
}

export function renderWebsearchSkill(port) {
    return `---
name: lui-web-search
description: Web search via browser bookmarklet. Extracts live search results from Google to answer questions requiring up-to-date information. Use when the user asks to search the web, look something up, find recent information, or you need data past your training cutoff. Returns JSON results with title, url, and snippet.
license: BSD-2-Clause
---

# lui-web-search

lui's search endpoint opens a Google search tab in the user's real
browser. The user clicks a one-time-installed \`lui-grab\` bookmarklet on
the resulting page; the bookmarklet POSTs the rendered results back to
lui, which returns them to you.

## Endpoint

\`\`\`
GET http://127.0.0.1:${port}/bsearch?q=<URL-ENCODED QUERY>
\`\`\`

- \`q\` (required): the search query. URL-encode it.

The request **blocks for up to 120 seconds** while waiting for the
user to click the bookmarklet. On timeout you'll get HTTP 504.

## Response

JSON object:

\`\`\`json
{
  "results": [
    {"title": "...", "url": "https://...", "snippet": "..."}
  ],
  "warnings": ["..."]
}
\`\`\`

\`results\` is always present. \`warnings\` is present only when the
bookmarklet had something to tell you — for example, when Google's
CSS class names rotated and the bookmarklet had to fall back to a
structural selector to find results. **If \`warnings\` is non-empty,
surface each warning verbatim to the user** at the end of your reply
(under a short heading like "Note from lui-grab:"), on top of your
normal answer. The user is the only one who can act on it (usually
by updating lui).

An HTTP 504 means the user did not click the bookmarklet in time
(probably they were AFK or the browser tab got buried). Other 4xx/5xx
or an empty \`results\` array means the search failed — say so plainly
rather than fabricating answers.

## How to invoke

\`\`\`sh
curl -sG 'http://127.0.0.1:${port}/bsearch' \\
  --data-urlencode 'q=rust async traits 2026'
\`\`\`

On Windows (PowerShell):

\`\`\`powershell
$q = [uri]::EscapeDataString('rust async traits 2026')
curl.exe -s "http://127.0.0.1:${port}/bsearch?q=$q"
\`\`\`

Read the JSON, then write your answer as normal prose with markdown
links. Do not paste the raw JSON back into the chat. If you need the
body of a specific page, fetch that page separately.

## When to use

- User asks to "search the web", "look up", "google", "find recent", etc.
- You need information that post-dates your training cutoff.
- You need a canonical URL for documentation, a release, a spec, or an issue.

Do not use this for fetching content from a URL the user already gave
you — just fetch that URL directly.

## Important: this requires user action

Each call pops a browser tab the user must click on. Before invoking
this for the first time in a conversation, **tell the user what's about
to happen** so they can be ready, e.g.:

> "I'm going to search the web for that. When I do, a Google tab will
> open in your browser — click the **lui-grab** bookmarklet on it. If
> you haven't installed lui-grab yet, visit
> \`http://127.0.0.1:${port}/setup\` and drag it to your bookmarks bar
> first. (This URL is also shown in the lui status panel.)"

Then call \`/bsearch\` and wait. If the call returns HTTP 504, the user
didn't click in time — most likely they don't have the bookmarklet
installed yet. Stop, point them at the setup page, wait for them to
say it's ready, then retry.

Be deliberate about when to search:
- One search at a time. Do not fire parallel searches.
- Pick the best query first instead of iterating with small variations.
- Don't search for things you already know.
`
}

function setupPageHtml(port) {
    const js = bookmarkletJs(port)
    const minified = js
        .split("\n")
        .map((l) => l.trim())
        .join("\n")
    const href = "javascript:" + encodeURIComponent(minified)
    const hrefAttr = href.replace(/"/g, "&quot;")
    return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>lui — install lui-grab</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 640px; margin: 2em auto; padding: 0 1em; color: #222; line-height: 1.5; }
  h1 { font-size: 1.4em; }
  .grab { display: inline-block; padding: 0.6em 1em; background: #2563eb; color: white !important; text-decoration: none; border-radius: 6px; font-weight: 600; margin: 1em 0; }
  .grab:hover { background: #1d4ed8; }
  code { background: #f3f4f6; padding: 0.1em 0.3em; border-radius: 3px; font-size: 0.9em; }
  ol li { margin-bottom: 0.5em; }
  .note { background: #fef3c7; padding: 0.8em 1em; border-radius: 6px; font-size: 0.9em; margin-top: 1.5em; }
</style>
</head>
<body>
<h1>Install <code>lui-grab</code></h1>
<p>Drag this button onto your <strong>bookmarks bar</strong>:</p>
<p><a class="grab" href="${hrefAttr}">lui-grab</a></p>

<h2>How it works</h2>
<ol>
  <li>When the model wants to search, lui opens a Google search tab in this browser.</li>
  <li>You click <code>lui-grab</code> in the bookmarks bar.</li>
  <li>The bookmarklet reads the rendered results and POSTs them to lui at <code>http://127.0.0.1:${port}/results</code>.</li>
  <li>The model's tool call returns and it continues.</li>
</ol>

<p>If your browser hides the bookmarks bar: in Chrome/Brave/Edge press <code>Cmd/Ctrl+Shift+B</code>; in Firefox right-click the toolbar &rarr; <em>Bookmarks Toolbar</em> &rarr; <em>Always Show</em>.</p>
</body>
</html>
`
}
