// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

//! Browser-mediated web-search HTTP server.
//!
//! Exposes `GET /bsearch?q=...` on 127.0.0.1:<port>. Each call opens a
//! Google search tab in the user's default browser; the user clicks a
//! one-time-installed `lui-grab` bookmarklet on the resulting page; the
//! bookmarklet reads the rendered DOM and POSTs `[{title, url, snippet}]`
//! to `/results?id=<id>`, which unblocks the original tool call.
//!
//! Why this design: every keyless scrape backend (DDG, Bing, Google) now
//! either captchas or serves a JS-only shim. Driving the user's real
//! browser sidesteps all of that — it's a real human session with the
//! user's real cookies, and the browser already executed the JS, so
//! reading `<h3>` tags from the DOM Just Works.
//!
//! The setup page at `GET /setup` hosts the draggable bookmarklet.
//! The companion skill `lui-web-search` tells opencode how to invoke
//! `/bsearch`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::oneshot;

use axum::{
    extract::{Query, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{Html as HtmlResponse, IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};

use crate::server::ServerState;

#[derive(Debug, Deserialize)]
pub struct SearchQuery {
    pub q: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[derive(Clone)]
struct AppState {
    server_state: Arc<Mutex<ServerState>>,
    // Pending browser-mediated searches keyed by request id. /bsearch
    // inserts a oneshot sender, opens the browser, and awaits the receiver.
    // The bookmarklet POSTs to /results?id=<id>, which fires the sender and
    // unblocks the tool call. std Mutex is fine — only held briefly to hash
    // a key in/out, never across an await.
    pending: Arc<Mutex<HashMap<String, oneshot::Sender<Vec<SearchResult>>>>>,
    port: u16,
}

// How long /bsearch waits for the user to click the bookmarklet before
// giving up. Long enough to find the right tab and click; short enough
// that a forgotten search doesn't tie up a tool call forever.
const BSEARCH_TIMEOUT: Duration = Duration::from_secs(120);

static NEXT_BSEARCH_ID: AtomicU64 = AtomicU64::new(1);

fn next_bsearch_id() -> String {
    let n = NEXT_BSEARCH_ID.fetch_add(1, Ordering::Relaxed);
    // Mix in seconds-since-epoch so ids don't collide across restarts in
    // case a stale browser tab tries to POST after we've restarted.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{:x}-{:x}", secs, n)
}

pub fn spawn(port: u16, server_state: Arc<Mutex<ServerState>>) {
    let state = AppState {
        server_state,
        pending: Arc::new(Mutex::new(HashMap::new())),
        port,
    };

    let app = Router::new()
        .route("/bsearch", get(handle_bsearch))
        .route("/results", post(handle_results).options(handle_results_preflight))
        .route("/setup", get(handle_setup))
        .route("/health", get(|| async { "ok" }))
        .with_state(state);

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));

    tokio::spawn(async move {
        let listener = match tokio::net::TcpListener::bind(addr).await {
            Ok(l) => l,
            Err(e) => {
                eprintln!("lui websearch: failed to bind {}: {}", addr, e);
                return;
            }
        };
        if let Err(e) = axum::serve(listener, app).await {
            eprintln!("lui websearch: server error: {}", e);
        }
    });
}


// ---------------------------------------------------------------------------
// Browser-mediated search: lui opens the user's browser to Google, the user
// clicks a one-time-installed bookmarklet on the resulting SERP, and the
// bookmarklet POSTs the rendered DOM's results back to lui. This sidesteps
// every anti-bot wall (it's a real human session) and Google's no-JS shim
// (the browser already executed the JS for us).
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ResultsQuery {
    // Both are optional because Google rewrites SERP URLs on load (via
    // history.replaceState) and can drop unknown params like `lui`. We
    // fall back to matching by `q` since the query param always survives.
    id: Option<String>,
    q: Option<String>,
}

fn cors_headers() -> HeaderMap {
    let mut h = HeaderMap::new();
    // `*` is fine here: the server binds 127.0.0.1 only and we don't use
    // cookies, so there's no credentialed-CORS pitfall.
    h.insert(
        header::ACCESS_CONTROL_ALLOW_ORIGIN,
        HeaderValue::from_static("*"),
    );
    h.insert(
        header::ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_static("POST, OPTIONS"),
    );
    h.insert(
        header::ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static("Content-Type"),
    );
    h
}

async fn handle_results_preflight() -> impl IntoResponse {
    (StatusCode::NO_CONTENT, cors_headers())
}

async fn handle_results(
    State(state): State<AppState>,
    Query(rq): Query<ResultsQuery>,
    Json(results): Json<Vec<SearchResult>>,
) -> impl IntoResponse {
    // Try the bookmarklet-provided id first; if it's missing or unknown
    // (Google stripped the `lui` param), fall back to matching the pending
    // search by query string.
    let id_match = rq.id.as_ref().and_then(|id| {
        if state.pending.lock().unwrap().contains_key(id) {
            Some(id.clone())
        } else {
            None
        }
    });
    let id_to_use = id_match.or_else(|| {
        let query = rq.q.as_ref().filter(|s| !s.is_empty())?;
        let active = state.server_state.lock().unwrap();
        active
            .active_searches
            .iter()
            .find_map(|(k, v)| if v == query { Some(k.clone()) } else { None })
    });
    let sender = id_to_use.and_then(|id| state.pending.lock().unwrap().remove(&id));
    match sender {
        Some(tx) => {
            // If the receiver was dropped (timeout fired), tx.send returns
            // Err — that's fine, the bookmarklet still gets a 200 so it can
            // close the tab. We just discard the orphan results.
            let _ = tx.send(results);
            (StatusCode::OK, cors_headers(), "ok")
        }
        None => (
            StatusCode::NOT_FOUND,
            cors_headers(),
            "no pending search with that id (timed out or already received)",
        ),
    }
}

async fn handle_bsearch(
    State(state): State<AppState>,
    Query(q): Query<SearchQuery>,
) -> Result<Json<Vec<SearchResult>>, (StatusCode, String)> {
    let query = q.q.trim().to_string();
    if query.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "missing q".into()));
    }

    let id = next_bsearch_id();
    let (tx, rx) = oneshot::channel();
    state.pending.lock().unwrap().insert(id.clone(), tx);

    {
        let mut st = state.server_state.lock().unwrap();
        st.websearch_total += 1;
        st.active_searches.insert(id.clone(), query.clone());
    }

    let google_url = format!(
        "https://www.google.com/search?q={}&lui={}",
        urlencoding::encode(&query),
        urlencoding::encode(&id),
    );
    if let Err(e) = open_in_browser(&google_url) {
        state.pending.lock().unwrap().remove(&id);
        state.server_state.lock().unwrap().active_searches.remove(&id);
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to open browser: {}", e),
        ));
    }

    let outcome = tokio::time::timeout(BSEARCH_TIMEOUT, rx).await;

    state.server_state.lock().unwrap().active_searches.remove(&id);

    match outcome {
        Ok(Ok(results)) => Ok(Json(results)),
        Ok(Err(_)) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "result channel closed unexpectedly".into(),
        )),
        Err(_) => {
            // Timeout fired; sweep the pending map so a late POST doesn't
            // see a phantom entry. (Race: another POST may have already
            // removed it — that's harmless.)
            state.pending.lock().unwrap().remove(&id);
            Err((
                StatusCode::GATEWAY_TIMEOUT,
                format!(
                    "user did not click the lui-grab bookmarklet within {}s",
                    BSEARCH_TIMEOUT.as_secs()
                ),
            ))
        }
    }
}

fn open_in_browser(url: &str) -> std::io::Result<()> {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open").arg(url).spawn()?;
    }
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open").arg(url).spawn()?;
    }
    #[cfg(target_os = "windows")]
    {
        // The empty "" is the window title arg `start` expects before the URL,
        // otherwise URLs containing & are mis-parsed.
        std::process::Command::new("cmd")
            .args(["/c", "start", "", url])
            .spawn()?;
    }
    Ok(())
}

async fn handle_setup(State(state): State<AppState>) -> HtmlResponse<String> {
    HtmlResponse(setup_page_html(state.port))
}

/// JS that runs when the user clicks the bookmarklet on a Google SERP.
/// Reads the rendered DOM (so we get whatever Google's JS produced —
/// no fragile HTML scraping), POSTs to lui, falls back to clipboard
/// if the POST is blocked (CSP / mixed content / lui not running).
fn bookmarklet_js(port: u16) -> String {
    // Note: kept readable here for maintenance; URL-encoded once at
    // render time below.
    format!(
        r#"(function(){{
  try {{
    var params = new URL(location.href).searchParams;
    var id = params.get('lui') || '';
    var q = params.get('q') || '';
    var nodes = document.querySelectorAll('div.g, div.tF2Cxc, div.MjjYud');
    var results = [];
    var seen = {{}};
    nodes.forEach(function(node) {{
      var h3 = node.querySelector('h3');
      if (!h3) return;
      var a = node.querySelector('a[href^="http"]');
      if (!a) return;
      var url = a.href;
      if (seen[url]) return;
      seen[url] = 1;
      var snipEl = node.querySelector('div.VwiC3b, span.aCOpRe, div[data-sncf]');
      results.push({{
        title: (h3.innerText || '').trim(),
        url: url,
        snippet: snipEl ? (snipEl.innerText || '').trim() : ''
      }});
    }});
    if (results.length === 0) {{
      alert('lui-grab: found no results on this page. Are you on a Google search results page?');
      return;
    }}
    fetch('http://127.0.0.1:{port}/results?id=' + encodeURIComponent(id) + '&q=' + encodeURIComponent(q), {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify(results)
    }}).then(function(r) {{
      if (r.ok) {{
        document.title = '\u2713 lui-grab: ' + results.length + ' results sent';
        try {{ window.close(); }} catch (e) {{}}
      }} else {{
        alert('lui-grab: server returned ' + r.status + ' (search may have timed out)');
      }}
    }}).catch(function(err) {{
      navigator.clipboard.writeText(JSON.stringify(results, null, 2))
        .then(function() {{ alert('lui-grab: server unreachable, ' + results.length + ' results copied to clipboard'); }})
        .catch(function() {{ alert('lui-grab failed: ' + err); }});
    }});
  }} catch (e) {{
    alert('lui-grab error: ' + e);
  }}
}})();"#
    )
}

fn setup_page_html(port: u16) -> String {
    let js = bookmarklet_js(port);
    // Minify the JS by stripping leading whitespace per line; bookmarklets
    // don't need the indentation and shorter URLs are friendlier in the
    // bookmark bar.
    let minified: String = js
        .lines()
        .map(|l| l.trim())
        .collect::<Vec<_>>()
        .join("");
    let href = format!("javascript:{}", urlencoding::encode(&minified));
    // Manually escape `"` for embedding the href in an HTML attribute. The
    // urlencoded JS contains no `"` itself.
    let href_attr = href.replace('"', "&quot;");

    format!(
        r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>lui — install lui-grab</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 640px; margin: 2em auto; padding: 0 1em; color: #222; line-height: 1.5; }}
  h1 {{ font-size: 1.4em; }}
  .grab {{ display: inline-block; padding: 0.6em 1em; background: #2563eb; color: white !important; text-decoration: none; border-radius: 6px; font-weight: 600; margin: 1em 0; }}
  .grab:hover {{ background: #1d4ed8; }}
  code {{ background: #f3f4f6; padding: 0.1em 0.3em; border-radius: 3px; font-size: 0.9em; }}
  ol li {{ margin-bottom: 0.5em; }}
  .note {{ background: #fef3c7; padding: 0.8em 1em; border-radius: 6px; font-size: 0.9em; margin-top: 1.5em; }}
</style>
</head>
<body>
<h1>Install <code>lui-grab</code></h1>
<p>Drag this button onto your <strong>bookmarks bar</strong>:</p>
<p><a class="grab" href="{href_attr}">lui-grab</a></p>

<h2>How it works</h2>
<ol>
  <li>When the model wants to search, lui opens a Google search tab in this browser.</li>
  <li>You click <code>lui-grab</code> in the bookmarks bar.</li>
  <li>The bookmarklet reads the rendered results and POSTs them to lui at <code>http://127.0.0.1:{port}/results</code>.</li>
  <li>The model's tool call returns and it continues.</li>
</ol>

<p>If your browser hides the bookmarks bar: in Chrome/Brave/Edge press <code>Cmd/Ctrl+Shift+B</code>; in Firefox right-click the toolbar &rarr; <em>Bookmarks Toolbar</em> &rarr; <em>Always Show</em>.</p>

<div class="note">
<strong>Privacy:</strong> the bookmarklet only runs when you click it, only on the current tab, and only sends data to <code>127.0.0.1</code>. No data leaves your machine.
</div>
</body>
</html>
"##
    )
}

