// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

//! Minimal local web-search HTTP server.
//!
//! Exposes `GET /search?q=...&n=10` on 127.0.0.1:<port> and returns a JSON
//! array of `{title, url, snippet}`. Backed by scraping DuckDuckGo's HTML
//! endpoint (html.duckduckgo.com/html/) — chosen over Google because it
//! serves captcha/consent walls far less aggressively.
//!
//! The TUI companion skill `lui-web-search` points opencode at this endpoint,
//! so Qwen (or any tool-using model) can "do a web search" with a single curl.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    routing::get,
    Router,
};
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};

use crate::server::ServerState;

#[derive(Debug, Deserialize)]
pub struct SearchQuery {
    pub q: String,
    #[serde(default)]
    pub n: Option<usize>,
}

#[derive(Debug, Serialize, Clone)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[derive(Clone)]
struct AppState {
    http: reqwest::Client,
    server_state: Arc<Mutex<ServerState>>,
}

pub fn spawn(port: u16, server_state: Arc<Mutex<ServerState>>) {
    let http = reqwest::Client::builder()
        .user_agent(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
             (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )
        .timeout(Duration::from_secs(15))
        .build()
        .expect("reqwest client");

    let state = AppState { http, server_state };

    let app = Router::new()
        .route("/search", get(handle_search))
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

async fn handle_search(
    State(state): State<AppState>,
    Query(q): Query<SearchQuery>,
) -> Result<Json<Vec<SearchResult>>, (StatusCode, String)> {
    let query = q.q.trim().to_string();
    if query.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "missing q".into()));
    }
    let n = q.n.unwrap_or(10).clamp(1, 25);

    {
        let mut st = state.server_state.lock().unwrap();
        st.websearch_active += 1;
        st.websearch_total += 1;
        st.websearch_last_query = query.clone();
    }

    let result = do_search(&state.http, &query, n).await;

    {
        let mut st = state.server_state.lock().unwrap();
        st.websearch_active = st.websearch_active.saturating_sub(1);
    }

    match result {
        Ok(results) => Ok(Json(results)),
        Err(e) => Err((StatusCode::BAD_GATEWAY, format!("search failed: {}", e))),
    }
}

async fn do_search(
    client: &reqwest::Client,
    query: &str,
    n: usize,
) -> Result<Vec<SearchResult>, String> {
    let url = format!(
        "https://html.duckduckgo.com/html/?q={}",
        urlencoding::encode(query)
    );
    let body = client
        .get(&url)
        .send()
        .await
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?
        .text()
        .await
        .map_err(|e| e.to_string())?;

    Ok(parse_ddg(&body, n))
}

fn parse_ddg(html: &str, n: usize) -> Vec<SearchResult> {
    let doc = Html::parse_document(html);

    // DDG HTML layout: each result is .result (or .web-result). Title+link
    // live in a.result__a, snippet in .result__snippet.
    let result_sel = Selector::parse("div.result, div.web-result").unwrap();
    let title_sel = Selector::parse("a.result__a").unwrap();
    let snippet_sel = Selector::parse(".result__snippet").unwrap();

    let mut out = Vec::new();
    for node in doc.select(&result_sel) {
        let (title, url) = match node.select(&title_sel).next() {
            Some(a) => {
                let title = a.text().collect::<Vec<_>>().join(" ").trim().to_string();
                let href = a.value().attr("href").unwrap_or("").to_string();
                (title, unwrap_ddg_redirect(&href))
            }
            None => continue,
        };
        if title.is_empty() || url.is_empty() {
            continue;
        }
        let snippet = node
            .select(&snippet_sel)
            .next()
            .map(|s| s.text().collect::<Vec<_>>().join(" ").trim().to_string())
            .unwrap_or_default();

        out.push(SearchResult {
            title,
            url,
            snippet,
        });
        if out.len() >= n {
            break;
        }
    }
    out
}

/// DDG returns links like `//duckduckgo.com/l/?uddg=<ENCODED_URL>&rut=...`.
/// Unwrap back to the real URL when we can; leave intact otherwise.
fn unwrap_ddg_redirect(href: &str) -> String {
    let normalized = if let Some(rest) = href.strip_prefix("//") {
        format!("https://{}", rest)
    } else {
        href.to_string()
    };
    if let Some(idx) = normalized.find("uddg=") {
        let tail = &normalized[idx + 5..];
        let encoded = tail.split('&').next().unwrap_or(tail);
        if let Ok(decoded) = urlencoding::decode(encoded) {
            return decoded.into_owned();
        }
    }
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unwraps_ddg_redirect() {
        let href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpath&rut=abc";
        assert_eq!(unwrap_ddg_redirect(href), "https://example.com/path");
    }

    #[test]
    fn passes_through_direct_url() {
        assert_eq!(
            unwrap_ddg_redirect("https://example.com/"),
            "https://example.com/"
        );
    }

    // Live smoke test — disabled by default because it hits the real network
    // and DDG's HTML can drift. Run with: cargo test live_ddg_smoke -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn live_ddg_smoke() {
        let client = reqwest::Client::builder()
            .user_agent(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
                 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )
            .timeout(std::time::Duration::from_secs(15))
            .build()
            .unwrap();
        let results = do_search(&client, "rust programming language", 5)
            .await
            .expect("search");
        assert!(!results.is_empty(), "expected at least one result");
        for r in &results {
            println!("- {}\n  {}\n  {}", r.title, r.url, r.snippet);
            assert!(!r.title.is_empty());
            assert!(r.url.starts_with("http"));
        }
    }
}
