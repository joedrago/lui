// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

//! Harness machinery: teach external tools (opencode, pi, …) how to reach
//! lui's llama-server.
//!
//! Every harness ships three things in its own file:
//!   - `pub const HARNESS: Harness` — metadata + fn pointers (the spec).
//!   - `fn apply(root, eff, inputs)` — the CST-surgical edits unique to this
//!     harness's config format.
//!   - `fn needs_backup(text)` — "is this a first-touch situation?"
//!
//! Everything else — parsing, serializing, atomic write, `.luibackup`,
//! SSH probe/write, skill file management, the SKILL.md body — lives here
//! and is shared. Adding a harness is a `src/harness/<name>.rs` plus one
//! entry in `HARNESSES`.

pub mod opencode;
pub mod pi;

use std::path::PathBuf;

use jsonc_parser::cst::{CstInputValue, CstObject, CstRootNode};
use jsonc_parser::ParseOptions;

use crate::settings::store::Effective;
use crate::ssh_tunnel::{ssh_run, SshTarget};

/// Where a harness's config file lives. Candidates are tried in order;
/// the first existing file wins, and a fresh install creates the first
/// candidate's filename.
pub struct ConfigFile {
    /// Home-relative directory, without leading `~/`. E.g. `.config/opencode`.
    pub dir: &'static str,
    /// Filenames tried in order. First entry is the default when no candidate exists.
    pub candidates: &'static [&'static str],
}

/// Universal inputs supplied to every harness apply. Harness-specific
/// settings are read by `apply` from `&Effective`.
pub struct HarnessInputs {
    pub model_name: String,
    /// OpenAI-shaped URL including `/v1`.
    pub base_url: String,
    pub ctx_size: u32,
    pub web_port: u16,
    pub websearch: bool,
}

/// The surgical CST edits unique to a harness: receives the parsed root
/// object, the full Effective view for reading harness-specific settings,
/// and the universal inputs.
pub type ApplyFn = fn(&CstObject, &Effective, &HarnessInputs);

/// Optional remote preflight check. If a harness provides one, we run
/// it before any SSH writes — typically to verify the harness's own
/// binary is installed on the remote.
pub type PreflightSshFn = fn(&SshTarget) -> Result<(), String>;

/// One declared harness. `pub const HARNESS: Harness = …` in each module
/// plus a reference in `HARNESSES` is enough — registry, CLI, local
/// apply, and SSH apply all find it by iteration.
pub struct Harness {
    pub name: &'static str,
    pub setting_name: &'static str, // "harness_opencode"
    pub flag_long: &'static str,    // "harness-opencode"
    pub default_on: bool,
    pub help: &'static [&'static str],
    pub config: ConfigFile,
    pub apply: ApplyFn,
    pub needs_backup: fn(&str) -> bool,
    pub preflight_ssh: Option<PreflightSshFn>,
}

/// Ordered list of every declared harness. Registry walks this to create
/// one `harness_X` bool per harness; main / ssh_tunnel iterate it to drive
/// updates.
pub const HARNESSES: &[&Harness] = &[&opencode::HARNESS, &pi::HARNESS];

// Tiny `CstInputValue` builders shared by per-harness `apply` fns.

pub fn s(v: impl Into<String>) -> CstInputValue {
    CstInputValue::String(v.into())
}
pub fn b(v: bool) -> CstInputValue {
    CstInputValue::Bool(v)
}
pub fn n(v: u32) -> CstInputValue {
    CstInputValue::from(v)
}
pub fn arr(v: Vec<CstInputValue>) -> CstInputValue {
    CstInputValue::Array(v)
}
pub fn obj<I: IntoIterator<Item = (&'static str, CstInputValue)>>(props: I) -> CstInputValue {
    CstInputValue::Object(
        props
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    )
}

/// Pick the local config file path for a harness: first existing
/// candidate, else the first declared candidate (creating it on save).
pub fn local_config_path(cf: &ConfigFile) -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let dir = home.join(cf.dir);
    for name in cf.candidates {
        let p = dir.join(name);
        if p.exists() {
            return p;
        }
    }
    dir.join(cf.candidates[0])
}

/// Skill directory: `<harness dir>/skills/lui-web-search`. Matches what
/// both opencode and pi expect; parameterized so harnesses don't each
/// hard-code their own path.
pub fn local_skill_dir(cf: &ConfigFile) -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(cf.dir).join("skills").join("lui-web-search")
}

/// Apply a harness locally: parse its config, let the harness edit the
/// CST, serialize, atomic-write with first-touch backup, then write (or
/// remove) the websearch SKILL.md.
pub fn apply_local(harness: &Harness, eff: &Effective, inputs: &HarnessInputs) {
    let path = local_config_path(&harness.config);
    let existing = std::fs::read_to_string(&path).unwrap_or_default();
    let should_backup = (harness.needs_backup)(&existing);
    let new_text = match transform(harness, &existing, eff, inputs) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "Warning: failed to update {} config {}: {}",
                harness.name,
                path.display(),
                e
            );
            return;
        }
    };

    if new_text != existing {
        if should_backup {
            let mut backup = path.as_os_str().to_os_string();
            backup.push(".luibackup");
            let backup = PathBuf::from(backup);
            if !backup.exists() {
                let _ = std::fs::write(&backup, &existing);
            }
        }
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        // Atomic write: temp file + rename. Rename is atomic on POSIX when
        // source and dest share a filesystem, which they do (same dir).
        let mut tmp = path.as_os_str().to_os_string();
        tmp.push(".luitmp");
        let tmp_path = PathBuf::from(tmp);
        if let Err(e) = std::fs::write(&tmp_path, &new_text) {
            eprintln!("Warning: failed to write {}: {}", tmp_path.display(), e);
            return;
        }
        if let Err(e) = std::fs::rename(&tmp_path, &path) {
            eprintln!(
                "Warning: failed to rename {} -> {}: {}",
                tmp_path.display(),
                path.display(),
                e
            );
            let _ = std::fs::remove_file(&tmp_path);
        }
    }

    let dir = local_skill_dir(&harness.config);
    let skill_path = dir.join("SKILL.md");
    if !inputs.websearch {
        if skill_path.exists() {
            let _ = std::fs::remove_file(&skill_path);
        }
        if dir.exists() {
            let _ = std::fs::remove_dir(&dir);
        }
        return;
    }
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("Warning: failed to create {}: {}", dir.display(), e);
        return;
    }
    if let Err(e) = std::fs::write(&skill_path, render_websearch_skill(inputs.web_port)) {
        eprintln!("Warning: failed to write {}: {}", skill_path.display(), e);
    }
}

/// Empty / non-object input is treated as `{}` so a fresh or corrupt
/// file still gets the lui block written rather than failing the whole
/// update.
fn transform(
    harness: &Harness,
    existing: &str,
    eff: &Effective,
    inputs: &HarnessInputs,
) -> Result<String, String> {
    let source = if existing.trim().is_empty() {
        "{}".to_string()
    } else {
        existing.to_string()
    };
    let root = CstRootNode::parse(&source, &ParseOptions::default())
        .map_err(|e| format!("parse: {}", e))?;
    let root_obj = root.object_value_or_set();
    (harness.apply)(&root_obj, eff, inputs);
    Ok(root.to_string())
}

/// Apply a harness over SSH. Mirrors `apply_local` but every file op
/// goes through `ssh_run`.
pub fn apply_remote(
    harness: &Harness,
    target: &SshTarget,
    remote_web_port: u16,
    inputs: &HarnessInputs,
    eff: &Effective,
) -> Result<(), String> {
    if let Some(preflight) = harness.preflight_ssh {
        preflight(target)?;
    }

    let (basename, existing) = fetch_remote(target, &harness.config)?;
    if (harness.needs_backup)(&existing) {
        // `cp -n` is POSIX "no-clobber" — preserves any prior backup.
        // `|| true` swallows failures if the source isn't there at all.
        let cmd = format!(
            "cp -n ~/{dir}/{base} ~/{dir}/{base}.luibackup 2>/dev/null || true",
            dir = harness.config.dir,
            base = basename
        );
        let _ = ssh_run(target, &[&cmd], None);
    }

    let contents = transform(harness, &existing, eff, inputs)?;
    let write_cmd = format!(
        "mkdir -p ~/{dir} && cat > ~/{dir}/{base}",
        dir = harness.config.dir,
        base = basename
    );
    ssh_run(target, &[&write_cmd], Some(contents.as_bytes()))?;

    let skill_dir = format!("~/{}/skills/lui-web-search", harness.config.dir);
    let skill_path = format!("{}/SKILL.md", skill_dir);
    if !inputs.websearch {
        let _ = ssh_run(
            target,
            &[&format!(
                "rm -f {} && rmdir {} 2>/dev/null || true",
                skill_path, skill_dir
            )],
            None,
        );
        return Ok(());
    }
    let body = render_websearch_skill(remote_web_port);
    ssh_run(
        target,
        &[&format!("mkdir -p {} && cat > {}", skill_dir, skill_path)],
        Some(body.as_bytes()),
    )?;

    Ok(())
}

/// Probe the remote for an existing config file by candidate order.
/// First line of stdout is the matching basename (or `MISSING`);
/// everything after the first newline is the file contents. Fresh-machine
/// fallback is the first declared candidate.
fn fetch_remote(target: &SshTarget, cf: &ConfigFile) -> Result<(String, String), String> {
    let mut probe = String::from("set -eu; ");
    for name in cf.candidates {
        probe.push_str(&format!(
            "if [ -f ~/{dir}/{name} ]; then echo {name}; cat ~/{dir}/{name}; exit 0; fi; ",
            dir = cf.dir,
            name = name,
        ));
    }
    probe.push_str("echo MISSING");
    let out = ssh_run(target, &[&probe], None)?;
    let (first_line, rest) = match out.split_once('\n') {
        Some(pair) => pair,
        None => (out.as_str(), ""),
    };
    let tag = first_line.trim();
    let basename = if tag == "MISSING" {
        cf.candidates[0].to_string()
    } else {
        tag.to_string()
    };
    Ok((basename, rest.to_string()))
}

/// The lui-web-search SKILL.md body, parameterized on the port where
/// `/bsearch` is reachable. Shared by every harness since the skill
/// instructions don't depend on which harness loads it.
pub fn render_websearch_skill(port: u16) -> String {
    format!(
        r#"---
name: lui-web-search
description: Browser-mediated web search for local/self-hosted models that have no native web-search tool. If you already have a native search tool (e.g. web_search, browse), use that instead — this skill requires the user to click a browser bookmarklet and blocks on their interaction. Returns JSON results with title, url, and snippet.
license: BSD-2-Clause
---

# lui-web-search

> **When to use this skill.** Only when you have no native web-search tool
> available. If you have one (Anthropic's `web_search`, OpenAI's browse, a
> native `search` tool, etc.), prefer that — it's faster, doesn't require a
> browser tab, and doesn't block on user interaction. This skill is the
> fallback for local/self-hosted models (llama.cpp, etc.) whose provider
> doesn't expose web search.

lui's search endpoint opens a Google search tab in the user's real
browser. The user clicks a one-time-installed `lui-grab` bookmarklet on
the resulting page; the bookmarklet POSTs the rendered results back to
lui, which returns them to you.

## Endpoint

```
GET http://127.0.0.1:{port}/bsearch?q=<URL-ENCODED QUERY>
```

- `q` (required): the search query. URL-encode it.

The request **blocks for up to 120 seconds** while waiting for the
user to click the bookmarklet. On timeout you'll get HTTP 504.

## Response

JSON object:

```json
{{
  "results": [
    {{"title": "...", "url": "https://...", "snippet": "..."}}
  ],
  "warnings": ["..."]
}}
```

`results` is always present. `warnings` is present only when the
bookmarklet had something to tell you — for example, when Google's
CSS class names rotated and the bookmarklet had to fall back to a
structural selector to find results. **If `warnings` is non-empty,
surface each warning verbatim to the user** at the end of your reply
(under a short heading like "Note from lui-grab:"), on top of your
normal answer. The user is the only one who can act on it (usually
by updating lui).

An HTTP 504 means the user did not click the bookmarklet in time
(probably they were AFK or the browser tab got buried). Other 4xx/5xx
or an empty `results` array means the search failed — say so plainly
rather than fabricating answers.

## How to invoke

```sh
curl -sG 'http://127.0.0.1:{port}/bsearch' \
  --data-urlencode 'q=rust async traits 2026'
```

On Windows (PowerShell):

```powershell
$q = [uri]::EscapeDataString('rust async traits 2026')
curl.exe -s "http://127.0.0.1:{port}/bsearch?q=$q"
```

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
> `http://127.0.0.1:{port}/setup` and drag it to your bookmarks bar
> first. (This URL is also shown in the lui status panel.)"

Then call `/bsearch` and wait. If the call returns HTTP 504, the user
didn't click in time — most likely they don't have the bookmarklet
installed yet. Stop, point them at the setup page, wait for them to
say it's ready, then retry.

Be deliberate about when to search:
- One search at a time. Do not fire parallel searches.
- Pick the best query first instead of iterating with small variations.
- Don't search for things you already know.
"#
    )
}

