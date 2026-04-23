// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

//! Auto-generated `--help` output. Walks the `Registry`, emits one row per
//! flag-bearing setting, and splices in the static narrative the registry
//! can't represent on its own (USAGE header, section preambles,
//! pass-through footer).
//!
//! Alignment is computed from the rendered flag signatures, so adding a
//! longer flag doesn't require retuning a magic column number — the
//! column shifts automatically.

use super::registry::{Registry, SECTIONS};
use super::setting::Setting;
use super::value::ValueKind;

/// Produce the complete `--help` text as a single allocated string. Pure
/// function of the registry — no I/O, safe to call from tests.
pub fn emit_help(reg: &Registry) -> String {
    let mut out = String::new();
    out.push_str(
        "lui — a friendly TUI wrapper for llama.cpp's llama-server\n\
         \n\
         USAGE:\n    \
         lui [OPTIONS] [NAME] [-- <EXTRA_LLAMA_ARGS>...]\n\
         \n",
    );

    // Build rendered rows once so we can compute a single alignment column
    // across every section. This matches today's behavior where every
    // description starts at the same column regardless of section.
    let rows: Vec<Row> = rows_for_help(reg);
    let sig_w = rows.iter().map(|r| r.signature.len()).max().unwrap_or(0);

    for section in SECTIONS {
        let in_section: Vec<&Row> = rows.iter().filter(|r| r.section == section.name).collect();
        if in_section.is_empty()
            && section.extra_rows.is_empty()
            && section.postamble.is_empty()
        {
            continue;
        }
        out.push_str(section.title);
        out.push('\n');
        if !section.preamble.is_empty() {
            out.push_str(section.preamble);
        }
        for r in in_section {
            // 4-space section indent + signature, then padding + description.
            out.push_str("    ");
            out.push_str(&r.signature);
            let gap = sig_w + 3 - r.signature.len();
            for _ in 0..gap {
                out.push(' ');
            }
            out.push_str(&r.description);
            out.push('\n');
            for c in &r.continuation {
                // Continuation lines align under the description column:
                // 4-space indent + signature width + 3-space gap.
                for _ in 0..(4 + sig_w + 3) {
                    out.push(' ');
                }
                out.push_str(c);
                out.push('\n');
            }
        }
        for er in section.extra_rows {
            // Synthetic aligned row: 4-space indent + 4-space short-flag
            // slot + signature + pad + description. Matches `emit_row`'s
            // column math so `<NAME>` sits under the flag column.
            out.push_str("        ");
            out.push_str(er.signature);
            let gap = (sig_w + 3).saturating_sub(er.signature.len() + 4);
            for _ in 0..gap {
                out.push(' ');
            }
            out.push_str(er.description);
            out.push('\n');
        }
        if !section.postamble.is_empty() {
            out.push_str(section.postamble);
        }
        out.push('\n');
    }

    out.push_str(
        "\nPass-through (`--`):\n    \
         Everything after `--` is appended to llama-server's argv. The pass-through\n    \
         list is scoped too: `lui --this -- --some-llama-flag` appends to the active\n    \
         model's extra_args; without --this, it appends to the global list.\n",
    );

    out
}

/// One formatted help row. A single setting may produce two rows (bool
/// with no_form pair — `--X` and `--no-X`).
struct Row {
    section: &'static str,
    signature: String,
    description: String,
    // Continuation lines to emit below this row, indented to `pad`.
    continuation: Vec<String>,
}

/// Walk the registry once and produce one (or two) rows per flag-bearing
/// setting. Storage-only entries (no short, no long) are skipped.
fn rows_for_help(reg: &Registry) -> Vec<Row> {
    let mut rows: Vec<Row> = Vec::new();
    for s in reg.iter_by_section() {
        if s.short.is_none() && s.long.is_none() {
            continue; // hidden: stored but not surfaced on the CLI
        }
        if s.kind == ValueKind::Bool && s.has_no_form() {
            // Two rows, one per form. help[0] describes the positive form,
            // help[1] (or a stock fallback) describes the `--no-X` form.
            // Neither row carries continuation text — bool flags with a
            // no-form pair are expected to fit on one line each.
            let pos_desc = s.help.first().copied().unwrap_or_default().to_string();
            let neg_desc = match s.help.get(1) {
                Some(line) => (*line).to_string(),
                None => format!("Inverse of --{}", s.long.unwrap()),
            };
            rows.push(Row {
                section: s.section,
                signature: signature_positive(s),
                description: pos_desc,
                continuation: Vec::new(),
            });
            // `--no-<long>` row. Aliases and placeholders are dropped —
            // the inverse form is always the primary long with `no-`
            // prepended. The 4-space prefix stands in for the short-flag
            // slot regardless of whether the positive form has a short;
            // the `--no-` form never claims the short.
            let long = s.long.expect("has_no_form implies long is Some");
            rows.push(Row {
                section: s.section,
                signature: format!("    --no-{}", long),
                description: neg_desc,
                continuation: Vec::new(),
            });
        } else {
            // Single-form setting. help[0] is the description; help[1..] is
            // multi-line continuation that hangs under the description
            // column.
            let desc = s.help.first().copied().unwrap_or_default().to_string();
            let cont: Vec<String> = if s.help.len() > 1 {
                s.help[1..].iter().map(|l| (*l).to_string()).collect()
            } else {
                Vec::new()
            };
            rows.push(Row {
                section: s.section,
                signature: signature_positive(s),
                description: desc,
                continuation: cont,
            });
        }
    }
    rows
}

/// Build the signature string for the primary (positive) flag form.
/// Examples: `-c, --ctx-size <N>` · `    --temp <F>` · `-h, --help` ·
/// `    --ngl, --gpu-layers <N>`.
fn signature_positive(s: &Setting) -> String {
    let mut out = String::new();
    match s.short {
        Some(c) => {
            out.push('-');
            out.push(c);
            if s.long.is_some() || !s.long_aliases.is_empty() {
                out.push_str(", ");
            }
        }
        None => out.push_str("    "),
    }
    let mut longs: Vec<String> = Vec::new();
    if let Some(l) = s.long {
        longs.push(format!("--{}", l));
    }
    for alias in s.long_aliases {
        longs.push(format!("--{}", alias));
    }
    out.push_str(&longs.join(", "));
    if s.takes_value() && !s.placeholder.is_empty() {
        out.push_str(" <");
        out.push_str(s.placeholder);
        out.push('>');
    }
    out
}

