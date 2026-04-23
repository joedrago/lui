// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

//! The single centralized settings registry.
//!
//! Every command-line option and persisted config knob is declared exactly
//! once in `registry::declare_all_settings`. All downstream code — help
//! output, argv parser, TOML IO, llama-server passthrough, UI payload —
//! walks the `Registry` rather than naming settings individually.

pub mod help;
pub mod migrate;
pub mod registry;
pub mod setting;
pub mod store;
pub mod value;

#[allow(unused_imports)]
pub use registry::{ExtraRow, LongLookup, Registry, Section, SECTIONS};
#[allow(unused_imports)]
pub use setting::{PassthroughMode, Scope, Setting};
#[allow(unused_imports)]
pub use store::{Config, Effective, Store};
#[allow(unused_imports)]
pub use value::{Value, ValueKind};
