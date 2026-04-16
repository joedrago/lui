// Copyright 2026 Joe Drago. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

use std::collections::HashMap;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

/// Reads select metadata from a GGUF file header without loading the full file.
/// Returns a map of key -> string representation of value.
pub fn read_gguf_metadata(path: &Path) -> io::Result<HashMap<String, String>> {
    let mut f = std::fs::File::open(path)?;
    let mut result = HashMap::new();

    // Magic: "GGUF" (4 bytes)
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a GGUF file",
        ));
    }

    // Version: u32
    let version = read_u32(&mut f)?;
    if version < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported GGUF version",
        ));
    }

    // Tensor count: u64
    let _tensor_count = read_u64(&mut f)?;

    // Metadata KV count: u64
    let kv_count = read_u64(&mut f)?;

    // Cap iteration to avoid reading huge files
    let max_kvs = kv_count.min(100);

    for _ in 0..max_kvs {
        let key = match read_gguf_string(&mut f) {
            Ok(k) => k,
            Err(_) => break,
        };

        let value_type = match read_u32(&mut f) {
            Ok(v) => v,
            Err(_) => break,
        };

        // We only care about certain keys
        let dominated = key.ends_with(".context_length")
            || key == "general.name"
            || key == "general.size_label"
            || key == "general.architecture";

        match value_type {
            // UINT8 = 0
            0 => {
                let v = read_u8(&mut f)?;
                if dominated {
                    result.insert(key, v.to_string());
                }
            }
            // INT8 = 1
            1 => {
                let mut buf = [0u8; 1];
                f.read_exact(&mut buf)?;
                if dominated {
                    result.insert(key, (buf[0] as i8).to_string());
                }
            }
            // UINT16 = 2
            2 => {
                let v = read_u16(&mut f)?;
                if dominated {
                    result.insert(key, v.to_string());
                }
            }
            // INT16 = 3
            3 => {
                let mut buf = [0u8; 2];
                f.read_exact(&mut buf)?;
                if dominated {
                    result.insert(key, i16::from_le_bytes(buf).to_string());
                }
            }
            // UINT32 = 4
            4 => {
                let v = read_u32(&mut f)?;
                if dominated {
                    result.insert(key, v.to_string());
                }
            }
            // INT32 = 5
            5 => {
                let mut buf = [0u8; 4];
                f.read_exact(&mut buf)?;
                if dominated {
                    result.insert(key, i32::from_le_bytes(buf).to_string());
                }
            }
            // FLOAT32 = 6
            6 => {
                let mut buf = [0u8; 4];
                f.read_exact(&mut buf)?;
                if dominated {
                    result.insert(key, format!("{}", f32::from_le_bytes(buf)));
                }
            }
            // BOOL = 7
            7 => {
                let v = read_u8(&mut f)?;
                if dominated {
                    result.insert(key, if v != 0 { "true" } else { "false" }.to_string());
                }
            }
            // STRING = 8
            8 => {
                let v = read_gguf_string(&mut f)?;
                if dominated {
                    result.insert(key, v);
                }
            }
            // ARRAY = 9
            9 => {
                // Skip arrays - read type and count, then skip elements
                let elem_type = read_u32(&mut f)?;
                let count = read_u64(&mut f)?;
                skip_array_elements(&mut f, elem_type, count)?;
            }
            // UINT64 = 10
            10 => {
                let v = read_u64(&mut f)?;
                if dominated {
                    result.insert(key, v.to_string());
                }
            }
            // INT64 = 11
            11 => {
                let mut buf = [0u8; 8];
                f.read_exact(&mut buf)?;
                if dominated {
                    result.insert(key, i64::from_le_bytes(buf).to_string());
                }
            }
            // FLOAT64 = 12
            12 => {
                let mut buf = [0u8; 8];
                f.read_exact(&mut buf)?;
                if dominated {
                    result.insert(key, format!("{}", f64::from_le_bytes(buf)));
                }
            }
            // Unknown type - bail
            _ => break,
        }
    }

    Ok(result)
}

fn read_u8(f: &mut std::fs::File) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    f.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u16(f: &mut std::fs::File) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    f.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32(f: &mut std::fs::File) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(f: &mut std::fs::File) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_gguf_string(f: &mut std::fs::File) -> io::Result<String> {
    let len = read_u64(f)? as usize;
    if len > 1_000_000 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "string too long",
        ));
    }
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn skip_array_elements(f: &mut std::fs::File, elem_type: u32, count: u64) -> io::Result<()> {
    let elem_size = match elem_type {
        0 | 1 | 7 => 1u64, // u8, i8, bool
        2 | 3 => 2,        // u16, i16
        4 | 5 | 6 => 4,    // u32, i32, f32
        10 | 11 | 12 => 8, // u64, i64, f64
        8 => {
            // Array of strings - read each one
            for _ in 0..count {
                let _ = read_gguf_string(f)?;
            }
            return Ok(());
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported array element type",
            ));
        }
    };
    let skip = elem_size * count;
    f.seek(SeekFrom::Current(skip as i64))?;
    Ok(())
}
