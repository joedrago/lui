use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LuiConfig {
    pub server: ServerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub hf_repo: String,
    #[serde(default = "default_ctx_size")]
    pub ctx_size: u32,
    #[serde(default = "default_gpu_layers")]
    pub gpu_layers: i32,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default)]
    pub extra_args: Vec<String>,
}

fn default_ctx_size() -> u32 {
    0
}
fn default_gpu_layers() -> i32 {
    -1
}
fn default_port() -> u16 {
    8080
}
fn default_host() -> String {
    "127.0.0.1".to_string()
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            model: String::new(),
            hf_repo: String::new(),
            ctx_size: default_ctx_size(),
            gpu_layers: default_gpu_layers(),
            port: default_port(),
            host: default_host(),
            extra_args: Vec::new(),
        }
    }
}

impl Default for LuiConfig {
    fn default() -> Self {
        LuiConfig {
            server: ServerConfig::default(),
        }
    }
}

fn config_path() -> PathBuf {
    let config_dir = dirs::config_dir().unwrap_or_else(|| PathBuf::from("."));
    config_dir.join("lui.toml")
}

pub fn load_config() -> LuiConfig {
    let path = config_path();
    if path.exists() {
        match std::fs::read_to_string(&path) {
            Ok(contents) => match toml::from_str(&contents) {
                Ok(config) => return config,
                Err(e) => eprintln!("Warning: failed to parse {}: {}", path.display(), e),
            },
            Err(e) => eprintln!("Warning: failed to read {}: {}", path.display(), e),
        }
    }
    LuiConfig::default()
}

pub fn save_config(config: &LuiConfig) {
    let path = config_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match toml::to_string_pretty(config) {
        Ok(contents) => {
            if let Err(e) = std::fs::write(&path, contents) {
                eprintln!("Warning: failed to write {}: {}", path.display(), e);
            }
        }
        Err(e) => eprintln!("Warning: failed to serialize config: {}", e),
    }
}

/// Derive a short model name for opencode from the hf_repo or model path.
/// e.g. "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M" -> "Qwen2.5-Coder-7B-Instruct"
/// e.g. "/path/to/qwen2.5-coder-7b-instruct-q4_k_m.gguf" -> "qwen2.5-coder-7b-instruct-q4_k_m"
pub fn derive_model_name(config: &ServerConfig) -> String {
    if !config.hf_repo.is_empty() {
        let repo = &config.hf_repo;
        // Take the part after '/' and before ':'
        let name = repo.split('/').last().unwrap_or(repo);
        let name = name.split(':').next().unwrap_or(name);
        // Strip common suffixes like -GGUF
        let name = name.strip_suffix("-GGUF").unwrap_or(name);
        name.to_string()
    } else if !config.model.is_empty() {
        // Extract filename without extension
        PathBuf::from(&config.model)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string()
    } else {
        "unknown".to_string()
    }
}

fn opencode_config_path() -> PathBuf {
    // opencode uses ~/.config/opencode/opencode.json (XDG-style), not ~/Library/Application Support/
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".config").join("opencode").join("opencode.json")
}

pub fn update_opencode_config(config: &ServerConfig) {
    let path = opencode_config_path();
    let model_name = derive_model_name(config);

    // Read existing config or start with empty object
    let mut json: serde_json::Value = if path.exists() {
        match std::fs::read_to_string(&path) {
            Ok(contents) => serde_json::from_str(&contents).unwrap_or(serde_json::json!({})),
            Err(_) => serde_json::json!({}),
        }
    } else {
        serde_json::json!({})
    };

    let obj = json.as_object_mut().unwrap();

    // Set top-level model
    obj.insert(
        "model".to_string(),
        serde_json::json!(format!("lui/{}", model_name)),
    );

    // Ensure provider section exists
    if !obj.contains_key("provider") {
        obj.insert("provider".to_string(), serde_json::json!({}));
    }

    let providers = obj.get_mut("provider").unwrap().as_object_mut().unwrap();

    // Create/update the "lui" provider
    providers.insert(
        "lui".to_string(),
        serde_json::json!({
            "name": "lui",
            "npm": "@ai-sdk/openai-compatible",
            "options": {
                "baseURL": format!("http://localhost:{}/v1", config.port),
                "toolParser": [
                    { "type": "raw-function-call" },
                    { "type": "json" }
                ]
            },
            "models": {
                &model_name: {
                    "name": &model_name,
                    "supportsToolCalls": true
                }
            }
        }),
    );

    // Write back
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(contents) = serde_json::to_string_pretty(&json) {
        if let Err(e) = std::fs::write(&path, contents) {
            eprintln!("Warning: failed to write {}: {}", path.display(), e);
        }
    }
}
