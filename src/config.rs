use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct Config {
    pub port: u16,
    #[serde(default)]
    pub host: String,
    #[serde(default)]
    pub debug: bool,
    pub auth_dir: String,
    #[serde(default)]
    #[allow(dead_code)]
    pub api_keys: Vec<String>,
    #[serde(default)]
    pub openai_compatibility: Vec<OpenAICompatProvider>,
    #[serde(default)]
    #[allow(dead_code)]
    pub request_retry: u32,
    #[serde(default = "default_true")]
    pub logging_to_file: bool,
    #[serde(default = "default_logs_size")]
    pub logs_max_total_size_mb: u64,
    #[serde(default = "default_log_level")]
    pub log_level: String,
    #[serde(default = "default_log_dir")]
    pub log_dir: String,
    /// When true, logs detailed per-turn debug info (instructions length,
    /// preview, tools count, input items) for every WebSocket request.
    /// Useful for diagnosing hallucination caused by missing system prompt.
    #[serde(default)]
    pub log_turn_debug: bool,
    /// Alias → backend model name map, populated once at load time.
    /// Used by `resolve_model()` to avoid per-request Vec allocation.
    #[serde(skip)]
    pub model_lookup: HashMap<String, String>,
}

fn default_true() -> bool {
    true
}
fn default_logs_size() -> u64 {
    10
}
fn default_log_level() -> String {
    "info".to_string()
}
fn default_log_dir() -> String {
    "logs".to_string()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct OpenAICompatProvider {
    pub name: String,
    #[allow(dead_code)]
    pub prefix: String,
    pub base_url: String,
    pub models: Vec<OpenAICompatModel>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAICompatModel {
    pub name: String,
    pub alias: String,
}

impl Config {
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let raw = std::fs::read_to_string(path)?;
        let mut cfg: Config = serde_yaml::from_str(&raw)?;
        if cfg.host.is_empty() {
            cfg.host = "0.0.0.0".to_string();
        }
        if cfg.auth_dir.starts_with('~') {
            cfg.auth_dir = shellexpand::tilde(&cfg.auth_dir).to_string();
        }
        if cfg.log_dir.starts_with('~') {
            cfg.log_dir = shellexpand::tilde(&cfg.log_dir).to_string();
        }
        if cfg.debug {
            cfg.log_level = "debug".to_string();
        }
        // Build the alias → backend model lookup map once.
        let model_pairs: Vec<(String, String)> = cfg
            .ericai_provider()
            .map(|p| {
                p.models
                    .iter()
                    .map(|m| (m.alias.clone(), m.name.clone()))
                    .collect()
            })
            .unwrap_or_default();
        for (alias, name) in model_pairs {
            cfg.model_lookup.insert(alias, name);
        }
        Ok(cfg)
    }

    pub fn ericai_provider(&self) -> Option<&OpenAICompatProvider> {
        self.openai_compatibility
            .iter()
            .find(|p| p.name == "ericai")
    }

    pub fn all_models(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        if let Some(p) = self.ericai_provider() {
            for m in &p.models {
                out.push((m.alias.clone(), m.name.clone()));
            }
        }
        out
    }

    /// Resolve a model alias to its backend model ID using the pre-built
    /// lookup map. Falls back to stripping an `ericai/` prefix, then to
    /// the alias itself. This replaces the per-request `all_models()` +
    /// `.find()` pattern on the hot path.
    pub fn resolve_model(&self, alias: &str) -> String {
        if let Some(name) = self.model_lookup.get(alias) {
            return name.clone();
        }
        if let Some(stripped) = alias.strip_prefix("ericai/") {
            return stripped.to_string();
        }
        alias.to_string()
    }
}

// Minimal shellexpand
mod shellexpand {
    pub fn tilde(s: &str) -> String {
        if let Ok(home) = std::env::var("HOME") {
            if s == "~" {
                return home;
            }
            if let Some(rest) = s.strip_prefix("~/") {
                return format!("{}/{}", home, rest);
            }
        }
        s.to_string()
    }
}
