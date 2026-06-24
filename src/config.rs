use serde::Deserialize;

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
}

fn default_true() -> bool { true }
fn default_logs_size() -> u64 { 10 }
fn default_log_level() -> String { "info".to_string() }
fn default_log_dir() -> String { "logs".to_string() }

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
        Ok(cfg)
    }

    pub fn ericai_provider(&self) -> Option<&OpenAICompatProvider> {
        self.openai_compatibility.iter().find(|p| p.name == "ericai")
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
}

// Minimal shellexpand
mod shellexpand {
    pub fn tilde(s: &str) -> String {
        if let Some(home) = std::env::var("HOME").ok() {
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
