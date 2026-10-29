use serde::Deserialize;
use std::collections::HashMap;
use std::collections::HashSet;

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
    pub api_keys: Vec<String>,
    /// Maximum requests per minute per client IP. 0 = unlimited.
    #[serde(default = "default_rate_limit")]
    pub rate_limit_per_minute: u32,
    /// Maximum concurrent in-flight requests per backend model before the proxy
    /// holds new requests (prevents upstream 429 storms). 0 = unlimited.
    #[serde(default = "default_max_concurrent_per_model")]
    pub max_concurrent_per_model: usize,
    /// Per-model override for max concurrent requests. Keyed by backend model ID
    /// (e.g. "zai-org/GLM-5.2-FP8": 1). If a model isn't listed here, falls
    /// back to max_concurrent_per_model.
    #[serde(default)]
    pub model_concurrency: HashMap<String, usize>,
    #[serde(default)]
    pub openai_compatibility: Vec<OpenAICompatProvider>,
    /// Retry budget for TRANSPORT-level upstream failures (reqwest never got
    /// response headers — connection dropped, DNS/TLS/socket blip), applied by
    /// send_upstream in main.rs. NOT for HTTP statuses: a 4xx/5xx from the
    /// backend is a real answer and re-sending it doubles the GPU work.
    /// Measured on this machine: ~1% of upstream attempts fail this way, and
    /// of those ~25% are followed within 2 minutes by an attempt that
    /// succeeds — i.e. the upstream was briefly unreachable, not down. The
    /// other ~75% have no next attempt at all (upstream genuinely down), where
    /// the retries cost only the delay below. 0 disables retrying.
    #[serde(default = "default_request_retry")]
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
    /// Optional Codex-style model catalog: `{"models":[{slug, context_window,
    /// input_modalities}]}`. Only needed for slugs the proxy does NOT declare
    /// itself — the provider blocks below are the source of truth for the proxy's
    /// own models, and Codex's catalog (its own `model_catalog_json` key points
    /// at `~/.codex/models.json`) only knows Anthropic-hosted slugs.
    /// Absent means no catalog: every window/modality comes from config.yaml.
    #[serde(default)]
    pub model_catalog: Option<String>,
    /// Alias → (provider index, backend model name) map, populated once at load
    /// time. Keyed by plain alias AND by `provider/alias`, so a client can name
    /// which machine serves a model when two providers carry the same family.
    #[serde(skip)]
    pub model_lookup: HashMap<String, UpstreamTarget>,
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
fn default_rate_limit() -> u32 {
    60
}
fn default_max_concurrent_per_model() -> usize {
    3
}
fn default_request_retry() -> u32 {
    3
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct OpenAICompatProvider {
    pub name: String,
    #[allow(dead_code)]
    pub prefix: String,
    pub base_url: String,
    /// Upstream key for THIS provider, sent as `Authorization: Bearer <key>` on
    /// every request forwarded to it. A llama-server started with `--api-key`
    /// rejects keyless requests, so a provider pointing at one must carry its
    /// key here. The client's own token is deliberately NOT forwarded: it
    /// authenticates the client against the proxy, not against the upstream.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Name of an environment variable holding this provider's upstream key.
    /// Resolved into `api_key` at load time, so a tracked config file never
    /// carries the secret itself.
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Logit bias applied to every model this provider serves, regardless of
    /// the name the client called it by. The sampler guards are a property of
    /// the served model's tokenizer, i.e. of the endpoint: the strata server
    /// is reached as "strata-ubuntu", "strata-proxy" and the declared model
    /// name alike (undeclared names fall back to the default provider with the
    /// key echoed as model ID), so a per-model guard would miss most traffic.
    /// A model-level `logit_bias` entry overrides individual tokens of this
    /// map.
    #[serde(default)]
    pub logit_bias: Option<HashMap<String, f64>>,
    pub models: Vec<OpenAICompatModel>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAICompatModel {
    pub name: String,
    pub alias: String,
    #[serde(default)]
    pub context_length: Option<u64>,
    /// Capability tags for routing (e.g. ["fast", "coding"]). Used by clients
    /// to resolve `tag:fast` / `tag:coding` to concrete models when the proxy
    /// is in the path (the proxy strips rayllm_metadata from /v1/models).
    /// Without this, subagents fall back to the session's default model
    /// (often an expensive thinking model) because tag resolution fails.
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    /// Input modalities this backend model accepts (e.g. ["text"],
    /// ["text","image"]). Declared here rather than in the Codex model
    /// catalog because the proxy's own model list is the source of truth for
    /// its models — Codex's catalog only knows Anthropic-hosted slugs.
    /// Absent means text-only.
    #[serde(default)]
    pub input_modalities: Option<Vec<String>>,
    /// Per-model logit bias, merged OVER the provider's map (see
    /// `effective_logit_bias`). Token IDs are tokenizer-specific, so most
    /// guards belong on the provider; use this only when one aliased entry
    /// behind the same endpoint needs a different bias.
    #[serde(default)]
    pub logit_bias: Option<HashMap<String, f64>>,
}

/// Which provider should serve a requested model, and the backend model ID that
/// provider expects.
#[derive(Debug, Clone)]
pub struct UpstreamTarget {
    pub provider_idx: usize,
    pub backend_model: String,
}

impl Config {
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let raw = std::fs::read_to_string(path)?;
        let mut cfg: Config = serde_yaml::from_str(&raw)?;
        if cfg.host.is_empty() {
            cfg.host = "127.0.0.1".to_string();
        }
        if cfg.auth_dir.starts_with('~') {
            cfg.auth_dir = shellexpand::tilde(&cfg.auth_dir).to_string();
        }
        if cfg.log_dir.starts_with('~') {
            cfg.log_dir = shellexpand::tilde(&cfg.log_dir).to_string();
        }
        // Expand the catalog path here so main.rs can open it directly — the
        // `shellexpand` helper is private to this module. `tilde` is a no-op
        // for absolute paths.
        cfg.model_catalog = cfg.model_catalog.as_ref().map(|p| shellexpand::tilde(p).to_string());
        if cfg.debug {
            cfg.log_level = "debug".to_string();
        }
        // Build the alias → (provider, backend model) lookup map once, across
        // EVERY provider. This used to walk only the default provider, which
        // pinned the whole proxy to a single base URL. First provider declaring
        // a plain alias wins; the `provider/alias` form is always registered and
        // is how a client picks a specific machine.
        for (idx, p) in cfg.openai_compatibility.iter().enumerate() {
            for m in &p.models {
                let target = UpstreamTarget{ provider_idx: idx, backend_model: m.name.clone() };
                if !cfg.model_lookup.contains_key(&m.alias) {
                    cfg.model_lookup.insert(m.alias.clone(), target.clone());
                }
                let qualified = format!("{}/{}", p.name, m.alias);
                if !cfg.model_lookup.contains_key(&qualified) {
                    cfg.model_lookup.insert(qualified, target.clone());
                }
            }
        }
        // Resolve per-provider upstream keys from env so the tracked config
        // file never carries a secret. An explicit `api-key` still wins.
        for p in cfg.openai_compatibility.iter_mut() {
            if p.api_key.is_none() {
                if let Some(name) = p.api_key_env.clone() {
                    p.api_key = std::env::var(name).ok();
                }
            }
        }
        Ok(cfg)
    }

    pub fn provider(&self) -> Option<&OpenAICompatProvider> {
        self.openai_compatibility
            .iter()
            .find(|p| p.name == "default")
            .or_else(|| self.openai_compatibility.first())
    }

    /// Index of the fallback provider `provider()` picks, so `upstream_for()`
    /// can hand back a client index even for models the catalog doesn't list.
    pub fn default_provider_idx(&self) -> usize {
        self.openai_compatibility
            .iter()
            .position(|p| p.name == "default")
            .unwrap_or_else(|| 0)
    }

    /// Resolve a requested model key to the provider that should serve it.
    /// `key` may be a plain alias, a `provider/alias` qualified alias, or an
    /// already-resolved backend model ID (the summarize path resolves before
    /// calling). None means "nothing declares it" — callers fall back to the
    /// default provider.
    pub fn upstream_target(&self, key: &str) -> Option<UpstreamTarget> {
        if let Some(t) = self.model_lookup.get(key) {
            return Some(t.clone());
        }
        if let Some(stripped) = key.strip_prefix("default/") {
            if let Some(t) = self.model_lookup.get(stripped) {
                return Some(t.clone());
            }
        }
        for (idx, p) in self.openai_compatibility.iter().enumerate() {
            if p.models.iter().any(|m| m.name == key) {
                return Some(UpstreamTarget{ provider_idx: idx, backend_model: key.to_string() });
            }
        }
        None
    }

    /// Everything one upstream request needs, from a single lookup: which
    /// provider serves this model, that provider's base URL and optional key,
    /// and the backend model ID. Unknown keys fall back to the default provider
    /// and echo the key as the model ID — the same fallbacks resolve_model() has
    /// always had, so a model the catalog doesn't list still reaches the
    /// default machine unchanged.
    pub fn upstream_for(&self, key: &str) -> Upstream {
        let target = self.upstream_target(key);
        let provider = target.as_ref()
            .and_then(|t| self.openai_compatibility.get(t.provider_idx))
            .or_else(|| self.provider());
        let provider_idx = target.as_ref().map(|t| t.provider_idx).unwrap_or_else(|| self.default_provider_idx());
        let backend_model = match target {
            Some(t) => t.backend_model,
            None => {
                if let Some(stripped) = key.strip_prefix("default/") {
                    stripped.to_string()
                } else {
                    key.to_string()
                }
            }
        };
        let (base_url, api_key, provider_name) = match provider {
            Some(p) => (p.base_url.clone(), p.api_key.clone(), p.name.clone()),
            None => ("".to_string(), None, "default".to_string()),
        };
        Upstream {
            base_url,
            api_key,
            backend_model,
            provider_name,
            provider_idx,
        }
    }

    /// Get tags for a model alias (for /v1/models metadata response). Matches a
    /// plain alias, a `provider/alias` qualified alias, or a backend model name.
    pub fn model_tags(&self, alias: &str) -> Option<&[String]> {
        self.model_entry(alias).and_then(|m| m.tags.as_deref())
    }

    /// Declared context window for a model alias, from `context_length` in the
    /// provider block. Same matching rules as `model_tags`.
    pub fn model_context_window(&self, alias: &str) -> Option<u64> {
        self.model_entry(alias).and_then(|m| m.context_length)
    }

    /// Declared input modalities for a model alias. Same matching rules as
    /// `model_tags`; absent means text-only.
    pub fn model_input_modalities(&self, alias: &str) -> Option<&[String]> {
        self.model_entry(alias).and_then(|m| m.input_modalities.as_deref())
    }

    /// The logit bias a request for `key` actually runs with: the serving
    /// provider's map with the model entry's map merged over it (a model can
    /// sharpen or relax a single token of the provider guard; it cannot
    /// remove tokens — deleting an entry belongs in config.yaml). None when
    /// neither declares anything. `key` may be an alias, a qualified alias,
    /// a backend model ID, or an undeclared name (which routes to the default
    /// provider — that fallback is exactly how the strata server receives
    /// names like "strata-ubuntu", so the provider map still reaches them).
    pub fn effective_logit_bias(&self, key: &str) -> Option<HashMap<String, f64>> {
        let provider_idx = self
            .upstream_target(key)
            .map(|t| t.provider_idx)
            .unwrap_or_else(|| self.default_provider_idx());
        let provider_bias = self
            .openai_compatibility
            .get(provider_idx)
            .and_then(|p| p.logit_bias.as_ref());
        let model_bias = self.model_entry(key).and_then(|m| m.logit_bias.as_ref());
        match (provider_bias, model_bias) {
            (None, None) => None,
            (Some(p), None) => Some(p.clone()),
            (None, Some(m)) => Some(m.clone()),
            (Some(p), Some(m)) => {
                let mut merged = p.clone();
                merged.extend(m.iter().map(|(k, v)| (k.clone(), *v)));
                Some(merged)
            }
        }
    }

    /// The provider model entry a key resolves to: plain alias, `provider/alias`
    /// qualified alias, or backend model name (the summarize path resolves to a
    /// backend ID before looking metadata up).
    fn model_entry(&self, alias: &str) -> Option<&OpenAICompatModel> {
        for p in self.openai_compatibility.iter() {
            for m in &p.models {
                let qualified = format!("{}/{}", p.name, m.alias);
                if m.alias == alias || m.name == alias || qualified == alias {
                    return Some(m);
                }
            }
        }
        None
    }

    /// Catalog for /v1/models: every model across every provider. Plain aliases
    /// are listed once (first provider declaring one wins); non-default
    /// providers also get a `provider/alias` entry so a client can name the
    /// machine explicitly.
    pub fn all_models(&self) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = Vec::new();
        let mut seen: HashSet<String> = std::collections::HashSet::new();
        for p in self.openai_compatibility.iter() {
            for m in &p.models {
                if seen.insert(m.alias.clone()) {
                    out.push((m.alias.clone(), m.name.clone()));
                }
                if p.name != "default" {
                    let qualified = format!("{}/{}", p.name, m.alias);
                    if seen.insert(qualified.clone()) {
                        out.push((qualified, m.name.clone()));
                    }
                }
            }
        }
        out
    }

    /// Resolve a model alias to its backend model ID. Falls back to stripping a
    /// `default/` prefix, then to the alias itself.
    pub fn resolve_model(&self, alias: &str) -> String {
        if let Some(t) = self.upstream_target(alias) {
            return t.backend_model;
        }
        if let Some(stripped) = alias.strip_prefix("default/") {
            return stripped.to_string();
        }
        alias.to_string()
    }
}

/// A resolved upstream: base URL, optional key, backend model ID, provider name.
pub struct Upstream {
    pub base_url: String,
    pub api_key: Option<String>,
    pub backend_model: String,
    pub provider_name: String,
    /// Index into `Config.openai_compatibility`, so a caller can pick the
    /// per-provider `reqwest::Client` (which carries that provider's upstream
    /// `Authorization` header) without re-resolving the model.
    pub provider_idx: usize,
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
