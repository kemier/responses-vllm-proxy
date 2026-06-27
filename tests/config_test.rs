use ericai_proxy::config::Config;

#[test]
fn test_load_minimal_config() {
    let yaml = r#"
port: 8317
host: "0.0.0.0"
debug: false
auth-dir: "/tmp/test-auth"
api_keys: []
request_retry: 3

openai-compatibility:
  - name: "ericai"
    prefix: "ericai"
    base-url: "https://example.com/v1"
    models:
      - name: zai-org/GLM-5.2-FP8
        alias: glm-5.2
      - name: deepseek-ai/DeepSeek-V4-Pro
        alias: deepseek-v4-pro
"#;
    let path = "/tmp/test_ericai_config.yaml";
    std::fs::write(path, yaml).unwrap();
    let cfg = Config::load(path).unwrap();
    assert_eq!(cfg.port, 8317);
    assert_eq!(cfg.host, "0.0.0.0");
    assert_eq!(cfg.auth_dir, "/tmp/test-auth");
    assert!(cfg.ericai_provider().is_some());
    let models = cfg.all_models();
    assert_eq!(models.len(), 2);
    assert_eq!(models[0], ("glm-5.2".to_string(), "zai-org/GLM-5.2-FP8".to_string()));
    assert_eq!(models[1], ("deepseek-v4-pro".to_string(), "deepseek-ai/DeepSeek-V4-Pro".to_string()));
}

#[test]
fn test_kebab_case_parsing() {
    let yaml = r#"
port: 9999
auth-dir: "/tmp/test"
openai-compatibility:
  - name: "ericai"
    prefix: "ericai"
    base-url: "https://api.example.com"
    models:
      - name: model-1
        alias: m1
"#;
    let path = "/tmp/test_ericai_config2.yaml";
    std::fs::write(path, yaml).unwrap();
    let cfg = Config::load(path).unwrap();
    assert_eq!(cfg.ericai_provider().unwrap().base_url, "https://api.example.com");
}

#[test]
fn test_default_logging_values() {
    let yaml = r#"
port: 8317
auth-dir: "/tmp/test"
openai-compatibility:
  - name: "ericai"
    prefix: "ericai"
    base-url: "https://example.com"
    models:
      - name: m1
        alias: a1
"#;
    let path = "/tmp/test_ericai_config3.yaml";
    std::fs::write(path, yaml).unwrap();
    let cfg = Config::load(path).unwrap();
    assert!(cfg.logging_to_file);
    assert_eq!(cfg.logs_max_total_size_mb, 10);
    assert_eq!(cfg.log_level, "info");
    assert_eq!(cfg.log_dir, "logs");
}

#[test]
fn test_debug_overrides_log_level() {
    let yaml = r#"
port: 8317
debug: true
auth-dir: "/tmp/test"
openai-compatibility:
  - name: "ericai"
    prefix: "ericai"
    base-url: "https://example.com"
    models:
      - name: m1
        alias: a1
"#;
    let path = "/tmp/test_ericai_config4.yaml";
    std::fs::write(path, yaml).unwrap();
    let cfg = Config::load(path).unwrap();
    assert_eq!(cfg.log_level, "debug");
}

#[test]
fn test_tilde_expansion() {
    let yaml = r#"
port: 8317
auth-dir: "~/.cli-proxy-api"
log_dir: "logs"
openai-compatibility:
  - name: "ericai"
    prefix: "ericai"
    base-url: "https://example.com"
    models:
      - name: m1
        alias: a1
"#;
    let path = "/tmp/test_ericai_config5.yaml";
    std::fs::write(path, yaml).unwrap();
    let cfg = Config::load(path).unwrap();
    let home = std::env::var("HOME").unwrap_or("/home/user".to_string());
    assert_eq!(cfg.auth_dir, format!("{}/.cli-proxy-api", home));
}

#[test]
fn test_no_ericai_provider() {
    let yaml = r#"
port: 8317
auth-dir: "/tmp/test"
openai-compatibility: []
"#;
    let path = "/tmp/test_ericai_config6.yaml";
    std::fs::write(path, yaml).unwrap();
    let cfg = Config::load(path).unwrap();
    assert!(cfg.ericai_provider().is_none());
    assert_eq!(cfg.all_models().len(), 0);
}

#[test]
fn test_custom_logging_config() {
    let yaml = r#"
port: 8317
auth-dir: "/tmp/test"
logging-to-file: false
logs-max-total-size-mb: 50
log-level: "debug"
log-dir: "/tmp/test-logs"
openai-compatibility:
  - name: "ericai"
    prefix: "ericai"
    base-url: "https://example.com"
    models:
      - name: m1
        alias: a1
"#;
    let path = "/tmp/test_ericai_config7.yaml";
    std::fs::write(path, yaml).unwrap();
    let cfg = Config::load(path).unwrap();
    assert!(!cfg.logging_to_file);
    assert_eq!(cfg.logs_max_total_size_mb, 50);
    assert_eq!(cfg.log_level, "debug");
    assert_eq!(cfg.log_dir, "/tmp/test-logs");
}

#[test]
fn test_resolve_model_known_alias() {
    let yaml = r#"
port: 8317
auth-dir: "/tmp/test"
openai-compatibility:
  - name: "ericai"
    prefix: "ericai"
    base-url: "https://example.com"
    models:
    - name: zai-org/GLM-5.2-FP8
      alias: glm-5.2
    - name: deepseek-ai/DeepSeek-V4-Pro
      alias: deepseek-v4-pro
"#;
    let path = "/tmp/test_ericai_resolve1.yaml";
    std::fs::write(path, yaml).unwrap();
    let cfg = Config::load(path).unwrap();
    assert_eq!(cfg.resolve_model("glm-5.2"), "zai-org/GLM-5.2-FP8");
    assert_eq!(cfg.resolve_model("deepseek-v4-pro"), "deepseek-ai/DeepSeek-V4-Pro");
}

#[test]
fn test_resolve_model_unknown_alias_no_prefix() {
    let yaml = r#"
port: 8317
auth-dir: "/tmp/test"
openai-compatibility:
  - name: "ericai"
    prefix: "ericai"
    base-url: "https://example.com"
    models:
    - name: m1
      alias: a1
"#;
    let path = "/tmp/test_ericai_resolve2.yaml";
    std::fs::write(path, yaml).unwrap();
    let cfg = Config::load(path).unwrap();
    // Unknown alias without prefix returns itself
    assert_eq!(cfg.resolve_model("some-random-model"), "some-random-model");
}

#[test]
fn test_resolve_model_ericai_prefix_stripped() {
    let yaml = r#"
port: 8317
auth-dir: "/tmp/test"
openai-compatibility:
  - name: "ericai"
    prefix: "ericai"
    base-url: "https://example.com"
    models:
    - name: m1
      alias: a1
"#;
    let path = "/tmp/test_ericai_resolve3.yaml";
    std::fs::write(path, yaml).unwrap();
    let cfg = Config::load(path).unwrap();
    // ericai/ prefix is stripped for unknown models
    assert_eq!(cfg.resolve_model("ericai/foo-bar"), "foo-bar");
}
