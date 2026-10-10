use vllm_proxy::config::Config;

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
  - name: "default"
    prefix: "default"
    base-url: "https://example.com/v1"
    models:
      - name: zai-org/GLM-5.2-FP8
        alias: glm-5.2
      - name: deepseek-ai/DeepSeek-V4-Pro
        alias: deepseek-v4-pro
"#;
    let path = "/tmp/test_vllm_config.yaml";
    std::fs::write(path, yaml).unwrap();
    let cfg = Config::load(path).unwrap();
    assert_eq!(cfg.port, 8317);
    assert_eq!(cfg.host, "0.0.0.0");
    assert_eq!(cfg.auth_dir, "/tmp/test-auth");
    assert!(cfg.provider().is_some());
    let models = cfg.all_models();
    assert_eq!(models.len(), 2);
    assert_eq!(
        models[0],
        ("glm-5.2".to_string(), "zai-org/GLM-5.2-FP8".to_string())
    );
    assert_eq!(
        models[1],
        (
            "deepseek-v4-pro".to_string(),
            "deepseek-ai/DeepSeek-V4-Pro".to_string()
        )
    );
}

#[test]
fn test_kebab_case_parsing() {
    let yaml = r#"
port: 9999
auth-dir: "/tmp/test"
openai-compatibility:
  - name: "default"
    prefix: "default"
    base-url: "https://api.example.com"
    models:
      - name: model-1
        alias: m1
"#;
    let path = "/tmp/test_vllm_config2.yaml";
    std::fs::write(path, yaml).unwrap();
    let cfg = Config::load(path).unwrap();
    assert_eq!(
        cfg.provider().unwrap().base_url,
        "https://api.example.com"
    );
}

#[test]
fn test_default_logging_values() {
    let yaml = r#"
port: 8317
auth-dir: "/tmp/test"
openai-compatibility:
  - name: "default"
    prefix: "default"
    base-url: "https://example.com"
    models:
      - name: m1
        alias: a1
"#;
    let path = "/tmp/test_vllm_config3.yaml";
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
  - name: "default"
    prefix: "default"
    base-url: "https://example.com"
    models:
      - name: m1
        alias: a1
"#;
    let path = "/tmp/test_vllm_config4.yaml";
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
  - name: "default"
    prefix: "default"
    base-url: "https://example.com"
    models:
      - name: m1
        alias: a1
"#;
    let path = "/tmp/test_vllm_config5.yaml";
    std::fs::write(path, yaml).unwrap();
    let cfg = Config::load(path).unwrap();
    let home = std::env::var("HOME").unwrap_or("/home/user".to_string());
    assert_eq!(cfg.auth_dir, format!("{}/.cli-proxy-api", home));
}

#[test]
fn test_no_provider() {
    let yaml = r#"
port: 8317
auth-dir: "/tmp/test"
openai-compatibility: []
"#;
    let path = "/tmp/test_vllm_config6.yaml";
    std::fs::write(path, yaml).unwrap();
    let cfg = Config::load(path).unwrap();
    assert!(cfg.provider().is_none());
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
  - name: "default"
    prefix: "default"
    base-url: "https://example.com"
    models:
      - name: m1
        alias: a1
"#;
    let path = "/tmp/test_vllm_config7.yaml";
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
  - name: "default"
    prefix: "default"
    base-url: "https://example.com"
    models:
    - name: zai-org/GLM-5.2-FP8
      alias: glm-5.2
    - name: deepseek-ai/DeepSeek-V4-Pro
      alias: deepseek-v4-pro
"#;
    let path = "/tmp/test_vllm_resolve1.yaml";
    std::fs::write(path, yaml).unwrap();
    let cfg = Config::load(path).unwrap();
    assert_eq!(cfg.resolve_model("glm-5.2"), "zai-org/GLM-5.2-FP8");
    assert_eq!(
        cfg.resolve_model("deepseek-v4-pro"),
        "deepseek-ai/DeepSeek-V4-Pro"
    );
}

#[test]
fn test_resolve_model_unknown_alias_no_prefix() {
    let yaml = r#"
port: 8317
auth-dir: "/tmp/test"
openai-compatibility:
  - name: "default"
    prefix: "default"
    base-url: "https://example.com"
    models:
    - name: m1
      alias: a1
"#;
    let path = "/tmp/test_vllm_resolve2.yaml";
    std::fs::write(path, yaml).unwrap();
    let cfg = Config::load(path).unwrap();
    // Unknown alias without prefix returns itself
    assert_eq!(cfg.resolve_model("some-random-model"), "some-random-model");
}

#[test]
fn test_resolve_model_default_prefix_stripped() {
    let yaml = r#"
port: 8317
auth-dir: "/tmp/test"
openai-compatibility:
  - name: "default"
    prefix: "default"
    base-url: "https://example.com"
    models:
    - name: m1
      alias: a1
"#;
    let path = "/tmp/test_vllm_resolve3.yaml";
    std::fs::write(path, yaml).unwrap();
    let cfg = Config::load(path).unwrap();
    // default/ prefix is stripped for unknown models
    assert_eq!(cfg.resolve_model("default/foo-bar"), "foo-bar");
}

#[test]
fn test_upstream_for_routes_across_machines() {
    let yaml = r#"
port: 8317
auth-dir: "/tmp/test"
openai-compatibility:
  - name: "default"
    prefix: "default"
    base-url: "http://192.168.124.14:8080/v1"
    models:
      - name: qwen3.8-flash-next-coder-iq1_m
        alias: strata-qwen
  - name: "mac"
    prefix: "mac"
    base-url: "http://127.0.0.1:11434/v1"
    models:
      - name: qwen3.8-4b-distill
        alias: qwen38-4b
"#;
    let path = "/tmp/test_vllm_route1.yaml";
    std::fs::write(path, yaml).unwrap();
    let cfg = Config::load(path).unwrap();
    let mac = cfg.upstream_for("qwen38-4b");
    assert_eq!(mac.base_url, "http://127.0.0.1:11434/v1");
    assert_eq!(mac.provider_name, "mac");
    assert_eq!(mac.backend_model, "qwen3.8-4b-distill");
    assert_eq!(mac.provider_idx, 1);
    let ubuntu = cfg.upstream_for("strata-qwen");
    assert_eq!(ubuntu.base_url, "http://192.168.124.14:8080/v1");
    assert_eq!(ubuntu.provider_name, "default");
    assert_eq!(ubuntu.provider_idx, 0);
    // A model the catalog doesn't list still reaches the default machine
    // with its name unchanged — routing must not swallow unknown slugs.
    let unknown = cfg.upstream_for("not-in-catalog");
    assert_eq!(unknown.base_url, "http://192.168.124.14:8080/v1");
    assert_eq!(unknown.backend_model, "not-in-catalog");
}

#[test]
fn test_qualified_alias_picks_machine() {
    // Two machines carrying the SAME alias: the plain alias resolves to the
    // first provider that declares it, the qualified form names the machine.
    let yaml = r#"
port: 8317
auth-dir: "/tmp/test"
openai-compatibility:
  - name: "default"
    prefix: "default"
    base-url: "http://192.168.124.14:8080/v1"
    models:
      - name: qwen-big
        alias: qwen
  - name: "mac"
    prefix: "mac"
    base-url: "http://127.0.0.1:11434/v1"
    models:
      - name: qwen-small
        alias: qwen
"#;
    let path = "/tmp/test_vllm_route2.yaml";
    std::fs::write(path, yaml).unwrap();
    let cfg = Config::load(path).unwrap();
    assert_eq!(cfg.resolve_model("qwen"), "qwen-big");
    assert_eq!(cfg.resolve_model("mac/qwen"), "qwen-small");
    assert_eq!(cfg.upstream_for("mac/qwen").base_url, "http://127.0.0.1:11434/v1");
    assert_eq!(cfg.upstream_for("qwen").base_url, "http://192.168.124.14:8080/v1");
}

#[test]
fn test_all_models_lists_both_machines() {
    let yaml = r#"
port: 8317
auth-dir: "/tmp/test"
openai-compatibility:
  - name: "default"
    prefix: "default"
    base-url: "http://192.168.124.14:8080/v1"
    models:
      - name: qwen-big
        alias: qwen
  - name: "mac"
    prefix: "mac"
    base-url: "http://127.0.0.1:11434/v1"
    models:
      - name: qwen-small
        alias: qwen
"#;
    let path = "/tmp/test_vllm_route3.yaml";
    std::fs::write(path, yaml).unwrap();
    let cfg = Config::load(path).unwrap();
    let models = cfg.all_models();
    // Plain alias appears once (first provider wins); the non-default machine
    // also gets a qualified entry so the picker can offer both.
    assert_eq!(models.len(), 2);
    assert_eq!(models[0], ("qwen".to_string(), "qwen-big".to_string()));
    assert_eq!(models[1], ("mac/qwen".to_string(), "qwen-small".to_string()));
}

#[test]
fn test_api_key_resolved_from_env() {
    std::env::set_var("MAC_OLLAMA_KEY_TEST", "sekret-token");
    let yaml = r#"
port: 8317
auth-dir: "/tmp/test"
openai-compatibility:
  - name: "default"
    prefix: "default"
    base-url: "http://192.168.124.14:8080/v1"
    models:
      - name: qwen-big
        alias: qwen
  - name: "mac"
    prefix: "mac"
    base-url: "http://127.0.0.1:11434/v1"
    api-key-env: "MAC_OLLAMA_KEY_TEST"
    models:
      - name: qwen-small
        alias: small
"#;
    let path = "/tmp/test_vllm_route4.yaml";
    std::fs::write(path, yaml).unwrap();
    let cfg = Config::load(path).unwrap();
    let mac = cfg.upstream_for("small");
    assert_eq!(mac.api_key, Some("sekret-token".to_string()));
    // The default provider has no key configured: nothing is invented.
    assert_eq!(cfg.upstream_for("qwen").api_key, None);
}
