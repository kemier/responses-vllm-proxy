use ericai_proxy::logger;

#[test]
fn test_extract_error_summary_from_json() {
    let body = r#"{"error":{"message":"Invalid model 'bad-model'","code":400}}"#;
    let summary = logger::extract_error_summary(body, 300);
    assert_eq!(summary, "Invalid model 'bad-model'");
}

#[test]
fn test_extract_error_summary_internal_message() {
    let body = r#"{"error":{"internal_message":"Internal server error","code":500}}"#;
    let summary = logger::extract_error_summary(body, 300);
    assert_eq!(summary, "Internal server error");
}

#[test]
fn test_extract_error_summary_top_level_message() {
    let body = r#"{"message":"Not found"}"#;
    let summary = logger::extract_error_summary(body, 300);
    assert_eq!(summary, "Not found");
}

#[test]
fn test_extract_error_summary_detail() {
    let body = r#"{"detail":"Not Found"}"#;
    let summary = logger::extract_error_summary(body, 300);
    assert_eq!(summary, "Not Found");
}

#[test]
fn test_extract_error_summary_non_json_fallback() {
    let body = "This is plain text error";
    let summary = logger::extract_error_summary(body, 300);
    assert_eq!(summary, "This is plain text error");
}

#[test]
fn test_extract_error_summary_non_json_truncated() {
    let body = "A".repeat(500);
    let summary = logger::extract_error_summary(&body, 100);
    assert!(summary.len() <= 104); // 100 + "..."
    assert!(summary.ends_with("..."));
}

#[test]
fn test_extract_error_summary_empty_body() {
    let summary = logger::extract_error_summary("", 300);
    assert_eq!(summary, "");
}

#[test]
fn test_extract_error_summary_nested_vllm_format() {
    // vLLM wraps errors differently
    let body = r#"{"generated_text":null,"error":{"message":"Invalid model 'bad'","internal_message":"OpenAIHTTPException: Invalid model 'bad'","code":400,"type":"OpenAIHTTPException"}}"#;
    let summary = logger::extract_error_summary(body, 300);
    assert_eq!(summary, "Invalid model 'bad'");
}

#[test]
fn test_extract_error_summary_unicode_safe() {
    // Ensure we don't panic on multi-byte chars at truncation boundary
    let body = "日本語のエラーメッセージ".repeat(50);
    let summary = logger::extract_error_summary(&body, 10);
    // Should not panic, should end with ...
    assert!(summary.ends_with("...") || summary.len() <= 10);
}
