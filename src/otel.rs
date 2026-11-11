//! OpenTelemetry span tracing (OTLP/HTTP) — the trace/span/parent_id model
//! from interview/tracing.md applied to this proxy.
//!
//! Enabled only when `OTEL_TRACES_ENDPOINT` is set at startup
//! (e.g. `http://localhost:4318/v1/traces` for otel-desktop-viewer / Jaeger).
//! When unset the global tracer provider stays NoOp and every helper below
//! is inert — the proxy behaves exactly as before.
//!
//! Span tree per request (the agent span vocabulary from tracing.md Q2):
//!
//!   POST /v1/responses      Server span — gen_ai.request.model = the alias
//!                            the client asked for
//!     └─ upstream.chat      Client span — gen_ai.response.model = the backend
//!                            this proxy actually resolved the alias to
//!
//! The requested-vs-resolved model pair is the gen_ai convention that closes
//! the map-Q52 "model only recorded at session level" gap: this proxy rewrites
//! aliases, so it is the one place both names are known.
//!
//! Failure attribution (tracing.md Q7): the first span with status=Error in
//! the tree names the layer that broke — proxy-side rejection (invalid JSON /
//! throttle / context overflow) marks the root span; an upstream transport
//! error or non-2xx marks the child.

use opentelemetry::{Context, KeyValue, global};
use opentelemetry::trace::{SpanKind, Status, Tracer, TraceContextExt};
use opentelemetry_sdk::trace::SdkTracerProvider;
use opentelemetry_otlp::{SpanExporter, WithExportConfig};
use serde_json::Value;

/// Install the OTLP/HTTP span pipeline if `OTEL_TRACES_ENDPOINT` is set.
/// Uses the simple (synchronous) exporter: spans are flushed as soon as
/// they end, so a dev run shows every request in the web UI without waiting
/// for a batch to fill or for process exit.
pub fn init_from_env() -> Option<SdkTracerProvider> {
    let endpoint = std::env::var("OTEL_TRACES_ENDPOINT").ok().filter(|s| !s.is_empty());
    if endpoint.is_none() {
        return None;
    }
    let exporter = SpanExporter::builder()
        .with_http()
        .with_endpoint(endpoint.unwrap())
        .build();
    let exporter = match exporter {
        Ok(exporter) => exporter,
        Err(e) => {
            eprintln!("otel: failed to build OTLP exporter: {}", e);
            return None;
        }
    };
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter)
        .build();
    global::set_tracer_provider(provider.clone());
    Some(provider)
}

/// Start a root span (one per request lifecycle) and return its context.
/// The span ends when the context is dropped — i.e. on every return path
/// of the handler, which is why handlers can early-return freely.
pub fn root_span(name: &str, kind: SpanKind, attrs: Vec<KeyValue>) -> Context {
    let tracer = global::tracer("vllm-proxy");
    let span = tracer
        .span_builder(name.to_string())
        .with_kind(kind)
        .with_attributes(attrs)
        .start(&tracer);
    Context::current().with_span(span)
}

/// Start a child span under `parent` and return the child context, so
/// grandchildren inherit the tree link (start_with_context is the explicit
/// parent threading — the OTel equivalent of this repo's conn_id capture-
/// then-reattach discipline in logger.rs).
pub fn child_span(parent: &Context, name: &str, kind: SpanKind, attrs: Vec<KeyValue>) -> Context {
    let tracer = global::tracer("vllm-proxy");
    let span = tracer
        .span_builder(name.to_string())
        .with_kind(kind)
        .with_attributes(attrs)
        .start_with_context(&tracer, parent);
    parent.with_span(span)
}

/// Attach attributes to the span of `cx` (e.g. the resolved model after
/// alias resolution).
pub fn add_attrs(cx: &Context, attrs: Vec<KeyValue>) {
    cx.span().set_attributes(attrs);
}

/// Mark the span of `cx` as failed — the web UI renders it red and the
/// first failing span in the tree is the layer attribution.
pub fn mark_error(cx: &Context, description: &str) {
    cx.span().set_status(Status::error(description.to_string()));
}

/// String attribute shorthand (callers stay free of OTel type imports).
pub fn kv(key: &str, value: &str) -> KeyValue {
    KeyValue::new(key.to_string(), value.to_string())
}

/// Server-kind root span (the inbound HTTP request boundary).
pub fn server_span(name: &str, attrs: Vec<KeyValue>) -> Context {
    root_span(name, SpanKind::Server, attrs)
}

/// Client-kind child span (an outbound call to the upstream vLLM).
pub fn client_span(parent: &Context, name: &str, attrs: Vec<KeyValue>) -> Context {
    child_span(parent, name, SpanKind::Client, attrs)
}

/// Sampling params the client actually sent (gen_ai.request.* convention).
/// Recorded only when the key is present in the request body — absent means
/// "backend default", which must stay distinguishable from "explicitly zero".
pub fn sampling_attrs(request: &Value) -> Vec<KeyValue> {
    let mut attrs: Vec<KeyValue> = vec![];
    for (json_key, attr_name) in [
        ("temperature", "gen_ai.request.temperature"),
        ("top_p", "gen_ai.request.top_p"),
        ("max_tokens", "gen_ai.request.max_tokens"),
    ] {
        if let Some(v) = request.get(json_key) {
            if let Some(f) = v.as_f64() {
                attrs.push(kv(attr_name, &format!("{}", f)));
            } else if let Some(s) = v.as_str() {
                attrs.push(kv(attr_name, s));
            }
        }
    }
    attrs
}

/// Attempt count on the upstream child span (Q52 "attempt count" gap):
/// how many tries send_upstream_retry used before this response landed.
pub fn attempt_attrs(attempts: u32) -> Vec<KeyValue> {
    vec![kv("proxy.attempt_count", &format!("{}", attempts))]
}

/// Estimated input tokens on the root span (Q52 "token counts" gap, partial):
/// the proxy's own estimate — exact upstream usage on the streaming path
/// lives inside the spawned SSE task and stays a documented gap.
pub fn estimated_input_attrs(estimated: usize) -> Vec<KeyValue> {
    vec![kv("proxy.estimated_input_tokens", &format!("{}", estimated))]
}

/// Config fingerprint backing proxy.config.version (Q52 "config version"
/// gap): written once at startup from an FNV-1a hash of the raw config
/// bytes, read by every root span. Mutex<Option<String>> is the simplest
/// write-once that keeps the getter total when hashing failed.
static CONFIG_VERSION: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

pub fn set_config_version(version: String) {
    let mut slot = CONFIG_VERSION.lock().unwrap();
    *slot = Some(version);
}

pub fn config_version() -> String {
    let slot = CONFIG_VERSION.lock().unwrap();
    (*slot).clone().unwrap_or_default()
}

/// Root-span attrs = operation name + config version (when the startup
/// fingerprint succeeded). Every root span gets this so a trace can be
/// pinned to the config.yaml revision that served it.
pub fn root_span_attrs(operation: &str) -> Vec<KeyValue> {
    let mut attrs: Vec<KeyValue> = vec![kv("gen_ai.operation.name", operation)];
    let version = config_version();
    if !version.is_empty() {
        attrs.push(kv("proxy.config.version", &version));
    }
    attrs
}
