#![allow(clippy::expect_used)]

use std::sync::Arc;

use codex_app_server_protocol::AuthMode;
use codex_core::ContentItem;
use codex_core::ModelClient;
use codex_core::ModelProviderInfo;
use codex_core::Prompt;
use codex_core::ResponseItem;
use codex_core::WireApi;
use codex_core::openai_models::models_manager::ModelsManager;
use codex_otel::otel_event_manager::OtelEventManager;
use codex_protocol::ConversationId;
use core_test_support::load_default_config_for_test;
use core_test_support::skip_if_no_network;
use futures::StreamExt;
use serde_json::Value;
use tempfile::TempDir;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;

async fn run_request(input: Vec<ResponseItem>) -> Value {
    let server = MockServer::start().await;

    let template = ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_raw(
            concat!(
                "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\"}}\n\n",
                "data: {\"type\":\"message_stop\",\"message\":{\"id\":\"msg_1\",\"usage\":",
                "{\"input_tokens\":1,\"output_tokens\":1,\"cache_creation_input_tokens\":0,",
                "\"cache_read_input_tokens\":0}}}\n\n",
            ),
            "text/event-stream",
        );

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(template)
        .expect(1)
        .mount(&server)
        .await;

    let provider = ModelProviderInfo {
        name: "mock-anthropic".into(),
        base_url: Some(format!("{}/v1", server.uri())),
        env_key: None,
        env_key_instructions: None,
        experimental_bearer_token: None,
        wire_api: WireApi::AnthropicMessages,
        query_params: None,
        http_headers: None,
        env_http_headers: None,
        request_max_retries: Some(0),
        stream_max_retries: Some(0),
        stream_idle_timeout_ms: Some(5_000),
        requires_openai_auth: false,
    };

    let codex_home = TempDir::new().expect("failed to create TempDir");
    let mut config = load_default_config_for_test(&codex_home);
    config.model_provider_id = provider.name.clone();
    config.model_provider = provider.clone();
    let effort = config.model_reasoning_effort;
    let summary = config.model_reasoning_summary;
    let model = ModelsManager::get_model_offline(config.model.as_deref());
    config.model = Some(model.clone());
    let config = Arc::new(config);

    let conversation_id = ConversationId::new();
    let model_family = ModelsManager::construct_model_family_offline(model.as_str(), &config);
    let otel_event_manager = OtelEventManager::new(
        conversation_id,
        model.as_str(),
        model_family.slug.as_str(),
        None,
        Some("test@test.com".to_string()),
        Some(AuthMode::ApiKey),
        false,
        "test".to_string(),
    );

    let client = ModelClient::new(
        Arc::clone(&config),
        None,
        model_family,
        otel_event_manager,
        provider,
        effort,
        summary,
        conversation_id,
        codex_protocol::protocol::SessionSource::Exec,
    );

    let mut prompt = Prompt::default();
    prompt.input = input;

    let mut stream = match client.stream(&prompt).await {
        Ok(s) => s,
        Err(e) => panic!("stream anthropic failed: {e}"),
    };
    while let Some(event) = stream.next().await {
        if let Err(e) = event {
            panic!("stream event error: {e}");
        }
    }

    let all_requests = server.received_requests().await.expect("received requests");
    let requests: Vec<_> = all_requests
        .iter()
        .filter(|req| req.method == "POST" && req.url.path().ends_with("/messages"))
        .collect();
    let request = requests
        .first()
        .unwrap_or_else(|| panic!("expected POST request to /messages"));
    match request.body_json() {
        Ok(v) => v,
        Err(e) => panic!("invalid json body: {e}"),
    }
}

fn user_message(text: &str) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: text.to_string(),
        }],
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn anthropic_messages_streams_to_messages_endpoint() {
    skip_if_no_network!();

    let body = run_request(vec![user_message("hello")]).await;

    assert!(body.get("model").and_then(|m| m.as_str()).is_some());
    assert!(body.get("system").and_then(|s| s.as_str()).is_some());
    assert_eq!(
        body.get("stream").and_then(serde_json::Value::as_bool),
        Some(true)
    );

    let messages = body
        .get("messages")
        .and_then(|m| m.as_array())
        .expect("messages array");
    assert_eq!(messages.len(), 1);
    assert_eq!(
        messages[0].get("role").and_then(|r| r.as_str()),
        Some("user")
    );
    assert_eq!(
        messages[0]
            .get("content")
            .and_then(|c| c.as_array())
            .and_then(|c| c.first())
            .and_then(|b| b.get("text"))
            .and_then(|t| t.as_str()),
        Some("hello")
    );

    let max_tokens = body
        .get("max_tokens")
        .and_then(serde_json::Value::as_i64)
        .expect("max_tokens");
    assert!(max_tokens > 0);
}
