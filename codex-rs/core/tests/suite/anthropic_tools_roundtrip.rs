#![allow(clippy::expect_used)]

use std::sync::Arc;

use codex_app_server_protocol::AuthMode;
use codex_core::CodexAuth;
use codex_core::ConversationManager;
use codex_core::ModelProviderInfo;
use codex_core::WireApi;
use codex_core::config::Config;
use codex_core::protocol::EventMsg;
use codex_core::protocol::Op;
use codex_core::protocol::SandboxPolicy;
use codex_protocol::config_types::ReasoningSummary;
use codex_protocol::user_input::UserInput;
use core_test_support::load_default_config_for_test;
use core_test_support::skip_if_no_network;
use core_test_support::wait_for_event;
use serde_json::Value;
use serde_json::json;
use tempfile::TempDir;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;

fn build_anthropic_sse(events: Vec<Value>) -> String {
    let mut body = String::new();
    for e in events {
        body.push_str("data: ");
        body.push_str(&e.to_string());
        body.push_str("\n\n");
    }
    body
}

fn build_anthropic_provider(server: &MockServer) -> ModelProviderInfo {
    ModelProviderInfo {
        name: "mock-anthropic-tools".to_string(),
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
    }
}

fn configure_anthropic_config(config: &mut Config, provider: &ModelProviderInfo) {
    config.model_provider_id = provider.name.clone();
    config.model_provider = provider.clone();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn anthropic_tool_use_round_trip_includes_tool_result() {
    skip_if_no_network!();

    let server = MockServer::start().await;

    // First request: tool_use from Anthropic with call_id "call-1" and name "update_plan".
    let first_body = build_anthropic_sse(vec![
        json!({
            "type": "message_start",
            "message": { "id": "msg_tool_1", "role": "assistant" }
        }),
        json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "call-1",
                "name": "update_plan",
                "input": {
                    "explanation": "Anthropic tool round-trip",
                    "plan": [
                        { "step": "step one", "status": "in_progress" }
                    ]
                }
            }
        }),
        json!({
            "type": "content_block_stop",
            "index": 0
        }),
        json!({
            "type": "message_stop",
            "message": {
                "id": "msg_tool_1",
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0
                }
            }
        }),
    ]);

    // Second request: simple text completion after tool_result.
    let second_body = build_anthropic_sse(vec![
        json!({
            "type": "message_start",
            "message": { "id": "msg_tool_2", "role": "assistant" }
        }),
        json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": { "type": "text" }
        }),
        json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": { "type": "text_delta", "text": "done" }
        }),
        json!({
            "type": "content_block_stop",
            "index": 0
        }),
        json!({
            "type": "message_stop",
            "message": {
                "id": "msg_tool_2",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0
                }
            }
        }),
    ]);

    let first_template = ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_raw(first_body, "text/event-stream");

    let second_template = ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_raw(second_body, "text/event-stream");

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(first_template)
        .expect(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(second_template)
        .expect(1)
        .mount(&server)
        .await;

    // Configure Codex to use the Anthropic wire API with our mock server.
    let provider = build_anthropic_provider(&server);
    let codex_home = TempDir::new().expect("failed to create TempDir");
    let mut config = load_default_config_for_test(&codex_home);
    configure_anthropic_config(&mut config, &provider);

    let conversation_manager =
        ConversationManager::with_models_provider(CodexAuth::from_api_key("dummy"), provider);
    let new_conversation = conversation_manager
        .new_conversation(config)
        .await
        .expect("create new conversation");
    let codex = new_conversation.conversation;

    codex
        .submit(Op::UserTurn {
            items: vec![UserInput::Text {
                text: "please update the plan".into(),
            }],
            final_output_json_schema: None,
            cwd: codex_home.path().to_path_buf(),
            approval_policy: codex_core::protocol::AskForApproval::Never,
            sandbox_policy: SandboxPolicy::DangerFullAccess,
            model: new_conversation.session_configured.model.clone(),
            effort: None,
            summary: ReasoningSummary::Auto,
        })
        .await
        .expect("submit user turn");

    wait_for_event(&codex, |event| matches!(event, EventMsg::TaskComplete(_))).await;

    let all_requests = server
        .received_requests()
        .await
        .expect("received requests from mock server");
    let messages_requests: Vec<_> = all_requests
        .iter()
        .filter(|req| req.method == "POST" && req.url.path().ends_with("/messages"))
        .collect();

    assert_eq!(
        messages_requests.len(),
        2,
        "expected two POST /v1/messages calls"
    );

    let first_body: Value = messages_requests[0]
        .body_json()
        .expect("first request body to be valid JSON");
    let second_body: Value = messages_requests[1]
        .body_json()
        .expect("second request body to be valid JSON");

    // First request: tools should be present and no tool_result messages yet.
    let tools = first_body
        .get("tools")
        .and_then(Value::as_array)
        .expect("tools array in first request");
    assert!(
        tools
            .iter()
            .any(|tool| tool.get("name").and_then(Value::as_str) == Some("update_plan")),
        "expected update_plan tool definition in Anthropic tools"
    );

    let first_messages = first_body
        .get("messages")
        .and_then(Value::as_array)
        .expect("messages array in first request");
    assert!(
        !first_messages.iter().any(|msg| {
            msg.get("content")
                .and_then(Value::as_array)
                .map(|blocks| {
                    blocks.iter().any(|block| {
                        block.get("type").and_then(Value::as_str) == Some("tool_result")
                    })
                })
                .unwrap_or(false)
        }),
        "first request should not include tool_result blocks"
    );

    // Second request: must include a tool_result block referencing call-1.
    let second_messages = second_body
        .get("messages")
        .and_then(Value::as_array)
        .expect("messages array in second request");

    let mut saw_tool_result = false;
    for message in second_messages {
        let role = message.get("role").and_then(Value::as_str);
        let Some(content) = message.get("content").and_then(Value::as_array) else {
            continue;
        };
        for block in content {
            if block.get("type").and_then(Value::as_str) == Some("tool_result")
                && block
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    == Some("call-1")
            {
                assert_eq!(role, Some("user"));

                let result_content = block
                    .get("content")
                    .and_then(Value::as_array)
                    .and_then(|items| items.first())
                    .expect("tool_result content items");
                assert_eq!(
                    result_content.get("type").and_then(Value::as_str),
                    Some("text")
                );
                let text = result_content
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                assert!(
                    text.contains("Plan updated"),
                    "expected tool_result text to mention 'Plan updated', got {text:?}"
                );

                saw_tool_result = true;
            }
        }
    }

    assert!(saw_tool_result, "expected tool_result block in second request");
}

