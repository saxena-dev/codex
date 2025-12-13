use std::time::Duration;

use anyhow::Result;
use codex_api::ApiError;
use codex_api::ResponseEvent;
use codex_api::SseTelemetry;
use codex_api::sse::anthropic::spawn_anthropic_stream;
use codex_client::StreamResponse;
use codex_client::TransportError;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use futures::StreamExt;
use futures::TryStreamExt;
use http::HeaderMap;
use http::StatusCode;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use tokio_util::io::ReaderStream;

fn build_anthropic_body(events: Vec<Value>) -> String {
    let mut body = String::new();
    for e in events {
        let kind = e
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| panic!("fixture event missing type in Anthropic SSE fixture: {e}"));
        body.push_str(&format!("event: {kind}\ndata: {e}\n\n"));
    }
    body
}

fn spawn_stream(body: String) -> codex_api::ResponseStream {
    let reader = ReaderStream::new(std::io::Cursor::new(body))
        .map_err(|err| TransportError::Network(err.to_string()));

    let response = StreamResponse {
        status: StatusCode::OK,
        headers: HeaderMap::new(),
        bytes: Box::pin(reader),
    };

    spawn_anthropic_stream(
        response,
        Duration::from_millis(500),
        None::<std::sync::Arc<dyn SseTelemetry>>,
    )
}

#[tokio::test]
async fn anthropic_text_only_stream_emits_deltas_and_completion() -> Result<()> {
    let message_start = json!({
        "type": "message_start",
        "message": { "id": "msg_1", "role": "assistant" }
    });

    let content_start = json!({
        "type": "content_block_start",
        "index": 0,
        "content_block": { "type": "text" }
    });

    let delta1 = json!({
        "type": "content_block_delta",
        "index": 0,
        "delta": { "type": "text_delta", "text": "Hello, " }
    });

    let delta2 = json!({
        "type": "content_block_delta",
        "index": 0,
        "delta": { "type": "text_delta", "text": "world" }
    });

    let content_stop = json!({
        "type": "content_block_stop",
        "index": 0
    });

    let message_stop = json!({
        "type": "message_stop",
        "message": {
            "id": "msg_1",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0
            }
        }
    });

    let body = build_anthropic_body(vec![
        message_start,
        content_start,
        delta1,
        delta2,
        content_stop,
        message_stop,
    ]);

    let mut stream = spawn_stream(body);
    let mut events = Vec::new();
    while let Some(ev) = stream.next().await {
        events.push(ev?);
    }

    let events: Vec<ResponseEvent> = events
        .into_iter()
        .filter(|ev| !matches!(ev, ResponseEvent::RateLimits(_)))
        .collect();

    assert_eq!(events.len(), 5);

    match &events[0] {
        ResponseEvent::OutputItemAdded(ResponseItem::Message { role, .. }) => {
            assert_eq!(role, "assistant");
        }
        other => panic!("unexpected first event: {other:?}"),
    }

    match &events[1] {
        ResponseEvent::OutputTextDelta(text) => {
            assert_eq!(text, "Hello, ");
        }
        other => panic!("unexpected second event: {other:?}"),
    }

    match &events[2] {
        ResponseEvent::OutputTextDelta(text) => {
            assert_eq!(text, "world");
        }
        other => panic!("unexpected third event: {other:?}"),
    }

    match &events[3] {
        ResponseEvent::OutputItemDone(ResponseItem::Message { role, content, .. }) => {
            assert_eq!(role, "assistant");
            let mut aggregated = String::new();
            for item in content {
                if let ContentItem::OutputText { text } = item {
                    aggregated.push_str(text);
                }
            }
            assert_eq!(aggregated, "Hello, world");
        }
        other => panic!("unexpected fourth event: {other:?}"),
    }

    match &events[4] {
        ResponseEvent::Completed {
            response_id,
            token_usage,
        } => {
            assert_eq!(response_id, "msg_1");
            let usage = token_usage
                .as_ref()
                .unwrap_or_else(|| panic!("expected token usage in Completed"));
            assert_eq!(usage.input_tokens, 10);
            assert_eq!(usage.cached_input_tokens, 0);
            assert_eq!(usage.output_tokens, 20);
            assert_eq!(usage.reasoning_output_tokens, 0);
            assert_eq!(usage.total_tokens, 30);
        }
        other => panic!("unexpected fifth event: {other:?}"),
    }

    Ok(())
}

#[tokio::test]
async fn anthropic_tool_use_stream_emits_function_call_and_completed() -> Result<()> {
    let message_start = json!({
        "type": "message_start",
        "message": { "id": "msg_tool", "role": "assistant" }
    });

    let tool_start = json!({
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "tool_use",
            "id": "call_1",
            "name": "get_weather",
            "input": { "location": "San Francisco" }
        }
    });

    let tool_stop = json!({
        "type": "content_block_stop",
        "index": 0
    });

    let message_stop = json!({
        "type": "message_stop",
        "message": {
            "id": "msg_tool",
            "usage": {
                "input_tokens": 5,
                "output_tokens": 1,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0
            }
        }
    });

    let body = build_anthropic_body(vec![message_start, tool_start, tool_stop, message_stop]);

    let mut stream = spawn_stream(body);
    let mut events = Vec::new();
    while let Some(ev) = stream.next().await {
        events.push(ev?);
    }

    let events: Vec<ResponseEvent> = events
        .into_iter()
        .filter(|ev| !matches!(ev, ResponseEvent::RateLimits(_)))
        .collect();

    assert_eq!(events.len(), 2);

    match &events[0] {
        ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
            name,
            arguments,
            call_id,
            ..
        }) => {
            assert_eq!(name, "get_weather");
            assert_eq!(call_id, "call_1");
            let args: Value = serde_json::from_str(arguments)?;
            assert_eq!(args, json!({ "location": "San Francisco" }));
        }
        other => panic!("unexpected first event: {other:?}"),
    }

    match &events[1] {
        ResponseEvent::Completed {
            response_id,
            token_usage,
        } => {
            assert_eq!(response_id, "msg_tool");
            let usage = token_usage
                .as_ref()
                .unwrap_or_else(|| panic!("expected token usage in Completed"));
            assert_eq!(usage.input_tokens, 5);
            assert_eq!(usage.cached_input_tokens, 0);
            assert_eq!(usage.output_tokens, 1);
            assert_eq!(usage.reasoning_output_tokens, 0);
            assert_eq!(usage.total_tokens, 6);
        }
        other => panic!("unexpected second event: {other:?}"),
    }

    Ok(())
}

#[tokio::test]
async fn anthropic_error_event_produces_stream_error() -> Result<()> {
    let error_event = json!({
        "type": "error",
        "error": {
            "type": "rate_limit_error",
            "code": "rate_limit",
            "message": "Too many requests"
        }
    });

    let body = build_anthropic_body(vec![error_event]);
    let mut stream = spawn_stream(body);

    let mut events = Vec::new();
    while let Some(ev) = stream.next().await {
        events.push(ev);
    }

    assert_eq!(events.len(), 1);

    match &events[0] {
        Err(ApiError::Stream(message)) => {
            assert!(message.contains("rate_limit_error"));
            assert!(message.contains("Too many requests"));
        }
        other => panic!("unexpected event: {other:?}"),
    }

    Ok(())
}
