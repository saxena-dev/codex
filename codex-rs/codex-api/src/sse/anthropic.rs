use crate::common::ResponseEvent;
use crate::common::ResponseStream;
use crate::error::ApiError;
use crate::telemetry::SseTelemetry;
use codex_client::ByteStream;
use codex_client::StreamResponse;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TokenUsage;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio::time::timeout;
use tracing::debug;
use tracing::trace;

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: i64,
    #[serde(default)]
    cache_creation_input_tokens: i64,
    #[serde(default)]
    cache_read_input_tokens: i64,
    output_tokens: i64,
}

impl From<AnthropicUsage> for TokenUsage {
    fn from(val: AnthropicUsage) -> Self {
        let cached_input_tokens = val.cache_creation_input_tokens + val.cache_read_input_tokens;
        let total_tokens = val.input_tokens + val.output_tokens;

        TokenUsage {
            input_tokens: val.input_tokens,
            cached_input_tokens,
            output_tokens: val.output_tokens,
            reasoning_output_tokens: 0,
            total_tokens,
        }
    }
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorBody {
    #[serde(rename = "type")]
    kind: Option<String>,
    message: Option<String>,
    code: Option<String>,
}

pub fn spawn_anthropic_stream(
    stream_response: StreamResponse,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
) -> ResponseStream {
    let (tx_event, rx_event) = mpsc::channel::<Result<ResponseEvent, ApiError>>(1600);
    tokio::spawn(async move {
        process_anthropic_sse(stream_response.bytes, tx_event, idle_timeout, telemetry).await;
    });

    ResponseStream { rx_event }
}

async fn process_anthropic_sse(
    stream: ByteStream,
    tx_event: mpsc::Sender<Result<ResponseEvent, ApiError>>,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
) {
    let mut stream = stream.eventsource();
    let mut full_text = String::new();
    let mut response_id: Option<String> = None;
    let mut usage: Option<TokenUsage> = None;
    let mut completed_sent = false;
    let mut text_item_started = false;

    loop {
        let start = Instant::now();
        let response = timeout(idle_timeout, stream.next()).await;
        if let Some(t) = telemetry.as_ref() {
            t.on_sse_poll(&response, start.elapsed());
        }

        let sse = match response {
            Ok(Some(Ok(sse))) => sse,
            Ok(Some(Err(e))) => {
                let _ = tx_event.send(Err(ApiError::Stream(e.to_string()))).await;
                return;
            }
            Ok(None) => {
                if !completed_sent {
                    let _ = tx_event
                        .send(Err(ApiError::Stream(
                            "stream closed before message_stop".to_string(),
                        )))
                        .await;
                }
                return;
            }
            Err(_) => {
                let _ = tx_event
                    .send(Err(ApiError::Stream(
                        "idle timeout waiting for Anthropic SSE".to_string(),
                    )))
                    .await;
                return;
            }
        };

        if sse.data.trim().is_empty() {
            continue;
        }

        trace!("Anthropic SSE event: {}", sse.data);

        let value: Value = match serde_json::from_str(&sse.data) {
            Ok(val) => val,
            Err(err) => {
                let _ = tx_event
                    .send(Err(ApiError::Stream(format!(
                        "failed to parse Anthropic SSE event: {err}"
                    ))))
                    .await;
                return;
            }
        };

        let Some(kind) = value.get("type").and_then(Value::as_str) else {
            debug!("Anthropic SSE event missing type: {}", sse.data);
            let _ = tx_event
                .send(Err(ApiError::Stream(
                    "Anthropic SSE event missing type field".to_string(),
                )))
                .await;
            return;
        };

        if let Some(msg) = extract_usage(&value) {
            usage = Some(msg);
        }

        match kind {
            "message_start" => {
                full_text.clear();
                text_item_started = false;
                if let Some(message) = value.get("message")
                    && let Some(id) = message.get("id").and_then(Value::as_str)
                {
                    response_id = Some(id.to_string());
                }
            }
            "content_block_start" => {
                if let Some(content_block) = value.get("content_block") {
                    let block_type = content_block
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    if block_type == "tool_use"
                        && let Err(err) =
                            handle_tool_use_block(&tx_event, content_block.to_owned()).await
                    {
                        let _ = tx_event.send(Err(err)).await;
                        return;
                    }
                }
            }
            "content_block_delta" => {
                if let Some(delta) = value.get("delta") {
                    let delta_type = delta.get("type").and_then(Value::as_str);
                    if (delta_type == Some("text")
                        || delta_type == Some("text_delta")
                        || delta_type.is_none())
                        && let Some(text) = delta.get("text").and_then(Value::as_str)
                    {
                        if !text_item_started {
                            // Emit an OutputItemAdded for the assistant message the first
                            // time we see a text delta so downstream consumers have an
                            // active item to attach deltas to.
                            let item = ResponseItem::Message {
                                id: None,
                                role: "assistant".to_string(),
                                content: vec![],
                            };
                            text_item_started = true;
                            if tx_event
                                .send(Ok(ResponseEvent::OutputItemAdded(item)))
                                .await
                                .is_err()
                            {
                                return;
                            }
                        }

                        full_text.push_str(text);
                        let event = ResponseEvent::OutputTextDelta(text.to_string());
                        if tx_event.send(Ok(event)).await.is_err() {
                            return;
                        }
                    }
                }
            }
            "message_delta" => {}
            "content_block_stop" => {}
            "message_stop" => {
                if let Some(message) = value.get("message") {
                    if response_id.is_none()
                        && let Some(id) = message.get("id").and_then(Value::as_str)
                    {
                        response_id = Some(id.to_string());
                    }
                    if let Some(message_usage) = extract_usage(message) {
                        usage = Some(message_usage);
                    }
                }

                if !full_text.is_empty() {
                    let item = ResponseItem::Message {
                        id: None,
                        role: "assistant".to_string(),
                        content: vec![ContentItem::OutputText {
                            text: full_text.clone(),
                        }],
                    };
                    if tx_event
                        .send(Ok(ResponseEvent::OutputItemDone(item)))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }

                let completed = ResponseEvent::Completed {
                    response_id: response_id.take().unwrap_or_default(),
                    token_usage: usage.clone(),
                };
                completed_sent = true;
                if tx_event.send(Ok(completed)).await.is_err() {
                    return;
                }
            }
            "error" => {
                let message = build_error_message(&value);
                let _ = tx_event.send(Err(ApiError::Stream(message))).await;
                return;
            }
            _ => {}
        }
    }
}

fn extract_usage(value: &Value) -> Option<TokenUsage> {
    let usage_val = value
        .get("usage")
        .or_else(|| value.get("message").and_then(|m| m.get("usage")))?;
    let usage: AnthropicUsage = serde_json::from_value(usage_val.clone()).ok()?;
    Some(TokenUsage::from(usage))
}

fn build_error_message(value: &Value) -> String {
    if let Some(err_val) = value.get("error")
        && let Ok(body) = serde_json::from_value::<AnthropicErrorBody>(err_val.clone())
    {
        let kind = body.kind.unwrap_or_else(|| "unknown".to_string());
        let message = body.message.unwrap_or_default();
        if let Some(code) = body.code {
            return format!("Anthropic error {kind} ({code}): {message}");
        }
        return format!("Anthropic error {kind}: {message}");
    }

    format!("Anthropic error event: {value}")
}

async fn handle_tool_use_block(
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
    content_block: Value,
) -> Result<(), ApiError> {
    let id = content_block
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::Stream("tool_use content_block missing id field".to_string()))?
        .to_string();

    let name = content_block
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::Stream("tool_use content_block missing name field".to_string()))?
        .to_string();

    let input = content_block.get("input").cloned().unwrap_or(Value::Null);

    let arguments = serde_json::to_string(&input).map_err(|err| {
        ApiError::Stream(format!("failed to encode tool_use input as JSON: {err}"))
    })?;

    let item = ResponseItem::FunctionCall {
        id: None,
        name,
        arguments,
        call_id: id,
    };

    if tx_event
        .send(Ok(ResponseEvent::OutputItemDone(item)))
        .await
        .is_err()
    {
        return Ok(());
    }

    Ok(())
}
