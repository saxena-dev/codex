use crate::client_common::tools::ToolSpec;
use crate::error::Result;
use crate::openai_models::model_family::ModelFamily;
pub use codex_api::common::ResponseEvent;
use codex_apply_patch::APPLY_PATCH_TOOL_INSTRUCTIONS;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::ResponseItem;
use futures::Stream;
use serde::Deserialize;
use serde_json::Value;
use std::borrow::Cow;
use std::collections::HashSet;
use std::ops::Deref;
use std::pin::Pin;
use std::task::Context;
use std::task::Poll;
use tokio::sync::mpsc;

/// Review thread system prompt. Edit `core/src/review_prompt.md` to customize.
pub const REVIEW_PROMPT: &str = include_str!("../review_prompt.md");

// Centralized templates for review-related user messages
pub const REVIEW_EXIT_SUCCESS_TMPL: &str = include_str!("../templates/review/exit_success.xml");
pub const REVIEW_EXIT_INTERRUPTED_TMPL: &str =
    include_str!("../templates/review/exit_interrupted.xml");

/// API request payload for a single model turn
#[derive(Default, Debug, Clone)]
pub struct Prompt {
    /// Conversation context input items.
    pub input: Vec<ResponseItem>,

    /// Tools available to the model, including additional tools sourced from
    /// external MCP servers.
    pub(crate) tools: Vec<ToolSpec>,

    /// Whether parallel tool calls are permitted for this prompt.
    pub(crate) parallel_tool_calls: bool,

    /// Optional override for the built-in BASE_INSTRUCTIONS.
    pub base_instructions_override: Option<String>,

    /// Optional the output schema for the model's response.
    pub output_schema: Option<Value>,
}

impl Prompt {
    pub(crate) fn get_full_instructions<'a>(&'a self, model: &'a ModelFamily) -> Cow<'a, str> {
        let base = self
            .base_instructions_override
            .as_deref()
            .unwrap_or(model.base_instructions.deref());
        // When there are no custom instructions, add apply_patch_tool_instructions if:
        // - the model needs special instructions (4.1)
        // AND
        // - there is no apply_patch tool present
        let is_apply_patch_tool_present = self.tools.iter().any(|tool| match tool {
            ToolSpec::Function(f) => f.name == "apply_patch",
            ToolSpec::Freeform(f) => f.name == "apply_patch",
            _ => false,
        });
        if self.base_instructions_override.is_none()
            && model.needs_special_apply_patch_instructions
            && !is_apply_patch_tool_present
        {
            Cow::Owned(format!("{base}\n{APPLY_PATCH_TOOL_INSTRUCTIONS}"))
        } else {
            Cow::Borrowed(base)
        }
    }

    pub(crate) fn get_formatted_input(&self) -> Vec<ResponseItem> {
        let mut input = self.input.clone();

        // when using the *Freeform* apply_patch tool specifically, tool outputs
        // should be structured text, not json. Do NOT reserialize when using
        // the Function tool - note that this differs from the check above for
        // instructions. We declare the result as a named variable for clarity.
        let is_freeform_apply_patch_tool_present = self.tools.iter().any(|tool| match tool {
            ToolSpec::Freeform(f) => f.name == "apply_patch",
            _ => false,
        });
        if is_freeform_apply_patch_tool_present {
            reserialize_shell_outputs(&mut input);
        }

        input
    }
}

fn reserialize_shell_outputs(items: &mut [ResponseItem]) {
    let mut shell_call_ids: HashSet<String> = HashSet::new();

    items.iter_mut().for_each(|item| match item {
        ResponseItem::LocalShellCall { call_id, id, .. } => {
            if let Some(identifier) = call_id.clone().or_else(|| id.clone()) {
                shell_call_ids.insert(identifier);
            }
        }
        ResponseItem::CustomToolCall {
            id: _,
            status: _,
            call_id,
            name,
            input: _,
        } => {
            if name == "apply_patch" {
                shell_call_ids.insert(call_id.clone());
            }
        }
        ResponseItem::CustomToolCallOutput { call_id, output } => {
            if shell_call_ids.remove(call_id)
                && let Some(structured) = parse_structured_shell_output(output)
            {
                *output = structured
            }
        }
        ResponseItem::FunctionCall { name, call_id, .. }
            if is_shell_tool_name(name) || name == "apply_patch" =>
        {
            shell_call_ids.insert(call_id.clone());
        }
        ResponseItem::FunctionCallOutput { call_id, output } => {
            if shell_call_ids.remove(call_id)
                && let Some(structured) = parse_structured_shell_output(&output.content)
            {
                output.content = structured
            }
        }
        _ => {}
    })
}

fn is_shell_tool_name(name: &str) -> bool {
    matches!(name, "shell" | "container.exec")
}

#[derive(Deserialize)]
struct ExecOutputJson {
    output: String,
    metadata: ExecOutputMetadataJson,
}

#[derive(Deserialize)]
struct ExecOutputMetadataJson {
    exit_code: i32,
    duration_seconds: f32,
}

fn parse_structured_shell_output(raw: &str) -> Option<String> {
    let parsed: ExecOutputJson = serde_json::from_str(raw).ok()?;
    Some(build_structured_output(&parsed))
}

fn build_structured_output(parsed: &ExecOutputJson) -> String {
    let mut sections = Vec::new();
    sections.push(format!("Exit code: {}", parsed.metadata.exit_code));
    sections.push(format!(
        "Wall time: {} seconds",
        parsed.metadata.duration_seconds
    ));

    let mut output = parsed.output.clone();
    if let Some((stripped, total_lines)) = strip_total_output_header(&parsed.output) {
        sections.push(format!("Total output lines: {total_lines}"));
        output = stripped.to_string();
    }

    sections.push("Output:".to_string());
    sections.push(output);

    sections.join("\n")
}

fn strip_total_output_header(output: &str) -> Option<(&str, u32)> {
    let after_prefix = output.strip_prefix("Total output lines: ")?;
    let (total_segment, remainder) = after_prefix.split_once('\n')?;
    let total_lines = total_segment.parse::<u32>().ok()?;
    let remainder = remainder.strip_prefix('\n').unwrap_or(remainder);
    Some((remainder, total_lines))
}

/// Builds an Anthropic Messages request body from the core `Prompt`.
///
/// This is an Anthropic-specific helper that keeps existing OpenAI payload
/// builders unchanged. It maps Codex `ResponseItem` history and tools into
/// the minimal Messages request shape:
///   - `model` comes from the configured model slug.
///   - `system` is derived from `Prompt::get_full_instructions`.
///   - `messages` encodes user/assistant text, function calls, and tool
///     results using Anthropic's content blocks.
///   - `tools` is produced by the Anthropic tools helper, which currently
///     returns an empty vector and will be fleshed out in a follow-up task.
///   - `tool_choice` is `"auto"` when parallel tool calls are enabled and
///     `null` otherwise.
///   - `max_tokens` is derived from the model family's effective context
///     window when available, or falls back to a conservative default.
///   - `stream` is always `true` to enable SSE.
pub fn build_anthropic_messages_body(
    prompt: &Prompt,
    model: &str,
    model_family: &ModelFamily,
) -> Result<Value> {
    let system = prompt.get_full_instructions(model_family).into_owned();
    let formatted_input = prompt.get_formatted_input();
    let messages = response_items_to_anthropic_messages(&formatted_input);
    let tools = tools::create_tools_json_for_anthropic_messages(&prompt.tools)?;
    let tool_choice = if prompt.parallel_tool_calls {
        Value::String("auto".to_string())
    } else {
        Value::Null
    };
    let max_tokens = compute_anthropic_max_tokens(model_family);

    Ok(serde_json::json!({
        "model": model,
        "system": system,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "max_tokens": max_tokens,
        "stream": true,
    }))
}

fn compute_anthropic_max_tokens(model_family: &ModelFamily) -> i64 {
    // Anthropic requires `max_tokens > 0`. For now, derive a simple upper
    // bound from the effective context window when available, reserving
    // roughly three quarters of the window for input and overhead and one
    // quarter for output. When no context window is known, fall back to a
    // conservative default.
    const DEFAULT_MAX_TOKENS: i64 = 1_024;

    let effective_context_window = model_family
        .context_window
        .map(|w| w.saturating_mul(model_family.effective_context_window_percent) / 100);

    let candidate = effective_context_window
        .map(|effective| effective / 4)
        .unwrap_or(DEFAULT_MAX_TOKENS);

    if candidate > 0 {
        candidate
    } else {
        DEFAULT_MAX_TOKENS
    }
}

fn response_items_to_anthropic_messages(items: &[ResponseItem]) -> Vec<Value> {
    let mut messages = Vec::new();

    for item in items {
        match item {
            ResponseItem::Message { role, content, .. } => {
                let content_blocks: Vec<Value> = content
                    .iter()
                    .filter_map(|c| match c {
                        ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                            Some(serde_json::json!({
                                "type": "text",
                                "text": text,
                            }))
                        }
                        ContentItem::InputImage { .. } => None,
                    })
                    .collect();

                if !content_blocks.is_empty() {
                    messages.push(serde_json::json!({
                        "role": role,
                        "content": content_blocks,
                    }));
                }
            }
            ResponseItem::FunctionCall {
                name,
                arguments,
                call_id,
                ..
            } => {
                let input = serde_json::from_str(arguments).unwrap_or(Value::Null);
                messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": call_id,
                        "name": name,
                        "input": input,
                    }],
                }));
            }
            ResponseItem::FunctionCallOutput { call_id, output } => {
                let inner_content: Vec<Value> = if let Some(items) = &output.content_items {
                    items
                        .iter()
                        .map(|item| match item {
                            FunctionCallOutputContentItem::InputText { text } => {
                                serde_json::json!({
                                    "type": "text",
                                    "text": text,
                                })
                            }
                            FunctionCallOutputContentItem::InputImage { image_url } => {
                                serde_json::json!({
                                    "type": "text",
                                    "text": image_url,
                                })
                            }
                        })
                        .collect()
                } else {
                    vec![serde_json::json!({
                        "type": "text",
                        "text": output.content,
                    })]
                };

                messages.push(serde_json::json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": inner_content,
                    }],
                }));
            }
            ResponseItem::Reasoning { .. }
            | ResponseItem::LocalShellCall { .. }
            | ResponseItem::CustomToolCall { .. }
            | ResponseItem::CustomToolCallOutput { .. }
            | ResponseItem::WebSearchCall { .. }
            | ResponseItem::GhostSnapshot { .. }
            | ResponseItem::CompactionSummary { .. }
            | ResponseItem::Other => {}
        }
    }

    messages
}

pub(crate) mod tools {
    use crate::error::Result;
    use crate::tools::spec::JsonSchema;
    use serde::Deserialize;
    use serde::Serialize;
    use serde_json::Value;

    /// When serialized as JSON, this produces a valid "Tool" in the OpenAI
    /// Responses API.
    #[derive(Debug, Clone, Serialize, PartialEq)]
    #[serde(tag = "type")]
    pub(crate) enum ToolSpec {
        #[serde(rename = "function")]
        Function(ResponsesApiTool),
        #[serde(rename = "local_shell")]
        LocalShell {},
        // TODO: Understand why we get an error on web_search although the API docs say it's supported.
        // https://platform.openai.com/docs/guides/tools-web-search?api-mode=responses#:~:text=%7B%20type%3A%20%22web_search%22%20%7D%2C
        #[serde(rename = "web_search")]
        WebSearch {},
        #[serde(rename = "custom")]
        Freeform(FreeformTool),
    }

    impl ToolSpec {
        pub(crate) fn name(&self) -> &str {
            match self {
                ToolSpec::Function(tool) => tool.name.as_str(),
                ToolSpec::LocalShell {} => "local_shell",
                ToolSpec::WebSearch {} => "web_search",
                ToolSpec::Freeform(tool) => tool.name.as_str(),
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct FreeformTool {
        pub(crate) name: String,
        pub(crate) description: String,
        pub(crate) format: FreeformToolFormat,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct FreeformToolFormat {
        pub(crate) r#type: String,
        pub(crate) syntax: String,
        pub(crate) definition: String,
    }

    #[derive(Debug, Clone, Serialize, PartialEq)]
    pub struct ResponsesApiTool {
        pub(crate) name: String,
        pub(crate) description: String,
        /// TODO: Validation. When strict is set to true, the JSON schema,
        /// `required` and `additional_properties` must be present. All fields in
        /// `properties` must be present in `required`.
        pub(crate) strict: bool,
        pub(crate) parameters: JsonSchema,
    }

    /// Builds Anthropic Messages tool definitions from core `ToolSpec`s.
    ///
    /// This helper is intentionally minimal for the initial Anthropic
    /// integration and currently returns an empty `tools` array. A follow-up
    /// task will flesh out the mapping to Anthropic's `input_schema` format.
    pub(crate) fn create_tools_json_for_anthropic_messages(
        _tools: &[ToolSpec],
    ) -> Result<Vec<Value>> {
        Ok(Vec::new())
    }
}

pub struct ResponseStream {
    pub(crate) rx_event: mpsc::Receiver<Result<ResponseEvent>>,
}

impl Stream for ResponseStream {
    type Item = Result<ResponseEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx_event.poll_recv(cx)
    }
}

#[cfg(test)]
mod tests {
    use crate::openai_models::model_family::find_family_for_model;
    use codex_api::ResponsesApiRequest;
    use codex_api::common::OpenAiVerbosity;
    use codex_api::common::TextControls;
    use codex_api::create_text_param_for_request;
    use codex_protocol::models::ContentItem;
    use codex_protocol::models::FunctionCallOutputContentItem;
    use codex_protocol::models::FunctionCallOutputPayload;
    use codex_protocol::models::ResponseItem;
    use pretty_assertions::assert_eq;

    use super::*;

    struct InstructionsTestCase {
        pub slug: &'static str,
        pub expects_apply_patch_instructions: bool,
    }
    #[test]
    fn get_full_instructions_no_user_content() {
        let prompt = Prompt {
            ..Default::default()
        };
        let test_cases = vec![
            InstructionsTestCase {
                slug: "gpt-3.5",
                expects_apply_patch_instructions: true,
            },
            InstructionsTestCase {
                slug: "gpt-4.1",
                expects_apply_patch_instructions: true,
            },
            InstructionsTestCase {
                slug: "gpt-4o",
                expects_apply_patch_instructions: true,
            },
            InstructionsTestCase {
                slug: "gpt-5",
                expects_apply_patch_instructions: true,
            },
            InstructionsTestCase {
                slug: "gpt-5.1",
                expects_apply_patch_instructions: false,
            },
            InstructionsTestCase {
                slug: "codex-mini-latest",
                expects_apply_patch_instructions: true,
            },
            InstructionsTestCase {
                slug: "gpt-oss:120b",
                expects_apply_patch_instructions: false,
            },
            InstructionsTestCase {
                slug: "gpt-5.1-codex",
                expects_apply_patch_instructions: false,
            },
            InstructionsTestCase {
                slug: "gpt-5.1-codex-max",
                expects_apply_patch_instructions: false,
            },
        ];
        for test_case in test_cases {
            let model_family = find_family_for_model(test_case.slug);
            let expected = if test_case.expects_apply_patch_instructions {
                format!(
                    "{}\n{}",
                    model_family.clone().base_instructions,
                    APPLY_PATCH_TOOL_INSTRUCTIONS
                )
            } else {
                model_family.clone().base_instructions
            };

            let full = prompt.get_full_instructions(&model_family);
            assert_eq!(full, expected);
        }
    }

    #[test]
    fn anthropic_body_includes_user_and_assistant_messages() {
        let model_family = find_family_for_model("gpt-5.1");
        let mut prompt = Prompt::default();
        prompt.input = vec![
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "hello".to_string(),
                }],
            },
            ResponseItem::Message {
                id: None,
                role: "assistant".to_string(),
                content: vec![ContentItem::OutputText {
                    text: "world".to_string(),
                }],
            },
        ];

        let body = build_anthropic_messages_body(&prompt, "test-model", &model_family)
            .expect("anthropic body");
        let messages = body
            .get("messages")
            .and_then(|m| m.as_array())
            .expect("messages array");

        assert_eq!(messages.len(), 2);
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
        assert_eq!(
            messages[1].get("role").and_then(|r| r.as_str()),
            Some("assistant")
        );
        assert_eq!(
            messages[1]
                .get("content")
                .and_then(|c| c.as_array())
                .and_then(|c| c.first())
                .and_then(|b| b.get("text"))
                .and_then(|t| t.as_str()),
            Some("world")
        );

        let max_tokens = body
            .get("max_tokens")
            .and_then(serde_json::Value::as_i64)
            .expect("max_tokens");
        assert!(max_tokens > 0);
        assert_eq!(
            body.get("model").and_then(|m| m.as_str()),
            Some("test-model")
        );
        assert_eq!(body.get("stream").and_then(serde_json::Value::as_bool), Some(true));
    }

    #[test]
    fn anthropic_body_maps_function_call_and_result() {
        let model_family = find_family_for_model("gpt-5.1");
        let mut prompt = Prompt::default();
        prompt.input = vec![
            ResponseItem::FunctionCall {
                id: None,
                name: "my_tool".to_string(),
                arguments: r#"{"arg":"value"}"#.to_string(),
                call_id: "call-1".to_string(),
            },
            ResponseItem::FunctionCallOutput {
                call_id: "call-1".to_string(),
                output: FunctionCallOutputPayload {
                    content: "result".to_string(),
                    content_items: Some(vec![FunctionCallOutputContentItem::InputText {
                        text: "result".to_string(),
                    }]),
                    success: Some(true),
                },
            },
        ];

        let body = build_anthropic_messages_body(&prompt, "test-model", &model_family)
            .expect("anthropic body");
        let messages = body
            .get("messages")
            .and_then(|m| m.as_array())
            .expect("messages array");

        assert_eq!(messages.len(), 2);

        let tool_use = messages[0]
            .get("content")
            .and_then(|c| c.as_array())
            .and_then(|c| c.first())
            .expect("tool_use block");
        assert_eq!(
            messages[0].get("role").and_then(|r| r.as_str()),
            Some("assistant")
        );
        assert_eq!(
            tool_use.get("type").and_then(|t| t.as_str()),
            Some("tool_use")
        );
        assert_eq!(tool_use.get("id").and_then(|v| v.as_str()), Some("call-1"));
        assert_eq!(
            tool_use.get("name").and_then(|v| v.as_str()),
            Some("my_tool")
        );
        assert_eq!(
            tool_use
                .get("input")
                .and_then(|i| i.get("arg"))
                .and_then(|v| v.as_str()),
            Some("value")
        );

        let tool_result = messages[1]
            .get("content")
            .and_then(|c| c.as_array())
            .and_then(|c| c.first())
            .expect("tool_result block");
        assert_eq!(
            messages[1].get("role").and_then(|r| r.as_str()),
            Some("user")
        );
        assert_eq!(
            tool_result.get("type").and_then(|t| t.as_str()),
            Some("tool_result")
        );
        assert_eq!(
            tool_result.get("tool_use_id").and_then(|v| v.as_str()),
            Some("call-1")
        );
        let result_content = tool_result
            .get("content")
            .and_then(|c| c.as_array())
            .and_then(|c| c.first())
            .expect("tool_result content block");
        assert_eq!(
            result_content.get("type").and_then(|t| t.as_str()),
            Some("text")
        );
        assert_eq!(
            result_content.get("text").and_then(|t| t.as_str()),
            Some("result")
        );
    }

    #[test]
    fn serializes_text_verbosity_when_set() {
        let input: Vec<ResponseItem> = vec![];
        let tools: Vec<serde_json::Value> = vec![];
        let req = ResponsesApiRequest {
            model: "gpt-5.1",
            instructions: "i",
            input: &input,
            tools: &tools,
            tool_choice: "auto",
            parallel_tool_calls: true,
            reasoning: None,
            store: false,
            stream: true,
            include: vec![],
            prompt_cache_key: None,
            text: Some(TextControls {
                verbosity: Some(OpenAiVerbosity::Low),
                format: None,
            }),
        };

        let v = serde_json::to_value(&req).expect("json");
        assert_eq!(
            v.get("text")
                .and_then(|t| t.get("verbosity"))
                .and_then(|s| s.as_str()),
            Some("low")
        );
    }

    #[test]
    fn serializes_text_schema_with_strict_format() {
        let input: Vec<ResponseItem> = vec![];
        let tools: Vec<serde_json::Value> = vec![];
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "answer": {"type": "string"}
            },
            "required": ["answer"],
        });
        let text_controls =
            create_text_param_for_request(None, &Some(schema.clone())).expect("text controls");

        let req = ResponsesApiRequest {
            model: "gpt-5.1",
            instructions: "i",
            input: &input,
            tools: &tools,
            tool_choice: "auto",
            parallel_tool_calls: true,
            reasoning: None,
            store: false,
            stream: true,
            include: vec![],
            prompt_cache_key: None,
            text: Some(text_controls),
        };

        let v = serde_json::to_value(&req).expect("json");
        let text = v.get("text").expect("text field");
        assert!(text.get("verbosity").is_none());
        let format = text.get("format").expect("format field");

        assert_eq!(
            format.get("name"),
            Some(&serde_json::Value::String("codex_output_schema".into()))
        );
        assert_eq!(
            format.get("type"),
            Some(&serde_json::Value::String("json_schema".into()))
        );
        assert_eq!(format.get("strict"), Some(&serde_json::Value::Bool(true)));
        assert_eq!(format.get("schema"), Some(&schema));
    }

    #[test]
    fn omits_text_when_not_set() {
        let input: Vec<ResponseItem> = vec![];
        let tools: Vec<serde_json::Value> = vec![];
        let req = ResponsesApiRequest {
            model: "gpt-5.1",
            instructions: "i",
            input: &input,
            tools: &tools,
            tool_choice: "auto",
            parallel_tool_calls: true,
            reasoning: None,
            store: false,
            stream: true,
            include: vec![],
            prompt_cache_key: None,
            text: None,
        };

        let v = serde_json::to_value(&req).expect("json");
        assert!(v.get("text").is_none());
    }
}
