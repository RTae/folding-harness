//! A [`ModelProvider`](crate::agent::ModelProvider) over the Claude Messages
//! API, by raw HTTP.
//!
//! There is no official Anthropic SDK for Rust, so this talks to
//! `POST /v1/messages` directly with `reqwest`. It is deliberately small: one
//! non-streaming completion, no tools, no server-side state. Everything the
//! harness asks a model is a single question with a single answer.
//!
//! # Custom endpoints
//!
//! The base URL, API version, model and effort are all configurable, so this
//! works against a proxy, a gateway, a router or a self-hosted endpoint as
//! well as the first-party API.
//!
//! Pointing at a non-first-party host changes what gets sent. Server-side
//! refusal fallback and `output_config.effort` are features of Anthropic's own
//! API; a gateway that does not understand them may reject the whole request
//! over an unknown field. So when the base URL is not the first-party one, the
//! request is reduced to the fields every Messages-compatible endpoint
//! accepts — `model`, `max_tokens`, `system`, `messages` — unless the extras
//! are asked for explicitly. [`Provider::describe`] reports which mode is
//! active, and the CLI prints it, because where prompts are sent should never
//! be a guess.

use serde::Deserialize;
use serde_json::{json, Value};

use crate::agent::{AgentError, ChatMessage, ModelProvider};

/// Pinned rather than tracking "latest": a silent API version change must not
/// alter how a recorded run behaves. Override it for an endpoint that wants an
/// older one.
pub const API_VERSION: &str = "2026-10-21";
pub const DEFAULT_MODEL: &str = "claude-opus-5-5";
pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

/// Opting into server-side fallback, so a safety decline on one model is
/// re-run on another inside the same call rather than failing the command.
const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";
/// Required when authenticating with an OAuth token rather than an API key.
const OAUTH_BETA: &str = "oauth-2025-04-20";

#[derive(Debug, Clone)]
enum Auth {
    ApiKey(String),
    /// A short-lived OAuth token, e.g. from `ant auth print-credentials
    /// --access-token`.
    Bearer(String),
}

#[derive(Debug, Clone)]
pub struct Provider {
    client: reqwest::Client,
    base_url: String,
    auth: Auth,
    model: String,
    api_version: String,
    /// `None` omits `output_config` entirely, for endpoints that reject it.
    effort: Option<String>,
    /// Send `fallbacks` and its beta header. First-party only by default.
    fallbacks: bool,
    max_tokens: u32,
}

/// Earlier name for [`Provider`], kept so existing callers keep working.
pub type AnthropicProvider = Provider;

impl Provider {
    /// Read configuration from the environment.
    ///
    /// Credentials: `ANTHROPIC_API_KEY`, else `ANTHROPIC_AUTH_TOKEN` (an OAuth
    /// token). Endpoint: `ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL`,
    /// `ANTHROPIC_API_VERSION`.
    pub fn from_env() -> Result<Self, AgentError> {
        let auth = match std::env::var("ANTHROPIC_API_KEY") {
            Ok(key) if !key.trim().is_empty() => Auth::ApiKey(key.trim().to_string()),
            _ => match std::env::var("ANTHROPIC_AUTH_TOKEN") {
                Ok(token) if !token.trim().is_empty() => Auth::Bearer(token.trim().to_string()),
                _ => {
                    return Err(AgentError::Provider(
                        "no credentials: set ANTHROPIC_API_KEY, or run `ant auth login` and \
                         export ANTHROPIC_AUTH_TOKEN=$(ant auth print-credentials --access-token)"
                            .to_string(),
                    ))
                }
            },
        };

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(180))
            .build()
            .map_err(|error| AgentError::Provider(error.to_string()))?;

        let base_url = std::env::var("ANTHROPIC_BASE_URL")
            .unwrap_or_else(|_| DEFAULT_BASE_URL.to_string())
            .trim_end_matches('/')
            .to_string();

        let provider = Provider {
            client,
            auth,
            model: std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string()),
            api_version: std::env::var("ANTHROPIC_API_VERSION")
                .unwrap_or_else(|_| API_VERSION.to_string()),
            effort: Some("high".to_string()),
            fallbacks: true,
            max_tokens: 16000,
            base_url: DEFAULT_BASE_URL.to_string(),
        };
        Ok(provider.with_base_url(base_url))
    }

    /// Set the endpoint. A host other than the first-party one switches the
    /// request to the compatible subset; re-enable the extras after this call
    /// if the gateway supports them.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        let base_url = base_url.into().trim_end_matches('/').to_string();
        let first_party = base_url == DEFAULT_BASE_URL;
        self.base_url = base_url;
        if !first_party {
            self.fallbacks = false;
            self.effort = None;
        }
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = version.into();
        self
    }

    /// `low` for small, frequent decisions; `high` where a wrong answer costs
    /// more than the tokens. `None` omits the field.
    pub fn with_effort(mut self, effort: Option<String>) -> Self {
        self.effort = effort;
        self
    }

    pub fn with_fallbacks(mut self, enabled: bool) -> Self {
        self.fallbacks = enabled;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens.max(1);
        self
    }

    pub fn model_name(&self) -> &str {
        &self.model
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn api_version(&self) -> &str {
        &self.api_version
    }

    pub fn is_first_party(&self) -> bool {
        self.base_url == DEFAULT_BASE_URL
    }

    /// Host part of the endpoint, for printing. Falls back to the whole URL if
    /// it cannot be split.
    pub fn host(&self) -> &str {
        self.base_url
            .split("://")
            .nth(1)
            .unwrap_or(&self.base_url)
            .split('/')
            .next()
            .unwrap_or(&self.base_url)
    }

    /// One line naming the model, the host, and whether prompts are leaving
    /// Anthropic's API. Printed by the CLI before any prompt is sent.
    pub fn describe(&self) -> String {
        if self.is_first_party() {
            format!("{} via {}", self.model, self.host())
        } else {
            format!(
                "{} via {} — a third-party endpoint: prompts go to that host, and \
                 Anthropic-specific request fields are omitted",
                self.model,
                self.host()
            )
        }
    }

    /// The request body. Separate from sending so it can be asserted on.
    pub fn request_body(&self, system: &str, messages: &[ChatMessage]) -> Value {
        let wire: Vec<Value> = messages
            .iter()
            .map(|message| json!({ "role": message.role.wire_name(), "content": message.text }))
            .collect();

        // Thinking is always on for the current model family, so the parameter
        // is omitted; depth is set through `output_config.effort` where the
        // endpoint supports it. Sampling parameters are rejected at the pinned
        // API version, so none are sent.
        let mut body = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": wire,
        });

        if let Some(map) = body.as_object_mut() {
            if let Some(effort) = &self.effort {
                map.insert("output_config".into(), json!({ "effort": effort }));
            }
            if self.fallbacks {
                map.insert("fallbacks".into(), json!("default"));
            }
        }
        body
    }

    fn beta_headers(&self) -> Vec<&str> {
        let mut betas = Vec::new();
        if self.fallbacks {
            betas.push(FALLBACK_BETA);
        }
        if matches!(self.auth, Auth::Bearer(_)) {
            betas.push(OAUTH_BETA);
        }
        betas
    }

    /// Send a tiny request to confirm the endpoint, credentials and model all
    /// work together. Used by `fh doctor --probe-model`.
    pub async fn probe(&self) -> Result<String, AgentError> {
        let probe = Provider {
            max_tokens: 64,
            effort: self.effort.as_ref().map(|_| "low".to_string()),
            ..self.clone()
        };
        probe
            .converse(
                "Reply with the single word OK.",
                &[ChatMessage::user("Reply with the single word OK.")],
            )
            .await
    }
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: String,
}

#[derive(Debug, Deserialize)]
struct StopDetails {
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    explanation: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MessageResponse {
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    stop_details: Option<StopDetails>,
    #[serde(default)]
    content: Vec<ContentBlock>,
}

/// Join the text blocks of a response, after checking it was not a refusal.
///
/// Exposed so the parsing is testable without a network call.
pub fn text_from_response(body: &str) -> Result<String, AgentError> {
    let response: MessageResponse = serde_json::from_str(body)
        .map_err(|error| AgentError::Provider(format!("unreadable response: {error}")))?;

    // A safety decline arrives as HTTP 200, so the status code alone does not
    // tell you the request succeeded.
    if response.stop_reason.as_deref() == Some("refusal") {
        let detail = response
            .stop_details
            .as_ref()
            .map(|details| {
                format!(
                    "{}{}",
                    details.category.as_deref().unwrap_or("unspecified"),
                    details
                        .explanation
                        .as_deref()
                        .map(|explanation| format!(": {explanation}"))
                        .unwrap_or_default()
                )
            })
            .unwrap_or_else(|| "no detail given".to_string());
        return Err(AgentError::Provider(format!(
            "the model declined this request ({detail})"
        )));
    }

    let text: String = response
        .content
        .iter()
        .filter(|block| block.kind == "text")
        .map(|block| block.text.as_str())
        .collect::<Vec<_>>()
        .join("");

    if text.trim().is_empty() {
        return Err(AgentError::Provider(format!(
            "no text in the response (stop_reason {})",
            response.stop_reason.as_deref().unwrap_or("unknown")
        )));
    }
    Ok(text)
}

#[async_trait::async_trait]
impl ModelProvider for Provider {
    fn model(&self) -> &str {
        &self.model
    }

    async fn complete(&self, system: &str, user: &str) -> Result<String, AgentError> {
        self.converse(system, &[ChatMessage::user(user)]).await
    }

    async fn converse(&self, system: &str, messages: &[ChatMessage]) -> Result<String, AgentError> {
        if messages.is_empty() {
            return Err(AgentError::Provider("no messages to send".to_string()));
        }

        let mut request = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("content-type", "application/json")
            .header("anthropic-version", &self.api_version);

        request = match &self.auth {
            Auth::ApiKey(key) => request.header("x-api-key", key),
            Auth::Bearer(token) => request.header("authorization", format!("Bearer {token}")),
        };

        let betas = self.beta_headers();
        if !betas.is_empty() {
            request = request.header("anthropic-beta", betas.join(","));
        }

        let response = request
            .json(&self.request_body(system, messages))
            .send()
            .await
            .map_err(|error| {
                AgentError::Provider(format!("could not reach {}: {error}", self.host()))
            })?;

        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|error| AgentError::Provider(error.to_string()))?;

        if !status.is_success() {
            // The hints matter most on a custom endpoint, where a 404 is far
            // more likely to be a wrong URL than a wrong request.
            let hint = match status.as_u16() {
                401 | 403 => " — check the API key for this endpoint",
                404 => " — check the base URL, and that this endpoint serves this model",
                400 if !self.is_first_party() => {
                    " — the endpoint rejected the request; try --api-version, or a model it serves"
                }
                429 => " — rate limited",
                _ => "",
            };
            return Err(AgentError::Provider(format!(
                "{} returned HTTP {}{hint}: {}",
                self.host(),
                status.as_u16(),
                text.chars().take(400).collect::<String>()
            )));
        }

        text_from_response(&text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider() -> Provider {
        Provider {
            client: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
            auth: Auth::ApiKey("test".to_string()),
            model: DEFAULT_MODEL.to_string(),
            api_version: API_VERSION.to_string(),
            effort: Some("high".to_string()),
            fallbacks: true,
            max_tokens: 16000,
        }
    }

    #[test]
    fn text_blocks_are_joined_and_non_text_blocks_ignored() {
        let body = r#"{
            "stop_reason": "end_turn",
            "content": [
                {"type": "thinking", "text": ""},
                {"type": "text", "text": "hello "},
                {"type": "text", "text": "world"}
            ]
        }"#;
        assert_eq!(text_from_response(body).unwrap(), "hello world");
    }

    #[test]
    fn a_refusal_is_an_error_even_though_the_status_was_ok() {
        let body = r#"{
            "stop_reason": "refusal",
            "stop_details": {"type": "refusal", "category": "bio", "explanation": "declined"},
            "content": []
        }"#;
        let error = text_from_response(body).expect_err("a refusal must not read as success");
        let message = error.to_string();
        assert!(message.contains("declined"), "{message}");
        assert!(message.contains("bio"), "{message}");
    }

    #[test]
    fn an_empty_response_is_an_error_not_an_empty_string() {
        let body = r#"{"stop_reason": "max_tokens", "content": []}"#;
        assert!(text_from_response(body).is_err());
    }

    #[test]
    fn unreadable_json_is_reported_as_such() {
        assert!(text_from_response("not json").is_err());
    }

    #[test]
    fn the_first_party_body_carries_effort_and_fallbacks() {
        let body = provider().request_body("sys", &[ChatMessage::user("hi")]);
        assert_eq!(body["model"], DEFAULT_MODEL);
        assert_eq!(body["output_config"]["effort"], "high");
        assert_eq!(body["fallbacks"], "default");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "hi");
        // Sampling parameters are rejected at this API version.
        assert!(body.get("temperature").is_none());
        assert!(body.get("top_p").is_none());
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn a_custom_endpoint_gets_the_compatible_subset() {
        let custom = provider().with_base_url("https://gateway.example.com/");
        assert!(!custom.is_first_party());
        assert_eq!(custom.base_url(), "https://gateway.example.com");
        assert_eq!(custom.host(), "gateway.example.com");

        let body = custom.request_body("sys", &[ChatMessage::user("hi")]);
        // Anthropic-specific fields are omitted: a gateway that does not know
        // them could reject the whole request.
        assert!(body.get("output_config").is_none());
        assert!(body.get("fallbacks").is_none());
        // The universal fields remain.
        assert_eq!(body["model"], DEFAULT_MODEL);
        assert_eq!(body["max_tokens"], 16000);
        assert_eq!(body["system"], "sys");
        assert!(body["messages"].is_array());

        assert!(custom.beta_headers().is_empty());
    }

    #[test]
    fn the_extras_can_be_re_enabled_for_a_gateway_that_supports_them() {
        let custom = provider()
            .with_base_url("https://gateway.example.com")
            .with_effort(Some("low".to_string()))
            .with_fallbacks(true);
        let body = custom.request_body("sys", &[ChatMessage::user("hi")]);
        assert_eq!(body["output_config"]["effort"], "low");
        assert_eq!(body["fallbacks"], "default");
        assert_eq!(custom.beta_headers(), vec![FALLBACK_BETA]);
    }

    #[test]
    fn the_api_version_and_model_are_overridable_for_an_older_endpoint() {
        let custom = provider()
            .with_base_url("https://gateway.example.com")
            .with_api_version("2023-06-01")
            .with_model("claude-opus-4-7");
        assert_eq!(custom.api_version(), "2023-06-01");
        assert_eq!(custom.model_name(), "claude-opus-4-7");
        assert_eq!(
            custom.request_body("s", &[ChatMessage::user("h")])["model"],
            "claude-opus-4-7"
        );
    }

    #[test]
    fn describe_names_the_host_and_flags_a_third_party_one() {
        let first = provider();
        assert!(first.describe().contains("api.anthropic.com"));
        assert!(!first.describe().contains("third-party"));

        let custom = provider().with_base_url("https://gateway.example.com");
        let described = custom.describe();
        assert!(described.contains("gateway.example.com"), "{described}");
        assert!(described.contains("third-party"), "{described}");
    }

    #[test]
    fn an_oauth_token_adds_its_beta_header_but_no_fallback_on_a_gateway() {
        let mut custom = provider().with_base_url("https://gateway.example.com");
        custom.auth = Auth::Bearer("token".to_string());
        assert_eq!(custom.beta_headers(), vec![OAUTH_BETA]);
    }

    #[test]
    fn multi_turn_messages_keep_their_roles_in_order() {
        let body = provider().request_body(
            "sys",
            &[
                ChatMessage::user("one"),
                ChatMessage::assistant("two"),
                ChatMessage::user("three"),
            ],
        );
        let messages = body["messages"].as_array().expect("an array");
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[2]["content"], "three");
    }
}
