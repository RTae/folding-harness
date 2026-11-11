//! A [`ModelProvider`](crate::agent::ModelProvider) over the Claude Messages
//! API, by raw HTTP.
//!
//! There is no official Anthropic SDK for Rust, so this talks to
//! `POST /v1/messages` directly with `reqwest`. It is deliberately small: one
//! non-streaming text completion, no tools, no conversation state. Everything
//! the harness asks a model is a single question with a single answer.

use serde::Deserialize;
use serde_json::json;

use crate::agent::{AgentError, ChatMessage, ModelProvider};

/// Pinned rather than tracking "latest": a silent API version change must not
/// alter how a recorded run behaves.
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
pub struct AnthropicProvider {
    client: reqwest::Client,
    base_url: String,
    auth: Auth,
    model: String,
    effort: String,
    max_tokens: u32,
}

impl AnthropicProvider {
    /// Read credentials from the environment.
    ///
    /// `ANTHROPIC_API_KEY` first, then `ANTHROPIC_AUTH_TOKEN` (an OAuth token).
    /// `ANTHROPIC_BASE_URL` and `ANTHROPIC_MODEL` override the defaults.
    pub fn from_env() -> Result<Self, AgentError> {
        let auth = match std::env::var("ANTHROPIC_API_KEY") {
            Ok(key) if !key.trim().is_empty() => Auth::ApiKey(key),
            _ => match std::env::var("ANTHROPIC_AUTH_TOKEN") {
                Ok(token) if !token.trim().is_empty() => Auth::Bearer(token),
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

        Ok(AnthropicProvider {
            client,
            base_url: std::env::var("ANTHROPIC_BASE_URL")
                .unwrap_or_else(|_| DEFAULT_BASE_URL.to_string())
                .trim_end_matches('/')
                .to_string(),
            auth,
            model: std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string()),
            effort: "high".to_string(),
            max_tokens: 16000,
        })
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// `low` for small, frequent decisions; `high` where a wrong answer costs
    /// more than the tokens.
    pub fn with_effort(mut self, effort: impl Into<String>) -> Self {
        self.effort = effort.into();
        self
    }

    pub fn model_name(&self) -> &str {
        &self.model
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
impl ModelProvider for AnthropicProvider {
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

        let wire: Vec<serde_json::Value> = messages
            .iter()
            .map(|message| json!({ "role": message.role.wire_name(), "content": message.text }))
            .collect();

        // Thinking is always on for this model family, so the parameter is
        // omitted; depth is set through `output_config.effort`. Sampling
        // parameters are rejected at this API version.
        let body = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "output_config": { "effort": self.effort },
            "fallbacks": "default",
            "messages": wire,
        });

        let mut betas = vec![FALLBACK_BETA];
        let mut request = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("content-type", "application/json")
            .header("anthropic-version", API_VERSION);

        request = match &self.auth {
            Auth::ApiKey(key) => request.header("x-api-key", key),
            Auth::Bearer(token) => {
                betas.push(OAUTH_BETA);
                request.header("authorization", format!("Bearer {token}"))
            }
        };

        let response = request
            .header("anthropic-beta", betas.join(","))
            .json(&body)
            .send()
            .await
            .map_err(|error| AgentError::Provider(error.to_string()))?;

        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|error| AgentError::Provider(error.to_string()))?;

        if !status.is_success() {
            return Err(AgentError::Provider(format!(
                "HTTP {}: {}",
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
}
