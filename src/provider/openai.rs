//! The OpenAI Chat Completions wire.
//!
//! `POST /v1/chat/completions`, which is what OpenAI serves and what most
//! OpenAI-compatible gateways, routers and local servers imitate. Supporting
//! it alongside Anthropic's Messages API is what lets one harness reach either
//! kind of endpoint.
//!
//! Deliberately conservative about the body: `model`, `messages`, and nothing
//! else unless asked for. A token cap is only sent when a profile sets one,
//! because the field for it has changed name across model generations
//! (`max_tokens`, then `max_completion_tokens`) and guessing wrong turns a
//! working request into a 400. Omitting it lets the endpoint apply its own
//! default, which is what you want for a few hundred tokens of plan JSON.

use serde::Deserialize;
use serde_json::{json, Value};

use crate::agent::{AgentError, ChatMessage, ModelProvider};

use super::{host_of, Credential};
use crate::settings::ProviderSettings;

pub const DEFAULT_BASE_URL: &str = "https://api.openai.com";
pub const DEFAULT_MODEL: &str = "gpt-5.2";

#[derive(Debug, Clone)]
pub struct OpenAiProvider {
    client: reqwest::Client,
    base_url: String,
    token: String,
    model: String,
    max_tokens: Option<u32>,
}

impl OpenAiProvider {
    pub fn build(settings: &ProviderSettings, credential: Credential) -> Result<Self, AgentError> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(180))
            .build()
            .map_err(|error| AgentError::Provider(error.to_string()))?;

        Ok(OpenAiProvider {
            client,
            base_url: settings
                .url
                .clone()
                .unwrap_or_else(|| DEFAULT_BASE_URL.to_string())
                .trim_end_matches('/')
                .to_string(),
            token: credential.secret().to_string(),
            model: settings
                .model
                .clone()
                .unwrap_or_else(|| DEFAULT_MODEL.to_string()),
            max_tokens: settings.max_tokens,
        })
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn host(&self) -> &str {
        host_of(&self.base_url)
    }

    pub fn is_first_party(&self) -> bool {
        self.base_url == DEFAULT_BASE_URL
    }

    pub fn describe(&self) -> String {
        if self.is_first_party() {
            format!("{} via {}", self.model, self.host())
        } else {
            format!(
                "{} via {} — a third-party endpoint: prompts go to that host",
                self.model,
                self.host()
            )
        }
    }

    pub fn request_body(&self, system: &str, messages: &[ChatMessage]) -> Value {
        let mut wire: Vec<Value> = Vec::with_capacity(messages.len() + 1);
        if !system.trim().is_empty() {
            wire.push(json!({ "role": "system", "content": system }));
        }
        for message in messages {
            wire.push(json!({
                "role": message.role.wire_name(),
                "content": message.text,
            }));
        }

        let mut body = json!({ "model": self.model, "messages": wire });
        if let (Some(map), Some(max_tokens)) = (body.as_object_mut(), self.max_tokens) {
            map.insert("max_tokens".into(), json!(max_tokens));
        }
        body
    }

    fn status_error(&self, status: u16, body: &str) -> AgentError {
        let hint = match status {
            401 | 403 => " — check the key for this endpoint",
            404 => " — check the base URL, and that this endpoint serves this model",
            400 => " — the endpoint rejected the request; check the model name",
            429 => " — rate limited",
            _ => "",
        };
        AgentError::Provider(format!(
            "{} returned HTTP {status}{hint}: {}",
            self.host(),
            body.chars().take(400).collect::<String>()
        ))
    }
}

#[derive(Debug, Deserialize)]
struct Message {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    refusal: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    #[serde(default)]
    message: Option<Message>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Completion {
    #[serde(default)]
    choices: Vec<Choice>,
}

/// Pull the reply out of a completion, or say why there is none.
pub fn text_from_response(body: &str) -> Result<String, AgentError> {
    let completion: Completion = serde_json::from_str(body)
        .map_err(|error| AgentError::Provider(format!("unreadable response: {error}")))?;

    let Some(choice) = completion.choices.first() else {
        return Err(AgentError::Provider(
            "the response carried no choices".to_string(),
        ));
    };

    // A refusal arrives as a field on the message, not as an HTTP error, so
    // reading `content` alone would treat a decline as an empty reply.
    if let Some(refusal) = choice
        .message
        .as_ref()
        .and_then(|message| message.refusal.as_ref())
        .filter(|text| !text.trim().is_empty())
    {
        return Err(AgentError::Provider(format!(
            "the model declined this request: {refusal}"
        )));
    }

    let text = choice
        .message
        .as_ref()
        .and_then(|message| message.content.clone())
        .unwrap_or_default();

    if text.trim().is_empty() {
        return Err(AgentError::Provider(format!(
            "no text in the response (finish_reason {})",
            choice.finish_reason.as_deref().unwrap_or("unknown")
        )));
    }
    Ok(text)
}

#[async_trait::async_trait]
impl ModelProvider for OpenAiProvider {
    fn model(&self) -> &str {
        &self.model
    }

    fn endpoint(&self) -> &str {
        self.host()
    }

    fn is_first_party(&self) -> bool {
        OpenAiProvider::is_first_party(self)
    }

    fn describe(&self) -> String {
        OpenAiProvider::describe(self)
    }

    async fn complete(&self, system: &str, user: &str) -> Result<String, AgentError> {
        self.converse(system, &[ChatMessage::user(user)]).await
    }

    async fn converse(&self, system: &str, messages: &[ChatMessage]) -> Result<String, AgentError> {
        if messages.is_empty() {
            return Err(AgentError::Provider("no messages to send".to_string()));
        }

        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .header("content-type", "application/json")
            .bearer_auth(&self.token)
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
            return Err(self.status_error(status.as_u16(), &text));
        }
        text_from_response(&text)
    }

    async fn converse_stream(
        &self,
        system: &str,
        messages: &[ChatMessage],
        deltas: std::sync::mpsc::Sender<String>,
    ) -> Result<String, AgentError> {
        use futures_util::StreamExt;

        if messages.is_empty() {
            return Err(AgentError::Provider("no messages to send".to_string()));
        }

        let mut body = self.request_body(system, messages);
        if let Some(map) = body.as_object_mut() {
            map.insert("stream".into(), json!(true));
        }

        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .header("content-type", "application/json")
            .bearer_auth(&self.token)
            .json(&body)
            .send()
            .await
            .map_err(|error| {
                AgentError::Provider(format!("could not reach {}: {error}", self.host()))
            })?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(self.status_error(status.as_u16(), &text));
        }

        let mut stream = response.bytes_stream();
        // Bytes, not text: a multi-byte character can straddle two chunks.
        let mut buffer: Vec<u8> = Vec::new();
        let mut text = String::new();
        let mut finish: Option<String> = None;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|error| {
                AgentError::Provider(format!("stream from {} broke: {error}", self.host()))
            })?;
            buffer.extend_from_slice(&chunk);

            while let Some(at) = buffer.windows(2).position(|pair| pair == b"\n\n") {
                let event: Vec<u8> = buffer.drain(..at + 2).collect();
                let event = String::from_utf8_lossy(&event);

                for line in event.lines() {
                    let Some(data) = line.strip_prefix("data:") else {
                        continue;
                    };
                    let data = data.trim();
                    if data.is_empty() || data == "[DONE]" {
                        continue;
                    }
                    let Ok(value) = serde_json::from_str::<Value>(data) else {
                        continue;
                    };
                    let choice = &value["choices"][0];
                    if let Some(piece) = choice["delta"]["content"].as_str() {
                        if !piece.is_empty() {
                            text.push_str(piece);
                            let _ = deltas.send(piece.to_string());
                        }
                    }
                    if let Some(refusal) = choice["delta"]["refusal"].as_str() {
                        if !refusal.trim().is_empty() {
                            return Err(AgentError::Provider(format!(
                                "the model declined this request: {refusal}"
                            )));
                        }
                    }
                    if let Some(reason) = choice["finish_reason"].as_str() {
                        finish = Some(reason.to_string());
                    }
                }
            }
        }

        if text.trim().is_empty() {
            return Err(AgentError::Provider(format!(
                "the stream produced no text (finish_reason {})",
                finish.as_deref().unwrap_or("unknown")
            )));
        }
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider() -> OpenAiProvider {
        OpenAiProvider::build(
            &ProviderSettings::default(),
            Credential::Bearer("test".to_string()),
        )
        .expect("builds")
    }

    #[test]
    fn the_body_carries_the_system_prompt_as_a_message() {
        let body = provider().request_body("be brief", &[ChatMessage::user("hi")]);
        let messages = body["messages"].as_array().expect("an array");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "be brief");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(body["model"], DEFAULT_MODEL);
    }

    #[test]
    fn no_token_cap_is_sent_unless_one_was_asked_for() {
        // The field for it has changed name across model generations, so
        // guessing turns a working request into a 400.
        let body = provider().request_body("", &[ChatMessage::user("hi")]);
        assert!(body.get("max_tokens").is_none());
        assert_eq!(
            body["messages"].as_array().unwrap().len(),
            1,
            "an empty system prompt adds no message"
        );

        let capped = OpenAiProvider::build(
            &ProviderSettings {
                max_tokens: Some(256),
                ..Default::default()
            },
            Credential::Bearer("test".to_string()),
        )
        .expect("builds");
        assert_eq!(
            capped.request_body("", &[ChatMessage::user("hi")])["max_tokens"],
            256
        );
    }

    #[test]
    fn a_reply_is_read_from_the_first_choice() {
        let body = r#"{"choices":[{"message":{"role":"assistant","content":"hello"},
                        "finish_reason":"stop"}]}"#;
        assert_eq!(text_from_response(body).unwrap(), "hello");
    }

    #[test]
    fn a_refusal_field_is_an_error_not_an_empty_reply() {
        let body = r#"{"choices":[{"message":{"role":"assistant","content":null,
                        "refusal":"I cannot help with that"},"finish_reason":"stop"}]}"#;
        let error = text_from_response(body).expect_err("a refusal must not read as success");
        assert!(error.to_string().contains("cannot help"), "{error}");
    }

    #[test]
    fn an_empty_or_malformed_response_is_an_error() {
        assert!(text_from_response(r#"{"choices":[]}"#).is_err());
        assert!(text_from_response(
            r#"{"choices":[{"message":{"content":""},"finish_reason":"length"}]}"#
        )
        .is_err());
        assert!(text_from_response("not json").is_err());
    }

    #[test]
    fn a_custom_endpoint_is_flagged_and_its_host_named() {
        let custom = OpenAiProvider::build(
            &ProviderSettings {
                url: Some("https://gw.test/".to_string()),
                model: Some("gpt-4o".to_string()),
                ..Default::default()
            },
            Credential::Bearer("test".to_string()),
        )
        .expect("builds");
        assert_eq!(custom.base_url(), "https://gw.test");
        assert_eq!(custom.host(), "gw.test");
        assert!(!custom.is_first_party());
        assert!(custom.describe().contains("third-party"));
        assert!(provider().describe().contains("api.openai.com"));
        assert!(!provider().describe().contains("third-party"));
    }
}
