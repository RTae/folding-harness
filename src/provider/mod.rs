//! Talking to a model.
//!
//! Two wire formats are supported, because endpoints in this space speak one
//! or the other: Anthropic's Messages API ([`anthropic`]) and OpenAI's Chat
//! Completions ([`openai`]). A profile says which, so one harness reaches both
//! first-party APIs and the many gateways that imitate one of them.
//!
//! # Credentials
//!
//! Never stored by this harness. A profile says where to *find* a credential,
//! not what it is:
//!
//! * a command whose standard output is the token, which is how an official
//!   login CLI or a password manager is used, or
//! * the name of an environment variable.
//!
//! Running a command named in a config file is a real capability, so it is
//! only ever one the user typed themselves, and the file holds the command
//! rather than the secret.

pub mod anthropic;
pub mod openai;
pub mod sse;

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::agent::{AgentError, ModelProvider};
use crate::settings::ProviderSettings;

pub use anthropic::{Provider, API_VERSION, DEFAULT_BASE_URL, DEFAULT_MODEL};
pub use openai::OpenAiProvider;

/// Kept so existing callers keep working.
pub type AnthropicProvider = Provider;

/// Which request and response shape an endpoint speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Wire {
    /// `POST /v1/messages`, Anthropic's Messages API.
    #[default]
    Anthropic,
    /// `POST /v1/chat/completions`, OpenAI's Chat Completions. Also what most
    /// OpenAI-compatible gateways and local servers speak.
    Openai,
}

impl Wire {
    pub fn as_str(&self) -> &'static str {
        match self {
            Wire::Anthropic => "anthropic",
            Wire::Openai => "openai",
        }
    }

    pub fn parse(name: &str) -> Option<Wire> {
        match name.trim().to_ascii_lowercase().as_str() {
            "anthropic" | "claude" | "messages" => Some(Wire::Anthropic),
            "openai" | "chatgpt" | "gpt" | "chat-completions" => Some(Wire::Openai),
            _ => None,
        }
    }

    pub fn names() -> &'static str {
        "anthropic, openai"
    }

    /// Where this wire goes when a profile names no URL.
    pub fn default_base_url(&self) -> &'static str {
        match self {
            Wire::Anthropic => DEFAULT_BASE_URL,
            Wire::Openai => openai::DEFAULT_BASE_URL,
        }
    }

    pub fn default_model(&self) -> &'static str {
        match self {
            Wire::Anthropic => DEFAULT_MODEL,
            Wire::Openai => openai::DEFAULT_MODEL,
        }
    }

    /// Environment variables checked when a profile names none, in order.
    pub fn default_key_envs(&self) -> &'static [&'static str] {
        match self {
            Wire::Anthropic => &["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"],
            Wire::Openai => &["OPENAI_API_KEY"],
        }
    }
}

/// A credential, once resolved. Held only for the life of the process.
#[derive(Debug, Clone)]
pub enum Credential {
    /// Sent as `x-api-key`, Anthropic's scheme.
    ApiKey(String),
    /// Sent as `Authorization: Bearer`.
    Bearer(String),
}

impl Credential {
    pub fn secret(&self) -> &str {
        match self {
            Credential::ApiKey(value) | Credential::Bearer(value) => value,
        }
    }
}

/// Run a command and take its standard output as a token.
///
/// This is how an official login is used without this harness reimplementing
/// OAuth: `ant auth print-credentials --access-token` exists for exactly this,
/// and a password manager works the same way.
pub fn token_from_command(command: &str) -> Result<String, AgentError> {
    let trimmed = command.trim();
    if trimmed.is_empty() {
        return Err(AgentError::Provider("empty credential command".to_string()));
    }

    // Split on whitespace rather than going through a shell: a config value
    // should not be able to chain commands or redirect.
    let mut parts = trimmed.split_whitespace();
    let program = parts.next().unwrap_or_default();
    let output = std::process::Command::new(program)
        .args(parts)
        .output()
        .map_err(|error| AgentError::Provider(format!("could not run '{program}': {error}")))?;

    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr);
        return Err(AgentError::Provider(format!(
            "'{trimmed}' failed: {}",
            detail.trim().chars().take(200).collect::<String>()
        )));
    }

    let token = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if token.is_empty() {
        return Err(AgentError::Provider(format!(
            "'{trimmed}' produced no output"
        )));
    }
    Ok(token)
}

/// Find a credential for these settings.
///
/// Order: the profile's command, then the profile's named variable, then the
/// wire's usual variables.
pub fn resolve_credential(
    settings: &ProviderSettings,
    wire: Wire,
) -> Result<Credential, AgentError> {
    if let Some(command) = &settings.key_command {
        let token = token_from_command(command)?;
        // A token fetched from a login flow is a bearer token; an API key
        // pasted through a command is still accepted by the same header on
        // the OpenAI wire, and by `x-api-key` on Anthropic's.
        return Ok(match wire {
            Wire::Anthropic if token.starts_with("sk-ant-") => Credential::ApiKey(token),
            Wire::Anthropic => Credential::Bearer(token),
            Wire::Openai => Credential::Bearer(token),
        });
    }

    let mut names: Vec<&str> = Vec::new();
    if let Some(name) = &settings.key_env {
        names.push(name);
    }
    names.extend_from_slice(wire.default_key_envs());

    for name in &names {
        if let Ok(value) = std::env::var(name) {
            let value = value.trim().to_string();
            if value.is_empty() {
                continue;
            }
            return Ok(match (wire, *name) {
                (Wire::Anthropic, "ANTHROPIC_AUTH_TOKEN") => Credential::Bearer(value),
                (Wire::Anthropic, _) => Credential::ApiKey(value),
                (Wire::Openai, _) => Credential::Bearer(value),
            });
        }
    }

    Err(AgentError::Provider(format!(
        "no credential for the {} wire. Set {}, or give the profile a command \
         with --key-command (for example `ant auth print-credentials --access-token`).",
        wire.as_str(),
        names.join(" or ")
    )))
}

/// Build a provider for these settings.
pub fn build(settings: &ProviderSettings) -> Result<Arc<dyn ModelProvider>, AgentError> {
    let wire = settings.wire.unwrap_or_default();
    let credential = resolve_credential(settings, wire)?;

    Ok(match wire {
        Wire::Anthropic => Arc::new(anthropic::Provider::build(settings, credential)?),
        Wire::Openai => Arc::new(openai::OpenAiProvider::build(settings, credential)?),
    })
}

/// One line naming the model and host, flagging a host that is not the
/// vendor's own.
///
/// Both wires need the same sentence, and a caller deciding whether prompts
/// are leaving the first-party API should not depend on which wire it is.
pub fn describe_endpoint(model: &str, base_url: &str, first_party: bool, note: &str) -> String {
    let host = host_of(base_url);
    if first_party {
        format!("{model} via {host}")
    } else {
        format!("{model} via {host} — a third-party endpoint: prompts go to that host{note}")
    }
}

/// An HTTP failure, with a hint about the likely cause.
///
/// The hints matter most on a custom endpoint, where a 404 is far more likely
/// to be a wrong URL than a wrong request.
pub fn status_error(host: &str, status: u16, body: &str, first_party: bool) -> AgentError {
    let hint = match status {
        401 | 403 => " — check the key for this endpoint",
        404 => " — check the base URL, and that this endpoint serves this model",
        400 if !first_party => {
            " — the endpoint rejected the request; check the model name, and try --api-version"
        }
        400 => " — the endpoint rejected the request; check the model name",
        429 => " — rate limited",
        _ => "",
    };
    AgentError::Provider(format!(
        "{host} returned HTTP {status}{hint}: {}",
        body.chars().take(400).collect::<String>()
    ))
}

/// Host part of a URL, for printing.
pub fn host_of(url: &str) -> &str {
    url.split("://")
        .nth(1)
        .unwrap_or(url)
        .split('/')
        .next()
        .unwrap_or(url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_names_round_trip_and_accept_the_obvious_aliases() {
        assert_eq!(Wire::parse("anthropic"), Some(Wire::Anthropic));
        assert_eq!(Wire::parse("claude"), Some(Wire::Anthropic));
        assert_eq!(Wire::parse("OpenAI"), Some(Wire::Openai));
        assert_eq!(Wire::parse("chatgpt"), Some(Wire::Openai));
        assert_eq!(Wire::parse("gpt"), Some(Wire::Openai));
        assert_eq!(Wire::parse("nonsense"), None);
        assert_eq!(Wire::default(), Wire::Anthropic);
    }

    #[test]
    fn each_wire_knows_its_own_defaults() {
        assert!(Wire::Anthropic.default_base_url().contains("anthropic"));
        assert!(Wire::Openai.default_base_url().contains("openai"));
        assert!(Wire::Anthropic
            .default_key_envs()
            .contains(&"ANTHROPIC_API_KEY"));
        assert!(Wire::Openai.default_key_envs().contains(&"OPENAI_API_KEY"));
    }

    #[test]
    fn a_credential_command_is_not_run_through_a_shell() {
        // A value from a config file must not be able to chain commands, so
        // the arguments go to the program rather than to a shell.
        let token = token_from_command("echo first; echo second").expect("echo runs");
        // Proof that no shell ran: `;` and the second `echo` were handed to
        // the first `echo` as ordinary arguments and printed literally. A
        // shell would have produced "first\nsecond" instead.
        assert_eq!(token, "first; echo second");
    }

    #[test]
    fn a_failing_or_silent_credential_command_is_an_error() {
        assert!(token_from_command("").is_err());
        assert!(token_from_command("definitely-not-a-program-9f3a").is_err());
        assert!(token_from_command("true").is_err(), "no output is an error");
    }

    #[test]
    fn a_credential_command_wins_over_the_environment() {
        let settings = ProviderSettings {
            key_command: Some("echo from-command".to_string()),
            ..Default::default()
        };
        let credential = resolve_credential(&settings, Wire::Openai).expect("resolves");
        assert_eq!(credential.secret(), "from-command");
    }

    #[test]
    fn a_missing_credential_says_what_to_set() {
        let settings = ProviderSettings {
            key_env: Some("FH_TEST_ABSENT_KEY".to_string()),
            ..Default::default()
        };
        std::env::remove_var("FH_TEST_ABSENT_KEY");
        // Only meaningful when the usual variables are also unset.
        if std::env::var("ANTHROPIC_API_KEY").is_ok()
            || std::env::var("ANTHROPIC_AUTH_TOKEN").is_ok()
        {
            return;
        }
        let error = resolve_credential(&settings, Wire::Anthropic).expect_err("none set");
        let message = error.to_string();
        assert!(message.contains("FH_TEST_ABSENT_KEY"), "{message}");
        assert!(message.contains("--key-command"), "{message}");
    }

    #[test]
    fn build_honours_the_profile_s_wire() {
        // Building by hand per wire is how a caller ends up pointing an
        // Anthropic client at an OpenAI endpoint, so one function decides.
        let openai = ProviderSettings {
            wire: Some(Wire::Openai),
            key_command: Some("echo token".to_string()),
            model: Some("gpt-4o".to_string()),
            ..Default::default()
        };
        let built = build(&openai).expect("builds");
        assert_eq!(built.model(), "gpt-4o");
        assert_eq!(built.endpoint(), "api.openai.com");

        let anthropic = ProviderSettings {
            wire: Some(Wire::Anthropic),
            key_command: Some("echo sk-ant-token".to_string()),
            model: Some("claude-opus-5-5".to_string()),
            ..Default::default()
        };
        let built = build(&anthropic).expect("builds");
        assert_eq!(built.endpoint(), "api.anthropic.com");

        // No wire named means the Anthropic default.
        let bare = ProviderSettings {
            key_command: Some("echo token".to_string()),
            ..Default::default()
        };
        assert_eq!(
            build(&bare).expect("builds").endpoint(),
            "api.anthropic.com"
        );
    }

    #[test]
    fn a_wire_reaches_the_endpoint_the_profile_names() {
        let settings = ProviderSettings {
            wire: Some(Wire::Openai),
            url: Some("https://gw.test".to_string()),
            key_command: Some("echo token".to_string()),
            ..Default::default()
        };
        assert_eq!(build(&settings).expect("builds").endpoint(), "gw.test");
    }

    #[test]
    fn hosts_are_extracted_for_printing() {
        assert_eq!(host_of("https://api.openai.com"), "api.openai.com");
        assert_eq!(host_of("https://gw.test/v1/path"), "gw.test");
        assert_eq!(host_of("gw.test"), "gw.test");
    }
}
