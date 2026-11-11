//! Remembered provider settings.
//!
//! Typing `--url https://gateway.example.com --model claude-opus-4-7` on every
//! command is the kind of thing a tool should do for you. These settings live
//! in a file, and a flag or an environment variable still wins when you want a
//! one-off.
//!
//! Precedence, highest first: a command-line flag, then the environment, then
//! this file, then the built-in default. That order means the file is a
//! convenience and never a trap: whatever you type on the command line is what
//! happens.
//!
//! **No API key is stored here.** Credentials come from `ANTHROPIC_API_KEY` or
//! `ANTHROPIC_AUTH_TOKEN` and are never written to disk by this harness. A
//! config file is the wrong place for a secret: it outlives the session, it is
//! easy to copy, and it is easy to commit by accident.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// What the harness remembers about where to send prompts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProviderSettings {
    pub url: Option<String>,
    pub model: Option<String>,
    pub api_version: Option<String>,
    pub effort: Option<String>,
    pub max_tokens: Option<u32>,
    /// Which request shape this endpoint speaks.
    pub wire: Option<crate::provider::Wire>,
    /// Environment variable holding the credential.
    pub key_env: Option<String>,
    /// Command whose standard output is the credential. This is how an
    /// official login CLI is used; the command is stored, never the secret.
    pub key_command: Option<String>,
}

impl ProviderSettings {
    pub fn is_empty(&self) -> bool {
        *self == ProviderSettings::default()
    }

    /// Apply `other` over `self`, field by field. Anything set in `other`
    /// wins; anything unset leaves `self` alone.
    pub fn overlay(&self, other: &ProviderSettings) -> ProviderSettings {
        ProviderSettings {
            url: other.url.clone().or_else(|| self.url.clone()),
            model: other.model.clone().or_else(|| self.model.clone()),
            api_version: other
                .api_version
                .clone()
                .or_else(|| self.api_version.clone()),
            effort: other.effort.clone().or_else(|| self.effort.clone()),
            max_tokens: other.max_tokens.or(self.max_tokens),
            wire: other.wire.or(self.wire),
            key_env: other.key_env.clone().or_else(|| self.key_env.clone()),
            key_command: other
                .key_command
                .clone()
                .or_else(|| self.key_command.clone()),
        }
    }

    /// What the environment asks for.
    pub fn from_env() -> ProviderSettings {
        let read = |name: &str| {
            std::env::var(name)
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        };
        ProviderSettings {
            url: read("ANTHROPIC_BASE_URL"),
            model: read("ANTHROPIC_MODEL"),
            api_version: read("ANTHROPIC_API_VERSION"),
            effort: read("ANTHROPIC_EFFORT"),
            max_tokens: read("ANTHROPIC_MAX_TOKENS").and_then(|value| value.parse().ok()),
            wire: read("FH_WIRE").and_then(|value| crate::provider::Wire::parse(&value)),
            key_env: None,
            key_command: None,
        }
    }

    /// Lines for `fh config show`, saying where each value came from.
    pub fn describe(saved: &ProviderSettings, env: &ProviderSettings) -> Vec<String> {
        let effective = saved.overlay(env);
        let line = |name: &str, value: Option<String>, from_env: bool, from_file: bool| {
            let source = if from_env {
                "environment"
            } else if from_file {
                "saved"
            } else {
                "default"
            };
            format!(
                "{:<12} {:<34} {}",
                name,
                value.unwrap_or_else(|| "(unset)".to_string()),
                source
            )
        };
        vec![
            line(
                "url",
                effective.url.clone(),
                env.url.is_some(),
                saved.url.is_some(),
            ),
            line(
                "model",
                effective.model.clone(),
                env.model.is_some(),
                saved.model.is_some(),
            ),
            line(
                "api_version",
                effective.api_version.clone(),
                env.api_version.is_some(),
                saved.api_version.is_some(),
            ),
            line(
                "effort",
                effective.effort.clone(),
                env.effort.is_some(),
                saved.effort.is_some(),
            ),
            line(
                "max_tokens",
                effective.max_tokens.map(|value| value.to_string()),
                env.max_tokens.is_some(),
                saved.max_tokens.is_some(),
            ),
            line(
                "wire",
                Some(effective.wire.unwrap_or_default().as_str().to_string()),
                env.wire.is_some(),
                saved.wire.is_some(),
            ),
            line(
                "key_env",
                effective.key_env.clone(),
                false,
                saved.key_env.is_some(),
            ),
            line(
                "key_command",
                effective.key_command.clone(),
                false,
                saved.key_command.is_some(),
            ),
        ]
    }
}

/// The built-in profile, always offered so the list is never empty.
pub const DEFAULT_PROFILE: &str = "anthropic";

/// Named provider profiles, and which one is in use.
///
/// One endpoint is the common case, but a lab usually has at least two: the
/// first-party API and whatever gateway the group pays for. Naming them means
/// switching is a word rather than a URL and a model id remembered correctly.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProviderConfig {
    /// Which profile is in use. `None` means the built-in default.
    pub active: Option<String>,
    pub profiles: std::collections::BTreeMap<String, ProviderSettings>,
}

impl ProviderConfig {
    /// Profile names, with the built-in default always present, sorted.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.profiles.keys().cloned().collect();
        if !names.iter().any(|name| name == DEFAULT_PROFILE) {
            names.push(DEFAULT_PROFILE.to_string());
        }
        names.sort();
        names
    }

    pub fn active_name(&self) -> String {
        match &self.active {
            Some(name) if self.profiles.contains_key(name) => name.clone(),
            Some(name) if name == DEFAULT_PROFILE => name.clone(),
            // A profile that was removed must not strand the configuration.
            _ => DEFAULT_PROFILE.to_string(),
        }
    }

    /// Settings of the active profile. The built-in default is empty, which
    /// means the first-party API and the default model.
    pub fn active_settings(&self) -> ProviderSettings {
        self.profiles
            .get(&self.active_name())
            .cloned()
            .unwrap_or_default()
    }

    pub fn get(&self, name: &str) -> Option<&ProviderSettings> {
        self.profiles.get(name)
    }

    /// One line per profile for a listing or a picker.
    pub fn summary(&self, name: &str) -> String {
        let settings = self.profiles.get(name).cloned().unwrap_or_default();
        let default_host = settings
            .wire
            .unwrap_or_default()
            .default_base_url()
            .to_string();
        let endpoint = settings
            .url
            .as_deref()
            .map(|url| {
                url.split("://")
                    .nth(1)
                    .unwrap_or(url)
                    .split('/')
                    .next()
                    .unwrap_or(url)
                    .to_string()
            })
            .unwrap_or_else(|| crate::provider::host_of(&default_host).to_string());
        let model = settings
            .model
            .clone()
            .unwrap_or_else(|| "(default model)".to_string());
        let wire = settings.wire.unwrap_or_default();
        format!("{model} via {endpoint} [{}]", wire.as_str())
    }
}

/// `$FH_CONFIG`, or `~/.folding_harness/provider.json`.
pub fn config_path() -> PathBuf {
    if let Ok(custom) = std::env::var("FH_CONFIG") {
        return PathBuf::from(custom);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home)
        .join(".folding_harness")
        .join("provider.json")
}

/// Read the whole configuration. A missing or unreadable file is not an
/// error: there is a working default behind it.
pub fn load_config() -> ProviderConfig {
    load_config_from(&config_path())
}

pub fn load_config_from(path: &std::path::Path) -> ProviderConfig {
    if let Ok(config) = crate::trace::read_json::<ProviderConfig>(path) {
        return config;
    }
    // An earlier version wrote a single flat setting. Keep it rather than
    // silently reverting someone to the defaults.
    match crate::trace::read_json::<ProviderSettings>(path) {
        Ok(flat) if !flat.is_empty() => {
            let mut profiles = std::collections::BTreeMap::new();
            profiles.insert("saved".to_string(), flat);
            ProviderConfig {
                active: Some("saved".to_string()),
                profiles,
            }
        }
        _ => ProviderConfig::default(),
    }
}

/// Settings of whichever profile is active.
pub fn load() -> ProviderSettings {
    load_config().active_settings()
}

pub fn load_from(path: &std::path::Path) -> ProviderSettings {
    load_config_from(path).active_settings()
}

pub fn save_config(config: &ProviderConfig) -> std::io::Result<PathBuf> {
    let path = config_path();
    save_config_to(&path, config)?;
    Ok(path)
}

pub fn save_config_to(path: &std::path::Path, config: &ProviderConfig) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    crate::trace::write_json(path, config)
}

/// Save settings into the active profile, creating it if needed.
pub fn save(settings: &ProviderSettings) -> std::io::Result<PathBuf> {
    let mut config = load_config();
    let name = config.active_name();
    config.profiles.insert(name.clone(), settings.clone());
    config.active = Some(name);
    save_config(&config)
}

pub fn save_to(path: &std::path::Path, settings: &ProviderSettings) -> std::io::Result<()> {
    let mut config = load_config_from(path);
    let name = config.active_name();
    config.profiles.insert(name.clone(), settings.clone());
    config.active = Some(name);
    save_config_to(path, &config)
}

pub fn clear() -> std::io::Result<()> {
    let path = config_path();
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(url: Option<&str>, model: Option<&str>) -> ProviderSettings {
        ProviderSettings {
            url: url.map(str::to_string),
            model: model.map(str::to_string),
            ..Default::default()
        }
    }

    #[test]
    fn an_overlay_fills_gaps_without_overwriting_what_is_set() {
        let saved = settings(Some("https://gateway"), Some("opus-4-7"));
        let flags = settings(None, Some("opus-5-5"));
        let merged = saved.overlay(&flags);

        assert_eq!(merged.url.as_deref(), Some("https://gateway"));
        assert_eq!(merged.model.as_deref(), Some("opus-5-5"), "the flag wins");
    }

    #[test]
    fn an_empty_overlay_changes_nothing() {
        let saved = settings(Some("https://gateway"), Some("opus-4-7"));
        assert_eq!(saved.overlay(&ProviderSettings::default()), saved);
        assert!(ProviderSettings::default().is_empty());
        assert!(!saved.is_empty());
    }

    #[test]
    fn settings_round_trip_through_disk() {
        let dir = std::env::temp_dir().join(format!("fh-settings-{}", crate::trace::now_ms()));
        let path = dir.join("provider.json");

        // Nothing saved yet reads as nothing set, not as an error.
        assert!(load_from(&path).is_empty());

        let mut original = settings(Some("https://gateway"), Some("opus-4-7"));
        original.max_tokens = Some(4096);
        save_to(&path, &original).expect("saves");
        assert_eq!(load_from(&path), original);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_corrupt_file_falls_back_to_defaults_rather_than_failing() {
        let dir = std::env::temp_dir().join(format!("fh-settings-bad-{}", crate::trace::now_ms()));
        std::fs::create_dir_all(&dir).expect("dir");
        let path = dir.join("provider.json");
        std::fs::write(&path, "not json at all").expect("writes");

        assert!(load_from(&path).is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_built_in_profile_is_always_offered() {
        let config = ProviderConfig::default();
        assert_eq!(config.names(), vec![DEFAULT_PROFILE]);
        assert_eq!(config.active_name(), DEFAULT_PROFILE);
        assert!(config.active_settings().is_empty(), "default means default");
    }

    #[test]
    fn profiles_are_listed_and_switched_by_name() {
        let mut config = ProviderConfig::default();
        config.profiles.insert(
            "gateway".to_string(),
            settings(Some("https://gw.test"), Some("opus-4-7")),
        );
        config.active = Some("gateway".to_string());

        assert_eq!(config.names(), vec!["anthropic", "gateway"]);
        assert_eq!(config.active_name(), "gateway");
        assert_eq!(config.active_settings().model.as_deref(), Some("opus-4-7"));
        assert!(config.summary("gateway").contains("gw.test"));
        assert!(config
            .summary(DEFAULT_PROFILE)
            .contains("api.anthropic.com"));
    }

    #[test]
    fn an_active_profile_that_was_removed_falls_back_rather_than_stranding() {
        let mut config = ProviderConfig {
            active: Some("gone".to_string()),
            ..Default::default()
        };
        assert_eq!(config.active_name(), DEFAULT_PROFILE);
        assert!(config.active_settings().is_empty());

        // And the same once other profiles exist.
        config.profiles.insert(
            "gateway".to_string(),
            settings(Some("https://gw.test"), None),
        );
        assert_eq!(config.active_name(), DEFAULT_PROFILE);
    }

    #[test]
    fn an_older_flat_config_file_is_kept_rather_than_discarded() {
        let dir =
            std::env::temp_dir().join(format!("fh-settings-migrate-{}", crate::trace::now_ms()));
        std::fs::create_dir_all(&dir).expect("dir");
        let path = dir.join("provider.json");

        // What the previous version wrote.
        std::fs::write(
            &path,
            r#"{"url":"https://gw.test","model":"opus-4-7","api_version":null,"effort":null,"max_tokens":null}"#,
        )
        .expect("writes");

        let config = load_config_from(&path);
        assert_eq!(config.active_name(), "saved");
        assert_eq!(
            config.active_settings().url.as_deref(),
            Some("https://gw.test"),
            "the earlier setting must survive the upgrade"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn saving_writes_into_the_active_profile() {
        let dir =
            std::env::temp_dir().join(format!("fh-settings-active-{}", crate::trace::now_ms()));
        let path = dir.join("provider.json");

        let mut config = ProviderConfig {
            active: Some("gateway".to_string()),
            ..Default::default()
        };
        config
            .profiles
            .insert("gateway".to_string(), ProviderSettings::default());
        save_config_to(&path, &config).expect("saves");

        save_to(&path, &settings(Some("https://gw.test"), Some("opus-4-7"))).expect("saves");

        let back = load_config_from(&path);
        assert_eq!(back.active_name(), "gateway");
        assert_eq!(
            back.get("gateway").and_then(|s| s.model.clone()).as_deref(),
            Some("opus-4-7")
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_description_says_where_each_value_came_from() {
        let saved = settings(Some("https://saved"), Some("saved-model"));
        let env = settings(None, Some("env-model"));
        let lines = ProviderSettings::describe(&saved, &env);

        let url = lines.iter().find(|line| line.starts_with("url")).unwrap();
        assert!(url.contains("https://saved"), "{url}");
        assert!(url.ends_with("saved"), "{url}");

        let model = lines.iter().find(|line| line.starts_with("model")).unwrap();
        assert!(model.contains("env-model"), "the environment wins: {model}");
        assert!(model.ends_with("environment"), "{model}");

        let effort = lines
            .iter()
            .find(|line| line.starts_with("effort"))
            .unwrap();
        assert!(effort.contains("(unset)"), "{effort}");
        assert!(effort.ends_with("default"), "{effort}");
    }

    #[test]
    fn no_field_can_hold_a_credential() {
        // A key in the config file would outlive the session and is easy to
        // commit by accident, so the type has nowhere to put one. Check the
        // field set exactly rather than by substring: `max_tokens` contains
        // "token" and would pass a looser test.
        let json = serde_json::to_value(ProviderSettings::default()).unwrap();
        let mut fields: Vec<&str> = json
            .as_object()
            .expect("an object")
            .keys()
            .map(String::as_str)
            .collect();
        fields.sort_unstable();
        assert_eq!(
            fields,
            [
                "api_version",
                "effort",
                "key_command",
                "key_env",
                "max_tokens",
                "model",
                "url",
                "wire"
            ]
        );
        // `key_env` and `key_command` say where to find a credential. Neither
        // holds one, which is the whole point of them.

        // And a file that tries to smuggle one in is rejected outright.
        for smuggled in [
            r#"{"url":"https://gateway","api_key":"sk-secret"}"#,
            r#"{"auth_token":"sk-secret"}"#,
            r#"{"key":"sk-secret"}"#,
        ] {
            assert!(
                serde_json::from_str::<ProviderSettings>(smuggled).is_err(),
                "{smuggled} should have been refused"
            );
        }
    }
}
