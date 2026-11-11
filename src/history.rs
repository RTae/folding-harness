//! Remembering chats, and tying them to the runs they produced.
//!
//! A design run already leaves a full record in its task directory. A
//! conversation left nothing: close the chat screen and the reasoning behind a
//! plan was gone, even though the run it started was carefully recorded. This
//! stores the conversation beside the runs and links the two, so "which
//! conversation produced this run, and what did we ask for" has an answer.
//!
//! A record is written after every exchange rather than on exit, so a crash or
//! a closed terminal does not lose the session.
//!
//! The record holds the sequences you supplied, so a resumed chat can validate
//! plans without being handed the FASTA files again. They are already on disk
//! in the task config; this is the same material in the same place.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::agent::{ChatMessage, ChatRole};
use crate::plan::Chain;
use crate::trace::{self, now_ms};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredRole {
    User,
    Assistant,
}

impl From<ChatRole> for StoredRole {
    fn from(role: ChatRole) -> Self {
        match role {
            ChatRole::User => StoredRole::User,
            ChatRole::Assistant => StoredRole::Assistant,
        }
    }
}

impl From<StoredRole> for ChatRole {
    fn from(role: StoredRole) -> Self {
        match role {
            StoredRole::User => ChatRole::User,
            StoredRole::Assistant => ChatRole::Assistant,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredMessage {
    pub role: StoredRole,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredChain {
    pub label: String,
    pub sequence: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRecord {
    pub id: String,
    pub started_at_ms: u64,
    pub updated_at_ms: u64,
    pub model: String,
    pub endpoint: String,
    pub seed: u64,
    pub binder: Vec<StoredChain>,
    pub target: Vec<StoredChain>,
    pub messages: Vec<StoredMessage>,
    /// Runs started from this conversation, newest last.
    #[serde(default)]
    pub task_ids: Vec<String>,
    /// Name of the last plan that validated.
    #[serde(default)]
    pub plan_name: Option<String>,
}

impl ChatRecord {
    pub fn new(model: &str, endpoint: &str, seed: u64, binder: &[Chain], target: &[Chain]) -> Self {
        let now = now_ms();
        let store = |chains: &[Chain]| -> Vec<StoredChain> {
            chains
                .iter()
                .map(|chain| StoredChain {
                    label: chain.label.clone(),
                    sequence: chain.sequence.clone(),
                })
                .collect()
        };
        ChatRecord {
            id: format!("{now}"),
            started_at_ms: now,
            updated_at_ms: now,
            model: model.to_string(),
            endpoint: endpoint.to_string(),
            seed,
            binder: store(binder),
            target: store(target),
            messages: Vec::new(),
            task_ids: Vec::new(),
            plan_name: None,
        }
    }

    pub fn set_messages(&mut self, messages: &[ChatMessage]) {
        self.messages = messages
            .iter()
            .map(|message| StoredMessage {
                role: message.role.into(),
                text: message.text.clone(),
            })
            .collect();
        self.updated_at_ms = now_ms();
    }

    pub fn to_messages(&self) -> Vec<ChatMessage> {
        self.messages
            .iter()
            .map(|message| ChatMessage {
                role: message.role.into(),
                text: message.text.clone(),
            })
            .collect()
    }

    pub fn binder_chains(&self) -> Vec<Chain> {
        self.binder
            .iter()
            .map(|chain| Chain::new(chain.label.clone(), chain.sequence.clone()))
            .collect()
    }

    pub fn target_chains(&self) -> Vec<Chain> {
        self.target
            .iter()
            .map(|chain| Chain::new(chain.label.clone(), chain.sequence.clone()))
            .collect()
    }

    /// How many turns the user took, which is the useful count in a listing.
    pub fn user_turns(&self) -> usize {
        self.messages
            .iter()
            .filter(|message| message.role == StoredRole::User)
            .count()
    }

    pub fn save(&self, root: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(root)?;
        trace::write_json(root.join(format!("{}.json", self.id)), self)
    }
}

/// `$FH_CHAT_ROOT`, or `~/.folding_harness/chats`.
pub fn chat_root() -> PathBuf {
    if let Ok(custom) = std::env::var("FH_CHAT_ROOT") {
        return PathBuf::from(custom);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".folding_harness").join("chats")
}

/// Every stored chat, newest first.
pub fn list(root: &Path) -> std::io::Result<Vec<ChatRecord>> {
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut found: Vec<ChatRecord> = std::fs::read_dir(root)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().map(|ext| ext == "json").unwrap_or(false))
        .filter_map(|path| trace::read_json::<ChatRecord>(path).ok())
        .collect();
    found.sort_by_key(|record| std::cmp::Reverse(record.updated_at_ms));
    Ok(found)
}

/// Load one chat by id, or the most recent when `id` is `None`.
pub fn load(root: &Path, id: Option<&str>) -> std::io::Result<ChatRecord> {
    match id {
        Some(id) => {
            // The id becomes a filename, so it has to be a plain name.
            let safe = !id.is_empty()
                && id.len() <= 128
                && id
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
            if !safe {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("'{id}' is not a chat id"),
                ));
            }
            trace::read_json(root.join(format!("{id}.json")))
        }
        None => list(root)?
            .into_iter()
            .next()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "no saved chats")),
    }
}

/// Note that a run was started from a chat.
pub fn record_run(root: &Path, chat_id: &str, task_id: &str) -> std::io::Result<()> {
    let mut record = load(root, Some(chat_id))?;
    if !record.task_ids.iter().any(|existing| existing == task_id) {
        record.task_ids.push(task_id.to_string());
    }
    record.updated_at_ms = now_ms();
    record.save(root)
}

/// "4m ago", "3h ago", "2d ago". Rough on purpose; a listing wants a glance.
pub fn ago(then_ms: u64) -> String {
    let now = now_ms();
    let seconds = now.saturating_sub(then_ms) / 1000;
    if seconds < 60 {
        return "just now".to_string();
    }
    let minutes = seconds / 60;
    if minutes < 60 {
        return format!("{minutes}m ago");
    }
    let hours = minutes / 60;
    if hours < 48 {
        return format!("{hours}h ago");
    }
    format!("{}d ago", hours / 24)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fh-history-{name}-{}", now_ms()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    fn record() -> ChatRecord {
        ChatRecord::new(
            "claude-opus-4-7",
            "kitcoding.com",
            7,
            &[Chain::new("H", "QVQLVESG")],
            &[Chain::new("A", "MKTVRQ")],
        )
    }

    #[test]
    fn a_chat_round_trips_through_disk() {
        let root = temp_root("roundtrip");
        let mut original = record();
        original.set_messages(&[
            ChatMessage::user("design the loops"),
            ChatMessage::assistant("{\"topology\":\"vhh\"}"),
        ]);
        original.save(&root).expect("saves");

        let loaded = load(&root, Some(&original.id)).expect("loads");
        assert_eq!(loaded, original);
        assert_eq!(loaded.user_turns(), 1);

        // The sequences come back, so a resumed chat can validate plans.
        assert_eq!(loaded.binder_chains()[0].sequence, "QVQLVESG");
        assert_eq!(loaded.target_chains()[0].label, "A");

        // And the messages come back as the agent's own type.
        let messages = loaded.to_messages();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, ChatRole::User);
        assert_eq!(messages[1].role, ChatRole::Assistant);

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn listing_puts_the_most_recent_first_and_load_defaults_to_it() {
        let root = temp_root("listing");
        let mut older = record();
        older.id = "1000".to_string();
        older.updated_at_ms = 1000;
        older.save(&root).expect("saves");

        let mut newer = record();
        newer.id = "2000".to_string();
        newer.updated_at_ms = 2000;
        newer.save(&root).expect("saves");

        let all = list(&root).expect("lists");
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].id, "2000");

        assert_eq!(load(&root, None).expect("loads newest").id, "2000");
        assert_eq!(load(&root, Some("1000")).expect("loads by id").id, "1000");

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_chat_id_cannot_escape_the_chat_directory() {
        let root = temp_root("traversal");
        for attempt in ["..", "../secrets", "/etc/passwd", "a/b", "a.b", ""] {
            assert!(
                load(&root, Some(attempt)).is_err(),
                "{attempt:?} should have been refused"
            );
        }
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn an_empty_directory_lists_nothing_rather_than_failing() {
        let root = temp_root("empty");
        assert!(list(&root).expect("lists").is_empty());
        assert!(load(&root, None).is_err());
        // A directory that was never created is also fine.
        assert!(list(&root.join("missing")).expect("lists").is_empty());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_run_is_recorded_against_its_chat_once() {
        let root = temp_root("link");
        let original = record();
        original.save(&root).expect("saves");

        record_run(&root, &original.id, "1700-demo").expect("records");
        record_run(&root, &original.id, "1700-demo").expect("records again");
        record_run(&root, &original.id, "1800-other").expect("records another");

        let loaded = load(&root, Some(&original.id)).expect("loads");
        assert_eq!(loaded.task_ids, vec!["1700-demo", "1800-other"]);

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn relative_times_read_sensibly() {
        let now = now_ms();
        assert_eq!(ago(now), "just now");
        assert_eq!(ago(now - 5 * 60 * 1000), "5m ago");
        assert_eq!(ago(now - 3 * 3_600_000), "3h ago");
        assert_eq!(ago(now - 5 * 24 * 3_600_000), "5d ago");
        // A timestamp from the future must not underflow.
        assert_eq!(ago(now + 10_000), "just now");
    }
}
