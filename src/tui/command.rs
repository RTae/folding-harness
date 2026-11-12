//! The slash commands the chat line understands.
//!
//! One table drives the parser, the suggestion list and the help text, so a
//! command cannot appear in one and be missing from another.

/// A slash command typed into the message line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Slash {
    Help,
    Provider,
    Spend,
    SetModel(String),
    SetUrl(String),
    Retry,
    Run,
    Save(Option<String>),
    Plan,
    Model,
    Clear,
    Quit,
    Unknown(String),
}

pub fn parse_slash(line: &str) -> Option<Slash> {
    let line = line.trim();
    let rest = line.strip_prefix('/')?;
    let mut parts = rest.splitn(2, char::is_whitespace);
    let name = parts.next().unwrap_or("").to_ascii_lowercase();
    let argument = parts.next().map(|text| text.trim().to_string());
    Some(match name.as_str() {
        "help" | "h" | "?" => Slash::Help,
        "provider" => Slash::Provider,
        "spend" | "usage" | "cost" => Slash::Spend,
        "model" | "m" => match argument {
            Some(value) if !value.is_empty() => Slash::SetModel(value),
            _ => Slash::Model,
        },
        "url" => match argument {
            Some(value) if !value.is_empty() => Slash::SetUrl(value),
            _ => Slash::Provider,
        },
        "retry" => Slash::Retry,
        "run" | "r" => Slash::Run,
        "save" | "s" => Slash::Save(argument.filter(|text| !text.is_empty())),
        "plan" | "p" => Slash::Plan,
        "clear" => Slash::Clear,
        "quit" | "q" | "exit" => Slash::Quit,
        other => Slash::Unknown(other.to_string()),
    })
}

/// The commands, their argument hint, and what they do.
///
/// One table drives the suggestion list and the help text, so the two cannot
/// drift apart.
pub const COMMANDS: &[(&str, &str, &str)] = &[
    ("/help", "", "this list"),
    (
        "/provider",
        "",
        "choose where prompts go, from your saved profiles",
    ),
    ("/spend", "", "tokens this session has used"),
    (
        "/model",
        " <name>",
        "use a different model, and remember it",
    ),
    (
        "/url",
        " <url>",
        "use a different endpoint, and remember it",
    ),
    ("/retry", "", "ask again, for a reply that was cut off"),
    ("/plan", "", "show the current plan summary again"),
    ("/run", "", "run the validated plan (same as Ctrl-R)"),
    (
        "/save",
        " [path]",
        "write the plan to task.yaml, or to [path]",
    ),
    (
        "/clear",
        "",
        "forget the conversation so far and start over",
    ),
    ("/quit", "", "leave without running (same as Esc)"),
];

const KEY_HELP: &str = "Enter sends, Alt-Enter adds a line. Pasting several lines keeps them as one
message. Left/Right and Home/End move within a line, Ctrl-W deletes a word,
Ctrl-U clears the message, Up/Down move between lines or recall what you typed
before. Tab completes a command. PgUp/PgDn scroll the transcript.";

pub fn help_text() -> String {
    let mut out = String::new();
    for (name, argument, what) in COMMANDS {
        out.push_str(&format!("{:<16}{what}\n", format!("{name}{argument}")));
    }
    out.push('\n');
    out.push_str(KEY_HELP);
    out
}

/// Commands matching what has been typed, while the message is still a bare
/// `/word`. An argument having been started means the command is chosen.
pub fn suggestions(input: &str) -> Vec<&'static (&'static str, &'static str, &'static str)> {
    let trimmed = input.trim_start();
    if !trimmed.starts_with('/') || trimmed.contains(char::is_whitespace) {
        return Vec::new();
    }
    let lowered = trimmed.to_ascii_lowercase();
    COMMANDS
        .iter()
        .filter(|(name, _, _)| name.starts_with(&lowered))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slash_commands_parse_with_aliases_and_arguments() {
        assert_eq!(parse_slash("/help"), Some(Slash::Help));
        assert_eq!(parse_slash("  /h  "), Some(Slash::Help));
        assert_eq!(parse_slash("/run"), Some(Slash::Run));
        assert_eq!(parse_slash("/save"), Some(Slash::Save(None)));
        assert_eq!(
            parse_slash("/save out/plan.yaml"),
            Some(Slash::Save(Some("out/plan.yaml".to_string())))
        );
        assert_eq!(parse_slash("/QUIT"), Some(Slash::Quit));
        assert_eq!(
            parse_slash("/nope"),
            Some(Slash::Unknown("nope".to_string()))
        );
        // Ordinary messages are not commands, including one that merely
        // mentions a slash.
        assert_eq!(parse_slash("design the loops"), None);
        assert_eq!(parse_slash("use a 50/50 split"), None);
    }

    #[test]
    fn the_help_text_is_generated_from_the_same_table() {
        let help = help_text();
        for (name, _, what) in COMMANDS {
            assert!(help.contains(name), "{name} missing from the help");
            assert!(help.contains(what), "{what:?} missing from the help");
        }
        assert!(help.contains("Tab completes a command"));
    }
}
