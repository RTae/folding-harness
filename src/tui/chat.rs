//! The chat screen: describe a run in prose, see the plan, confirm it.
//!
//! This is the conversational entry point. You type what you want, the model
//! replies, and the plan is put through the harness's own validation before it
//! is shown — so the pane on the right is not the model's claim about what will
//! happen, it is what the typed config actually resolves to: which residues may
//! change, which may not, where the gates sit. A plan that does not validate
//! shows the error instead, and the model is told what was wrong so it can
//! correct itself.
//!
//! Nothing runs until you press Ctrl-R.
//!
//! Replies stream in as they are generated. The model call happens on the Tokio
//! runtime while this loop keeps drawing, so the interface stays responsive and
//! a slow reply cannot wedge it.

use std::io::{self, IsTerminal};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ratatui::backend::{Backend, CrosstermBackend};
use ratatui::crossterm::event::{
    self, Event as TermEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers,
};
use ratatui::crossterm::event::{DisableBracketedPaste, EnableBracketedPaste};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::layout::{Constraint, Direction, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Wrap};
use ratatui::{Frame, Terminal};

use crate::agent::{ChatMessage, ModelProvider};
use crate::history::{ChatRecord, StoredRole};
use crate::plan::{self, Chain, Plan};
use crate::settings::{self, ProviderConfig, ProviderSettings};
use crate::TaskConfig;

/// What the model-call worker sends back.
enum Incoming {
    /// A piece of the reply, as it is generated.
    Delta(String),
    /// The reply finished; carries the whole text.
    Done(String),
    Failed(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Entry {
    You,
    Model,
    Harness,
}

impl Entry {
    fn label(&self) -> &'static str {
        match self {
            Entry::You => "you",
            Entry::Model => "model",
            Entry::Harness => "harness",
        }
    }

    fn color(&self) -> Color {
        match self {
            Entry::You => Color::Cyan,
            Entry::Model => Color::White,
            Entry::Harness => Color::Yellow,
        }
    }

    fn body_color(&self) -> Color {
        match self {
            Entry::Harness => Color::Yellow,
            _ => Color::Gray,
        }
    }
}

/// Wrap text to a width, preserving explicit newlines. Done by hand so the
/// transcript's line count is known exactly and scrolling is predictable.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(8);
    let mut out = Vec::new();
    for paragraph in text.split('\n') {
        if paragraph.trim().is_empty() {
            out.push(String::new());
            continue;
        }

        // Keep the line's own indentation and repeat it on continuations, so
        // nested JSON stays readable instead of collapsing to a flat block.
        let body = paragraph.trim_start();
        let depth = paragraph.chars().count() - body.chars().count();
        let indent = " ".repeat(depth.min(width.saturating_sub(8)));
        let width = width - indent.chars().count();

        let mut line = String::new();
        let push = |line: String, out: &mut Vec<String>| {
            out.push(format!("{indent}{line}"));
        };

        for word in body.split_whitespace() {
            if line.is_empty() {
                // A single word longer than the pane is hard-split.
                if word.chars().count() > width {
                    let mut rest = word;
                    while rest.chars().count() > width {
                        let head: String = rest.chars().take(width).collect();
                        push(head, &mut out);
                        rest = &rest[rest
                            .char_indices()
                            .nth(width)
                            .map(|(index, _)| index)
                            .unwrap_or(rest.len())..];
                    }
                    line.push_str(rest);
                } else {
                    line.push_str(word);
                }
            } else if line.chars().count() + 1 + word.chars().count() <= width {
                line.push(' ');
                line.push_str(word);
            } else {
                push(std::mem::take(&mut line), &mut out);
                line.push_str(word);
            }
        }
        if !line.is_empty() {
            push(line, &mut out);
        }
    }
    out
}

/// A single-line editor with a cursor and a history ring.
///
/// Hand-written because `ratatui` draws widgets and leaves input to the
/// application. Chars are the unit throughout, so a multi-byte character
/// counts as one position.
#[derive(Debug, Default)]
pub struct TextInput {
    chars: Vec<char>,
    cursor: usize,
    history: Vec<String>,
    /// Where we are when scrolling back; `None` means "editing a fresh line".
    recall: Option<usize>,
    /// The line being edited, stashed while scrolling through history.
    stashed: Option<Vec<char>>,
}

impl TextInput {
    pub fn text(&self) -> String {
        self.chars.iter().collect()
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    fn insert(&mut self, character: char) {
        self.chars
            .insert(self.cursor.min(self.chars.len()), character);
        self.cursor += 1;
    }

    /// Insert pasted text at the cursor, newlines and all.
    ///
    /// A paste arrives as one event when the terminal has bracketed paste, so
    /// its newlines are content rather than a run of Enter presses. One
    /// trailing newline is dropped: copying a line usually takes its line
    /// ending with it, and nobody means to end on a blank line.
    fn insert_str(&mut self, text: &str) {
        let text = text.strip_suffix('\n').unwrap_or(text);
        let text = text.strip_suffix('\r').unwrap_or(text);
        for character in text.chars() {
            // Carriage returns and tabs would corrupt the layout.
            match character {
                '\r' => continue,
                '\t' => {
                    for _ in 0..4 {
                        self.insert(' ');
                    }
                }
                _ => self.insert(character),
            }
        }
    }

    /// The logical lines of the buffer.
    fn lines(&self) -> Vec<String> {
        self.text()
            .split('\n')
            .map(|line| line.to_string())
            .collect()
    }

    fn line_count(&self) -> usize {
        1 + self.chars.iter().filter(|c| **c == '\n').count()
    }

    /// Where the cursor sits, as a line and a column within it.
    fn row_col(&self) -> (usize, usize) {
        let mut row = 0;
        let mut col = 0;
        for character in self.chars.iter().take(self.cursor) {
            if *character == '\n' {
                row += 1;
                col = 0;
            } else {
                col += 1;
            }
        }
        (row, col)
    }

    /// Index of the first character of a given line.
    fn line_start(&self, row: usize) -> usize {
        let mut seen = 0;
        for (index, character) in self.chars.iter().enumerate() {
            if seen == row {
                return index;
            }
            if *character == '\n' {
                seen += 1;
            }
        }
        self.chars.len()
    }

    fn line_len(&self, row: usize) -> usize {
        self.lines()
            .get(row)
            .map(|line| line.chars().count())
            .unwrap_or(0)
    }

    /// Up and Down move between lines when the message has several, and
    /// otherwise walk the history, which is what a shell does.
    fn up(&mut self) {
        let (row, col) = self.row_col();
        if row == 0 {
            self.recall_older();
            return;
        }
        let target = row - 1;
        self.cursor = self.line_start(target) + col.min(self.line_len(target));
    }

    fn down(&mut self) {
        let (row, col) = self.row_col();
        if row + 1 >= self.line_count() {
            self.recall_newer();
            return;
        }
        let target = row + 1;
        self.cursor = self.line_start(target) + col.min(self.line_len(target));
    }

    fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.chars.remove(self.cursor);
        }
    }

    fn delete(&mut self) {
        if self.cursor < self.chars.len() {
            self.chars.remove(self.cursor);
        }
    }

    /// Ctrl-W: delete the word before the cursor, including trailing spaces.
    fn delete_word(&mut self) {
        while self.cursor > 0 && self.chars[self.cursor - 1].is_whitespace() {
            self.cursor -= 1;
            self.chars.remove(self.cursor);
        }
        while self.cursor > 0 && !self.chars[self.cursor - 1].is_whitespace() {
            self.cursor -= 1;
            self.chars.remove(self.cursor);
        }
    }

    fn left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    fn right(&mut self) {
        self.cursor = (self.cursor + 1).min(self.chars.len());
    }

    fn home(&mut self) {
        let (row, _) = self.row_col();
        self.cursor = self.line_start(row);
    }

    fn end(&mut self) {
        let (row, _) = self.row_col();
        self.cursor = self.line_start(row) + self.line_len(row);
    }

    fn clear(&mut self) {
        self.chars.clear();
        self.cursor = 0;
        self.recall = None;
        self.stashed = None;
    }

    /// Take the line, push it onto the history, and reset.
    fn take(&mut self) -> String {
        let text: String = self.chars.iter().collect();
        if !text.trim().is_empty() && self.history.last().map(String::as_str) != Some(text.as_str())
        {
            self.history.push(text.clone());
        }
        self.clear();
        text
    }

    fn set(&mut self, text: &str) {
        self.chars = text.chars().collect();
        self.cursor = self.chars.len();
    }

    /// Up: an older line from the history.
    fn recall_older(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let next = match self.recall {
            None => {
                self.stashed = Some(self.chars.clone());
                self.history.len() - 1
            }
            Some(0) => 0,
            Some(index) => index - 1,
        };
        self.recall = Some(next);
        let text = self.history[next].clone();
        self.set(&text);
    }

    /// Down: a newer line, then back to whatever was being typed.
    fn recall_newer(&mut self) {
        match self.recall {
            None => {}
            Some(index) if index + 1 < self.history.len() => {
                self.recall = Some(index + 1);
                let text = self.history[index + 1].clone();
                self.set(&text);
            }
            Some(_) => {
                self.recall = None;
                let stashed = self.stashed.take().unwrap_or_default();
                self.chars = stashed;
                self.cursor = self.chars.len();
            }
        }
    }

    /// The lines to draw and where the cursor sits among them.
    ///
    /// Only the cursor's own line scrolls horizontally, so the rest stay put
    /// while you move along a long line.
    fn view(&self, width: usize, max_rows: usize) -> (Vec<String>, usize, usize) {
        let width = width.max(4);
        let max_rows = max_rows.max(1);
        let lines = self.lines();
        let (row, col) = self.row_col();

        // Scroll vertically so the cursor's line is on screen.
        let first = if row >= max_rows {
            row + 1 - max_rows
        } else {
            0
        };
        let visible: Vec<String> = lines.iter().skip(first).take(max_rows).cloned().collect();

        let cursor_row = row - first;
        let mut shown = visible;
        let mut cursor_col = col;
        if let Some(line) = shown.get_mut(cursor_row) {
            if line.chars().count() >= width {
                let start = col.saturating_sub(width - 1);
                *line = line.chars().skip(start).take(width).collect();
                cursor_col = col - start;
            }
        }
        (shown, cursor_row, cursor_col)
    }
}

/// A slash command typed into the message line.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Slash {
    Help,
    Provider,
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

fn parse_slash(line: &str) -> Option<Slash> {
    let line = line.trim();
    let rest = line.strip_prefix('/')?;
    let mut parts = rest.splitn(2, char::is_whitespace);
    let name = parts.next().unwrap_or("").to_ascii_lowercase();
    let argument = parts.next().map(|text| text.trim().to_string());
    Some(match name.as_str() {
        "help" | "h" | "?" => Slash::Help,
        "provider" => Slash::Provider,
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

const HELP: &str = "/help           this list
/provider       choose where prompts go, from your saved profiles
/model <name>   use a different model, and remember it
/url <url>      use a different endpoint, and remember it
/retry          ask again, for a reply that was cut off
/plan           show the current plan summary again
/run            run the validated plan (same as Ctrl-R)
/save [path]    write the plan to task.yaml, or to [path]
/model          which model and endpoint this session is using
/clear          forget the conversation so far and start over
/quit           leave without running (same as Esc)

Enter sends, Alt-Enter adds a line. Pasting several lines keeps them as one
message. Left/Right and Home/End move within a line, Ctrl-W deletes a word,
Ctrl-U clears the message, Up/Down move between lines or recall what you typed
before. PgUp/PgDn scroll the transcript.";

pub struct ChatApp {
    provider: Arc<dyn ModelProvider>,
    runtime: tokio::runtime::Handle,
    binder: Vec<Chain>,
    target: Vec<Chain>,
    seed: u64,

    transcript: Vec<(Entry, String)>,
    /// What is sent to the model: the transcript without harness notes.
    history: Vec<ChatMessage>,
    input: TextInput,
    waiting: Option<Receiver<Incoming>>,
    /// Index into `transcript` of the reply currently streaming in.
    streaming_into: Option<usize>,
    started_waiting: Option<Instant>,
    frame: usize,
    scroll: usize,
    follow: bool,

    plan: Option<Plan>,
    plan_error: Option<String>,
    status: String,
    quit: bool,
    /// Set when the user confirms; returned to the caller to run.
    confirmed: Option<TaskConfig>,

    /// Saved after every exchange, so closing the terminal does not lose the
    /// session.
    record: ChatRecord,
    chat_root: std::path::PathBuf,

    /// Settings the screen can change, and the means to apply them.
    settings: ProviderSettings,
    /// Where those settings live. Held rather than read from the environment
    /// on each use, so a test can point it somewhere harmless without
    /// mutating the process environment, which races other tests.
    config_path: std::path::PathBuf,
    rebuild: Option<ProviderFactory>,
    /// Open when choosing a provider profile.
    picker: Option<Picker>,
}

/// The provider chooser: a list of profile names and where the cursor is.
#[derive(Debug, Clone)]
struct Picker {
    config: ProviderConfig,
    names: Vec<String>,
    selected: usize,
}

impl Picker {
    fn open(config: ProviderConfig) -> Option<Picker> {
        let names = config.names();
        if names.is_empty() {
            return None;
        }
        let active = config.active_name();
        let selected = names.iter().position(|name| *name == active).unwrap_or(0);
        Some(Picker {
            config,
            names,
            selected,
        })
    }

    fn move_by(&mut self, delta: isize) {
        if self.names.is_empty() {
            return;
        }
        let count = self.names.len() as isize;
        let next = (self.selected as isize + delta).rem_euclid(count);
        self.selected = next as usize;
    }

    fn chosen(&self) -> Option<&String> {
        self.names.get(self.selected)
    }
}

/// Builds a provider from settings. The screen cannot construct one itself:
/// it holds a trait object and the concrete type lives a layer up, which is
/// also what lets the tests run against a stub.
pub type ProviderFactory =
    Box<dyn Fn(&ProviderSettings) -> Result<Arc<dyn ModelProvider>, String> + Send>;

impl ChatApp {
    pub fn new(
        provider: Arc<dyn ModelProvider>,
        runtime: tokio::runtime::Handle,
        binder: Vec<Chain>,
        target: Vec<Chain>,
        seed: u64,
    ) -> Self {
        let mut transcript = vec![(
            Entry::Harness,
            format!(
                "Connected to {}. Describe the run you want and I will turn it into a plan.\n\n\
Sequences you supplied: {} binder chain(s), {} target chain(s). The model is never asked to \
invent a sequence — it only chooses which spans are mutable, what to pin, and the search \
parameters.\n\nThe plan on the right is what the config actually resolves to, not what the model \
says it does. Nothing runs until you press Ctrl-R. Type /help for commands.",
                provider.model(),
                binder.len(),
                target.len()
            ),
        )];

        if binder.is_empty() || target.is_empty() {
            transcript.push((
                Entry::Harness,
                "No sequences were supplied, so a plan cannot be validated. Restart with \
--binder and --target."
                    .to_string(),
            ));
        }

        let record = ChatRecord::new(
            provider.model(),
            provider.endpoint(),
            seed,
            &binder,
            &target,
        );

        ChatApp {
            record,
            chat_root: crate::history::chat_root(),
            settings: settings::load(),
            config_path: settings::config_path(),
            rebuild: None,
            picker: None,
            provider,
            runtime,
            binder,
            target,
            seed,
            transcript,
            history: Vec::new(),
            input: TextInput::default(),
            waiting: None,
            streaming_into: None,
            started_waiting: None,
            frame: 0,
            scroll: 0,
            follow: true,
            plan: None,
            plan_error: None,
            status: String::new(),
            quit: false,
            confirmed: None,
        }
    }

    /// The confirmed config, with the id of the conversation that produced it.
    pub fn confirmed(self) -> Option<(TaskConfig, String)> {
        let id = self.record.id.clone();
        self.confirmed.map(|config| (config, id))
    }

    pub fn chat_id(&self) -> &str {
        &self.record.id
    }

    /// Let the screen change where prompts go.
    pub fn with_provider_factory(mut self, factory: ProviderFactory) -> Self {
        self.rebuild = Some(factory);
        self
    }

    /// Point the remembered settings at a particular file.
    pub fn with_config_path(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.config_path = path.into();
        self.settings = settings::load_config_from(&self.config_path).active_settings();
        self
    }

    fn open_picker(&mut self) {
        if self.rebuild.is_none() {
            self.status = "this session cannot change provider".to_string();
            return;
        }
        match Picker::open(settings::load_config_from(&self.config_path)) {
            None => self.status = "no provider profiles".to_string(),
            Some(picker) => self.picker = Some(picker),
        }
    }

    /// Switch to the highlighted profile.
    fn choose_profile(&mut self) {
        let Some(picker) = self.picker.take() else {
            return;
        };
        let Some(name) = picker.chosen().cloned() else {
            return;
        };

        let mut config = picker.config;
        config.active = Some(name.clone());
        let settings = config.active_settings();

        let Some(rebuild) = self.rebuild.as_ref() else {
            return;
        };
        match rebuild(&settings) {
            Err(error) => self.note(format!("Could not switch to {name}: {error}")),
            Ok(provider) => {
                let description = format!("{} via {}", provider.model(), provider.endpoint());
                self.provider = provider;
                self.settings = settings;
                match settings::save_config_to(&self.config_path, &config) {
                    Ok(()) => self.note(format!("Now using {name}: {description}.")),
                    Err(error) => self.note(format!(
                        "Now using {name}: {description}, but it could not be remembered: \
{error}"
                    )),
                }
            }
        }
    }

    /// Apply changed settings to this session and remember them.
    ///
    /// The change is saved only once the provider is built, so a typo in a
    /// model name does not become the setting every later command inherits.
    fn apply_settings(&mut self, changed: ProviderSettings, what: &str) {
        let Some(rebuild) = self.rebuild.as_ref() else {
            self.status = "this session cannot change provider".to_string();
            return;
        };
        match rebuild(&changed) {
            Err(error) => {
                self.note(format!("Could not switch {what}: {error}"));
            }
            Ok(provider) => {
                let description = format!("{} via {}", provider.model(), provider.endpoint());
                self.provider = provider;
                self.settings = changed;
                match settings::save_to(&self.config_path, &self.settings)
                    .map(|()| self.config_path.clone())
                {
                    Ok(path) => self.note(format!(
                        "Now using {description}. Remembered in {}.",
                        path.display()
                    )),
                    Err(error) => self.note(format!(
                        "Now using {description}, but it could not be remembered: {error}"
                    )),
                }
            }
        }
    }

    /// Carry on a stored conversation. The transcript is rebuilt so the screen
    /// opens where it left off.
    pub fn resumed(mut self, record: ChatRecord) -> Self {
        self.transcript.push((
            Entry::Harness,
            format!(
                "Resumed chat {} from {}, {} message(s).",
                record.id,
                crate::history::ago(record.updated_at_ms),
                record.messages.len()
            ),
        ));
        for message in &record.messages {
            let entry = match message.role {
                StoredRole::User => Entry::You,
                StoredRole::Assistant => Entry::Model,
            };
            self.transcript.push((entry, message.text.clone()));
        }
        self.history = record.to_messages();

        // A stored plan is not replayed as validated: re-deriving it from the
        // last reply keeps one definition of what a valid plan is.
        if let Some(last) = record
            .messages
            .iter()
            .rev()
            .find(|message| message.role == StoredRole::Assistant)
        {
            let text = last.text.clone();
            self.validate(&text);
        }

        self.record = record;
        self
    }

    /// Write the conversation to disk. A failure is shown but does not
    /// interrupt the session: losing the history is better than losing the
    /// chat.
    fn remember(&mut self) {
        self.record.set_messages(&self.history);
        if let Some(plan) = self.plan.as_ref() {
            self.record.plan_name = Some(plan.config.name.clone());
        }
        if let Err(error) = self.record.save(&self.chat_root) {
            self.status = format!("could not save the chat: {error}");
        }
    }

    fn note(&mut self, text: impl Into<String>) {
        self.transcript.push((Entry::Harness, text.into()));
        self.follow = true;
    }

    fn submit(&mut self) {
        if self.waiting.is_some() {
            self.status = "still waiting for the last reply".to_string();
            return;
        }
        if self.input.is_empty() {
            return;
        }
        let line = self.input.take();

        if let Some(command) = parse_slash(&line) {
            self.run_slash(command);
            return;
        }

        self.transcript.push((Entry::You, line.clone()));
        self.history.push(ChatMessage::user(line));
        self.follow = true;
        self.ask();
    }

    fn run_slash(&mut self, command: Slash) {
        match command {
            Slash::Help => self.note(HELP),
            Slash::Quit => self.quit = true,
            Slash::Model => {
                let text = format!(
                    "model {} — {} binder chain(s), {} target chain(s), seed {}",
                    self.provider.model(),
                    self.binder.len(),
                    self.target.len(),
                    self.seed
                );
                self.note(text);
            }
            Slash::Provider => self.open_picker(),
            Slash::SetModel(model) => {
                let mut changed = self.settings.clone();
                changed.model = Some(model.clone());
                self.apply_settings(changed, &format!("to {model}"));
            }
            Slash::SetUrl(url) => {
                let mut changed = self.settings.clone();
                changed.url = Some(url.clone());
                self.apply_settings(changed, &format!("to {url}"));
            }
            Slash::Clear => {
                self.history.clear();
                self.plan = None;
                self.plan_error = None;
                self.note("Conversation cleared. The next message starts a fresh one.");
                self.remember();
            }
            Slash::Plan => match self.plan.as_ref() {
                None => self.note("No validated plan yet."),
                Some(plan) => {
                    let summary = plan.summary().join("\n");
                    self.note(summary);
                }
            },
            Slash::Retry => self.retry(),
            Slash::Run => self.confirm_plan(),
            Slash::Save(path) => self.save(path.as_deref()),
            Slash::Unknown(name) => {
                let text = format!("No command /{name}. Type /help for the list.");
                self.note(text);
            }
        }
    }

    /// Ask again without adding a message, for a reply that was cut off.
    fn retry(&mut self) {
        if self.waiting.is_some() {
            self.status = "still waiting for the last reply".to_string();
            return;
        }
        if self.history.is_empty() {
            self.status = "nothing to retry yet".to_string();
            return;
        }
        self.note("Asking again.");
        self.ask();
    }

    /// Start a streaming request for the current history.
    fn ask(&mut self) {
        let (tx, rx) = mpsc::channel();
        self.waiting = Some(rx);
        self.started_waiting = Some(Instant::now());

        // The reply accumulates into its own transcript entry.
        self.transcript.push((Entry::Model, String::new()));
        self.streaming_into = Some(self.transcript.len() - 1);

        let provider = Arc::clone(&self.provider);
        let system = plan::system_prompt();
        let mut messages = self.history.clone();
        // The first user turn carries the chain inventory, so the model knows
        // the lengths it must respect.
        if messages.len() == 1 {
            if let Some(first) = messages.first_mut() {
                first.text = plan::user_prompt(&first.text, &self.binder, &self.target);
            }
        }

        let delta_tx = tx.clone();
        self.runtime.spawn(async move {
            let (piece_tx, piece_rx) = mpsc::channel::<String>();
            // Forward each piece on to the interface as it arrives.
            let forwarder = std::thread::spawn(move || {
                while let Ok(piece) = piece_rx.recv() {
                    if delta_tx.send(Incoming::Delta(piece)).is_err() {
                        break;
                    }
                }
            });

            let outcome = provider.converse_stream(&system, &messages, piece_tx).await;
            let _ = forwarder.join();

            let _ = tx.send(match outcome {
                Ok(reply) => Incoming::Done(reply),
                Err(error) => Incoming::Failed(error.to_string()),
            });
        });
    }

    pub fn poll_worker(&mut self) {
        loop {
            let Some(receiver) = self.waiting.as_ref() else {
                return;
            };
            match receiver.try_recv() {
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => {
                    self.finish_streaming();
                    self.status = "the request thread stopped without replying".to_string();
                    return;
                }
                Ok(Incoming::Delta(piece)) => {
                    if let Some(index) = self.streaming_into {
                        if let Some((_, text)) = self.transcript.get_mut(index) {
                            text.push_str(&piece);
                        }
                    }
                    self.follow = true;
                }
                Ok(Incoming::Failed(detail)) => {
                    // Text that already arrived is worth keeping: a stream cut
                    // off at the end may still carry a complete plan, and
                    // throwing it away would waste the whole turn.
                    let partial = self
                        .streaming_into
                        .and_then(|index| self.transcript.get(index))
                        .map(|(_, text)| text.clone())
                        .filter(|text| !text.trim().is_empty());

                    self.drop_empty_stream_entry();
                    self.finish_streaming();

                    match partial {
                        None => self.note(format!("the model call failed: {detail}")),
                        Some(text) => {
                            self.note(format!(
                                "the reply stopped early: {detail}\nWhat arrived is above, and \
is checked below in case it was already complete."
                            ));
                            self.history.push(ChatMessage::assistant(text.clone()));
                            self.validate(&text);
                            self.remember();
                        }
                    }
                    return;
                }
                Ok(Incoming::Done(reply)) => {
                    // Prefer the assembled reply: a dropped delta would
                    // otherwise leave the transcript subtly wrong.
                    if let Some(index) = self.streaming_into {
                        if let Some((_, text)) = self.transcript.get_mut(index) {
                            *text = reply.clone();
                        }
                    }
                    self.finish_streaming();
                    self.history.push(ChatMessage::assistant(reply.clone()));
                    self.follow = true;
                    self.validate(&reply);
                    self.remember();
                    return;
                }
            }
        }
    }

    fn drop_empty_stream_entry(&mut self) {
        if let Some(index) = self.streaming_into {
            if self
                .transcript
                .get(index)
                .map(|(_, text)| text.is_empty())
                .unwrap_or(false)
            {
                self.transcript.remove(index);
            }
        }
    }

    fn finish_streaming(&mut self) {
        self.waiting = None;
        self.streaming_into = None;
        self.started_waiting = None;
    }

    /// Put the reply through the harness's own validation.
    fn validate(&mut self, reply: &str) {
        if self.binder.is_empty() || self.target.is_empty() {
            return;
        }
        match plan::plan_from_reply(reply, &self.binder, &self.target, self.seed) {
            Ok(plan) => {
                self.plan_error = None;
                let summary = plan.scope.summary();
                self.plan = Some(plan);
                self.note(format!(
                    "Plan validated: {summary}. Ctrl-R or /run to start it, /save to write \
task.yaml, or keep talking to change it."
                ));
            }
            Err(plan::PlanError::NoJson { .. }) => {
                // Conversation without a plan in it is normal.
            }
            Err(error) => {
                // The reply is already in the transcript above, so the note and
                // the plan pane show the reason only.
                let brief = error.brief();
                self.plan_error = Some(brief.clone());

                if error.looks_truncated() {
                    self.note(format!(
                        "That plan was cut off before it finished, so it was not accepted:\n\n\
{brief}\n\nType /retry to ask again."
                    ));
                } else {
                    self.note(format!(
                        "That plan did not validate, so it was not accepted:\n\n{brief}"
                    ));
                }

                // The model is told the full error, including its own reply, so
                // it can see exactly where the problem was.
                self.history.push(ChatMessage::user(format!(
                    "The harness rejected that plan: {error}\nReply with corrected plan JSON."
                )));
            }
        }
    }

    fn confirm_plan(&mut self) {
        match self.plan.take() {
            Some(plan) => {
                self.confirmed = Some(plan.config);
                self.quit = true;
            }
            None => self.status = "no validated plan to run yet".to_string(),
        }
    }

    fn save(&mut self, path: Option<&str>) {
        let destination = path.unwrap_or("task.yaml").to_string();
        let Some(plan) = self.plan.as_ref() else {
            self.status = "no validated plan to save yet".to_string();
            return;
        };
        match plan.config.to_yaml() {
            Err(error) => self.status = format!("could not serialise: {error}"),
            Ok(yaml) => match std::fs::write(&destination, yaml) {
                Ok(()) => {
                    self.status = format!("written to {destination}");
                    self.note(format!("Saved the plan to {destination}."));
                }
                Err(error) => self.status = format!("could not write {destination}: {error}"),
            },
        }
    }

    /// Pasted text arrives whole, so its newlines never read as Enter.
    pub fn on_paste(&mut self, text: &str) {
        if self.waiting.is_some() {
            self.status = "still waiting for the last reply".to_string();
            return;
        }
        self.input.insert_str(text);
        let lines = text.lines().count();
        if lines > 1 {
            self.status = format!("pasted {lines} lines; Enter sends, Alt-Enter adds a line");
        }
    }

    pub fn on_key(&mut self, key: KeyEvent) {
        if key.kind != KeyEventKind::Press {
            return;
        }
        self.status.clear();

        // The picker takes the keys it needs, so a stray Enter cannot send a
        // message while a profile is highlighted.
        if self.picker.is_some() {
            match key.code {
                KeyCode::Up | KeyCode::Char('k') => {
                    if let Some(picker) = self.picker.as_mut() {
                        picker.move_by(-1);
                    }
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    if let Some(picker) = self.picker.as_mut() {
                        picker.move_by(1);
                    }
                }
                KeyCode::Enter => self.choose_profile(),
                KeyCode::Esc => self.picker = None,
                _ => {}
            }
            return;
        }

        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('c') => self.quit = true,
                KeyCode::Char('u') => self.input.clear(),
                KeyCode::Char('w') => self.input.delete_word(),
                KeyCode::Char('a') => self.input.home(),
                KeyCode::Char('e') => self.input.end(),
                KeyCode::Char('r') => self.confirm_plan(),
                KeyCode::Char('s') => self.save(None),
                _ => {}
            }
            return;
        }

        // Alt-Enter adds a line instead of sending, for writing a long message
        // by hand. Most terminals cannot distinguish Shift-Enter, so Alt is
        // the one that works widely.
        if key.modifiers.contains(KeyModifiers::ALT) && key.code == KeyCode::Enter {
            self.input.insert('\n');
            return;
        }

        match key.code {
            KeyCode::Esc => self.quit = true,
            // `?` is a character you might want to type, so the shortcut is
            // F1; /help does the same thing.
            KeyCode::F(1) => self.note(HELP),
            KeyCode::Enter => self.submit(),
            KeyCode::Backspace => self.input.backspace(),
            KeyCode::Delete => self.input.delete(),
            KeyCode::Left => self.input.left(),
            KeyCode::Right => self.input.right(),
            KeyCode::Home => self.input.home(),
            KeyCode::End => self.input.end(),
            KeyCode::Up => self.input.up(),
            KeyCode::Down => self.input.down(),
            KeyCode::PageUp => {
                self.follow = false;
                self.scroll = self.scroll.saturating_sub(8);
            }
            KeyCode::PageDown => self.scroll = self.scroll.saturating_add(8),
            KeyCode::Char(character) => self.input.insert(character),
            _ => {}
        }
    }

    pub fn draw(&mut self, frame: &mut Frame) {
        let area = frame.area();
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(40), Constraint::Length(46)])
            .split(area);

        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(6),
                Constraint::Length(self.input_height()),
                Constraint::Length(1),
            ])
            .split(columns[0]);

        self.draw_transcript(frame, rows[0]);
        self.draw_input(frame, rows[1]);
        self.draw_footer(frame, rows[2]);
        self.draw_plan(frame, columns[1]);

        if self.picker.is_some() {
            self.draw_picker(frame, area);
        }
    }

    fn draw_picker(&self, frame: &mut Frame, area: Rect) {
        let Some(picker) = self.picker.as_ref() else {
            return;
        };
        let active = picker.config.active_name();

        let mut lines: Vec<Line> = vec![
            Line::from(Span::styled(
                "Choose where prompts go.",
                Style::default().fg(Color::White),
            )),
            Line::from(""),
        ];
        for (index, name) in picker.names.iter().enumerate() {
            let here = index == picker.selected;
            let mut style = Style::default().fg(Color::White);
            if here {
                // Reversed video rather than a colour: it swaps foreground
                // and background, so the highlight is visible on every theme.
                style = style.add_modifier(Modifier::REVERSED | Modifier::BOLD);
            }
            lines.push(Line::from(vec![
                Span::styled(
                    format!(" {} ", if here { '>' } else { ' ' }),
                    Style::default().fg(Color::Cyan),
                ),
                Span::styled(format!("{name:<14}"), style),
                Span::styled(
                    format!(" {}", picker.config.summary(name)),
                    Style::default().fg(Color::Gray),
                ),
                Span::styled(
                    if *name == active { "  (current)" } else { "" }.to_string(),
                    Style::default().fg(Color::Green),
                ),
            ]));
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Up/Down choose, Enter switch, Esc cancel.",
            Style::default().fg(Color::Gray),
        )));
        lines.push(Line::from(Span::styled(
            "Add one with: fh config add <name> --url ... --model ...",
            Style::default().fg(Color::Gray),
        )));

        let height = lines.len() as u16 + 2;
        let width = 68.min(area.width);
        let popup = Rect {
            x: area.x + (area.width.saturating_sub(width)) / 2,
            y: area.y + (area.height.saturating_sub(height)) / 2,
            width,
            height: height.min(area.height),
        };

        frame.render_widget(Clear, popup);
        frame.render_widget(
            Paragraph::new(lines).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" provider \u{2014} Esc to close "),
            ),
            popup,
        );
    }

    fn draw_transcript(&mut self, frame: &mut Frame, area: Rect) {
        let width = area.width.saturating_sub(4) as usize;
        let height = area.height.saturating_sub(2) as usize;

        let mut lines: Vec<Line> = Vec::new();
        for (entry, text) in &self.transcript {
            lines.push(Line::from(vec![
                Span::styled(
                    "\u{258c} ",
                    Style::default()
                        .fg(entry.color())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    entry.label().to_string(),
                    Style::default()
                        .fg(entry.color())
                        .add_modifier(Modifier::BOLD),
                ),
            ]));

            let mut in_code = false;
            for wrapped in wrap(text, width) {
                if wrapped.trim_start().starts_with("```") {
                    in_code = !in_code;
                    continue;
                }
                lines.push(Line::from(Span::styled(
                    format!("  {wrapped}"),
                    if in_code {
                        Style::default().fg(Color::LightCyan)
                    } else {
                        Style::default().fg(entry.body_color())
                    },
                )));
            }
            lines.push(Line::from(""));
        }

        if self.waiting.is_some() {
            const SPINNER: [&str; 4] = ["\u{2846}", "\u{2807}", "\u{280b}", "\u{2838}"];
            let elapsed = self
                .started_waiting
                .map(|at| at.elapsed().as_secs())
                .unwrap_or(0);
            lines.push(Line::from(Span::styled(
                format!(
                    "  {} {}",
                    SPINNER[(self.frame / 3) % SPINNER.len()],
                    if elapsed > 2 {
                        format!("thinking, {elapsed}s")
                    } else {
                        "thinking".to_string()
                    }
                ),
                Style::default().fg(Color::Gray),
            )));
        }

        let total = lines.len();
        if self.follow {
            self.scroll = total.saturating_sub(height);
        } else {
            self.scroll = self.scroll.min(total.saturating_sub(1));
        }

        let title = if total > height {
            format!(
                " conversation  {}-{} of {} ",
                (self.scroll + 1).min(total),
                (self.scroll + height).min(total),
                total
            )
        } else {
            " conversation ".to_string()
        };

        let visible: Vec<ListItem> = lines
            .into_iter()
            .skip(self.scroll)
            .take(height)
            .map(ListItem::new)
            .collect();

        frame.render_widget(
            List::new(visible).block(Block::default().borders(Borders::ALL).title(title)),
            area,
        );
    }

    /// How tall the message box wants to be, borders included.
    fn input_height(&self) -> u16 {
        const MAX_ROWS: usize = 8;
        2 + self.input.line_count().clamp(1, MAX_ROWS) as u16
    }

    fn draw_input(&self, frame: &mut Frame, area: Rect) {
        let busy = self.waiting.is_some();
        let inner_width = area.width.saturating_sub(4) as usize;
        let rows = area.height.saturating_sub(2) as usize;
        let (lines, cursor_row, cursor_col) = self.input.view(inner_width, rows);

        let title = if self.input.line_count() > 1 {
            format!(" message \u{2014} {} lines ", self.input.line_count())
        } else {
            " message ".to_string()
        };

        let body: Vec<Line> = if busy && self.input.is_empty() {
            vec![Line::from(vec![
                Span::styled("> ", Style::default().fg(Color::Cyan)),
                Span::styled(
                    "waiting for the model\u{2026}",
                    Style::default().fg(Color::Gray),
                ),
            ])]
        } else {
            lines
                .iter()
                .enumerate()
                .map(|(index, line)| {
                    Line::from(vec![
                        Span::styled(
                            if index == 0 { "> " } else { "  " },
                            Style::default().fg(Color::Cyan),
                        ),
                        Span::styled(line.clone(), Style::default().fg(Color::White)),
                    ])
                })
                .collect()
        };

        frame.render_widget(
            Paragraph::new(body).block(Block::default().borders(Borders::ALL).title(title)),
            area,
        );

        if !busy || !self.input.is_empty() {
            // +1 for the border, +2 for the "> " prompt.
            frame.set_cursor_position(Position::new(
                area.x + 3 + cursor_col as u16,
                area.y + 1 + cursor_row as u16,
            ));
        }
    }

    fn draw_footer(&self, frame: &mut Frame, area: Rect) {
        let mut spans = vec![Span::styled(
            "Enter send   Ctrl-R run   F1 or /help for keys   PgUp/PgDn scroll   Esc quit",
            Style::default().fg(Color::Gray),
        )];
        if !self.status.is_empty() {
            spans.push(Span::raw("  |  "));
            spans.push(Span::styled(
                self.status.clone(),
                Style::default().fg(Color::Yellow),
            ));
        }
        frame.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    fn draw_plan(&self, frame: &mut Frame, area: Rect) {
        let mut lines: Vec<Line> = Vec::new();

        match (&self.plan, &self.plan_error) {
            (Some(plan), _) => {
                for line in plan.summary() {
                    let style = if line.contains("cannot change") || line.contains("never changes")
                    {
                        Style::default().fg(Color::Green)
                    } else if line.starts_with("  ") {
                        Style::default().fg(Color::Gray)
                    } else {
                        Style::default().fg(Color::White)
                    };
                    lines.push(Line::from(Span::styled(line, style)));
                }
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "Ctrl-R to run",
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                )));
            }
            (None, Some(error)) => {
                lines.push(Line::from(Span::styled(
                    "last plan was rejected",
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                )));
                lines.push(Line::from(""));
                for wrapped in wrap(error, area.width.saturating_sub(4) as usize) {
                    lines.push(Line::from(Span::styled(
                        wrapped,
                        Style::default().fg(Color::Red),
                    )));
                }
            }
            (None, None) => {
                lines.push(Line::from(Span::styled(
                    "no plan yet",
                    Style::default()
                        .fg(Color::Gray)
                        .add_modifier(Modifier::BOLD),
                )));
                lines.push(Line::from(""));
                for wrapped in wrap(
                    "Describe the binder, which spans should be designed, anything to keep fixed, \
and how long to search. Example: \"design the three CDR loops of this VHH, keep H:52 fixed, \
20 cycles of 6 candidates\".",
                    area.width.saturating_sub(4) as usize,
                ) {
                    lines.push(Line::from(Span::styled(
                        wrapped,
                        Style::default().fg(Color::Gray),
                    )));
                }
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "chains supplied",
                    Style::default().fg(Color::Cyan),
                )));
                for chain in &self.binder {
                    lines.push(Line::from(format!(
                        "  binder {} — {} residues",
                        chain.label,
                        chain.len()
                    )));
                }
                for chain in &self.target {
                    lines.push(Line::from(format!(
                        "  target {} — {} residues",
                        chain.label,
                        chain.len()
                    )));
                }
            }
        }

        frame.render_widget(
            Paragraph::new(lines)
                .block(Block::default().borders(Borders::ALL).title(" plan "))
                .wrap(Wrap { trim: false }),
            area,
        );
    }
}

/// Run the chat screen. Returns the config the user confirmed, if any.
pub fn run(
    provider: Arc<dyn ModelProvider>,
    runtime: tokio::runtime::Handle,
    binder: Vec<Chain>,
    target: Vec<Chain>,
    seed: u64,
    resume: Option<ChatRecord>,
    factory: Option<ProviderFactory>,
) -> io::Result<Option<(TaskConfig, String)>> {
    let mut app = ChatApp::new(provider, runtime, binder, target, seed);
    if let Some(factory) = factory {
        app = app.with_provider_factory(factory);
    }
    if let Some(record) = resume {
        app = app.resumed(record);
    }

    // Entering raw mode on a non-terminal fails with "Device not configured",
    // which says nothing useful. Check first.
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        return Err(io::Error::other(
            "this needs an interactive terminal: run it directly, not through a pipe \
             and not with stdin or stdout redirected",
        ));
    }

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    // Bracketed paste makes the terminal wrap a paste in markers, so it
    // arrives as one event rather than as a burst of keystrokes with every
    // newline read as Enter.
    execute!(stdout, EnterAlternateScreen, EnableBracketedPaste)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout))?;

    let outcome = event_loop(&mut terminal, &mut app);

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        DisableBracketedPaste,
        LeaveAlternateScreen
    )?;
    terminal.show_cursor()?;

    outcome?;
    Ok(app.confirmed())
}

fn event_loop<B: Backend>(terminal: &mut Terminal<B>, app: &mut ChatApp) -> io::Result<()> {
    let mut last_tick = Instant::now();
    loop {
        terminal.draw(|frame| app.draw(frame))?;

        // Short poll so streamed text appears promptly.
        if event::poll(Duration::from_millis(40))? {
            match event::read()? {
                TermEvent::Key(key) => app.on_key(key),
                TermEvent::Paste(text) => app.on_paste(&text),
                _ => {}
            }
        }

        app.poll_worker();

        if last_tick.elapsed() >= Duration::from_millis(120) {
            app.frame = app.frame.wrapping_add(1);
            last_tick = Instant::now();
        }

        if app.quit {
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AgentError, ChatRole};

    struct StubProvider {
        reply: String,
    }

    #[async_trait::async_trait]
    impl ModelProvider for StubProvider {
        fn model(&self) -> &str {
            "stub"
        }
        async fn complete(&self, _system: &str, _user: &str) -> Result<String, AgentError> {
            Ok(self.reply.clone())
        }
    }

    fn chains() -> (Vec<Chain>, Vec<Chain>) {
        (
            vec![Chain::new(
                "H",
                "QVQLVESGGGLVQAGGSLRLSCAASGYTFTSYAMGWFRQAPGKEREFVAAISWSGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAGSYYDYWGQGTQVTVSS",
            )],
            vec![Chain::new("A", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQ")],
        )
    }

    fn app_with(reply: &str) -> ChatApp {
        let (binder, target) = chains();
        ChatApp::new(
            Arc::new(StubProvider {
                reply: reply.to_string(),
            }),
            tokio::runtime::Handle::current(),
            binder,
            target,
            7,
        )
    }

    const GOOD_PLAN: &str = r#"{"name":"demo","topology":"vhh","regions":[{"name":"loop1","chain":"H","start":26,"end":33}],"cycles":4}"#;

    #[test]
    fn wrapping_respects_width_and_newlines() {
        let lines = wrap("one two three four five", 9);
        assert!(
            lines.iter().all(|line| line.chars().count() <= 9),
            "{lines:?}"
        );
        assert_eq!(wrap("a\n\nb", 20), vec!["a", "", "b"]);

        let long = wrap("QVQLVESGGGLVQAGGSLRLSCAAS", 10);
        assert!(
            long.iter().all(|line| line.chars().count() <= 10),
            "{long:?}"
        );
        assert_eq!(long.concat(), "QVQLVESGGGLVQAGGSLRLSCAAS");
    }

    #[test]
    fn wrapping_keeps_each_line_s_own_indentation() {
        // JSON in a reply is only readable if nesting survives the wrap.
        let json = "{\n  \"regions\": [\n    {\"name\": \"cdr1\"}\n  ]\n}";
        let lines = wrap(json, 40);
        assert_eq!(lines[0], "{");
        assert_eq!(lines[1], "  \"regions\": [");
        assert_eq!(lines[2], "    {\"name\": \"cdr1\"}");
        assert_eq!(lines[3], "  ]");
        assert_eq!(lines[4], "}");

        // A continuation of an indented line keeps the same indentation.
        let long = "    aaa bbb ccc ddd eee fff ggg hhh";
        let wrapped = wrap(long, 20);
        assert!(wrapped.len() > 1, "{wrapped:?}");
        assert!(
            wrapped.iter().all(|line| line.starts_with("    ")),
            "{wrapped:?}"
        );
        assert!(
            wrapped.iter().all(|line| line.chars().count() <= 20),
            "{wrapped:?}"
        );
    }

    #[tokio::test]
    async fn a_reply_cut_off_at_the_end_is_kept_and_still_checked() {
        let mut app = app_with("");
        let (tx, rx) = mpsc::channel();
        app.waiting = Some(rx);
        app.transcript.push((Entry::Model, String::new()));
        app.streaming_into = Some(app.transcript.len() - 1);

        // A complete plan arrives, then the stream reports a refusal.
        tx.send(Incoming::Delta(GOOD_PLAN.to_string())).unwrap();
        tx.send(Incoming::Failed(
            "the model declined this request (no category given)".to_string(),
        ))
        .unwrap();
        app.poll_worker();

        // The text is not thrown away, and it was validated anyway.
        assert!(app
            .transcript
            .iter()
            .any(|(entry, text)| *entry == Entry::Model && text.contains("vhh")));
        assert!(
            app.plan.is_some(),
            "a complete plan must survive a late refusal"
        );
        assert_eq!(app.history.len(), 1);
    }

    #[tokio::test]
    async fn a_stream_that_fails_with_nothing_said_is_just_an_error() {
        let mut app = app_with("");
        let (tx, rx) = mpsc::channel();
        app.waiting = Some(rx);
        app.transcript.push((Entry::Model, String::new()));
        app.streaming_into = Some(app.transcript.len() - 1);

        tx.send(Incoming::Failed("connection reset".to_string()))
            .unwrap();
        app.poll_worker();

        assert!(app.plan.is_none());
        assert!(app.history.is_empty(), "nothing was said, nothing to keep");
        let (entry, text) = app.transcript.last().expect("a note");
        assert_eq!(*entry, Entry::Harness);
        assert!(text.contains("connection reset"), "{text}");
    }

    #[test]
    fn the_input_line_edits_at_the_cursor() {
        let mut input = TextInput::default();
        for character in "hello".chars() {
            input.insert(character);
        }
        assert_eq!(input.text(), "hello");
        assert_eq!(input.cursor(), 5);

        input.left();
        input.left();
        input.insert('X');
        assert_eq!(input.text(), "helXlo");
        assert_eq!(input.cursor(), 4);

        input.home();
        assert_eq!(input.cursor(), 0);
        input.delete();
        assert_eq!(input.text(), "elXlo");
        input.end();
        input.backspace();
        assert_eq!(input.text(), "elXl");
    }

    #[test]
    fn ctrl_w_deletes_one_word_including_its_spaces() {
        let mut input = TextInput::default();
        for character in "design the loops   ".chars() {
            input.insert(character);
        }
        input.delete_word();
        assert_eq!(input.text(), "design the ");
        input.delete_word();
        assert_eq!(input.text(), "design ");
    }

    #[test]
    fn the_cursor_cannot_leave_the_line() {
        let mut input = TextInput::default();
        input.left();
        input.backspace();
        assert_eq!(input.cursor(), 0);
        assert_eq!(input.text(), "");
        input.insert('a');
        input.right();
        input.right();
        assert_eq!(input.cursor(), 1);
    }

    #[test]
    fn history_recalls_what_was_typed_and_comes_back() {
        let mut input = TextInput::default();
        input.set("first");
        assert_eq!(input.take(), "first");
        input.set("second");
        assert_eq!(input.take(), "second");

        input.set("draft");
        input.recall_older();
        assert_eq!(input.text(), "second");
        input.recall_older();
        assert_eq!(input.text(), "first");
        input.recall_older();
        assert_eq!(input.text(), "first", "stops at the oldest");
        input.recall_newer();
        assert_eq!(input.text(), "second");
        input.recall_newer();
        assert_eq!(input.text(), "draft", "returns to the unsent line");
    }

    #[test]
    fn a_blank_line_is_not_remembered_and_duplicates_collapse() {
        let mut input = TextInput::default();
        input.set("   ");
        input.take();
        input.set("same");
        input.take();
        input.set("same");
        input.take();
        assert_eq!(input.history, vec!["same"]);
    }

    #[test]
    fn a_pasted_block_is_one_message_not_many() {
        // The bug this fixes: a multi-line paste used to read as a burst of
        // Enter presses, sending the first line and stranding the rest.
        let mut app = tokio::runtime::Runtime::new()
            .expect("runtime")
            .block_on(async { app_with("") });
        app.on_paste("first line\nsecond line\nthird line\n");

        assert_eq!(app.input.line_count(), 3, "{:?}", app.input.text());
        assert_eq!(app.input.text(), "first line\nsecond line\nthird line");
        assert!(app.history.is_empty(), "nothing should have been sent");
        assert!(app.waiting.is_none(), "no request should have started");
        assert!(app.status.contains("3 lines"), "{}", app.status);
    }

    #[test]
    fn a_paste_keeps_its_shape_and_drops_only_a_trailing_newline() {
        let mut input = TextInput::default();
        input.insert_str("a\nb\n");
        assert_eq!(input.text(), "a\nb");

        // Carriage returns are dropped and tabs become spaces, so neither
        // corrupts the layout.
        let mut windows = TextInput::default();
        windows.insert_str("one\r\ntwo\r\n");
        assert_eq!(windows.text(), "one\ntwo");

        let mut tabbed = TextInput::default();
        tabbed.insert_str("a\tb");
        assert_eq!(tabbed.text(), "a    b");
    }

    #[test]
    fn the_cursor_walks_a_multi_line_message() {
        let mut input = TextInput::default();
        input.insert_str("alpha\nbe\ngamma");
        assert_eq!(input.line_count(), 3);

        input.home();
        // Cursor is on the last line, at its start.
        assert_eq!(input.row_col(), (2, 0));

        input.up();
        assert_eq!(input.row_col(), (1, 0));
        input.end();
        assert_eq!(input.row_col(), (1, 2), "end of the short middle line");

        // Moving up from a short line clamps to the longer line's end.
        input.up();
        let (row, col) = input.row_col();
        assert_eq!(row, 0);
        assert!(col <= 5);

        input.down();
        assert_eq!(input.row_col().0, 1);
    }

    #[test]
    fn up_recalls_history_only_on_the_first_line() {
        let mut input = TextInput::default();
        input.set("remembered");
        input.take();

        // Single line: Up walks the history.
        input.up();
        assert_eq!(input.text(), "remembered");

        // Several lines: Up moves within the message instead.
        input.clear();
        input.insert_str("one\ntwo");
        input.up();
        assert_eq!(
            input.text(),
            "one\ntwo",
            "history must not overwrite a draft"
        );
        assert_eq!(input.row_col().0, 0);
    }

    #[test]
    fn the_box_grows_with_the_message_and_stops_growing() {
        let mut input = TextInput::default();
        assert_eq!(input.line_count(), 1);
        input.insert_str(&"x\n".repeat(20));
        assert_eq!(input.line_count(), 20);

        // The view shows a window of the lines, not all twenty.
        let (lines, row, _) = input.view(40, 8);
        assert_eq!(lines.len(), 8);
        assert!(row < 8, "the cursor must stay on screen");
    }

    #[test]
    fn the_visible_window_follows_the_cursor() {
        let mut input = TextInput::default();
        for character in "abcdefghijklmnop".chars() {
            input.insert(character);
        }
        // At the end of a long line the tail is visible and the cursor sits
        // just past it — the cursor takes a column of its own, so the text is
        // at most width-1 characters.
        let (lines, _, column) = input.view(8, 1);
        let text = &lines[0];
        assert!(text.chars().count() <= 8, "{text:?}");
        assert!(text.ends_with('p'), "{text}");
        assert!(column < 8, "cursor column {column} is off-pane");
        assert_eq!(column, text.chars().count());

        input.home();
        let (lines, _, column) = input.view(8, 1);
        assert!(lines[0].starts_with('a'), "{:?}", lines[0]);
        assert_eq!(column, 0);

        // Somewhere in the middle, the cursor is still on-pane.
        for _ in 0..6 {
            input.right();
        }
        let (lines, _, column) = input.view(8, 1);
        assert!(column < 8, "cursor column {column} is off-pane");
        assert!(!lines[0].is_empty());
    }

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

    #[tokio::test]
    async fn a_slash_command_does_not_reach_the_model() {
        let mut app = app_with("");
        app.input.set("/help");
        app.submit();
        assert!(app.waiting.is_none(), "no request should have started");
        assert!(app.history.is_empty(), "nothing was sent");
        assert_eq!(
            app.transcript.last().map(|(entry, _)| *entry),
            Some(Entry::Harness)
        );

        app.input.set("/quit");
        app.submit();
        assert!(app.quit);
    }

    #[tokio::test]
    async fn switching_model_rebuilds_the_provider_and_is_remembered() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct Named(String);
        #[async_trait::async_trait]
        impl ModelProvider for Named {
            fn model(&self) -> &str {
                &self.0
            }
            fn endpoint(&self) -> &str {
                "example.test"
            }
            async fn complete(&self, _: &str, _: &str) -> Result<String, AgentError> {
                Ok(String::new())
            }
        }

        // A file of its own, so this test does not race any other through the
        // process environment.
        let path = std::env::temp_dir().join(format!("fh-chatcfg-{}.json", crate::trace::now_ms()));

        static BUILDS: AtomicUsize = AtomicUsize::new(0);
        let mut app = app_with("")
            .with_config_path(&path)
            .with_provider_factory(Box::new(|settings| {
                BUILDS.fetch_add(1, Ordering::SeqCst);
                Ok(Arc::new(Named(
                    settings.model.clone().unwrap_or_else(|| "none".to_string()),
                )) as Arc<dyn ModelProvider>)
            }));

        // Point the saved settings somewhere harmless for the duration.
        let path = std::env::temp_dir().join(format!("fh-chatcfg-{}.json", crate::trace::now_ms()));
        std::env::set_var("FH_CONFIG", &path);

        app.input.set("/model claude-opus-4-7");
        app.submit();

        assert_eq!(BUILDS.load(Ordering::SeqCst), 1, "the provider was rebuilt");
        assert_eq!(app.provider.model(), "claude-opus-4-7");
        assert_eq!(app.settings.model.as_deref(), Some("claude-opus-4-7"));
        assert_eq!(
            crate::settings::load_from(&path).model.as_deref(),
            Some("claude-opus-4-7"),
            "the choice should outlive the session"
        );

        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn a_provider_that_will_not_build_leaves_the_session_alone() {
        let before = {
            let app = app_with("");
            app.provider.model().to_string()
        };
        let mut app =
            app_with("").with_provider_factory(Box::new(|_| Err("no such model".to_string())));

        app.input.set("/model nonsense");
        app.submit();

        assert_eq!(
            app.provider.model(),
            before,
            "a failed switch must not change the session"
        );
        assert!(app.settings.model.is_none(), "and must not be remembered");
        let note = &app.transcript.last().expect("a note").1;
        assert!(note.contains("no such model"), "{note}");
    }

    #[tokio::test]
    async fn the_picker_lists_profiles_and_switches_on_enter() {
        use crate::settings::{ProviderConfig, ProviderSettings};

        let path = std::env::temp_dir().join(format!("fh-picker-{}.json", crate::trace::now_ms()));
        let mut config = ProviderConfig::default();
        config.profiles.insert(
            "gateway".to_string(),
            ProviderSettings {
                url: Some("https://gw.test".to_string()),
                model: Some("opus-4-7".to_string()),
                ..Default::default()
            },
        );
        crate::settings::save_config_to(&path, &config).expect("saves");

        struct Named(String);
        #[async_trait::async_trait]
        impl ModelProvider for Named {
            fn model(&self) -> &str {
                &self.0
            }
            fn endpoint(&self) -> &str {
                "gw.test"
            }
            async fn complete(&self, _: &str, _: &str) -> Result<String, AgentError> {
                Ok(String::new())
            }
        }

        let mut app = app_with("")
            .with_config_path(&path)
            .with_provider_factory(Box::new(|settings| {
                Ok(Arc::new(Named(
                    settings.model.clone().unwrap_or_else(|| "default".into()),
                )) as Arc<dyn ModelProvider>)
            }));

        app.input.set("/provider");
        app.submit();
        let picker = app.picker.as_ref().expect("the picker opened");
        assert_eq!(picker.names, vec!["anthropic", "gateway"]);

        // Enter while it is open must choose, not send a message.
        app.on_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        let before = app.history.len();
        app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

        assert!(app.picker.is_none(), "the picker closed");
        assert_eq!(app.history.len(), before, "Enter must not have sent");
        // The chosen profile is live in this session and remembered for later.
        assert_eq!(app.provider.model(), "opus-4-7");
        assert_eq!(app.settings.model.as_deref(), Some("opus-4-7"));
        assert_eq!(
            crate::settings::load_config_from(&path).active_name(),
            "gateway"
        );

        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn escape_closes_the_picker_without_switching() {
        let mut app = app_with("")
            .with_provider_factory(Box::new(|_| Err("should not be built".to_string())));
        app.input.set("/provider");
        app.submit();
        assert!(app.picker.is_some());

        app.on_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(app.picker.is_none());
        assert!(!app.quit, "Esc in the picker must not quit the chat");
    }

    #[tokio::test]
    async fn provider_commands_parse() {
        assert_eq!(
            parse_slash("/model claude-opus-4-7"),
            Some(Slash::SetModel("claude-opus-4-7".to_string()))
        );
        assert_eq!(
            parse_slash("/url https://gateway.example.com"),
            Some(Slash::SetUrl("https://gateway.example.com".to_string()))
        );
        // Bare forms ask rather than set.
        assert_eq!(parse_slash("/model"), Some(Slash::Model));
        assert_eq!(parse_slash("/url"), Some(Slash::Provider));
        assert_eq!(parse_slash("/provider"), Some(Slash::Provider));
    }

    #[tokio::test]
    async fn clear_forgets_the_conversation_and_the_plan() {
        let mut app = app_with("");
        app.validate(GOOD_PLAN);
        assert!(app.plan.is_some());
        app.input.set("/clear");
        app.submit();
        assert!(app.plan.is_none());
        assert!(app.history.is_empty());
    }

    #[tokio::test]
    async fn streamed_pieces_accumulate_then_are_replaced_by_the_whole_reply() {
        let mut app = app_with("");
        let (tx, rx) = mpsc::channel();
        app.waiting = Some(rx);
        app.transcript.push((Entry::Model, String::new()));
        app.streaming_into = Some(app.transcript.len() - 1);

        tx.send(Incoming::Delta("Hel".to_string())).unwrap();
        tx.send(Incoming::Delta("lo".to_string())).unwrap();
        app.poll_worker();
        assert_eq!(
            app.transcript.last().map(|(_, text)| text.as_str()),
            Some("Hello")
        );
        assert!(app.waiting.is_some(), "still streaming");

        tx.send(Incoming::Done("Hello there".to_string())).unwrap();
        app.poll_worker();
        assert_eq!(
            app.transcript
                .iter()
                .find(|(entry, _)| *entry == Entry::Model)
                .map(|(_, text)| text.as_str()),
            Some("Hello there")
        );
        assert!(app.waiting.is_none());
        assert_eq!(app.history.len(), 1);
        assert_eq!(app.history[0].role, ChatRole::Assistant);
    }

    #[tokio::test]
    async fn a_failure_removes_the_empty_reply_and_explains() {
        let mut app = app_with("");
        let (tx, rx) = mpsc::channel();
        app.waiting = Some(rx);
        app.transcript.push((Entry::Model, String::new()));
        app.streaming_into = Some(app.transcript.len() - 1);
        let before = app.transcript.len();

        tx.send(Incoming::Failed("connection reset".to_string()))
            .unwrap();
        app.poll_worker();

        // The blank model entry is gone, replaced by a harness note.
        assert_eq!(app.transcript.len(), before);
        let (entry, text) = app.transcript.last().expect("a note");
        assert_eq!(*entry, Entry::Harness);
        assert!(text.contains("connection reset"), "{text}");
        assert!(app.waiting.is_none());
    }

    #[tokio::test]
    async fn a_valid_reply_becomes_a_runnable_plan() {
        let mut app = app_with("");
        app.validate(GOOD_PLAN);
        assert!(app.plan.is_some());
        assert!(app.plan_error.is_none());

        app.on_key(KeyEvent::new(KeyCode::Char('r'), KeyModifiers::CONTROL));
        assert!(app.quit);
        let (config, chat_id) = app.confirmed().expect("a config was confirmed");
        assert_eq!(config.name, "demo");
        assert_eq!(config.design.cycles, 4);
        assert!(!chat_id.is_empty(), "the run must know its conversation");
    }

    #[tokio::test]
    async fn an_invalid_plan_is_rejected_and_fed_back_to_the_model() {
        let mut app = app_with("");
        app.validate(
            r#"{"topology":"vhh","regions":[{"name":"x","chain":"H","start":900,"end":950}]}"#,
        );
        assert!(app.plan.is_none());
        assert!(app.plan_error.is_some());
        assert!(app
            .history
            .last()
            .map(|message| message.text.contains("rejected"))
            .unwrap_or(false));

        app.on_key(KeyEvent::new(KeyCode::Char('r'), KeyModifiers::CONTROL));
        assert!(!app.quit);
        assert!(app.confirmed().is_none());
    }

    #[tokio::test]
    async fn a_rejected_plan_does_not_repeat_the_reply_on_screen() {
        let mut app = app_with("");
        // A long reply that fails the schema. The transcript already holds it,
        // so neither the note nor the plan pane should echo it again.
        let reply = format!(
            "{{\"topology\":\"vhh\",\"regions\":[],\"notes\":\"{}\"}}",
            "x".repeat(600)
        );
        app.validate(&reply);

        let shown = app.plan_error.clone().expect("an error is shown");
        assert!(
            !shown.contains("xxxxx"),
            "the plan pane repeated the reply: {shown}"
        );
        assert!(shown.len() < 200, "the error should be one line: {shown}");

        let (_, note) = app.transcript.last().expect("a harness note");
        assert!(!note.contains("xxxxx"), "the note repeated the reply");

        // The model still gets the full error, reply included, so it can fix it.
        let told = &app.history.last().expect("told the model").text;
        assert!(told.contains("rejected"), "{told}");
    }

    #[tokio::test]
    async fn a_cut_off_plan_says_so_and_offers_retry() {
        let mut app = app_with("");
        // Valid as far as it goes, then the stream stopped.
        // Valid as far as it goes: the inner object closes, the outer never
        // does, which is exactly how a cut-off stream looks.
        app.validate("{\"topology\": \"vhh\", \"gates\": {\"min_iptm\": 0.25}");

        let note = &app.transcript.last().expect("a note").1;
        assert!(note.contains("cut off"), "{note}");
        assert!(note.contains("/retry"), "{note}");

        // And a reply that simply stops mid-object, with nothing closed at
        // all, is also a truncation rather than "no JSON here".
        let mut stopped = app_with("");
        stopped.validate("{\"name\": \"exploratory-run\", \"topology\": \"vhh\", \"regions\": [");
        let note = &stopped.transcript.last().expect("a note").1;
        assert!(note.contains("cut off"), "{note}");
    }

    #[tokio::test]
    async fn retry_asks_again_without_adding_a_message() {
        let mut app = app_with("");
        app.history.push(ChatMessage::user("design the loops"));
        let before = app.history.len();

        app.input.set("/retry");
        app.submit();

        assert_eq!(app.history.len(), before, "retry adds no new message");
        assert!(app.waiting.is_some(), "it should have asked again");
    }

    #[tokio::test]
    async fn retry_with_nothing_to_retry_says_so() {
        let mut app = app_with("");
        app.input.set("/retry");
        app.submit();
        assert!(app.waiting.is_none());
        assert!(app.status.contains("nothing to retry"), "{}", app.status);
    }

    #[tokio::test]
    async fn plain_conversation_without_json_is_not_an_error() {
        let mut app = app_with("");
        app.validate("What is the target, and which loops should I design?");
        assert!(app.plan.is_none());
        assert!(app.plan_error.is_none());
    }

    #[tokio::test]
    async fn typing_and_sending_moves_the_line_into_the_transcript() {
        let mut app = app_with("hello");
        for character in "design the loops".chars() {
            app.on_key(KeyEvent::new(KeyCode::Char(character), KeyModifiers::NONE));
        }
        assert_eq!(app.input.text(), "design the loops");
        app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(app.input.is_empty());
        assert!(app.waiting.is_some());
        assert_eq!(app.history.len(), 1);
        assert_eq!(app.history[0].role, ChatRole::User);

        // Enter while a request is in flight is refused, with a reason.
        app.on_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE));
        app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(app.history.len(), 1);
        assert!(app.status.contains("waiting"), "{}", app.status);
    }

    #[tokio::test]
    async fn a_resumed_chat_reopens_where_it_left_off() {
        use crate::history::{ChatRecord, StoredMessage, StoredRole};

        let (binder, target) = chains();
        let mut record = ChatRecord::new("stub", "localhost", 7, &binder, &target);
        record.messages = vec![
            StoredMessage {
                role: StoredRole::User,
                text: "design the loops".to_string(),
            },
            StoredMessage {
                role: StoredRole::Assistant,
                text: GOOD_PLAN.to_string(),
            },
        ];
        let id = record.id.clone();

        let app = app_with("").resumed(record);

        // The conversation the model will see is the stored one.
        assert_eq!(app.history.len(), 2);
        assert_eq!(app.history[0].text, "design the loops");
        assert_eq!(app.history[1].role, ChatRole::Assistant);

        // The screen shows it too, under both speakers.
        assert!(app
            .transcript
            .iter()
            .any(|(entry, text)| *entry == Entry::You && text == "design the loops"));
        assert!(app
            .transcript
            .iter()
            .any(|(entry, _)| *entry == Entry::Model));

        // The plan is re-derived from the last reply rather than trusted from
        // the file, so one definition of validity applies.
        assert!(app.plan.is_some(), "the stored plan should re-validate");
        assert_eq!(app.chat_id(), id, "it carries on the same record");
    }

    #[tokio::test]
    async fn the_screen_renders_at_several_sizes() {
        use ratatui::backend::TestBackend;

        let mut app = app_with("");
        app.validate(GOOD_PLAN);

        for (width, height) in [(140, 44), (100, 30), (60, 16)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("terminal");
            terminal
                .draw(|frame| app.draw(frame))
                .unwrap_or_else(|error| panic!("failed at {width}x{height}: {error}"));
        }

        // A rejected plan renders a different pane.
        let mut rejected = app_with("");
        rejected.validate(
            r#"{"topology":"vhh","regions":[{"name":"x","chain":"H","start":900,"end":950}]}"#,
        );
        let mut terminal = Terminal::new(TestBackend::new(100, 30)).expect("terminal");
        terminal
            .draw(|frame| rejected.draw(frame))
            .expect("renders the rejection");

        // And so does a long transcript mid-stream, with the scroll indicator.
        let mut busy = app_with("");
        let (_tx, rx) = mpsc::channel();
        busy.waiting = Some(rx);
        busy.started_waiting = Some(Instant::now());
        for index in 0..40 {
            busy.transcript
                .push((Entry::Model, format!("reply number {index} ")));
        }
        let mut terminal = Terminal::new(TestBackend::new(100, 24)).expect("terminal");
        terminal
            .draw(|frame| busy.draw(frame))
            .expect("renders while streaming");
    }
}
