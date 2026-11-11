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
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::layout::{Constraint, Direction, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};
use ratatui::{Frame, Terminal};

use crate::agent::{ChatMessage, ModelProvider};
use crate::plan::{self, Chain, Plan};
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
        let mut line = String::new();
        for word in paragraph.split_whitespace() {
            if line.is_empty() {
                // A single word longer than the pane is hard-split.
                if word.chars().count() > width {
                    let mut rest = word;
                    while rest.chars().count() > width {
                        let head: String = rest.chars().take(width).collect();
                        out.push(head);
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
                out.push(std::mem::take(&mut line));
                line.push_str(word);
            }
        }
        if !line.is_empty() {
            out.push(line);
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
        self.cursor = 0;
    }

    fn end(&mut self) {
        self.cursor = self.chars.len();
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

    /// The visible slice and the cursor's column within it, for a given width.
    fn view(&self, width: usize) -> (String, usize) {
        let width = width.max(4);
        if self.chars.len() < width {
            return (self.text(), self.cursor);
        }
        // Keep the cursor in view, scrolling horizontally.
        let start = self.cursor.saturating_sub(width - 1);
        let text: String = self.chars[start..].iter().take(width).collect();
        (text, self.cursor - start)
    }
}

/// A slash command typed into the message line.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Slash {
    Help,
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
        "run" | "r" => Slash::Run,
        "save" | "s" => Slash::Save(argument.filter(|text| !text.is_empty())),
        "plan" | "p" => Slash::Plan,
        "model" | "m" => Slash::Model,
        "clear" => Slash::Clear,
        "quit" | "q" | "exit" => Slash::Quit,
        other => Slash::Unknown(other.to_string()),
    })
}

const HELP: &str = "/help           this list
/plan           show the current plan summary again
/run            run the validated plan (same as Ctrl-R)
/save [path]    write the plan to task.yaml, or to [path]
/model          which model and endpoint this session is using
/clear          forget the conversation so far and start over
/quit           leave without running (same as Esc)

Enter sends. Left/Right and Home/End move the cursor, Ctrl-W deletes a word,
Ctrl-U clears the line, Up/Down recall what you typed before. PgUp/PgDn scroll
the transcript.";

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
}

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

        ChatApp {
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

    pub fn confirmed(self) -> Option<TaskConfig> {
        self.confirmed
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
            Slash::Clear => {
                self.history.clear();
                self.plan = None;
                self.plan_error = None;
                self.note("Conversation cleared. The next message starts a fresh one.");
            }
            Slash::Plan => match self.plan.as_ref() {
                None => self.note("No validated plan yet."),
                Some(plan) => {
                    let summary = plan.summary().join("\n");
                    self.note(summary);
                }
            },
            Slash::Run => self.confirm_plan(),
            Slash::Save(path) => self.save(path.as_deref()),
            Slash::Unknown(name) => {
                let text = format!("No command /{name}. Type /help for the list.");
                self.note(text);
            }
        }
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
                    self.drop_empty_stream_entry();
                    self.finish_streaming();
                    self.note(format!("the model call failed: {detail}"));
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
                let detail = error.to_string();
                self.plan_error = Some(detail.clone());
                self.note(format!(
                    "That plan did not validate, so it was not accepted:\n\n{detail}"
                ));
                // Tell the model what was wrong, so the next turn can fix it.
                self.history.push(ChatMessage::user(format!(
                    "The harness rejected that plan: {detail}\nReply with corrected plan JSON."
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

    pub fn on_key(&mut self, key: KeyEvent) {
        if key.kind != KeyEventKind::Press {
            return;
        }
        self.status.clear();

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

        match key.code {
            KeyCode::Esc => self.quit = true,
            KeyCode::Enter => self.submit(),
            KeyCode::Backspace => self.input.backspace(),
            KeyCode::Delete => self.input.delete(),
            KeyCode::Left => self.input.left(),
            KeyCode::Right => self.input.right(),
            KeyCode::Home => self.input.home(),
            KeyCode::End => self.input.end(),
            KeyCode::Up => self.input.recall_older(),
            KeyCode::Down => self.input.recall_newer(),
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
                Constraint::Length(3),
                Constraint::Length(1),
            ])
            .split(columns[0]);

        self.draw_transcript(frame, rows[0]);
        self.draw_input(frame, rows[1]);
        self.draw_footer(frame, rows[2]);
        self.draw_plan(frame, columns[1]);
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
                Style::default().fg(Color::DarkGray),
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

    fn draw_input(&self, frame: &mut Frame, area: Rect) {
        let busy = self.waiting.is_some();
        let inner_width = area.width.saturating_sub(4) as usize;
        let (shown, cursor_column) = self.input.view(inner_width);

        let (text, style) = if busy && self.input.is_empty() {
            (
                "waiting for the model…".to_string(),
                Style::default().fg(Color::DarkGray),
            )
        } else {
            (shown, Style::default().fg(Color::White))
        };

        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("> ", Style::default().fg(Color::Cyan)),
                Span::styled(text, style),
            ]))
            .block(Block::default().borders(Borders::ALL).title(" message ")),
            area,
        );

        if !busy || !self.input.is_empty() {
            // +1 for the border, +2 for the "> " prompt.
            frame.set_cursor_position(Position::new(area.x + 3 + cursor_column as u16, area.y + 1));
        }
    }

    fn draw_footer(&self, frame: &mut Frame, area: Rect) {
        let mut spans = vec![Span::styled(
            "Enter send   Ctrl-R run   /help commands   PgUp/PgDn scroll   Esc quit",
            Style::default().fg(Color::DarkGray),
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
                    Style::default().fg(Color::DarkGray),
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
                        Style::default().fg(Color::DarkGray),
                    )));
                }
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "chains supplied",
                    Style::default().fg(Color::DarkGray),
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
) -> io::Result<Option<TaskConfig>> {
    let mut app = ChatApp::new(provider, runtime, binder, target, seed);

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
    execute!(stdout, EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout))?;

    let outcome = event_loop(&mut terminal, &mut app);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
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
            if let TermEvent::Key(key) = event::read()? {
                app.on_key(key);
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
    fn the_visible_window_follows_the_cursor() {
        let mut input = TextInput::default();
        for character in "abcdefghijklmnop".chars() {
            input.insert(character);
        }
        // At the end of a long line the tail is visible and the cursor sits
        // just past it — the cursor takes a column of its own, so the text is
        // at most width-1 characters.
        let (text, column) = input.view(8);
        assert!(text.chars().count() <= 8, "{text:?}");
        assert!(text.ends_with('p'), "{text}");
        assert!(column < 8, "cursor column {column} is off-pane");
        assert_eq!(column, text.chars().count());

        input.home();
        let (text, column) = input.view(8);
        assert!(text.starts_with('a'), "{text}");
        assert_eq!(column, 0);

        // Somewhere in the middle, the cursor is still on-pane.
        for _ in 0..6 {
            input.right();
        }
        let (text, column) = input.view(8);
        assert!(column < 8, "cursor column {column} is off-pane");
        assert!(!text.is_empty());
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
        let config = app.confirmed().expect("a config was confirmed");
        assert_eq!(config.name, "demo");
        assert_eq!(config.design.cycles, 4);
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
