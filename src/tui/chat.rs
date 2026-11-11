//! The chat screen: describe a run in prose, see the plan, confirm it.
//!
//! This is the conversational entry point. You type what you want, the model
//! replies with a plan, and the plan is put through the harness's own
//! validation before it is shown to you — so the pane on the right is not the
//! model's claim about what will happen, it is what the typed config actually
//! resolves to: which residues may change, which may not, where the gates sit.
//! A plan that does not validate shows the error instead, and the model is told
//! what was wrong so it can correct itself.
//!
//! Nothing runs until you press Ctrl-R.
//!
//! The model call happens on the Tokio runtime while this loop keeps drawing,
//! so the interface stays responsive and a slow reply cannot wedge it.

use std::io;
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
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};
use ratatui::{Frame, Terminal};

use crate::agent::{ChatMessage, ModelProvider};
use crate::plan::{self, Chain, Plan};
use crate::TaskConfig;

/// What the model-call worker sends back.
enum Incoming {
    Reply(String),
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

pub struct ChatApp {
    provider: Arc<dyn ModelProvider>,
    runtime: tokio::runtime::Handle,
    binder: Vec<Chain>,
    target: Vec<Chain>,
    seed: u64,

    transcript: Vec<(Entry, String)>,
    /// What is sent to the model: the transcript without harness notes.
    history: Vec<ChatMessage>,
    input: String,
    waiting: Option<Receiver<Incoming>>,
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
        let mut transcript = Vec::new();
        transcript.push((
            Entry::Harness,
            format!(
                "Connected to {}. Describe the run you want and I will turn it into a plan.\n\n\
Sequences you supplied: {} binder chain(s), {} target chain(s). I will never ask the model to \
invent a sequence — it only chooses which spans are mutable, what to pin, and the search \
parameters.\n\nThe plan on the right is what the config actually resolves to, not what the model \
says it does. Nothing runs until you press Ctrl-R.",
                provider.model(),
                binder.len(),
                target.len()
            ),
        ));

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
            input: String::new(),
            waiting: None,
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

    fn send(&mut self) {
        let text = self.input.trim().to_string();
        if text.is_empty() || self.waiting.is_some() {
            return;
        }
        self.input.clear();
        self.transcript.push((Entry::You, text.clone()));
        self.history.push(ChatMessage::user(text));
        self.follow = true;

        let (tx, rx) = mpsc::channel();
        self.waiting = Some(rx);

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

        self.runtime.spawn(async move {
            let outcome = provider.converse(&system, &messages).await;
            let _ = tx.send(match outcome {
                Ok(reply) => Incoming::Reply(reply),
                Err(error) => Incoming::Failed(error.to_string()),
            });
        });
    }

    fn poll_worker(&mut self) {
        let Some(receiver) = self.waiting.as_ref() else {
            return;
        };
        match receiver.try_recv() {
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                self.waiting = None;
                self.status = "the request thread stopped without replying".to_string();
            }
            Ok(Incoming::Failed(detail)) => {
                self.waiting = None;
                self.transcript
                    .push((Entry::Harness, format!("the model call failed: {detail}")));
                self.follow = true;
            }
            Ok(Incoming::Reply(reply)) => {
                self.waiting = None;
                self.transcript.push((Entry::Model, reply.clone()));
                self.history.push(ChatMessage::assistant(reply.clone()));
                self.follow = true;
                self.validate(&reply);
            }
        }
    }

    /// Put the reply through the harness's own validation.
    fn validate(&mut self, reply: &str) {
        if self.binder.is_empty() || self.target.is_empty() {
            return;
        }
        match plan::plan_from_reply(reply, &self.binder, &self.target, self.seed) {
            Ok(plan) => {
                self.plan_error = None;
                self.transcript.push((
                    Entry::Harness,
                    format!(
                        "Plan validated: {}. Press Ctrl-R to run it, Ctrl-S to save it as \
task.yaml, or keep talking to change it.",
                        plan.scope.summary()
                    ),
                ));
                self.plan = Some(plan);
            }
            Err(plan::PlanError::NoJson) => {
                // Conversation without a plan in it is normal.
            }
            Err(error) => {
                let detail = error.to_string();
                self.plan_error = Some(detail.clone());
                self.transcript.push((
                    Entry::Harness,
                    format!("That plan did not validate, so it was not accepted:\n\n{detail}"),
                ));
                // Tell the model what was wrong, so the next turn can fix it.
                self.history.push(ChatMessage::user(format!(
                    "The harness rejected that plan: {detail}\nReply with corrected plan JSON."
                )));
            }
        }
    }

    pub fn on_key(&mut self, key: KeyEvent) {
        if key.kind != KeyEventKind::Press {
            return;
        }

        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('c') => self.quit = true,
                KeyCode::Char('u') => self.input.clear(),
                KeyCode::Char('r') => match self.plan.take() {
                    Some(plan) => {
                        self.confirmed = Some(plan.config);
                        self.quit = true;
                    }
                    None => self.status = "no validated plan to run yet".to_string(),
                },
                KeyCode::Char('s') => self.save(),
                _ => {}
            }
            return;
        }

        match key.code {
            KeyCode::Esc => self.quit = true,
            KeyCode::Enter => self.send(),
            KeyCode::Backspace => {
                self.input.pop();
            }
            KeyCode::PageUp => {
                self.follow = false;
                self.scroll = self.scroll.saturating_sub(8);
            }
            KeyCode::PageDown => {
                self.scroll = self.scroll.saturating_add(8);
            }
            KeyCode::End => self.follow = true,
            KeyCode::Char(character) => self.input.push(character),
            _ => {}
        }
    }

    fn save(&mut self) {
        let Some(plan) = self.plan.as_ref() else {
            self.status = "no validated plan to save yet".to_string();
            return;
        };
        match plan.config.to_yaml() {
            Err(error) => self.status = format!("could not serialise: {error}"),
            Ok(yaml) => match std::fs::write("task.yaml", yaml) {
                Ok(()) => {
                    self.status = "written to task.yaml".to_string();
                    self.transcript
                        .push((Entry::Harness, "Saved the plan to task.yaml.".to_string()));
                }
                Err(error) => self.status = format!("could not write task.yaml: {error}"),
            },
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
            lines.push(Line::from(Span::styled(
                format!("{}:", entry.label()),
                Style::default()
                    .fg(entry.color())
                    .add_modifier(Modifier::BOLD),
            )));
            for wrapped in wrap(text, width) {
                lines.push(Line::from(Span::styled(
                    format!("  {wrapped}"),
                    Style::default().fg(match entry {
                        Entry::Harness => Color::Yellow,
                        _ => Color::Gray,
                    }),
                )));
            }
            lines.push(Line::from(""));
        }

        if self.waiting.is_some() {
            const SPINNER: [&str; 4] = ["|", "/", "-", "\\"];
            lines.push(Line::from(Span::styled(
                format!("  {} thinking", SPINNER[(self.frame / 3) % SPINNER.len()]),
                Style::default().fg(Color::DarkGray),
            )));
        }

        let total = lines.len();
        if self.follow {
            self.scroll = total.saturating_sub(height);
        } else {
            self.scroll = self.scroll.min(total.saturating_sub(1));
        }

        let visible: Vec<ListItem> = lines
            .into_iter()
            .skip(self.scroll)
            .take(height)
            .map(ListItem::new)
            .collect();

        frame.render_widget(
            List::new(visible).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" conversation "),
            ),
            area,
        );
    }

    fn draw_input(&self, frame: &mut Frame, area: Rect) {
        let busy = self.waiting.is_some();
        let shown = if busy {
            "waiting for the model…".to_string()
        } else {
            format!("{}\u{2588}", self.input)
        };
        frame.render_widget(
            Paragraph::new(shown)
                .style(Style::default().fg(if busy { Color::DarkGray } else { Color::White }))
                .wrap(Wrap { trim: false })
                .block(Block::default().borders(Borders::ALL).title(" message ")),
            area,
        );
    }

    fn draw_footer(&self, frame: &mut Frame, area: Rect) {
        let mut spans = vec![Span::styled(
            "Enter send   Ctrl-R run plan   Ctrl-S save   Ctrl-U clear   PgUp/PgDn scroll   Esc quit",
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

        if event::poll(Duration::from_millis(80))? {
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

    #[test]
    fn wrapping_respects_width_and_newlines() {
        let lines = wrap("one two three four five", 9);
        assert!(
            lines.iter().all(|line| line.chars().count() <= 9),
            "{lines:?}"
        );
        assert_eq!(wrap("a\n\nb", 20), vec!["a", "", "b"]);

        // A word longer than the pane is split rather than overflowing.
        let long = wrap("QVQLVESGGGLVQAGGSLRLSCAAS", 10);
        assert!(
            long.iter().all(|line| line.chars().count() <= 10),
            "{long:?}"
        );
        assert_eq!(long.concat(), "QVQLVESGGGLVQAGGSLRLSCAAS");
    }

    #[tokio::test]
    async fn a_valid_reply_becomes_a_runnable_plan() {
        let mut app = app_with("");
        app.validate(
            r#"{"name":"demo","topology":"vhh","regions":[{"name":"loop1","chain":"H","start":26,"end":33}],"cycles":4}"#,
        );
        assert!(app.plan.is_some());
        assert!(app.plan_error.is_none());

        // Ctrl-R confirms it.
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
        // The rejection is appended to the history so the next turn sees it.
        assert!(app
            .history
            .last()
            .map(|message| message.text.contains("rejected"))
            .unwrap_or(false));

        // Ctrl-R must not run anything.
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
        assert_eq!(app.input, "design the loops");
        app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(app.input.is_empty());
        assert!(app.waiting.is_some());
        assert_eq!(
            app.transcript.last().map(|(entry, _)| *entry),
            Some(Entry::You)
        );
        assert_eq!(app.history.len(), 1);
        assert_eq!(app.history[0].role, ChatRole::User);

        // A second Enter while a request is in flight is ignored.
        app.on_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE));
        app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(app.history.len(), 1);
    }

    #[tokio::test]
    async fn the_screen_renders_at_several_sizes() {
        use ratatui::backend::TestBackend;

        let mut app = app_with("");
        app.validate(
            r#"{"name":"demo","topology":"vhh","regions":[{"name":"loop1","chain":"H","start":26,"end":33}]}"#,
        );

        for (width, height) in [(140, 44), (100, 30), (60, 16)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("terminal");
            terminal
                .draw(|frame| app.draw(frame))
                .unwrap_or_else(|error| panic!("failed at {width}x{height}: {error}"));
        }

        // And with a rejected plan, which renders a different pane.
        let mut rejected = app_with("");
        rejected.validate(
            r#"{"topology":"vhh","regions":[{"name":"x","chain":"H","start":900,"end":950}]}"#,
        );
        let mut terminal = Terminal::new(TestBackend::new(100, 30)).expect("terminal");
        terminal
            .draw(|frame| rejected.draw(frame))
            .expect("renders the rejection");
    }
}
