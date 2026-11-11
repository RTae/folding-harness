//! The terminal dashboard.
//!
//! The reference harness this project takes its shape from inspects runs in a
//! browser dashboard. Everything it shows can be shown in a terminal instead,
//! which keeps inspection available over SSH on a cluster login node with no
//! port forwarding: a loss curve, the retained population, the lineage of
//! parents and children, the event stream, and the predicted backbone drawn
//! with braille glyphs.
//!
//! It reads the task directory and holds no connection to the running process,
//! so it works the same on a live run and on one that finished last week — and
//! a crashed dashboard cannot disturb a design run.

pub mod chat;
pub mod structure;

use std::collections::HashMap;
use std::io::{self, IsTerminal};
use std::time::{Duration, Instant};

use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::{
    self, Event as TermEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers,
};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::canvas::{Canvas, Line as CanvasLine, Points};
use ratatui::widgets::{
    Axis, Block, Borders, Chart, Clear, Dataset, Gauge, GraphType, List, ListItem, Paragraph, Row,
    Table, TableState, Tabs, Wrap,
};
use ratatui::{Frame, Terminal};

use crate::trace::{
    read_after, Event as TraceEvent, PopulationSnapshot, RunManifest, StructureSnapshot, TaskPaths,
    TraceRecord,
};
use crate::{loss_breakdown, Candidate, CandidateId, MutableScope, Scaffold, TaskConfig};

use structure::{backbone_segments, Camera, ConfidenceBand};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Overview,
    Population,
    Lineage,
    Structure,
    Events,
}

impl Tab {
    pub const ALL: [Tab; 5] = [
        Tab::Overview,
        Tab::Population,
        Tab::Lineage,
        Tab::Structure,
        Tab::Events,
    ];

    pub fn title(&self) -> &'static str {
        match self {
            Tab::Overview => "Overview",
            Tab::Population => "Population",
            Tab::Lineage => "Lineage",
            Tab::Structure => "Structure",
            Tab::Events => "Events",
        }
    }

    fn index(&self) -> usize {
        Tab::ALL.iter().position(|tab| tab == self).unwrap_or(0)
    }

    /// Match a pane by name, case-insensitively, for `--pane`.
    pub fn from_name(name: &str) -> Option<Tab> {
        let wanted = name.trim().to_ascii_lowercase();
        Tab::ALL
            .iter()
            .copied()
            .find(|tab| tab.title().to_ascii_lowercase() == wanted)
    }

    pub fn names() -> String {
        Tab::ALL
            .iter()
            .map(|tab| tab.title().to_ascii_lowercase())
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn next(&self) -> Tab {
        Tab::ALL[(self.index() + 1) % Tab::ALL.len()]
    }

    fn previous(&self) -> Tab {
        Tab::ALL[(self.index() + Tab::ALL.len() - 1) % Tab::ALL.len()]
    }
}

pub struct App {
    paths: TaskPaths,
    scaffold: Option<Scaffold>,
    scope: Option<MutableScope>,
    config: Option<TaskConfig>,
    snapshot: Option<PopulationSnapshot>,
    manifest: Option<RunManifest>,
    events: Vec<TraceRecord>,
    last_seq: u64,
    tab: Tab,
    table_state: TableState,
    camera: Camera,
    structures: HashMap<CandidateId, Option<StructureSnapshot>>,
    event_scroll: Option<usize>,
    follow_best: bool,
    status: String,
    show_help: bool,
    should_quit: bool,
}

impl App {
    pub fn new(paths: TaskPaths) -> Self {
        let mut table_state = TableState::default();
        table_state.select(Some(0));
        App {
            paths,
            scaffold: None,
            scope: None,
            config: None,
            snapshot: None,
            manifest: None,
            events: Vec::new(),
            last_seq: 0,
            tab: Tab::Overview,
            table_state,
            camera: Camera::default(),
            structures: HashMap::new(),
            event_scroll: None,
            follow_best: true,
            status: String::new(),
            show_help: false,
            should_quit: false,
        }
    }

    pub fn should_quit(&self) -> bool {
        self.should_quit
    }

    /// Re-read the task directory. Missing files are not an error: a run that
    /// has only just started has a trace but no population snapshot yet.
    pub fn refresh(&mut self) {
        if self.config.is_none() {
            if let Ok(config) = TaskConfig::load(self.paths.config()) {
                if let Ok((scaffold, scope)) = config.resolve() {
                    self.scaffold = Some(scaffold);
                    self.scope = Some(scope);
                }
                self.config = Some(config);
            }
        }

        if let Ok(manifest) = crate::trace::read_json::<RunManifest>(self.paths.manifest()) {
            self.manifest = Some(manifest);
        }
        if let Ok(snapshot) = crate::trace::read_json::<PopulationSnapshot>(self.paths.population())
        {
            self.snapshot = Some(snapshot);
        }

        match read_after(self.paths.trace(), self.last_seq) {
            Ok(fresh) => {
                if let Some(last) = fresh.last() {
                    self.last_seq = last.seq;
                }
                self.events.extend(fresh);
            }
            Err(error) => self.status = format!("trace unreadable: {error}"),
        }

        let count = self.members().len();
        if self.follow_best && count > 0 {
            // Members arrive ranked best-first.
            self.table_state.select(Some(0));
        } else if let Some(selected) = self.table_state.selected() {
            if count == 0 {
                self.table_state.select(None);
            } else if selected >= count {
                self.table_state.select(Some(count - 1));
            }
        }
    }

    pub fn members(&self) -> &[Candidate] {
        self.snapshot
            .as_ref()
            .map(|snapshot| snapshot.members.as_slice())
            .unwrap_or(&[])
    }

    pub fn selected(&self) -> Option<&Candidate> {
        let index = self.table_state.selected()?;
        self.members().get(index)
    }

    /// Load and cache a candidate's trace. A `None` entry means "looked and
    /// did not find", so a missing file is not re-read every frame.
    fn structure_for(&mut self, id: CandidateId) -> Option<&StructureSnapshot> {
        if !self.structures.contains_key(&id) {
            let loaded =
                crate::trace::read_json::<StructureSnapshot>(self.paths.structure(id)).ok();
            self.structures.insert(id, loaded);
        }
        self.structures.get(&id).and_then(|entry| entry.as_ref())
    }

    pub fn on_key(&mut self, key: KeyEvent) {
        if key.kind != KeyEventKind::Press {
            return;
        }

        if key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('c')) {
            self.should_quit = true;
            return;
        }

        // While the overlay is up it swallows the keys that would otherwise
        // quit or move the selection, so a glance at the help cannot end the
        // session by accident.
        if self.show_help {
            match key.code {
                KeyCode::Char('?') | KeyCode::F(1) | KeyCode::Esc | KeyCode::Enter => {
                    self.show_help = false
                }
                KeyCode::Char('q') => self.should_quit = true,
                _ => {}
            }
            return;
        }

        match key.code {
            KeyCode::Char('?') | KeyCode::F(1) => self.show_help = true,
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            KeyCode::Tab => self.tab = self.tab.next(),
            KeyCode::BackTab => self.tab = self.tab.previous(),
            KeyCode::Char(digit @ '1'..='5') => {
                let index = digit as usize - '1' as usize;
                self.tab = Tab::ALL[index];
            }
            KeyCode::Char('f') => {
                self.follow_best = !self.follow_best;
                self.status = if self.follow_best {
                    "following the best candidate".to_string()
                } else {
                    "selection pinned".to_string()
                };
            }
            KeyCode::Down | KeyCode::Char('j') => self.move_selection(1),
            KeyCode::Up | KeyCode::Char('k') => self.move_selection(-1),
            KeyCode::Left | KeyCode::Char('h') => self.camera.turn(-0.15, 0.0),
            KeyCode::Right | KeyCode::Char('l') => self.camera.turn(0.15, 0.0),
            KeyCode::Char('u') => self.camera.turn(0.0, 0.12),
            KeyCode::Char('d') => self.camera.turn(0.0, -0.12),
            KeyCode::Char('+') | KeyCode::Char('=') => self.camera.scale(1.2),
            KeyCode::Char('-') | KeyCode::Char('_') => self.camera.scale(1.0 / 1.2),
            KeyCode::Char('0') => self.camera.reset(),
            KeyCode::PageDown => self.scroll_events(10),
            KeyCode::PageUp => self.scroll_events(-10),
            KeyCode::End => self.event_scroll = None,
            _ => {}
        }
    }

    fn move_selection(&mut self, delta: isize) {
        if matches!(self.tab, Tab::Events) {
            self.scroll_events(delta);
            return;
        }
        let count = self.members().len();
        if count == 0 {
            return;
        }
        self.follow_best = false;
        let current = self.table_state.selected().unwrap_or(0) as isize;
        let next = (current + delta).clamp(0, count as isize - 1);
        self.table_state.select(Some(next as usize));
    }

    fn scroll_events(&mut self, delta: isize) {
        let count = self.events.len();
        if count == 0 {
            return;
        }
        let current = self.event_scroll.unwrap_or(count.saturating_sub(1)) as isize;
        let next = (current + delta).clamp(0, count as isize - 1);
        self.event_scroll = Some(next as usize);
    }

    pub fn draw(&mut self, frame: &mut Frame) {
        let area = frame.area();
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),
                Constraint::Length(2),
                Constraint::Min(8),
                Constraint::Length(1),
            ])
            .split(area);

        self.draw_header(frame, rows[0]);
        self.draw_tabs(frame, rows[1]);
        match self.tab {
            Tab::Overview => self.draw_overview(frame, rows[2]),
            Tab::Population => self.draw_population(frame, rows[2]),
            Tab::Lineage => self.draw_lineage(frame, rows[2]),
            Tab::Structure => self.draw_structure(frame, rows[2]),
            Tab::Events => self.draw_events(frame, rows[2]),
        }
        self.draw_footer(frame, rows[3]);

        if self.show_help {
            self.draw_help(frame, area);
        }
    }

    /// A centred box of the given size, clamped to the area.
    fn centred(area: Rect, width: u16, height: u16) -> Rect {
        let width = width.min(area.width);
        let height = height.min(area.height);
        Rect {
            x: area.x + (area.width.saturating_sub(width)) / 2,
            y: area.y + (area.height.saturating_sub(height)) / 2,
            width,
            height,
        }
    }

    /// Every key, grouped, with the current pane's group marked.
    ///
    /// The footer only has room for a handful, and a key you have to remember
    /// is a key you will not use.
    fn draw_help(&self, frame: &mut Frame, area: Rect) {
        let mut lines: Vec<Line> = Vec::new();
        let heading = |text: &str, active: bool| {
            Line::from(Span::styled(
                if active {
                    format!("{text}  (this pane)")
                } else {
                    text.to_string()
                },
                Style::default()
                    .fg(if active { Color::Cyan } else { Color::White })
                    .add_modifier(Modifier::BOLD),
            ))
        };
        let key = |keys: &str, what: &str| {
            Line::from(vec![
                Span::styled(format!("  {keys:<22}"), Style::default().fg(Color::White)),
                Span::styled(what.to_string(), Style::default().fg(Color::Gray)),
            ])
        };

        lines.push(heading("panes", false));
        lines.push(key("1 2 3 4 5", "jump to a pane"));
        lines.push(key("Tab / Shift-Tab", "next / previous pane"));
        lines.push(Line::from(""));

        lines.push(heading(
            "candidates",
            matches!(self.tab, Tab::Overview | Tab::Population | Tab::Lineage),
        ));
        lines.push(key("j / Down", "next candidate"));
        lines.push(key("k / Up", "previous candidate"));
        lines.push(key("f", "follow the best candidate"));
        lines.push(Line::from(""));

        lines.push(heading("structure", matches!(self.tab, Tab::Structure)));
        lines.push(key("h / l", "turn left / right"));
        lines.push(key("u / d", "tilt up / down"));
        lines.push(key("+ / -", "zoom in / out"));
        lines.push(key("0", "reset the view"));
        lines.push(Line::from(""));

        lines.push(heading("events", matches!(self.tab, Tab::Events)));
        lines.push(key("j / k", "scroll one line"));
        lines.push(key("PgUp / PgDn", "scroll a page"));
        lines.push(key("End", "follow new events again"));
        lines.push(Line::from(""));

        lines.push(heading("session", false));
        lines.push(key("? / F1", "open or close this help"));
        lines.push(key("q / Esc / Ctrl-C", "quit the dashboard"));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  Quitting does not stop the run.",
            Style::default().fg(Color::Gray),
        )));
        lines.push(Line::from(Span::styled(
            "  Reopen this view with `fh board`.",
            Style::default().fg(Color::Gray),
        )));

        let height = lines.len() as u16 + 2;
        let popup = Self::centred(area, 62, height);

        frame.render_widget(Clear, popup);
        frame.render_widget(
            Paragraph::new(lines).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" keys — ? or Esc to close "),
            ),
            popup,
        );
    }

    fn draw_header(&self, frame: &mut Frame, area: Rect) {
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
            .split(area);

        let name = self
            .snapshot
            .as_ref()
            .map(|s| s.name.clone())
            .or_else(|| self.manifest.as_ref().map(|m| m.name.clone()))
            .unwrap_or_else(|| "(no task loaded)".to_string());

        let mut lines = vec![Line::from(vec![
            Span::styled(
                name,
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(
                self.manifest
                    .as_ref()
                    .map(|m| m.task_id.clone())
                    .unwrap_or_default(),
                Style::default().fg(Color::Gray),
            ),
        ])];

        if let Some(manifest) = &self.manifest {
            lines.push(Line::from(vec![
                Span::styled("backend ", Style::default().fg(Color::Gray)),
                Span::raw(manifest.backend.clone()),
                Span::styled("  model ", Style::default().fg(Color::Gray)),
                Span::raw(manifest.model.clone()),
                Span::styled("  protocol ", Style::default().fg(Color::Gray)),
                Span::raw(manifest.protocol.clone()),
                Span::styled("  seed ", Style::default().fg(Color::Gray)),
                Span::raw(manifest.seed.to_string()),
            ]));
            lines.push(Line::from(vec![
                Span::styled("config ", Style::default().fg(Color::Gray)),
                Span::raw(manifest.config_sha256.chars().take(12).collect::<String>()),
                Span::styled("  folds ", Style::default().fg(Color::Gray)),
                Span::raw(manifest.folds.max(self.folds()).to_string()),
            ]));
        }
        if let Some(snapshot) = &self.snapshot {
            lines.push(Line::from(vec![
                Span::styled("scope ", Style::default().fg(Color::Gray)),
                Span::raw(snapshot.scope.clone()),
            ]));
        }

        frame.render_widget(
            Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" run ")),
            columns[0],
        );

        let (cycle, planned, running) = self
            .snapshot
            .as_ref()
            .map(|s| (s.cycle, s.cycles_planned.max(1), s.running))
            .unwrap_or((0, 1, false));
        let ratio = (f64::from(cycle) / f64::from(planned)).clamp(0.0, 1.0);

        let stats = self.snapshot.as_ref().map(|s| s.stats);
        let best = stats
            .and_then(|s| s.best_loss)
            .map(|l| format!("{l:.4}"))
            .unwrap_or_else(|| "-".to_string());
        let mean = stats
            .and_then(|s| s.mean_loss)
            .map(|l| format!("{l:.4}"))
            .unwrap_or_else(|| "-".to_string());

        let inner = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(1)])
            .split(
                Block::default()
                    .borders(Borders::ALL)
                    .title(if running { " running " } else { " finished " })
                    .inner(columns[1]),
            );

        frame.render_widget(
            Block::default().borders(Borders::ALL).title(if running {
                " running "
            } else {
                " finished "
            }),
            columns[1],
        );
        frame.render_widget(
            Gauge::default()
                .gauge_style(Style::default().fg(if running { Color::Cyan } else { Color::Green }))
                .ratio(ratio)
                .label(format!("cycle {cycle}/{planned}")),
            inner[0],
        );
        frame.render_widget(
            Paragraph::new(vec![
                Line::from(vec![
                    Span::styled("best loss ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        best,
                        Style::default()
                            .fg(Color::Green)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled("   mean ", Style::default().fg(Color::Gray)),
                    Span::raw(mean),
                ]),
                Line::from(vec![
                    Span::styled("population ", Style::default().fg(Color::Gray)),
                    Span::raw(self.members().len().to_string()),
                    Span::styled("   events ", Style::default().fg(Color::Gray)),
                    Span::raw(self.events.len().to_string()),
                ]),
            ]),
            inner[1],
        );
    }

    fn folds(&self) -> usize {
        self.snapshot.as_ref().map(|s| s.folds).unwrap_or(0)
    }

    fn draw_tabs(&self, frame: &mut Frame, area: Rect) {
        let titles: Vec<Line> = Tab::ALL
            .iter()
            .enumerate()
            .map(|(index, tab)| {
                Line::from(vec![
                    Span::styled(format!("{} ", index + 1), Style::default().fg(Color::Gray)),
                    Span::raw(tab.title()),
                ])
            })
            .collect();

        frame.render_widget(
            Tabs::new(titles)
                .select(self.tab.index())
                .highlight_style(
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )
                .divider(" "),
            area,
        );
    }

    fn draw_overview(&mut self, frame: &mut Frame, area: Rect) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
            .split(area);

        // --- loss curve ---
        let curve: Vec<(f64, f64)> = self
            .snapshot
            .as_ref()
            .map(|snapshot| {
                snapshot
                    .loss_curve
                    .iter()
                    .enumerate()
                    .map(|(index, loss)| ((index + 1) as f64, *loss))
                    .collect()
            })
            .unwrap_or_default();

        if curve.len() < 2 {
            frame.render_widget(
                Paragraph::new("waiting for a second completed cycle to plot the curve")
                    .style(Style::default().fg(Color::Gray))
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title(" best loss per cycle "),
                    ),
                rows[0],
            );
        } else {
            let max_x = curve.last().map(|point| point.0).unwrap_or(1.0);
            let min_y = curve.iter().map(|p| p.1).fold(f64::MAX, f64::min);
            let max_y = curve.iter().map(|p| p.1).fold(f64::MIN, f64::max);
            let pad = ((max_y - min_y) * 0.12).max(0.002);

            let datasets = vec![Dataset::default()
                .name("best")
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Green))
                .data(&curve)];

            frame.render_widget(
                Chart::new(datasets)
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title(" best loss per cycle (lower is better) "),
                    )
                    .x_axis(
                        Axis::default()
                            .style(Style::default().fg(Color::Gray))
                            .bounds([1.0, max_x.max(2.0)])
                            .labels(vec![
                                Span::raw("1"),
                                Span::raw(format!("{:.0}", max_x.max(2.0))),
                            ]),
                    )
                    .y_axis(
                        Axis::default()
                            .style(Style::default().fg(Color::Gray))
                            .bounds([min_y - pad, max_y + pad])
                            .labels(vec![
                                Span::raw(format!("{:.3}", min_y - pad)),
                                Span::raw(format!("{:.3}", max_y + pad)),
                            ]),
                    ),
                rows[0],
            );
        }

        // --- cycle table and counters ---
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
            .split(rows[1]);

        let records = self
            .snapshot
            .as_ref()
            .map(|snapshot| snapshot.records.clone())
            .unwrap_or_default();

        let cycle_rows: Vec<Row> = records
            .iter()
            .rev()
            .take(20)
            .map(|record| {
                let improved = if record.improved { "+" } else { " " };
                Row::new(vec![
                    record.cycle.to_string(),
                    record.skill.to_string(),
                    format!("{}/{}", record.admitted, record.proposed),
                    record
                        .best_loss
                        .map(|l| format!("{l:.4}"))
                        .unwrap_or_else(|| "-".into()),
                    improved.to_string(),
                ])
                .style(if record.improved {
                    Style::default().fg(Color::Green)
                } else {
                    Style::default()
                })
            })
            .collect();

        frame.render_widget(
            Table::new(
                cycle_rows,
                [
                    Constraint::Length(5),
                    Constraint::Min(16),
                    Constraint::Length(7),
                    Constraint::Length(9),
                    Constraint::Length(2),
                ],
            )
            .header(
                Row::new(vec!["cyc", "skill", "adm", "best", ""])
                    .style(Style::default().fg(Color::Gray)),
            )
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" cycles (newest first) "),
            ),
            columns[0],
        );

        let mut lines: Vec<Line> = Vec::new();
        lines.push(Line::from(Span::styled(
            "rejections by kind",
            Style::default().fg(Color::Gray),
        )));
        let mut counts: HashMap<String, usize> = HashMap::new();
        for record in &self.events {
            if let TraceEvent::Rejected { kind, .. } = &record.event {
                *counts.entry(kind.clone()).or_insert(0) += 1;
            }
            if let TraceEvent::FoldFailed { .. } = &record.event {
                *counts.entry("fold_failed".to_string()).or_insert(0) += 1;
            }
        }
        if counts.is_empty() {
            lines.push(Line::from(Span::styled(
                "  none yet",
                Style::default().fg(Color::Gray),
            )));
        } else {
            let mut pairs: Vec<(String, usize)> = counts.into_iter().collect();
            pairs.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
            for (kind, count) in pairs {
                lines.push(Line::from(vec![
                    Span::raw("  "),
                    Span::styled(format!("{count:>4} "), Style::default().fg(Color::Red)),
                    Span::raw(kind),
                ]));
            }
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "last routing decision",
            Style::default().fg(Color::Gray),
        )));
        let routed = self
            .events
            .iter()
            .rev()
            .find_map(|record| match &record.event {
                TraceEvent::SkillRouted { legal, blocked, .. } => {
                    Some((legal.clone(), blocked.clone()))
                }
                _ => None,
            });
        match routed {
            None => lines.push(Line::from(Span::styled(
                "  not yet",
                Style::default().fg(Color::Gray),
            ))),
            Some((legal, blocked)) => {
                lines.push(Line::from(vec![
                    Span::raw("  legal: "),
                    Span::styled(legal.join(", "), Style::default().fg(Color::Green)),
                ]));
                for reason in blocked.iter().take(4) {
                    lines.push(Line::from(vec![
                        Span::raw("  - "),
                        Span::styled(reason.clone(), Style::default().fg(Color::Gray)),
                    ]));
                }
            }
        }

        frame.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: true })
                .block(Block::default().borders(Borders::ALL).title(" search ")),
            columns[1],
        );
    }

    fn draw_population(&mut self, frame: &mut Frame, area: Rect) {
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(56), Constraint::Percentage(44)])
            .split(area);

        let members: Vec<Candidate> = self.members().to_vec();
        let rows: Vec<Row> = members
            .iter()
            .enumerate()
            .map(|(rank, candidate)| {
                let metrics = candidate.metrics.as_ref();
                Row::new(vec![
                    format!("{}", rank + 1),
                    candidate.id.to_string(),
                    candidate
                        .parent
                        .map(|p| p.to_string())
                        .unwrap_or_else(|| "-".into()),
                    candidate.origin.label().to_string(),
                    candidate
                        .loss
                        .map(|l| format!("{l:.4}"))
                        .unwrap_or_else(|| "-".into()),
                    metrics
                        .map(|m| format!("{:.3}", m.iptm))
                        .unwrap_or_else(|| "-".into()),
                    metrics
                        .map(|m| format!("{:.1}", m.interface_plddt))
                        .unwrap_or_else(|| "-".into()),
                    metrics
                        .map(|m| m.clashes.to_string())
                        .unwrap_or_else(|| "-".into()),
                    candidate.substitutions.len().to_string(),
                ])
            })
            .collect();

        let table = Table::new(
            rows,
            [
                Constraint::Length(4),
                Constraint::Length(7),
                Constraint::Length(7),
                Constraint::Min(14),
                Constraint::Length(8),
                Constraint::Length(6),
                Constraint::Length(6),
                Constraint::Length(4),
                Constraint::Length(4),
            ],
        )
        .header(
            Row::new(vec![
                "#", "id", "parent", "origin", "loss", "ipTM", "iface", "cl", "sub",
            ])
            .style(Style::default().fg(Color::Gray)),
        )
        .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED | Modifier::BOLD))
        .highlight_symbol("> ")
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" retained population (best first) "),
        );

        frame.render_stateful_widget(table, columns[0], &mut self.table_state);
        self.draw_candidate_detail(frame, columns[1]);
    }

    fn draw_candidate_detail(&mut self, frame: &mut Frame, area: Rect) {
        let selected = self.selected().cloned();
        let Some(candidate) = selected else {
            frame.render_widget(
                Paragraph::new("no candidate selected")
                    .style(Style::default().fg(Color::Gray))
                    .block(Block::default().borders(Borders::ALL).title(" detail ")),
                area,
            );
            return;
        };

        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(13), Constraint::Min(4)])
            .split(area);

        let mut lines: Vec<Line> = vec![Line::from(vec![
            Span::styled(
                candidate.id.to_string(),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  cycle ", Style::default().fg(Color::Gray)),
            Span::raw(candidate.cycle.to_string()),
            Span::styled("  via ", Style::default().fg(Color::Gray)),
            Span::styled(
                candidate.origin.label().to_string(),
                Style::default().fg(Color::Cyan),
            ),
        ])];

        lines.push(Line::from(Span::styled(
            candidate.rationale.clone(),
            Style::default().fg(Color::Gray),
        )));
        lines.push(Line::from(""));

        if let (Some(metrics), Some(config)) = (candidate.metrics.as_ref(), self.config.as_ref()) {
            let breakdown = loss_breakdown(metrics, &config.loss);
            lines.push(Line::from(Span::styled(
                format!("loss {:.4} by term", breakdown.total),
                Style::default().fg(Color::Gray),
            )));
            let mut terms = breakdown.terms.clone();
            terms.sort_by(|a, b| b.contribution.total_cmp(&a.contribution));
            for term in terms.iter().take(8) {
                let width = (term.contribution / breakdown.total.max(1e-9) * 18.0).round() as usize;
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("{:>19} ", term.name),
                        Style::default().fg(Color::Gray),
                    ),
                    Span::styled(
                        "\u{2588}".repeat(width.min(18)),
                        Style::default().fg(if term.contribution > breakdown.total * 0.25 {
                            Color::Red
                        } else {
                            Color::Yellow
                        }),
                    ),
                    Span::raw(format!(" {:.4}", term.contribution)),
                ]));
            }
        }

        frame.render_widget(
            Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" detail ")),
            rows[0],
        );

        self.draw_sequence(frame, rows[1], &candidate);
    }

    /// The binder sequence, marking what may change and what did.
    fn draw_sequence(&self, frame: &mut Frame, area: Rect, candidate: &Candidate) {
        let Some(scaffold) = self.scaffold.as_ref() else {
            frame.render_widget(
                Paragraph::new("task.yaml not readable, so the mutable scope is unknown")
                    .style(Style::default().fg(Color::Gray))
                    .block(Block::default().borders(Borders::ALL).title(" sequence ")),
                area,
            );
            return;
        };

        let mut lines: Vec<Line> = Vec::new();
        for chain in scaffold.binder_chains() {
            let founder: Vec<char> = chain.sequence.chars().collect();
            let current: Vec<char> = candidate
                .sequence
                .chain(chain.id)
                .unwrap_or(&chain.sequence)
                .chars()
                .collect();

            lines.push(Line::from(vec![
                Span::styled(
                    format!("chain {} ", chain.label),
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("{} residues", current.len()),
                    Style::default().fg(Color::Gray),
                ),
            ]));

            let width = area.width.saturating_sub(10).max(20) as usize;
            for (row, window) in current.chunks(width).enumerate() {
                let start = row * width;
                let mut spans = vec![Span::styled(
                    format!("{:>5} ", start + 1),
                    Style::default().fg(Color::Gray),
                )];
                for (offset, residue) in window.iter().enumerate() {
                    let position = start + offset;
                    let mutable = self
                        .scope
                        .as_ref()
                        .map(|scope| scope.is_mutable(chain.id, position))
                        .unwrap_or(false);
                    let changed = founder.get(position) != Some(residue);
                    let style = match (mutable, changed) {
                        (_, true) => Style::default()
                            .fg(Color::Black)
                            .bg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                        (true, false) => Style::default().fg(Color::Green),
                        (false, false) => Style::default().fg(Color::Gray),
                    };
                    spans.push(Span::styled(residue.to_string(), style));
                }
                lines.push(Line::from(spans));
            }
        }

        frame.render_widget(
            Paragraph::new(lines).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" sequence: grey fixed, green mutable, yellow changed "),
            ),
            area,
        );
    }

    fn draw_lineage(&mut self, frame: &mut Frame, area: Rect) {
        let members: Vec<Candidate> = self.members().to_vec();
        let best = members.first().map(|candidate| candidate.id);
        let selected = self.selected().map(|candidate| candidate.id);

        let mut children: HashMap<Option<CandidateId>, Vec<&Candidate>> = HashMap::new();
        let present: Vec<CandidateId> = members.iter().map(|c| c.id).collect();
        for candidate in &members {
            // A parent evicted from the population is no longer a node, so its
            // children are shown as roots rather than hidden.
            let key = match candidate.parent {
                Some(parent) if present.contains(&parent) => Some(parent),
                _ => None,
            };
            children.entry(key).or_default().push(candidate);
        }
        for list in children.values_mut() {
            list.sort_by(|a, b| a.loss_or_inf().total_cmp(&b.loss_or_inf()));
        }

        let mut lines: Vec<Line> = Vec::new();
        let mut stack: Vec<(Option<CandidateId>, String)> = vec![(None, String::new())];
        let mut guard = 0usize;

        while let Some((parent, prefix)) = stack.pop() {
            guard += 1;
            if guard > 4096 {
                break;
            }
            let Some(list) = children.get(&parent) else {
                continue;
            };
            let count = list.len();
            // Push in reverse so the best child renders first.
            for (index, candidate) in list.iter().enumerate().rev() {
                let last = index + 1 == count;
                let branch = if prefix.is_empty() {
                    String::new()
                } else if last {
                    format!("{prefix}\u{2514}\u{2500} ")
                } else {
                    format!("{prefix}\u{251c}\u{2500} ")
                };
                let child_prefix = if prefix.is_empty() {
                    "  ".to_string()
                } else if last {
                    format!("{prefix}   ")
                } else {
                    format!("{prefix}\u{2502}  ")
                };

                let mut style = Style::default();
                if Some(candidate.id) == best {
                    style = style.fg(Color::Green).add_modifier(Modifier::BOLD);
                }
                if Some(candidate.id) == selected {
                    style = style.add_modifier(Modifier::REVERSED);
                }

                lines.push(Line::from(vec![
                    Span::styled(branch, Style::default().fg(Color::Gray)),
                    Span::styled(candidate.id.to_string(), style),
                    Span::styled(
                        format!("  {:.4}  ", candidate.loss.unwrap_or(f64::NAN)),
                        Style::default().fg(Color::White),
                    ),
                    Span::styled(
                        candidate.origin.label().to_string(),
                        Style::default().fg(Color::Cyan),
                    ),
                    Span::styled(
                        format!("  cycle {}", candidate.cycle),
                        Style::default().fg(Color::Gray),
                    ),
                ]));

                stack.push((Some(candidate.id), child_prefix));
            }
        }

        if lines.is_empty() {
            lines.push(Line::from(Span::styled(
                "no candidates yet",
                Style::default().fg(Color::Gray),
            )));
        }

        frame.render_widget(
            Paragraph::new(lines).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" lineage of retained candidates (recorded parents, not a phylogeny) "),
            ),
            area,
        );
    }

    fn draw_structure(&mut self, frame: &mut Frame, area: Rect) {
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(30), Constraint::Length(34)])
            .split(area);

        let selected = self.selected().cloned();
        let Some(candidate) = selected else {
            frame.render_widget(
                Paragraph::new("no candidate selected")
                    .style(Style::default().fg(Color::Gray))
                    .block(Block::default().borders(Borders::ALL).title(" structure ")),
                area,
            );
            return;
        };

        let camera = self.camera;
        let snapshot = self.structure_for(candidate.id).cloned();

        let Some(snapshot) = snapshot else {
            frame.render_widget(
                Paragraph::new(format!(
                    "no stored backbone for {}\n\nthe service returned no alpha-carbon trace for this candidate",
                    candidate.id
                ))
                .style(Style::default().fg(Color::Gray))
                .wrap(Wrap { trim: true })
                .block(Block::default().borders(Borders::ALL).title(" structure ")),
                area,
            );
            return;
        };

        let points = structure::project(&snapshot.ca_trace, &camera);
        let segments = backbone_segments(&points);

        let bucket = |band: ConfidenceBand| -> Vec<(f64, f64)> {
            points
                .iter()
                .filter(|point| point.designed && ConfidenceBand::of(point.plddt) == band)
                .map(|point| (point.x, point.y))
                .collect()
        };
        let very_high = bucket(ConfidenceBand::VeryHigh);
        let confident = bucket(ConfidenceBand::Confident);
        let low = bucket(ConfidenceBand::Low);
        let very_low = bucket(ConfidenceBand::VeryLow);
        let target: Vec<(f64, f64)> = points
            .iter()
            .filter(|point| !point.designed)
            .map(|point| (point.x, point.y))
            .collect();

        let (x_bounds, y_bounds) =
            structure::canvas_bounds(columns[0].width, columns[0].height, camera.zoom);

        let canvas = Canvas::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(format!(" {} backbone ", candidate.id)),
            )
            .marker(Marker::Braille)
            .x_bounds(x_bounds)
            .y_bounds(y_bounds)
            .paint(move |ctx| {
                for (from, to) in &segments {
                    let color = if from.designed {
                        match ConfidenceBand::of(from.plddt) {
                            ConfidenceBand::VeryHigh => Color::Blue,
                            ConfidenceBand::Confident => Color::Cyan,
                            ConfidenceBand::Low => Color::Yellow,
                            ConfidenceBand::VeryLow => Color::Red,
                        }
                    } else {
                        Color::Gray
                    };
                    ctx.draw(&CanvasLine {
                        x1: from.x,
                        y1: from.y,
                        x2: to.x,
                        y2: to.y,
                        color,
                    });
                }
                ctx.layer();
                ctx.draw(&Points {
                    coords: &target,
                    color: Color::Gray,
                });
                ctx.draw(&Points {
                    coords: &very_low,
                    color: Color::Red,
                });
                ctx.draw(&Points {
                    coords: &low,
                    color: Color::Yellow,
                });
                ctx.draw(&Points {
                    coords: &confident,
                    color: Color::Cyan,
                });
                ctx.draw(&Points {
                    coords: &very_high,
                    color: Color::Blue,
                });
            });

        frame.render_widget(canvas, columns[0]);

        let designed = snapshot.ca_trace.iter().filter(|r| r.designed).count();
        let mut lines = vec![
            Line::from(vec![
                Span::styled("candidate ", Style::default().fg(Color::Gray)),
                Span::raw(candidate.id.to_string()),
            ]),
            Line::from(vec![
                Span::styled("residues  ", Style::default().fg(Color::Gray)),
                Span::raw(format!("{} ({designed} designed)", snapshot.ca_trace.len())),
            ]),
            Line::from(vec![
                Span::styled("view      ", Style::default().fg(Color::Gray)),
                Span::raw(format!(
                    "yaw {:.2} pitch {:.2} zoom {:.2}",
                    camera.yaw, camera.pitch, camera.zoom
                )),
            ]),
            Line::from(""),
            Line::from(Span::styled("confidence", Style::default().fg(Color::Gray))),
        ];

        for (band, color) in [
            (ConfidenceBand::VeryHigh, Color::Blue),
            (ConfidenceBand::Confident, Color::Cyan),
            (ConfidenceBand::Low, Color::Yellow),
            (ConfidenceBand::VeryLow, Color::Red),
        ] {
            lines.push(Line::from(vec![
                Span::styled("  \u{2588}\u{2588} ", Style::default().fg(color)),
                Span::raw(band.label()),
            ]));
        }
        lines.push(Line::from(vec![
            Span::styled("  \u{2588}\u{2588} ", Style::default().fg(Color::Gray)),
            Span::raw("target (fixed)"),
        ]));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "h/l yaw  u/d pitch  +/- zoom  0 reset",
            Style::default().fg(Color::Gray),
        )));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "a synthetic backbone when the fixture backend produced it",
            Style::default().fg(Color::Gray),
        )));

        frame.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: true })
                .block(Block::default().borders(Borders::ALL).title(" view ")),
            columns[1],
        );
    }

    fn draw_events(&mut self, frame: &mut Frame, area: Rect) {
        let height = area.height.saturating_sub(2) as usize;
        let total = self.events.len();
        let anchor = self.event_scroll.unwrap_or(total.saturating_sub(1));
        let start = anchor.saturating_sub(height.saturating_sub(1));
        let first_at = self.events.first().map(|record| record.at_ms).unwrap_or(0);

        let items: Vec<ListItem> = self
            .events
            .iter()
            .skip(start)
            .take(height)
            .map(|record| {
                let kind = record.event.kind();
                let color = match kind {
                    "admitted" | "founder_scored" => Color::Green,
                    "rejected" | "fold_failed" | "task_failed" => Color::Red,
                    "skill_chosen" | "skill_routed" => Color::Cyan,
                    "cycle_started" | "cycle_finished" => Color::Blue,
                    "stagnation" => Color::Yellow,
                    "note" => Color::Magenta,
                    "task_started" | "task_finished" => Color::White,
                    _ => Color::Gray,
                };
                let seconds = (record.at_ms.saturating_sub(first_at)) as f64 / 1000.0;
                ListItem::new(Line::from(vec![
                    Span::styled(
                        format!("{:>6.1}s ", seconds),
                        Style::default().fg(Color::Gray),
                    ),
                    Span::styled(format!("{kind:<16}"), Style::default().fg(color)),
                    Span::raw(record.event.summary()),
                ]))
            })
            .collect();

        let title = match self.event_scroll {
            None => format!(" events ({total}, following) "),
            Some(at) => format!(" events ({}/{total}, End to follow) ", at + 1),
        };

        frame.render_widget(
            List::new(items).block(Block::default().borders(Borders::ALL).title(title)),
            area,
        );
    }

    fn draw_footer(&self, frame: &mut Frame, area: Rect) {
        let keys = match self.tab {
            Tab::Structure => {
                "? help   q quit   1-5/Tab panes   j/k select   h/l/u/d rotate   +/- zoom   0 reset"
            }
            Tab::Events => "? help   q quit   1-5/Tab panes   j/k scroll   PgUp/PgDn page",
            _ => "? help   q quit   1-5/Tab panes   j/k select   f follow-best",
        };
        let mut spans = vec![Span::styled(keys, Style::default().fg(Color::Gray))];
        if !self.status.is_empty() {
            spans.push(Span::raw("  |  "));
            spans.push(Span::styled(
                self.status.clone(),
                Style::default().fg(Color::Yellow),
            ));
        }
        frame.render_widget(
            Paragraph::new(Line::from(spans)).alignment(Alignment::Left),
            area,
        );
    }
}

/// Render one pane once, as plain text, without taking over the terminal.
///
/// For looking at a finished run over a pipe, pasting a backbone into a note,
/// or checking a pane in CI. Colour carries the confidence bands in the live
/// dashboard and is dropped here; the braille still carries the shape.
pub fn snapshot(
    paths: TaskPaths,
    tab: Tab,
    width: u16,
    height: u16,
    camera: Option<Camera>,
    keys: bool,
) -> io::Result<String> {
    use ratatui::backend::TestBackend;

    let mut app = App::new(paths);
    app.refresh();
    app.tab = tab;
    if let Some(camera) = camera {
        app.camera = camera;
    }
    app.show_help = keys;

    let mut terminal = Terminal::new(TestBackend::new(width.max(40), height.max(12)))?;
    terminal.draw(|frame| app.draw(frame))?;

    let buffer = terminal.backend().buffer();
    let area = buffer.area;
    let cells = buffer.content();

    let mut out = String::new();
    for y in 0..area.height {
        let mut line = String::new();
        for x in 0..area.width {
            let index = y as usize * area.width as usize + x as usize;
            if let Some(cell) = cells.get(index) {
                line.push_str(cell.symbol());
            }
        }
        out.push_str(line.trim_end());
        out.push('\n');
    }
    Ok(out)
}

/// Run the dashboard until the user quits.
pub fn run(paths: TaskPaths, tick: Duration) -> io::Result<()> {
    let mut app = App::new(paths);
    app.refresh();

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
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let outcome = event_loop(&mut terminal, &mut app, tick);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    outcome
}

fn event_loop<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
    tick: Duration,
) -> io::Result<()> {
    let mut last_refresh = Instant::now();
    loop {
        terminal.draw(|frame| app.draw(frame))?;

        if event::poll(Duration::from_millis(100))? {
            if let TermEvent::Key(key) = event::read()? {
                app.on_key(key);
            }
        }

        if last_refresh.elapsed() >= tick {
            app.refresh();
            last_refresh = Instant::now();
        }

        if app.should_quit() {
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Terminal themes vary, and ANSI bright-black disappears on several of
    /// them. Anything carrying meaning has to survive that, so the dimmest
    /// colour either screen uses is `Gray`.
    #[test]
    fn no_pane_draws_meaning_in_a_colour_that_vanishes_on_some_themes() {
        // Split so this test's own source does not match the needle.
        let needle = concat!("Color::", "DarkGray");
        for (name, source) in [
            ("tui/mod.rs", include_str!("mod.rs")),
            ("tui/chat.rs", include_str!("chat.rs")),
        ] {
            let uses = source.lines().filter(|line| line.contains(needle)).count();
            assert_eq!(uses, 0, "{name} still draws in {needle}");
        }
    }

    #[test]
    fn tabs_cycle_in_both_directions() {
        assert_eq!(Tab::Overview.next(), Tab::Population);
        assert_eq!(Tab::Events.next(), Tab::Overview);
        assert_eq!(Tab::Overview.previous(), Tab::Events);
        assert_eq!(Tab::Overview.index(), 0);
        assert_eq!(Tab::Events.index(), 4);
    }

    #[test]
    fn the_help_overlay_opens_and_does_not_quit_by_accident() {
        let dir = std::env::temp_dir().join(format!("fh-tui-help-{}", crate::trace::now_ms()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let mut app = App::new(TaskPaths::new(&dir));

        assert!(!app.show_help);
        app.on_key(KeyEvent::new(KeyCode::Char('?'), KeyModifiers::NONE));
        assert!(app.show_help);

        // While it is open, keys that would normally move or quit are
        // swallowed — except `q`, which still means quit.
        app.on_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.tab, Tab::Overview, "the pane must not change");
        app.on_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(!app.show_help, "Esc closes the help");
        assert!(!app.should_quit(), "Esc must not also quit");

        app.on_key(KeyEvent::new(KeyCode::F(1), KeyModifiers::NONE));
        assert!(app.show_help);
        app.on_key(KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE));
        assert!(app.should_quit());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_help_overlay_renders_over_every_pane_at_any_size() {
        use ratatui::backend::TestBackend;

        let dir = std::env::temp_dir().join(format!("fh-tui-helpdraw-{}", crate::trace::now_ms()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let mut app = App::new(TaskPaths::new(&dir));
        app.show_help = true;

        for (width, height) in [(140, 44), (100, 30), (40, 12)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("terminal");
            for tab in Tab::ALL {
                app.tab = tab;
                terminal
                    .draw(|frame| app.draw(frame))
                    .unwrap_or_else(|error| panic!("{tab:?} at {width}x{height}: {error}"));
            }
        }

        // The overlay names the keys, and marks the pane you are on.
        let mut terminal = Terminal::new(TestBackend::new(100, 40)).expect("terminal");
        app.tab = Tab::Structure;
        terminal.draw(|frame| app.draw(frame)).expect("renders");
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(text.contains("zoom in / out"), "keys missing");
        assert!(text.contains("this pane"), "the active group is not marked");
        // Nothing in the overlay may be clipped by its own border: every
        // entry has to fit the inner width.
        assert!(
            text.contains("follow the best candidate"),
            "a description was cut off"
        );
        assert!(
            text.contains("Reopen this view with `fh board`."),
            "the closing note was cut off"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_empty_task_directory_does_not_panic() {
        let dir = std::env::temp_dir().join(format!("fh-tui-empty-{}", crate::trace::now_ms()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let mut app = App::new(TaskPaths::new(&dir));
        app.refresh();
        assert!(app.members().is_empty());
        assert!(app.selected().is_none());
        // Keys on an empty run must be harmless.
        app.on_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
        app.on_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.tab, Tab::Population);
        app.on_key(KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE));
        assert!(app.should_quit());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Produce a real finished run, then render every pane into an off-screen
    /// buffer. Layout and indexing panics are the usual way a dashboard breaks,
    /// and they only appear when something is actually drawn.
    #[tokio::test]
    async fn every_pane_renders_against_a_finished_run() {
        use ratatui::backend::TestBackend;
        use std::sync::Arc;

        const YAML: &str = r#"
name: tui-render-test
binder:
  topology: vhh
  chains:
    - label: H
      sequence: QVQLVESGGGLVQAGGSLRLSCAASGYTFTSYAMGWFRQAPGKEREFVAAISWSGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAGSYYDYWGQGTQVTVSS
  regions:
    - name: loop1
      chain: H
      start: 26
      end: 33
  fixed_residues: ["H:30"]
target:
  chains:
    - label: A
      sequence: MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG
design:
  cycles: 5
  batch_size: 4
  seed: 11
gates:
  min_iptm: 0.2
  min_interface_plddt: 40.0
  max_pae_interface: 30.0
  max_clashes: 2
  min_interface_contacts: 5
  max_hydrophobic_patch: 0.9
fold:
  model: fixture
  execution_mode: fixture
"#;

        let dir = std::env::temp_dir().join(format!("fh-tui-render-{}", crate::trace::now_ms()));
        std::fs::create_dir_all(&dir).expect("temp dir");

        let config = TaskConfig::from_yaml(YAML).expect("config parses");
        let (scaffold, scope) = config.resolve().expect("resolves");
        let seed = config.design.seed;
        let mut orchestrator = crate::orchestrator::Orchestrator::prepare(
            config,
            Arc::new(crate::fold::FixtureBackend::planted(
                &scaffold, &scope, seed,
            )),
            Box::new(crate::agent::HeuristicAgent::new(seed)),
            TaskPaths::new(&dir),
            crate::TaskId("render".to_string()),
        )
        .await
        .expect("prepares");
        orchestrator.run().await.expect("runs");

        let mut app = App::new(TaskPaths::new(&dir));
        app.refresh();
        assert!(!app.members().is_empty(), "nothing to draw");
        assert!(app.selected().is_some());

        // A wide terminal, then a cramped one: the constraints have to hold at
        // both, and a narrow pane is where subtraction underflows.
        for (width, height) in [(140, 44), (100, 30), (40, 12)] {
            let mut terminal =
                Terminal::new(TestBackend::new(width, height)).expect("test terminal");
            for tab in Tab::ALL {
                app.tab = tab;
                terminal
                    .draw(|frame| app.draw(frame))
                    .unwrap_or_else(|error| panic!("{tab:?} failed at {width}x{height}: {error}"));
            }
        }

        // The overview actually shows the run, rather than rendering blank.
        let mut terminal = Terminal::new(TestBackend::new(140, 44)).expect("test terminal");
        app.tab = Tab::Overview;
        terminal.draw(|frame| app.draw(frame)).expect("renders");
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(text.contains("tui-render-test"), "task name missing");
        assert!(text.contains("fixture"), "backend missing");

        // The structure pane draws the stored backbone for the selection.
        app.tab = Tab::Structure;
        terminal.draw(|frame| app.draw(frame)).expect("renders");
        let structure: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(
            structure.contains("backbone") || structure.contains("no stored backbone"),
            "structure pane drew neither a backbone nor an explanation"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
