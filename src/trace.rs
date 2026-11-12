//! What a run leaves behind.
//!
//! Three things, all on disk under one task directory:
//!
//! * `trace.jsonl` — an append-only event log, one JSON object per line, in the
//!   order things happened. Written with a flush per line so a terminal UI can
//!   tail a live run.
//! * `population.json` / `best.json` / `structures/` — snapshots of current
//!   state, rewritten atomically so a reader never sees half a file.
//! * `run.json` — the manifest: seed, backend, protocol version, config digest,
//!   service version. Enough to say what produced a result, which is the part
//!   most pipelines leave implicit.

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::contract::CaResidue;
use crate::{Candidate, CandidateId, CycleRecord, PopulationStats};

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Everything that happens in a run, in the order it happened.
///
/// Skill names and rejection reasons are carried as strings rather than enums
/// so an old trace stays readable after the vocabulary changes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    TaskStarted {
        task_id: String,
        name: String,
        backend: String,
        model: String,
        protocol: String,
        cycles: u32,
        seed: u64,
        scope: String,
        gates: String,
    },
    TaskResumed {
        /// The cycle the recorded run completed. This run continues from the
        /// next one.
        cycle: u32,
        /// Total cycles planned now, counting the completed ones.
        cycles_planned: u32,
        seed: u64,
        /// Always true on a resume: the position in the generator's stream is
        /// not recorded, so the generator is seeded anew from `seed` and the
        /// choices after this point diverge from one uninterrupted run.
        generator_reseeded: bool,
    },
    FounderScored {
        candidate: CandidateId,
        loss: f64,
    },
    CycleStarted {
        cycle: u32,
        parent: CandidateId,
        population: usize,
        best_loss: Option<f64>,
    },
    SkillRouted {
        cycle: u32,
        legal: Vec<String>,
        /// `skill: reason` for each skill the router refused.
        blocked: Vec<String>,
    },
    SkillChosen {
        cycle: u32,
        skill: String,
        agent: String,
        decided_by: String,
        rationale: String,
    },
    ProposalsGenerated {
        cycle: u32,
        skill: String,
        count: usize,
    },
    FoldStarted {
        cycle: u32,
        candidate: CandidateId,
    },
    FoldFinished {
        cycle: u32,
        candidate: CandidateId,
        elapsed_ms: u64,
    },
    FoldFailed {
        cycle: u32,
        candidate: CandidateId,
        detail: String,
    },
    Scored {
        cycle: u32,
        candidate: CandidateId,
        loss: f64,
        iptm: f64,
        interface_plddt: f64,
    },
    Rejected {
        cycle: u32,
        candidate: CandidateId,
        kind: String,
        reason: String,
    },
    Admitted {
        cycle: u32,
        candidate: CandidateId,
        parent: Option<CandidateId>,
        loss: f64,
        outcome: String,
        substitutions: String,
    },
    CycleFinished {
        cycle: u32,
        admitted: usize,
        rejected: usize,
        best_loss: Option<f64>,
        mean_loss: Option<f64>,
        improved: bool,
    },
    Stagnation {
        cycle: u32,
        cycles_without_improvement: u32,
    },
    Note {
        cycle: u32,
        source: String,
        text: String,
    },
    TaskFinished {
        cycles_completed: u32,
        folds: usize,
        best: Option<CandidateId>,
        best_loss: Option<f64>,
        elapsed_ms: u64,
    },
    TaskFailed {
        cycle: u32,
        detail: String,
    },
}

impl Event {
    /// Short label for the event stream pane.
    pub fn kind(&self) -> &'static str {
        match self {
            Event::TaskStarted { .. } => "task_started",
            Event::TaskResumed { .. } => "task_resumed",
            Event::FounderScored { .. } => "founder_scored",
            Event::CycleStarted { .. } => "cycle_started",
            Event::SkillRouted { .. } => "skill_routed",
            Event::SkillChosen { .. } => "skill_chosen",
            Event::ProposalsGenerated { .. } => "proposals",
            Event::FoldStarted { .. } => "fold_started",
            Event::FoldFinished { .. } => "fold_finished",
            Event::FoldFailed { .. } => "fold_failed",
            Event::Scored { .. } => "scored",
            Event::Rejected { .. } => "rejected",
            Event::Admitted { .. } => "admitted",
            Event::CycleFinished { .. } => "cycle_finished",
            Event::Stagnation { .. } => "stagnation",
            Event::Note { .. } => "note",
            Event::TaskFinished { .. } => "task_finished",
            Event::TaskFailed { .. } => "task_failed",
        }
    }

    pub fn cycle(&self) -> Option<u32> {
        match self {
            Event::CycleStarted { cycle, .. }
            | Event::SkillRouted { cycle, .. }
            | Event::SkillChosen { cycle, .. }
            | Event::ProposalsGenerated { cycle, .. }
            | Event::FoldStarted { cycle, .. }
            | Event::FoldFinished { cycle, .. }
            | Event::FoldFailed { cycle, .. }
            | Event::Scored { cycle, .. }
            | Event::Rejected { cycle, .. }
            | Event::Admitted { cycle, .. }
            | Event::CycleFinished { cycle, .. }
            | Event::Stagnation { cycle, .. }
            | Event::Note { cycle, .. }
            | Event::TaskResumed { cycle, .. }
            | Event::TaskFailed { cycle, .. } => Some(*cycle),
            _ => None,
        }
    }

    /// The candidate an event names, when it names one. A resume reads this
    /// back to find the highest id the recorded run ever assigned.
    pub fn candidate(&self) -> Option<CandidateId> {
        match self {
            Event::FounderScored { candidate, .. }
            | Event::FoldStarted { candidate, .. }
            | Event::FoldFinished { candidate, .. }
            | Event::FoldFailed { candidate, .. }
            | Event::Scored { candidate, .. }
            | Event::Rejected { candidate, .. }
            | Event::Admitted { candidate, .. } => Some(*candidate),
            _ => None,
        }
    }

    /// One line for the event pane and `fh run` stdout.
    pub fn summary(&self) -> String {
        match self {
            Event::TaskStarted {
                name,
                backend,
                model,
                cycles,
                ..
            } => format!("task '{name}' on {backend} ({model}), {cycles} cycles"),
            Event::TaskResumed {
                cycle,
                cycles_planned,
                seed,
                ..
            } => format!(
                "resumed after cycle {cycle} of {cycles_planned}, seed {seed}; the generator \
                 was re-seeded, so choices from here diverge from one uninterrupted run"
            ),
            Event::FounderScored { candidate, loss } => {
                format!("founder {candidate} scored, loss {loss:.4}")
            }
            Event::CycleStarted {
                cycle,
                parent,
                population,
                ..
            } => format!("cycle {cycle}: parent {parent}, population {population}"),
            Event::SkillRouted { legal, blocked, .. } => format!(
                "router: legal [{}]{}",
                legal.join(", "),
                if blocked.is_empty() {
                    String::new()
                } else {
                    format!(", blocked {}", blocked.len())
                }
            ),
            Event::SkillChosen {
                skill, rationale, ..
            } => format!("chose {skill}: {rationale}"),
            Event::ProposalsGenerated { count, skill, .. } => {
                format!("{skill} produced {count} proposal(s)")
            }
            Event::FoldStarted { candidate, .. } => format!("folding {candidate}"),
            Event::FoldFinished {
                candidate,
                elapsed_ms,
                ..
            } => format!("folded {candidate} in {elapsed_ms}ms"),
            Event::FoldFailed {
                candidate, detail, ..
            } => format!("fold failed for {candidate}: {detail}"),
            Event::Scored {
                candidate,
                loss,
                iptm,
                ..
            } => format!("{candidate} loss {loss:.4} (ipTM {iptm:.3})"),
            Event::Rejected {
                candidate, reason, ..
            } => format!("rejected {candidate}: {reason}"),
            Event::Admitted {
                candidate,
                loss,
                substitutions,
                ..
            } => format!("admitted {candidate} loss {loss:.4} [{substitutions}]"),
            Event::CycleFinished {
                cycle,
                admitted,
                rejected,
                best_loss,
                improved,
                ..
            } => format!(
                "cycle {cycle} done: +{admitted} / -{rejected}, best {}{}",
                best_loss
                    .map(|l| format!("{l:.4}"))
                    .unwrap_or_else(|| "none".into()),
                if *improved { " (improved)" } else { "" }
            ),
            Event::Stagnation {
                cycles_without_improvement,
                ..
            } => format!("stagnating: {cycles_without_improvement} cycles without improvement"),
            Event::Note { source, text, .. } => format!("{source}: {text}"),
            Event::TaskFinished {
                cycles_completed,
                folds,
                best_loss,
                elapsed_ms,
                ..
            } => format!(
                "finished {cycles_completed} cycles, {folds} folds, best {}, {:.1}s",
                best_loss
                    .map(|l| format!("{l:.4}"))
                    .unwrap_or_else(|| "none".into()),
                *elapsed_ms as f64 / 1000.0
            ),
            Event::TaskFailed { detail, .. } => format!("task failed: {detail}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceRecord {
    pub seq: u64,
    pub at_ms: u64,
    #[serde(flatten)]
    pub event: Event,
}

/// Append-only JSONL writer, flushed per record so a reader tailing the file
/// sees events as they happen.
pub struct TraceWriter {
    file: File,
    path: PathBuf,
    seq: u64,
}

impl TraceWriter {
    /// Start a fresh trace, truncating any existing file.
    pub fn create(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)?;
        Ok(TraceWriter { file, path, seq: 0 })
    }

    /// Continue an existing trace, numbering after the last readable record.
    pub fn append(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let seq =
            read_all(&path).map(|records| records.last().map(|record| record.seq).unwrap_or(0))?;
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(TraceWriter { file, path, seq })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn write(&mut self, event: Event) -> io::Result<TraceRecord> {
        self.seq += 1;
        let record = TraceRecord {
            seq: self.seq,
            at_ms: now_ms(),
            event,
        };
        let line = serde_json::to_string(&record)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        writeln!(self.file, "{line}")?;
        self.file.flush()?;
        Ok(record)
    }
}

/// Read a trace. Unparsable lines are skipped rather than failing the read:
/// the last line of a live trace can be half-written when a reader gets to it.
pub fn read_all(path: impl AsRef<Path>) -> io::Result<Vec<TraceRecord>> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(Vec::new());
    }
    let reader = BufReader::new(File::open(path)?);
    Ok(reader
        .lines()
        .map_while(Result::ok)
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str::<TraceRecord>(&line).ok())
        .collect())
}

/// Records after `seq`, for incremental tailing.
pub fn read_after(path: impl AsRef<Path>, seq: u64) -> io::Result<Vec<TraceRecord>> {
    Ok(read_all(path)?
        .into_iter()
        .filter(|record| record.seq > seq)
        .collect())
}

/// Snapshot of the retained population, rewritten each cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationSnapshot {
    pub task_id: String,
    pub name: String,
    pub cycle: u32,
    pub cycles_planned: u32,
    pub updated_at_ms: u64,
    pub scope: String,
    pub stats: PopulationStats,
    pub members: Vec<Candidate>,
    pub loss_curve: Vec<f64>,
    pub records: Vec<CycleRecord>,
    pub folds: usize,
    pub running: bool,
    /// Admission counters of the retained population, so a resumed run
    /// restores what the recorded run reported instead of starting at zero.
    #[serde(default)]
    pub admitted_total: usize,
    #[serde(default)]
    pub rejected_total: usize,
}

/// Alpha-carbon trace for one candidate, so the viewer can draw it without
/// parsing a structure file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureSnapshot {
    pub candidate: CandidateId,
    pub cycle: u32,
    pub loss: Option<f64>,
    pub ca_trace: Vec<CaResidue>,
}

/// How a run was produced. The part that makes a result citable rather than
/// merely reported.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    pub task_id: String,
    pub name: String,
    pub harness_version: String,
    pub started_at_ms: u64,
    pub finished_at_ms: Option<u64>,
    pub seed: u64,
    pub backend: String,
    pub model: String,
    pub protocol: String,
    pub service_version: Option<String>,
    /// SHA-256 of the resolved task config, so a rerun can be shown to have
    /// used the same inputs.
    pub config_sha256: String,
    pub cycles_planned: u32,
    pub cycles_completed: u32,
    pub folds: usize,
    pub best_candidate: Option<CandidateId>,
    pub best_loss: Option<f64>,
    /// The chat this run was started from, when it was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_id: Option<String>,
    /// What the run spent: folds, backend time, and model tokens if any.
    /// Across a resume this is the sum of both runs' spend.
    #[serde(default)]
    pub usage: crate::usage::Usage,
    /// The cycle this run resumed from, when it resumed. The run continued
    /// from `resumed_from_cycle + 1`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resumed_from_cycle: Option<u32>,
    /// True when the run resumed: the position in the generator's stream is
    /// not recorded, so the generator was seeded anew from `seed` and the
    /// choices after the resume diverge from one uninterrupted run.
    #[serde(default)]
    pub generator_reseeded: bool,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Write JSON atomically: a reader either sees the previous file or the new
/// one, never a partial write.
pub fn write_json<T: Serialize>(path: impl AsRef<Path>, value: &T) -> io::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.tmp");
    let text = serde_json::to_string_pretty(value)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    fs::write(&temporary, text)?;
    fs::rename(&temporary, path)
}

pub fn read_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> io::Result<T> {
    let text = fs::read_to_string(path)?;
    serde_json::from_str(&text).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Layout of one task directory.
#[derive(Debug, Clone)]
pub struct TaskPaths {
    root: PathBuf,
}

impl TaskPaths {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        TaskPaths { root: root.into() }
    }

    /// `$FH_TASK_ROOT`, or `~/.folding_harness/tasks`.
    pub fn default_root() -> PathBuf {
        if let Ok(custom) = std::env::var("FH_TASK_ROOT") {
            return PathBuf::from(custom);
        }
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join(".folding_harness").join("tasks")
    }

    pub fn under_default_root(task_id: &str) -> Self {
        TaskPaths::new(Self::default_root().join(task_id))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn ensure(&self) -> io::Result<()> {
        fs::create_dir_all(self.structures())
    }

    pub fn trace(&self) -> PathBuf {
        self.root.join("trace.jsonl")
    }

    pub fn config(&self) -> PathBuf {
        self.root.join("task.yaml")
    }

    pub fn manifest(&self) -> PathBuf {
        self.root.join("run.json")
    }

    pub fn population(&self) -> PathBuf {
        self.root.join("population.json")
    }

    pub fn best(&self) -> PathBuf {
        self.root.join("best.json")
    }

    pub fn structures(&self) -> PathBuf {
        self.root.join("structures")
    }

    pub fn structure(&self, candidate: CandidateId) -> PathBuf {
        self.structures().join(format!("{candidate}.json"))
    }

    /// Task directories under a root, newest first by directory name.
    pub fn list(root: impl AsRef<Path>) -> io::Result<Vec<TaskPaths>> {
        let root = root.as_ref();
        if !root.exists() {
            return Ok(Vec::new());
        }
        let mut found: Vec<PathBuf> = fs::read_dir(root)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect();
        found.sort();
        found.reverse();
        Ok(found.into_iter().map(TaskPaths::new).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fh-trace-test-{name}-{}", now_ms()));
        fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    #[test]
    fn records_are_numbered_and_read_back_in_order() {
        let dir = temp_dir("order");
        let path = dir.join("trace.jsonl");
        let mut writer = TraceWriter::create(&path).expect("writer");

        writer
            .write(Event::Note {
                cycle: 1,
                source: "test".into(),
                text: "first".into(),
            })
            .expect("write");
        writer
            .write(Event::Note {
                cycle: 2,
                source: "test".into(),
                text: "second".into(),
            })
            .expect("write");

        let records = read_all(&path).expect("read");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].seq, 1);
        assert_eq!(records[1].seq, 2);
        assert_eq!(records[1].event.cycle(), Some(2));

        let tail = read_after(&path, 1).expect("tail");
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0].seq, 2);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_half_written_line_does_not_break_a_reader() {
        let dir = temp_dir("partial");
        let path = dir.join("trace.jsonl");
        let mut writer = TraceWriter::create(&path).expect("writer");
        writer
            .write(Event::Stagnation {
                cycle: 3,
                cycles_without_improvement: 3,
            })
            .expect("write");

        // Simulate a reader arriving mid-write.
        let mut file = OpenOptions::new().append(true).open(&path).expect("open");
        write!(file, "{{\"seq\":2,\"at_ms\":1,\"event\":\"no").expect("partial");
        file.flush().expect("flush");

        let records = read_all(&path).expect("read");
        assert_eq!(records.len(), 1);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn appending_continues_the_sequence() {
        let dir = temp_dir("append");
        let path = dir.join("trace.jsonl");
        {
            let mut writer = TraceWriter::create(&path).expect("writer");
            writer
                .write(Event::Note {
                    cycle: 1,
                    source: "a".into(),
                    text: "x".into(),
                })
                .expect("write");
        }
        let mut writer = TraceWriter::append(&path).expect("append");
        let record = writer
            .write(Event::Note {
                cycle: 2,
                source: "b".into(),
                text: "y".into(),
            })
            .expect("write");
        assert_eq!(record.seq, 2);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn json_is_written_atomically_and_leaves_no_temp_file() {
        let dir = temp_dir("atomic");
        let path = dir.join("population.json");
        write_json(&path, &vec![1u32, 2, 3]).expect("write");
        let back: Vec<u32> = read_json(&path).expect("read");
        assert_eq!(back, vec![1, 2, 3]);
        assert!(!dir.join("population.json.tmp").exists());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_config_digest_is_stable_and_sensitive() {
        assert_eq!(sha256_hex(b"name: demo"), sha256_hex(b"name: demo"));
        assert_ne!(sha256_hex(b"name: demo"), sha256_hex(b"name: demo2"));
        assert_eq!(sha256_hex(b"").len(), 64);
    }

    /// The resume fields are additive: a `run.json` or `population.json`
    /// written before they existed still reads, with defaults in their place.
    #[test]
    fn artifacts_from_before_the_resume_fields_still_read() {
        let manifest_json = r#"{
            "task_id": "1-demo",
            "name": "demo",
            "harness_version": "0.1.0",
            "started_at_ms": 1,
            "finished_at_ms": null,
            "seed": 7,
            "backend": "fixture",
            "model": "fixture",
            "protocol": "fh.fold/v1",
            "service_version": null,
            "config_sha256": "00",
            "cycles_planned": 4,
            "cycles_completed": 4,
            "folds": 9,
            "best_candidate": null,
            "best_loss": null
        }"#;
        let manifest: RunManifest =
            serde_json::from_str(manifest_json).expect("old manifest reads");
        assert_eq!(manifest.resumed_from_cycle, None);
        assert!(!manifest.generator_reseeded);
        assert!(manifest.usage.is_empty());
        assert!(manifest.chat_id.is_none());

        let snapshot_json = r#"{
            "task_id": "1-demo",
            "name": "demo",
            "cycle": 4,
            "cycles_planned": 4,
            "updated_at_ms": 1,
            "scope": "8 positions",
            "stats": {"size": 0, "best_loss": null, "mean_loss": null, "worst_loss": null},
            "members": [],
            "loss_curve": [],
            "records": [],
            "folds": 9,
            "running": false
        }"#;
        let snapshot: PopulationSnapshot =
            serde_json::from_str(snapshot_json).expect("old snapshot reads");
        assert_eq!(snapshot.admitted_total, 0);
        assert_eq!(snapshot.rejected_total, 0);
    }

    #[test]
    fn a_resumed_event_names_the_candidate_of_nothing_and_keeps_its_cycle() {
        let event = Event::TaskResumed {
            cycle: 3,
            cycles_planned: 8,
            seed: 7,
            generator_reseeded: true,
        };
        assert_eq!(event.kind(), "task_resumed");
        assert_eq!(event.cycle(), Some(3));
        assert_eq!(event.candidate(), None);
        assert!(event.summary().contains("re-seeded"));

        let scored = Event::Scored {
            cycle: 4,
            candidate: CandidateId(12),
            loss: 0.5,
            iptm: 0.7,
            interface_plddt: 80.0,
        };
        assert_eq!(scored.candidate(), Some(CandidateId(12)));
    }
}
