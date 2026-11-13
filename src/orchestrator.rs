//! The design cycle.
//!
//! One cycle, in order:
//!
//! 1. select a parent from the retained population,
//! 2. ask the router which skills are legal in the current search state,
//! 3. ask the agent to choose one of them,
//! 4. run that skill's executor to get validated proposals,
//! 5. fold the proposed candidates,
//! 6. apply hard gates,
//! 7. offer survivors to the population,
//! 8. record everything and snapshot state to disk.
//!
//! Steps 1–4 and 6–8 are deterministic given the task seed. Step 5 is the only
//! one that leaves the process. That separation is what makes a run replayable:
//! the same config, seed and backend produce the same trace.
//!
//! A resumed run is the one exception. The manifest records the seed but not
//! the position in the generator's stream, so [`Orchestrator::resume`] seeds a
//! fresh generator and the resumed run's parent selection and proposals
//! diverge from what one uninterrupted run would have produced. The trace and
//! the manifest both record the resume and the re-seed rather than hiding it.

use std::sync::Arc;
use std::time::Instant;

use crate::RunRng;

use tokio::task::JoinSet;

use crate::agent::{AgentError, DesignAgent};
use crate::contract::FoldResult;
use crate::fold::{
    job_spec_for, metrics_from, verify_output, FoldBackend, FoldError, Verification,
};
use crate::skills::{executor, DesignSkill, Legality, Router, SkillContext};
use crate::trace::{
    now_ms, Event, PopulationSnapshot, RunManifest, StructureSnapshot, TaskPaths, TraceRecord,
    TraceWriter,
};
use crate::{
    AdmissionOutcome, BinderSequence, Candidate, CandidateId, ConfigError, CycleOutcome,
    MutableScope, Population, Proposal, ProposalError, RejectReason, Scaffold, SearchHistory,
    SkillId, TaskConfig, TaskId,
};

#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error(transparent)]
    Fold(#[from] FoldError),
    #[error(transparent)]
    Agent(#[from] AgentError),
    #[error(transparent)]
    Proposal(#[from] ProposalError),
    #[error("input/output error: {0}")]
    Io(#[from] std::io::Error),
    #[error("population is empty, so there is no parent to design from")]
    NoParent,
    #[error("cycle {cycle}: the router left no legal skills; check design.skills and service capabilities")]
    NoLegalSkills { cycle: u32 },
    #[error("skill '{skill}' was chosen but has no executor in this build")]
    NoExecutor { skill: String },
    #[error("the founder sequence could not be folded: {0}")]
    FounderFoldFailed(String),
    #[error("there is no founder to design from. {reason}. Check the service against the contract in src/contract.rs")]
    FounderSequenceMismatch { reason: String },
    #[error("cannot read run.json in {dir}: {detail}. `fh resume` continues a recorded run, so point --path at a task directory that `fh run` created")]
    NoManifest {
        dir: std::path::PathBuf,
        detail: String,
    },
    #[error("cannot read population.json in {dir}: {detail}. A resume continues the population the recorded run retained, so run the task again with `fh run`")]
    NoSnapshot {
        dir: std::path::PathBuf,
        detail: String,
    },
    #[error("population.json in {dir} retains no members, so there is no parent to design from; run the task again with `fh run`")]
    EmptySnapshot { dir: std::path::PathBuf },
    #[error("population.json in {dir} holds candidate {candidate}, which the recorded run could not have produced: {reason}. A resume folds the members it restores, so restore population.json from a copy the recorded run wrote, or run the task again with `fh run`")]
    SnapshotMemberInvalid {
        dir: std::path::PathBuf,
        candidate: CandidateId,
        reason: String,
    },
    #[error("the run in {dir} already completed its {completed} planned cycles; pass --cycles with a larger total to extend it")]
    AlreadyComplete {
        dir: std::path::PathBuf,
        completed: u32,
    },
    #[error("--cycles {requested} is below the {completed} cycles already completed in {dir}; pass a larger total to extend the run")]
    CyclesBelowCompleted {
        dir: std::path::PathBuf,
        completed: u32,
        requested: u32,
    },
    #[error("the config does not match the recorded run: {path} digests to {current}, but run.json beside it records {recorded}. A resumed run continues one search, so it needs the config that search used. Resume from the task.yaml saved in the task directory, or start a new task with `fh run`.")]
    ConfigMismatch {
        path: std::path::PathBuf,
        recorded: String,
        current: String,
    },
}

#[derive(Debug, Clone)]
pub struct RunSummary {
    pub task_id: TaskId,
    pub root: std::path::PathBuf,
    pub cycles_completed: u32,
    pub folds: usize,
    pub best: Option<Candidate>,
    pub elapsed_ms: u64,
}

/// Called for every trace record as it is written, so a caller can stream a
/// run without re-reading the file.
pub type EventSink = Box<dyn FnMut(&TraceRecord) + Send>;

pub struct Orchestrator {
    config: TaskConfig,
    scaffold: Scaffold,
    scope: MutableScope,
    population: Population,
    history: SearchHistory,
    backend: Arc<dyn FoldBackend>,
    agent: Box<dyn DesignAgent>,
    router: Router,
    trace: TraceWriter,
    paths: TaskPaths,
    manifest: RunManifest,
    rng: RunRng,
    task_id: TaskId,
    next_id: u32,
    folds: usize,
    meter: crate::usage::Meter,
    /// What the recorded run spent before a resume. Added to the meter's
    /// snapshot so `run.json` reports both runs' spend. A fresh run has no
    /// recorded spend, and that is known, so its base is
    /// [`Usage::no_model_calls`](crate::usage::Usage::no_model_calls) rather
    /// than the unknown a default carries.
    usage_base: crate::usage::Usage,
    /// The first cycle `run` executes. One for a fresh run, one past the
    /// recorded run's last completed cycle for a resume.
    start_cycle: u32,
    /// The cycle a resumed run continues from, for the `TaskResumed` event.
    resumed_from: Option<u32>,
    on_event: Option<EventSink>,
}

impl Orchestrator {
    /// Resolve the config, check the backend, and open the task directory.
    ///
    /// The health check is not ceremony: the router's legal set is derived from
    /// the capabilities the service advertises, so a run that cannot ask the
    /// service what it can do would have to guess.
    pub async fn prepare(
        config: TaskConfig,
        backend: Arc<dyn FoldBackend>,
        agent: Box<dyn DesignAgent>,
        paths: TaskPaths,
        task_id: TaskId,
    ) -> Result<Self, RunError> {
        let (scaffold, scope) = config.resolve()?;
        paths.ensure()?;

        let health = backend.health().await?;
        let router = Router::from_capability_strings(
            config.design.skills.iter().copied(),
            &health.capabilities,
        );

        let config_yaml = config.to_yaml()?;
        std::fs::write(paths.config(), &config_yaml)?;

        let manifest = RunManifest {
            task_id: task_id.0.clone(),
            name: config.name.clone(),
            harness_version: env!("CARGO_PKG_VERSION").to_string(),
            started_at_ms: now_ms(),
            finished_at_ms: None,
            seed: config.design.seed,
            backend: backend.name().to_string(),
            model: config.fold.model.clone(),
            protocol: health.protocol.clone(),
            service_version: Some(health.version.clone()),
            config_sha256: crate::trace::sha256_hex(config_yaml.as_bytes()),
            cycles_planned: config.design.cycles,
            cycles_completed: 0,
            folds: 0,
            best_candidate: None,
            best_loss: None,
            chat_id: None,
            usage: crate::usage::Usage::default(),
            resumed_from_cycle: None,
            generator_reseeded: false,
            resumed_protocol: None,
            resumed_service_version: None,
        };
        crate::trace::write_json(paths.manifest(), &manifest)?;

        let trace = TraceWriter::create(paths.trace())?;
        let population = Population::new(config.population.clone());
        let rng = crate::rng_from_seed(config.design.seed);

        Ok(Orchestrator {
            config,
            scaffold,
            scope,
            population,
            history: SearchHistory::new(),
            backend,
            agent,
            router,
            trace,
            paths,
            manifest,
            rng,
            task_id,
            next_id: 0,
            folds: 0,
            meter: crate::usage::Meter::new(),
            usage_base: crate::usage::Usage::no_model_calls(),
            start_cycle: 1,
            resumed_from: None,
            on_event: None,
        })
    }

    /// Rebuild the search state from a task directory a recorded run left, so
    /// [`run`](Self::run) continues from the last completed cycle.
    ///
    /// The config must digest to the `config_sha256` the manifest recorded, or
    /// two searches would mix into one population and one lineage. `cycles_total`
    /// extends the planned total past what the config says. The digest is
    /// checked before it applies, so extending a run is not a config change.
    ///
    /// The generator cannot be restored: the manifest records the seed but not
    /// the position in its stream. This re-seeds from `design.seed`, and the
    /// manifest, the trace and `run.json` say so.
    pub async fn resume(
        mut config: TaskConfig,
        backend: Arc<dyn FoldBackend>,
        agent: Box<dyn DesignAgent>,
        paths: TaskPaths,
        cycles_total: Option<u32>,
    ) -> Result<Self, RunError> {
        let dir = paths.root().to_path_buf();
        let mut manifest: RunManifest =
            crate::trace::read_json(paths.manifest()).map_err(|error| RunError::NoManifest {
                dir: dir.clone(),
                detail: error.to_string(),
            })?;
        let snapshot: PopulationSnapshot =
            crate::trace::read_json(paths.population()).map_err(|error| RunError::NoSnapshot {
                dir: dir.clone(),
                detail: error.to_string(),
            })?;

        let config_yaml = config.to_yaml()?;
        let digest = crate::trace::sha256_hex(config_yaml.as_bytes());
        if digest != manifest.config_sha256 {
            return Err(RunError::ConfigMismatch {
                path: paths.config(),
                recorded: manifest.config_sha256.clone(),
                current: digest,
            });
        }

        // population.json is written after every completed cycle, so its
        // `cycle` is the completed count even when the run stopped before it
        // could write a final run.json.
        let completed = snapshot.cycle;
        match cycles_total {
            Some(total) if total < completed => {
                return Err(RunError::CyclesBelowCompleted {
                    dir,
                    completed,
                    requested: total,
                })
            }
            Some(total) => config.design.cycles = total,
            // A resume that was killed leaves a manifest whose
            // `cycles_planned` is larger than the config's own count, so a
            // bare resume continues that plan rather than report the run
            // complete against a total the manifest does not carry.
            None => config.design.cycles = config.design.cycles.max(manifest.cycles_planned),
        }
        if completed >= config.design.cycles {
            return Err(RunError::AlreadyComplete { dir, completed });
        }
        if snapshot.members.is_empty() {
            return Err(RunError::EmptySnapshot { dir });
        }

        let (scaffold, scope) = config.resolve()?;

        // A resume makes the snapshot's members live: they are selected as
        // parents, and `job_spec_for` folds their chains. `Candidate` and
        // `BinderSequence` both deserialize, so a hand-edited or truncated
        // population.json parses, and without this check it could smuggle a
        // sequence past the invariant that `propose` and `from_proposal`
        // otherwise carry. Every member the recorded run retained was folded,
        // scored and traced, so each one is checked against the config that
        // produced it and the trace that recorded it, and the whole resume is
        // refused on the first defect.
        let records = crate::trace::read_all(paths.trace())?;
        let scored_losses: std::collections::HashMap<CandidateId, f64> = records
            .iter()
            .filter_map(|record| match &record.event {
                Event::Scored {
                    candidate, loss, ..
                }
                | Event::FounderScored { candidate, loss } => Some((*candidate, *loss)),
                _ => None,
            })
            .collect();
        for member in &snapshot.members {
            if let Some(reason) =
                restored_member_defect(member, &scaffold, &scope, &config.loss, &scored_losses)
            {
                return Err(RunError::SnapshotMemberInvalid {
                    dir,
                    candidate: member.id,
                    reason,
                });
            }
        }

        let health = backend.health().await?;
        let router = Router::from_capability_strings(
            config.design.skills.iter().copied(),
            &health.capabilities,
        );

        let (admitted, rejected) = if snapshot.admitted_total > 0 || snapshot.rejected_total > 0 {
            (snapshot.admitted_total, snapshot.rejected_total)
        } else {
            // A snapshot written before the counters were recorded carries
            // only the cycle records. The founder is the extra admission.
            // The rebuilt `rejected` also counts gate failures and fold
            // failures, which the population's own counter never saw, so it
            // is higher than the recorded run reported.
            (
                1 + snapshot.records.iter().map(|r| r.admitted).sum::<usize>(),
                snapshot.records.iter().map(|r| r.rejected).sum(),
            )
        };
        let population = Population::restore(
            config.population.clone(),
            snapshot.members.clone(),
            admitted,
            rejected,
        );
        let history = SearchHistory::restore(snapshot.records.clone());

        // Ids continue after the highest one the recorded run ever assigned,
        // not just the retained ones: a collision would silently overwrite a
        // candidate and corrupt the lineage. The trace names every candidate
        // it folded, admitted or rejected. `read_all` skips a half-written
        // last line, but `fold_all` writes `FoldStarted` for a whole batch
        // before any later event, so the highest id stays readable.
        let next_id = records
            .iter()
            .filter_map(|record| record.event.candidate())
            .map(|id| id.0)
            .chain(snapshot.members.iter().map(|member| member.id.0))
            .max()
            .unwrap_or(0);

        let trace = TraceWriter::append(paths.trace())?;

        let usage_base = manifest.usage;
        manifest.harness_version = env!("CARGO_PKG_VERSION").to_string();
        manifest.finished_at_ms = None;
        // `protocol` and `service_version` keep describing the run that
        // produced the recorded artifacts: `folds`, `usage` and
        // `best_candidate` span both runs, so overwriting them would name,
        // for the whole result, a service that produced only part of it. The
        // resumed run's own values go in the resume fields.
        manifest.resumed_protocol = Some(health.protocol.clone());
        manifest.resumed_service_version = Some(health.version.clone());
        manifest.cycles_planned = config.design.cycles;
        manifest.cycles_completed = completed;
        manifest.folds = snapshot.folds;
        manifest.resumed_from_cycle = Some(completed);
        manifest.generator_reseeded = true;

        paths.ensure()?;
        crate::trace::write_json(paths.manifest(), &manifest)?;

        let rng = crate::rng_from_seed(config.design.seed);
        let task_id = TaskId(manifest.task_id.clone());
        Ok(Orchestrator {
            config,
            scaffold,
            scope,
            population,
            history,
            backend,
            agent,
            router,
            trace,
            paths,
            manifest,
            rng,
            task_id,
            next_id,
            folds: snapshot.folds,
            meter: crate::usage::Meter::new(),
            usage_base,
            start_cycle: completed + 1,
            resumed_from: Some(completed),
            on_event: None,
        })
    }

    /// Record which conversation started this run, so the manifest says where
    /// the plan came from. Rewrites `run.json` straight away, so the link
    /// survives even if the run does not finish.
    pub fn attributed_to(mut self, chat_id: Option<String>) -> Result<Self, RunError> {
        if chat_id.is_some() {
            self.manifest.chat_id = chat_id;
            self.save_manifest()?;
        }
        Ok(self)
    }

    pub fn on_event(mut self, sink: EventSink) -> Self {
        self.on_event = Some(sink);
        self
    }

    pub fn scope(&self) -> &MutableScope {
        &self.scope
    }

    pub fn population(&self) -> &Population {
        &self.population
    }

    /// Write `run.json`. Called whenever a field of the manifest changes, so a
    /// run that stops early still leaves an accurate one on disk.
    fn save_manifest(&self) -> Result<(), RunError> {
        crate::trace::write_json(self.paths.manifest(), &self.manifest)?;
        Ok(())
    }

    /// Persist the counters a later resume reads back: folds and the spend of
    /// this run added to what the recorded run spent. The meter holds the fold
    /// counts and the agent reports the model spend, so a resumed run sums
    /// three sources rather than two.
    fn record_progress(&mut self) -> Result<(), RunError> {
        self.manifest.folds = self.folds;
        self.manifest.usage = self
            .usage_base
            .summed_with(self.meter.snapshot())
            .summed_with(self.agent.usage());
        self.save_manifest()
    }

    fn emit(&mut self, event: Event) -> Result<(), RunError> {
        let record = self.trace.write(event)?;
        if let Some(sink) = self.on_event.as_mut() {
            sink(&record);
        }
        Ok(())
    }

    fn next_candidate_id(&mut self) -> CandidateId {
        self.next_id += 1;
        CandidateId(self.next_id)
    }

    /// The completion step both arms share: emit `TaskFinished`, mark the run
    /// finished in the manifest, name the best candidate the population
    /// holds, persist the counters through [`Self::record_progress`], and
    /// write the final snapshot with `running` cleared.
    ///
    /// `cycles_completed` is what the event and the snapshot report: `run`
    /// passes the manifest's count, the founder-only arm passes zero.
    /// Returns the best candidate and the elapsed time for the caller's
    /// [`RunSummary`].
    fn finish(
        &mut self,
        cycles_completed: u32,
        started: Instant,
    ) -> Result<(Option<Candidate>, u64), RunError> {
        let best = self.population.best().cloned();
        let elapsed_ms = started.elapsed().as_millis() as u64;

        self.emit(Event::TaskFinished {
            cycles_completed,
            folds: self.folds,
            best: best.as_ref().map(|candidate| candidate.id),
            best_loss: best.as_ref().and_then(|candidate| candidate.loss),
            elapsed_ms,
        })?;

        self.manifest.finished_at_ms = Some(now_ms());
        self.manifest.best_candidate = best.as_ref().map(|candidate| candidate.id);
        self.manifest.best_loss = best.as_ref().and_then(|candidate| candidate.loss);
        self.record_progress()?;
        self.snapshot(cycles_completed, false)?;

        Ok((best, elapsed_ms))
    }

    /// Fold the input sequence and stop, without searching.
    ///
    /// The control arm of a benchmark: it shows what the starting sequence
    /// scores, so a search can be compared against doing nothing.
    pub async fn seed_founder_only(&mut self) -> Result<RunSummary, RunError> {
        let started = Instant::now();
        self.announce()?;
        self.seed_founder().await?;

        // The arm closes its manifest through the same completion step `run`
        // uses, so a benchmark's floor arm leaves a task directory that reads
        // as finished rather than one `fh ls` and the web view report as
        // still running forever.
        let (best, elapsed_ms) = self.finish(0, started)?;

        Ok(RunSummary {
            task_id: self.task_id.clone(),
            root: self.paths.root().to_path_buf(),
            cycles_completed: 0,
            folds: self.folds,
            best,
            elapsed_ms,
        })
    }

    fn announce(&mut self) -> Result<(), RunError> {
        // The event describes the service this run talks to. On a resume that
        // is the resumed run's checked protocol, not the recorded run's.
        let protocol = self
            .manifest
            .resumed_protocol
            .clone()
            .unwrap_or_else(|| self.manifest.protocol.clone());
        self.emit(Event::TaskStarted {
            task_id: self.task_id.0.clone(),
            name: self.config.name.clone(),
            backend: self.backend.name().to_string(),
            model: self.config.fold.model.clone(),
            protocol,
            cycles: self.config.design.cycles,
            seed: self.config.design.seed,
            scope: self.scope.summary(),
            gates: format!("{:?}", self.config.gates),
        })
    }

    pub async fn run(&mut self) -> Result<RunSummary, RunError> {
        let started = Instant::now();
        self.announce()?;

        match self.resumed_from {
            // A resumed run keeps the population and the history the recorded
            // run left, so it does not seed a founder again.
            Some(cycle) => {
                self.emit(Event::TaskResumed {
                    cycle,
                    cycles_planned: self.config.design.cycles,
                    seed: self.config.design.seed,
                    generator_reseeded: true,
                })?;
            }
            None => match self.seed_founder().await {
                Ok(()) => {}
                Err(error) => {
                    self.emit(Event::TaskFailed {
                        cycle: 0,
                        detail: error.to_string(),
                    })?;
                    self.record_progress()?;
                    return Err(error);
                }
            },
        }

        for cycle in self.start_cycle..=self.config.design.cycles {
            if let Err(error) = self.run_cycle(cycle).await {
                self.emit(Event::TaskFailed {
                    cycle,
                    detail: error.to_string(),
                })?;
                self.record_progress()?;
                return Err(error);
            }
            self.manifest.cycles_completed = cycle;
            self.record_progress()?;
            self.snapshot(cycle, true)?;
        }

        let (best, elapsed_ms) = self.finish(self.manifest.cycles_completed, started)?;

        Ok(RunSummary {
            task_id: self.task_id.clone(),
            root: self.paths.root().to_path_buf(),
            cycles_completed: self.manifest.cycles_completed,
            folds: self.folds,
            best,
            elapsed_ms,
        })
    }

    /// Re-verify one fold result against the sequence that was sent, and
    /// record the outcome in the trace.
    ///
    /// Returns the [`RejectReason`] when the service reported a different
    /// sequence, after emitting `Event::Rejected`. The caller decides what a
    /// rejection means: `run_cycle` drops the candidate and continues,
    /// `seed_founder` fails the run. Returns `None` when the check matched or
    /// when there was nothing to check, which emits a note instead. `subject`
    /// names what was not re-verified in that note.
    async fn verify_and_record(
        &mut self,
        cycle: u32,
        candidate: CandidateId,
        sequence: &BinderSequence,
        result: &FoldResult,
        subject: &str,
    ) -> Result<Option<RejectReason>, RunError> {
        match verify_output(&self.scaffold, sequence, result) {
            Verification::Mismatch(reason) => {
                self.emit(Event::Rejected {
                    cycle,
                    candidate,
                    kind: reason.kind().to_string(),
                    reason: reason.to_string(),
                })?;
                Ok(Some(reason))
            }
            Verification::NothingToCheck { reason } => {
                self.emit(Event::Note {
                    cycle,
                    source: "verification".to_string(),
                    text: format!("{subject} not re-verified: {reason}"),
                })?;
                Ok(None)
            }
            Verification::Matched => Ok(None),
        }
    }

    /// Fold the input sequence as cycle 0 and admit it.
    ///
    /// The founder is admitted even if it would fail the gates: it is the
    /// user's starting point, and refusing it would leave the search with no
    /// parent at all. A note records the gate it failed, so nothing is hidden.
    async fn seed_founder(&mut self) -> Result<(), RunError> {
        let id = self.next_candidate_id();
        let founder = Candidate::founder(id, BinderSequence::from_scaffold(&self.scaffold));

        self.emit(Event::FoldStarted {
            cycle: 0,
            candidate: id,
        })?;

        let job = job_spec_for(
            &format!("{}-{}", self.config.name, id),
            &self.scaffold,
            &founder.sequence,
            &self.config.fold,
        );
        let result = self
            .backend
            .fold(job)
            .await
            .map_err(|error| RunError::FounderFoldFailed(error.to_string()))?;
        self.folds += 1;
        self.meter.record_fold(result.elapsed_ms);

        self.emit(Event::FoldFinished {
            cycle: 0,
            candidate: id,
            elapsed_ms: result.elapsed_ms,
        })?;

        // Re-verify before scoring, so a service that answered with a
        // different sequence never becomes the founder.
        if let Some(reason) = self
            .verify_and_record(0, id, &founder.sequence, &result, "founder sequence")
            .await?
        {
            return Err(RunError::FounderSequenceMismatch {
                reason: reason.to_string(),
            });
        }

        let metrics = metrics_from(&result.confidence);
        if let Err(reason) = self.config.gates.check(&metrics) {
            self.emit(Event::Note {
                cycle: 0,
                source: "gates".to_string(),
                text: format!("founder admitted despite failing a gate: {reason}"),
            })?;
        }

        let scored = founder.scored(metrics, &self.config.loss);
        let loss = scored.loss_or_inf();
        self.write_structure(&scored, &result)?;

        match self.population.admit(scored) {
            Ok(_) => {
                self.emit(Event::FounderScored {
                    candidate: id,
                    loss,
                })?;
                Ok(())
            }
            Err(reason) => {
                self.emit(Event::Rejected {
                    cycle: 0,
                    candidate: id,
                    kind: reason.kind().to_string(),
                    reason: reason.to_string(),
                })?;
                Err(RunError::NoParent)
            }
        }
    }

    async fn run_cycle(&mut self, cycle: u32) -> Result<(), RunError> {
        let state = self.history.state(
            cycle,
            self.population.len(),
            self.config.design.stagnation_window,
        );

        if state.stagnating {
            self.emit(Event::Stagnation {
                cycle,
                cycles_without_improvement: state.cycles_without_improvement,
            })?;
        }

        let parent = self
            .population
            .select_parent(&mut self.rng)
            .cloned()
            .ok_or(RunError::NoParent)?;

        self.emit(Event::CycleStarted {
            cycle,
            parent: parent.id,
            population: self.population.len(),
            best_loss: self.population.best().and_then(|c| c.loss),
        })?;

        // Legality first, choice second: the agent cannot reach past this set.
        let assessed = self.router.assess_all(&state);
        let legal: Vec<SkillId> = assessed
            .iter()
            .filter(|(_, verdict)| verdict.is_legal())
            .map(|(skill, _)| *skill)
            .collect();
        let blocked: Vec<String> = assessed
            .iter()
            .filter(|(_, verdict)| !verdict.is_legal())
            .map(|(skill, verdict)| format!("{skill}: {}", verdict.reason()))
            .collect();

        self.emit(Event::SkillRouted {
            cycle,
            legal: legal.iter().map(|skill| skill.to_string()).collect(),
            blocked,
        })?;

        if legal.is_empty() {
            return Err(RunError::NoLegalSkills { cycle });
        }

        let decision = self.agent.choose(&legal, &state).await?;
        debug_assert!(
            legal.contains(&decision.skill),
            "an agent must not return an illegal skill"
        );
        if !legal.contains(&decision.skill) {
            return Err(RunError::NoLegalSkills { cycle });
        }

        self.history.note_choice(decision.skill);
        self.emit(Event::SkillChosen {
            cycle,
            skill: decision.skill.to_string(),
            agent: self.agent.name().to_string(),
            decided_by: decision.decided_by.clone(),
            rationale: decision.rationale.clone(),
        })?;

        let skill: Box<dyn DesignSkill> =
            executor(decision.skill).ok_or_else(|| RunError::NoExecutor {
                skill: decision.skill.to_string(),
            })?;

        let proposals = {
            let context = SkillContext {
                scope: &self.scope,
                parent: &parent,
                cycle,
                batch_size: self.config.design.batch_size,
                max_substitutions: self.config.design.max_substitutions,
                state: &state,
            };
            skill.propose(&context, &mut self.rng)?
        };

        let proposal_count = proposals.len();
        self.history.note_proposals(decision.skill, proposal_count);
        self.emit(Event::ProposalsGenerated {
            cycle,
            skill: decision.skill.to_string(),
            count: proposal_count,
        })?;

        let candidates = self.materialise(cycle, &proposals, &parent)?;
        let results = self.fold_all(cycle, &candidates).await?;

        let mut admitted = 0usize;
        let mut rejected = 0usize;

        for (candidate, outcome) in candidates.into_iter().zip(results) {
            match outcome {
                Err(error) => {
                    self.emit(Event::FoldFailed {
                        cycle,
                        candidate: candidate.id,
                        detail: error.to_string(),
                    })?;
                    self.history.note_rejected(decision.skill);
                    rejected += 1;
                }
                Ok(result) => {
                    self.emit(Event::FoldFinished {
                        cycle,
                        candidate: candidate.id,
                        elapsed_ms: result.elapsed_ms,
                    })?;

                    // Re-verify before scoring, so a service that answered
                    // with a different sequence never reaches the gates or
                    // the loss. A mismatch is recorded, and the run moves on
                    // to the next candidate.
                    if self
                        .verify_and_record(
                            cycle,
                            candidate.id,
                            &candidate.sequence,
                            &result,
                            "sequence",
                        )
                        .await?
                        .is_some()
                    {
                        self.history.note_rejected(decision.skill);
                        rejected += 1;
                        continue;
                    }

                    let metrics = metrics_from(&result.confidence);
                    let scored = candidate.scored(metrics.clone(), &self.config.loss);

                    self.emit(Event::Scored {
                        cycle,
                        candidate: scored.id,
                        loss: scored.loss_or_inf(),
                        iptm: metrics.iptm,
                        interface_plddt: metrics.interface_plddt,
                    })?;

                    // Gates before admission, so a gate failure is recorded
                    // against the candidate rather than disappearing into a
                    // ranking.
                    if let Err(reason) = self.config.gates.check(&metrics) {
                        self.emit(Event::Rejected {
                            cycle,
                            candidate: scored.id,
                            kind: reason.kind().to_string(),
                            reason: reason.to_string(),
                        })?;
                        self.history.note_rejected(decision.skill);
                        rejected += 1;
                        continue;
                    }

                    let substitutions = describe_substitutions(&scored, &parent);
                    let id = scored.id;
                    let loss = scored.loss_or_inf();
                    let parent_id = scored.parent;
                    self.write_structure(&scored, &result)?;

                    match self.population.admit(scored) {
                        Ok(outcome) => {
                            self.history.note_admitted(decision.skill);
                            admitted += 1;
                            self.emit(Event::Admitted {
                                cycle,
                                candidate: id,
                                parent: parent_id,
                                loss,
                                outcome: describe_admission(outcome),
                                substitutions,
                            })?;
                        }
                        Err(reason) => {
                            self.history.note_rejected(decision.skill);
                            rejected += 1;
                            self.emit(Event::Rejected {
                                cycle,
                                candidate: id,
                                kind: reason.kind().to_string(),
                                reason: reason.to_string(),
                            })?;
                        }
                    }
                }
            }
        }

        let stats = self.population.stats();
        let record = self
            .history
            .record_cycle(
                CycleOutcome {
                    cycle,
                    skill: decision.skill,
                    // What the skill actually returned, not the configured
                    // batch size: a skill may produce fewer, and recording the
                    // batch size would understate its admission rate — which
                    // is the number the agent chooses on.
                    proposed: proposal_count,
                    admitted,
                    rejected,
                    best_loss: stats.best_loss,
                    mean_loss: stats.mean_loss,
                },
                self.config.design.stagnation_epsilon,
            )
            .clone();

        self.emit(Event::CycleFinished {
            cycle,
            admitted,
            rejected,
            best_loss: record.best_loss,
            mean_loss: record.mean_loss,
            improved: record.improved,
        })?;

        if cycle % 4 == 0 {
            let state = self.history.state(
                cycle,
                self.population.len(),
                self.config.design.stagnation_window,
            );
            if let Some(note) = self.agent.reflect(&state).await? {
                self.emit(Event::Note {
                    cycle,
                    source: "reflection".to_string(),
                    text: note,
                })?;
            }
        }

        Ok(())
    }

    /// Turn validated proposals into candidates with ids.
    fn materialise(
        &mut self,
        cycle: u32,
        proposals: &[Proposal],
        parent: &Candidate,
    ) -> Result<Vec<Candidate>, RunError> {
        let mut out = Vec::with_capacity(proposals.len());
        for proposal in proposals {
            let id = self.next_candidate_id();
            out.push(Candidate::from_proposal(
                id,
                cycle,
                proposal,
                &parent.sequence,
            )?);
        }
        Ok(out)
    }

    /// Fold a batch, `design.concurrency` at a time.
    ///
    /// Results come back in candidate order regardless of completion order, so
    /// the trace does not depend on which fold happened to finish first. The
    /// timestamps on concurrent folds are therefore approximate; the
    /// `elapsed_ms` the service reports is not.
    async fn fold_all(
        &mut self,
        cycle: u32,
        candidates: &[Candidate],
    ) -> Result<Vec<Result<FoldResult, FoldError>>, RunError> {
        for candidate in candidates {
            self.emit(Event::FoldStarted {
                cycle,
                candidate: candidate.id,
            })?;
        }

        let concurrency = self.config.design.concurrency.max(1);
        let mut collected: Vec<(CandidateId, Result<FoldResult, FoldError>)> = Vec::new();

        for chunk in candidates.chunks(concurrency) {
            let mut set: JoinSet<(CandidateId, Result<FoldResult, FoldError>)> = JoinSet::new();
            for candidate in chunk {
                let backend = Arc::clone(&self.backend);
                let job = job_spec_for(
                    &format!("{}-{}", self.config.name, candidate.id),
                    &self.scaffold,
                    &candidate.sequence,
                    &self.config.fold,
                );
                let id = candidate.id;
                set.spawn(async move {
                    let outcome = backend.fold(job).await;
                    (id, outcome)
                });
            }

            while let Some(joined) = set.join_next().await {
                match joined {
                    Ok(pair) => collected.push(pair),
                    Err(error) => {
                        // A panicking fold task must not take the run down.
                        collected.push((
                            CandidateId(u32::MAX),
                            Err(FoldError::Transport(format!("fold task failed: {error}"))),
                        ));
                    }
                }
            }
        }

        self.folds += collected.len();
        for (_, outcome) in &collected {
            let elapsed = outcome
                .as_ref()
                .map(|result| result.elapsed_ms)
                .unwrap_or(0);
            self.meter.record_fold(elapsed);
        }

        let mut ordered = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let position = collected
                .iter()
                .position(|(id, _)| *id == candidate.id)
                .unwrap_or(0);
            if collected.get(position).map(|(id, _)| *id) == Some(candidate.id) {
                ordered.push(collected.remove(position).1);
            } else {
                ordered.push(Err(FoldError::Transport(
                    "no result returned for this candidate".to_string(),
                )));
            }
        }
        Ok(ordered)
    }

    fn write_structure(&self, candidate: &Candidate, result: &FoldResult) -> Result<(), RunError> {
        if result.structure.ca_trace.is_empty() {
            return Ok(());
        }
        let snapshot = StructureSnapshot {
            candidate: candidate.id,
            cycle: candidate.cycle,
            loss: candidate.loss,
            ca_trace: result.structure.ca_trace.clone(),
        };
        crate::trace::write_json(self.paths.structure(candidate.id), &snapshot)?;
        Ok(())
    }

    fn snapshot(&self, cycle: u32, running: bool) -> Result<(), RunError> {
        let snapshot = PopulationSnapshot {
            task_id: self.task_id.0.clone(),
            name: self.config.name.clone(),
            cycle,
            cycles_planned: self.config.design.cycles,
            updated_at_ms: now_ms(),
            scope: self.scope.summary(),
            stats: self.population.stats(),
            members: self.population.ranked().into_iter().cloned().collect(),
            loss_curve: self.history.loss_curve(),
            records: self.history.records().to_vec(),
            folds: self.folds,
            running,
            admitted_total: self.population.admitted_total(),
            rejected_total: self.population.rejected_total(),
        };
        crate::trace::write_json(self.paths.population(), &snapshot)?;
        if let Some(best) = self.population.best() {
            crate::trace::write_json(self.paths.best(), best)?;
        }
        Ok(())
    }
}

fn describe_admission(outcome: AdmissionOutcome) -> String {
    match outcome {
        AdmissionOutcome::Admitted => "admitted".to_string(),
        AdmissionOutcome::Evicted(id) => format!("admitted, evicted {id}"),
        AdmissionOutcome::Replaced(id) => format!("admitted, replaced {id}"),
    }
}

fn describe_substitutions(candidate: &Candidate, parent: &Candidate) -> String {
    candidate
        .substitutions
        .iter()
        .map(|substitution| {
            let from = parent
                .sequence
                .chain(substitution.chain)
                .and_then(|seq| seq.chars().nth(substitution.position))
                .unwrap_or('?');
            format!(
                "{}:{} {}>{}",
                substitution.chain,
                substitution.position + 1,
                from,
                substitution.to
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Convenience for the `Legality` of each skill, for `fh validate`.
pub fn routing_table(router: &Router, state: &crate::SearchState) -> Vec<(SkillId, Legality)> {
    router.assess_all(state)
}

/// What one member of a `population.json` must look like to have come from a
/// recorded run of this config: the scaffold's binder chains at the scaffold's
/// lengths, the config's residue at every position outside the mutable scope,
/// and the loss its own metrics and trace produce. Returns why the member
/// could not have been recorded, or `None` when it passes. `scored` maps a
/// candidate id to the loss its `Scored` or `FounderScored` event recorded.
///
/// Pure function, no I/O: the caller parses the snapshot and the trace and
/// passes the values in, the way `Population::restore` takes them.
fn restored_member_defect(
    member: &Candidate,
    scaffold: &Scaffold,
    scope: &MutableScope,
    weights: &crate::LossWeights,
    scored: &std::collections::HashMap<CandidateId, f64>,
) -> Option<String> {
    let binders: Vec<&crate::scaffold::Chain> = scaffold.binder_chains().collect();

    let extras: Vec<String> = member
        .sequence
        .iter()
        .map(|(id, _)| id)
        .filter(|id| !binders.iter().any(|chain| chain.id == *id))
        .map(|id| id.to_string())
        .collect();
    if !extras.is_empty() {
        return Some(format!(
            "it carries {}, which the config's binder does not have",
            extras.join(", ")
        ));
    }

    for chain in &binders {
        let Some(sequence) = member.sequence.chain(chain.id) else {
            return Some(format!(
                "it has no chain {}, which the config's binder has",
                chain.label
            ));
        };
        let found = sequence.chars().count();
        if found != chain.len() {
            return Some(format!(
                "its chain {} is {found} residues long, but the config says {}",
                chain.label,
                chain.len()
            ));
        }
    }

    for chain in &binders {
        let configured: Vec<char> = chain.sequence.chars().collect();
        let current: Vec<char> = member
            .sequence
            .chain(chain.id)
            .expect("the chain check above passed")
            .chars()
            .collect();
        for position in 0..configured.len() {
            if !scope.is_mutable(chain.id, position) && current[position] != configured[position] {
                return Some(format!(
                    "its chain {} position {} is outside the mutable scope and holds {}, \
                     but the config says {}",
                    chain.label,
                    position + 1,
                    current[position],
                    configured[position]
                ));
            }
        }
    }

    let (Some(recorded_loss), Some(metrics)) = (member.loss, &member.metrics) else {
        return Some("it is not scored, so no recorded run retained it".to_string());
    };
    // A tolerance rather than exact equality: serde_json parses decimal text
    // to within an ULP of the double the recorded run computed from, so the
    // recomputed loss can land one bit off the recorded one without any edit.
    // An edit that moves the loss enough to matter is orders of magnitude
    // above the tolerance. The trace check below stays exact for a different
    // reason: serde_json round-trips an f64 through decimal text without
    // loss, so the two files hold the same double even though they are two
    // files.
    let recomputed = crate::loss(metrics, weights);
    if (recorded_loss - recomputed).abs() > 1e-12 {
        return Some(format!(
            "its loss {recorded_loss} is not the {recomputed} that its own metrics score \
             under the config's loss weights"
        ));
    }
    match scored.get(&member.id) {
        None => Some(
            "trace.jsonl records no scored loss for it, so the recorded run never folded it"
                .to_string(),
        ),
        Some(traced) if *traced != recorded_loss => Some(format!(
            "its loss {recorded_loss} is not the {traced} that trace.jsonl recorded when it was scored"
        )),
        Some(_) => None,
    }
}

/// Why a candidate was turned away, counted by kind — for the run summary.
pub fn rejection_counts(records: &[TraceRecord]) -> Vec<(String, usize)> {
    let mut counts: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for record in records {
        if let Event::Rejected { kind, .. } = &record.event {
            *counts.entry(kind.clone()).or_insert(0) += 1;
        }
    }
    counts.into_iter().collect()
}

/// Re-exported so callers do not need [`crate::RejectReason`] directly.
pub use crate::RejectReason as Rejection;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::HeuristicAgent;
    use crate::fold::FixtureBackend;
    use crate::ChainRole;

    const YAML: &str = r#"
name: run-test
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
    - name: loop2
      chain: H
      start: 51
      end: 58
  fixed_residues: ["H:52"]
target:
  chains:
    - label: A
      sequence: MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG
design:
  cycles: 8
  batch_size: 5
  seed: 4242
  concurrency: 3
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

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("fh-orchestrator-{name}-{}", crate::trace::now_ms()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    async fn run_once(dir: &std::path::Path, seed: u64) -> (RunSummary, Vec<Candidate>, Scaffold) {
        let mut config = TaskConfig::from_yaml(YAML).expect("config parses");
        config.design.seed = seed;
        let (scaffold, scope) = config.resolve().expect("resolves");

        let backend = Arc::new(FixtureBackend::planted(&scaffold, &scope, seed));
        let agent = Box::new(HeuristicAgent::new(seed));
        let task_id = TaskId("test".to_string());

        let mut orchestrator = Orchestrator::prepare(
            config,
            backend,
            agent,
            crate::trace::TaskPaths::new(dir),
            task_id,
        )
        .await
        .expect("prepares");

        let summary = orchestrator.run().await.expect("runs");
        let members = orchestrator.population().members().to_vec();
        (summary, members, scaffold)
    }

    /// Run a fresh task for `cycles` cycles in `dir`, leaving the artifacts a
    /// resume reads back.
    async fn run_stopped(dir: &std::path::Path, seed: u64, cycles: u32) -> RunSummary {
        let mut config = TaskConfig::from_yaml(YAML).expect("config parses");
        config.design.seed = seed;
        config.design.cycles = cycles;
        let (scaffold, scope) = config.resolve().expect("resolves");

        let backend = Arc::new(FixtureBackend::planted(&scaffold, &scope, seed));
        let mut orchestrator = Orchestrator::prepare(
            config,
            backend,
            Box::new(HeuristicAgent::new(seed)),
            crate::trace::TaskPaths::new(dir),
            TaskId("test".to_string()),
        )
        .await
        .expect("prepares");
        orchestrator.run().await.expect("runs")
    }

    /// Resume the task in `dir` and extend it to `total` cycles, loading the
    /// config back from the recorded task.yaml the way `fh resume` does.
    async fn run_resumed(
        dir: &std::path::Path,
        seed: u64,
        total: u32,
    ) -> (RunSummary, Orchestrator) {
        let paths = crate::trace::TaskPaths::new(dir);
        let config = TaskConfig::load(paths.config()).expect("recorded config loads");
        let (scaffold, scope) = config.resolve().expect("resolves");

        let backend = Arc::new(FixtureBackend::planted(&scaffold, &scope, seed));
        let mut orchestrator = Orchestrator::resume(
            config,
            backend,
            Box::new(HeuristicAgent::new(seed)),
            paths,
            Some(total),
        )
        .await
        .expect("resumes");
        let summary = orchestrator.run().await.expect("runs");
        (summary, orchestrator)
    }

    /// Unwrap the error of a refused resume. `Orchestrator` has no `Debug`,
    /// so `expect_err` cannot be used on the result.
    fn resume_error(result: Result<Orchestrator, RunError>) -> RunError {
        match result {
            Ok(_) => panic!("the resume must be refused"),
            Err(error) => error,
        }
    }

    /// The position check the invariant tests pin: every residue outside the
    /// mutable scope is still exactly what the config said it was, including
    /// the explicitly pinned residue inside a mutable region.
    fn assert_positions_untouched(
        members: &[Candidate],
        scaffold: &Scaffold,
        scope: &MutableScope,
    ) {
        for chain in scaffold
            .chains
            .iter()
            .filter(|c| c.role == ChainRole::Binder)
        {
            let founder: Vec<char> = chain.sequence.chars().collect();
            for member in members {
                let current: Vec<char> = member
                    .sequence
                    .chain(chain.id)
                    .expect("member has this chain")
                    .chars()
                    .collect();
                assert_eq!(current.len(), founder.len(), "{} changed length", member.id);
                for position in 0..founder.len() {
                    if !scope.is_mutable(chain.id, position) {
                        assert_eq!(
                            current[position],
                            founder[position],
                            "{} changed immutable {} position {} (one-based {})",
                            member.id,
                            chain.label,
                            position,
                            position + 1
                        );
                    }
                }
                // The explicitly pinned residue, stated separately: it sits
                // inside a mutable region, so only the pin protects it.
                assert_eq!(
                    current[51], founder[51],
                    "{} changed pinned H:52",
                    member.id
                );
            }
        }
    }

    /// The property the whole harness exists to guarantee, checked against a
    /// finished run rather than a single call: after eight cycles of mutation
    /// and whole-region redesign, every residue outside the mutable scope is
    /// still exactly what the config said it was.
    #[tokio::test]
    async fn no_run_ever_changes_a_position_outside_the_mutable_scope() {
        let dir = temp_dir("invariant");
        let (summary, members, scaffold) = run_once(&dir, 4242).await;

        assert_eq!(summary.cycles_completed, 8);
        assert!(
            summary.folds > 8,
            "expected real folding, got {}",
            summary.folds
        );
        assert!(members.len() > 1, "nothing was admitted");

        let config = TaskConfig::from_yaml(YAML).expect("config parses");
        let (_, scope) = config.resolve().expect("resolves");

        assert_positions_untouched(&members, &scaffold, &scope);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The same invariant across a resume: the restored population and four
    /// more cycles of mutation must not touch a position outside the scope.
    #[tokio::test]
    async fn a_resumed_run_also_keeps_every_position_outside_the_mutable_scope() {
        let dir = temp_dir("resume-invariant");
        run_stopped(&dir, 4242, 4).await;
        let (summary, orchestrator) = run_resumed(&dir, 4242, 8).await;

        assert_eq!(summary.cycles_completed, 8);
        assert!(
            summary.folds > 8,
            "expected real folding across both runs, got {}",
            summary.folds
        );
        let members = orchestrator.population().members().to_vec();
        assert!(members.len() > 1, "nothing was admitted");

        let config = TaskConfig::from_yaml(YAML).expect("config parses");
        let (scaffold, scope) = config.resolve().expect("resolves");
        assert_positions_untouched(&members, &scaffold, &scope);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn a_run_stopped_early_resumes_from_the_recorded_population() {
        let dir = temp_dir("resume-basic");
        run_stopped(&dir, 4242, 3).await;

        let paths = crate::trace::TaskPaths::new(&dir);
        let before: crate::trace::PopulationSnapshot =
            crate::trace::read_json(paths.population()).expect("snapshot reads");
        assert_eq!(before.cycle, 3);
        assert_eq!(before.records.len(), 3);
        assert!(!before.members.is_empty());

        // The restored population is the recorded one, member for member,
        // before a single new cycle runs.
        let config = TaskConfig::load(paths.config()).expect("recorded config loads");
        let (scaffold, scope) = config.resolve().expect("resolves");
        let backend = Arc::new(FixtureBackend::planted(&scaffold, &scope, 4242));
        let mut orchestrator = Orchestrator::resume(
            config,
            backend,
            Box::new(HeuristicAgent::new(4242)),
            paths.clone(),
            Some(8),
        )
        .await
        .expect("resumes");
        assert_eq!(orchestrator.population().members(), &before.members[..]);

        let summary = orchestrator.run().await.expect("runs");
        assert_eq!(summary.cycles_completed, 8);

        let after: crate::trace::PopulationSnapshot =
            crate::trace::read_json(paths.population()).expect("snapshot reads");
        assert_eq!(after.records.len(), 8);
        assert_eq!(&after.records[..3], &before.records[..]);
        assert!(!after.running);

        let manifest: RunManifest =
            crate::trace::read_json(paths.manifest()).expect("manifest reads");
        assert_eq!(manifest.cycles_completed, 8);
        assert_eq!(manifest.cycles_planned, 8);
        assert_eq!(manifest.resumed_from_cycle, Some(3));
        assert!(manifest.generator_reseeded);
        assert_eq!(manifest.folds, summary.folds);
        assert!(manifest.finished_at_ms.is_some());
        assert_eq!(manifest.best_candidate, summary.best.as_ref().map(|c| c.id));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn candidate_ids_after_a_resume_do_not_collide_with_ids_before_it() {
        let dir = temp_dir("resume-ids");
        run_stopped(&dir, 11, 3).await;

        let paths = crate::trace::TaskPaths::new(&dir);
        let before = crate::trace::read_all(paths.trace()).expect("trace reads");
        let highest_before = before
            .iter()
            .filter_map(|record| record.event.candidate())
            .map(|id| id.0)
            .max()
            .expect("the founder was folded");

        run_resumed(&dir, 11, 8).await;

        let after = crate::trace::read_all(paths.trace()).expect("trace reads");
        let resumed_at = after
            .iter()
            .position(|record| matches!(record.event, Event::TaskResumed { .. }))
            .expect("the resume is recorded");
        let new_ids: Vec<u32> = after[resumed_at..]
            .iter()
            .filter_map(|record| record.event.candidate())
            .map(|id| id.0)
            .collect();
        assert!(!new_ids.is_empty(), "the resumed run folded nothing");
        assert!(
            new_ids.iter().all(|id| *id > highest_before),
            "an id after the resume reuses a recorded candidate: {new_ids:?} vs {highest_before}"
        );

        let snapshot: crate::trace::PopulationSnapshot =
            crate::trace::read_json(paths.population()).expect("snapshot reads");
        let mut ids: Vec<u32> = snapshot.members.iter().map(|member| member.id.0).collect();
        let count = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), count, "two retained members share an id");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn a_resumed_trace_keeps_the_earlier_records_and_continues_the_numbering() {
        let dir = temp_dir("resume-trace");
        run_stopped(&dir, 5, 2).await;

        let paths = crate::trace::TaskPaths::new(&dir);
        let before = crate::trace::read_all(paths.trace()).expect("trace reads");
        assert!(matches!(
            before.last().map(|record| &record.event),
            Some(Event::TaskFinished { .. })
        ));

        run_resumed(&dir, 5, 6).await;

        let after = crate::trace::read_all(paths.trace()).expect("trace reads");
        assert!(after.len() > before.len());
        assert_eq!(
            &after[..before.len()],
            &before[..],
            "the earlier records must survive untouched"
        );
        for (index, record) in after.iter().enumerate() {
            assert_eq!(record.seq, index as u64 + 1, "sequence numbers stay dense");
        }
        assert!(matches!(
            &after[before.len()].event,
            Event::TaskStarted { .. }
        ));
        assert!(matches!(
            &after[before.len() + 1].event,
            Event::TaskResumed {
                cycle: 2,
                generator_reseeded: true,
                ..
            }
        ));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn a_config_that_does_not_match_the_recorded_digest_is_refused() {
        let dir = temp_dir("resume-mismatch");
        run_stopped(&dir, 3, 2).await;

        let paths = crate::trace::TaskPaths::new(&dir);
        let mut config = TaskConfig::load(paths.config()).expect("recorded config loads");
        config.gates.min_iptm += 0.05;
        let (scaffold, scope) = config.resolve().expect("resolves");

        let backend = Arc::new(FixtureBackend::planted(&scaffold, &scope, 3));
        let error = resume_error(
            Orchestrator::resume(
                config,
                backend,
                Box::new(HeuristicAgent::new(3)),
                paths.clone(),
                None,
            )
            .await,
        );

        match &error {
            RunError::ConfigMismatch {
                path,
                recorded,
                current,
            } => {
                assert_eq!(*path, paths.config());
                assert_ne!(recorded, current);
                let message = error.to_string();
                assert!(message.contains("task.yaml"), "{message}");
                assert!(
                    message.contains(recorded.as_str()),
                    "the refusal must name the recorded digest: {message}"
                );
                assert!(
                    message.contains(current.as_str()),
                    "the refusal must name the digest it found: {message}"
                );
            }
            other => panic!("expected a config mismatch, got {other:?}"),
        }

        // The refusal wrote no resume event into the recorded trace.
        let records = crate::trace::read_all(paths.trace()).expect("trace reads");
        assert!(!records
            .iter()
            .any(|record| matches!(record.event, Event::TaskResumed { .. })));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn a_finished_run_refuses_a_resume_that_does_not_extend_it() {
        let dir = temp_dir("resume-done");
        run_stopped(&dir, 2, 3).await;

        let paths = crate::trace::TaskPaths::new(&dir);
        let config = TaskConfig::load(paths.config()).expect("recorded config loads");
        let (scaffold, scope) = config.resolve().expect("resolves");

        let backend = Arc::new(FixtureBackend::planted(&scaffold, &scope, 2));
        let error = resume_error(
            Orchestrator::resume(
                config.clone(),
                backend,
                Box::new(HeuristicAgent::new(2)),
                paths.clone(),
                None,
            )
            .await,
        );
        assert!(
            matches!(error, RunError::AlreadyComplete { completed: 3, .. }),
            "{error:?}"
        );

        let backend = Arc::new(FixtureBackend::planted(&scaffold, &scope, 2));
        let error = resume_error(
            Orchestrator::resume(
                config,
                backend,
                Box::new(HeuristicAgent::new(2)),
                paths.clone(),
                Some(2),
            )
            .await,
        );
        match error {
            RunError::CyclesBelowCompleted {
                completed,
                requested,
                ..
            } => {
                assert_eq!(completed, 3);
                assert_eq!(requested, 2);
            }
            other => panic!("expected a cycles refusal, got {other:?}"),
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn the_same_seed_reproduces_the_same_run() {
        let first_dir = temp_dir("repro-a");
        let second_dir = temp_dir("repro-b");

        let (first, first_members, _) = run_once(&first_dir, 99).await;
        let (second, second_members, _) = run_once(&second_dir, 99).await;

        assert_eq!(
            first.best.as_ref().and_then(|c| c.loss),
            second.best.as_ref().and_then(|c| c.loss)
        );
        assert_eq!(first.folds, second.folds);
        assert_eq!(
            first_members.iter().map(|c| c.id).collect::<Vec<_>>(),
            second_members.iter().map(|c| c.id).collect::<Vec<_>>()
        );
        assert_eq!(
            first.best.as_ref().map(|c| c.sequence.clone()),
            second.best.as_ref().map(|c| c.sequence.clone())
        );

        std::fs::remove_dir_all(&first_dir).ok();
        std::fs::remove_dir_all(&second_dir).ok();
    }

    #[tokio::test]
    async fn a_run_leaves_a_readable_trace_and_manifest() {
        let dir = temp_dir("artifacts");
        let (summary, _, _) = run_once(&dir, 7).await;
        let paths = crate::trace::TaskPaths::new(&dir);

        let records = crate::trace::read_all(paths.trace()).expect("trace reads");
        assert!(!records.is_empty());
        assert!(matches!(
            records.first().map(|r| &r.event),
            Some(Event::TaskStarted { .. })
        ));
        assert!(matches!(
            records.last().map(|r| &r.event),
            Some(Event::TaskFinished { .. })
        ));
        // Sequence numbers are dense and ordered.
        for (index, record) in records.iter().enumerate() {
            assert_eq!(record.seq, index as u64 + 1);
        }

        let manifest: RunManifest =
            crate::trace::read_json(paths.manifest()).expect("manifest reads");
        assert_eq!(manifest.cycles_completed, 8);
        assert_eq!(manifest.seed, 7);
        assert_eq!(manifest.config_sha256.len(), 64);
        assert!(manifest.finished_at_ms.is_some());
        assert_eq!(manifest.best_candidate, summary.best.map(|c| c.id));

        let snapshot: crate::trace::PopulationSnapshot =
            crate::trace::read_json(paths.population()).expect("population reads");
        assert!(!snapshot.members.is_empty());
        assert!(!snapshot.running);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A gate set above the founder's score cannot admit anything. The run
    /// must still complete and say so, rather than hanging or panicking.
    #[tokio::test]
    async fn an_unreachable_gate_rejects_everything_without_failing_the_run() {
        let dir = temp_dir("gates");
        let mut config = TaskConfig::from_yaml(YAML).expect("config parses");
        config.gates.min_iptm = 0.99;
        config.design.cycles = 3;
        let (scaffold, scope) = config.resolve().expect("resolves");

        let backend = Arc::new(FixtureBackend::planted(&scaffold, &scope, 1));
        let orchestrator = Orchestrator::prepare(
            config,
            backend,
            Box::new(HeuristicAgent::new(1)),
            crate::trace::TaskPaths::new(&dir),
            TaskId("gates".to_string()),
        )
        .await
        .expect("prepares");

        let mut orchestrator = orchestrator;
        let summary = orchestrator.run().await.expect("completes");
        assert_eq!(summary.cycles_completed, 3);
        // Only the founder, which is admitted despite the gate so the search
        // has a parent at all.
        assert_eq!(orchestrator.population().len(), 1);

        let records = crate::trace::read_all(crate::trace::TaskPaths::new(&dir).trace())
            .expect("trace reads");
        assert!(records.iter().any(|record| matches!(
            &record.event,
            Event::Note { source, .. } if source == "gates"
        )));
        assert!(records
            .iter()
            .any(|record| matches!(&record.event, Event::Rejected { .. })));

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Delegates every decision to the heuristic and reports a fixed model
    /// spend, standing in for a run driven by a provider that counts.
    struct SpendingAgent {
        inner: HeuristicAgent,
        spent: crate::usage::Usage,
    }

    #[async_trait::async_trait]
    impl DesignAgent for SpendingAgent {
        fn name(&self) -> &str {
            self.inner.name()
        }

        async fn choose(
            &mut self,
            legal: &[SkillId],
            state: &crate::SearchState,
        ) -> Result<crate::agent::Decision, AgentError> {
            self.inner.choose(legal, state).await
        }

        fn usage(&self) -> crate::usage::Usage {
            self.spent
        }
    }

    /// Delegates every decision to the heuristic and reports uncounted model
    /// spend, standing in for an `LlmAgent` over a provider that keeps the
    /// `ModelProvider::usage()` default.
    struct NonCountingAgent {
        inner: HeuristicAgent,
    }

    #[async_trait::async_trait]
    impl DesignAgent for NonCountingAgent {
        fn name(&self) -> &str {
            self.inner.name()
        }

        async fn choose(
            &mut self,
            legal: &[SkillId],
            state: &crate::SearchState,
        ) -> Result<crate::agent::Decision, AgentError> {
            self.inner.choose(legal, state).await
        }

        fn usage(&self) -> crate::usage::Usage {
            crate::usage::Usage::uncounted()
        }
    }

    /// Rewrite the `usage` block of the recorded `run.json` in `dir`, the way
    /// the corrupt-artifact tests edit `population.json`.
    fn rewrite_recorded_usage(dir: &std::path::Path, usage: serde_json::Value) {
        let paths = crate::trace::TaskPaths::new(dir);
        let mut value: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(paths.manifest()).expect("manifest reads"),
        )
        .expect("manifest is json");
        value["usage"] = usage;
        std::fs::write(
            paths.manifest(),
            serde_json::to_string_pretty(&value).expect("encode"),
        )
        .expect("manifest writes");
    }

    #[tokio::test]
    async fn an_agent_that_reports_spend_has_it_folded_into_the_manifest() {
        let dir = temp_dir("agent-spend");
        let mut config = TaskConfig::from_yaml(YAML).expect("config parses");
        config.design.cycles = 2;
        let (scaffold, scope) = config.resolve().expect("resolves");

        let spent = crate::usage::Usage {
            calls: 3,
            input_tokens: 900,
            output_tokens: 120,
            folds: 0,
            fold_ms: 0,
            calls_counted: Some(true),
        };
        let backend = Arc::new(FixtureBackend::planted(&scaffold, &scope, 11));
        let mut orchestrator = Orchestrator::prepare(
            config,
            backend,
            Box::new(SpendingAgent {
                inner: HeuristicAgent::new(11),
                spent,
            }),
            crate::trace::TaskPaths::new(&dir),
            TaskId("agent-spend".to_string()),
        )
        .await
        .expect("prepares");

        orchestrator.run().await.expect("completes");

        let manifest: RunManifest =
            crate::trace::read_json(crate::trace::TaskPaths::new(&dir).manifest())
                .expect("manifest reads");
        assert_eq!(manifest.usage.calls, spent.calls);
        assert_eq!(manifest.usage.input_tokens, spent.input_tokens);
        assert_eq!(manifest.usage.output_tokens, spent.output_tokens);
        assert_eq!(
            manifest.usage.calls_counted,
            Some(true),
            "a counted agent's manifest says its model spend was counted"
        );
        assert!(
            manifest.usage.folds > 0,
            "the run's own folds must still land in the usage"
        );
        assert_eq!(manifest.usage.folds, manifest.folds as u64);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn an_agent_that_makes_no_model_calls_leaves_the_manifest_carrying_only_folds() {
        let dir = temp_dir("no-model-spend");
        let mut config = TaskConfig::from_yaml(YAML).expect("config parses");
        config.design.cycles = 2;
        let (scaffold, scope) = config.resolve().expect("resolves");

        let backend = Arc::new(FixtureBackend::planted(&scaffold, &scope, 12));
        let mut orchestrator = Orchestrator::prepare(
            config,
            backend,
            Box::new(HeuristicAgent::new(12)),
            crate::trace::TaskPaths::new(&dir),
            TaskId("no-model-spend".to_string()),
        )
        .await
        .expect("prepares");

        orchestrator.run().await.expect("completes");

        let manifest: RunManifest =
            crate::trace::read_json(crate::trace::TaskPaths::new(&dir).manifest())
                .expect("manifest reads");
        assert_eq!(manifest.usage.calls, 0);
        assert_eq!(manifest.usage.total_tokens(), 0);
        assert_eq!(
            manifest.usage.calls_counted,
            Some(true),
            "no model was called, and the manifest says that rather than staying silent"
        );
        assert_eq!(manifest.usage.folds, manifest.folds as u64);
        assert!(manifest.usage.folds > 0);
        // The summary names the spend that happened and never claims a token
        // count for calls that were not made.
        let summary = manifest.usage.summary();
        assert!(summary.contains("fold(s)"), "{summary}");
        assert!(!summary.contains("call(s)"), "{summary}");
        assert!(!summary.contains("uncounted"), "{summary}");
        assert!(!summary.contains("unknown"), "{summary}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The three sources of a resumed manifest's usage, end to end: the spend
    /// run.json recorded, the folds this run's meter counted, and the model
    /// calls this run's agent reports. Every counter is pinned separately, so
    /// dropping `usage_base` from the sum or adding it twice fails here.
    #[tokio::test]
    async fn a_resumed_run_sums_recorded_spend_with_the_meter_and_the_agent() {
        let dir = temp_dir("resume-usage");
        run_stopped(&dir, 4242, 3).await;
        let paths = crate::trace::TaskPaths::new(&dir);

        let before: crate::trace::PopulationSnapshot =
            crate::trace::read_json(paths.population()).expect("snapshot reads");

        // Give the recorded run a spend of its own by editing run.json the
        // way the corrupt-artifact tests edit population.json.
        rewrite_recorded_usage(
            &dir,
            serde_json::json!({
                "calls": 10,
                "input_tokens": 1000,
                "output_tokens": 200,
                "folds": 7,
                "fold_ms": 3500,
                "calls_counted": true,
            }),
        );

        let spent = crate::usage::Usage {
            calls: 3,
            input_tokens: 900,
            output_tokens: 120,
            folds: 0,
            fold_ms: 0,
            calls_counted: Some(true),
        };
        let config = TaskConfig::load(paths.config()).expect("recorded config loads");
        let (scaffold, scope) = config.resolve().expect("resolves");
        let backend = Arc::new(FixtureBackend::planted(&scaffold, &scope, 4242));
        let mut orchestrator = Orchestrator::resume(
            config,
            backend,
            Box::new(SpendingAgent {
                inner: HeuristicAgent::new(4242),
                spent,
            }),
            paths.clone(),
            Some(6),
        )
        .await
        .expect("resumes");
        let summary = orchestrator.run().await.expect("runs");

        // The meter holds only the resumed run's folds: `self.folds` starts
        // at the snapshot's count and the summary spans both runs. The
        // fixture reports `elapsed_ms` equal to its latency and `planted`
        // uses zero latency, so the meter adds no fold_ms.
        let new_folds = summary.folds - before.folds;
        assert!(new_folds > 0, "the resumed run folded nothing");

        let manifest: RunManifest =
            crate::trace::read_json(paths.manifest()).expect("manifest reads");
        assert_eq!(manifest.usage.calls, 10 + spent.calls);
        assert_eq!(manifest.usage.input_tokens, 1000 + spent.input_tokens);
        assert_eq!(manifest.usage.output_tokens, 200 + spent.output_tokens);
        assert_eq!(manifest.usage.folds, 7 + new_folds as u64);
        assert_eq!(manifest.usage.fold_ms, 3500);
        assert_eq!(
            manifest.usage.calls_counted,
            Some(true),
            "counted base plus counted run stays counted"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A run whose agent wraps a provider that does not count must not write
    /// `calls: 0` and stop: zeros in a manifest read as free calls, so the
    /// manifest carries `calls_counted: false` and the summary says the spend
    /// was not counted.
    #[tokio::test]
    async fn an_agent_that_does_not_count_writes_an_uncounted_manifest_rather_than_zero_spend() {
        let dir = temp_dir("uncounted-spend");
        let mut config = TaskConfig::from_yaml(YAML).expect("config parses");
        config.design.cycles = 2;
        let (scaffold, scope) = config.resolve().expect("resolves");

        let backend = Arc::new(FixtureBackend::planted(&scaffold, &scope, 13));
        let mut orchestrator = Orchestrator::prepare(
            config,
            backend,
            Box::new(NonCountingAgent {
                inner: HeuristicAgent::new(13),
            }),
            crate::trace::TaskPaths::new(&dir),
            TaskId("uncounted-spend".to_string()),
        )
        .await
        .expect("prepares");

        orchestrator.run().await.expect("completes");

        let paths = crate::trace::TaskPaths::new(&dir);
        let manifest: RunManifest =
            crate::trace::read_json(paths.manifest()).expect("manifest reads");
        assert_eq!(manifest.usage.calls, 0);
        assert_eq!(manifest.usage.calls_counted, Some(false));
        assert!(!manifest.usage.is_empty());
        assert!(manifest.usage.summary().contains("uncounted"));

        // The flag is on the wire, so a later reader sees the claim.
        let raw = std::fs::read_to_string(paths.manifest()).expect("manifest reads");
        assert!(raw.contains("\"calls_counted\": false"), "{raw}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A recorded `run.json` from before `calls_counted` existed has counters
    /// but no claim. Resuming over it must sum into unknown, not into the
    /// fresh run's counted state: part of the total predates counting.
    #[tokio::test]
    async fn a_resume_over_a_manifest_that_predates_counting_reads_as_unknown() {
        let dir = temp_dir("resume-unknown");
        run_stopped(&dir, 4343, 3).await;

        // The old wire form: no calls_counted field at all.
        rewrite_recorded_usage(
            &dir,
            serde_json::json!({
                "calls": 4,
                "input_tokens": 400,
                "output_tokens": 80,
                "folds": 7,
                "fold_ms": 3500,
            }),
        );

        let spent = crate::usage::Usage {
            calls: 1,
            input_tokens: 10,
            output_tokens: 5,
            folds: 0,
            fold_ms: 0,
            calls_counted: Some(true),
        };
        let paths = crate::trace::TaskPaths::new(&dir);
        let config = TaskConfig::load(paths.config()).expect("recorded config loads");
        let (scaffold, scope) = config.resolve().expect("resolves");
        let backend = Arc::new(FixtureBackend::planted(&scaffold, &scope, 4343));
        let mut orchestrator = Orchestrator::resume(
            config,
            backend,
            Box::new(SpendingAgent {
                inner: HeuristicAgent::new(4343),
                spent,
            }),
            paths.clone(),
            Some(6),
        )
        .await
        .expect("resumes");
        orchestrator.run().await.expect("runs");

        let manifest: RunManifest =
            crate::trace::read_json(paths.manifest()).expect("manifest reads");
        assert_eq!(manifest.usage.calls, 4 + spent.calls);
        assert_eq!(
            manifest.usage.calls_counted, None,
            "an old base makes the total unknown, not counted and not uncounted"
        );
        let summary = manifest.usage.summary();
        assert!(summary.contains("counting not recorded"), "{summary}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The two sources of a resume may disagree: the recorded run counted,
    /// this run's agent does not. The total contains unmeasured spend, so it
    /// reports uncounted rather than the recorded run's counted state.
    #[tokio::test]
    async fn a_resume_whose_sources_disagree_about_counting_reports_uncounted() {
        let dir = temp_dir("resume-disagree");
        run_stopped(&dir, 4444, 3).await;

        rewrite_recorded_usage(
            &dir,
            serde_json::json!({
                "calls": 4,
                "input_tokens": 400,
                "output_tokens": 80,
                "folds": 7,
                "fold_ms": 3500,
                "calls_counted": true,
            }),
        );

        let paths = crate::trace::TaskPaths::new(&dir);
        let config = TaskConfig::load(paths.config()).expect("recorded config loads");
        let (scaffold, scope) = config.resolve().expect("resolves");
        let backend = Arc::new(FixtureBackend::planted(&scaffold, &scope, 4444));
        let mut orchestrator = Orchestrator::resume(
            config,
            backend,
            Box::new(NonCountingAgent {
                inner: HeuristicAgent::new(4444),
            }),
            paths.clone(),
            Some(6),
        )
        .await
        .expect("resumes");
        orchestrator.run().await.expect("runs");

        let manifest: RunManifest =
            crate::trace::read_json(paths.manifest()).expect("manifest reads");
        assert_eq!(manifest.usage.calls, 4, "the recorded calls still sum");
        assert_eq!(manifest.usage.calls_counted, Some(false));

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The founder-only control arm persists the same counters `run` does:
    /// the founder's fold and the agent's spend land in run.json rather than
    /// staying in memory.
    #[tokio::test]
    async fn the_founder_only_arm_records_its_fold_and_the_agent_spend() {
        let dir = temp_dir("founder-only-usage");
        let config = TaskConfig::from_yaml(YAML).expect("config parses");
        let (scaffold, scope) = config.resolve().expect("resolves");

        let spent = crate::usage::Usage {
            calls: 1,
            input_tokens: 400,
            output_tokens: 60,
            folds: 0,
            fold_ms: 0,
            calls_counted: Some(true),
        };
        let backend = Arc::new(FixtureBackend::planted(&scaffold, &scope, 21));
        let mut orchestrator = Orchestrator::prepare(
            config,
            backend,
            Box::new(SpendingAgent {
                inner: HeuristicAgent::new(21),
                spent,
            }),
            crate::trace::TaskPaths::new(&dir),
            TaskId("founder-only".to_string()),
        )
        .await
        .expect("prepares");

        let summary = orchestrator.seed_founder_only().await.expect("completes");
        assert_eq!(summary.folds, 1, "the founder fold is the arm's only fold");

        let manifest: RunManifest =
            crate::trace::read_json(crate::trace::TaskPaths::new(&dir).manifest())
                .expect("manifest reads");
        assert_eq!(manifest.folds, 1);
        assert_eq!(manifest.usage.folds, 1);
        // The fixture reports `elapsed_ms` equal to its latency and `planted`
        // uses zero latency, so the fold costs no recorded time.
        assert_eq!(manifest.usage.fold_ms, 0);
        assert_eq!(manifest.usage.calls, spent.calls);
        assert_eq!(manifest.usage.input_tokens, spent.input_tokens);
        assert_eq!(manifest.usage.output_tokens, spent.output_tokens);
        assert_eq!(manifest.usage.calls_counted, Some(true));

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The founder-only arm closes its manifest the way `run` closes one: the
    /// task directory reads as finished, and the best it names is the founder
    /// the population actually holds. Without this, `fh ls` and the web view
    /// report every completed benchmark floor as still running, forever.
    #[tokio::test]
    async fn the_founder_only_arm_leaves_a_manifest_that_reads_as_finished() {
        let dir = temp_dir("founder-only-manifest");
        let config = TaskConfig::from_yaml(YAML).expect("config parses");
        let (scaffold, scope) = config.resolve().expect("resolves");

        let backend = Arc::new(FixtureBackend::planted(&scaffold, &scope, 21));
        let mut orchestrator = Orchestrator::prepare(
            config,
            backend,
            Box::new(HeuristicAgent::new(21)),
            crate::trace::TaskPaths::new(&dir),
            TaskId("founder-only".to_string()),
        )
        .await
        .expect("prepares");

        let summary = orchestrator.seed_founder_only().await.expect("completes");
        let founder = orchestrator
            .population()
            .best()
            .cloned()
            .expect("the founder is admitted");
        let founder_loss = founder.loss.expect("the founder is scored");

        let paths = crate::trace::TaskPaths::new(&dir);
        let manifest: RunManifest =
            crate::trace::read_json(paths.manifest()).expect("manifest reads");
        assert!(
            manifest.finished_at_ms.is_some(),
            "the arm never marked the run finished"
        );
        assert_eq!(manifest.cycles_completed, 0);
        assert_eq!(manifest.best_candidate, Some(founder.id));
        // The loss the population holds, not a literal a fixture change
        // would silently invalidate.
        assert_eq!(manifest.best_loss, Some(founder_loss));
        // The counters `record_progress` persists survive the completion step.
        assert_eq!(manifest.folds, 1);
        assert_eq!(manifest.usage.folds, 1);

        let snapshot: crate::trace::PopulationSnapshot =
            crate::trace::read_json(paths.population()).expect("population reads");
        assert!(!snapshot.running);
        assert_eq!(snapshot.cycle, 0);
        assert_eq!(snapshot.members.len(), 1);
        assert_eq!(snapshot.members[0].loss, Some(founder_loss));

        // What `fh bench` reads from the summary does not move.
        assert_eq!(summary.cycles_completed, 0);
        assert_eq!(summary.folds, 1);
        assert_eq!(summary.best.as_ref().map(|c| c.id), Some(founder.id));

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Wraps the fixture and substitutes one residue on every fold from
    /// `corrupt_from` onward, counting calls. The founder folds first, so
    /// `corrupt_from: 1` admits the founder and corrupts every candidate from
    /// the cycles, while `corrupt_from: 0` corrupts the founder itself.
    struct SubstitutingBackend {
        inner: FixtureBackend,
        calls: std::sync::atomic::AtomicUsize,
        corrupt_from: usize,
    }

    #[async_trait::async_trait]
    impl FoldBackend for SubstitutingBackend {
        fn name(&self) -> &str {
            "substituting-fixture"
        }

        async fn health(&self) -> Result<crate::contract::Health, FoldError> {
            self.inner.health().await
        }

        async fn fold(
            &self,
            job: crate::contract::JobSpec,
        ) -> Result<crate::contract::FoldResult, FoldError> {
            let mut result = self.inner.fold(job).await?;
            let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if call >= self.corrupt_from {
                let residue = result
                    .structure
                    .ca_trace
                    .iter_mut()
                    .find(|residue| residue.designed)
                    .expect("the fixture traces every designed residue");
                residue.residue = if residue.residue == 'A' { 'C' } else { 'A' };
            }
            Ok(result)
        }
    }

    /// The end-to-end form of the re-verification: a service that quietly
    /// answers with a different sequence has every candidate rejected under
    /// `sequence_mismatch`, and the trace records the chain, the position and
    /// both residues.
    #[tokio::test]
    async fn a_backend_that_substitutes_a_residue_has_its_candidates_rejected() {
        let dir = temp_dir("verification");
        let mut config = TaskConfig::from_yaml(YAML).expect("config parses");
        config.design.cycles = 2;
        let (scaffold, scope) = config.resolve().expect("resolves");

        let backend = Arc::new(SubstitutingBackend {
            inner: FixtureBackend::planted(&scaffold, &scope, 5),
            calls: std::sync::atomic::AtomicUsize::new(0),
            corrupt_from: 1,
        });
        let mut orchestrator = Orchestrator::prepare(
            config,
            backend,
            Box::new(HeuristicAgent::new(5)),
            crate::trace::TaskPaths::new(&dir),
            TaskId("verification".to_string()),
        )
        .await
        .expect("prepares");

        let summary = orchestrator.run().await.expect("completes");
        assert_eq!(summary.cycles_completed, 2);
        // Only the founder survives: every candidate came back corrupted.
        assert_eq!(orchestrator.population().len(), 1);

        let records = crate::trace::read_all(crate::trace::TaskPaths::new(&dir).trace())
            .expect("trace reads");
        let reason = records
            .iter()
            .find_map(|record| match &record.event {
                Event::Rejected { kind, reason, .. } if kind == "sequence_mismatch" => {
                    Some(reason.clone())
                }
                _ => None,
            })
            .expect("a substituted residue is rejected");
        // The first designed residue of chain H, which no proposal can reach
        // because it sits outside the mutable scope.
        assert!(reason.contains("chain H"), "{reason}");
        assert!(reason.contains("position 1"), "{reason}");
        assert!(reason.contains("sent Q"), "{reason}");
        assert!(reason.contains("returned A"), "{reason}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The founder half of re-verification: a service that answers with a
    /// different sequence for the founder fails the run, and the error names
    /// the cause rather than reporting an empty population.
    #[tokio::test]
    async fn a_founder_the_service_changed_fails_the_run_naming_the_cause() {
        let dir = temp_dir("founder-verification");
        let config = TaskConfig::from_yaml(YAML).expect("config parses");
        let (scaffold, scope) = config.resolve().expect("resolves");

        let backend = Arc::new(SubstitutingBackend {
            inner: FixtureBackend::planted(&scaffold, &scope, 5),
            calls: std::sync::atomic::AtomicUsize::new(0),
            corrupt_from: 0,
        });
        let mut orchestrator = Orchestrator::prepare(
            config,
            backend,
            Box::new(HeuristicAgent::new(5)),
            crate::trace::TaskPaths::new(&dir),
            TaskId("founder-verification".to_string()),
        )
        .await
        .expect("prepares");

        let error = orchestrator.run().await.expect_err("the run must fail");
        assert!(
            matches!(error, RunError::FounderSequenceMismatch { .. }),
            "expected a founder sequence mismatch, got {error:?}"
        );
        // The full message, pinned exactly: the consequence, the cause with
        // its chain, position and both residues, and the way out.
        assert_eq!(
            error.to_string(),
            "there is no founder to design from. sequence mismatch on chain H: \
             position 1: the harness sent Q, the service returned A. \
             Check the service against the contract in src/contract.rs"
        );

        // The rejection lands in the trace one event before the failure, so
        // nothing is hidden by the run stopping.
        let records = crate::trace::read_all(crate::trace::TaskPaths::new(&dir).trace())
            .expect("trace reads");
        assert!(records.iter().any(|record| matches!(
            &record.event,
            Event::Rejected { kind, .. } if kind == "sequence_mismatch"
        )));

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Rewrite population.json the way a user editing the file would: read it
    /// as JSON, apply `corrupt`, write it back.
    fn corrupt_population(dir: &std::path::Path, corrupt: impl FnOnce(&mut serde_json::Value)) {
        let path = crate::trace::TaskPaths::new(dir).population();
        let mut value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("snapshot reads"))
                .expect("snapshot is json");
        corrupt(&mut value);
        std::fs::write(&path, serde_json::to_string_pretty(&value).expect("encode"))
            .expect("snapshot writes");
    }

    /// Attempt the resume of the task in `dir` and return the refusal.
    async fn refused_resume(dir: &std::path::Path, seed: u64) -> RunError {
        let paths = crate::trace::TaskPaths::new(dir);
        let config = TaskConfig::load(paths.config()).expect("recorded config loads");
        let (scaffold, scope) = config.resolve().expect("resolves");
        let backend = Arc::new(FixtureBackend::planted(&scaffold, &scope, seed));
        resume_error(
            Orchestrator::resume(
                config,
                backend,
                Box::new(HeuristicAgent::new(seed)),
                paths,
                Some(8),
            )
            .await,
        )
    }

    /// Assert the refusal is about a restored member, that its message names
    /// population.json and the candidate, and that the reason contains
    /// `expected`. Returns the candidate the refusal named.
    fn member_refusal(error: &RunError, expected: &str) -> CandidateId {
        match error {
            RunError::SnapshotMemberInvalid {
                candidate, reason, ..
            } => {
                let message = error.to_string();
                assert!(message.contains("population.json"), "{message}");
                assert!(
                    message.contains(&candidate.to_string()),
                    "the refusal must name the candidate: {message}"
                );
                assert!(reason.contains(expected), "{reason}");
                *candidate
            }
            other => panic!("expected a restored-member refusal, got {other:?}"),
        }
    }

    /// Harm one of an unvalidated restore: `job_spec_for` falls back to the
    /// scaffold's starting sequence for a binder chain the candidate lacks, so
    /// a restored parent missing that chain would have it folded silently and
    /// its child scored as designed. Resume refuses the member instead.
    #[tokio::test]
    async fn a_restored_member_missing_a_binder_chain_is_refused() {
        let dir = temp_dir("resume-missing-chain");
        run_stopped(&dir, 4242, 2).await;

        let paths = crate::trace::TaskPaths::new(&dir);
        let before: crate::trace::PopulationSnapshot =
            crate::trace::read_json(paths.population()).expect("snapshot reads");
        let victim = before.members[0].id;

        let manifest_before = std::fs::read(paths.manifest()).expect("manifest reads");
        let trace_before = std::fs::read(paths.trace()).expect("trace reads");

        corrupt_population(&dir, |value| {
            value["members"][0]["sequence"]["chains"]
                .as_object_mut()
                .expect("chains")
                .clear();
        });

        let error = refused_resume(&dir, 4242).await;
        assert_eq!(member_refusal(&error, "no chain H"), victim);

        // The refusal wrote nothing into the recorded artifacts.
        let after: crate::trace::PopulationSnapshot =
            crate::trace::read_json(paths.population()).expect("snapshot reads");
        assert_eq!(after.cycle, before.cycle);
        assert_eq!(
            std::fs::read(paths.manifest()).expect("manifest reads"),
            manifest_before,
            "a refused resume rewrote run.json"
        );
        assert_eq!(
            std::fs::read(paths.trace()).expect("trace reads"),
            trace_before,
            "a refused resume appended to trace.jsonl"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Harm two of an unvalidated restore: `apply_to` checks only the
    /// positions a proposal substitutes, so a restored parent with an edited
    /// immutable position would propagate that residue into its children.
    /// Resume refuses the member instead.
    #[tokio::test]
    async fn a_restored_member_with_an_edited_immutable_position_is_refused() {
        let dir = temp_dir("resume-edited-position");
        run_stopped(&dir, 4242, 2).await;

        // Position 1 of chain H is outside both mutable regions, which start
        // at 26 and 51 (one-based).
        corrupt_population(&dir, |value| {
            let chains = value["members"][0]["sequence"]["chains"]
                .as_object_mut()
                .expect("chains");
            let key = chains.keys().next().expect("a binder chain").clone();
            let mut residues: Vec<char> =
                chains[&key].as_str().expect("sequence").chars().collect();
            residues[0] = if residues[0] == 'A' { 'C' } else { 'A' };
            chains.insert(
                key,
                serde_json::Value::String(residues.into_iter().collect()),
            );
        });

        let error = refused_resume(&dir, 4242).await;
        member_refusal(&error, "outside the mutable scope");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Harm three of an unvalidated restore: a fabricated loss makes the
    /// member `population.best()`, and completion writes it to `best.json`
    /// and `manifest.best_candidate`. Both shapes are refused: a loss the
    /// member's own metrics do not score, and a self-consistent loss no fold
    /// in the trace ever recorded.
    #[tokio::test]
    async fn a_restored_member_with_a_fabricated_loss_is_refused() {
        let dir = temp_dir("resume-fabricated-loss");
        run_stopped(&dir, 4242, 2).await;

        // No recorded loss is negative: every penalty is clamped to 0..1.
        corrupt_population(&dir, |value| {
            value["members"][0]["loss"] = serde_json::json!(-1.0);
        });
        let error = refused_resume(&dir, 4242).await;
        member_refusal(&error, "loss weights");

        // The fabrication the weights check cannot catch: rewrite a metric
        // and recompute the loss from it, so member and weights agree. The
        // fixture never reports iptm 1.0, so the recomputed loss differs
        // from the one the trace recorded when this candidate was scored.
        let paths = crate::trace::TaskPaths::new(&dir);
        let mut snapshot: crate::trace::PopulationSnapshot =
            crate::trace::read_json(paths.population()).expect("snapshot reads");
        let victim = &mut snapshot.members[0];
        victim.metrics.as_mut().expect("members are scored").iptm = 1.0;
        let config = TaskConfig::load(paths.config()).expect("recorded config loads");
        victim.loss = Some(crate::loss(
            victim.metrics.as_ref().expect("members are scored"),
            &config.loss,
        ));
        crate::trace::write_json(paths.population(), &snapshot).expect("snapshot writes");

        let error = refused_resume(&dir, 4242).await;
        member_refusal(&error, "trace.jsonl");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A resume killed after extending the plan leaves a manifest whose
    /// `cycles_planned` is larger than the config's own count. A bare resume
    /// continues the manifest's plan rather than report the run complete
    /// against a total the manifest does not carry.
    #[tokio::test]
    async fn a_bare_resume_continues_the_plan_the_manifest_recorded() {
        let dir = temp_dir("resume-plan");
        run_stopped(&dir, 4242, 3).await;

        let paths = crate::trace::TaskPaths::new(&dir);
        let mut manifest: RunManifest =
            crate::trace::read_json(paths.manifest()).expect("manifest reads");
        assert_eq!(manifest.cycles_completed, 3);
        manifest.cycles_planned = 6;
        crate::trace::write_json(paths.manifest(), &manifest).expect("manifest writes");

        let config = TaskConfig::load(paths.config()).expect("recorded config loads");
        assert_eq!(config.design.cycles, 3, "the config still plans three");
        let (scaffold, scope) = config.resolve().expect("resolves");
        let backend = Arc::new(FixtureBackend::planted(&scaffold, &scope, 4242));
        let mut orchestrator = Orchestrator::resume(
            config,
            backend,
            Box::new(HeuristicAgent::new(4242)),
            paths.clone(),
            None,
        )
        .await
        .expect("resumes");
        let summary = orchestrator.run().await.expect("runs");
        assert_eq!(summary.cycles_completed, 6);

        let manifest: RunManifest =
            crate::trace::read_json(paths.manifest()).expect("manifest reads");
        assert_eq!(manifest.cycles_planned, 6);
        assert_eq!(manifest.cycles_completed, 6);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Wraps the fixture and reports a different service version, the way a
    /// service upgraded between the recorded run and the resume would.
    struct UpgradedBackend {
        inner: FixtureBackend,
        version: String,
    }

    #[async_trait::async_trait]
    impl FoldBackend for UpgradedBackend {
        fn name(&self) -> &str {
            "upgraded-fixture"
        }

        async fn health(&self) -> Result<crate::contract::Health, FoldError> {
            let mut health = self.inner.health().await?;
            health.version = self.version.clone();
            Ok(health)
        }

        async fn fold(
            &self,
            job: crate::contract::JobSpec,
        ) -> Result<crate::contract::FoldResult, FoldError> {
            self.inner.fold(job).await
        }
    }

    /// `folds`, `usage` and `best_candidate` span both runs, so the manifest
    /// must not name the resumed run's service for the whole result. The
    /// original fields keep describing the run that produced them, and the
    /// resume's health-check values land in the resume fields.
    #[tokio::test]
    async fn a_resume_records_its_service_version_beside_the_original() {
        let dir = temp_dir("resume-service-version");
        run_stopped(&dir, 4242, 2).await;

        let paths = crate::trace::TaskPaths::new(&dir);
        let before: RunManifest =
            crate::trace::read_json(paths.manifest()).expect("manifest reads");

        let config = TaskConfig::load(paths.config()).expect("recorded config loads");
        let (scaffold, scope) = config.resolve().expect("resolves");
        let backend = Arc::new(UpgradedBackend {
            inner: FixtureBackend::planted(&scaffold, &scope, 4242),
            version: "9.9.9-test".to_string(),
        });
        let mut orchestrator = Orchestrator::resume(
            config,
            backend,
            Box::new(HeuristicAgent::new(4242)),
            paths.clone(),
            Some(6),
        )
        .await
        .expect("resumes");
        orchestrator.run().await.expect("runs");

        let after: RunManifest = crate::trace::read_json(paths.manifest()).expect("manifest reads");
        assert_eq!(after.protocol, before.protocol);
        assert_eq!(after.service_version, before.service_version);
        assert_eq!(after.resumed_protocol, Some(before.protocol.clone()));
        assert_eq!(
            after.resumed_service_version,
            Some("9.9.9-test".to_string())
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A truncated artifact exists, so the refusal must carry the read error
    /// rather than claim there is no such file.
    #[tokio::test]
    async fn a_corrupt_artifact_is_reported_as_corrupt_not_missing() {
        let dir = temp_dir("resume-corrupt");
        run_stopped(&dir, 4242, 2).await;
        let paths = crate::trace::TaskPaths::new(&dir);
        let manifest: RunManifest =
            crate::trace::read_json(paths.manifest()).expect("manifest reads");

        std::fs::write(paths.manifest(), r#"{"task_id": "trunc"#).expect("write");
        let error = refused_resume(&dir, 4242).await;
        match &error {
            RunError::NoManifest { detail, .. } => {
                let message = error.to_string();
                assert!(message.contains("run.json"), "{message}");
                assert!(!detail.is_empty(), "{message}");
                assert!(message.contains(detail.as_str()), "{message}");
            }
            other => panic!("expected a run.json read failure, got {other:?}"),
        }

        crate::trace::write_json(paths.manifest(), &manifest).expect("manifest writes");
        std::fs::write(paths.population(), r#"{"members": ["#).expect("write");
        let error = refused_resume(&dir, 4242).await;
        match &error {
            RunError::NoSnapshot { detail, .. } => {
                let message = error.to_string();
                assert!(message.contains("population.json"), "{message}");
                assert!(!detail.is_empty(), "{message}");
                assert!(message.contains(detail.as_str()), "{message}");
            }
            other => panic!("expected a population.json read failure, got {other:?}"),
        }

        std::fs::remove_dir_all(&dir).ok();
    }
}
