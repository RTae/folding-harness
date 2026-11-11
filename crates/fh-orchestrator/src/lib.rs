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

use std::sync::Arc;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;
use tokio::task::JoinSet;

use fh_agent::{AgentError, DesignAgent};
use fh_contract::FoldResult;
use fh_core::{
    AdmissionOutcome, Candidate, CandidateId, ConfigError, MutableScope, Population, Proposal,
    ProposalError, RejectReason, Scaffold, SearchHistory, SkillId, TaskConfig, TaskId,
};
use fh_fold::{job_spec_for, metrics_from, FoldBackend, FoldError};
use fh_skills::{executor, DesignSkill, Legality, Router, SkillContext};
use fh_trace::{
    now_ms, Event, PopulationSnapshot, RunManifest, StructureSnapshot, TaskPaths, TraceRecord,
    TraceWriter,
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
    rng: StdRng,
    task_id: TaskId,
    next_id: u32,
    folds: usize,
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
            config_sha256: fh_trace::sha256_hex(config_yaml.as_bytes()),
            cycles_planned: config.design.cycles,
            cycles_completed: 0,
            folds: 0,
            best_candidate: None,
            best_loss: None,
        };
        fh_trace::write_json(paths.manifest(), &manifest)?;

        let trace = TraceWriter::create(paths.trace())?;
        let population = Population::new(config.population.clone());
        let rng = StdRng::seed_from_u64(config.design.seed);

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
            on_event: None,
        })
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

    pub async fn run(&mut self) -> Result<RunSummary, RunError> {
        let started = Instant::now();

        self.emit(Event::TaskStarted {
            task_id: self.task_id.0.clone(),
            name: self.config.name.clone(),
            backend: self.backend.name().to_string(),
            model: self.config.fold.model.clone(),
            protocol: self.manifest.protocol.clone(),
            cycles: self.config.design.cycles,
            seed: self.config.design.seed,
            scope: self.scope.summary(),
            gates: format!("{:?}", self.config.gates),
        })?;

        match self.seed_founder().await {
            Ok(()) => {}
            Err(error) => {
                self.emit(Event::TaskFailed {
                    cycle: 0,
                    detail: error.to_string(),
                })?;
                return Err(error);
            }
        }

        for cycle in 1..=self.config.design.cycles {
            if let Err(error) = self.run_cycle(cycle).await {
                self.emit(Event::TaskFailed {
                    cycle,
                    detail: error.to_string(),
                })?;
                return Err(error);
            }
            self.manifest.cycles_completed = cycle;
            self.snapshot(cycle, true)?;
        }

        let best = self.population.best().cloned();
        let elapsed_ms = started.elapsed().as_millis() as u64;

        self.emit(Event::TaskFinished {
            cycles_completed: self.manifest.cycles_completed,
            folds: self.folds,
            best: best.as_ref().map(|candidate| candidate.id),
            best_loss: best.as_ref().and_then(|candidate| candidate.loss),
            elapsed_ms,
        })?;

        self.manifest.finished_at_ms = Some(now_ms());
        self.manifest.folds = self.folds;
        self.manifest.best_candidate = best.as_ref().map(|candidate| candidate.id);
        self.manifest.best_loss = best.as_ref().and_then(|candidate| candidate.loss);
        fh_trace::write_json(self.paths.manifest(), &self.manifest)?;
        self.snapshot(self.manifest.cycles_completed, false)?;

        Ok(RunSummary {
            task_id: self.task_id.clone(),
            root: self.paths.root().to_path_buf(),
            cycles_completed: self.manifest.cycles_completed,
            folds: self.folds,
            best,
            elapsed_ms,
        })
    }

    /// Fold the input sequence as cycle 0 and admit it.
    ///
    /// The founder is admitted even if it would fail the gates: it is the
    /// user's starting point, and refusing it would leave the search with no
    /// parent at all. A note records the gate it failed, so nothing is hidden.
    async fn seed_founder(&mut self) -> Result<(), RunError> {
        let id = self.next_candidate_id();
        let founder = Candidate::founder(id, fh_core::BinderSequence::from_scaffold(&self.scaffold));

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

        self.emit(Event::FoldFinished {
            cycle: 0,
            candidate: id,
            elapsed_ms: result.elapsed_ms,
        })?;

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

        self.history
            .note_proposals(decision.skill, proposals.len());
        self.emit(Event::ProposalsGenerated {
            cycle,
            skill: decision.skill.to_string(),
            count: proposals.len(),
        })?;

        let candidates = self.materialise(cycle, &proposals, &parent)?;
        let results = self.fold_all(cycle, &candidates).await?;

        let mut admitted = 0usize;
        let mut rejected = 0usize;

        for (candidate, outcome) in candidates.into_iter().zip(results.into_iter()) {
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
                cycle,
                decision.skill,
                self.config.design.batch_size,
                admitted,
                rejected,
                stats.best_loss,
                stats.mean_loss,
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

    fn write_structure(
        &self,
        candidate: &Candidate,
        result: &FoldResult,
    ) -> Result<(), RunError> {
        if result.structure.ca_trace.is_empty() {
            return Ok(());
        }
        let snapshot = StructureSnapshot {
            candidate: candidate.id,
            cycle: candidate.cycle,
            loss: candidate.loss,
            ca_trace: result.structure.ca_trace.clone(),
        };
        fh_trace::write_json(self.paths.structure(candidate.id), &snapshot)?;
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
        };
        fh_trace::write_json(self.paths.population(), &snapshot)?;
        if let Some(best) = self.population.best() {
            fh_trace::write_json(self.paths.best(), best)?;
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
pub fn routing_table(router: &Router, state: &fh_core::SearchState) -> Vec<(SkillId, Legality)> {
    router.assess_all(state)
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

/// Re-exported so callers do not need `fh_core::RejectReason` directly.
pub use fh_core::RejectReason as Rejection;
#[allow(unused_imports)]
use RejectReason as _;
