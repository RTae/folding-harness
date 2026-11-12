//! Running a design cycle, and the policy benchmark.
use super::*;

pub(super) async fn run(
    task: PathBuf,
    cycles: Option<u32>,
    seed: Option<u64>,
    url: Option<String>,
    task_root: Option<PathBuf>,
    open_board: bool,
    latency: Duration,
) -> Result<()> {
    let mut config =
        TaskConfig::load(&task).with_context(|| format!("loading {}", task.display()))?;
    if let Some(cycles) = cycles {
        config.design.cycles = cycles;
    }
    if let Some(seed) = seed {
        config.design.seed = seed;
    }
    if let Some(url) = url {
        config.fold.url = Some(url);
        config.fold.execution_mode = ExecutionMode::Service;
    }

    run_config(config, task_root, open_board, latency, None).await
}

/// Prepare and run an already-resolved config. Shared by `fh run`, `fh task`
/// and the chat screen, so all three behave identically once a plan exists.
pub(super) async fn run_config(
    config: TaskConfig,
    task_root: Option<PathBuf>,
    open_board: bool,
    latency: Duration,
    chat_id: Option<String>,
) -> Result<()> {
    let (scaffold, scope) = config.resolve().context("resolving the scaffold")?;
    let backend = backend_for(&config, &scaffold, &scope, latency)?;
    let agent: Box<dyn DesignAgent> = Box::new(HeuristicAgent::new(config.design.seed));

    let task_id = TaskId::new(&config.name);
    let paths = match task_root {
        Some(root) => TaskPaths::new(root.join(&task_id.0)),
        None => TaskPaths::under_default_root(&task_id.0),
    };

    println!("task {} -> {}", task_id, paths.root().display());
    println!("scope: {}", scope.summary());

    let orchestrator = Orchestrator::prepare(
        config,
        Arc::clone(&backend),
        agent,
        paths.clone(),
        task_id.clone(),
    )
    .await
    .context("preparing the run")?
    .attributed_to(chat_id.clone())
    .context("recording the chat this run came from")?;

    // Both directions of the link, so either record finds the other.
    if let Some(chat_id) = &chat_id {
        let root = fh::history::chat_root();
        if let Err(error) = fh::history::record_run(&root, chat_id, &task_id.0) {
            println!("note: could not link this run to chat {chat_id}: {error}");
        }
    }

    drive(orchestrator, paths, open_board).await
}

/// Continue a recorded run from its last completed cycle.
///
/// The config comes from the task.yaml saved in the task directory, and
/// `Orchestrator::resume` refuses a run whose digest does not match what
/// run.json recorded. `--cycles` extends the planned total without touching
/// the config, so extending is not a config change.
pub(super) async fn resume(
    path: Option<PathBuf>,
    task: Option<String>,
    cycles: Option<u32>,
    open_board: bool,
    latency: Duration,
) -> Result<()> {
    let dir = resolve_task_dir(path, task)?;
    let config = TaskConfig::load(dir.config())
        .with_context(|| format!("loading the recorded config {}", dir.config().display()))?;
    let (scaffold, scope) = config.resolve().context("resolving the scaffold")?;
    let backend = backend_for(&config, &scaffold, &scope, latency)?;
    let agent: Box<dyn DesignAgent> = Box::new(HeuristicAgent::new(config.design.seed));

    println!("resuming {}", dir.root().display());
    println!("scope: {}", scope.summary());

    let orchestrator = Orchestrator::resume(config, backend, agent, dir.clone(), cycles)
        .await
        .context("resuming the run")?;

    drive(orchestrator, dir, open_board).await
}

/// Run a prepared orchestrator, streaming events to stdout or under the
/// dashboard. Shared by `fh run` and `fh resume`, so both behave identically
/// once the orchestrator exists.
async fn drive(orchestrator: Orchestrator, paths: TaskPaths, open_board: bool) -> Result<()> {
    if open_board {
        // stdout belongs to the dashboard, so no event sink here.
        let mut orchestrator = orchestrator;
        let run_handle = tokio::spawn(async move { orchestrator.run().await });

        let board_paths = paths.clone();
        let board_handle = tokio::task::spawn_blocking(move || {
            fh::tui::run(board_paths, Duration::from_millis(400))
        });

        let board_result = board_handle.await.context("dashboard thread")?;

        if run_handle.is_finished() {
            match run_handle.await.context("run task")? {
                Ok(summary) => report(&summary),
                Err(error) => return Err(anyhow!("run failed: {error}")),
            }
        } else {
            run_handle.abort();
            println!(
                "dashboard closed while the run was still going; it was stopped. Partial results: {}",
                paths.root().display()
            );
        }
        board_result.context("dashboard")?;
        return Ok(());
    }

    let mut orchestrator = orchestrator.on_event(Box::new(|record| {
        println!(
            "{:>5} {:<16} {}",
            record.seq,
            record.event.kind(),
            record.event.summary()
        );
    }));

    let summary = orchestrator.run().await.context("running the task")?;
    report(&summary);
    Ok(())
}

pub(super) async fn bench(
    task: Option<PathBuf>,
    seeds: u64,
    cycles: u32,
    policy: Vec<String>,
    json: Option<PathBuf>,
) -> Result<()> {
    use fh::bench::{self, Policy};

    let config = match &task {
        Some(path) => {
            TaskConfig::load(path).with_context(|| format!("loading {}", path.display()))?
        }
        None => TaskConfig::from_yaml(EXAMPLE_TASK).context("the built-in example")?,
    };

    let policies: Vec<Policy> = if policy.is_empty() {
        Policy::ALL.to_vec()
    } else {
        policy
            .iter()
            .map(|name| {
                Policy::parse(name)
                    .ok_or_else(|| anyhow!("no policy '{name}'; choose from: {}", Policy::names()))
            })
            .collect::<Result<_>>()?
    };

    let seed_list: Vec<u64> = (0..seeds.max(1)).map(|index| 1000 + index).collect();
    let root = std::env::temp_dir().join(format!("fh-bench-{}", fh::trace::now_ms()));

    println!(
        "{} policies x {} seeds x {} cycles, against the fixture's planted optimum",
        policies.len(),
        seed_list.len(),
        cycles
    );
    println!("recovery is the share of planted positions the best candidate found");
    println!();

    let total = policies.len() * seed_list.len();
    let mut done = 0usize;
    // Overwrite one line on a terminal; print a line each on a pipe, where
    // carriage returns would run the whole sweep together.
    let interactive = std::io::IsTerminal::is_terminal(&std::io::stdout());
    let outcomes = bench::sweep(&config, &policies, &seed_list, cycles, &root, |outcome| {
        done += 1;
        use std::io::Write;
        if interactive {
            print!(
                "\r  {done}/{total}  {} seed {} recovery {:.0}%          ",
                outcome.policy,
                outcome.seed,
                outcome.recovery * 100.0
            );
            std::io::stdout().flush().ok();
        } else {
            println!(
                "  {done}/{total}  {} seed {} recovery {:.0}%",
                outcome.policy,
                outcome.seed,
                outcome.recovery * 100.0
            );
        }
    })
    .await
    .map_err(|error| anyhow!("{error}"))?;
    if interactive {
        println!("\r{:<62}", "");
    } else {
        println!();
    }

    println!(
        "{:<14} {:>5} {:>10} {:>10} {:>10} {:>8} {:>12}",
        "policy", "runs", "recovery", "worst", "best", "folds", "per 1k folds"
    );
    let mut baseline: Option<f64> = None;
    for policy in &policies {
        let mine: Vec<_> = outcomes
            .iter()
            .filter(|outcome| outcome.policy == policy.as_str())
            .cloned()
            .collect();
        let summary = bench::summarise(*policy, &mine);
        if *policy == Policy::Random {
            baseline = Some(summary.mean_recovery);
        }
        println!(
            "{:<14} {:>5} {:>9.1}% {:>9.1}% {:>9.1}% {:>8.0} {:>12.2}",
            summary.policy,
            summary.runs,
            summary.mean_recovery * 100.0,
            summary.worst_recovery * 100.0,
            summary.best_recovery * 100.0,
            summary.mean_folds,
            summary.recovery_per_1k_folds
        );
    }

    if let Some(random) = baseline {
        let heuristic = outcomes
            .iter()
            .filter(|outcome| outcome.policy == "heuristic")
            .map(|outcome| outcome.recovery)
            .collect::<Vec<_>>();
        if !heuristic.is_empty() {
            let mean = heuristic.iter().sum::<f64>() / heuristic.len() as f64;
            println!();
            if random <= f64::EPSILON {
                println!("the random control recovered nothing, so there is no ratio to report");
            } else {
                println!(
                    "the shipped policy recovers {:.2}x what a uniform draw does",
                    mean / random
                );
            }
            if mean <= random {
                println!("it is not beating the control, which is worth looking into");
            }
        }
    }

    if let Some(path) = json {
        fh::trace::write_json(&path, &outcomes)
            .with_context(|| format!("writing {}", path.display()))?;
        println!();
        println!("raw outcomes written to {}", path.display());
    }

    std::fs::remove_dir_all(&root).ok();
    println!();
    println!("This measures orchestration against a synthetic landscape.");
    println!("It says nothing about whether a design would bind.");
    Ok(())
}
