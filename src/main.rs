//! `fh` — the harness command line.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};

use fh::agent::{DesignAgent, HeuristicAgent, ModelProvider};
use fh::fold::{FixtureBackend, FoldBackend, HttpBackend};
use fh::orchestrator::Orchestrator;
use fh::plan::Chain as PlanChain;
use fh::provider::AnthropicProvider;
use fh::skills::{executor, Router};
use fh::trace::{RunManifest, TaskPaths};
use fh::{Capability, ExecutionMode, SearchState, SkillId, TaskConfig, TaskId};

const EXAMPLE_TASK: &str = include_str!("../examples/vhh_demo.yaml");

#[derive(Parser)]
#[command(
    name = "fh",
    version,
    about = "A design harness: orchestration, gates and inspection around a folding backend",
    long_about = "fh runs a design cycle over a folding backend and records everything it did.\n\
                  The default backend is a deterministic fixture, so `fh run` works with no GPU."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Where to send prompts, and how to shape the request.
///
/// A custom `--url` is for a proxy, gateway or self-hosted endpoint. The
/// harness prints the host before sending anything, and drops the
/// Anthropic-specific request fields, which a gateway may not accept.
#[derive(clap::Args, Debug, Clone)]
struct ProviderOptions {
    /// Endpoint base URL, e.g. https://gateway.example.com. Defaults to
    /// $ANTHROPIC_BASE_URL, else the first-party API.
    #[arg(long)]
    url: Option<String>,
    /// Model name. Defaults to $ANTHROPIC_MODEL, else claude-opus-5-5. A
    /// custom endpoint usually needs this set to something it serves.
    #[arg(long)]
    model: Option<String>,
    /// `anthropic-version` header, for an endpoint that wants an older one
    /// (e.g. 2023-06-01). Defaults to $ANTHROPIC_API_VERSION.
    #[arg(long)]
    api_version: Option<String>,
    /// Send output_config.effort: low, medium, high, xhigh or max. Omitted by
    /// default on a custom endpoint; set it if the endpoint supports it.
    #[arg(long)]
    effort: Option<String>,
}

#[derive(Subcommand)]
enum Command {
    /// Write an example task config.
    Init {
        #[arg(default_value = "task.yaml")]
        path: PathBuf,
        /// Overwrite an existing file.
        #[arg(long)]
        force: bool,
    },
    /// Check a task config and print the mutable scope and routing table.
    Validate { task: PathBuf },
    /// Run a design task.
    Run {
        task: PathBuf,
        /// Override design.cycles.
        #[arg(long)]
        cycles: Option<u32>,
        /// Override design.seed.
        #[arg(long)]
        seed: Option<u64>,
        /// Point at a compute service, implying execution_mode: service.
        #[arg(long)]
        url: Option<String>,
        /// Parent directory for the task folder.
        #[arg(long)]
        task_root: Option<PathBuf>,
        /// Open the dashboard instead of streaming events to stdout.
        #[arg(long)]
        board: bool,
        /// Artificial per-fold delay for the fixture backend, to watch a run.
        #[arg(long, default_value_t = 0)]
        latency_ms: u64,
    },
    /// Chat with a model to set up a run, see the validated plan, then confirm.
    Chat {
        /// Binder chain: a FASTA path, or a bare sequence. Repeatable.
        #[arg(long, required = true)]
        binder: Vec<String>,
        /// Target chain: a FASTA path, or a bare sequence. Repeatable.
        #[arg(long, required = true)]
        target: Vec<String>,
        #[arg(long, default_value_t = 20261111)]
        seed: u64,
        #[command(flatten)]
        provider: ProviderOptions,
        #[arg(long)]
        task_root: Option<PathBuf>,
        /// Artificial per-fold delay once a confirmed run starts.
        #[arg(long, default_value_t = 0)]
        latency_ms: u64,
    },
    /// One-shot version of `chat`: describe the run in prose on the command line.
    Task {
        /// What you want, in prose.
        prose: String,
        #[arg(long, required = true)]
        binder: Vec<String>,
        #[arg(long, required = true)]
        target: Vec<String>,
        #[arg(long, default_value_t = 20261111)]
        seed: u64,
        #[command(flatten)]
        provider: ProviderOptions,
        /// Write the plan here instead of task.yaml.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Start the run after the plan is confirmed.
        #[arg(long)]
        run: bool,
        /// Skip the confirmation prompt.
        #[arg(long)]
        yes: bool,
        #[arg(long)]
        task_root: Option<PathBuf>,
    },
    /// Open the dashboard on a task directory.
    Board {
        /// Task directory. Defaults to the most recent task.
        #[arg(long)]
        path: Option<PathBuf>,
        /// Task id under the default task root.
        #[arg(long)]
        task: Option<String>,
        #[arg(long, default_value_t = 400)]
        tick_ms: u64,
    },
    /// Serve a stub compute service speaking the fold contract.
    Serve {
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Plant the fixture's optimum from this task, so the service produces
        /// a search landscape rather than flat scores.
        #[arg(long)]
        task: Option<PathBuf>,
        #[arg(long, default_value_t = 0)]
        latency_ms: u64,
    },
    /// Check the local setup, and a task config and its backend if given.
    Doctor {
        task: Option<PathBuf>,
        /// Also send one tiny request to the model endpoint to prove the URL,
        /// key and model work together. This costs a few tokens.
        #[arg(long)]
        probe_model: bool,
        #[command(flatten)]
        provider: ProviderOptions,
    },
    /// List the design skills this build knows about.
    Skills,
    /// List recorded runs.
    Ls,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Init { path, force } => init(path, force),
        Command::Validate { task } => validate(task),
        Command::Run {
            task,
            cycles,
            seed,
            url,
            task_root,
            board,
            latency_ms,
        } => {
            run(
                task,
                cycles,
                seed,
                url,
                task_root,
                board,
                Duration::from_millis(latency_ms),
            )
            .await
        }
        Command::Chat {
            binder,
            target,
            seed,
            provider,
            task_root,
            latency_ms,
        } => {
            chat(
                binder,
                target,
                seed,
                provider,
                task_root,
                Duration::from_millis(latency_ms),
            )
            .await
        }
        Command::Task {
            prose,
            binder,
            target,
            seed,
            provider,
            out,
            run,
            yes,
            task_root,
        } => {
            task_from_prose(
                prose, binder, target, seed, provider, out, run, yes, task_root,
            )
            .await
        }
        Command::Board {
            path,
            task,
            tick_ms,
        } => board(path, task, Duration::from_millis(tick_ms)),
        Command::Serve {
            port,
            task,
            latency_ms,
        } => serve(port, task, Duration::from_millis(latency_ms)).await,
        Command::Doctor {
            task,
            probe_model,
            provider,
        } => doctor(task, probe_model, provider).await,
        Command::Skills => {
            skills();
            Ok(())
        }
        Command::Ls => list_runs(),
    }
}

fn init(path: PathBuf, force: bool) -> Result<()> {
    if path.exists() && !force {
        return Err(anyhow!(
            "{} already exists; pass --force to overwrite it",
            path.display()
        ));
    }
    std::fs::write(&path, EXAMPLE_TASK).with_context(|| format!("writing {}", path.display()))?;
    println!("wrote {}", path.display());
    println!("next: fh validate {}", path.display());
    Ok(())
}

fn validate(task: PathBuf) -> Result<()> {
    let config = TaskConfig::load(&task).with_context(|| format!("loading {}", task.display()))?;
    let (scaffold, scope) = config.resolve().context("resolving the scaffold")?;

    println!("task        {}", config.name);
    println!("topology    {}", scaffold.topology.as_str());
    for chain in &scaffold.chains {
        println!(
            "chain {:<4} {:>4} residues  {:?}",
            chain.label,
            chain.len(),
            chain.role
        );
    }
    println!();
    println!("scope       {}", scope.summary());
    for region in scope.regions() {
        let mutable = scope.region_positions(&region.name).len();
        let pinned = region.len() - mutable;
        println!(
            "  {:<10} residues {}..{} ({} mutable{})",
            region.name,
            region.start + 1,
            region.end,
            mutable,
            if pinned > 0 {
                format!(", {pinned} pinned")
            } else {
                String::new()
            }
        );
    }

    println!();
    println!(
        "search      {} cycles x {} proposals, seed {}, concurrency {}",
        config.design.cycles,
        config.design.batch_size,
        config.design.seed,
        config.design.concurrency
    );
    println!(
        "population  capacity {}, min hamming {}, parent temperature {}",
        config.population.capacity,
        config.population.min_hamming,
        config.population.parent_temperature
    );
    println!(
        "fold        {} via {:?}{}",
        config.fold.model,
        config.fold.execution_mode,
        config
            .fold
            .url
            .as_ref()
            .map(|url| format!(" at {url}"))
            .unwrap_or_default()
    );

    // Routing is shown for a service that offers folding only, since checking
    // the real capability set means connecting to one.
    println!();
    println!("routing (assuming a fold-only service, population 1):");
    let router = Router::new(config.design.skills.iter().copied(), [Capability::Fold]);
    let mut state = SearchState::bootstrap();
    state.population_size = 1;
    for (skill, verdict) in router.assess_all(&state) {
        let mark = if verdict.is_legal() { "+" } else { "-" };
        println!("  {mark} {:<18} {}", skill.to_string(), verdict.reason());
    }

    println!();
    println!("config is valid");
    Ok(())
}

fn backend_for(
    config: &TaskConfig,
    scaffold: &fh::Scaffold,
    scope: &fh::MutableScope,
    latency: Duration,
) -> Result<Arc<dyn FoldBackend>> {
    match config.fold.execution_mode {
        ExecutionMode::Fixture => Ok(Arc::new(
            FixtureBackend::planted(scaffold, scope, config.design.seed).with_latency(latency),
        )),
        ExecutionMode::Service => {
            let url = config
                .fold
                .url
                .clone()
                .ok_or_else(|| anyhow!("fold.url is required for execution_mode: service"))?;
            Ok(Arc::new(HttpBackend::new(
                url,
                config.fold.token.clone(),
                config.fold.poll_interval_secs,
                config.fold.timeout_secs,
            )?))
        }
    }
}

async fn run(
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

    run_config(config, task_root, open_board, latency).await
}

/// Prepare and run an already-resolved config. Shared by `fh run`, `fh task`
/// and the chat screen, so all three behave identically once a plan exists.
async fn run_config(
    config: TaskConfig,
    task_root: Option<PathBuf>,
    open_board: bool,
    latency: Duration,
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
    .context("preparing the run")?;

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

fn report(summary: &fh::orchestrator::RunSummary) {
    println!();
    println!(
        "{} cycles, {} folds, {:.1}s",
        summary.cycles_completed,
        summary.folds,
        summary.elapsed_ms as f64 / 1000.0
    );
    match &summary.best {
        Some(best) => {
            println!("best {}", best.summary());
            if let Some(metrics) = &best.metrics {
                println!(
                    "     ipTM {:.3}  interface pLDDT {:.1}  PAE {:.1}  contacts {}  clashes {}",
                    metrics.iptm,
                    metrics.interface_plddt,
                    metrics.pae_interface,
                    metrics.interface_contacts,
                    metrics.clashes
                );
            }
            for (chain, sequence) in best.sequence.iter() {
                println!("     {chain}: {sequence}");
            }
        }
        None => println!("no candidate survived the gates"),
    }
    println!();
    println!("results {}", summary.root.display());
    println!("inspect fh board --path {}", summary.root.display());
    println!(
        "         (numbers above come from the configured backend; a fixture run predicts nothing)"
    );
}

/// Load chains from FASTA paths or bare sequences. The first unlabelled chain
/// gets `default_label`, so `--binder QVQL...` works without a FASTA header.
fn load_chains(values: &[String], default_label: &str) -> Result<Vec<PlanChain>> {
    let mut out = Vec::new();
    for (index, value) in values.iter().enumerate() {
        let label = if index == 0 {
            default_label.to_string()
        } else {
            format!("{default_label}{}", index + 1)
        };
        let records = fh::fasta::load_or_literal(value, &label)
            .with_context(|| format!("reading chain from '{value}'"))?;
        for record in records {
            out.push(PlanChain::new(record.label, record.sequence));
        }
    }
    Ok(out)
}

/// Build the provider from the environment, then apply the flags.
///
/// Order matters: `with_base_url` resets the Anthropic-specific extras, so an
/// explicit `--effort` after it still wins.
fn make_provider(options: &ProviderOptions) -> Result<AnthropicProvider> {
    let mut provider = AnthropicProvider::from_env().map_err(|error| anyhow!("{error}"))?;
    if let Some(url) = &options.url {
        provider = provider.with_base_url(url.clone());
    }
    if let Some(model) = &options.model {
        provider = provider.with_model(model.clone());
    }
    if let Some(version) = &options.api_version {
        provider = provider.with_api_version(version.clone());
    }
    if let Some(effort) = &options.effort {
        provider = provider.with_effort(Some(effort.clone()));
    }
    Ok(provider)
}

fn confirm(question: &str) -> Result<bool> {
    use std::io::Write;
    print!("{question} [y/N] ");
    std::io::stdout().flush()?;
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    Ok(matches!(
        line.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

async fn chat(
    binder: Vec<String>,
    target: Vec<String>,
    seed: u64,
    options: ProviderOptions,
    task_root: Option<PathBuf>,
    latency: Duration,
) -> Result<()> {
    let binder_chains = load_chains(&binder, "H")?;
    let target_chains = load_chains(&target, "A")?;
    let built = make_provider(&options)?;
    println!("model {}", built.describe());
    let provider: Arc<dyn ModelProvider> = Arc::new(built);

    // The screen runs on a blocking thread so the runtime stays free to make
    // the model call while the interface keeps drawing.
    let runtime = tokio::runtime::Handle::current();
    let confirmed = tokio::task::spawn_blocking(move || {
        fh::tui::chat::run(provider, runtime, binder_chains, target_chains, seed)
    })
    .await
    .context("chat thread")?
    .context("running the chat screen")?;

    match confirmed {
        None => {
            println!("no plan confirmed; nothing ran");
            Ok(())
        }
        Some(config) => {
            println!("running the confirmed plan: {}", config.name);
            run_config(config, task_root, true, latency).await
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn task_from_prose(
    prose: String,
    binder: Vec<String>,
    target: Vec<String>,
    seed: u64,
    options: ProviderOptions,
    out: Option<PathBuf>,
    start: bool,
    yes: bool,
    task_root: Option<PathBuf>,
) -> Result<()> {
    let binder_chains = load_chains(&binder, "H")?;
    let target_chains = load_chains(&target, "A")?;
    let provider = make_provider(&options)?;

    println!("model {}", provider.describe());
    println!("asking for a plan...");
    let plan = fh::plan::draft(&provider, &prose, &binder_chains, &target_chains, seed)
        .await
        .map_err(|error| anyhow!("{error}"))?;

    println!();
    for line in plan.summary() {
        println!("{line}");
    }
    if let Some(notes) = &plan.config.notes {
        println!();
        println!("notes       {notes}");
    }
    println!();
    println!("A prediction is not a measurement; results need experimental validation.");
    println!();

    let destination = out.unwrap_or_else(|| PathBuf::from("task.yaml"));
    let yaml = plan.config.to_yaml().context("serialising the plan")?;
    std::fs::write(&destination, yaml)
        .with_context(|| format!("writing {}", destination.display()))?;
    println!("plan written to {}", destination.display());

    if !start {
        println!("run it with: fh run {}", destination.display());
        return Ok(());
    }
    if !yes && !confirm("Start this run?")? {
        println!("not started");
        return Ok(());
    }

    run_config(plan.config, task_root, false, Duration::ZERO).await
}

fn resolve_task_dir(path: Option<PathBuf>, task: Option<String>) -> Result<TaskPaths> {
    if let Some(path) = path {
        return Ok(TaskPaths::new(path));
    }
    if let Some(task) = task {
        return Ok(TaskPaths::under_default_root(&task));
    }
    let root = TaskPaths::default_root();
    let found = TaskPaths::list(&root).with_context(|| format!("listing {}", root.display()))?;
    found.into_iter().next().ok_or_else(|| {
        anyhow!(
            "no runs under {}; run one first with `fh run`",
            root.display()
        )
    })
}

fn board(path: Option<PathBuf>, task: Option<String>, tick: Duration) -> Result<()> {
    let paths = resolve_task_dir(path, task)?;
    if !paths.root().exists() {
        return Err(anyhow!("{} does not exist", paths.root().display()));
    }
    fh::tui::run(paths, tick).context("running the dashboard")
}

async fn serve(port: u16, task: Option<PathBuf>, latency: Duration) -> Result<()> {
    let backend: Arc<dyn FoldBackend> = match task {
        Some(task) => {
            let config =
                TaskConfig::load(&task).with_context(|| format!("loading {}", task.display()))?;
            let (scaffold, scope) = config.resolve()?;
            println!(
                "planting the fixture optimum from {} (seed {})",
                task.display(),
                config.design.seed
            );
            Arc::new(
                FixtureBackend::planted(&scaffold, &scope, config.design.seed)
                    .with_latency(latency),
            )
        }
        None => {
            println!("no --task given: the fixture will return flat scores");
            Arc::new(FixtureBackend::default().with_latency(latency))
        }
    };

    let addr: SocketAddr = ([127, 0, 0, 1], port).into();
    println!("serving {} on http://{addr}", fh::contract::PROTOCOL);
    println!("  GET  http://{addr}{}", fh::contract::routes::HEALTH);
    println!("  POST http://{addr}{}", fh::contract::routes::FOLD);
    println!();
    println!("point a task at it with: fh run <task.yaml> --url http://{addr}");

    fh::fold::server::serve(addr, backend)
        .await
        .map_err(|error| anyhow!("serving: {error}"))
}

async fn doctor(task: Option<PathBuf>, probe_model: bool, options: ProviderOptions) -> Result<()> {
    println!("fh          {}", env!("CARGO_PKG_VERSION"));
    println!("protocol    {}", fh::contract::PROTOCOL);

    let root = TaskPaths::default_root();
    println!("task root   {}", root.display());
    match std::fs::create_dir_all(&root) {
        Ok(()) => println!("            writable"),
        Err(error) => println!("            NOT writable: {error}"),
    }

    if probe_model {
        println!();
        match make_provider(&options) {
            Err(error) => {
                println!("model       NOT CONFIGURED");
                println!("            {error}");
                return Err(anyhow!("model provider not configured"));
            }
            Ok(provider) => {
                println!("model       {}", provider.describe());
                println!("            version {}", provider.api_version());
                print!("            probing ... ");
                use std::io::Write;
                std::io::stdout().flush().ok();
                match provider.probe().await {
                    Ok(reply) => {
                        println!("ok: {}", reply.trim().chars().take(60).collect::<String>())
                    }
                    Err(error) => {
                        println!("FAILED");
                        println!("            {error}");
                        return Err(anyhow!("model probe failed"));
                    }
                }
            }
        }
    }

    let Some(task) = task else {
        println!();
        println!("pass a task config to check it too: fh doctor task.yaml");
        return Ok(());
    };

    println!();
    match TaskConfig::load(&task) {
        Err(error) => {
            println!("config      INVALID: {error}");
            return Err(anyhow!("config did not load"));
        }
        Ok(config) => {
            println!("config      {} parses", task.display());
            match config.resolve() {
                Err(error) => {
                    println!("scaffold    INVALID: {error}");
                    return Err(anyhow!("scaffold did not resolve"));
                }
                Ok((scaffold, scope)) => {
                    println!("scaffold    ok, {}", scope.summary());
                    let backend = backend_for(&config, &scaffold, &scope, Duration::ZERO)?;
                    print!("backend     {} ... ", backend.name());
                    match backend.health().await {
                        Ok(health) => {
                            println!("ok");
                            println!(
                                "            {} {} protocol {}",
                                health.service, health.version, health.protocol
                            );
                            println!(
                                "            capabilities {}",
                                health.capabilities.join(", ")
                            );
                            if !health.gpus.is_empty() {
                                println!("            {} gpu(s) reported", health.gpus.len());
                            }
                        }
                        Err(error) => {
                            println!("UNREACHABLE");
                            println!("            {error}");
                            return Err(anyhow!("backend health check failed"));
                        }
                    }
                }
            }
        }
    }

    println!();
    println!("ready");
    Ok(())
}

fn skills() {
    println!("design skills in this build:");
    println!();
    for skill in SkillId::ALL {
        match executor(skill) {
            Some(implementation) => {
                println!("  {:<18} {}", skill.to_string(), implementation.describe());
                match skill.required_capability() {
                    Some(capability) => {
                        println!("  {:<18} needs capability '{}'", "", capability.as_str())
                    }
                    None => println!("  {:<18} needs no service capability", ""),
                }
            }
            None => {
                let needs = skill
                    .required_capability()
                    .map(|capability| format!(" (would need '{}')", capability.as_str()))
                    .unwrap_or_default();
                println!("  {:<18} not implemented yet{needs}", skill.to_string());
            }
        }
        println!();
    }
}

fn list_runs() -> Result<()> {
    let root = TaskPaths::default_root();
    let found = TaskPaths::list(&root).with_context(|| format!("listing {}", root.display()))?;

    if found.is_empty() {
        println!("no runs under {}", root.display());
        return Ok(());
    }

    println!(
        "{:<34} {:<18} {:>7} {:>10}  state",
        "task", "name", "cycles", "best"
    );
    for paths in found.iter().take(40) {
        let id = paths
            .root()
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_default();
        match fh::trace::read_json::<RunManifest>(paths.manifest()) {
            Ok(manifest) => println!(
                "{:<34} {:<18} {:>7} {:>10}  {}",
                id,
                manifest.name.chars().take(18).collect::<String>(),
                format!("{}/{}", manifest.cycles_completed, manifest.cycles_planned),
                manifest
                    .best_loss
                    .map(|loss| format!("{loss:.4}"))
                    .unwrap_or_else(|| "-".to_string()),
                if manifest.finished_at_ms.is_some() {
                    "finished"
                } else {
                    "incomplete"
                }
            ),
            Err(_) => println!("{:<34} {:<18} {:>7} {:>10}  no manifest", id, "-", "-", "-"),
        }
    }
    Ok(())
}
