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
const EXAMPLE_BINDER: &str = include_str!("../examples/binder.fasta");
const EXAMPLE_TARGET: &str = include_str!("../examples/target.fasta");

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
    /// Response token cap. Lower it if an endpoint rejects the default.
    #[arg(long)]
    max_tokens: Option<u32>,
    /// Request shape this endpoint speaks: anthropic or openai.
    #[arg(long)]
    wire: Option<String>,
    /// Environment variable holding the credential.
    #[arg(long)]
    key_env: Option<String>,
    /// Command whose output is the credential, for an official login CLI or a
    /// password manager. The command is stored, never the secret.
    #[arg(long)]
    key_command: Option<String>,
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
        /// Not needed when resuming, which reuses the stored sequences.
        #[arg(long)]
        binder: Vec<String>,
        /// Target chain: a FASTA path, or a bare sequence. Repeatable.
        #[arg(long)]
        target: Vec<String>,
        /// Carry on a stored chat by id. See `fh chats`.
        #[arg(long)]
        resume: Option<String>,
        /// Carry on the most recent chat.
        #[arg(long = "continue")]
        continue_last: bool,
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
        /// Render one frame as plain text and exit, instead of opening the
        /// dashboard. Works over a pipe and needs no terminal.
        #[arg(long)]
        snapshot: bool,
        /// Which pane to snapshot: overview, population, lineage, structure
        /// or events.
        #[arg(long, default_value = "overview")]
        pane: String,
        #[arg(long, default_value_t = 120)]
        width: u16,
        #[arg(long, default_value_t = 40)]
        height: u16,
        /// Rotate the structure view before rendering: yaw,pitch in radians.
        #[arg(long)]
        view: Option<String>,
        /// Draw the key list over the pane. With --snapshot, prints it.
        #[arg(long)]
        keys: bool,
    },
    /// Serve the browser view of recorded runs on localhost.
    Web {
        #[arg(long, default_value_t = 8750)]
        port: u16,
        /// Task root to read. Defaults to $FH_TASK_ROOT or ~/.folding_harness/tasks.
        #[arg(long)]
        task_root: Option<PathBuf>,
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
        /// Probe over the streaming path as well, which is what `fh chat`
        /// uses. An endpoint can serve ordinary requests and not stream.
        #[arg(long)]
        probe_stream: bool,
        #[command(flatten)]
        provider: ProviderOptions,
    },
    /// List the design skills this build knows about.
    Skills,
    /// List recorded runs.
    Ls,
    /// Measure search policies against the fixture's planted optimum.
    Bench {
        /// Task config to benchmark. Defaults to the built-in example.
        task: Option<PathBuf>,
        /// How many seeds to run per policy.
        #[arg(long, default_value_t = 5)]
        seeds: u64,
        #[arg(long, default_value_t = 20)]
        cycles: u32,
        /// Policies to compare. Defaults to all of them.
        #[arg(long)]
        policy: Vec<String>,
        /// Write the raw outcomes here as JSON.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Explain how to sign in for each wire, and what is already set.
    Login,
    /// Show or change the remembered provider settings.
    Config {
        #[command(subcommand)]
        action: Option<ConfigAction>,
    },
    /// List saved chats.
    Chats {
        /// Print the whole transcript of one chat.
        #[arg(long)]
        show: Option<String>,
    },
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Print the settings and where each value came from.
    Show,
    /// List the provider profiles.
    List,
    /// Switch to a profile.
    Use { name: String },
    /// Create or update a named profile.
    Add {
        name: String,
        #[command(flatten)]
        provider: ProviderOptions,
    },
    /// Delete a profile.
    Remove { name: String },
    /// Change the active profile's settings.
    Set {
        #[command(flatten)]
        provider: ProviderOptions,
    },
    /// Forget everything saved.
    Clear,
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
            resume,
            continue_last,
            seed,
            provider,
            task_root,
            latency_ms,
        } => {
            chat(
                binder,
                target,
                resume,
                continue_last,
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
            snapshot,
            pane,
            width,
            height,
            view,
            keys,
        } => board(
            path,
            task,
            Duration::from_millis(tick_ms),
            snapshot,
            pane,
            width,
            height,
            view,
            keys,
        ),
        Command::Web { port, task_root } => web(port, task_root).await,
        Command::Serve {
            port,
            task,
            latency_ms,
        } => serve(port, task, Duration::from_millis(latency_ms)).await,
        Command::Doctor {
            task,
            probe_model,
            probe_stream,
            provider,
        } => doctor(task, probe_model, probe_stream, provider).await,
        Command::Skills => {
            skills();
            Ok(())
        }
        Command::Ls => list_runs(),
        Command::Bench {
            task,
            seeds,
            cycles,
            policy,
            json,
        } => bench(task, seeds, cycles, policy, json).await,
        Command::Login => login(),
        Command::Config { action } => config(action),
        Command::Chats { show } => list_chats(show),
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

    // The FASTA files go beside the config, because `fh chat` and `fh task`
    // need sequences as files or arguments and the first thing anyone does is
    // reach for the example ones.
    let directory = path.parent().unwrap_or(std::path::Path::new("."));
    let mut sequences = Vec::new();
    for (name, contents) in [
        ("binder.fasta", EXAMPLE_BINDER),
        ("target.fasta", EXAMPLE_TARGET),
    ] {
        let destination = directory.join(name);
        if destination.exists() && !force {
            println!("kept {} (already there)", destination.display());
        } else {
            std::fs::write(&destination, contents)
                .with_context(|| format!("writing {}", destination.display()))?;
            println!("wrote {}", destination.display());
        }
        sequences.push(destination);
    }

    println!();
    println!("The sequences in these files are placeholders. Replace them before");
    println!("doing anything real; the target is synthetic, not a known protein.");
    println!();
    println!("next:");
    println!("  fh validate {}", path.display());
    println!("  fh run {}", path.display());
    if let [binder, target] = sequences.as_slice() {
        println!(
            "  fh chat --binder {} --target {}   (needs ANTHROPIC_API_KEY)",
            binder.display(),
            target.display()
        );
    }
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

    run_config(config, task_root, open_board, latency, None).await
}

/// Prepare and run an already-resolved config. Shared by `fh run`, `fh task`
/// and the chat screen, so all three behave identically once a plan exists.
async fn run_config(
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
impl ProviderOptions {
    fn as_settings(&self) -> fh::settings::ProviderSettings {
        fh::settings::ProviderSettings {
            url: self.url.clone(),
            model: self.model.clone(),
            api_version: self.api_version.clone(),
            effort: self.effort.clone(),
            max_tokens: self.max_tokens,
            wire: self.wire.as_deref().and_then(fh::provider::Wire::parse),
            key_env: self.key_env.clone(),
            key_command: self.key_command.clone(),
        }
    }
}

/// Build the provider from the saved settings, the environment and the flags.
///
/// Precedence, highest first: flag, environment, saved file, default. The
/// saved file is a convenience, never a trap.
fn effective_settings(options: &ProviderOptions) -> fh::settings::ProviderSettings {
    fh::settings::load()
        .overlay(&fh::settings::ProviderSettings::from_env())
        .overlay(&options.as_settings())
}

fn make_provider(options: &ProviderOptions) -> Result<Arc<dyn ModelProvider>> {
    let settings = effective_settings(options);
    if let Some(name) = &options.wire {
        if fh::provider::Wire::parse(name).is_none() {
            return Err(anyhow!(
                "no wire '{name}'; choose from: {}",
                fh::provider::Wire::names()
            ));
        }
    }
    fh::provider::build(&settings).map_err(|error| anyhow!("{error}"))
}

/// A gateway can return its own error as an ordinary HTTP 200 reply, which
/// then reads as though the model said it. Say where the text came from.
fn third_party_note(provider: &Arc<dyn ModelProvider>) -> String {
    if provider.is_first_party() {
        return String::new();
    }
    format!(
        "\n\nThat came back from {} with HTTP 200. A gateway may return its own error as an \
         ordinary reply, so the text above is not necessarily the model's words. Try another \
         --model (this endpoint is known to behave differently per model), a lower \
         --max-tokens, or the first-party API.",
        provider.endpoint()
    )
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

#[allow(clippy::too_many_arguments)]
async fn chat(
    binder: Vec<String>,
    target: Vec<String>,
    resume: Option<String>,
    continue_last: bool,
    seed: u64,
    options: ProviderOptions,
    task_root: Option<PathBuf>,
    latency: Duration,
) -> Result<()> {
    let chat_root = fh::history::chat_root();

    // Resuming brings its own sequences, so `--binder` and `--target` are only
    // required for a fresh chat.
    let record = if continue_last || resume.is_some() {
        let loaded =
            fh::history::load(&chat_root, resume.as_deref()).with_context(|| match &resume {
                Some(id) => format!("loading chat {id}"),
                None => "loading the most recent chat".to_string(),
            })?;
        println!(
            "resuming chat {} from {} ({} message(s))",
            loaded.id,
            fh::history::ago(loaded.updated_at_ms),
            loaded.messages.len()
        );
        Some(loaded)
    } else {
        None
    };

    let (binder_chains, target_chains) = match &record {
        Some(loaded) if binder.is_empty() && target.is_empty() => {
            (loaded.binder_chains(), loaded.target_chains())
        }
        _ => {
            if binder.is_empty() || target.is_empty() {
                return Err(anyhow!(
                    "a new chat needs --binder and --target (a FASTA path or a sequence).\n\
                     `fh init` writes example binder.fasta and target.fasta.\n\
                     To carry on an earlier chat instead, use --continue or --resume <id>."
                ));
            }
            (load_chains(&binder, "H")?, load_chains(&target, "A")?)
        }
    };

    let provider = make_provider(&options)?;
    println!("model {}", provider.describe());

    // The screen can rebuild its provider when /model or /url changes it, and
    // the flags given on this command still win over whatever it saves.
    let overrides = options.as_settings();
    let factory: fh::tui::chat::ProviderFactory = Box::new(move |settings| {
        let merged = settings.overlay(&overrides);
        let mut built = AnthropicProvider::from_env().map_err(|error| error.to_string())?;
        if let Some(url) = &merged.url {
            built = built.with_base_url(url.clone());
        }
        if let Some(model) = &merged.model {
            built = built.with_model(model.clone());
        }
        if let Some(version) = &merged.api_version {
            built = built.with_api_version(version.clone());
        }
        if let Some(effort) = &merged.effort {
            built = built.with_effort(Some(effort.clone()));
        }
        if let Some(max_tokens) = merged.max_tokens {
            built = built.with_max_tokens(max_tokens);
        }
        Ok(Arc::new(built) as Arc<dyn ModelProvider>)
    });

    // The screen runs on a blocking thread so the runtime stays free to make
    // the model call while the interface keeps drawing.
    let runtime = tokio::runtime::Handle::current();
    let confirmed = tokio::task::spawn_blocking(move || {
        fh::tui::chat::run(
            provider,
            runtime,
            binder_chains,
            target_chains,
            seed,
            record,
            Some(factory),
        )
    })
    .await
    .context("chat thread")?
    .context("running the chat screen")?;

    match confirmed {
        None => {
            println!("no plan confirmed; nothing ran");
            println!("the conversation is saved: fh chats");
            Ok(())
        }
        Some((config, chat_id)) => {
            println!("running the confirmed plan: {}", config.name);
            run_config(config, task_root, true, latency, Some(chat_id)).await
        }
    }
}

async fn bench(
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

/// Report what credentials are available and how to get one.
///
/// This harness does not implement OAuth. Device flow, PKCE and token refresh
/// are security-sensitive, and the vendors' own CLIs already do them properly,
/// so a profile names a command whose output is a token instead.
fn login() -> Result<()> {
    use fh::provider::Wire;

    let present = |program: &str| {
        std::process::Command::new("sh")
            .args(["-c", &format!("command -v {program}")])
            .output()
            .map(|out| out.status.success())
            .unwrap_or(false)
    };
    let set = |name: &str| {
        std::env::var(name)
            .map(|value| !value.trim().is_empty())
            .unwrap_or(false)
    };
    let mark = |ok: bool| if ok { "yes" } else { "no " };

    println!("This harness never stores a credential. A profile names where to");
    println!("find one: an environment variable, or a command whose output is a token.");
    println!();

    println!("anthropic wire");
    println!(
        "  ANTHROPIC_API_KEY set      {}",
        mark(set("ANTHROPIC_API_KEY"))
    );
    println!(
        "  ANTHROPIC_AUTH_TOKEN set   {}",
        mark(set("ANTHROPIC_AUTH_TOKEN"))
    );
    println!("  `ant` CLI installed        {}", mark(present("ant")));
    println!();
    println!("  An API key from console.anthropic.com is the simplest:");
    println!("    export ANTHROPIC_API_KEY=sk-ant-...");
    println!();
    println!("  Or sign in with the official CLI and let a profile read the token:");
    println!("    ant auth login");
    println!("    fh config add anthropic --wire anthropic \\");
    println!("        --key-command 'ant auth print-credentials --access-token'");
    println!();

    println!("openai wire");
    println!(
        "  OPENAI_API_KEY set         {}",
        mark(set("OPENAI_API_KEY"))
    );
    println!("  `codex` CLI installed      {}", mark(present("codex")));
    println!();
    println!("  Use an API key from platform.openai.com:");
    println!("    export OPENAI_API_KEY=sk-...");
    println!("    fh config add openai --wire openai --model gpt-5.2");
    println!();
    println!("  A `codex login` with a ChatGPT account authorises the Codex client,");
    println!("  not arbitrary API clients, so this harness will not read that token");
    println!("  out of ~/.codex and reuse it. Use an API key for programmatic access.");
    println!();

    println!("wires available: {}", Wire::names());
    Ok(())
}

fn config(action: Option<ConfigAction>) -> Result<()> {
    use fh::settings::{self, ProviderSettings};

    match action.unwrap_or(ConfigAction::Show) {
        ConfigAction::List => {
            let config = settings::load_config();
            let active = config.active_name();
            println!("{}", settings::config_path().display());
            println!();
            for name in config.names() {
                println!(
                    "{} {:<14} {}",
                    if name == active { "*" } else { " " },
                    name,
                    config.summary(&name)
                );
            }
            println!();
            println!("fh config use <name>                        switch");
            println!("fh config add <name> --url ... --model ...  create or update");
        }
        ConfigAction::Use { name } => {
            let mut config = settings::load_config();
            if name != settings::DEFAULT_PROFILE && !config.profiles.contains_key(&name) {
                return Err(anyhow!(
                    "no profile '{name}'; you have: {}",
                    config.names().join(", ")
                ));
            }
            config.active = Some(name.clone());
            settings::save_config(&config).context("saving")?;
            println!("now using {name}: {}", config.summary(&name));
        }
        ConfigAction::Add { name, provider } => {
            let wanted = provider.as_settings();
            if wanted.is_empty() {
                return Err(anyhow!(
                    "nothing to set; pass at least one of --url, --model, --api-version, \
                     --effort or --max-tokens"
                ));
            }
            let mut config = settings::load_config();
            let merged = config
                .get(&name)
                .cloned()
                .unwrap_or_default()
                .overlay(&wanted);
            config.profiles.insert(name.clone(), merged);
            config.active = Some(name.clone());
            settings::save_config(&config).context("saving")?;
            println!("saved profile {name}: {}", config.summary(&name));
            println!("and switched to it");
        }
        ConfigAction::Remove { name } => {
            let mut config = settings::load_config();
            if config.profiles.remove(&name).is_none() {
                return Err(anyhow!("no profile '{name}'"));
            }
            if config.active.as_deref() == Some(name.as_str()) {
                config.active = None;
            }
            settings::save_config(&config).context("saving")?;
            println!("removed {name}; now using {}", config.active_name());
        }
        ConfigAction::Show => {
            let config = settings::load_config();
            println!("profile      {}", config.active_name());
            let saved = settings::load();
            let env = ProviderSettings::from_env();
            println!("{}", settings::config_path().display());
            if saved.is_empty() {
                println!("(nothing saved yet)");
            }
            println!();
            println!("{:<12} {:<34} from", "setting", "value");
            for line in ProviderSettings::describe(&saved, &env) {
                println!("{line}");
            }
            println!();
            println!("Credentials are never saved here. Set ANTHROPIC_API_KEY in your");
            println!("environment; a config file outlives the session and is easy to commit.");
            println!();
            println!("fh config set --url https://gateway.example.com --model claude-opus-4-7");
        }
        ConfigAction::Set { provider } => {
            let wanted = provider.as_settings();
            if wanted.is_empty() {
                return Err(anyhow!(
                    "nothing to set; pass at least one of --url, --model, --api-version, \
                     --effort or --max-tokens"
                ));
            }
            let merged = settings::load().overlay(&wanted);
            let path = settings::save(&merged).context("saving the settings")?;
            println!("saved to {}", path.display());
            for line in ProviderSettings::describe(&merged, &ProviderSettings::default()) {
                println!("{line}");
            }
        }
        ConfigAction::Clear => {
            settings::clear().context("clearing the settings")?;
            println!("cleared {}", settings::config_path().display());
        }
    }
    Ok(())
}

fn list_chats(show: Option<String>) -> Result<()> {
    let root = fh::history::chat_root();

    if let Some(id) = show {
        let record =
            fh::history::load(&root, Some(&id)).with_context(|| format!("loading chat {id}"))?;
        println!(
            "chat {}  {}  model {}{}",
            record.id,
            fh::history::ago(record.updated_at_ms),
            record.model,
            if record.endpoint.is_empty() {
                String::new()
            } else {
                format!(" via {}", record.endpoint)
            }
        );
        if !record.task_ids.is_empty() {
            println!("runs {}", record.task_ids.join(", "));
        }
        println!();
        for message in &record.messages {
            let who = match message.role {
                fh::history::StoredRole::User => "you",
                fh::history::StoredRole::Assistant => "model",
            };
            println!("{who}:");
            for line in message.text.lines() {
                println!("  {line}");
            }
            println!();
        }
        return Ok(());
    }

    let chats = fh::history::list(&root).with_context(|| format!("listing {}", root.display()))?;
    if chats.is_empty() {
        println!("no saved chats under {}", root.display());
        println!("start one with: fh chat --binder binder.fasta --target target.fasta");
        return Ok(());
    }

    println!(
        "{:<16} {:<10} {:<22} {:>5} {:>5}  plan",
        "id", "when", "model", "turns", "runs"
    );
    for record in chats.iter().take(40) {
        println!(
            "{:<16} {:<10} {:<22} {:>5} {:>5}  {}",
            record.id,
            fh::history::ago(record.updated_at_ms),
            record.model.chars().take(22).collect::<String>(),
            record.user_turns(),
            record.task_ids.len(),
            record.plan_name.clone().unwrap_or_default()
        );
    }
    println!();
    println!("fh chats --show <id>        print a transcript");
    println!("fh chat --resume <id>       carry one on");
    Ok(())
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
    let plan = fh::plan::draft(
        provider.as_ref(),
        &prose,
        &binder_chains,
        &target_chains,
        seed,
    )
    .await
    .map_err(|error| anyhow!("{error}{}", third_party_note(&provider)))?;

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

    run_config(plan.config, task_root, false, Duration::ZERO, None).await
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

#[allow(clippy::too_many_arguments)]
fn board(
    path: Option<PathBuf>,
    task: Option<String>,
    tick: Duration,
    snapshot: bool,
    pane: String,
    width: u16,
    height: u16,
    view: Option<String>,
    keys: bool,
) -> Result<()> {
    let paths = resolve_task_dir(path, task)?;
    if !paths.root().exists() {
        return Err(anyhow!("{} does not exist", paths.root().display()));
    }

    if !snapshot {
        return fh::tui::run(paths, tick).context("running the dashboard");
    }

    let tab = fh::tui::Tab::from_name(&pane).ok_or_else(|| {
        anyhow!(
            "no pane called '{pane}'; choose one of: {}",
            fh::tui::Tab::names()
        )
    })?;

    let camera =
        match view.as_deref() {
            None => None,
            Some(text) => {
                let mut parts = text.split(',').map(str::trim);
                let yaw: f64 = parts.next().unwrap_or("").parse().map_err(|_| {
                    anyhow!("--view wants yaw,pitch in radians, e.g. --view 0.9,0.3")
                })?;
                let pitch: f64 = parts.next().unwrap_or("0").parse().map_err(|_| {
                    anyhow!("--view wants yaw,pitch in radians, e.g. --view 0.9,0.3")
                })?;
                Some(fh::tui::structure::Camera {
                    yaw,
                    pitch,
                    zoom: 1.0,
                })
            }
        };

    let text =
        fh::tui::snapshot(paths, tab, width, height, camera, keys).context("rendering the pane")?;
    print!("{text}");
    Ok(())
}

async fn web(port: u16, task_root: Option<PathBuf>) -> Result<()> {
    let root = task_root.unwrap_or_else(TaskPaths::default_root);
    let runs = TaskPaths::list(&root).unwrap_or_default();

    // Loopback only: this serves local run data and authenticates nobody.
    let addr: SocketAddr = ([127, 0, 0, 1], port).into();
    println!("reading {}", root.display());
    println!(
        "{} run(s) found{}",
        runs.len(),
        if runs.is_empty() {
            "; start one with `fh run task.yaml`"
        } else {
            ""
        }
    );
    println!();
    println!("open http://{addr}");
    println!("ctrl-c to stop");

    fh::web::serve(addr, root)
        .await
        .map_err(|error| anyhow!("serving: {error}"))
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

async fn doctor(
    task: Option<PathBuf>,
    probe_model: bool,
    probe_stream: bool,
    options: ProviderOptions,
) -> Result<()> {
    println!("fh          {}", env!("CARGO_PKG_VERSION"));
    println!("protocol    {}", fh::contract::PROTOCOL);

    let root = TaskPaths::default_root();
    println!("task root   {}", root.display());
    match std::fs::create_dir_all(&root) {
        Ok(()) => println!("            writable"),
        Err(error) => println!("            NOT writable: {error}"),
    }

    if probe_model || probe_stream {
        println!();
        match make_provider(&options) {
            Err(error) => {
                println!("model       NOT CONFIGURED");
                println!("            {error}");
                return Err(anyhow!("model provider not configured"));
            }
            Ok(provider) => {
                println!("model       {}", provider.describe());
                println!(
                    "            wire {}",
                    effective_settings(&options)
                        .wire
                        .unwrap_or_default()
                        .as_str()
                );
                print!("            probing ... ");
                use std::io::Write;
                std::io::stdout().flush().ok();
                if probe_model {
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
                } else {
                    println!("skipped");
                }

                if probe_stream {
                    print!("            streaming ... ");
                    std::io::stdout().flush().ok();
                    match provider.probe_stream().await {
                        Ok((reply, pieces)) => println!(
                            "ok: {pieces} piece(s), {}",
                            reply.trim().chars().take(40).collect::<String>()
                        ),
                        Err(error) => {
                            println!("FAILED");
                            println!("            {error}");
                            println!(
                                "            `fh chat` streams, so it will not work against \
                                 this endpoint; `fh task` does not stream and still will."
                            );
                            return Err(anyhow!("streaming probe failed"));
                        }
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
