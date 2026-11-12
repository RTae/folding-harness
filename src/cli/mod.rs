//! The command line: the arguments, and the one match that dispatches them.
//!
//! These modules belong to the binary, not the library, so `anyhow` and the
//! terminal stay out of `fh` itself. `main.rs` declares them; `lib.rs` does
//! not.

mod chat;
mod config;
mod run;
mod setup;
mod view;

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
use fh::skills::{executor, Router};
use fh::trace::{RunManifest, TaskPaths};
use fh::{Capability, ExecutionMode, SearchState, SkillId, TaskConfig, TaskId};

pub(crate) const EXAMPLE_TASK: &str = include_str!("../../examples/vhh_demo.yaml");
pub(crate) const EXAMPLE_BINDER: &str = include_str!("../../examples/binder.fasta");
pub(crate) const EXAMPLE_TARGET: &str = include_str!("../../examples/target.fasta");

#[derive(Parser)]
#[command(
    name = "fh",
    version,
    about = "A design harness: orchestration, gates and inspection around a folding backend",
    long_about = "fh runs a design cycle over a folding backend and records everything it did.\n\
                  The default backend is a deterministic fixture, so `fh run` works with no GPU."
)]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Where to send prompts, and how to shape the request.
///
/// A custom `--url` is for a proxy, gateway or self-hosted endpoint. The
/// harness prints the host before sending anything, and drops the
/// Anthropic-specific request fields, which a gateway may not accept.
#[derive(clap::Args, Debug, Clone)]
pub struct ProviderOptions {
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
pub enum Command {
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
pub enum ConfigAction {
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

/// Run the command the arguments name.
///
/// One match, so every subcommand is visible in one place and `main` stays a
/// line long.
pub async fn dispatch(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Init { path, force } => setup::init(path, force),
        Command::Validate { task } => setup::validate(task),
        Command::Run {
            task,
            cycles,
            seed,
            url,
            task_root,
            board,
            latency_ms,
        } => {
            run::run(
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
            chat::chat(
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
            chat::task_from_prose(
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
        } => view::board(
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
        Command::Web { port, task_root } => view::web(port, task_root).await,
        Command::Serve {
            port,
            task,
            latency_ms,
        } => view::serve(port, task, Duration::from_millis(latency_ms)).await,
        Command::Doctor {
            task,
            probe_model,
            probe_stream,
            provider,
        } => setup::doctor(task, probe_model, probe_stream, provider).await,
        Command::Skills => {
            view::skills();
            Ok(())
        }
        Command::Ls => view::list_runs(),
        Command::Bench {
            task,
            seeds,
            cycles,
            policy,
            json,
        } => run::bench(task, seeds, cycles, policy, json).await,
        Command::Login => config::login(),
        Command::Config { action } => config::config(action),
        Command::Chats { show } => chat::list_chats(show),
    }
}

pub(crate) fn backend_for(
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

/// Load chains from FASTA paths or bare sequences. The first unlabelled chain
/// gets `default_label`, so `--binder QVQL...` works without a FASTA header.
pub(crate) fn load_chains(values: &[String], default_label: &str) -> Result<Vec<PlanChain>> {
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

/// Build the provider from the saved settings, the environment and the flags.
///
/// Precedence, highest first: flag, environment, saved file, default. The
/// saved file is a convenience, never a trap.
pub(crate) fn effective_settings(options: &ProviderOptions) -> fh::settings::ProviderSettings {
    fh::settings::load()
        .overlay(&fh::settings::ProviderSettings::from_env())
        .overlay(&options.as_settings())
}

pub(crate) fn make_provider(options: &ProviderOptions) -> Result<Arc<dyn ModelProvider>> {
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
pub(crate) fn third_party_note(provider: &Arc<dyn ModelProvider>) -> String {
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

pub(crate) fn confirm(question: &str) -> Result<bool> {
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

pub(crate) fn resolve_task_dir(path: Option<PathBuf>, task: Option<String>) -> Result<TaskPaths> {
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

pub(crate) fn report(summary: &fh::orchestrator::RunSummary) {
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
    if let Ok(manifest) =
        fh::trace::read_json::<fh::trace::RunManifest>(summary.root.join("run.json"))
    {
        if !manifest.usage.is_empty() {
            println!("spent {}", manifest.usage.summary());
        }
    }
    println!();
    println!("results {}", summary.root.display());
    println!("inspect fh board --path {}", summary.root.display());
    println!(
        "         (numbers above come from the configured backend; a fixture run predicts nothing)"
    );
}
