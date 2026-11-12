//! Writing a starting task, checking one, and checking the environment.
use super::*;

pub(super) fn init(path: PathBuf, force: bool) -> Result<()> {
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

pub(super) fn validate(task: PathBuf) -> Result<()> {
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
            "  {:<10} residues {} ({} mutable{})",
            region.name,
            region.display_span(),
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

pub(super) async fn doctor(
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
