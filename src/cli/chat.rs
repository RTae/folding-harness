//! The conversational entry points: the chat screen, one-shot prose, and the
//! record of past conversations.
use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) async fn chat(
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

    // The screen can rebuild its provider when /model, /url or /provider
    // changes it, and the flags given on this command still win over whatever
    // it saves. Building goes through the one function that knows about
    // wires: a hand-rolled copy here once pointed an Anthropic client at an
    // OpenAI endpoint.
    let overrides = options.as_settings();
    let factory: fh::tui::chat::ProviderFactory = Box::new(move |settings| {
        fh::provider::build(&settings.overlay(&overrides)).map_err(|error| error.to_string())
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

#[allow(clippy::too_many_arguments)]
pub(super) async fn task_from_prose(
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
    let spent = provider.usage();
    if !spent.is_empty() {
        println!();
        println!("spent       {}", spent.summary());
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

pub(super) fn list_chats(show: Option<String>) -> Result<()> {
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
