//! Looking at runs: the dashboard, the browser view, the fixture fold
//! service, and the plain listings.
use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn board(
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

pub(super) async fn web(port: u16, task_root: Option<PathBuf>) -> Result<()> {
    let root = task_root.unwrap_or_else(TaskPaths::default_root);
    let runs = TaskPaths::list(&root).unwrap_or_default();

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
    println!("open {}", fh::web::loopback_url(port));
    println!("ctrl-c to stop");

    fh::web::serve(port, root)
        .await
        .map_err(|error| anyhow!("serving: {error}"))
}

pub(super) async fn serve(port: u16, task: Option<PathBuf>, latency: Duration) -> Result<()> {
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

    // Loopback by default: the service authenticates nobody.
    let addr = fh::fold::server::loopback_addr(port);
    println!("serving {} on http://{addr}", fh::contract::PROTOCOL);
    println!("  GET  http://{addr}{}", fh::contract::routes::HEALTH);
    println!("  POST http://{addr}{}", fh::contract::routes::FOLD);
    println!();
    println!("point a task at it with: fh run <task.yaml> --url http://{addr}");

    fh::fold::server::serve(addr, backend)
        .await
        .map_err(|error| anyhow!("serving: {error}"))
}

pub(super) fn skills() {
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

pub(super) fn list_runs() -> Result<()> {
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
