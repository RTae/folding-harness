//! Provider profiles and signing in.
use super::*;

/// Report what credentials are available and how to get one.
///
/// This harness does not implement OAuth. Device flow, PKCE and token refresh
/// are security-sensitive, and the vendors' own CLIs already do them properly,
/// so a profile names a command whose output is a token instead.
pub(super) fn login() -> Result<()> {
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

pub(super) fn config(action: Option<ConfigAction>) -> Result<()> {
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
