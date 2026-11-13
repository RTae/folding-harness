//! A local web view of recorded runs.
//!
//! The terminal dashboard is the one that works over SSH with nothing
//! installed. This is the other half: a browser gives real pixels, so the
//! structure can be dragged around smoothly and the loss curve can be read at
//! a glance.
//!
//! It is still one binary. The page, its stylesheet and its script are
//! compiled in with `include_str!`, so there is no asset directory to ship, no
//! Node, and no build step. The server reads the same task directory the
//! dashboard reads and holds no state of its own, so it cannot disturb a run
//! and a run does not need it.
//!
//! It binds to loopback, and [`serve`] is what makes that true. It takes a
//! port and binds the loopback address itself, and the router it serves is
//! `pub(crate)`, so no external crate can obtain that router and bind it to a
//! wider address. The data is a local research run, and nothing here
//! authenticates anybody.

use std::collections::HashMap;
use std::net::Ipv4Addr;
use std::path::{Path, PathBuf};

use axum::extract::{Path as UrlPath, Query, State};
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::Serialize;

use crate::trace::{self, PopulationSnapshot, RunManifest, StructureSnapshot, TaskPaths};
use crate::CandidateId;

const INDEX: &str = include_str!("index.html");
const SCRIPT: &str = include_str!("app.js");
const STYLE: &str = include_str!("app.css");

#[derive(Clone)]
struct WebState {
    root: PathBuf,
}

/// One row of the run picker.
#[derive(Debug, Serialize)]
struct RunRow {
    id: String,
    name: String,
    cycles_completed: u32,
    cycles_planned: u32,
    best_loss: Option<f64>,
    finished: bool,
    backend: String,
    model: String,
}

#[derive(Debug, Serialize)]
struct RunDetail {
    manifest: Option<RunManifest>,
    population: Option<PopulationSnapshot>,
    scope: Option<ScopeView>,
    /// The spend line, rendered server-side.
    ///
    /// The page shows this string instead of deriving a line from the
    /// manifest's numeric `usage` fields, which cannot say whether the model
    /// counts were measured. `None` means the terminal would print nothing:
    /// no spend worth a line. The numeric fields stay in the JSON, because
    /// `manifest` serializes whole.
    spend: Option<String>,
}

/// The mutable scope, resolved server-side.
///
/// The browser gets a plain boolean per residue rather than the task YAML, so
/// the page never has to re-derive which positions may change. One definition
/// of the scope, in Rust, shared by both interfaces.
#[derive(Debug, Serialize)]
struct ScopeView {
    summary: String,
    chains: Vec<ChainView>,
    regions: Vec<RegionView>,
}

#[derive(Debug, Serialize)]
struct ChainView {
    id: u8,
    label: String,
    role: &'static str,
    founder: String,
    /// One entry per residue of a binder chain; empty for the target.
    mutable: Vec<bool>,
}

#[derive(Debug, Serialize)]
struct RegionView {
    name: String,
    chain: u8,
    /// One-based inclusive, matching how the config is written.
    start: usize,
    end: usize,
    mutable: usize,
}

fn scope_view(paths: &TaskPaths) -> Option<ScopeView> {
    let config = crate::TaskConfig::load(paths.config()).ok()?;
    let (scaffold, scope) = config.resolve().ok()?;

    let chains = scaffold
        .chains
        .iter()
        .map(|chain| ChainView {
            id: chain.id.0,
            label: chain.label.clone(),
            role: match chain.role {
                crate::ChainRole::Binder => "binder",
                crate::ChainRole::Target => "target",
            },
            founder: chain.sequence.clone(),
            mutable: match chain.role {
                crate::ChainRole::Binder => (0..chain.len())
                    .map(|position| scope.is_mutable(chain.id, position))
                    .collect(),
                crate::ChainRole::Target => Vec::new(),
            },
        })
        .collect();

    let regions = scope
        .regions()
        .iter()
        .map(|region| {
            let span = region.display_span();
            RegionView {
                name: region.name.clone(),
                chain: region.chain.0,
                start: span.start,
                end: span.end,
                mutable: scope.region_positions(&region.name).len(),
            }
        })
        .collect();

    Some(ScopeView {
        summary: scope.summary(),
        chains,
        regions,
    })
}

/// Reject anything that is not a plain directory name.
///
/// Task ids reach this from a URL, so a joined path could otherwise climb out
/// of the task root. The syntax rule is [`trace::check_id`], shared with
/// `history::load`. The id must also be a directory that is actually there.
fn task_dir(root: &Path, id: &str) -> Option<TaskPaths> {
    if trace::check_id("task id", id).is_err() {
        return None;
    }
    let candidate = root.join(id);
    if candidate.is_dir() {
        Some(TaskPaths::new(candidate))
    } else {
        None
    }
}

fn asset(body: &'static str, content_type: &'static str) -> Response {
    ([(header::CONTENT_TYPE, content_type)], body).into_response()
}

pub(crate) fn app(root: PathBuf) -> Router {
    Router::new()
        .route("/", get(|| async { Html(INDEX) }))
        .route(
            "/app.js",
            get(|| async { asset(SCRIPT, "text/javascript; charset=utf-8") }),
        )
        .route(
            "/app.css",
            get(|| async { asset(STYLE, "text/css; charset=utf-8") }),
        )
        .route("/api/runs", get(runs))
        .route("/api/runs/:id", get(run_detail))
        .route("/api/runs/:id/trace", get(run_trace))
        .route("/api/runs/:id/structure/:candidate", get(structure))
        .with_state(WebState { root })
}

async fn runs(State(state): State<WebState>) -> impl IntoResponse {
    let found = TaskPaths::list(&state.root).unwrap_or_default();
    let rows: Vec<RunRow> = found
        .iter()
        .filter_map(|paths| {
            let id = paths.root().file_name()?.to_string_lossy().to_string();
            let manifest: RunManifest = trace::read_json(paths.manifest()).ok()?;
            Some(RunRow {
                id,
                name: manifest.name,
                cycles_completed: manifest.cycles_completed,
                cycles_planned: manifest.cycles_planned,
                best_loss: manifest.best_loss,
                finished: manifest.finished_at_ms.is_some(),
                backend: manifest.backend,
                model: manifest.model,
            })
        })
        .collect();
    Json(rows)
}

async fn run_detail(
    State(state): State<WebState>,
    UrlPath(id): UrlPath<String>,
) -> impl IntoResponse {
    let Some(paths) = task_dir(&state.root, &id) else {
        return (StatusCode::NOT_FOUND, "no such run").into_response();
    };
    let manifest: Option<RunManifest> = trace::read_json(paths.manifest()).ok();
    // The same rendering and the same silence the terminal report uses:
    // `Usage::summary` when there is spend to report, and no line when
    // `is_empty` says the terminal would print nothing.
    let spend = manifest
        .as_ref()
        .filter(|manifest| !manifest.usage.is_empty())
        .map(|manifest| manifest.usage.summary());
    Json(RunDetail {
        manifest,
        population: trace::read_json(paths.population()).ok(),
        scope: scope_view(&paths),
        spend,
    })
    .into_response()
}

async fn run_trace(
    State(state): State<WebState>,
    UrlPath(id): UrlPath<String>,
    Query(params): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    let Some(paths) = task_dir(&state.root, &id) else {
        return (StatusCode::NOT_FOUND, "no such run").into_response();
    };
    let after: u64 = params
        .get("after")
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let records = trace::read_after(paths.trace(), after).unwrap_or_default();
    Json(records).into_response()
}

async fn structure(
    State(state): State<WebState>,
    UrlPath((id, candidate)): UrlPath<(String, u32)>,
) -> impl IntoResponse {
    let Some(paths) = task_dir(&state.root, &id) else {
        return (StatusCode::NOT_FOUND, "no such run").into_response();
    };
    match trace::read_json::<StructureSnapshot>(paths.structure(CandidateId(candidate))) {
        Ok(snapshot) => Json(snapshot).into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "no stored structure").into_response(),
    }
}

/// The loopback address for `port`. The bind and the printed URL both take
/// their address from here, so the two cannot drift apart.
fn loopback_addr(port: u16) -> std::net::SocketAddr {
    (Ipv4Addr::LOCALHOST, port).into()
}

/// Bind `port` on the loopback address. Port 0 asks the OS to pick a free
/// port, which the tests use.
async fn bind_loopback(port: u16) -> std::io::Result<tokio::net::TcpListener> {
    tokio::net::TcpListener::bind(loopback_addr(port)).await
}

/// The URL of the address [`serve`] binds on `port`, formatted for printing.
///
/// `serve` runs until the process stops and returns nothing, so a caller
/// cannot read the bound address back. This function and `bind_loopback` both
/// take the address from [`loopback_addr`], which keeps the URL a caller
/// prints from drifting away from the address the server binds.
pub fn loopback_url(port: u16) -> String {
    format!("http://{}", loopback_addr(port))
}

/// Bind the loopback address on `port` and serve until the process stops.
///
/// The page serves run data, including the sequences from a task config, and
/// authenticates nobody, so the loopback decision lives here rather than in
/// each caller. A caller picks the port and cannot widen the address.
pub async fn serve(
    port: u16,
    root: PathBuf,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let listener = bind_loopback(port).await?;
    axum::serve(listener, app(root)).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A temp task root that removes itself when the test ends. `Drop` runs
    /// even when an assertion fails and the test panics, which a trailing
    /// `remove_dir_all` does not.
    struct TempRoot {
        path: PathBuf,
    }

    impl TempRoot {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("fh-web-{name}-{}", trace::now_ms()));
            std::fs::create_dir_all(dir.join("1700-demo")).expect("task dir");
            Self { path: dir }
        }
    }

    impl Drop for TempRoot {
        fn drop(&mut self) {
            // A failure to remove must not mask the panic that is unwinding.
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn a_task_id_cannot_climb_out_of_the_root() {
        let root = TempRoot::new("traversal");
        // The real one resolves.
        assert!(task_dir(&root.path, "1700-demo").is_some());

        // These must not, whatever the filesystem would make of them.
        for attempt in [
            "..",
            "../..",
            "../../etc",
            "..%2f..",
            "/etc/passwd",
            "1700-demo/../..",
            ".hidden",
            "",
            "a/b",
            "a\\b",
        ] {
            assert!(
                task_dir(&root.path, attempt).is_none(),
                "{attempt:?} should have been refused"
            );
        }

        // A name that is well-formed but absent is still refused.
        assert!(task_dir(&root.path, "1700-nothing").is_none());
    }

    #[test]
    fn an_over_long_id_is_refused() {
        let root = TempRoot::new("long");
        assert!(task_dir(&root.path, &"a".repeat(129)).is_none());
    }

    /// Every element the script looks up has to exist, or a pane silently
    /// stops updating. The two files are compiled in together, so this can be
    /// checked at build time rather than discovered in a browser.
    #[test]
    fn every_element_the_script_looks_up_exists_in_the_page() {
        // Ids the script creates itself rather than finding in the page.
        const GENERATED: &[&str] = &["curve-marker"];

        let mut defined: Vec<&str> = Vec::new();
        let mut rest = INDEX;
        while let Some(at) = rest.find("id=\"") {
            rest = &rest[at + 4..];
            if let Some(end) = rest.find('"') {
                defined.push(&rest[..end]);
                rest = &rest[end..];
            }
        }

        let mut looked_up: Vec<&str> = Vec::new();
        let mut rest = SCRIPT;
        while let Some(at) = rest.find("el(\"") {
            rest = &rest[at + 4..];
            if let Some(end) = rest.find('"') {
                looked_up.push(&rest[..end]);
                rest = &rest[end..];
            }
        }

        assert!(!looked_up.is_empty(), "the script looks nothing up");
        for id in &looked_up {
            assert!(
                defined.contains(id) || GENERATED.contains(id),
                "the script looks up '{id}', which the page does not define"
            );
        }
    }

    #[test]
    fn the_page_loads_the_assets_this_server_serves() {
        assert!(INDEX.contains("/app.js"), "the page never loads the script");
        assert!(
            INDEX.contains("/app.css"),
            "the page never loads the styles"
        );
        // And the routes exist for both.
        let routes = [("/app.js", SCRIPT), ("/app.css", STYLE)];
        for (path, body) in routes {
            assert!(!body.is_empty(), "{path} is empty");
        }
    }

    /// Every event kind the trace can record has a case in the script's
    /// `summarise`, or the event pane renders it blank: a kind chip with no
    /// text. Both files compile into the binary, so this is checked at build
    /// time rather than discovered in a browser. The kinds in `BLANK` may
    /// render with no text, because the chip alone says enough for them.
    #[test]
    fn every_event_kind_the_trace_records_has_a_case_in_the_script() {
        const BLANK: &[&str] = &["fold_started", "fold_finished"];

        // The kinds Rust emits: the string literals in the `Event::kind()`
        // match in src/trace.rs. Both files are in this crate, so the source
        // is available to the test.
        const TRACE_SOURCE: &str = include_str!("../trace.rs");
        let after_signature = TRACE_SOURCE
            .split_once("pub fn kind(&self) -> &'static str")
            .expect("trace.rs defines Event::kind")
            .1;
        let kind_body = after_signature
            .split_once("\n    }\n")
            .expect("Event::kind ends")
            .0;
        let mut rust_kinds: Vec<&str> = Vec::new();
        let mut rest = kind_body;
        while let Some(at) = rest.find("=> \"") {
            rest = &rest[at + 4..];
            if let Some(end) = rest.find('"') {
                rust_kinds.push(&rest[..end]);
                rest = &rest[end..];
            }
        }
        assert!(
            rust_kinds.len() >= 15,
            "found only {} kinds in Event::kind, the parse broke",
            rust_kinds.len()
        );

        // The kinds the script summarises: the `case` labels inside
        // `summarise`. The scan stays inside that function, because the
        // script has other switches whose labels are not event kinds.
        let summarise = SCRIPT
            .split_once("function summarise(")
            .expect("the script defines summarise")
            .1;
        let summarise_body = summarise.split_once("\n}\n").expect("summarise ends").0;
        let mut script_cases: Vec<&str> = Vec::new();
        let mut rest = summarise_body;
        while let Some(at) = rest.find("case \"") {
            rest = &rest[at + 6..];
            if let Some(end) = rest.find('"') {
                script_cases.push(&rest[..end]);
                rest = &rest[end..];
            }
        }
        assert!(
            !script_cases.is_empty(),
            "found no cases in summarise, the parse broke"
        );

        for kind in &rust_kinds {
            assert!(
                script_cases.contains(kind) || BLANK.contains(kind),
                "the event pane renders '{kind}' blank: summarise in app.js has no case for it"
            );
        }
        for case in &script_cases {
            assert!(
                rust_kinds.contains(case),
                "summarise in app.js has a case for '{case}', which the trace never records"
            );
        }
    }

    /// A run the picker can show. `runs` skips a task directory whose
    /// `run.json` is missing or unreadable, so the list stays empty without
    /// this.
    fn write_manifest(paths: &TaskPaths) {
        write_manifest_with_usage(paths, Default::default());
    }

    /// The same run with a chosen spend snapshot.
    fn write_manifest_with_usage(paths: &TaskPaths, usage: crate::usage::Usage) {
        let manifest = RunManifest {
            task_id: "1700-demo".to_string(),
            name: "demo".to_string(),
            harness_version: "test".to_string(),
            started_at_ms: 0,
            finished_at_ms: None,
            seed: 7,
            backend: "fixture".to_string(),
            model: "none".to_string(),
            protocol: "test".to_string(),
            service_version: None,
            config_sha256: "0".to_string(),
            cycles_planned: 2,
            cycles_completed: 1,
            folds: 3,
            best_candidate: None,
            best_loss: Some(0.5),
            chat_id: None,
            usage,
            resumed_from_cycle: None,
            generator_reseeded: false,
            resumed_protocol: None,
            resumed_service_version: None,
        };
        let encoded = serde_json::to_vec(&manifest).expect("encode manifest");
        std::fs::write(paths.manifest(), encoded).expect("write manifest");
    }

    /// The page and the terminal read the same manifest, so they must say the
    /// same thing about what the run spent. The assertions compare the served
    /// `spend` against `Usage::summary()` for the same `Usage`, which is what
    /// the terminal report prints, so the two renderings cannot drift without
    /// this test failing. The empty case pins the shared silence: the terminal
    /// prints no spend line for it, so the page must receive none.
    #[tokio::test]
    async fn the_browser_receives_the_spend_line_the_terminal_prints() {
        use crate::usage::Usage;

        let root = TempRoot::new("spend");
        let cases: Vec<(&str, Usage)> = vec![
            // The source counted and made no model call: folds only, and no
            // model wording, because a counted zero is a measurement.
            (
                "1700-no-calls",
                Usage {
                    folds: 3,
                    fold_ms: 900,
                    ..Usage::no_model_calls()
                },
            ),
            // The source counted its calls: the totals are measurements.
            (
                "1700-counted",
                Usage {
                    calls: 2,
                    input_tokens: 100,
                    output_tokens: 40,
                    folds: 3,
                    fold_ms: 900,
                    calls_counted: Some(true),
                },
            ),
            // The source does not count: calls may have happened that the
            // zero totals do not show.
            ("1700-uncounted", Usage::uncounted()),
            // A manifest written before the counted flag existed, carrying
            // real totals. Nobody said whether they were counted, so the
            // rendering must say that rather than imply the totals are
            // measurements.
            (
                "1700-predates-the-flag",
                Usage {
                    calls: 5,
                    input_tokens: 410,
                    output_tokens: 85,
                    folds: 3,
                    fold_ms: 900,
                    calls_counted: None,
                },
            ),
            // Nothing on record at all: the terminal prints no spend line.
            ("1700-empty", Usage::default()),
        ];
        for (id, usage) in &cases {
            let paths = TaskPaths::new(root.path.join(id));
            std::fs::create_dir_all(paths.root()).expect("task dir");
            write_manifest_with_usage(&paths, *usage);
        }

        let listener = bind_loopback(0).await.expect("bind");
        let addr = listener.local_addr().expect("bound address");
        let served = root.path.clone();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app(served)).await;
        });

        for (id, usage) in &cases {
            let body: serde_json::Value = reqwest::get(format!("http://{addr}/api/runs/{id}"))
                .await
                .expect("request the run detail")
                .json()
                .await
                .expect("read the run detail");
            let expected = (!usage.is_empty()).then(|| usage.summary());
            assert_eq!(
                body["spend"],
                serde_json::to_value(expected).expect("encodes"),
                "the page and the terminal disagree about {id}"
            );
        }
    }

    /// The script reads the rendered `spend` field and nothing else. If it
    /// ever went back to deriving a line from `usage.calls` and `usage.folds`,
    /// the state logic would exist twice, in two languages, and only one copy
    /// would be under test.
    #[test]
    fn the_script_reads_the_rendered_spend_line_and_never_rederives_it() {
        let detail = RunDetail {
            manifest: None,
            population: None,
            scope: None,
            spend: Some("an uncounted number of model calls".to_string()),
        };
        let json = serde_json::to_value(&detail).expect("encodes");
        assert_eq!(
            json["spend"], "an uncounted number of model calls",
            "the server must send the field under the name the script reads"
        );
        assert!(
            SCRIPT.contains("detail.spend"),
            "the script never reads the rendered spend line"
        );
        // Assert on the identifier rather than on `usage.calls`, because the
        // code this replaced read `usage?.calls`, and a pin on the dotted
        // spelling would miss a regression written the original way. The
        // script has no legitimate reason to name `usage` at all now, so any
        // occurrence means the state logic came back into JavaScript.
        assert!(
            !SCRIPT.contains("usage"),
            "the script names `usage` again, so it is re-deriving the spend \
             line instead of reading the one the server rendered"
        );
    }

    #[tokio::test]
    async fn binding_a_port_gives_a_socket_on_the_loopback_address() {
        let listener = bind_loopback(0).await.expect("bind");
        let addr = listener.local_addr().expect("bound address");
        assert!(
            addr.ip().is_loopback(),
            "serve would bind {addr}, which is not loopback"
        );
    }

    /// The binary prints the URL from [`loopback_url`] and `serve` binds the
    /// address from `bind_loopback`. Both must name the loopback address for
    /// the same port, or the printed URL lies about where the server listens.
    #[test]
    fn the_formatted_url_is_the_loopback_url_for_that_port() {
        let url = loopback_url(8750);
        assert_eq!(url, "http://127.0.0.1:8750");
        let authority = url.strip_prefix("http://").expect("an http URL");
        let addr: std::net::SocketAddr = authority.parse().expect("host and port");
        assert!(addr.ip().is_loopback(), "{addr} is not loopback");
        assert_eq!(addr.port(), 8750);
    }

    /// The exact-string test above pins the format. This one pins the
    /// agreement: a change that moves only the binder, say to `::1`, makes
    /// the printed URL lie about where the server listens, and fails here.
    #[tokio::test]
    async fn the_printed_url_names_the_address_the_server_binds() {
        let listener = bind_loopback(0).await.expect("bind");
        let bound = listener.local_addr().expect("bound address");
        let url = loopback_url(bound.port());
        let authority = url.strip_prefix("http://").expect("an http URL");
        let printed: std::net::SocketAddr = authority.parse().expect("host and port");
        assert_eq!(
            printed, bound,
            "serve binds {bound} but the printed URL names {printed}"
        );
    }

    #[tokio::test]
    async fn serving_a_run_directory_returns_the_runs_list() {
        let root = TempRoot::new("serve");
        write_manifest(&TaskPaths::new(root.path.join("1700-demo")));

        // Bind first, so the port is listening before the request goes out.
        let listener = bind_loopback(0).await.expect("bind");
        let addr = listener.local_addr().expect("bound address");
        let served = root.path.clone();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app(served)).await;
        });

        let body: serde_json::Value = reqwest::get(format!("http://{addr}/api/runs"))
            .await
            .expect("request the runs list")
            .json()
            .await
            .expect("read the runs list");
        let rows = body.as_array().expect("the runs list is an array");
        assert_eq!(rows.len(), 1, "expected only 1700-demo, got {body}");
        assert_eq!(rows[0]["id"], "1700-demo");
        assert_eq!(rows[0]["name"], "demo");
        // `write_manifest` leaves `finished_at_ms` unset, so this pins the
        // negative half. Without it a handler hardcoded to `finished: true`
        // would pass, and the founder-only test would be the only guard.
        assert_eq!(
            rows[0]["finished"], false,
            "an unfinished run must not be listed as finished"
        );
    }

    /// The config for a real founder-only run. Small, because the arm folds
    /// the founder once and never asks the agent or the router anything.
    const FOUNDER_YAML: &str = r#"
name: founder-web
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
target:
  chains:
    - label: A
      sequence: MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQ
design:
  cycles: 1
  batch_size: 2
  seed: 21
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

    /// The run picker reads "finished" from the manifest's `finished_at_ms`.
    /// A founder-only run, the floor arm `fh bench` writes, must be listed as
    /// finished once it completes. The task directory here comes from a real
    /// orchestrator run, not a hand-written manifest, so the test fails if the
    /// arm ever stops closing its manifest again.
    #[tokio::test]
    async fn a_completed_founder_only_run_is_listed_as_finished() {
        use crate::agent::HeuristicAgent;
        use crate::fold::FixtureBackend;
        use crate::orchestrator::Orchestrator;
        use crate::{TaskConfig, TaskId};

        let root = TempRoot::new("founder-finished");
        let config = TaskConfig::from_yaml(FOUNDER_YAML).expect("config parses");
        let (scaffold, scope) = config.resolve().expect("resolves");
        let backend = std::sync::Arc::new(FixtureBackend::planted(
            &scaffold,
            &scope,
            config.design.seed,
        ));
        let mut orchestrator = Orchestrator::prepare(
            config,
            backend,
            Box::new(HeuristicAgent::new(21)),
            TaskPaths::new(root.path.join("1700-demo")),
            TaskId("founder".to_string()),
        )
        .await
        .expect("prepares");
        let summary = orchestrator.seed_founder_only().await.expect("completes");
        let founder_loss = summary
            .best
            .as_ref()
            .and_then(|candidate| candidate.loss)
            .expect("the founder is scored");

        let listener = bind_loopback(0).await.expect("bind");
        let addr = listener.local_addr().expect("bound address");
        let served = root.path.clone();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app(served)).await;
        });

        let body: serde_json::Value = reqwest::get(format!("http://{addr}/api/runs"))
            .await
            .expect("request the runs list")
            .json()
            .await
            .expect("read the runs list");
        let rows = body.as_array().expect("the runs list is an array");
        assert_eq!(rows.len(), 1, "expected only 1700-demo, got {body}");
        assert_eq!(rows[0]["id"], "1700-demo");
        assert_eq!(
            rows[0]["finished"], true,
            "a completed founder-only run must not be listed as still running"
        );
        assert_eq!(rows[0]["cycles_completed"], 0);
        assert_eq!(
            rows[0]["best_loss"].as_f64(),
            Some(founder_loss),
            "the picker must show the founder's loss"
        );
    }

    #[tokio::test]
    async fn the_page_and_its_assets_are_served_with_their_types() {
        // The assets are compiled in, so this also proves they exist.
        assert!(INDEX.contains("<!doctype html>") || INDEX.contains("<!DOCTYPE html>"));
        assert!(SCRIPT.contains("function"));
        assert!(!STYLE.is_empty());

        let response = asset(SCRIPT, "text/javascript; charset=utf-8");
        assert_eq!(
            response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("text/javascript; charset=utf-8")
        );
    }
}
