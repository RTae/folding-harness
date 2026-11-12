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
        .map(|region| RegionView {
            name: region.name.clone(),
            chain: region.chain.0,
            start: region.start + 1,
            end: region.end,
            mutable: scope.region_positions(&region.name).len(),
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
/// of the task root. The id must also be a directory that is actually there.
fn task_dir(root: &Path, id: &str) -> Option<TaskPaths> {
    let safe = !id.is_empty()
        && id.len() <= 128
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
        && !id.starts_with('.');
    if !safe {
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
    Json(RunDetail {
        manifest: trace::read_json(paths.manifest()).ok(),
        population: trace::read_json(paths.population()).ok(),
        scope: scope_view(&paths),
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

    /// A run the picker can show. `runs` skips a task directory whose
    /// `run.json` is missing or unreadable, so the list stays empty without
    /// this.
    fn write_manifest(paths: &TaskPaths) {
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
            usage: Default::default(),
        };
        let encoded = serde_json::to_vec(&manifest).expect("encode manifest");
        std::fs::write(paths.manifest(), encoded).expect("write manifest");
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
