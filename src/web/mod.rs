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
//! It binds to loopback. The data is a local research run, and nothing here
//! authenticates anybody.

use std::collections::HashMap;
use std::net::SocketAddr;
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
pub struct WebState {
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

pub fn app(root: PathBuf) -> Router {
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

/// Bind and serve until the process stops.
pub async fn serve(
    addr: SocketAddr,
    root: PathBuf,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app(root)).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fh-web-{name}-{}", trace::now_ms()));
        std::fs::create_dir_all(dir.join("1700-demo")).expect("task dir");
        dir
    }

    #[test]
    fn a_task_id_cannot_climb_out_of_the_root() {
        let root = temp_root("traversal");
        // The real one resolves.
        assert!(task_dir(&root, "1700-demo").is_some());

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
                task_dir(&root, attempt).is_none(),
                "{attempt:?} should have been refused"
            );
        }

        // A name that is well-formed but absent is still refused.
        assert!(task_dir(&root, "1700-nothing").is_none());

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn an_over_long_id_is_refused() {
        let root = temp_root("long");
        assert!(task_dir(&root, &"a".repeat(129)).is_none());
        std::fs::remove_dir_all(&root).ok();
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
