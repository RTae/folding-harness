//! A stub compute service: puts any [`FoldBackend`] behind the `fh.fold/v1`
//! HTTP contract.
//!
//! Its purpose is to exercise the real network path in development, so that
//! the only difference between `execution_mode: fixture` and
//! `execution_mode: service` is whether the bytes crossed a socket. It is also
//! the reference for anyone implementing the service side in another language:
//! the handlers below are the whole protocol.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};

use crate::contract::{
    major_version, ErrorBody, FoldRequest, JobAccepted, JobState, JobStatus, PROTOCOL,
};

use super::FoldBackend;

#[derive(Clone)]
pub struct ServiceState {
    backend: Arc<dyn FoldBackend>,
    jobs: Arc<Mutex<HashMap<String, JobStatus>>>,
    counter: Arc<AtomicU64>,
    /// Set on the first recovered poison so the recovery is recorded once
    /// rather than on every later request.
    poison_recorded: Arc<AtomicBool>,
}

impl ServiceState {
    pub fn new(backend: Arc<dyn FoldBackend>) -> Self {
        ServiceState {
            backend,
            jobs: Arc::new(Mutex::new(HashMap::new())),
            counter: Arc::new(AtomicU64::new(1)),
            poison_recorded: Arc::new(AtomicBool::new(false)),
        }
    }

    fn next_job_id(&self) -> String {
        format!("job-{:06}", self.counter.fetch_add(1, Ordering::SeqCst))
    }

    /// Lock the job table, recovering it if a thread panicked while holding
    /// the lock. The mutex is poisoned only by a panic, and every critical
    /// section on this table is a map operation that a panic cannot leave
    /// structurally broken, so the jobs and their count stay readable. What
    /// recovery must not do is hide the panic: a poisoned lock means a worker
    /// died, and reporting a healthy queue over it would fabricate a
    /// measurement. The first recovery is recorded on the service's stderr,
    /// where the panic hook's own output already names the thread and site.
    fn lock_jobs(&self) -> MutexGuard<'_, HashMap<String, JobStatus>> {
        match self.jobs.lock() {
            Ok(jobs) => jobs,
            Err(poisoned) => {
                if !self.poison_recorded.swap(true, Ordering::SeqCst) {
                    eprintln!(
                        "[fh fold server] a thread panicked while holding the job table; \
                         recovered the lock, job states written before the panic may be stale"
                    );
                }
                poisoned.into_inner()
            }
        }
    }
}

pub fn app(backend: Arc<dyn FoldBackend>) -> Router {
    router(ServiceState::new(backend))
}

fn router(state: ServiceState) -> Router {
    Router::new()
        .route(crate::contract::routes::HEALTH, get(health))
        .route(crate::contract::routes::FOLD, post(submit))
        .route("/v1/fold/:job_id", get(status))
        .with_state(state)
}

async fn health(State(state): State<ServiceState>) -> impl IntoResponse {
    match state.backend.health().await {
        Ok(mut report) => {
            report.queue_depth = state
                .lock_jobs()
                .values()
                .filter(|job| !job.state.is_terminal())
                .count() as u32;
            (StatusCode::OK, Json(report)).into_response()
        }
        Err(error) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorBody {
                error: "backend_unavailable".to_string(),
                detail: Some(error.to_string()),
            }),
        )
            .into_response(),
    }
}

async fn submit(
    State(state): State<ServiceState>,
    Json(request): Json<FoldRequest>,
) -> impl IntoResponse {
    if major_version(&request.protocol) != major_version(PROTOCOL) {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorBody {
                error: "protocol_mismatch".to_string(),
                detail: Some(format!(
                    "service speaks {PROTOCOL}, request declared {}",
                    request.protocol
                )),
            }),
        )
            .into_response();
    }

    let job_id = state.next_job_id();
    let queue_position = {
        let mut jobs = state.lock_jobs();
        let waiting = jobs.values().filter(|j| !j.state.is_terminal()).count() as u32;
        jobs.insert(
            job_id.clone(),
            JobStatus {
                job_id: job_id.clone(),
                state: JobState::Queued,
                message: None,
                result: None,
            },
        );
        waiting
    };

    // Accept now, compute later: the client gets an id it can poll or
    // reattach to, and a long fold does not hold a connection open.
    let worker_state = state.clone();
    let worker_job_id = job_id.clone();
    tokio::spawn(async move {
        {
            let mut jobs = worker_state.lock_jobs();
            if let Some(job) = jobs.get_mut(&worker_job_id) {
                job.state = JobState::Running;
            }
        }

        let outcome = worker_state.backend.fold(request.job).await;

        let mut jobs = worker_state.lock_jobs();
        if let Some(job) = jobs.get_mut(&worker_job_id) {
            match outcome {
                Ok(mut result) => {
                    result.job_id = worker_job_id.clone();
                    job.state = JobState::Succeeded;
                    job.result = Some(result);
                }
                Err(error) => {
                    job.state = JobState::Failed;
                    job.message = Some(error.to_string());
                }
            }
        }
    });

    (
        StatusCode::ACCEPTED,
        Json(JobAccepted {
            job_id,
            queue_position: Some(queue_position),
        }),
    )
        .into_response()
}

async fn status(
    State(state): State<ServiceState>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    // A poisoned table still holds its jobs, so recover rather than report
    // `unknown_job` for a job that exists.
    let found = state.lock_jobs().get(&job_id).cloned();

    match found {
        Some(job) => (StatusCode::OK, Json(job)).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(ErrorBody {
                error: "unknown_job".to_string(),
                detail: Some(format!("no job {job_id}")),
            }),
        )
            .into_response(),
    }
}

/// Bind and serve until the process is stopped.
pub async fn serve(
    addr: SocketAddr,
    backend: Arc<dyn FoldBackend>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app(backend)).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{routes, ChainSpec, Health, JobSpec, MoleculeKind};
    use crate::fold::FixtureBackend;

    fn state() -> ServiceState {
        ServiceState::new(Arc::new(FixtureBackend::default()))
    }

    fn queued_job(job_id: &str) -> JobStatus {
        JobStatus {
            job_id: job_id.to_string(),
            state: JobState::Queued,
            message: None,
            result: None,
        }
    }

    fn job_spec() -> JobSpec {
        JobSpec {
            name: "poison-check".to_string(),
            model: "fixture".to_string(),
            chains: vec![ChainSpec {
                label: "H".to_string(),
                sequence: "QVQL".to_string(),
                kind: MoleculeKind::Protein,
                designed: true,
            }],
            seeds: vec![1],
            diffusion_samples: 1,
            diffusion_steps: 8,
            use_msa: false,
            need_atom_confidence: false,
        }
    }

    /// Panic on a spawned thread while holding the job table lock. That is the
    /// only way the mutex becomes poisoned. The panic stays on its thread:
    /// `join` returns `Err` and the test continues.
    fn poison_the_job_table(state: &ServiceState) {
        let jobs = Arc::clone(&state.jobs);
        let thread = std::thread::spawn(move || {
            let _held = jobs.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            panic!("a worker died holding the job table");
        });
        assert!(thread.join().is_err(), "the thread should have panicked");
    }

    /// Serve `state` on an ephemeral port and return its base URL.
    async fn serve_state(state: ServiceState) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let address = listener.local_addr().expect("address");
        tokio::spawn(async move {
            let _ = axum::serve(listener, router(state)).await;
        });
        format!("http://{address}")
    }

    /// The old fallback reported `queue_depth: 0` on a poisoned table, so a
    /// dead worker looked like an idle service. The count must come from the
    /// recovered table instead.
    #[tokio::test]
    async fn health_after_a_poisoning_panic_answers_with_the_real_queue_depth() {
        let state = state();
        state
            .jobs
            .lock()
            .expect("fresh lock")
            .insert("job-000001".to_string(), queued_job("job-000001"));
        poison_the_job_table(&state);

        let url = serve_state(state).await;
        let response = reqwest::get(format!("{url}{}", routes::HEALTH))
            .await
            .expect("health answers");
        assert_eq!(response.status(), StatusCode::OK);
        let health: Health = response.json().await.expect("health body");
        assert_eq!(health.queue_depth, 1);
    }

    /// A poisoned table must not take the service down for every later
    /// request: submit still accepts work after the poisoning panic.
    #[tokio::test]
    async fn submit_after_a_poisoning_panic_is_accepted_not_refused() {
        let state = state();
        poison_the_job_table(&state);

        let url = serve_state(state).await;
        let response = reqwest::Client::new()
            .post(format!("{url}{}", routes::FOLD))
            .json(&FoldRequest::new(job_spec()))
            .send()
            .await
            .expect("submit answers");
        assert_eq!(response.status(), StatusCode::ACCEPTED);
    }

    /// The status route used to turn a lock failure into a 404, claiming the
    /// job never existed. A job recorded before the panic is still found.
    #[tokio::test]
    async fn a_job_recorded_before_a_poisoning_panic_is_still_found() {
        let state = state();
        state
            .jobs
            .lock()
            .expect("fresh lock")
            .insert("job-000001".to_string(), queued_job("job-000001"));
        poison_the_job_table(&state);

        let url = serve_state(state).await;
        let response = reqwest::get(format!("{url}{}", routes::job("job-000001")))
            .await
            .expect("status answers");
        assert_eq!(response.status(), StatusCode::OK);
        let status: JobStatus = response.json().await.expect("status body");
        assert_eq!(status.job_id, "job-000001");
        assert_eq!(status.state, JobState::Queued);
    }
}
