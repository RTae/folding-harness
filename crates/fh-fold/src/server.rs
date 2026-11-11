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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};

use fh_contract::{
    major_version, ErrorBody, FoldRequest, JobAccepted, JobState, JobStatus, PROTOCOL,
};

use crate::FoldBackend;

#[derive(Clone)]
pub struct ServiceState {
    backend: Arc<dyn FoldBackend>,
    jobs: Arc<Mutex<HashMap<String, JobStatus>>>,
    counter: Arc<AtomicU64>,
}

impl ServiceState {
    pub fn new(backend: Arc<dyn FoldBackend>) -> Self {
        ServiceState {
            backend,
            jobs: Arc::new(Mutex::new(HashMap::new())),
            counter: Arc::new(AtomicU64::new(1)),
        }
    }

    fn next_job_id(&self) -> String {
        format!("job-{:06}", self.counter.fetch_add(1, Ordering::SeqCst))
    }
}

pub fn app(backend: Arc<dyn FoldBackend>) -> Router {
    Router::new()
        .route(fh_contract::routes::HEALTH, get(health))
        .route(fh_contract::routes::FOLD, post(submit))
        .route("/v1/fold/:job_id", get(status))
        .with_state(ServiceState::new(backend))
}

async fn health(State(state): State<ServiceState>) -> impl IntoResponse {
    match state.backend.health().await {
        Ok(mut report) => {
            report.queue_depth = state
                .jobs
                .lock()
                .map(|jobs| {
                    jobs.values()
                        .filter(|job| !job.state.is_terminal())
                        .count() as u32
                })
                .unwrap_or(0);
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
        let mut jobs = match state.jobs.lock() {
            Ok(jobs) => jobs,
            Err(_) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorBody {
                        error: "job_table_poisoned".to_string(),
                        detail: None,
                    }),
                )
                    .into_response()
            }
        };
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
            if let Ok(mut jobs) = worker_state.jobs.lock() {
                if let Some(job) = jobs.get_mut(&worker_job_id) {
                    job.state = JobState::Running;
                }
            }
        }

        let outcome = worker_state.backend.fold(request.job).await;

        if let Ok(mut jobs) = worker_state.jobs.lock() {
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
    let found = state
        .jobs
        .lock()
        .ok()
        .and_then(|jobs| jobs.get(&job_id).cloned());

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
