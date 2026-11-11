//! HTTP client for a compute service speaking `fh.fold/v1`.
//!
//! Submission is accept-then-poll rather than one long blocking request. That
//! costs a little protocol complexity and buys three things a blocking POST
//! cannot: a fold that outlives the client process, a job id to reattach to
//! after a crash, and a queue the service can report on.

use std::time::{Duration, Instant};

use async_trait::async_trait;
use fh_contract::{
    major_version, routes, FoldRequest, FoldResult, Health, JobAccepted, JobSpec, JobState,
    JobStatus, PROTOCOL,
};

use crate::{FoldBackend, FoldError};

#[derive(Debug, Clone)]
pub struct HttpBackend {
    client: reqwest::Client,
    base: String,
    token: Option<String>,
    poll: Duration,
    timeout: Duration,
    name: String,
}

impl HttpBackend {
    pub fn new(
        base: impl Into<String>,
        token: Option<String>,
        poll_secs: f64,
        timeout_secs: u64,
    ) -> Result<Self, FoldError> {
        let base = base.into().trim_end_matches('/').to_string();
        let client = reqwest::Client::builder()
            // Per-request timeout only; the overall wait is bounded by
            // `timeout` across polls.
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| FoldError::Transport(e.to_string()))?;
        Ok(HttpBackend {
            client,
            name: format!("service({base})"),
            base,
            token,
            poll: Duration::from_secs_f64(poll_secs.max(0.1)),
            timeout: Duration::from_secs(timeout_secs.max(1)),
        })
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base, path)
    }

    fn authorized(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.token {
            Some(token) => builder.bearer_auth(token),
            None => builder,
        }
    }

    async fn send<T: serde::de::DeserializeOwned>(
        &self,
        builder: reqwest::RequestBuilder,
    ) -> Result<T, FoldError> {
        let response = self
            .authorized(builder)
            .send()
            .await
            .map_err(|e| FoldError::Transport(e.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| FoldError::Transport(e.to_string()))?;

        if !status.is_success() {
            return Err(FoldError::Service {
                status: status.as_u16(),
                body: body.chars().take(512).collect(),
            });
        }

        serde_json::from_str(&body).map_err(|e| FoldError::Malformed(format!("{e}: {body:.256}")))
    }

    /// Fetch one job's status. Public so a `reattach` command can poll a job
    /// submitted by an earlier process.
    pub async fn job_status(&self, job_id: &str) -> Result<JobStatus, FoldError> {
        self.send(self.client.get(self.url(&routes::job(job_id))))
            .await
    }
}

#[async_trait]
impl FoldBackend for HttpBackend {
    fn name(&self) -> &str {
        &self.name
    }

    async fn health(&self) -> Result<Health, FoldError> {
        let health: Health = self.send(self.client.get(self.url(routes::HEALTH))).await?;
        if major_version(&health.protocol) != major_version(PROTOCOL) {
            return Err(FoldError::Protocol {
                service: health.protocol.clone(),
                harness: PROTOCOL.to_string(),
            });
        }
        Ok(health)
    }

    async fn fold(&self, job: JobSpec) -> Result<FoldResult, FoldError> {
        let accepted: JobAccepted = self
            .send(
                self.client
                    .post(self.url(routes::FOLD))
                    .json(&FoldRequest::new(job)),
            )
            .await?;

        let started = Instant::now();
        loop {
            let status = self.job_status(&accepted.job_id).await?;
            match status.state {
                JobState::Succeeded => {
                    return status.result.ok_or_else(|| {
                        FoldError::Malformed(format!(
                            "job {} succeeded without a result",
                            accepted.job_id
                        ))
                    });
                }
                JobState::Failed => {
                    return Err(FoldError::Rejected(
                        status
                            .message
                            .unwrap_or_else(|| "service reported failure".to_string()),
                    ));
                }
                JobState::Queued | JobState::Running => {
                    if started.elapsed() > self.timeout {
                        return Err(FoldError::Timeout {
                            job_id: accepted.job_id,
                            secs: self.timeout.as_secs(),
                        });
                    }
                    tokio::time::sleep(self.poll).await;
                }
            }
        }
    }
}
