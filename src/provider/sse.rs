//! Reading server-sent events from a streamed response.
//!
//! Both wires stream the same way: `data:` lines carrying JSON, events
//! separated by a blank line. Only the meaning of the JSON differs, so the
//! reading is shared and each wire supplies a handler.
//!
//! The buffer holds bytes rather than text on purpose. A chunk boundary can
//! fall in the middle of a multi-byte character, and decoding each chunk on
//! its own would corrupt it.

use futures_util::StreamExt;
use serde_json::Value;

use crate::agent::AgentError;

/// What a wire's handler decides about one event.
pub enum Step {
    /// Keep reading.
    Continue,
    /// Stop with this error, for an event that reports a failure.
    Fail(AgentError),
}

/// Read a response body as server-sent events, handing each `data:` payload to
/// `on_event`.
///
/// `[DONE]` and unparsable payloads are skipped: a keep-alive or a comment is
/// not an error, and neither is a field a wire has added since.
pub async fn read_events<F>(
    response: reqwest::Response,
    host: &str,
    mut on_event: F,
) -> Result<(), AgentError>
where
    F: FnMut(&Value) -> Step,
{
    let mut stream = response.bytes_stream();
    let mut buffer: Vec<u8> = Vec::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk
            .map_err(|error| AgentError::Provider(format!("stream from {host} broke: {error}")))?;
        buffer.extend_from_slice(&chunk);

        while let Some(at) = buffer.windows(2).position(|pair| pair == b"\n\n") {
            let event: Vec<u8> = buffer.drain(..at + 2).collect();
            let event = String::from_utf8_lossy(&event);

            for line in event.lines() {
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim();
                if data.is_empty() || data == "[DONE]" {
                    continue;
                }
                let Ok(value) = serde_json::from_str::<Value>(data) else {
                    continue;
                };
                if let Step::Fail(error) = on_event(&value) {
                    return Err(error);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serve a fixed body in chunks of a given size, and return its address.
    async fn serve(body: &'static str, chunk_size: usize) -> String {
        use axum::routing::post;
        use axum::Router;

        let app = Router::new().route(
            "/",
            post(move || async move {
                let chunks: Vec<Result<Vec<u8>, std::io::Error>> = body
                    .as_bytes()
                    .chunks(chunk_size)
                    .map(|chunk| Ok(chunk.to_vec()))
                    .collect();
                axum::body::Body::from_stream(futures_util::stream::iter(chunks))
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let address = listener.local_addr().expect("address");
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        format!("http://{address}/")
    }

    async fn collect(body: &'static str, chunk_size: usize) -> Result<Vec<String>, AgentError> {
        let url = serve(body, chunk_size).await;
        let response = reqwest::Client::new()
            .post(url)
            .send()
            .await
            .expect("request");

        let mut seen = Vec::new();
        read_events(response, "test", |value| {
            if value["type"] == "boom" {
                return Step::Fail(AgentError::Provider("boom".to_string()));
            }
            if let Some(text) = value["text"].as_str() {
                seen.push(text.to_string());
            }
            Step::Continue
        })
        .await?;
        Ok(seen)
    }

    const BODY: &str = "event: a\n\
data: {\"text\":\"Hello \"}\n\
\n\
event: ping\n\
data: {\"type\":\"ping\"}\n\
\n\
data: {\"text\":\"世界\"}\n\
\n\
data: [DONE]\n\
\n";

    #[tokio::test]
    async fn events_split_across_awkward_chunks_reassemble() {
        // Seven bytes cuts through JSON and through the three-byte characters.
        assert_eq!(collect(BODY, 7).await.unwrap(), vec!["Hello ", "世界"]);
    }

    #[tokio::test]
    async fn one_byte_at_a_time_still_reassembles() {
        assert_eq!(collect(BODY, 1).await.unwrap(), vec!["Hello ", "世界"]);
    }

    #[tokio::test]
    async fn the_whole_body_at_once_reads_the_same() {
        assert_eq!(
            collect(BODY, BODY.len()).await.unwrap(),
            vec!["Hello ", "世界"]
        );
    }

    const RUBBISH: &str = "data: not json\n\
\n\
: a comment\n\
\n\
data: {\"text\":\"kept\"}\n\
\n";

    #[tokio::test]
    async fn unreadable_payloads_and_comments_are_skipped_not_fatal() {
        assert_eq!(collect(RUBBISH, 5).await.unwrap(), vec!["kept"]);
    }

    const FAILING: &str = "data: {\"text\":\"before\"}\n\
\n\
data: {\"type\":\"boom\"}\n\
\n\
data: {\"text\":\"after\"}\n\
\n";

    #[tokio::test]
    async fn a_handler_can_stop_the_stream() {
        let error = collect(FAILING, 6)
            .await
            .expect_err("the handler failed it");
        assert!(error.to_string().contains("boom"), "{error}");
    }
}
