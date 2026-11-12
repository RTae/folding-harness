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

/// Handle one event's bytes: hand each `data:` payload to `on_event`.
///
/// `[DONE]` and unparsable payloads are skipped: a keep-alive or a comment is
/// not an error, and neither is a field a wire has added since. An event with
/// no `data:` line yields no handler call.
fn dispatch_event<F>(event: &[u8], on_event: &mut F) -> Result<(), AgentError>
where
    F: FnMut(&Value) -> Step,
{
    let event = String::from_utf8_lossy(event);

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
    Ok(())
}

/// The event delimiters the specification allows: a blank line written with
/// CRLF, LF or CR endings. The longest comes first only for readability. The
/// three cannot match at the same offset, because their first two bytes
/// differ, so the scan order carries no meaning.
const DELIMITERS: [&[u8]; 3] = [b"\r\n\r\n", b"\n\n", b"\r\r"];

/// The index just past the first complete event delimiter in `buffer`.
///
/// A partial delimiter at the tail does not match. Its missing bytes can
/// still arrive in the next chunk, and only a complete delimiter ends an
/// event.
fn find_delimiter_end(buffer: &[u8]) -> Option<usize> {
    (0..buffer.len()).find_map(|at| {
        DELIMITERS
            .into_iter()
            .find(|delimiter| buffer[at..].starts_with(delimiter))
            .map(|delimiter| at + delimiter.len())
    })
}

/// Read a response body as server-sent events, handing each `data:` payload to
/// `on_event`.
///
/// The blank line between events may be written with CRLF, LF or CR endings.
/// All three delimit an event.
///
/// A final event that arrives without its terminating blank line is flushed to
/// `on_event` when the stream ends, under the same skipping rules as a
/// delimited one.
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

        while let Some(end) = find_delimiter_end(&buffer) {
            let event: Vec<u8> = buffer.drain(..end).collect();
            dispatch_event(&event, &mut on_event)?;
        }
    }

    if !buffer.is_empty() {
        dispatch_event(&buffer, &mut on_event)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serve `app` on an ephemeral loopback port and return its address.
    async fn spawn_server(app: axum::Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let address = listener.local_addr().expect("address");
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        format!("http://{address}/")
    }

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
        spawn_server(app).await
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

    /// The last event has no terminating blank line, not even a newline.
    const UNTERMINATED: &str = "data: {\"text\":\"first\"}\n\
    \n\
    data: {\"text\":\"last\"}";

    #[tokio::test]
    async fn a_final_event_without_a_blank_line_still_arrives() {
        assert_eq!(
            collect(UNTERMINATED, 5).await.unwrap(),
            vec!["first", "last"]
        );
    }

    #[tokio::test]
    async fn an_unterminated_final_event_survives_one_byte_chunks() {
        assert_eq!(
            collect(UNTERMINATED, 1).await.unwrap(),
            vec!["first", "last"]
        );
    }

    #[tokio::test]
    async fn an_unterminated_final_event_survives_a_single_chunk() {
        assert_eq!(
            collect(UNTERMINATED, UNTERMINATED.len()).await.unwrap(),
            vec!["first", "last"]
        );
    }

    const TRAILING_NO_DATA: &str = "data: {\"text\":\"kept\"}\n\
    \n\
    event: ping\n";

    #[tokio::test]
    async fn a_trailing_fragment_without_a_data_line_yields_nothing() {
        assert_eq!(collect(TRAILING_NO_DATA, 5).await.unwrap(), vec!["kept"]);
    }

    const TRAILING_DONE: &str = "data: {\"text\":\"kept\"}\n\
    \n\
    data: [DONE]";

    #[tokio::test]
    async fn a_trailing_done_without_a_blank_line_is_not_an_event() {
        assert_eq!(collect(TRAILING_DONE, 5).await.unwrap(), vec!["kept"]);
    }

    const FAILING_TAIL: &str = "data: {\"text\":\"before\"}\n\
    \n\
    data: {\"type\":\"boom\"}";

    #[tokio::test]
    async fn a_handler_failure_on_the_flushed_event_is_returned() {
        let error = collect(FAILING_TAIL, 6)
            .await
            .expect_err("the handler failed it");
        assert!(error.to_string().contains("boom"), "{error}");
    }

    /// The same stream as `BODY`, written with CRLF line endings.
    const CRLF_BODY: &str = "event: a\r\n\
    data: {\"text\":\"Hello \"}\r\n\
    \r\n\
    event: ping\r\n\
    data: {\"type\":\"ping\"}\r\n\
    \r\n\
    data: {\"text\":\"世界\"}\r\n\
    \r\n\
    data: [DONE]\r\n\
    \r\n";

    #[tokio::test]
    async fn crlf_events_split_across_awkward_chunks_reassemble() {
        // Seven bytes cuts through JSON, through the three-byte characters
        // and through the four-byte delimiters.
        assert_eq!(collect(CRLF_BODY, 7).await.unwrap(), vec!["Hello ", "世界"]);
    }

    #[tokio::test]
    async fn crlf_events_reassemble_one_byte_at_a_time() {
        assert_eq!(collect(CRLF_BODY, 1).await.unwrap(), vec!["Hello ", "世界"]);
    }

    #[tokio::test]
    async fn the_whole_crlf_body_at_once_reads_the_same() {
        assert_eq!(
            collect(CRLF_BODY, CRLF_BODY.len()).await.unwrap(),
            vec!["Hello ", "世界"]
        );
    }

    /// Events delimited by `\r\r`, a blank line written with CR endings. Each
    /// event carries a single `data:` line, because `dispatch_event` splits an
    /// event's lines on `\n` and a CR-separated `event:` field would not be
    /// recognised. That is the existing per-event rule, not part of delimiter
    /// recognition.
    const CR_BODY: &str = "data: {\"text\":\"one\"}\r\rdata: {\"text\":\"two\"}\r\r";

    #[tokio::test]
    async fn cr_delimited_events_all_arrive() {
        assert_eq!(collect(CR_BODY, 1).await.unwrap(), vec!["one", "two"]);
        assert_eq!(
            collect(CR_BODY, CR_BODY.len()).await.unwrap(),
            vec!["one", "two"]
        );
    }

    /// Two CRLF-delimited events. The first `data:` line is 18 bytes, so
    /// chunk sizes of 19, 20 and 21 cut the first `\r\n\r\n` after the `\r`,
    /// after the `\r\n` and after the `\r\n\r`. A partial delimiter sits at
    /// the buffer tail until the next chunk completes it.
    const SPLIT_PAIR: &str = "data: {\"text\":\"x\"}\r\n\r\ndata: {\"text\":\"y\"}\r\n\r\n";

    #[tokio::test]
    async fn a_crlf_delimiter_split_across_chunks_still_delivers() {
        for chunk_size in [19, 20, 21] {
            assert_eq!(
                collect(SPLIT_PAIR, chunk_size).await.unwrap(),
                vec!["x", "y"],
                "chunk size {chunk_size}"
            );
        }
    }

    /// One body that mixes all three delimiters.
    const MIXED_DELIMITERS: &str = "data: {\"text\":\"lf\"}\n\
    \n\
    data: {\"text\":\"crlf\"}\r\n\
    \r\n\
    data: {\"text\":\"cr\"}\r\
    \r";

    #[tokio::test]
    async fn a_body_mixing_delimiters_delivers_every_event() {
        assert_eq!(
            collect(MIXED_DELIMITERS, 7).await.unwrap(),
            vec!["lf", "crlf", "cr"]
        );
    }

    #[tokio::test]
    async fn a_crlf_event_reaches_the_handler_before_the_stream_ends() {
        use axum::routing::post;
        use axum::Router;
        use futures_util::stream;
        use std::sync::Arc;
        use std::time::Duration;
        use tokio::sync::{mpsc, oneshot};

        const FIRST: &str = "data: {\"text\":\"first\"}\r\n\r\n";
        const SECOND: &str = "data: {\"text\":\"second\"}\r\n\r\n";

        // The test owns the response body's channel. The server sends the
        // first event and then whatever the channel carries, so the test
        // holds the stream open until the handler reports the first event,
        // and only then sends the rest and closes. A parser that dispatches
        // only when the stream ends never runs its handler while the test
        // holds the body open, so it fails the bounded wait below instead
        // of passing.
        let (body_tx, body_rx) = mpsc::channel::<Vec<u8>>(4);
        let body = Arc::new(tokio::sync::Mutex::new(Some(body_rx)));
        let app = Router::new().route(
            "/",
            post(move || {
                let body = Arc::clone(&body);
                async move {
                    let first = async { Ok::<Vec<u8>, std::io::Error>(FIRST.as_bytes().to_vec()) };
                    let rest = match body.lock().await.take() {
                        Some(body_rx) => stream::unfold(body_rx, |mut body_rx| async move {
                            body_rx
                                .recv()
                                .await
                                .map(|chunk| (Ok::<Vec<u8>, std::io::Error>(chunk), body_rx))
                        })
                        .boxed(),
                        None => stream::empty().boxed(),
                    };
                    axum::body::Body::from_stream(stream::once(first).chain(rest))
                }
            }),
        );
        let url = spawn_server(app).await;

        let response = reqwest::Client::new()
            .post(url)
            .send()
            .await
            .expect("request");

        let (ack_tx, ack_rx) = oneshot::channel::<()>();
        let reader = tokio::spawn(async move {
            let mut ack = Some(ack_tx);
            let mut seen = Vec::new();
            let result = read_events(response, "test", |value| {
                let Some(text) = value["text"].as_str() else {
                    return Step::Continue;
                };
                seen.push(text.to_string());
                if text == "first" {
                    if let Some(ack) = ack.take() {
                        let _ = ack.send(());
                    }
                }
                Step::Continue
            })
            .await;
            (result, seen)
        });

        // The timeout is a deadlock guard, not a timing assertion. Waiting
        // for the handler's ack needs a bound because a regression to
        // end-of-stream dispatch would otherwise block this test forever.
        // Ten seconds is far beyond a healthy run, which acks as soon as
        // the first chunk arrives.
        let acked = tokio::time::timeout(Duration::from_secs(10), ack_rx)
            .await
            .is_ok();

        if !acked {
            // Close the body so the reader finishes and the failure can
            // name what the handler did see.
            drop(body_tx);
            let (_result, seen) = reader.await.expect("reader task");
            panic!(
                "incremental dispatch did not happen: the handler never saw \
                 the first event while the stream was open, and saw {seen:?} \
                 only once closing the stream flushed the buffer"
            );
        }

        body_tx
            .send(SECOND.as_bytes().to_vec())
            .await
            .expect("body channel open");
        drop(body_tx);

        let (result, seen) = reader.await.expect("reader task");
        result.unwrap();
        assert_eq!(seen, vec!["first", "second"]);
    }
}
