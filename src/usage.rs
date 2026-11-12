//! What a run spent.
//!
//! Two things cost money here and neither was being counted: tokens sent to a
//! model, and time on a folding backend. The open harnesses in this space
//! generally meter neither, which makes "was that policy worth it" a question
//! nobody can answer. A manifest that records the seed and the config but not
//! the bill is only half a record.
//!
//! Counters are atomic and shared, so a provider can record a call from
//! whichever task made it while the interface reads totals without locking.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// A snapshot of what has been spent.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Model calls that returned something.
    pub calls: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Folds submitted to a backend.
    pub folds: u64,
    /// Time backends reported for those folds.
    pub fold_ms: u64,
}

impl Usage {
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    /// Both snapshots added together, field by field. One source may hold the
    /// fold counts while another reports the model spend, and a resumed run
    /// adds what the recorded run spent to what this one spends. The manifest
    /// carries the total.
    pub fn summed_with(self, other: Usage) -> Usage {
        Usage {
            calls: self.calls + other.calls,
            input_tokens: self.input_tokens + other.input_tokens,
            output_tokens: self.output_tokens + other.output_tokens,
            folds: self.folds + other.folds,
            fold_ms: self.fold_ms + other.fold_ms,
        }
    }

    pub fn is_empty(&self) -> bool {
        *self == Usage::default()
    }

    /// One line for a footer or a summary.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if self.calls > 0 {
            parts.push(format!(
                "{} call(s), {} in / {} out tokens",
                self.calls, self.input_tokens, self.output_tokens
            ));
        }
        if self.folds > 0 {
            parts.push(format!(
                "{} fold(s), {:.1}s of backend time",
                self.folds,
                self.fold_ms as f64 / 1000.0
            ));
        }
        if parts.is_empty() {
            "nothing spent yet".to_string()
        } else {
            parts.join("; ")
        }
    }
}

/// A shared, thread-safe tally.
///
/// Cloning shares the counters rather than copying them, so a provider handed
/// a clone still reports into the same totals.
#[derive(Debug, Clone, Default)]
pub struct Meter {
    inner: Arc<Counters>,
}

#[derive(Debug, Default)]
struct Counters {
    calls: AtomicU64,
    input_tokens: AtomicU64,
    output_tokens: AtomicU64,
    folds: AtomicU64,
    fold_ms: AtomicU64,
}

impl Meter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one model call. Token counts are optional because not every
    /// endpoint reports them, and a gateway may strip the field.
    pub fn record_call(&self, input_tokens: Option<u64>, output_tokens: Option<u64>) {
        self.inner.calls.fetch_add(1, Ordering::Relaxed);
        if let Some(tokens) = input_tokens {
            self.inner.input_tokens.fetch_add(tokens, Ordering::Relaxed);
        }
        if let Some(tokens) = output_tokens {
            self.inner
                .output_tokens
                .fetch_add(tokens, Ordering::Relaxed);
        }
    }

    pub fn record_fold(&self, elapsed_ms: u64) {
        self.inner.folds.fetch_add(1, Ordering::Relaxed);
        self.inner.fold_ms.fetch_add(elapsed_ms, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> Usage {
        Usage {
            calls: self.inner.calls.load(Ordering::Relaxed),
            input_tokens: self.inner.input_tokens.load(Ordering::Relaxed),
            output_tokens: self.inner.output_tokens.load(Ordering::Relaxed),
            folds: self.inner.folds.load(Ordering::Relaxed),
            fold_ms: self.inner.fold_ms.load(Ordering::Relaxed),
        }
    }
}

/// Token counts from a response body, whichever wire it came from.
///
/// Anthropic reports `usage.input_tokens` and `usage.output_tokens`; OpenAI
/// reports `usage.prompt_tokens` and `usage.completion_tokens`. Both are
/// optional: a gateway may not pass them through, and a missing count is left
/// uncounted rather than guessed.
pub fn tokens_from_body(body: &serde_json::Value) -> (Option<u64>, Option<u64>) {
    let usage = &body["usage"];
    let read =
        |names: [&str; 2]| -> Option<u64> { names.iter().find_map(|name| usage[*name].as_u64()) };
    (
        read(["input_tokens", "prompt_tokens"]),
        read(["output_tokens", "completion_tokens"]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_meter_adds_up_and_shares_its_counters_when_cloned() {
        let meter = Meter::new();
        let other = meter.clone();

        meter.record_call(Some(100), Some(20));
        other.record_call(Some(50), None);
        other.record_fold(1500);

        let usage = meter.snapshot();
        assert_eq!(usage.calls, 2, "a clone reports into the same totals");
        assert_eq!(usage.input_tokens, 150);
        assert_eq!(usage.output_tokens, 20, "a missing count is not guessed");
        assert_eq!(usage.folds, 1);
        assert_eq!(usage.fold_ms, 1500);
        assert_eq!(usage.total_tokens(), 170);
        assert_eq!(meter.snapshot(), other.snapshot());
    }

    #[test]
    fn summed_with_adds_every_counter_and_leaves_the_other_snapshot_alone() {
        let folds = Usage {
            calls: 0,
            input_tokens: 0,
            output_tokens: 0,
            folds: 4,
            fold_ms: 900,
        };
        let calls = Usage {
            calls: 2,
            input_tokens: 300,
            output_tokens: 40,
            folds: 0,
            fold_ms: 0,
        };
        assert_eq!(
            folds.summed_with(calls),
            Usage {
                calls: 2,
                input_tokens: 300,
                output_tokens: 40,
                folds: 4,
                fold_ms: 900,
            }
        );
        assert_eq!(folds.summed_with(Usage::default()), folds);
    }

    #[test]
    fn a_resumed_run_reports_both_spends_summed() {
        let recorded = Usage {
            calls: 2,
            input_tokens: 100,
            output_tokens: 40,
            folds: 5,
            fold_ms: 2500,
        };
        let fresh = Usage {
            calls: 1,
            input_tokens: 10,
            output_tokens: 5,
            folds: 3,
            fold_ms: 900,
        };
        assert_eq!(
            recorded.summed_with(fresh),
            Usage {
                calls: 3,
                input_tokens: 110,
                output_tokens: 45,
                folds: 8,
                fold_ms: 3400,
            }
        );
    }

    #[test]
    fn an_untouched_meter_reports_nothing_rather_than_zeroes_dressed_up() {
        let usage = Meter::new().snapshot();
        assert!(usage.is_empty());
        assert_eq!(usage.summary(), "nothing spent yet");
    }

    #[test]
    fn a_summary_names_both_kinds_of_spend() {
        let meter = Meter::new();
        meter.record_call(Some(1200), Some(340));
        meter.record_fold(2500);
        let summary = meter.snapshot().summary();
        assert!(summary.contains("1 call(s)"), "{summary}");
        assert!(summary.contains("1200 in"), "{summary}");
        assert!(summary.contains("1 fold(s)"), "{summary}");
        assert!(summary.contains("2.5s"), "{summary}");
    }

    #[test]
    fn token_counts_are_read_from_either_wire() {
        let anthropic = serde_json::json!({
            "usage": {"input_tokens": 120, "output_tokens": 45}
        });
        assert_eq!(tokens_from_body(&anthropic), (Some(120), Some(45)));

        let openai = serde_json::json!({
            "usage": {"prompt_tokens": 200, "completion_tokens": 60, "total_tokens": 260}
        });
        assert_eq!(tokens_from_body(&openai), (Some(200), Some(60)));
    }

    #[test]
    fn a_response_without_usage_counts_nothing_rather_than_zero() {
        // A gateway that strips the field must not look like a free call.
        let bare = serde_json::json!({"content": []});
        assert_eq!(tokens_from_body(&bare), (None, None));

        let partial = serde_json::json!({"usage": {"input_tokens": 10}});
        assert_eq!(tokens_from_body(&partial), (Some(10), None));
    }
}
