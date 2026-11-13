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
    /// Whether the model calls behind these totals were counted.
    ///
    /// `Some(true)`: the source counted its calls, so `calls` is a measurement
    /// and `calls: 0` means no model was called. The token fields can still
    /// undercount under a counted source, because a gateway may strip the
    /// usage body from a response, which leaves the call counted and its tokens
    /// absent. `Some(false)`: the source does not count, so calls may have
    /// happened that these totals do not show. `None`: nobody said. A
    /// `run.json` written before this field existed reads back as `None`, which
    /// stays unknown rather than turning into a claim either way.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calls_counted: Option<bool>,
}

impl Usage {
    /// Zero spend, counted. The source made no model calls and says so, which
    /// is a measurement of zero rather than silence about it.
    pub fn no_model_calls() -> Self {
        Usage {
            calls_counted: Some(true),
            ..Usage::default()
        }
    }

    /// Zero spend, not counted. The source does not count model calls, so any
    /// call it made is missing from these totals. Reporting zeros without
    /// this flag would claim the calls were free.
    pub fn uncounted() -> Self {
        Usage {
            calls_counted: Some(false),
            ..Usage::default()
        }
    }

    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    /// Both snapshots added together, field by field. One source may hold the
    /// fold counts while another reports the model spend, and a resumed run
    /// adds what the recorded run spent to what this one spends. The manifest
    /// carries the total.
    ///
    /// The sum is counted only when both sources counted. One source that
    /// does not count leaves part of the total unmeasured, so the sum reports
    /// uncounted. Failing that, one source that never said, such as a
    /// recorded manifest from before `calls_counted` existed, leaves the sum
    /// unknown.
    pub fn summed_with(self, other: Usage) -> Usage {
        Usage {
            calls: self.calls + other.calls,
            input_tokens: self.input_tokens + other.input_tokens,
            output_tokens: self.output_tokens + other.output_tokens,
            folds: self.folds + other.folds,
            fold_ms: self.fold_ms + other.fold_ms,
            calls_counted: match (self.calls_counted, other.calls_counted) {
                (Some(false), _) | (_, Some(false)) => Some(false),
                (Some(true), Some(true)) => Some(true),
                _ => None,
            },
        }
    }

    /// Nothing to report: every counter is zero and no uncounted model call
    /// is on record. A snapshot that says its model spend was not counted is
    /// never empty, because silence about it would read as zero.
    pub fn is_empty(&self) -> bool {
        self.calls == 0
            && self.input_tokens == 0
            && self.output_tokens == 0
            && self.folds == 0
            && self.fold_ms == 0
            && self.calls_counted != Some(false)
    }

    /// One line for a footer or a summary.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        match self.calls_counted {
            Some(true) if self.calls > 0 => parts.push(format!(
                "{} call(s), {} in / {} out tokens",
                self.calls, self.input_tokens, self.output_tokens
            )),
            Some(true) => {}
            Some(false) => parts.push("an uncounted number of model calls".to_string()),
            None if self.calls > 0 => parts.push(format!(
                "{} call(s), {} in / {} out tokens, counting not recorded",
                self.calls, self.input_tokens, self.output_tokens
            )),
            None => parts.push("model calls of unknown count".to_string()),
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

    /// The meter is a counting device: every call and fold it shows went
    /// through `record_call` or `record_fold`, so its snapshots report their
    /// model spend as counted. A zero call count from a meter means the meter
    /// saw no model calls, not that counting was skipped.
    pub fn snapshot(&self) -> Usage {
        Usage {
            calls: self.inner.calls.load(Ordering::Relaxed),
            input_tokens: self.inner.input_tokens.load(Ordering::Relaxed),
            output_tokens: self.inner.output_tokens.load(Ordering::Relaxed),
            folds: self.inner.folds.load(Ordering::Relaxed),
            fold_ms: self.inner.fold_ms.load(Ordering::Relaxed),
            calls_counted: Some(true),
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
        assert_eq!(
            usage.calls_counted,
            Some(true),
            "a meter counts, so its snapshot says so"
        );
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
            calls_counted: Some(true),
        };
        let calls = Usage {
            calls: 2,
            input_tokens: 300,
            output_tokens: 40,
            folds: 0,
            fold_ms: 0,
            calls_counted: Some(true),
        };
        assert_eq!(
            folds.summed_with(calls),
            Usage {
                calls: 2,
                input_tokens: 300,
                output_tokens: 40,
                folds: 4,
                fold_ms: 900,
                calls_counted: Some(true),
            }
        );
        assert_eq!(folds.summed_with(Usage::no_model_calls()), folds);
    }

    #[test]
    fn a_sum_is_counted_only_when_every_source_counted() {
        let counted = Usage {
            calls: 2,
            calls_counted: Some(true),
            ..Usage::default()
        };
        // One source that does not count leaves part of the total unmeasured,
        // whichever side it sits on.
        assert_eq!(
            counted.summed_with(Usage::uncounted()).calls_counted,
            Some(false)
        );
        assert_eq!(
            Usage::uncounted().summed_with(counted).calls_counted,
            Some(false)
        );
        // One source that never said, such as a manifest from before the
        // field existed, leaves the total unknown rather than counted.
        assert_eq!(counted.summed_with(Usage::default()).calls_counted, None);
        assert_eq!(Usage::default().summed_with(counted).calls_counted, None);
        // Uncounted beats unknown: the total definitely misses spend.
        assert_eq!(
            Usage::uncounted()
                .summed_with(Usage::default())
                .calls_counted,
            Some(false)
        );
    }

    #[test]
    fn a_resumed_run_reports_both_spends_summed() {
        let recorded = Usage {
            calls: 2,
            input_tokens: 100,
            output_tokens: 40,
            folds: 5,
            fold_ms: 2500,
            calls_counted: Some(true),
        };
        let fresh = Usage {
            calls: 1,
            input_tokens: 10,
            output_tokens: 5,
            folds: 3,
            fold_ms: 900,
            calls_counted: Some(true),
        };
        assert_eq!(
            recorded.summed_with(fresh),
            Usage {
                calls: 3,
                input_tokens: 110,
                output_tokens: 45,
                folds: 8,
                fold_ms: 3400,
                calls_counted: Some(true),
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
    fn usage_from_a_manifest_that_predates_the_field_reads_as_unknown() {
        // The wire form an old run.json carries: counters, no calls_counted.
        let old = serde_json::json!({
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "folds": 4,
            "fold_ms": 900,
        });
        let usage: Usage = serde_json::from_value(old).expect("old wire form reads");
        assert_eq!(
            usage.calls_counted, None,
            "an old manifest must not be read as a claim either way"
        );
        assert_eq!(usage.folds, 4);

        let new = serde_json::json!({
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "folds": 0,
            "fold_ms": 0,
            "calls_counted": false,
        });
        let usage: Usage = serde_json::from_value(new).expect("new wire form reads");
        assert_eq!(usage.calls_counted, Some(false));
    }

    #[test]
    fn the_counted_field_serializes_only_when_somebody_said() {
        let json = serde_json::to_value(Usage::default()).expect("encodes");
        assert!(
            json.get("calls_counted").is_none(),
            "unknown stays absent on the wire, so a re-read stays unknown"
        );
        let json = serde_json::to_value(Usage::uncounted()).expect("encodes");
        assert_eq!(json["calls_counted"], serde_json::json!(false));
    }

    #[test]
    fn uncounted_spend_is_never_empty_and_never_silent() {
        let usage = Usage::uncounted();
        assert!(
            !usage.is_empty(),
            "silence about uncounted calls reads as zero"
        );
        assert_eq!(usage.summary(), "an uncounted number of model calls");
    }

    #[test]
    fn counted_zero_calls_says_nothing_about_models_rather_than_reporting_zero_tokens() {
        let usage = Usage {
            folds: 2,
            fold_ms: 400,
            ..Usage::no_model_calls()
        };
        assert!(!usage.is_empty());
        let summary = usage.summary();
        assert!(!summary.contains("call(s)"), "{summary}");
        assert!(!summary.contains("unknown"), "{summary}");
        assert!(!summary.contains("uncounted"), "{summary}");
        assert!(summary.contains("2 fold(s)"), "{summary}");
    }

    #[test]
    fn unknown_spend_says_unknown_in_both_of_its_shapes() {
        // No counters at all: the source never said whether calls happened.
        let unknown = Usage {
            folds: 3,
            fold_ms: 700,
            ..Usage::default()
        };
        let summary = unknown.summary();
        assert!(
            summary.contains("model calls of unknown count"),
            "{summary}"
        );

        // Counters from before the field existed: the numbers show, flagged.
        let inherited = Usage {
            calls: 5,
            input_tokens: 500,
            output_tokens: 90,
            ..Usage::default()
        };
        let summary = inherited.summary();
        assert!(summary.contains("5 call(s)"), "{summary}");
        assert!(summary.contains("counting not recorded"), "{summary}");
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
