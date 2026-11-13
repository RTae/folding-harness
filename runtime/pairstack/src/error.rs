//! The crate's error type.
//!
//! Written by hand rather than derived with `thiserror`, because the library
//! carries no dependencies at all. The numerical core is intended to compile
//! for other toolchains later, and every dependency is a portability question.

use std::fmt;
use std::path::PathBuf;

/// Everything that can fail in this crate.
///
/// Each variant names the tensor, file or operation at fault and says what
/// disagreed, so a failure points at the stage that broke.
#[derive(Debug)]
pub enum Error {
    /// A tensor buffer holds a number of values its shape cannot account for.
    Length {
        /// What was being built when the disagreement appeared.
        context: String,
        /// The shape the tensor was declared with.
        shape: Vec<usize>,
        /// The number of values the buffer actually holds.
        len: usize,
    },
    /// Two tensors met in an operation that requires equal shapes, or a
    /// parameter's length disagrees with the axis it applies to.
    Mismatch {
        /// The operation that refused the pair.
        context: String,
        /// What the operation required.
        expected: String,
        /// What it was given.
        got: String,
    },
    /// A fixture file is missing, mis-sized, undecodable or non-finite.
    Fixture {
        /// The file that failed to load.
        path: PathBuf,
        /// Why it was rejected.
        reason: String,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Length {
                context,
                shape,
                len,
            } => {
                let want: usize = shape.iter().product();
                write!(
                    f,
                    "{context}: shape {shape:?} holds {want} values but the buffer has {len}. \
                     Fix the code that produced this buffer."
                )
            }
            Error::Mismatch {
                context,
                expected,
                got,
            } => write!(f, "{context}: expected {expected}, got {got}."),
            Error::Fixture { path, reason } => write!(
                f,
                "{}: {reason}. Check the file against fixtures/manifest.json \
                 or regenerate the fixtures with tools/pairstack-oracle/generate_fixtures.py.",
                path.display()
            ),
        }
    }
}

impl std::error::Error for Error {}
