//! `fh` — the harness command line.
//!
//! This is the whole binary: parse the arguments, hand them to `cli::dispatch`.
//! Everything else lives in `cli/`, and the design itself lives in the library.

mod cli;

use anyhow::Result;
use clap::Parser;

#[tokio::main]
async fn main() -> Result<()> {
    cli::dispatch(cli::Cli::parse()).await
}
