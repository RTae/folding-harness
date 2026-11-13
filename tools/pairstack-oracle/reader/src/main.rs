//! Reads the pair-stack oracle fixtures and checks them against the manifest.
//!
//! The oracle proves that a Rust program can load every dumped tensor with no
//! Python runtime present. For each manifest entry this program checks that
//! the file holds exactly `prod(shape) * 4` bytes, decodes it as
//! little-endian f32, and rejects any non-finite value. It prints what it
//! loaded and exits nonzero on the first mismatch.
//!
//! Usage: `cargo run --release -- [FIXTURE_DIR]`, default `../fixtures`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

#[derive(Deserialize)]
struct Manifest {
    meta: Meta,
    tensors: BTreeMap<String, Entry>,
    weights: BTreeMap<String, BTreeMap<String, Entry>>,
}

#[derive(Deserialize)]
struct Meta {
    foundry_commit: String,
    seed: u64,
    python: String,
    torch: String,
}

#[derive(Deserialize)]
struct Entry {
    path: String,
    shape: Vec<usize>,
    dtype: String,
}

fn check_entry(root: &Path, name: &str, entry: &Entry) -> Result<usize> {
    if entry.dtype != "float32" {
        bail!(
            "{name}: manifest dtype is {}, expected float32",
            entry.dtype
        );
    }
    let count: usize = entry.shape.iter().product();
    let path = root.join(&entry.path);
    let bytes =
        std::fs::read(&path).with_context(|| format!("{name}: cannot read {}", path.display()))?;
    if bytes.len() != count * 4 {
        bail!(
            "{name}: {} holds {} bytes, shape {:?} needs {}",
            entry.path,
            bytes.len(),
            entry.shape,
            count * 4
        );
    }
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        let value = f32::from_le_bytes(chunk.try_into().unwrap());
        if !value.is_finite() {
            bail!("{name}: element {i} is {value}, expected a finite f32");
        }
    }
    println!("{name:<44} {:?} {} values", entry.shape, count);
    Ok(count)
}

fn main() -> Result<()> {
    let arg = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../fixtures".to_string());
    let root = PathBuf::from(arg);
    let manifest_path = root.join("manifest.json");
    let text = std::fs::read_to_string(&manifest_path)
        .with_context(|| format!("cannot read {}", manifest_path.display()))?;
    let manifest: Manifest =
        serde_json::from_str(&text).context("manifest.json does not match the expected schema")?;

    println!(
        "foundry {} seed {} python {} torch {}",
        manifest.meta.foundry_commit, manifest.meta.seed, manifest.meta.python, manifest.meta.torch
    );

    let mut tensor_count = 0usize;
    let mut values = 0usize;
    for (name, entry) in &manifest.tensors {
        values += check_entry(&root, name, entry)?;
        tensor_count += 1;
    }
    for (op, weights) in &manifest.weights {
        for (key, entry) in weights {
            values += check_entry(&root, &format!("{op}/{key}"), entry)?;
            tensor_count += 1;
        }
    }
    println!("loaded {tensor_count} tensors, {values} float32 values, all finite");
    Ok(())
}
