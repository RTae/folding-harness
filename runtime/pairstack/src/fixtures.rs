//! Test-only loader for the oracle fixtures under
//! `tools/pairstack-oracle/fixtures/`.
//!
//! Reads `manifest.json`, then each headerless little-endian float32 file, and
//! exposes tensors by name with their shapes. The validation mirrors
//! `tools/pairstack-oracle/reader/src/main.rs`: the file must hold exactly
//! `prod(shape) * 4` bytes and every value must be finite. That logic is
//! copied rather than shared, because importing the reader would couple the
//! runtime crate to a development tool, and the crate is meant to stay
//! movable and dependency-free.
//!
//! The directory is found through `PAIRSTACK_FIXTURES`, or by walking up from
//! the crate manifest to the repository tree that holds the oracle. Moving the
//! crate inside the repository therefore does not break the tests.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::tensor::Tensor;
use crate::tri_attn::TriangleAttnWeights;
use crate::tri_mul::TriangleMulWeights;
use crate::Error;

#[derive(Deserialize)]
struct Manifest {
    meta: Meta,
    tensors: BTreeMap<String, Entry>,
    weights: BTreeMap<String, BTreeMap<String, Entry>>,
}

#[derive(Deserialize)]
struct Meta {
    constants: Constants,
}

#[derive(Deserialize)]
struct Constants {
    triangle_attention_n_head: usize,
    triangle_attention_d_head: usize,
}

#[derive(Deserialize)]
struct Entry {
    path: String,
    shape: Vec<usize>,
    dtype: String,
}

/// The loaded fixture manifest, able to produce any dumped tensor on demand.
pub struct Fixtures {
    root: PathBuf,
    manifest: Manifest,
}

impl Fixtures {
    /// Opens the fixture directory at `root`.
    pub fn open(root: &Path) -> Result<Self, Error> {
        let manifest_path = root.join("manifest.json");
        let text = std::fs::read_to_string(&manifest_path).map_err(|e| Error::Fixture {
            path: manifest_path.clone(),
            reason: format!("cannot read the manifest ({e})"),
        })?;
        let manifest: Manifest = serde_json::from_str(&text).map_err(|e| Error::Fixture {
            path: manifest_path,
            reason: format!("manifest.json does not match the expected schema ({e})"),
        })?;
        Ok(Self {
            root: root.to_path_buf(),
            manifest,
        })
    }

    /// Finds and opens the fixture directory. Panics with a message that says
    /// where it looked when the fixtures are absent.
    pub fn load() -> Self {
        let root = Self::discover();
        Self::open(&root).unwrap_or_else(|e| panic!("{e}"))
    }

    fn discover() -> PathBuf {
        if let Ok(dir) = std::env::var("PAIRSTACK_FIXTURES") {
            let path = PathBuf::from(&dir);
            assert!(
                path.join("manifest.json").is_file(),
                "PAIRSTACK_FIXTURES is set to {dir}, but that directory holds no \
                 manifest.json. Point the variable at the fixture directory, or unset \
                 it to search from the crate manifest upward."
            );
            return path;
        }
        let mut dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        loop {
            let candidate = dir.join("tools/pairstack-oracle/fixtures");
            if candidate.join("manifest.json").is_file() {
                return candidate;
            }
            match dir.parent() {
                Some(parent) => dir = parent,
                None => panic!(
                    "cannot find tools/pairstack-oracle/fixtures at or above {}. Set \
                     PAIRSTACK_FIXTURES to the directory that holds manifest.json.",
                    env!("CARGO_MANIFEST_DIR")
                ),
            }
        }
    }

    /// The number of activation tensors the manifest names.
    pub fn tensor_count(&self) -> usize {
        self.manifest.tensors.len()
    }

    /// The number of weight tensors the manifest names.
    pub fn weight_count(&self) -> usize {
        self.manifest.weights.values().map(BTreeMap::len).sum()
    }

    /// Loads a dumped activation tensor by its manifest name, for example
    /// `tri_attn_end.bias`.
    pub fn tensor(&self, name: &str) -> Tensor {
        self.try_tensor(name).unwrap_or_else(|e| panic!("{e}"))
    }

    /// The `Result` form of [`Fixtures::tensor`], for tests that pin the
    /// loader's rejections.
    pub fn try_tensor(&self, name: &str) -> Result<Tensor, Error> {
        let entry = self
            .manifest
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("manifest.json names no tensor {name}"));
        self.read(entry)
    }

    /// Assembles the dumped weights of one triangle multiplication.
    pub fn triangle_mul_weights(&self, op: &str) -> TriangleMulWeights {
        TriangleMulWeights {
            norm_in_weight: self.weight(op, "norm_in.weight").data().to_vec(),
            norm_in_bias: self.weight(op, "norm_in.bias").data().to_vec(),
            p_in_weight: self.weight(op, "p_in.weight"),
            g_in_weight: self.weight(op, "g_in.weight"),
            norm_out_weight: self.weight(op, "norm_out.weight").data().to_vec(),
            norm_out_bias: self.weight(op, "norm_out.bias").data().to_vec(),
            p_out_weight: self.weight(op, "p_out.weight"),
            g_out_weight: self.weight(op, "g_out.weight"),
        }
    }

    /// Assembles the dumped weights of one triangle attention, with the head
    /// count and head width the manifest records.
    pub fn triangle_attn_weights(&self, op: &str) -> TriangleAttnWeights {
        TriangleAttnWeights {
            norm_weight: self.weight(op, "norm.weight").data().to_vec(),
            norm_bias: self.weight(op, "norm.bias").data().to_vec(),
            to_q_weight: self.weight(op, "to_q.weight"),
            to_k_weight: self.weight(op, "to_k.weight"),
            to_v_weight: self.weight(op, "to_v.weight"),
            to_b_weight: self.weight(op, "to_b.weight"),
            to_g_weight: self.weight(op, "to_g.weight"),
            to_g_bias: self.weight(op, "to_g.bias").data().to_vec(),
            to_out_weight: self.weight(op, "to_out.weight"),
            to_out_bias: self.weight(op, "to_out.bias").data().to_vec(),
            n_head: self.manifest.meta.constants.triangle_attention_n_head,
            d_head: self.manifest.meta.constants.triangle_attention_d_head,
        }
    }

    /// Loads one weight tensor of one operation by its `state_dict` key, for
    /// example `("tri_mul_outgoing", "p_in.weight")`.
    pub fn weight(&self, op: &str, key: &str) -> Tensor {
        let entry = self
            .manifest
            .weights
            .get(op)
            .unwrap_or_else(|| panic!("manifest.json names no operation {op}"))
            .get(key)
            .unwrap_or_else(|| panic!("operation {op} has no weight {key}"));
        self.read(entry).unwrap_or_else(|e| panic!("{e}"))
    }

    fn read(&self, entry: &Entry) -> Result<Tensor, Error> {
        let path = self.root.join(&entry.path);
        if entry.dtype != "float32" {
            return Err(Error::Fixture {
                path,
                reason: format!("manifest dtype is {}, expected float32", entry.dtype),
            });
        }
        let count: usize = entry.shape.iter().product();
        let bytes = std::fs::read(&path).map_err(|e| Error::Fixture {
            path: path.clone(),
            reason: format!("cannot read the file ({e})"),
        })?;
        if bytes.len() != count * 4 {
            return Err(Error::Fixture {
                path,
                reason: format!(
                    "the file holds {} bytes but shape {:?} needs {}",
                    bytes.len(),
                    entry.shape,
                    count * 4
                ),
            });
        }
        let mut data = Vec::with_capacity(count);
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            let value = f32::from_le_bytes(chunk.try_into().expect("chunk of four bytes"));
            if !value.is_finite() {
                return Err(Error::Fixture {
                    path,
                    reason: format!("element {i} is {value}, expected a finite f32"),
                });
            }
            data.push(value);
        }
        Tensor::new(entry.shape.clone(), data)
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::Fixtures;

    #[test]
    fn the_manifest_accounts_for_every_fixture_file() {
        let fx = Fixtures::load();
        assert_eq!(fx.tensor_count(), 64, "activation tensors");
        assert_eq!(fx.weight_count(), 36, "weight tensors");
    }

    #[test]
    fn all_four_operations_share_one_dumped_input() {
        let fx = Fixtures::load();
        let first = fx.tensor("tri_mul_outgoing.input");
        for name in [
            "tri_mul_incoming.input",
            "tri_attn_start.input",
            "tri_attn_end.input",
        ] {
            assert_eq!(
                fx.tensor(name).data(),
                first.data(),
                "{name} differs from tri_mul_outgoing.input; the oracle recipe \
                 feeds one shared pair to all four operations"
            );
        }
    }

    #[test]
    fn the_loader_rejects_a_file_whose_size_disagrees_with_the_manifest() {
        let dir = scratch_dir("size");
        write_min_manifest(&dir);
        std::fs::write(dir.join("t.f32"), [0u8; 4]).unwrap();
        let fx = Fixtures::open(&dir).unwrap();
        let err = fx.try_tensor("t").unwrap_err();
        let message = err.to_string();
        assert!(message.contains("t.f32"), "{message}");
        assert!(message.contains("4 bytes"), "{message}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_loader_rejects_a_non_finite_value() {
        let dir = scratch_dir("finite");
        write_min_manifest(&dir);
        // Two f32 values: a NaN and a 1.0.
        let mut bytes = f32::NAN.to_le_bytes().to_vec();
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        std::fs::write(dir.join("t.f32"), bytes).unwrap();
        let fx = Fixtures::open(&dir).unwrap();
        let err = fx.try_tensor("t").unwrap_err();
        assert!(err.to_string().contains("NaN"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn scratch_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("pairstack-fixtures-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_min_manifest(dir: &Path) {
        std::fs::write(
            dir.join("manifest.json"),
            r#"{"meta": {"constants": {"triangle_attention_n_head": 4, "triangle_attention_d_head": 32}},
                "tensors": {"t": {"path": "t.f32", "shape": [2], "dtype": "float32"}},
                "weights": {}}"#,
        )
        .unwrap();
    }
}
