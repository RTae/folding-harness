//! A flat row-major f32 tensor with an explicit shape.
//!
//! Hand-rolled rather than built on `ndarray`. The fixture shapes are small,
//! the largest being (1, 13, 13, 128), and the operations need a fixed handful
//! of layout moves: reshape, transpose of the last two axes, and a split of
//! the last axis. A dependency-free core keeps a later GPU toolchain port to
//! adding a backend beside this code rather than replacing an array crate.
//!
//! Storage is always row-major (C order), matching the fixture files, so a
//! loaded tensor needs no reorder and a dump needs none either.

use crate::Error;

/// A dense f32 tensor in row-major order.
#[derive(Debug, Clone)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    /// Builds a tensor, checking that the buffer holds exactly `prod(shape)`
    /// values. Use this for every tensor that comes from outside the crate,
    /// such as a fixture file or a weight buffer.
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, Error> {
        let want: usize = shape.iter().product();
        if want != data.len() {
            return Err(Error::Length {
                context: "cannot build a tensor".to_string(),
                shape,
                len: data.len(),
            });
        }
        Ok(Self { shape, data })
    }

    /// Builds a tensor from internal code that computed the buffer length from
    /// the same shape. The length check still runs in a debug build, so a
    /// logic error inside the crate fails its tests rather than reading out of
    /// bounds.
    pub(crate) fn internal(shape: Vec<usize>, data: Vec<f32>) -> Self {
        let want: usize = shape.iter().product();
        debug_assert_eq!(want, data.len(), "shape {shape:?} vs {} values", data.len());
        Self { shape, data }
    }

    /// The tensor's dimensions.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The values in row-major order.
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// The number of values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the tensor holds no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Applies a function to every value.
    #[must_use]
    pub fn map(&self, f: impl Fn(f32) -> f32) -> Self {
        Self::internal(
            self.shape.clone(),
            self.data.iter().map(|v| f(*v)).collect(),
        )
    }

    /// Combines two equally shaped tensors value by value.
    pub fn zip_map(&self, other: &Self, f: impl Fn(f32, f32) -> f32) -> Result<Self, Error> {
        if self.shape != other.shape {
            return Err(Error::Mismatch {
                context: "cannot combine two tensors value by value".to_string(),
                expected: format!("shape {:?}", self.shape),
                got: format!("shape {:?}", other.shape),
            });
        }
        Ok(Self::internal(
            self.shape.clone(),
            self.data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| f(*a, *b))
                .collect(),
        ))
    }

    /// Adds two equally shaped tensors.
    pub fn add(&self, other: &Self) -> Result<Self, Error> {
        self.zip_map(other, |a, b| a + b)
    }

    /// Multiplies two equally shaped tensors value by value.
    pub fn mul(&self, other: &Self) -> Result<Self, Error> {
        self.zip_map(other, |a, b| a * b)
    }

    /// Regroups the same row-major values under a new shape.
    pub fn reshaped(&self, shape: Vec<usize>) -> Result<Self, Error> {
        Self::new(shape, self.data.clone())
    }

    /// Swaps two axes. For an ending-node attention this is foundry's
    /// transpose `b i j d -> b j i d`, which swaps the two sequence axes at
    /// indices 1 and 2 of a `(B, L, L, C)` tensor, leaving the channel axis
    /// where it is.
    pub fn transpose_axes(&self, a: usize, b: usize) -> Self {
        let ndim = self.shape.len();
        assert!(
            a < ndim && b < ndim && a != b,
            "transpose_axes({a}, {b}) outside a {ndim}-axis shape"
        );
        let in_strides = row_major_strides(&self.shape);
        let mut out_shape = self.shape.clone();
        out_shape.swap(a, b);
        let out_strides = row_major_strides(&out_shape);
        let mut data = vec![0.0f32; self.len()];
        let mut coords = vec![0usize; ndim];
        for (src, v) in self.data.iter().enumerate() {
            let mut rem = src;
            for d in 0..ndim {
                coords[d] = rem / in_strides[d];
                rem %= in_strides[d];
            }
            coords.swap(a, b);
            let mut dst = 0;
            for d in 0..ndim {
                dst += coords[d] * out_strides[d];
            }
            data[dst] = *v;
        }
        Self::internal(out_shape, data)
    }

    /// Splits the last axis into two equal halves, returning the low half and
    /// the high half. This is the split foundry applies to the combined
    /// projection `p_in` and gate `g_in` outputs.
    pub fn split_last_half(&self) -> Result<(Self, Self), Error> {
        let ndim = self.shape.len();
        let width = self.shape[ndim - 1];
        if width % 2 != 0 {
            return Err(Error::Mismatch {
                context: "cannot split the last axis into two halves".to_string(),
                expected: "an even last axis".to_string(),
                got: format!("{width}"),
            });
        }
        let half = width / 2;
        let rows = self.len() / width;
        let mut low = vec![0.0f32; rows * half];
        let mut high = vec![0.0f32; rows * half];
        for (r, row) in self.data.chunks_exact(width).enumerate() {
            low[r * half..(r + 1) * half].copy_from_slice(&row[..half]);
            high[r * half..(r + 1) * half].copy_from_slice(&row[half..]);
        }
        let mut shape = self.shape.clone();
        shape[ndim - 1] = half;
        Ok((
            Self::internal(shape.clone(), low),
            Self::internal(shape, high),
        ))
    }
}

/// Row-major strides for a shape: the value stride of each axis.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::Tensor;

    #[test]
    fn transpose_axes_swaps_the_two_named_axes() {
        let t = Tensor::new(vec![2, 3], (0..6).map(|v| v as f32).collect()).unwrap();
        let got = t.transpose_axes(0, 1);
        assert_eq!(got.shape(), &[3, 2]);
        // Row-major [[0,1,2],[3,4,5]] transposes to [[0,3],[1,4],[2,5]].
        assert_eq!(got.data(), &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn transpose_axes_leaves_the_other_axes_alone() {
        // (1, 2, 3, 1): swapping axes 1 and 2 of a pair tensor moves whole
        // channel rows and keeps the channel axis last.
        let t = Tensor::new(vec![1, 2, 3, 1], (0..6).map(|v| v as f32).collect()).unwrap();
        let got = t.transpose_axes(1, 2);
        assert_eq!(got.shape(), &[1, 3, 2, 1]);
        // [i][j] -> [j][i]: [[0,1,2],[3,4,5]] becomes [[0,3],[1,4],[2,5]].
        assert_eq!(got.data(), &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn transpose_axes_round_trips() {
        let t = Tensor::new(vec![2, 3, 4], (0..24).map(|v| v as f32).collect()).unwrap();
        assert_eq!(t.transpose_axes(1, 2).transpose_axes(1, 2).data(), t.data());
    }

    #[test]
    fn split_last_half_partitions_the_last_axis() {
        let t = Tensor::new(vec![2, 4], (0..8).map(|v| v as f32).collect()).unwrap();
        let (low, high) = t.split_last_half().unwrap();
        assert_eq!(low.shape(), &[2, 2]);
        assert_eq!(low.data(), &[0.0, 1.0, 4.0, 5.0]);
        assert_eq!(high.data(), &[2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn a_buffer_that_disagrees_with_its_shape_is_rejected() {
        let err = Tensor::new(vec![2, 3], vec![0.0; 5]).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("shape [2, 3]"), "{message}");
        assert!(message.contains("5"), "{message}");
    }
}
