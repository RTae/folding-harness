//! Drawing an alpha-carbon trace in a terminal.
//!
//! Braille glyphs give a 2×4 dot grid per cell, so a modest pane is a few
//! hundred addressable points — enough for a recognisable backbone. The maths
//! here is deliberately plain: rotate, project orthographically, normalise to
//! the canvas. No perspective, because a foreshortened backbone is harder to
//! read at this resolution, not easier.

use fh_contract::CaResidue;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    /// Rotation about the vertical axis, radians.
    pub yaw: f64,
    /// Rotation about the horizontal axis, radians.
    pub pitch: f64,
    /// 1.0 fits the whole structure in the pane.
    pub zoom: f64,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            yaw: 0.6,
            pitch: 0.25,
            zoom: 1.0,
        }
    }
}

impl Camera {
    pub fn turn(&mut self, yaw: f64, pitch: f64) {
        self.yaw = (self.yaw + yaw).rem_euclid(std::f64::consts::TAU);
        // Stop short of the poles, where the projection degenerates.
        self.pitch = (self.pitch + pitch).clamp(-1.4, 1.4);
    }

    pub fn scale(&mut self, factor: f64) {
        self.zoom = (self.zoom * factor).clamp(0.25, 6.0);
    }

    pub fn reset(&mut self) {
        *self = Camera::default();
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Projected {
    pub x: f64,
    pub y: f64,
    /// Rotated z, for depth ordering. Larger is nearer the viewer.
    pub depth: f64,
    pub plddt: f32,
    pub designed: bool,
    pub chain_index: u8,
    pub index: u32,
}

fn rotate(point: [f64; 3], yaw: f64, pitch: f64) -> [f64; 3] {
    let (sin_yaw, cos_yaw) = yaw.sin_cos();
    let (sin_pitch, cos_pitch) = pitch.sin_cos();

    // Yaw about Y.
    let x1 = point[0] * cos_yaw + point[2] * sin_yaw;
    let y1 = point[1];
    let z1 = -point[0] * sin_yaw + point[2] * cos_yaw;

    // Pitch about X.
    let x2 = x1;
    let y2 = y1 * cos_pitch - z1 * sin_pitch;
    let z2 = y1 * sin_pitch + z1 * cos_pitch;

    [x2, y2, z2]
}

/// Project a trace into canvas coordinates spanning roughly `[-zoom, zoom]`.
///
/// Centres on the centroid and divides by the largest extent, so a short
/// peptide and a large complex both fill the pane.
pub fn project(trace: &[CaResidue], camera: &Camera) -> Vec<Projected> {
    if trace.is_empty() {
        return Vec::new();
    }

    let count = trace.len() as f64;
    let centroid = trace.iter().fold([0.0f64; 3], |mut acc, residue| {
        acc[0] += f64::from(residue.xyz[0]) / count;
        acc[1] += f64::from(residue.xyz[1]) / count;
        acc[2] += f64::from(residue.xyz[2]) / count;
        acc
    });

    let rotated: Vec<[f64; 3]> = trace
        .iter()
        .map(|residue| {
            rotate(
                [
                    f64::from(residue.xyz[0]) - centroid[0],
                    f64::from(residue.xyz[1]) - centroid[1],
                    f64::from(residue.xyz[2]) - centroid[2],
                ],
                camera.yaw,
                camera.pitch,
            )
        })
        .collect();

    let extent = rotated
        .iter()
        .flat_map(|point| [point[0].abs(), point[1].abs()])
        .fold(0.0f64, f64::max)
        .max(1e-6);

    trace
        .iter()
        .zip(rotated.iter())
        .map(|(residue, point)| Projected {
            x: point[0] / extent * camera.zoom,
            y: point[1] / extent * camera.zoom,
            depth: point[2],
            plddt: residue.plddt,
            designed: residue.designed,
            chain_index: residue.chain_index,
            index: residue.index,
        })
        .collect()
}

/// Confidence bands, in the familiar four-step scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceBand {
    VeryHigh,
    Confident,
    Low,
    VeryLow,
}

impl ConfidenceBand {
    pub fn of(plddt: f32) -> Self {
        if plddt >= 90.0 {
            ConfidenceBand::VeryHigh
        } else if plddt >= 70.0 {
            ConfidenceBand::Confident
        } else if plddt >= 50.0 {
            ConfidenceBand::Low
        } else {
            ConfidenceBand::VeryLow
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            ConfidenceBand::VeryHigh => "very high (>=90)",
            ConfidenceBand::Confident => "confident (70-90)",
            ConfidenceBand::Low => "low (50-70)",
            ConfidenceBand::VeryLow => "very low (<50)",
        }
    }
}

/// Consecutive residues of the same chain, as segments to draw.
pub fn backbone_segments(points: &[Projected]) -> Vec<(Projected, Projected)> {
    points
        .windows(2)
        .filter(|pair| {
            pair[0].chain_index == pair[1].chain_index && pair[1].index == pair[0].index + 1
        })
        .map(|pair| (pair[0], pair[1]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(count: u32) -> Vec<CaResidue> {
        (0..count)
            .map(|index| CaResidue {
                chain_index: 0,
                index,
                residue: 'A',
                xyz: [index as f32 * 3.0, (index as f32 * 0.5).sin() * 4.0, 2.0],
                plddt: 80.0,
                designed: true,
            })
            .collect()
    }

    #[test]
    fn an_empty_trace_projects_to_nothing() {
        assert!(project(&[], &Camera::default()).is_empty());
    }

    #[test]
    fn projection_is_centred_and_bounded_by_zoom() {
        let points = project(&trace(40), &Camera::default());
        assert_eq!(points.len(), 40);

        let mean_x: f64 = points.iter().map(|p| p.x).sum::<f64>() / points.len() as f64;
        let mean_y: f64 = points.iter().map(|p| p.y).sum::<f64>() / points.len() as f64;
        assert!(mean_x.abs() < 0.5, "mean x {mean_x}");
        assert!(mean_y.abs() < 0.5, "mean y {mean_y}");

        for point in &points {
            assert!(point.x.abs() <= 1.0001, "x out of bounds: {}", point.x);
            assert!(point.y.abs() <= 1.0001, "y out of bounds: {}", point.y);
        }
    }

    #[test]
    fn zoom_scales_the_projection() {
        let wide = project(
            &trace(20),
            &Camera {
                zoom: 1.0,
                ..Camera::default()
            },
        );
        let close = project(
            &trace(20),
            &Camera {
                zoom: 2.0,
                ..Camera::default()
            },
        );
        let span = |points: &[Projected]| {
            points.iter().map(|p| p.x).fold(f64::MIN, f64::max)
                - points.iter().map(|p| p.x).fold(f64::MAX, f64::min)
        };
        assert!((span(&close) / span(&wide) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn segments_do_not_bridge_chains_or_gaps() {
        let mut residues = trace(4);
        residues[2].chain_index = 1;
        let points = project(&residues, &Camera::default());
        let segments = backbone_segments(&points);
        // 0-1 same chain, 1-2 crosses chains, 2-3 same chain again.
        assert_eq!(segments.len(), 2);
    }

    #[test]
    fn confidence_bands_split_at_the_usual_thresholds() {
        assert_eq!(ConfidenceBand::of(95.0), ConfidenceBand::VeryHigh);
        assert_eq!(ConfidenceBand::of(90.0), ConfidenceBand::VeryHigh);
        assert_eq!(ConfidenceBand::of(89.9), ConfidenceBand::Confident);
        assert_eq!(ConfidenceBand::of(70.0), ConfidenceBand::Confident);
        assert_eq!(ConfidenceBand::of(69.0), ConfidenceBand::Low);
        assert_eq!(ConfidenceBand::of(49.0), ConfidenceBand::VeryLow);
    }

    #[test]
    fn the_camera_clamps_pitch_and_zoom() {
        let mut camera = Camera::default();
        camera.turn(0.0, 100.0);
        assert!(camera.pitch <= 1.4);
        camera.turn(0.0, -100.0);
        assert!(camera.pitch >= -1.4);
        camera.scale(1000.0);
        assert!(camera.zoom <= 6.0);
        camera.scale(0.0001);
        assert!(camera.zoom >= 0.25);
    }
}
