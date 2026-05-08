use serde::{Deserialize, Serialize};

/// Camera intrinsic parameters for undistortion and projection
use opencv::prelude::MatTraitConst;
#[derive(Debug, Clone, Copy)]
pub struct CameraIntrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub k1: f64,
    pub k2: f64,
}

impl Default for CameraIntrinsics {
    fn default() -> Self {
        Self {
            fx: 1389.7,
            fy: 1387.1,
            cx: 954.007,
            cy: 558.896,
            k1: 0.1378,
            k2: -0.2564,
        }
    }
}

impl CameraIntrinsics {
    /// Create camera matrix (3x3) for OpenCV functions
    ///
    /// # Errors
    /// Returns OpenCV error if matrix creation fails
    pub fn camera_matrix(&self) -> opencv::Result<opencv::core::Mat> {
        let data = [
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ];
        let mat_ref = opencv::core::Mat::from_slice_2d(&data)?;
        mat_ref.try_clone()
    }

    /// Create distortion coefficients vector [k1, k2, 0, 0]
    ///
    /// # Errors
    /// Returns OpenCV error if vector creation fails
    pub fn dist_coeffs(&self) -> opencv::Result<opencv::core::Mat> {
        let data = [self.k1, self.k2, 0.0, 0.0];
        let mat_ref = opencv::core::Mat::from_slice(&data)?;
        mat_ref.try_clone()
    }
}

/// Relative pose change between two consecutive frames
#[derive(Debug, Clone)]
pub struct PoseDelta {
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    pub frame_idx: usize,
    pub scale: f64,
    pub inliers_count: usize,
    pub is_unreliable: bool,
}

impl Default for PoseDelta {
    fn default() -> Self {
        Self {
            dx: 0.0,
            dy: 0.0,
            dz: 0.0,
            frame_idx: 0,
            scale: 1.0,
            inliers_count: 0,
            is_unreliable: false,
        }
    }
}

/// Pair of matched points between two frames
#[derive(Debug, Clone)]
pub struct FramePair {
    pub prev_points: Vec<opencv::core::Point2f>,
    pub curr_points: Vec<opencv::core::Point2f>,
}

impl FramePair {
    pub fn new(
        prev_points: Vec<opencv::core::Point2f>,
        curr_points: Vec<opencv::core::Point2f>,
    ) -> Self {
        Self {
            prev_points,
            curr_points,
        }
    }

    /// Convert to OpenCV Vector types for homography estimation
    pub fn to_opencv_vectors(
        &self,
    ) -> (
        opencv::core::Vector<opencv::core::Point2f>,
        opencv::core::Vector<opencv::core::Point2f>,
    ) {
        (
            opencv::core::Vector::from(self.prev_points.clone()),
            opencv::core::Vector::from(self.curr_points.clone()),
        )
    }
}

pub struct Point {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

pub struct Rotation {
    pub yaw: f64,
    pub pitch: f64,
    pub roll: f64,
}

pub struct Frame {
    pub id: u32,
    pub data: opencv::core::Mat,
    pub pos: Point,
    pub rot: Rotation,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CsvRow {
    pub translation_x: f64,
    pub translation_y: f64,
    pub translation_z: f64,
    pub frame_num: u32,
}

/// Trajectory record for CSV output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryRecord {
    pub frame_idx: usize,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub scale: f64,
    pub inliers: usize,
    pub dt_sec: f64,
}

impl TrajectoryRecord {
    pub fn to_csv_row(&self) -> Vec<String> {
        vec![
            self.frame_idx.to_string(),
            format!("{:.6}", self.x),
            format!("{:.6}", self.y),
            format!("{:.6}", self.z),
            format!("{:.6}", self.scale),
            self.inliers.to_string(),
            format!("{:.6}", self.dt_sec),
        ]
    }
}

/// Ground truth trajectory record from CSV
#[derive(Debug, Clone, Deserialize)]
pub struct GroundTruthRecord {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub frame_idx: u32,
}

impl GroundTruthRecord {
    /// Load ground truth records from CSV file
    ///
    /// # Arguments
    /// * `path` - Path to the CSV file (format: x,y,z,frame_idx)
    ///
    /// # Returns
    /// Vector of GroundTruthRecord
    pub fn load_from_csv(path: &str) -> Result<Vec<Self>, csv::Error> {
        let mut reader = csv::Reader::from_path(path)?;
        let mut records = Vec::new();
        for result in reader.deserialize() {
            let record: GroundTruthRecord = result?;
            records.push(record);
        }
        Ok(records)
    }
}

/// Tracking metrics for evaluation
#[derive(Debug, Clone, Default)]
pub struct TrackingMetrics {
    pub rmse_x: f64,
    pub rmse_y: f64,
    pub rmse_z: f64,
    pub mae_x: f64,
    pub mae_y: f64,
    pub mae_z: f64,
    pub num_samples: usize,
}

impl TrackingMetrics {
    /// Compute metrics from estimated and ground truth trajectories
    pub fn compute(estimated: &[TrajectoryRecord], ground_truth: &[GroundTruthRecord]) -> Self {
        let mut metrics = Self::default();
        metrics.num_samples = 0;
        
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;
        let mut sum_sq_z = 0.0;
        let mut sum_abs_x = 0.0;
        let mut sum_abs_y = 0.0;
        let mut sum_abs_z = 0.0;
        
        // Build a hash map for ground truth for faster lookup
        let gt_map: std::collections::HashMap<usize, &GroundTruthRecord> = ground_truth
            .iter()
            .map(|r| (r.frame_idx as usize, r))
            .collect();
        
        for est in estimated {
            if let Some(&gt) = gt_map.get(&est.frame_idx) {
                let dx = est.x - gt.x;
                let dy = est.y - gt.y;
                let dz = est.z - gt.z;
                
                sum_sq_x += dx * dx;
                sum_sq_y += dy * dy;
                sum_sq_z += dz * dz;
                
                sum_abs_x += dx.abs();
                sum_abs_y += dy.abs();
                sum_abs_z += dz.abs();
                
                metrics.num_samples += 1;
            }
        }
        
        if metrics.num_samples > 0 {
            let n = metrics.num_samples as f64;
            metrics.rmse_x = (sum_sq_x / n).sqrt();
            metrics.rmse_y = (sum_sq_y / n).sqrt();
            metrics.rmse_z = (sum_sq_z / n).sqrt();
            
            metrics.mae_x = sum_abs_x / n;
            metrics.mae_y = sum_abs_y / n;
            metrics.mae_z = sum_abs_z / n;
        }
        
        metrics
    }
}

impl std::fmt::Display for TrackingMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Tracking Metrics ===")?;
        writeln!(f, "Samples: {}", self.num_samples)?;
        writeln!(f, "RMSE (x, y, z): {:.6}, {:.6}, {:.6}", self.rmse_x, self.rmse_y, self.rmse_z)?;
        writeln!(f, "MAE (x, y, z): {:.6}, {:.6}, {:.6}", self.mae_x, self.mae_y, self.mae_z)?;
        Ok(())
    }
}
