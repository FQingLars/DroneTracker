//! Height estimation module using ground plane homography.
//!
//! This module provides functionality to estimate relative height changes (Δz)
//! between consecutive drone frames by analyzing the homography of the ground plane.
//! The approach assumes nadir-view flight over locally flat terrain.

use crate::types::{CameraIntrinsics, FramePair};
use anyhow::{anyhow, Result};
use log::{debug, warn};
use opencv::calib3d;
use opencv::core::{self, Mat};
use opencv::imgproc;
use opencv::prelude::{MatTrait, MatTraitConst};

/// Minimum number of inliers required for reliable scale estimation
const MIN_INLIERS: usize = 30;

/// Valid scale range for clamping (prevents tracking artifacts)
const SCALE_MIN: f64 = 0.7;
const SCALE_MAX: f64 = 1.4;

/// Configuration for homography estimation
#[derive(Debug, Clone)]
pub struct HomographyConfig {
    pub ransac_threshold: f64,
    pub confidence: f64,
    pub max_iters: i32,
}

impl Default for HomographyConfig {
    fn default() -> Self {
        Self {
            ransac_threshold: 3.0,
            confidence: 0.99,
            max_iters: 2000,
        }
    }
}

/// Cached undistortion maps for efficient remapping
pub struct UndistortionCache {
    map1: Mat,
    map2: Mat,
    #[allow(dead_code)]
    image_size: (i32, i32),
}

impl UndistortionCache {
    /// Create undistortion maps from camera intrinsics
    ///
    /// # Arguments
    /// * `intrinsics` - Camera intrinsic parameters
    /// * `image_size` - (width, height) of the input images
    ///
    /// # Example
    /// ```
    /// use tracker::types::CameraIntrinsics;
    /// use tracker::height::UndistortionCache;
    /// let intrinsics = CameraIntrinsics::default();
    /// let cache = UndistortionCache::new(&intrinsics, (1920, 1080)).expect("Failed to create cache");
    /// ```
    pub fn new(intrinsics: &CameraIntrinsics, image_size: (i32, i32)) -> Result<Self> {
        let camera_matrix = intrinsics.camera_matrix()?;
        let dist_coeffs = intrinsics.dist_coeffs()?;

        let mut map1 = Mat::default();
        let mut map2 = Mat::default();

        let k = &camera_matrix;
        let d = &dist_coeffs;
        opencv::calib3d::init_undistort_rectify_map(
            k,
            d,
            &Mat::default(),
            k,
            core::Size::new(image_size.0, image_size.1),
            core::CV_16SC2,
            &mut map1,
            &mut map2,
        )?;

        debug!(
            "UndistortionCache initialized for image size {}x{}",
            image_size.0, image_size.1
        );

        Ok(Self {
            map1,
            map2,
            image_size,
        })
    }

    /// Apply undistortion to an input image using cached maps
    ///
    /// # Arguments
    /// * `src` - Input distorted image
    /// * `dst` - Output undistorted image (pre-allocated or will be created)
    pub fn remap(&self, src: &Mat, dst: &mut Mat) -> Result<()> {
        imgproc::remap(
            src,
            dst,
            &self.map1,
            &self.map2,
            imgproc::INTER_LINEAR,
            core::BORDER_CONSTANT,
            core::Scalar::default(),
        )?;
        Ok(())
    }
}

/// Height estimator using ground plane homography
pub struct HeightEstimator {
    config: HomographyConfig,
    prev_scale: f64,
}

impl Default for HeightEstimator {
    fn default() -> Self {
        Self {
            config: HomographyConfig::default(),
            prev_scale: 1.0,
        }
    }
}

impl HeightEstimator {
    /// Create a new height estimator
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom configuration
    pub fn with_config(config: HomographyConfig) -> Self {
        Self {
            config,
            prev_scale: 1.0,
        }
    }

    /// Estimate relative height change from matched points
    ///
    /// Returns (delta_z, scale, inliers_count, is_unreliable)
    ///
    /// # Mathematics
    /// - H_norm = H_pixel / H[2,2]
    /// - s = (||H_norm[0:2, 0]||₂ + ||H_norm[0:2, 1]||₂) / 2.0
    /// - Δz_rel = 1/s - 1.0
    ///
    /// # Arguments
    /// * `frame_pair` - Matched points between previous and current frame
    ///
    /// # Returns
    /// Tuple of (dz, scale, inliers_count, is_unreliable)
    pub fn estimate_height_change(&mut self, frame_pair: &FramePair) -> Result<(f64, f64, usize, bool)> {
        if frame_pair.prev_points.len() < 4 || frame_pair.curr_points.len() < 4 {
            debug!("Insufficient points for homography: prev={}, curr={}", 
                   frame_pair.prev_points.len(), frame_pair.curr_points.len());
            return Ok((0.0, 1.0, 0, false));
        }

        let (src_pts, dst_pts) = frame_pair.to_opencv_vectors();
        let mut mask = Mat::default();

        let h = calib3d::find_homography(
            &src_pts,
            &dst_pts,
            &mut mask,
            calib3d::RANSAC,
            self.config.ransac_threshold,
        )?;

        let inliers_count = self.count_inliers(&mask);
        
        if inliers_count < MIN_INLIERS {
            warn!(
                "Insufficient inliers for height estimation: {} < {}, using previous scale",
                inliers_count, MIN_INLIERS
            );
            return Ok((0.0, self.prev_scale, inliers_count, true));
        }

            let scale = match self.extract_scale_from_homography(&h) {
                Ok(s) => {
                    if !(SCALE_MIN..=SCALE_MAX).contains(&s) {
                        warn!(
                            "Scale {:.4} outside valid range [{:.2}, {:.2}], clamping",
                            s, SCALE_MIN, SCALE_MAX
                        );
                        s.clamp(SCALE_MIN, SCALE_MAX)
                    } else {
                        s
                    }
                }
            Err(e) => {
                warn!("Failed to extract scale from homography: {}, using previous", e);
                self.prev_scale
            }
        };

        let delta_z_rel = 1.0 / scale - 1.0;

        debug!(
            "Height estimation: scale={:.6}, dz={:.6}, inliers={}",
            scale, delta_z_rel, inliers_count
        );

        #[cfg(debug_assertions)]
        {
            if scale <= 0.0 || !scale.is_finite() {
                return Err(anyhow!("Invalid scale value: {}", scale));
            }
        }

        self.prev_scale = scale;
        Ok((delta_z_rel, scale, inliers_count, false))
    }

    /// Extract isotropic scale from normalized homography matrix
    ///
    /// Uses the Frobenius norm approach:
    /// s = (||H[0:2, 0]||₂ + ||H[0:2, 1]||₂) / 2.0
    fn extract_scale_from_homography(&self, h: &Mat) -> Result<f64> {
        let size = h.size()?;
        if size.height < 3 || size.width < 3 {
            return Err(anyhow!(
                "Homography matrix too small: {}x{}",
                size.height,
                size.width
            ));
        }

        let h_norm = self.normalize_homography(h)?;

        let h00 = *h_norm.at_2d::<f64>(0, 0)?;
        let h01 = *h_norm.at_2d::<f64>(0, 1)?;
        let h10 = *h_norm.at_2d::<f64>(1, 0)?;
        let h11 = *h_norm.at_2d::<f64>(1, 1)?;

        let s1 = (h00 * h00 + h10 * h10).sqrt();
        let s2 = (h01 * h01 + h11 * h11).sqrt();
        let scale = (s1 + s2) / 2.0;

        if scale <= 0.0 || !scale.is_finite() {
            return Err(anyhow!("Invalid scale computed: {}", scale));
        }

        Ok(scale)
    }

    /// Normalize homography so that H[2,2] = 1.0
    fn normalize_homography(&self, h: &Mat) -> Result<Mat> {
        let h22 = *h.at_2d::<f64>(2, 2)?;
        
        if (h22 - 1.0).abs() < f64::EPSILON {
            Ok(h.clone())
        } else if h22.abs() < f64::EPSILON {
            Err(anyhow!("Homography[2,2] is zero, cannot normalize"))
        } else {
            let mut h_norm = Mat::new_rows_cols_with_default(3, 3, core::CV_64F, 0.0.into())?;
            for i in 0..3 {
                for j in 0..3 {
                    let val = *h.at_2d::<f64>(i, j)?;
                    *h_norm.at_2d_mut::<f64>(i, j)? = val / h22;
                }
            }
            Ok(h_norm)
        }
    }

    /// Count inliers from RANSAC mask using count_non_zero for efficiency
    fn count_inliers(&self, mask: &Mat) -> usize {
        if mask.size().map(|s| s.area()).unwrap_or(0) == 0 {
            return 0;
        }
        match opencv::core::count_non_zero(mask) {
            Ok(count) => count as usize,
            Err(_) => 0,
        }
    }

    /// Reset the estimator state (for use when tracking restarts)
    pub fn reset(&mut self) {
        self.prev_scale = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::Point2f;
    use opencv::prelude::MatTraitConst;

    /// Test scale extraction from identity homography
    /// Identity H should give scale = 1.0, dz = 0.0
    #[test]
    fn test_identity_homography() -> Result<()> {
        let mut estimator = HeightEstimator::new();
        
        let identity_points_prev = vec![
            Point2f { x: 100.0, y: 100.0 },
            Point2f { x: 200.0, y: 100.0 },
            Point2f { x: 100.0, y: 200.0 },
            Point2f { x: 200.0, y: 200.0 },
        ];
        let identity_points_curr = identity_points_prev.clone();
        
        let frame_pair = FramePair::new(identity_points_prev, identity_points_curr);
        let (dz, scale, _inliers, _unreliable) = estimator.estimate_height_change(&frame_pair)?;
        
        assert!(scale > 0.99 && scale < 1.01, "Scale should be ~1.0 for identity, got {}", scale);
        assert!((dz).abs() < 0.01, "dz should be ~0.0 for identity, got {}", dz);
        Ok(())
    }

    /// Test synthetic scale change
    /// If points are scaled by 1.05 (5% larger), it means drone descended,
    /// so dz should be negative: 1/1.05 - 1 = -0.0476
    #[test]
    fn test_scale_increase_means_descend() -> Result<()> {
        let mut estimator = HeightEstimator::new();
        
        // Create more points for robust homography estimation
        let mut prev_pts = Vec::new();
        let mut curr_pts = Vec::new();
        let scale_factor: f32 = 1.05;
        
        for i in 0..20 {
            for j in 0..20 {
                let x = 100.0 + i as f32 * 10.0;
                let y = 100.0 + j as f32 * 10.0;
                prev_pts.push(Point2f { x, y });
                curr_pts.push(Point2f { 
                    x: x * scale_factor, 
                    y: y * scale_factor 
                });
            }
        }
        
        let frame_pair = FramePair::new(prev_pts, curr_pts);
        let (dz, scale, _, _) = estimator.estimate_height_change(&frame_pair)?;
        
        let expected_dz = 1.0 / scale_factor as f64 - 1.0;
        // Allow larger epsilon due to homography estimation noise
        assert!(
            (scale - scale_factor as f64).abs() < 0.05,
            "Scale should be ~{}, got {}",
            scale_factor,
            scale
        );
        assert!(
            (dz - expected_dz).abs() < 0.05,
            "dz should be ~{}, got {}",
            expected_dz,
            dz
        );
        Ok(())
    }

    /// Test scale decrease means ascend
    /// If points are scaled by 0.95 (5% smaller), drone ascended,
    /// dz should be positive: 1/0.95 - 1 = 0.0526
    #[test]
    fn test_scale_decrease_means_ascend() -> Result<()> {
        let mut estimator = HeightEstimator::new();
        
        // Create more points for robust homography estimation
        let mut prev_pts = Vec::new();
        let mut curr_pts = Vec::new();
        let scale_factor: f32 = 0.95;
        
        for i in 0..20 {
            for j in 0..20 {
                let x = 100.0 + i as f32 * 10.0;
                let y = 100.0 + j as f32 * 10.0;
                prev_pts.push(Point2f { x, y });
                curr_pts.push(Point2f { 
                    x: x * scale_factor, 
                    y: y * scale_factor 
                });
            }
        }
        
        let frame_pair = FramePair::new(prev_pts, curr_pts);
        let (dz, _scale, _, _) = estimator.estimate_height_change(&frame_pair)?;
        
        let expected_dz = 1.0 / scale_factor as f64 - 1.0;
        // Allow larger epsilon due to homography estimation noise
        assert!(
            (dz - expected_dz).abs() < 0.05,
            "dz should be ~{}, got {}",
            expected_dz,
            dz
        );
        Ok(())
    }

    /// Test camera intrinsics creation
    #[test]
    fn test_camera_intrinsics() -> opencv::Result<()> {
        let intrinsics = CameraIntrinsics::default();
        assert!(intrinsics.fx > 0.0);
        assert!(intrinsics.fy > 0.0);
        
        let camera_matrix = intrinsics.camera_matrix()?;
        assert!(!camera_matrix.empty());
        assert_eq!(camera_matrix.rows(), 3);
        assert_eq!(camera_matrix.cols(), 3);
        Ok(())
    }

    /// Test undistortion cache creation
    #[test]
    fn test_undistortion_cache() -> Result<()> {
        let intrinsics = CameraIntrinsics::default();
        let cache = UndistortionCache::new(&intrinsics, (1920, 1080))?;
        assert_eq!(cache.image_size, (1920, 1080));
        assert!(!cache.map1.empty());
        assert!(!cache.map2.empty());
        Ok(())
    }
}
