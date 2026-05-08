use crate::types::FramePair;
use anyhow::{anyhow, Result};
use log::{debug, info, warn};
use opencv::calib3d;
use opencv::core::{self, DMatch, KeyPoint, Mat, NORM_HAMMING, Point2f, Vector};
use opencv::features2d::{BFMatcher, DescriptorMatcherTrait, Feature2DTrait, ORB};
use opencv::imgproc;
use opencv::prelude::*;

/// GMC Tracker for estimating camera motion between frames
///
/// Uses ORB feature detection and matching with RANSAC homography
/// to estimate 2D motion in pixel coordinates.
pub struct GmcTracker {
    detector: core::Ptr<ORB>,
    matcher: core::Ptr<BFMatcher>,
    prev_frame: Option<Mat>,
    prev_kp: Option<Vector<KeyPoint>>,
    prev_desc: Option<Mat>,
}

impl GmcTracker {
    /// Create a new GMC tracker with default ORB parameters
    pub fn new() -> Result<Self> {
        info!("[GMC] Creating GmcTracker...");
        let detector = ORB::create(
            2000, 1.2, 8, 31, 0, 2,
            opencv::features2d::ORB_ScoreType::HARRIS_SCORE, 31, 20,
        )?;
        info!("[GMC] ORB created: nfeatures=2000, score=HARRIS_SCORE");
        let matcher = BFMatcher::create(NORM_HAMMING, false)?;
        info!("[GMC] BFMatcher created: NORM_HAMMING");
        info!("[GMC] GmcTracker created successfully");
        Ok(Self {
            detector,
            matcher,
            prev_frame: None,
            prev_kp: None,
            prev_desc: None,
        })
    }

    /// Process a new frame and return matched points with raw motion
    ///
    /// Returns (Option<FramePair>, Option<compensated_frame>)
    pub fn process(
        &mut self,
        frame: &Mat,
    ) -> Result<(Option<FramePair>, Option<Mat>)> {
        let gray = Self::to_grayscale(frame)?;
        debug!("[GMC] Frame converted to grayscale, size: {:?}", gray.size());

        let mut kp = Vector::new();
        let mut desc = Mat::default();
        self.detector
            .detect_and_compute(&gray, &core::Mat::default(), &mut kp, &mut desc, false)?;
        debug!(
            "[GMC] Detection: {} keypoints, {} descriptors",
            kp.len(),
            desc.rows()
        );

        let result = if let (Some(prev_kp), Some(prev_desc)) = (&self.prev_kp, &self.prev_desc) {
            let matches = Self::match_features(&mut self.matcher, prev_desc, &desc)?;
            debug!("[GMC] After filtering: {} matches", matches.len());

            if matches.len() >= 4 {
                let (prev_points, curr_points) = Self::extract_points(&kp, prev_kp, &matches)?;
                debug!(
                    "[GMC] Prepared points: prev={}, curr={}",
                    prev_points.len(),
                    curr_points.len()
                );

                let frame_pair = FramePair::new(prev_points, curr_points);

                let size = frame.size()?;
                let mut compensated = Mat::default();
                imgproc::warp_perspective(
                    frame,
                    &mut compensated,
                    &frame_pair_to_homography(&frame_pair)?,
                    size,
                    imgproc::INTER_LINEAR,
                    core::BORDER_CONSTANT,
                    core::Scalar::default(),
                )?;
                debug!("[GMC] Compensation done");

                (Some(frame_pair), Some(compensated))
            } else {
                warn!("[GMC] Insufficient matches (< 4), skipping");
                (None, None)
            }
        } else {
            debug!("[GMC] First frame, saving as previous");
            (None, None)
        };

        self.prev_frame = Some(gray.clone());
        self.prev_kp = Some(kp);
        self.prev_desc = Some(desc);
        Ok(result)
    }

    /// Get raw motion from matched points (dx, dy in pixels)
    ///
    /// This extracts the translation component from the homography matrix
    pub fn get_raw_motion(&self, frame_pair: &FramePair) -> Result<(f64, f64)> {
        let h = frame_pair_to_homography(frame_pair)?;
        let _h00 = *h.at_2d::<f64>(0, 0)?;
        let _h01 = *h.at_2d::<f64>(0, 1)?;
        let h02 = *h.at_2d::<f64>(0, 2)?;
        let _h10 = *h.at_2d::<f64>(1, 0)?;
        let _h11 = *h.at_2d::<f64>(1, 1)?;
        let h12 = *h.at_2d::<f64>(1, 2)?;

        let dx = h02;
        let dy = h12;

        Ok((dx, dy))
    }

    /// Correct raw motion by scale factor from height estimation
    ///
    /// Formula: (dx, dy) = (dx_raw / s, dy_raw / s)
    pub fn correct_by_scale(
        &self,
        dx_raw: f64,
        dy_raw: f64,
        scale: f64,
    ) -> Result<(f64, f64)> {
        if scale <= 0.0 || !scale.is_finite() {
            return Err(anyhow!("Invalid scale value: {}", scale));
        }
        Ok((dx_raw / scale, dy_raw / scale))
    }

    /// Reset tracker state
    pub fn reset(&mut self) {
        info!("[GMC] Resetting tracker");
        self.prev_frame = None;
        self.prev_kp = None;
        self.prev_desc = None;
    }

    fn to_grayscale(src: &Mat) -> Result<Mat> {
        let channels = src.channels();
        debug!(
            "[GMC] to_grayscale: channels={}, size={:?}",
            channels,
            src.size()
        );
        if channels == 1 {
            Ok(src.clone())
        } else {
            let mut dst = Mat::default();
            imgproc::cvt_color(
                src,
                &mut dst,
                imgproc::COLOR_BGR2GRAY,
                0,
                core::AlgorithmHint::ALGO_HINT_DEFAULT,
            )?;
            debug!("[GMC] to_grayscale: converted BGR->GRAY");
            Ok(dst)
        }
    }

    fn match_features(
        matcher: &mut core::Ptr<BFMatcher>,
        query_desc: &Mat,
        train_desc: &Mat,
    ) -> Result<Vector<DMatch>> {
        let mut matches = Vector::new();
        DescriptorMatcherTrait::clear(matcher)?;
        let train_vec: Vector<Mat> = Vector::from(vec![train_desc.clone()]);
        matcher.add(&train_vec)?;
        matcher.match_(query_desc, &mut matches, &core::Mat::default())?;
        debug!("[GMC] match_features: {} raw matches", matches.len());

        if matches.is_empty() {
            return Ok(matches);
        }

        let min_dist = matches
            .iter()
            .map(|m| m.distance)
            .fold(f32::INFINITY, f32::min);
        let threshold = (min_dist * 1.5).max(30.0);
        debug!(
            "[GMC] match_features: min_dist={:.2}, threshold={:.2}",
            min_dist, threshold
        );

        let filtered: Vec<_> = matches
            .iter()
            .filter(|m| m.distance <= threshold)
            .collect();

        debug!(
            "[GMC] match_features: {} matches after filtering",
            filtered.len()
        );

        Ok(Vector::from(filtered))
    }

    fn extract_points(
        curr_kp: &Vector<KeyPoint>,
        prev_kp: &Vector<KeyPoint>,
        matches: &Vector<DMatch>,
    ) -> Result<(Vec<Point2f>, Vec<Point2f>)> {
        let mut prev_points = Vec::with_capacity(matches.len());
        let mut curr_points = Vec::with_capacity(matches.len());

        for m in matches.iter() {
            let prev_kp_idx = m.query_idx as usize;
            let curr_kp_idx = m.train_idx as usize;

            let prev_k = prev_kp
                .get(prev_kp_idx)
                .map_err(|e| anyhow!("Failed to get prev keypoint: {}", e))?;
            let curr_k = curr_kp
                .get(curr_kp_idx)
                .map_err(|e| anyhow!("Failed to get curr keypoint: {}", e))?;

            prev_points.push(Point2f {
                x: prev_k.pt().x,
                y: prev_k.pt().y,
            });
            curr_points.push(Point2f {
                x: curr_k.pt().x,
                y: curr_k.pt().y,
            });
        }

        debug!(
            "[GMC] extract_points: extracted {} point pairs",
            prev_points.len()
        );
        Ok((prev_points, curr_points))
    }
}

/// Convert FramePair to homography matrix using RANSAC
fn frame_pair_to_homography(frame_pair: &FramePair) -> Result<Mat> {
    let src_pts: Vector<Point2f> = Vector::from(frame_pair.prev_points.clone());
    let dst_pts: Vector<Point2f> = Vector::from(frame_pair.curr_points.clone());
    let mut mask = core::Mat::default();

    let h = calib3d::find_homography(
        &src_pts,
        &dst_pts,
        &mut mask,
        calib3d::RANSAC,
        3.0,
    )?;

    if h.empty() {
        return Err(anyhow!("Homography computation failed - empty matrix"));
    }

    Ok(h)
}

/// Extract FramePair from GMC processing result for height estimation
pub fn frame_pair_to_points(frame_pair: &FramePair) -> (Vec<Point2f>, Vec<Point2f>) {
    (
        frame_pair.prev_points.clone(),
        frame_pair.curr_points.clone(),
    )
}


