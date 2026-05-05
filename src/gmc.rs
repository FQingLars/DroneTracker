use opencv::calib3d;
use opencv::core::{self, DMatch, KeyPoint, Mat, NORM_HAMMING, Point2f, Vector};
use opencv::features2d::{BFMatcher, DescriptorMatcherTrait, Feature2DTrait, ORB};
use opencv::imgproc;
use opencv::prelude::*;
use std::error::Error;

pub struct GmcTracker {
    detector: core::Ptr<ORB>,
    matcher: core::Ptr<BFMatcher>,
    prev_frame: Option<Mat>,
    prev_kp: Option<Vector<KeyPoint>>,
    prev_desc: Option<Mat>,
}

impl GmcTracker {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let detector = ORB::create(
            2000, 1.2, 8, 31, 0, 2,
            opencv::features2d::ORB_ScoreType::HARRIS_SCORE, 31, 20,
        )?;
        let matcher = BFMatcher::create(NORM_HAMMING, true)?;
        Ok(Self { detector, matcher, prev_frame: None, prev_kp: None, prev_desc: None })
    }

    pub fn process(&mut self, frame: &Mat) -> Result<(Option<Mat>, Option<Mat>), Box<dyn Error>> {
        let gray = Self::to_grayscale(frame)?;
        let mut kp = Vector::new();
        let mut desc = Mat::default();
        self.detector.detect_and_compute(&gray, &core::Mat::default(), &mut kp, &mut desc, false)?;

        let result = if let (Some(prev_kp), Some(prev_desc)) = (&self.prev_kp, &self.prev_desc) {
            let matches = Self::match_features(&mut self.matcher, prev_desc, &desc)?;
            if matches.len() >= 4 {
                let src_pts = Self::extract_points(prev_kp, &matches, true)?;
                let dst_pts = Self::extract_points(&kp, &matches, false)?;
                let mut mask = core::Mat::default();
                let h = calib3d::find_homography(&src_pts, &dst_pts, &mut mask, calib3d::RANSAC, 3.0)?;

                let size = frame.size()?;
                let mut compensated = Mat::default();
                imgproc::warp_perspective(
                    frame, &mut compensated, &h, size,
                    imgproc::INTER_LINEAR, core::BORDER_CONSTANT, core::Scalar::default(),
                )?;

                (Some(h), Some(compensated))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        self.prev_frame = Some(gray.clone());
        self.prev_kp = Some(kp);
        self.prev_desc = Some(desc);
        Ok(result)
    }

    pub fn reset(&mut self) {
        self.prev_frame = None;
        self.prev_kp = None;
        self.prev_desc = None;
    }

    fn to_grayscale(src: &Mat) -> Result<Mat, Box<dyn Error>> {
        if src.channels() == 1 {
            Ok(src.clone())
        } else {
            let mut dst = Mat::default();
            imgproc::cvt_color(src, &mut dst, imgproc::COLOR_BGR2GRAY, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
            Ok(dst)
        }
    }

    fn match_features(
        matcher: &mut core::Ptr<BFMatcher>,
        desc1: &Mat,
        _desc2: &Mat,
    ) -> Result<Vector<DMatch>, Box<dyn Error>> {
        let mut matches = Vector::new();
        matcher.match_(desc1, &mut matches, &core::Mat::default())?;
        if matches.is_empty() { return Ok(matches); }
        let min_dist = matches.iter().map(|m| m.distance).fold(f32::INFINITY, f32::min);
        let threshold = (min_dist * 1.5).max(30.0);
        let filtered: Vec<_> = matches.iter()
            .filter(|m| m.distance <= threshold)
            .collect();
        Ok(Vector::from(filtered))
    }

    fn extract_points(
        kp: &Vector<KeyPoint>,
        matches: &Vector<DMatch>,
        is_src: bool,
    ) -> Result<Vector<Point2f>, Box<dyn Error>> {
        let pts: Vec<Point2f> = matches.iter().map(|m| {
            let idx = if is_src { m.query_idx } else { m.train_idx };
            let k = kp.get(idx as usize).unwrap();
            let pt = k.pt();
            Point2f { x: pt.x, y: pt.y }
        }).collect();
        Ok(Vector::from(pts))
    }
}
