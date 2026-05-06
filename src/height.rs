use opencv::calib3d::{self, decompose_homography_mat};
use opencv::core::{self, DMatch, KeyPoint, Mat, NORM_HAMMING, Point2f, Vector};
use opencv::features2d::{BFMatcher, DescriptorMatcherTrait, Feature2DTrait, ORB};
use opencv::imgproc;
use opencv::prelude::*;
use std::error::Error;

pub struct HeightEstimator {
    k: Mat,
    reference_height: f64,
    detector: core::Ptr<ORB>,
    matcher: core::Ptr<BFMatcher>,
    prev_height: f64,
}

impl HeightEstimator {
    pub fn new(focal_px: f64, cx: f64, cy: f64, reference_height: f64) -> Result<Self, Box<dyn Error>> {
        let k = core::Mat::from_slice_2d(&[
            &[focal_px, 0.0, cx],
            &[0.0, focal_px, cy],
            &[0.0, 0.0, 1.0],
        ])?;

        let detector = ORB::create(
            1500, 1.2, 8, 31, 0, 2,
            opencv::features2d::ORB_ScoreType::HARRIS_SCORE, 31, 20,
        )?;

        let matcher = BFMatcher::create(NORM_HAMMING, false)?;

        Ok(Self {
            k,
            reference_height,
            detector,
            matcher,
            prev_height: reference_height,
        })
    }

    pub fn estimate(&mut self, frame0: &Mat, frame1: &Mat) -> Result<f64, Box<dyn Error>> {
        let gray0 = Self::to_grayscale(frame0)?;
        let gray1 = Self::to_grayscale(frame1)?;

        let (kp0, desc0) = self.detect(&gray0)?;
        let (kp1, desc1) = self.detect(&gray1)?;

        if kp0.len() < 4 || kp1.len() < 4 {
            return Ok(self.prev_height);
        }

        let matches = Self::match_features(&mut self.matcher, &desc0, &desc1)?;

        if matches.len() < 4 {
            return Ok(self.prev_height);
        }

        let src_pts = Self::extract_points(&kp0, &matches, true)?;
        let dst_pts = Self::extract_points(&kp1, &matches, false)?;

        let mut mask = core::Mat::default();
        let h = calib3d::find_homography(&src_pts, &dst_pts, &mut mask, calib3d::RANSAC, 3.0)?;

        let mut rs = Vector::<Mat>::new();
        let mut ts = Vector::<Mat>::new();
        let mut ns = Vector::<Mat>::new();
        decompose_homography_mat(&h, &self.k, &mut rs, &mut ts, &mut ns)?;

        if rs.is_empty() {
            return Ok(self.prev_height);
        }

        let (_, t, n) = Self::select_valid_pose(&rs, &ts, &ns)?;

        let t_norm = Self::norm(&t)?;
        if t_norm < 1e-6 {
            return Ok(self.prev_height);
        }

        // Используем n_z и t_z для оценки относительного изменения высоты
        let nz = *n.at_2d::<f64>(2, 0)?;
        let tz = *t.at_2d::<f64>(2, 0)?;

        // Если нормаль направлена к камере (nz > 0), то высота пропорциональна 1/||t||
        // Используем предыдущую высоту как опорную
        let height = if nz > 0.0 {
            self.prev_height / t_norm.max(0.1)
        } else {
            self.prev_height
        };

        // Ограничиваем изменение высоты (чтобы не улетать)
        let new_height = if height > 0.0 && height < 500.0 {
            height
        } else {
            self.prev_height
        };

        self.prev_height = new_height;
        Ok(new_height.max(0.1))
    }

    fn select_valid_pose(
        rs: &Vector<Mat>,
        ts: &Vector<Mat>,
        ns: &Vector<Mat>,
    ) -> Result<(Mat, Mat, Mat), Box<dyn Error>> {
        let mut best_score = f64::MIN;
        let mut best_idx = 0;

        for i in 0..rs.len() {
            let _r = rs.get(i)?;
            let t = ts.get(i)?;
            let n = ns.get(i)?;

            let nz = *n.at_2d::<f64>(2, 0)?;
        let tz = *t.at_2d::<f64>(2, 0)?;

            if nz <= 0.0 {
                continue;
            }

            let mut score = nz;
            if tz < 0.0 {
                score += 1.0;
            }

            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        if best_score == f64::MIN {
            best_idx = 0;
        }

        Ok((rs.get(best_idx)?, ts.get(best_idx)?, ns.get(best_idx)?))
    }

    fn norm(t: &Mat) -> Result<f64, Box<dyn Error>> {
        let tx = *t.at_2d::<f64>(0, 0)?;
        let ty = *t.at_2d::<f64>(1, 0)?;
        let tz = *t.at_2d::<f64>(2, 0)?;
        Ok((tx*tx + ty*ty + tz*tz).sqrt())
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

    fn detect(&mut self, img: &Mat) -> Result<(Vector<KeyPoint>, Mat), Box<dyn Error>> {
        let mut kp = Vector::new();
        let mut desc = Mat::default();
        self.detector.detect_and_compute(img, &core::Mat::default(), &mut kp, &mut desc, false)?;
        Ok((kp, desc))
    }

    fn match_features(
        matcher: &mut core::Ptr<BFMatcher>,
        desc0: &Mat,
        desc1: &Mat,
    ) -> Result<Vector<DMatch>, Box<dyn Error>> {
        let mut matches = Vector::new();
        opencv::prelude::DescriptorMatcherTrait::clear(matcher)?;
        let train_vec: Vector<Mat> = Vector::from(vec![desc1.clone()]);
        matcher.add(&train_vec)?;
        matcher.match_(desc0, &mut matches, &core::Mat::default())?;

        if matches.is_empty() {
            return Ok(matches);
        }

        let min_dist = matches.iter().map(|m| m.distance).fold(f32::INFINITY, f32::min);
        let threshold = (min_dist * 0.75).max(10.0);

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
