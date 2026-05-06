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
        eprintln!("[GMC] Создание GmcTracker...");
        let detector = ORB::create(
            2000, 1.2, 8, 31, 0, 2,
            opencv::features2d::ORB_ScoreType::HARRIS_SCORE, 31, 20,
        )?;
        eprintln!("[GMC] ORB создан: nfeatures=2000, score=HARRIS_SCORE");
        let matcher = BFMatcher::create(NORM_HAMMING, false)?;
        eprintln!("[GMC] BFMatcher создан: NORM_HAMMING, crossCheck=true");
        eprintln!("[GMC] GmcTracker успешно создан");
        Ok(Self { detector, matcher, prev_frame: None, prev_kp: None, prev_desc: None })
    }

    pub fn process(&mut self, frame: &Mat) -> Result<(Option<Mat>, Option<Mat>), Box<dyn Error>> {
        let gray = Self::to_grayscale(frame)?;
        eprintln!("[GMC] Кадр преобразован в grayscale, размер: {:?}", gray.size());

        let mut kp = Vector::new();
        let mut desc = Mat::default();
        self.detector.detect_and_compute(&gray, &core::Mat::default(), &mut kp, &mut desc, false)?;
        eprintln!("[GMC] Детекция: найдено {} ключевых точек, дескрипторов: {}", kp.len(), desc.rows());

        let result = if let (Some(prev_kp), Some(prev_desc)) = (&self.prev_kp, &self.prev_desc) {
            eprintln!("[GMC] Сопоставление с предыдущим кадром: {} vs {} точек", prev_kp.len(), kp.len());
            let matches = Self::match_features(&mut self.matcher, prev_desc, &desc)?;
            eprintln!("[GMC] После фильтрации: {} матчей", matches.len());

            if matches.len() >= 4 {
                let src_pts = Self::extract_points(prev_kp, &matches, true)?;
                let dst_pts = Self::extract_points(&kp, &matches, false)?;
                eprintln!("[GMC] Подготовлено точек: src={}, dst={}", src_pts.len(), dst_pts.len());

                let mut mask = core::Mat::default();
                let h = calib3d::find_homography(&src_pts, &dst_pts, &mut mask, calib3d::RANSAC, 3.0)?;
                eprintln!("[GMC] Гомография вычислена: {:?}", h.size());

                let size = frame.size()?;
                let mut compensated = Mat::default();
                imgproc::warp_perspective(
                    frame, &mut compensated, &h, size,
                    imgproc::INTER_LINEAR, core::BORDER_CONSTANT, core::Scalar::default(),
                )?;
                eprintln!("[GMC] Компенсация выполнена");

                (Some(h), Some(compensated))
            } else {
                eprintln!("[GMC] Недостаточно матчей (< 4), пропуск");
                (None, None)
            }
        } else {
            eprintln!("[GMC] Первый кадр, сохранение как предыдущий");
            (None, None)
        };

        self.prev_frame = Some(gray.clone());
        self.prev_kp = Some(kp);
        self.prev_desc = Some(desc);
        Ok(result)
    }

    pub fn reset(&mut self) {
        eprintln!("[GMC] Сброс трекера");
        self.prev_frame = None;
        self.prev_kp = None;
        self.prev_desc = None;
    }

    fn to_grayscale(src: &Mat) -> Result<Mat, Box<dyn Error>> {
        let channels = src.channels();
        eprintln!("[GMC] to_grayscale: каналов={}, размер={:?}", channels, src.size());
        if channels == 1 {
            Ok(src.clone())
        } else {
            let mut dst = Mat::default();
            imgproc::cvt_color(src, &mut dst, imgproc::COLOR_BGR2GRAY, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
            eprintln!("[GMC] to_grayscale: конвертировано BGR->GRAY");
            Ok(dst)
        }
    }

    fn match_features(
        matcher: &mut core::Ptr<BFMatcher>,
        query_desc: &Mat,
        train_desc: &Mat,
    ) -> Result<Vector<DMatch>, Box<dyn Error>> {
        let mut matches = Vector::new();
        opencv::prelude::DescriptorMatcherTrait::clear(matcher)?;
        let train_vec: Vector<Mat> = Vector::from(vec![train_desc.clone()]);
        matcher.add(&train_vec)?;
        matcher.match_(query_desc, &mut matches, &core::Mat::default())?;
        eprintln!("[GMC] match_features: {} сырых матчей", matches.len());

        if matches.is_empty() {
            return Ok(matches);
        }

        let min_dist = matches.iter().map(|m| m.distance).fold(f32::INFINITY, f32::min);
        let threshold = (min_dist * 1.5).max(30.0);
        eprintln!("[GMC] match_features: min_dist={:.2}, threshold={:.2}", min_dist, threshold);

        let filtered: Vec<_> = matches.iter()
            .filter(|m| m.distance <= threshold)
            .collect();

        eprintln!("[GMC] match_features: после фильтрации {} матчей", filtered.len());

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
        eprintln!("[GMC] extract_points: извлечено {} точек", pts.len());
        Ok(Vector::from(pts))
    }
}
