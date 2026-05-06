use opencv::calib3d::{self, decompose_homography_mat};
use opencv::core::{self, DMatch, KeyPoint, Mat, NORM_HAMMING, Point2f, Rect, Vector};
use opencv::features2d::{BFMatcher, DescriptorMatcherTrait, Feature2DTrait, ORB};
use opencv::imgproc;
use opencv::prelude::*;
use std::error::Error;
use crate::types::Frame;

pub struct HeightEstimator {
    k: Mat,
    reference_height: f64,
    detector: core::Ptr<ORB>,
    matcher: core::Ptr<BFMatcher>,
}

impl HeightEstimator {
    pub fn new(focal_px: f64, cx: f64, cy: f64, reference_height: f64) -> Result<Self, Box<dyn Error>> {
        eprintln!("[HEIGHT] Создание HeightEstimator: focal={:.1}, cx={:.1}, cy={:.1}, ref_height={:.1}",
            focal_px, cx, cy, reference_height);
        let k = core::Mat::from_slice_2d(&[
            &[focal_px, 0.0, cx],
            &[0.0, focal_px, cy],
            &[0.0, 0.0, 1.0],
        ])?;
        eprintln!("[HEIGHT] Матрица камеры K создана: {:?}", k.size());

        let detector = ORB::create(
            1500, 1.2, 8, 31, 0, 2,
            opencv::features2d::ORB_ScoreType::HARRIS_SCORE, 31, 20,
        )?;
        eprintln!("[HEIGHT] ORB создан: nfeatures=1500, score=HARRIS_SCORE");

        let matcher = BFMatcher::create(NORM_HAMMING, true)?;
        eprintln!("[HEIGHT] BFMatcher создан: NORM_HAMMING, crossCheck=true");
        eprintln!("[HEIGHT] HeightEstimator успешно создан");

        Ok(Self { k, reference_height, detector, matcher })
    }

    pub fn estimate(&mut self, frame0: &Frame, frame1: &Frame, roi: Option<Rect>) -> Result<f64, Box<dyn Error>> {
        eprintln!("[HEIGHT] Оценка высоты между кадрами {} и {}", frame0.id, frame1.id);
        let gray0 = Self::to_grayscale(&frame0.data)?;
        let gray1 = Self::to_grayscale(&frame1.data)?;

        let (kp0, desc0) = self.detect(&gray0)?;
        let (kp1, desc1) = self.detect(&gray1)?;
        eprintln!("[HEIGHT] Детекция завершена: кадр0={} точек, кадр1={} точек", kp0.len(), kp1.len());

        let (kp0_f, desc0_f) = Self::filter_by_roi(&kp0, &desc0, roi)?;
        let (kp1_f, desc1_f) = Self::filter_by_roi(&kp1, &desc1, roi)?;
        eprintln!("[HEIGHT] После фильтрации ROI: кадр0_f={} точек, кадр1_f={} точек", kp0_f.len(), kp1_f.len());

        if kp0_f.len() < 10 || kp1_f.len() < 10 {
            eprintln!("[HEIGHT] ОШИБКА: недостаточно фичей в ROI ({} и {})", kp0_f.len(), kp1_f.len());
            return Err("Недостаточно фичей в ROI".into());
        }

        let matches = Self::match_features(&mut self.matcher, &desc0_f, &desc1_f)?;
        eprintln!("[HEIGHT] Матчей после фильтрации: {}", matches.len());

        if matches.len() < 8 {
            eprintln!("[HEIGHT] ОШИБКА: слишком мало матчей для RANSAC ({})", matches.len());
            return Err("Слишком мало матчей для RANSAC".into());
        }

        let src_pts = Self::extract_points(&kp0_f, &matches, true)?;
        let dst_pts = Self::extract_points(&kp1_f, &matches, false)?;
        eprintln!("[HEIGHT] Точки для гомографии: src={}, dst={}", src_pts.len(), dst_pts.len());

        let mut mask = core::Mat::default();
        let h = calib3d::find_homography(&src_pts, &dst_pts, &mut mask, calib3d::RANSAC, 3.0)?;
        eprintln!("[HEIGHT] Гомография вычислена: {:?}", h.size());

        let mut rs = Vector::<Mat>::new();
        let mut ts = Vector::<Mat>::new();
        let mut ns = Vector::<Mat>::new();
        decompose_homography_mat(&h, &self.k, &mut rs, &mut ts, &mut ns)?;
        eprintln!("[HEIGHT] Декомпозиция: получено {} решений", rs.len());

        if rs.is_empty() {
            eprintln!("[HEIGHT] ОШИБКА: декомпозиция не вернула решений");
            return Err("Декомпозиция гомографии не вернула решений".into());
        }

        let (t_valid, n_valid) = Self::select_valid_pose(&rs, &ts, &ns)?;

        let t_z = t_valid.at_2d::<f64>(2, 0)?.abs().max(1e-6);
        let n_z = *n_valid.at_2d::<f64>(2, 0)?;
        eprintln!("[HEIGHT] t_z={:.6}, n_z={:.6}", t_z, n_z);

        let height = self.reference_height * (n_z / t_z);
        eprintln!("[HEIGHT] Результат высоты: {:.4} (ref_height={:.2}, n_z/t_z={:.6})", height, self.reference_height, n_z / t_z);

        Ok(height.max(0.1))
    }

    fn to_grayscale(src: &Mat) -> Result<Mat, Box<dyn Error>> {
        let channels = src.channels();
        eprintln!("[HEIGHT] to_grayscale: channels={}, size={:?}", channels, src.size());
        if channels == 1 {
            Ok(src.clone())
        } else {
            let mut dst = Mat::default();
            imgproc::cvt_color(src, &mut dst, imgproc::COLOR_BGR2GRAY, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
            eprintln!("[HEIGHT] to_grayscale: конвертировано BGR->GRAY");
            Ok(dst)
        }
    }

    fn detect(&mut self, img: &Mat) -> Result<(Vector<KeyPoint>, Mat), Box<dyn Error>> {
        let mut kp = Vector::new();
        let mut desc = Mat::default();
        self.detector.detect_and_compute(img, &core::Mat::default(), &mut kp, &mut desc, false)?;
        eprintln!("[HEIGHT] detect: {} ключевых точек, дескрипторов: {} строк", kp.len(), desc.rows());
        Ok((kp, desc))
    }

    fn filter_by_roi(
        kp: &Vector<KeyPoint>,
        desc: &Mat,
        roi: Option<Rect>,
    ) -> Result<(Vector<KeyPoint>, Mat), Box<dyn Error>> {
        let Some(r) = roi else {
            eprintln!("[HEIGHT] filter_by_roi: ROI не задан, возвращаем все {} точек", kp.len());
            return Ok((kp.clone(), desc.clone()));
        };
        eprintln!("[HEIGHT] filter_by_roi: прямоугольник ({},{},{}x{})", r.x, r.y, r.width, r.height);

        let valid_indices: Vec<usize> = kp.iter().enumerate()
            .filter(|(_, k)| {
                let pt = k.pt();
                pt.x >= r.x as f32 && pt.x <= (r.x + r.width) as f32 &&
                    pt.y >= r.y as f32 && pt.y <= (r.y + r.height) as f32
            })
            .map(|(i, _)| i)
            .collect();

        eprintln!("[HEIGHT] filter_by_roi: {} точек попали в ROI из {}", valid_indices.len(), kp.len());

        if valid_indices.is_empty() {
            eprintln!("[HEIGHT] filter_by_roi: ни одна точка не попала в ROI");
            return Ok((Vector::new(), Mat::default()));
        }

        let filtered_kp: Vec<_> = valid_indices.iter()
            .map(|&i| kp.get(i).unwrap().clone())
            .collect();
        let out_kp = Vector::from(filtered_kp);

        let mut out_desc = Mat::default();
        if !desc.empty() {
            for &idx in &valid_indices {
                let row = desc.row(idx as i32)?;
                out_desc.push_back(&row)?;
            }
        }

        eprintln!("[HEIGHT] filter_by_roi: результат {} точек, дескрипторов {} строк", out_kp.len(), out_desc.rows());
        Ok((out_kp, out_desc))
    }

    fn match_features(
        matcher: &mut core::Ptr<BFMatcher>,
        desc1: &Mat,
        _desc2: &Mat,
    ) -> Result<Vector<DMatch>, Box<dyn Error>> {
        let mut matches = Vector::new();
        matcher.match_(desc1, &mut matches, &core::Mat::default())?;
        eprintln!("[HEIGHT] match_features: {} сырых матчей", matches.len());

        if matches.is_empty() {
            return Ok(matches);
        }

        let min_dist = matches.iter().map(|m| m.distance).fold(f32::INFINITY, f32::min);
        let threshold = (min_dist * 0.75).max(10.0);
        eprintln!("[HEIGHT] match_features: min_dist={:.2}, threshold={:.2}", min_dist, threshold);

        let filtered: Vec<_> = matches.iter()
            .filter(|m| m.distance <= threshold)
            .collect();
        eprintln!("[HEIGHT] match_features: после фильтрации {} матчей", filtered.len());

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
        eprintln!("[HEIGHT] extract_points: извлечено {} точек", pts.len());
        Ok(Vector::from(pts))
    }

    fn select_valid_pose(
        rs: &Vector<Mat>,
        ts: &Vector<Mat>,
        ns: &Vector<Mat>,
    ) -> Result<(Mat, Mat), Box<dyn Error>> {
        let mut best_idx = 0;
        let mut max_nz = f64::MIN;

        for i in 0..rs.len() {
            let n = ns.get(i)?;
            let nz = *n.at_2d::<f64>(2, 0)?;
            eprintln!("[HEIGHT] select_valid_pose: решение {}: n_z={:.6}", i, nz);
            if nz > max_nz {
                max_nz = nz;
                best_idx = i;
            }
        }

        eprintln!("[HEIGHT] select_valid_pose: выбрано решение {} с n_z={:.6}", best_idx, max_nz);
        Ok((ts.get(best_idx)?, ns.get(best_idx)?))
    }
}
