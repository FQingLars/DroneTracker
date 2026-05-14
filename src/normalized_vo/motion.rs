// ============================================================================
// motion.rs — Оценка движения: существенная матрица, выделение позы,
//              триангуляция точек в нормированном пространстве.
// ============================================================================
// Математика:
//   E = [t]_× R  (существенная матрица 3×3)
//   x2^T * E * x1 = 0  (эпиполярное ограничение)
//   E = U * diag(1,1,0) * V^T  (сингулярное ограничение)
//   Из E извлекаются 4 возможные (R, t), выбирается одна по
//   критерию положительной глубины для всех триангулированных точек.
// ============================================================================

use nalgebra::{Matrix3, Vector3, SVD, Unit};
use crate::normalized_vo::types::{
    NormCoords, NormPoint3, Pose, Rotation, UnitTranslation, VoError, MotionEstimate,
};

/// Минимальное количество точек для 8-точечного алгоритма
const MIN_POINTS_8PT: usize = 8;

/// Порог для RANSAC (в нормированных координатах)
const RANSAC_THRESHOLD: f64 = 1e-3;
/// Количество итераций RANSAC
const RANSAC_ITER: usize = 200;

// ===================== 8-ТОЧЕЧНЫЙ АЛГОРИТМ ================================

/// Вычисляет существенную матрицу E по набору соответствующих точек
/// в нормированных координатах. Использует 8-точечный алгоритм с RANSAC.
///
/// # Аргументы
/// * `points1` — точки в первом кадре (нормированные координаты)
/// * `points2` — соответствующие точки во втором кадре
///
/// # Возвращает
/// Существенную матрицу E (3×3)
///
/// # Ошибки
/// * `InsufficientMatches` — если точек меньше 8
/// * `DegenerateMotion` — если точки коллинеарны или движение вырождено
pub fn compute_essential_matrix(
    points1: &[NormCoords],
    points2: &[NormCoords],
) -> Result<(Matrix3<f64>, Vec<usize>), VoError> {
    if points1.len() < MIN_POINTS_8PT || points2.len() < MIN_POINTS_8PT {
        return Err(VoError::InsufficientMatches(points1.len().min(points2.len())));
    }

    if points1.len() != points2.len() {
        return Err(VoError::InsufficientMatches(
            points1.len().min(points2.len()),
        ));
    }

    let n = points1.len();

    // RANSAC: ищем лучшую E с максимальным числом inliers
    let mut best_inliers = Vec::new();
    let mut best_E = Matrix3::zeros();
    let mut max_inliers = 0usize;

    for iter in 0..RANSAC_ITER {
        // Выбираем 8 случайных точек с помощью встроенного LCG
        // (без внешней зависимости rand)
        let sample = random_sample_8(n, iter as u64);

        let sampled1: Vec<NormCoords> = sample.iter().map(|&i| points1[i]).collect();
        let sampled2: Vec<NormCoords> = sample.iter().map(|&i| points2[i]).collect();

        let E = estimate_eight_point(&sampled1, &sampled2);

        // Считаем inliers по критерию Самсона (Sampson distance)
        let mut inliers = Vec::new();
        for i in 0..n {
            let x1 = points1[i];
            let x2 = points2[i];

            // Переводим в однородные координаты для умножения на E (3×3)
            let x1_h = Vector3::new(x1.x, x1.y, 1.0);
            let x2_h = Vector3::new(x2.x, x2.y, 1.0);

            // Вычисляем расстояние Самсона:
            // d = (x2^T * E * x1)^2 / ( (E*x1)_1^2 + (E*x1)_2^2 + (E^T*x2)_1^2 + (E^T*x2)_2^2 )
            let ex1 = &E * x1_h;
            let etx2 = &E.transpose() * x2_h;
            let x2t_ex1 = x2_h.dot(&ex1);

            let denom = ex1.x * ex1.x + ex1.y * ex1.y +
                        etx2.x * etx2.x + etx2.y * etx2.y;

            if denom < 1e-12 {
                continue;
            }

            let dist = (x2t_ex1 * x2t_ex1) / denom;
            if dist < RANSAC_THRESHOLD {
                inliers.push(i);
            }
        }

        if inliers.len() > max_inliers {
            max_inliers = inliers.len();
            best_inliers = inliers;
            best_E = E;
        }
    }

    if max_inliers < 8 {
        return Err(VoError::InsufficientMatches(max_inliers));
    }

    // Уточняем E по всем inliers
    let inliers_pts1: Vec<NormCoords> = best_inliers.iter().map(|&i| points1[i]).collect();
    let inliers_pts2: Vec<NormCoords> = best_inliers.iter().map(|&i| points2[i]).collect();
    best_E = estimate_eight_point(&inliers_pts1, &inliers_pts2);

    // Нормировка: след(E * E^T) = 2 (общая нормировка для E)
    let norm_e = (best_E * best_E.transpose()).trace().sqrt();
    if norm_e > 1e-12 {
        best_E /= norm_e;
    }

    // Проверка антигаллюцинации: детерминант E должен быть ~0
    // (сингулярное ограничение)
    let det_e = best_E.determinant();
    if det_e.abs() > 1e-3 {
        // Принудительно накладываем сингулярное ограничение
        best_E = enforce_singularity(&best_E);
    }

    Ok((best_E, best_inliers))
}

// ==================== ВСПОМОГАТЕЛЬНЫЙ LCG (RANDOM) ========================

/// Простейший LCG (Linear Congruential Generator) для RANSAC.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        SimpleRng { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // Константы из библиотеки glibc
        self.state = self.state.wrapping_mul(6364136223846793005)
                           .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_usize(&mut self, max: usize) -> usize {
        if max == 0 { return 0; }
        (self.next_u64() % max as u64) as usize
    }
}

/// Выбирает 8 случайных различных индексов из [0, n).
fn random_sample_8(n: usize, seed: u64) -> Vec<usize> {
    let mut rng = SimpleRng::new(seed.wrapping_add(1));
    let mut available: Vec<usize> = (0..n).collect();
    let mut sample = Vec::with_capacity(8);
    for j in 0..8 {
        let idx = rng.next_usize(n - j);
        sample.push(available.swap_remove(idx));
    }
    sample
}

/// 8-точечный алгоритм вычисления существенной матрицы.
///
/// # Математика
/// Для каждой пары точек (x1, x2) составляем уравнение:
/// [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] * vec(E) = 0
/// где x = x1, y = y1, x' = x2, y' = y2
///
/// Решаем Af = 0 через SVD: f — последняя строка V^T.
fn estimate_eight_point(points1: &[NormCoords], points2: &[NormCoords]) -> Matrix3<f64> {
    let n = points1.len();
    assert!(n >= 8, "8-точечный алгоритм требует минимум 8 точек");

    // Строим матрицу A размером n×9
    let mut a = vec![0.0f64; n * 9];

    for i in 0..n {
        let x1 = points1[i].x;
        let y1 = points1[i].y;
        let x2 = points2[i].x;
        let y2 = points2[i].y;

        // Строка: [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
        a[i * 9]     = x2 * x1;
        a[i * 9 + 1] = x2 * y1;
        a[i * 9 + 2] = x2;
        a[i * 9 + 3] = y2 * x1;
        a[i * 9 + 4] = y2 * y1;
        a[i * 9 + 5] = y2;
        a[i * 9 + 6] = x1;
        a[i * 9 + 7] = y1;
        a[i * 9 + 8] = 1.0;
    }

    // SVD: A = U * D * V^T
    // Используем nalgebra: строим матрицу n×9
    // К сожалению, nalgebra требует размеры на этапе компиляции для малых матриц,
    // поэтому используем динамическую матрицу
    let a_mat = nalgebra::DMatrix::from_row_slice(n, 9, &a);
    let svd = SVD::new(a_mat, true, true);
    
    // Решение: последний столбец V (соответствует минимальному сингулярному числу)
    let v_t = svd.v_t.expect("SVD V^T не вычислена");
    // Последняя строка V^T = последний столбец V
    let e_vec = v_t.row(v_t.nrows() - 1);

    // Преобразуем вектор 9×1 в матрицу 3×3
    let mut E = Matrix3::zeros();
    E[(0, 0)] = e_vec[0]; E[(0, 1)] = e_vec[1]; E[(0, 2)] = e_vec[2];
    E[(1, 0)] = e_vec[3]; E[(1, 1)] = e_vec[4]; E[(1, 2)] = e_vec[5];
    E[(2, 0)] = e_vec[6]; E[(2, 1)] = e_vec[7]; E[(2, 2)] = e_vec[8];

    E
}

/// Накладывает сингулярное ограничение: E = U * diag(1, 1, 0) * V^T
fn enforce_singularity(E: &Matrix3<f64>) -> Matrix3<f64> {
    let svd = SVD::new(E.clone(), true, true);
    let u = svd.u.expect("U не вычислена");
    let v_t = svd.v_t.expect("V^T не вычислена");

    // Приводим сингулярные числа к (1, 1, 0)
    let sigma = Matrix3::from_diagonal(&nalgebra::Vector3::new(1.0, 1.0, 0.0));
    u * sigma * v_t
}

/// Извлекает 4 возможные пары (R, t) из существенной матрицы E.
///
/// # Математика
/// E = U * diag(1,1,0) * V^T
/// W = [[0,-1,0],[1,0,0],[0,0,1]]
///
/// Четыре решения:
///   R1 = U * W * V^T,  t1 = ±u_3
///   R2 = U * W^T * V^T, t2 = ±u_3
/// где u_3 — третий столбец U.
pub fn decompose_essential_matrix(E: &Matrix3<f64>) -> [Pose; 4] {
    let svd = SVD::new(E.clone(), true, true);
    let u = svd.u.expect("U не вычислена");
    let v_t = svd.v_t.expect("V^T не вычислена");
    let v = v_t.transpose();

    // Матрица W
    let w = Matrix3::new(
        0.0, -1.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 0.0, 1.0,
    );
    let w_t = w.transpose();

    // Третий столбец U — направление перемещения
    let u3 = u.column(2).into_owned();

    // Проверка: детерминанты U и V должны быть > 0 для корректной ротации
    let det_u = u.determinant();
    let det_v = v.determinant();
    
    let u_adj = if det_u < 0.0 { -u } else { u };
    let v_adj = if det_v < 0.0 { -v } else { v };
    let u3_adj = u_adj.column(2).into_owned();

    // Четыре комбинации
    let r1 = u_adj * w * v_adj.transpose();
    let r2 = u_adj * w_t * v_adj.transpose();
    let t_pos = Unit::try_new(u3_adj, 1e-12)
        .unwrap_or_else(|| Unit::new_normalize(Vector3::new(1.0, 0.0, 0.0)));
    let t_neg = Unit::new_normalize(-u3_adj);

    [
        Pose::new(r1, t_pos.into_inner()),
        Pose::new(r1, t_neg.into_inner()),
        Pose::new(r2, t_pos.into_inner()),
        Pose::new(r2, t_neg.into_inner()),
    ]
}

/// Выбирает правильную позу среди 4 кандидатов по критерию:
/// максимальное количество триангулированных точек перед обеими камерами.
pub fn disambiguate_pose(
    poses: &[Pose; 4],
    points1: &[NormCoords],
    points2: &[NormCoords],
) -> Option<Pose> {
    let mut best_pose = None;
    let mut max_positive = 0usize;

    for pose in poses {
        let mut positive_count = 0;

        for (&x1, &x2) in points1.iter().zip(points2.iter()) {
            // Триангулируем точку с помощью DLT
            if let Some(point_3d) = triangulate_point(&Pose::identity(), pose, x1, x2) {
                // Проверка: точка перед обеими камерами
                // В первой камере (I, 0): Z-координата = point_3d.z
                // Во второй камере (R, t): Z' = (R*point_3d + t).z
                let in_front1 = point_3d.z > 1e-12;
                let p2 = pose.rotation * point_3d + pose.translation.into_inner();
                let in_front2 = p2.z > 1e-12;

                if in_front1 && in_front2 {
                    positive_count += 1;
                }
            }
        }

        if positive_count > max_positive {
            max_positive = positive_count;
            best_pose = Some(pose.clone());
        }
    }

    if max_positive == 0 {
        // Антигаллюцинация: ни одна поза не даёт точек перед камерами
        return None;
    }

    best_pose
}

// ========================== ТРИАНГУЛЯЦИЯ ===================================

/// Линейная триангуляция методом DLT (Direct Linear Transform).
///
/// # Математика
/// Для каждой камеры с проективной матрицей P = [R|t] и точкой x:
///   x × (P * X) = 0
///
/// Это даёт два независимых уравнения на X:
///   x * P3^T - P1^T = 0
///   y * P3^T - P2^T = 0
///
/// где P1, P2, P3 — строки матрицы P.
///
/// Для двух камер собираем матрицу A (4×4) и решаем AX = 0 через SVD.
pub fn triangulate_point(
    pose1: &Pose,
    pose2: &Pose,
    x1: NormCoords,
    x2: NormCoords,
) -> Option<NormPoint3> {
    // Проективные матрицы
    // P1 = [R1|t1], P2 = [R2|t2]
    let p1_rows = pose_to_rows(pose1);
    let p2_rows = pose_to_rows(pose2);

    // Строим матрицу A (4×4)
    // Для первой камеры: x1 * P1[2] - P1[0] = 0, y1 * P1[2] - P1[1] = 0
    // Для второй камеры: x2 * P2[2] - P2[0] = 0, y2 * P2[2] - P2[1] = 0
    let mut a = nalgebra::SMatrix::<f64, 4, 4>::zeros();

    // Строка 0: x1 * P1_3 - P1_1
    for j in 0..4 {
        a[(0, j)] = x1.x * p1_rows[2][j] - p1_rows[0][j];
    }
    // Строка 1: y1 * P1_3 - P1_2
    for j in 0..4 {
        a[(1, j)] = x1.y * p1_rows[2][j] - p1_rows[1][j];
    }
    // Строка 2: x2 * P2_3 - P2_1
    for j in 0..4 {
        a[(2, j)] = x2.x * p2_rows[2][j] - p2_rows[0][j];
    }
    // Строка 3: y2 * P2_3 - P2_2
    for j in 0..4 {
        a[(3, j)] = x2.y * p2_rows[2][j] - p2_rows[1][j];
    }

    // SVD: A = U * D * V^T
    let svd = SVD::new(a, true, true);
    let v_t = svd.v_t.expect("V^T не вычислена");
    // Решение: последняя строка V^T (собственный вектор, соотв. мин. сингулярному числу)
    let homog = v_t.row(v_t.nrows() - 1);

    // Де-гомогенизация: X = (x/w, y/w, z/w)
    let w = homog[3];
    if w.abs() < 1e-12 {
        return None; // Точка на бесконечности
    }

    Some(NormPoint3::new(
        homog[0] / w,
        homog[1] / w,
        homog[2] / w,
    ))
}

/// Преобразует позу (R, t) в проективную матрицу [R|t] размером 3×4.
fn pose_to_rows(pose: &Pose) -> [[f64; 4]; 3] {
    let r = &pose.rotation;
    let t = pose.translation.into_inner();
    [
        [r[(0, 0)], r[(0, 1)], r[(0, 2)], t.x],
        [r[(1, 0)], r[(1, 1)], r[(1, 2)], t.y],
        [r[(2, 0)], r[(2, 1)], r[(2, 2)], t.z],
    ]
}

/// Триангулирует множество точек.
pub fn triangulate_points(
    pose1: &Pose,
    pose2: &Pose,
    points1: &[NormCoords],
    points2: &[NormCoords],
) -> Vec<(NormPoint3, bool)> {
    points1
        .iter()
        .zip(points2.iter())
        .map(|(x1, x2)| {
            let pt = triangulate_point(pose1, pose2, *x1, *x2);
            match pt {
                Some(p) => (p, true),
                None => (NormPoint3::zeros(), false),
            }
        })
        .collect()
}

/// Вычисляет среднюю глубину триангулированных точек и нормирует
/// координаты так, чтобы средняя глубина была равна 1.
///
/// Это фиксирует масштаб в нормированном пространстве.
pub fn normalize_depth(points: &mut [NormPoint3]) {
    if points.is_empty() {
        return;
    }
    let mean_z: f64 = points.iter().map(|p| p.z.abs()).sum::<f64>() / points.len() as f64;
    if mean_z > 1e-12 {
        for p in points.iter_mut() {
            *p /= mean_z;
        }
    }
}

impl Pose {
    /// Тождественная поза: камера в начале координат, направление взгляда вдоль +Z.
    /// t = (0, 0, 1) — камера смещена вдоль оси Z (условность нормированного пространства).
    pub fn identity() -> Self {
        Pose {
            rotation: Matrix3::identity(),
            translation: Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
        }
    }
}

/// Вычисляет среднюю ошибку повторной проекции для набора точек и позы.
pub fn compute_reprojection_error(
    pose: &Pose,
    world_points: &[NormPoint3],
    observed: &[NormCoords],
) -> f64 {
    if world_points.is_empty() {
        return 0.0;
    }

    let total_err: f64 = world_points
        .iter()
        .zip(observed.iter())
        .filter_map(|(wp, obs)| {
            pose.project(wp).map(|proj| (proj - obs).norm_squared())
        })
        .sum();

    (total_err / world_points.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Vector3, Unit};

    /// Тест: проверка 8-точечного алгоритма на синтетических данных
    #[test]
    fn test_eight_point_known_motion() {
        // Создаём синтетическое движение: поворот на 5° вокруг Y,
        // перемещение вдоль X
        let angle = 5.0f64.to_radians();
        let true_R = Matrix3::new(
            angle.cos(), 0.0, angle.sin(),
            0.0, 1.0, 0.0,
            -angle.sin(), 0.0, angle.cos(),
        );
        let true_t = Vector3::new(0.1, 0.0, 0.0);
        let _true_t_unit = Unit::new_normalize(true_t);

        // Генерируем 3D точки по сетке (детерминированно)
        let mut points1 = Vec::new();
        let mut points2 = Vec::new();

        for i in 0..7 {
            for j in 0..7 {
                let pt_3d = Vector3::new(
                    -0.6 + i as f64 * 0.2,
                    -0.6 + j as f64 * 0.2,
                    3.0 + (i + j) as f64 * 0.5,
                );

                // Проецируем в первый кадр (I, 0)
                let x1 = NormCoords::new(pt_3d.x / pt_3d.z, pt_3d.y / pt_3d.z);

                // Проецируем во второй кадр
                let pt2 = true_R * pt_3d + true_t;
                if pt2.z <= 0.0 { continue; }
                let x2 = NormCoords::new(pt2.x / pt2.z, pt2.y / pt2.z);

                points1.push(x1);
                points2.push(x2);
            }
        }

        // Вычисляем E
        let (E, inliers) = compute_essential_matrix(&points1, &points2)
            .expect("8-точечный алгоритм должен успешно работать");
        assert!(inliers.len() >= 8, "Должно быть минимум 8 inliers");

        // Декомпозируем E
        let poses = decompose_essential_matrix(&E);
        let best_pose = disambiguate_pose(&poses, &points1, &points2)
            .expect("Должна быть выбрана поза");

        // Проверяем: угол между осями Z должен быть мал
        let z1 = Vector3::z();
        let z2 = best_pose.rotation * Vector3::z();
        let angle_err = z1.angle(&z2).abs();
        assert!(
            angle_err < 0.2,
            "Ошибка угла поворота слишком велика: {} рад",
            angle_err
        );
    }

    /// Тест: проверка триангуляции
    #[test]
    fn test_triangulation() {
        // Первая камера: идентичная (начало координат, ориентирована по Z)
        let pose1 = Pose::identity();

        // Вторая камера: повёрнута на 2° вокруг Y, сдвинута вбок
        let angle = 2.0f64.to_radians();
        let r2 = Matrix3::new(
            angle.cos(), 0.0, angle.sin(),
            0.0, 1.0, 0.0,
            -angle.sin(), 0.0, angle.cos(),
        );
        let pose2 = Pose::new(r2, Vector3::new(0.3, 0.0, 0.1));

        let true_pt = NormPoint3::new(0.5, -0.3, 5.0);

        let x1 = pose1.project(&true_pt).unwrap();
        let x2 = pose2.project(&true_pt).unwrap();

        let recovered = triangulate_point(&pose1, &pose2, x1, x2)
            .expect("Триангуляция должна сойтись");

        // Восстановленная точка должна быть коллинеарна истинной
        let ratio = recovered.z / true_pt.z;
        assert!(
            ratio > 0.0,
            "Глубина должна быть положительной, получена {}",
            ratio
        );
        assert!(
            (recovered.x - true_pt.x * ratio).abs() < 1e-3 &&
            (recovered.y - true_pt.y * ratio).abs() < 1e-3,
            "Триангуляция неверна: получено {:?}, истина {:?}, ratio={}",
            recovered, true_pt, ratio,
        );
    }
}
