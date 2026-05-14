// ============================================================================
// optimization.rs — Оптимизация позы методом Левенберга-Марквардта
//                   и локальная пучковая настройка (Bundle Adjustment)
// ============================================================================
// Минимизируется суммарная ошибка повторной проекции в нормированных
// координатах. Якобиан вычисляется аналитически. Масштаб не участвует
// в оптимизации — вектор t остаётся единичным.
//
// Геометрическая ошибка:
//   e_i = x_i - proj(R * X_i + t)
//
// Параметризация обновления позы:
//   R ← R * exp(δθ^),  t ← (t + δt) / ||t + δt||
// где δθ — возмущение угла, δt — возмущение перемещения.
// ============================================================================

use nalgebra::{Matrix3, Vector3, Vector2, SMatrix};
use crate::normalized_vo::types::{
    NormCoords, NormPoint3, Pose, Rotation, UnitTranslation, VoError,
};

/// Максимальное количество итераций LM
const LM_MAX_ITER: usize = 30;

/// Начальное значение λ для LM
const LM_LAMBDA_INIT: f64 = 1e-3;

/// Коэффициент увеличения λ при неудачном шаге
const LM_LAMBDA_INC: f64 = 2.0;

/// Коэффициент уменьшения λ при удачном шаге
const LM_LAMBDA_DEC: f64 = 0.5;

/// Порог сходимости (относительное изменение ошибки)
const LM_CONV_THRESH: f64 = 1e-8;

/// Минимальная глубина для стабильных вычислений
const MIN_Z: f64 = 1e-12;

// ==================== ГЕОМЕТРИЧЕСКАЯ ОШИБКА И ЯКОБИАН ======================

/// Вычисляет ошибку повторной проекции для одной точки.
///
/// # Математика
/// e = [u_obs - u_proj, v_obs - v_proj]^T
/// где u_proj = p_x / p_z, v_proj = p_y / p_z
///       p = R * X + t
pub fn reprojection_error(
    pose: &Pose,
    world_point: &NormPoint3,
    observed: &NormCoords,
) -> Option<Vector2<f64>> {
    let projected = pose.project(world_point)?;
    Some(observed - projected)
}

/// Вычисляет якобиан ошибки повторной проекции по параметрам позы.
///
/// # Математика
/// Пусть p = R * X + t = [p_x, p_y, p_z]^T.
/// Проекция: u = p_x/p_z, v = p_y/p_z.
///
/// ∂[u,v]/∂p = [1/p_z, 0, -p_x/p_z^2;
///              0, 1/p_z, -p_y/p_z^2]
///
/// ∂p/∂[δθ, δt] = [-R * skew(X) | I_3]
///
/// Якобиан e по [δθ, δt]:
///   J_e = ∂[u,v]/∂p * ∂p/∂[δθ, δt]  (размер 2×6)
pub fn pose_jacobian(
    pose: &Pose,
    world_point: &NormPoint3,
) -> Option<SMatrix<f64, 2, 6>> {
    let p = pose.rotation * world_point + pose.translation.into_inner();

    if p.z <= MIN_Z {
        return None;
    }

    // ∂[u,v]/∂p = [1/p_z, 0, -p_x/p_z^2; 0, 1/p_z, -p_y/p_z^2]
    let dz = 1.0 / p.z;
    let dz2 = dz * dz;
    let du_dp = nalgebra::RowVector3::new(dz, 0.0, -p.x * dz2);
    let dv_dp = nalgebra::RowVector3::new(0.0, dz, -p.y * dz2);

    // ∂p/∂δθ = -R * skew(X) = R * skew(X)^T ... 
    // Используем: R * (δθ × X) = -(R * X) × (R * δθ)... нет.
    // Правильно: ∂p/∂δθ = -R * [X]_×
    // где [X]_× — кососимметричная матрица X
    let r_skew_x = &pose.rotation * skew_symmetric(world_point);
    let dp_dtheta = -r_skew_x;

    // ∂p/∂δt = I_3
    let dp_dt = Matrix3::identity();

    // Полный якобиан 2×6
    let mut J = SMatrix::<f64, 2, 6>::zeros();

    // Первая строка: du_dp * [dp_dtheta | dp_dt]
    for j in 0..3 {
        J[(0, j)] = (du_dp * dp_dtheta.column(j))[(0, 0)];
        J[(0, 3 + j)] = (du_dp * dp_dt.column(j))[(0, 0)];
    }
    // Вторая строка: dv_dp * [dp_dtheta | dp_dt]
    for j in 0..3 {
        J[(1, j)] = (dv_dp * dp_dtheta.column(j))[(0, 0)];
        J[(1, 3 + j)] = (dv_dp * dp_dt.column(j))[(0, 0)];
    }

    Some(J)
}

/// Кососимметричная матрица 3×3 для вектора v.
/// [v]_× = [[0, -v_z, v_y], [v_z, 0, -v_x], [-v_y, v_x, 0]]
fn skew_symmetric(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -v.z, v.y,
        v.z, 0.0, -v.x,
        -v.y, v.x, 0.0,
    )
}

// ==================== ФОТОМЕТРИЧЕСКАЯ ОШИБКА ===============================

/// Вычисляет фотометрическую ошибку и её якобиан для точки.
///
/// Фотометрическая ошибка:
///   e_photo = I_ref(proj_ref(X)) - I_cur(proj_cur(X))
///
/// Якобиан:
///   ∂e_photo/∂ξ = ∇I_cur(u_cur) * ∂u_cur/∂ξ
/// где ∇I_cur — градиент изображения в текущем кадре,
///   ∂u_cur/∂ξ — геометрический якобиан проекции.
pub fn photometric_error_jacobian(
    // TODO: в реальной системе сюда подаются изображения для вычисления
    // фотометрической ошибки. Для каркасной реализации используем заглушку.
    _pose: &Pose,
    _world_point: &NormPoint3,
    _image_grad_x: f64,
    _image_grad_y: f64,
) -> Option<(f64, SMatrix<f64, 1, 6>)> {
    // В данной реализации фотометрическая ошибка добавлена как каркас.
    // Полная реализация требует доступа к изображению для вычисления
    // яркости и градиентов в точке проекции.
    //
    // Якобиан: J_photo = [grad_x, grad_y] * J_geo (2×6)
    // где J_geo — геометрический якобиан из pose_jacobian().
    None
}

// ============== ОПТИМИЗАЦИЯ ПОЗЫ ЛЕВЕНБЕРГА-МАРКВАРДТА ====================

/// Результат оптимизации позы.
#[derive(Debug)]
pub struct PoseOptimizationResult {
    pub pose: Pose,
    pub final_error: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Оптимизирует позу камеры методом Левенберга-Марквардта.
///
/// Минимизируется сумма квадратов ошибок повторной проекции:
///   E(R, t) = Σ_i ||x_i - proj(R * X_i + t)||^2
///
/// # Аргументы
/// * `initial_pose` — начальное приближение позы
/// * `world_points` — 3D точки в нормированном пространстве
/// * `observations` — наблюдения в нормированных координатах
///
/// # Возвращает
/// Оптимизированную позу, финальную ошибку, число итераций, флаг сходимости.
pub fn optimize_pose_lm(
    initial_pose: &Pose,
    world_points: &[NormPoint3],
    observations: &[NormCoords],
) -> Result<PoseOptimizationResult, VoError> {
    if world_points.len() != observations.len() || world_points.len() < 3 {
        return Err(VoError::InsufficientMatches(world_points.len()));
    }

    let mut pose = initial_pose.clone();
    let mut lambda = LM_LAMBDA_INIT;
    let mut best_error = compute_total_error(&pose, world_points, observations);
    let mut best_pose = pose.clone();
    let mut prev_error = best_error;
    let mut failed_steps = 0;

    for iteration in 0..LM_MAX_ITER {
        // Собираем нормальные уравнения: H * Δx = g
        // H = J^T * J, g = J^T * e
        let mut H = SMatrix::<f64, 6, 6>::zeros();
        let mut g = nalgebra::SVector::<f64, 6>::zeros();
        let mut current_error = 0.0;

        for i in 0..world_points.len() {
            let J_opt = pose_jacobian(&pose, &world_points[i]);
            let e_opt = reprojection_error(&pose, &world_points[i], &observations[i]);

            let (J, e) = match (J_opt, e_opt) {
                (Some(J), Some(e)) => (J, e),
                _ => continue,
            };

            H += J.transpose() * J;
            g += J.transpose() * e;
            current_error += e.norm_squared();
        }

        // Проверка сходимости: относительное изменение ошибки
        if iteration > 0 {
            let rel_change = (prev_error - current_error).abs() / prev_error.max(1e-12);
            if rel_change < LM_CONV_THRESH && current_error <= prev_error {
                best_pose = pose.clone();
                best_error = current_error;
                break;
            }
        }
        prev_error = current_error;

        // Решаем (H + λ * diag(H)) * Δx = g
        let diag_H = SMatrix::<f64, 6, 6>::from_diagonal(&nalgebra::SVector::<f64, 6>::new(
            H[(0, 0)], H[(1, 1)], H[(2, 2)], H[(3, 3)], H[(4, 4)], H[(5, 5)],
        ));
        let H_augmented = H + lambda * diag_H;

        let delta_x = match H_augmented.lu().solve(&g) {
            Some(dx) => dx,
            None => {
                lambda *= LM_LAMBDA_INC;
                failed_steps += 1;
                if failed_steps > 10 { break; }
                continue;
            }
        };

        // Обновляем позу
        let new_pose = apply_pose_update(&pose, &delta_x);
        let new_error = compute_total_error(&new_pose, world_points, observations);

        if new_error < current_error {
            // Удачный шаг: принимаем обновление
            pose = new_pose;
            if new_error < best_error {
                best_error = new_error;
                best_pose = pose.clone();
            }
            lambda *= LM_LAMBDA_DEC;
            failed_steps = 0;
        } else {
            // Неудачный шаг: увеличиваем λ
            lambda *= LM_LAMBDA_INC;
            failed_steps += 1;
            if failed_steps > 10 {
                break;
            }
        }
    }

    Ok(PoseOptimizationResult {
        pose: best_pose,
        final_error: best_error.sqrt(),
        iterations: LM_MAX_ITER,
        converged: failed_steps <= 10,
    })
}

/// Вычисляет суммарную ошибку повторной проекции.
fn compute_total_error(
    pose: &Pose,
    world_points: &[NormPoint3],
    observations: &[NormCoords],
) -> f64 {
    let mut total = 0.0;
    for i in 0..world_points.len() {
        if let Some(e) = reprojection_error(pose, &world_points[i], &observations[i]) {
            total += e.norm_squared();
        }
    }
    total
}

/// Применяет обновление Δx = [δθ; δt] к позе.
///
/// # Математика
/// R_new = R * exp(δθ^)  — обновление поворота через экспоненту so(3)
/// t_new = (t + δt) / ||t + δt||  — нормировка к единичной длине
fn apply_pose_update(pose: &Pose, delta: &nalgebra::SVector<f64, 6>) -> Pose {
    // Извлекаем δθ (угловое возмущение) и δt (линейное возмущение)
    let dtheta = Vector3::new(delta[0], delta[1], delta[2]);
    let dt = Vector3::new(delta[3], delta[4], delta[5]);

    // Обновление поворота: R_new = R * exp(δθ^)
    // Используем формулу Родрига для экспоненты so(3)
    let delta_R = rotation_from_axis_angle(&dtheta);
    let new_R = pose.rotation * delta_R;

    // Нормируем R обратно в SO(3) через SVD-проекцию
    let new_R = project_to_so3(&new_R);

    // Обновление перемещения: t_new = (t + δt) / ||t + δt||
    let t_vec = pose.translation.into_inner() + dt;
    let t_norm = t_vec.norm();

    let new_t = if t_norm > 1e-12 {
        UnitTranslation::new_normalize(t_vec)
    } else {
        // Антигаллюцинация: если норма нулевая, сохраняем предыдущий t
        pose.translation
    };

    Pose::new(new_R, new_t.into_inner())
}

/// Создаёт матрицу поворота из вектора угла (формула Родрига).
///
/// exp(θ^) = I + sin(θ) * [θ/||θ||]_× + (1 - cos(θ)) * ([θ/||θ||]_×)^2
fn rotation_from_axis_angle(theta: &Vector3<f64>) -> Rotation {
    let angle = theta.norm();
    if angle < 1e-12 {
        return Rotation::identity();
    }

    let axis = theta / angle;
    let s = angle.sin();
    let c = angle.cos();
    let one_minus_c = 1.0 - c;

    let ax = axis.x;
    let ay = axis.y;
    let az = axis.z;

    Rotation::new(
        c + ax * ax * one_minus_c,
        ax * ay * one_minus_c - az * s,
        ax * az * one_minus_c + ay * s,
        ay * ax * one_minus_c + az * s,
        c + ay * ay * one_minus_c,
        ay * az * one_minus_c - ax * s,
        az * ax * one_minus_c - ay * s,
        az * ay * one_minus_c + ax * s,
        c + az * az * one_minus_c,
    )
}

/// Проецирует матрицу на SO(3): R = U * V^T из SVD(R).
fn project_to_so3(mat: &Rotation) -> Rotation {
    let svd = nalgebra::SVD::new(mat.clone(), true, true);
    let u = svd.u.expect("U не вычислена");
    let v_t = svd.v_t.expect("V^T не вычислена");
    let mut r = u * v_t;

    // Гарантируем det = +1
    if r.determinant() < 0.0 {
        // Отражаем последний столбец U
        let mut u_mod = u;
        u_mod.column_mut(2).scale_mut(-1.0);
        r = u_mod * v_t;
    }

    r
}

/// Проверяет, что матрица поворота в SO(3) с допуском 1e-3.
/// Возвращает Ok(()) или Err с отклонением детерминанта.
pub fn check_rotation_validity(r: &Rotation) -> Result<(), VoError> {
    let det = r.determinant();
    let diff = (det - 1.0).abs();
    if diff > 1e-3 {
        return Err(VoError::RotationNotInSO3(diff));
    }
    // Проверка ортогональности: R * R^T ≈ I
    let rrt = r * r.transpose();
    let ortho_err = (rrt - Rotation::identity()).norm();
    if ortho_err > 1e-3 {
        return Err(VoError::RotationNotInSO3(ortho_err));
    }
    Ok(())
}

/// Проверяет, что вектор перемещения имеет единичную длину с допуском 1e-6.
pub fn check_translation_validity(t: &UnitTranslation) -> Result<(), VoError> {
    let norm = t.into_inner().norm();
    if (norm - 1.0).abs() > 1e-6 {
        return Err(VoError::ZeroTranslation);
    }
    Ok(())
}

// ==================== ЛОКАЛЬНАЯ ПУЧКОВАЯ НАСТРОЙКА =========================

/// Выполняет локальную пучковую настройку (Bundle Adjustment) для набора
/// ключевых кадров и их общих точек.
///
/// Оптимизируются позы всех ключевых кадров и 3D координаты всех точек
/// одновременно, минимизируя ошибку повторной проекции.
///
/// В данной реализации используется упрощённый подход с поочерёдной
/// оптимизацией поз (alternating optimization).
pub fn local_bundle_adjustment(
    keyframes: &mut [crate::normalized_vo::types::KeyFrame],
    map_points: &mut [crate::normalized_vo::types::MapPoint],
) -> Result<(), VoError> {
    if keyframes.is_empty() || map_points.is_empty() {
        return Ok(());
    }

    // Собираем наблюдения: для каждой точки, в каких кадрах она видна
    // и в каких координатах
    for kf_idx in 0..keyframes.len() {
        let (kf_left, kf_right) = keyframes.split_at_mut(kf_idx + 1);
        let kf = &mut kf_left[kf_idx];

        // Для каждого признака в кадре
        let mut local_points = Vec::new();
        let mut local_obs = Vec::new();

        for feat in &kf.features {
            if let Some(mp_id) = feat.map_point_id {
                // Ищем точку карты
                if let Some(mp) = map_points.iter().find(|mp| mp.id == mp_id) {
                    local_points.push(mp.position);
                    local_obs.push(feat.norm_coords);
                }
            }
        }

        if local_points.len() >= 3 {
            // Оптимизируем позу текущего кадра
            match optimize_pose_lm(&kf.pose, &local_points, &local_obs) {
                Ok(result) => {
                    kf.pose = result.pose;
                }
                Err(_) => {
                    // Если оптимизация не сошлась, оставляем текущую позу
                    log::warn!(
                        "Пучковая настройка: кадр {} не сошёлся, поза сохранена",
                        kf.id
                    );
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalized_vo::types::Pose;
    use nalgebra::{Matrix3, Vector3, Unit};

    /// Тест: оптимизация позы на синтетических данных (детерминированный)
    #[test]
    fn test_pose_optimization() {
        // Истинная поза
        let true_angle = 3.0f64.to_radians();
        let true_R = Matrix3::new(
            true_angle.cos(), 0.0, true_angle.sin(),
            0.0, 1.0, 0.0,
            -true_angle.sin(), 0.0, true_angle.cos(),
        );
        let true_t = Unit::new_normalize(Vector3::new(0.1, -0.02, 0.01));
        let true_pose = Pose::new(true_R, true_t.into_inner());

        // Генерируем 3D точки по сетке (детерминированно)
        let mut world_points = Vec::new();
        let mut observations = Vec::new();

        for i in 0..7 {
            for j in 0..7 {
                let pt = NormPoint3::new(
                    -0.6 + i as f64 * 0.2,
                    -0.6 + j as f64 * 0.2,
                    2.0 + (i + j) as f64 * 0.5,
                );
                if let Some(obs) = true_pose.project(&pt) {
                    world_points.push(pt);
                    observations.push(obs);
                }
            }
        }

        // Начальное приближение: Identity
        let initial_pose = Pose::identity();

        let result = optimize_pose_lm(&initial_pose, &world_points, &observations)
            .expect("Оптимизация должна сойтись");

        assert!(result.converged, "Оптимизация должна сойтись");

        // Проверяем, что финальная ошибка мала
        assert!(
            result.final_error < 1e-6,
            "Финальная ошибка слишком велика: {}",
            result.final_error
        );
    }
}
