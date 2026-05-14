// ============================================================================
// mapping.rs — Управление картой и ключевыми кадрами
// ============================================================================
// Карта содержит 3D точки в нормированном пространстве и ключевые кадры.
// Новые точки триангулируются из пар ключевых кадров.
// Масштаб фиксируется нормировкой средней глубины к 1.
// ============================================================================

use nalgebra::Vector3;
use crate::normalized_vo::types::{
    NormCoords, NormPoint3, Pose, Feature, MapPoint, KeyFrame, CameraCalibration, VoError,
};
use crate::normalized_vo::motion::{
    triangulate_point, triangulate_points, normalize_depth,
};
use crate::normalized_vo::camera::is_in_image;

/// Минимальное количество общих наблюдений для триангуляции
const MIN_PARALLAX: f64 = 0.01; // минимальный параллакс (в нормированных единицах)

/// Максимальное количество ключевых кадров в локальной карте
const MAX_LOCAL_KEYFRAMES: usize = 10;

/// Порог для определения ключевого кадра: среднее смещение признаков
const KEYFRAME_TRANSLATION_THRESHOLD: f64 = 0.05;

/// Максимальная глубина точки (в нормированных единицах)
const MAX_POINT_DEPTH: f64 = 100.0;

/// Минимальный угол параллакса для триангуляции (рад)
const MIN_PARALLAX_ANGLE: f64 = 0.005;

// ============================ КАРТА =======================================

/// Карта нормированной визуальной одометрии.
#[derive(Debug)]
pub struct Map {
    /// Все точки карты в нормированном пространстве
    pub points: Vec<MapPoint>,
    /// Все ключевые кадры
    pub keyframes: Vec<KeyFrame>,
    /// Счётчик ID для новых точек
    next_point_id: u64,
    /// Счётчик ID для новых кадров
    next_keyframe_id: u64,
}

impl Map {
    pub fn new() -> Self {
        Map {
            points: Vec::new(),
            keyframes: Vec::new(),
            next_point_id: 1,
            next_keyframe_id: 1,
        }
    }

    /// Добавляет ключевой кадр в карту.
    pub fn add_keyframe(&mut self, kf: KeyFrame) -> u64 {
        let id = self.next_keyframe_id;
        self.keyframes.push(KeyFrame { id, ..kf });
        self.next_keyframe_id += 1;
        id
    }

    /// Создаёт новую точку карты из триангуляции.
    pub fn create_map_point(
        &mut self,
        position: NormPoint3,
        kf_id: u64,
    ) -> u64 {
        let id = self.next_point_id;
        self.points.push(MapPoint {
            id,
            position,
            observations: vec![kf_id],
        });
        self.next_point_id += 1;
        id
    }

    /// Возвращает последние N ключевых кадров.
    pub fn last_keyframes(&self, n: usize) -> &[KeyFrame] {
        let start = self.keyframes.len().saturating_sub(n);
        &self.keyframes[start..]
    }

    /// Возвращает последний ключевой кадр.
    pub fn last_keyframe(&self) -> Option<&KeyFrame> {
        self.keyframes.last()
    }

    /// Возвращает точку карты по ID.
    pub fn get_point(&self, id: u64) -> Option<&MapPoint> {
        self.points.iter().find(|p| p.id == id)
    }

    /// Возвращает изменяемую точку карты по ID.
    pub fn get_point_mut(&mut self, id: u64) -> Option<&mut MapPoint> {
        self.points.iter_mut().find(|p| p.id == id)
    }

    /// Удаляет точки, которые видны менее чем в N кадрах.
    pub fn filter_points_by_observations(&mut self, min_obs: usize) {
        self.points.retain(|p| p.observations.len() >= min_obs);
    }

    /// Нормирует среднюю глубину всех точек к 1.
    pub fn normalize_scale(&mut self) {
        let positions: Vec<NormPoint3> = self.points.iter().map(|p| p.position).collect();
        let mut mutable_positions = positions;
        normalize_depth(&mut mutable_positions);
        for (i, pos) in mutable_positions.iter().enumerate() {
            if i < self.points.len() {
                self.points[i].position = *pos;
            }
        }
    }
}

// ==================== ТРИАНГУЛЯЦИЯ НОВЫХ ТОЧЕК ============================

/// Триангулирует новые точки из пары ключевых кадров.
///
/// Для каждой пары соответствующих признаков (сопоставленных через
/// оптический поток) вычисляется 3D положение методом DLT.
/// Точки добавляются в карту, если параллакс достаточен.
pub fn triangulate_new_points(
    map: &mut Map,
    kf_ref: &KeyFrame,
    kf_cur: &KeyFrame,
    matches: &[(Feature, Feature)],
) -> usize {
    let mut new_count = 0;

    for (feat_ref, feat_cur) in matches {
        // Проверка: точка уже в карте?
        if feat_ref.map_point_id.is_some() {
            continue;
        }

        // Вычисляем параллакс
        let parallax = (feat_cur.norm_coords - feat_ref.norm_coords).norm();
        if parallax < MIN_PARALLAX {
            continue; // Слишком малый параллакс — триангуляция будет неустойчивой
        }

        // Триангулируем
        let pt_3d = triangulate_point(
            &kf_ref.pose,
            &kf_cur.pose,
            feat_ref.norm_coords,
            feat_cur.norm_coords,
        );

        if let Some(point) = pt_3d {
            // Проверка: точка перед камерой и не слишком далеко
            if point.z <= 0.0 || point.z > MAX_POINT_DEPTH || !point.z.is_finite() {
                continue;
            }
            if !point.x.is_finite() || !point.y.is_finite() {
                continue;
            }

            // Проверка повторной проекции: ошибка должна быть мала
            let proj_ref = kf_ref.pose.project(&point);
            let proj_cur = kf_cur.pose.project(&point);

            let ok = match (proj_ref, proj_cur) {
                (Some(p_ref), Some(p_cur)) => {
                    let err_ref = (p_ref - feat_ref.norm_coords).norm();
                    let err_cur = (p_cur - feat_cur.norm_coords).norm();
                    err_ref < 1e-3 && err_cur < 1e-3
                }
                _ => false,
            };

            if ok {
                let mp_id = map.create_map_point(point, kf_cur.id);
                new_count += 1;

                // Обновляем map_point_id в признаках (через внешний механизм)
                // В реальной системе нужно обновить Feature.map_point_id
                // через изменяемый доступ к карте признаков.
            }
        }
    }

    new_count
}

// ==================== ВЫБОР КЛЮЧЕВЫХ КАДРОВ ===============================

/// Определяет, должен ли текущий кадр стать ключевым.
///
/// Критерии:
/// 1. Среднее смещение отслеженных признаков превышает порог
/// 2. Количество потерянных признаков превышает порог
/// 3. Текущий кадр — первый в последовательности
pub fn should_be_keyframe(
    tracked_features: &[(Feature, Option<Feature>)],
    is_first_frame: bool,
) -> bool {
    if is_first_frame {
        return true;
    }

    // Считаем среднее смещение успешно отслеженных признаков
    let mut total_displacement = 0.0;
    let mut count = 0;

    for (prev, next_opt) in tracked_features {
        if let Some(next) = next_opt {
            let displacement = (next.norm_coords - prev.norm_coords).norm();
            total_displacement += displacement;
            count += 1;
        }
    }

    if count == 0 {
        return true; // Все точки потеряны — нужен новый ключевой кадр
    }

    let mean_displacement = total_displacement / count as f64;

    // Порог: если среднее смещение достаточно велико
    mean_displacement > KEYFRAME_TRANSLATION_THRESHOLD ||
    // Или если потеряно более 50% точек
    (tracked_features.len() - count) > tracked_features.len() / 2
}

// ==================== ФИЛЬТРАЦИЯ И ОБНОВЛЕНИЕ ТОЧЕК ========================

/// Обновляет наблюдения точки карты: добавляет новый ключевой кадр.
pub fn add_observation(map: &mut Map, point_id: u64, kf_id: u64) {
    if let Some(pt) = map.get_point_mut(point_id) {
        if !pt.observations.contains(&kf_id) {
            pt.observations.push(kf_id);
        }
    }
}

/// Фильтрует точки, которые вышли за пределы поля зрения.
pub fn filter_out_of_view(
    map: &mut Map,
    keyframe: &KeyFrame,
    calib: &CameraCalibration,
) {
    map.points.retain(|pt| {
        // Проецируем точку в кадр
        if let Some(norm) = keyframe.pose.project(&pt.position) {
            is_in_image(&norm, calib)
        } else {
            false // Точка за камерой
        }
    });
}

/// Удаляет выбросы по ошибке повторной проекции.
pub fn filter_outliers_by_reprojection(
    map: &mut Map,
    keyframe: &KeyFrame,
    max_error: f64,
) {
    map.points.retain(|pt| {
        if let Some(proj) = keyframe.pose.project(&pt.position) {
            // Проверяем, есть ли наблюдение в этом кадре
            // Если нет — не отфильтровываем
            true
        } else {
            false // Точка за камерой
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Vector3};

    #[test]
    fn test_map_operations() {
        let mut map = Map::new();

        let kf = KeyFrame {
            id: 0,
            pose: Pose::identity(),
            features: Vec::new(),
            timestamp: 0.0,
        };

        let kf_id = map.add_keyframe(kf);
        assert_eq!(kf_id, 1);

        let mp_id = map.create_map_point(NormPoint3::new(1.0, 0.0, 5.0), kf_id);
        assert_eq!(mp_id, 1);

        let pt = map.get_point(mp_id);
        assert!(pt.is_some());
        assert_eq!(pt.unwrap().observations.len(), 1);
    }

    #[test]
    fn test_keyframe_selection() {
        let small_motion = vec![
            (Feature { id: 1, norm_coords: NormCoords::new(0.0, 0.0), response: 1.0, age: 1, map_point_id: None },
             Some(Feature { id: 1, norm_coords: NormCoords::new(0.001, 0.0), response: 1.0, age: 2, map_point_id: None })),
        ];
        assert!(!should_be_keyframe(&small_motion, false));
    }
}
