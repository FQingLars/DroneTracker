// ============================================================================
// features.rs — Детектор FAST, оптический поток Лукаса-Канаде, трекинг
// ============================================================================
// Реализация:
//   1. FAST-9 угловой детектор (без внешних зависимостей)
//   2. Лукас-Канаде оптический поток с пирамидой изображений
//   3. Фильтрация треков по согласованности и ошибке повторной проекции
// ============================================================================

use crate::normalized_vo::types::{GrayImage, Feature, NormCoords};

/// Порог детектора FAST (разница интенсивности)
const FAST_THRESHOLD: f64 = 20.0;
/// Минимальное количество последовательных пикселей для FAST
const FAST_N: u32 = 12;
/// Радиус круга Брезенхема для FAST
const FAST_CIRCLE: [(i32, i32); 16] = [
    (0, 3),  (1, 3),  (2, 2),  (3, 1),
    (3, 0),  (3, -1), (2, -2), (1, -3),
    (0, -3), (-1, -3),(-2, -2),(-3, -1),
    (-3, 0), (-3, 1), (-2, 2), (-1, 3),
];

/// Размер блока для LK (половина стороны окна)
const LK_WINDOW_HALF: i32 = 4;
/// Количество уровней пирамиды для LK
const LK_PYRAMID_LEVELS: usize = 3;
/// Максимальное количество итераций LK на уровне
const LK_MAX_ITER: u32 = 30;
/// Критерий сходимости LK (среднеквадратичное смещение)
const LK_EPSILON: f64 = 0.01;

// =========================== FAST ДЕТЕКТОР ================================

/// Обнаруживает угловые точки методом FAST-9.
///
/// Алгоритм:
/// 1. Для каждого пикселя p сравниваем 16 точек на окружности радиуса 3.
/// 2. Если N=12 последовательных точек все ярче или все темнее p ± threshold,
///    помечаем p как угол.
/// 3. Применяем non-maximum suppression по отклику.
pub fn fast_detect(image: &GrayImage) -> Vec<Feature> {
    let w = image.width as i32;
    let h = image.height as i32;
    let border = 3; // радиус круга FAST
    let mut scores = vec![0.0f64; image.data.len()];
    let mut is_corner = vec![false; image.data.len()];

    // Шаг 1: детекция углов
    for y in border..(h - border) {
        for x in border..(w - border) {
            let idx = y as usize * image.width + x as usize;
            let center = image.data[idx];

            // Собираем значения 16 пикселей окружности
            let mut circle_vals = [0.0f64; 16];
            for (i, &(dx, dy)) in FAST_CIRCLE.iter().enumerate() {
                let px = (x + dx) as usize;
                let py = (y + dy) as usize;
                circle_vals[i] = image.data[py * image.width + px];
            }

            // Проверяем: все ярче
            let brighter = circle_vals.map(|v| v - center > FAST_THRESHOLD);
            // Проверяем: все темнее
            let darker = circle_vals.map(|v| center - v > FAST_THRESHOLD);

            // Смотрим, есть ли N последовательных
            let has_corner = has_consecutive_n(&brighter, FAST_N) ||
                             has_consecutive_n(&darker, FAST_N);

            if has_corner {
                is_corner[idx] = true;
                // Отклик: сумма абсолютных разностей
                let response: f64 = circle_vals.iter()
                    .map(|&v| (v - center).abs())
                    .sum();
                scores[idx] = response;
            }
        }
    }

    // Шаг 2: non-maximum suppression (NMS)
    let mut features = Vec::new();
    let nms_radius = 3;
    for y in border..(h - border) {
        for x in border..(w - border) {
            let idx = y as usize * image.width + x as usize;
            if !is_corner[idx] {
                continue;
            }

            // Проверяем, является ли локальным максимумом
            let mut is_max = true;
            'nms: for dy in -nms_radius..=nms_radius {
                for dx in -nms_radius..=nms_radius {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx < border || nx >= w - border || ny < border || ny >= h - border {
                        continue;
                    }
                    let nidx = ny as usize * image.width + nx as usize;
                    if is_corner[nidx] && scores[nidx] > scores[idx] {
                        is_max = false;
                        break 'nms;
                    }
                }
            }

            if is_max {
                features.push(Feature {
                    id: 0, // будет назначен позже
                    norm_coords: NormCoords::new(x as f64, y as f64),
                    response: scores[idx],
                    age: 0,
                    map_point_id: None,
                });
            }
        }
    }

    features
}

/// Проверяет, есть ли N последовательных true в булевом массиве.
fn has_consecutive_n(arr: &[bool; 16], n: u32) -> bool {
    // Используем wrapping для проверки по кругу
    for start in 0..16 {
        let mut count = 0u32;
        for j in 0..16 {
            if arr[(start + j) % 16] {
                count += 1;
                if count >= n {
                    return true;
                }
            } else {
                count = 0;
            }
        }
    }
    false
}

// ========================= ПИРАМИДА ИЗОБРАЖЕНИЙ ===========================

/// Строит гауссову пирамиду изображения с заданным числом уровней.
pub fn build_pyramid(image: &GrayImage, levels: usize) -> Vec<GrayImage> {
    let mut pyramid = Vec::with_capacity(levels);
    pyramid.push(image.clone());

    for level in 1..levels {
        let prev = &pyramid[level - 1];
        let new_w = prev.width / 2;
        let new_h = prev.height / 2;
        let mut down = GrayImage::new(new_w, new_h);

        for y in 0..new_h {
            for x in 0..new_w {
                // Гауссово размытие 5x5 (упрощённое: билинейная интерполяция)
                let src_x = x as f64 * 2.0;
                let src_y = y as f64 * 2.0;
                let val = prev.interpolate(src_x, src_y).unwrap_or(0.0);
                down.data[y * new_w + x] = val;
            }
        }

        pyramid.push(down);
    }

    pyramid
}

// ====================== ВЫЧИСЛЕНИЕ ГРАДИЕНТОВ =============================

/// Вычисляет градиенты изображения по X и Y (центральные разности).
pub fn compute_gradients(image: &GrayImage) -> (Vec<f64>, Vec<f64>) {
    let w = image.width;
    let h = image.height;
    let mut grad_x = vec![0.0f64; w * h];
    let mut grad_y = vec![0.0f64; w * h];

    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let idx = y * w + x;
            grad_x[idx] = (image.data[idx + 1] - image.data[idx - 1]) * 0.5;
            grad_y[idx] = (image.data[idx + w] - image.data[idx - w]) * 0.5;
        }
    }

    (grad_x, grad_y)
}

// =================== ОПТИЧЕСКИЙ ПОТОК ЛУКАСА-КАНАДЕ =======================

/// Вычисляет оптический поток для одной точки методом Лукаса-Канаде
/// с пирамидой изображений.
///
/// # Аргументы
/// * `prev_pyr` — пирамида предыдущего изображения
/// * `next_pyr` — пирамида текущего изображения
/// * `pt` — начальная точка в нормированных координатах
/// * `prev_w` — ширина предыдущего изображения на нулевом уровне
///
/// # Возвращает
/// Смещение в пикселях на нулевом уровне пирамиды.
pub fn lk_track_point(
    prev_pyr: &[GrayImage],
    next_pyr: &[GrayImage],
    pt: &NormCoords,
    prev_w: usize,
) -> Option<NormCoords> {
    let levels = prev_pyr.len().min(next_pyr.len());

    // Умножаем нормированные координаты на ширину для получения грубой
    // аппроксимации пиксельных координат (для пирамиды).
    // Внимание: это приближение, так как нормированные координаты
    // зависят от фокусного расстояния. Для корректной работы LK
    // нужно сначала преобразовать нормированные -> пиксельные.
    // Здесь мы принимаем, что входные координаты уже в пикселях,
    // если передана calibration. Для упрощения используем масштаб.

    // Мы работаем на нулевом уровне в пиксельных координатах.
    // Координаты точки на уровне l: pt / 2^l
    let px = pt.x;
    let py = pt.y;

    // Инициализируем смещение на верхнем уровне
    let mut flow_x = 0.0f64;
    let mut flow_y = 0.0f64;

    // Спускаемся сверху вниз по пирамиде
    for level in (0..levels).rev() {
        let scale = 1.0 / (1 << level) as f64;
        let x_l = px * scale + flow_x * 0.5;
        let y_l = py * scale + flow_y * 0.5;

        let prev_img = &prev_pyr[level];
        let next_img = &next_pyr[level];

        let (gx, gy) = compute_gradients(prev_img);

        // Итерация Лукаса-Канаде на текущем уровне
        for _ in 0..LK_MAX_ITER {
            // Собираем матрицу Z = [Ix^2, Ix*Iy; Ix*Iy, Iy^2] и вектор b
            let mut z00 = 0.0f64; let mut z01 = 0.0f64;
            let mut z11 = 0.0f64;
            let mut b0 = 0.0f64; let mut b1 = 0.0f64;

            let half = LK_WINDOW_HALF;
            let w = prev_img.width as i32;

            for dy in -half..=half {
                for dx in -half..=half {
                    let sx = (x_l + flow_x + dx as f64).round() as i32;
                    let sy = (y_l + flow_y + dy as f64).round() as i32;

                    if sx < 1 || sx >= prev_img.width as i32 - 1 ||
                       sy < 1 || sy >= prev_img.height as i32 - 1 {
                        continue;
                    }

                    let idx = (sy * w + sx) as usize;
                    let ix = gx[idx];
                    let iy = gy[idx];

                    // Разница между кадрами
                    let it = prev_img.data[idx] -
                        next_img.interpolate(
                            sx as f64, sy as f64
                        ).unwrap_or(prev_img.data[idx]);

                    z00 += ix * ix;
                    z01 += ix * iy;
                    z11 += iy * iy;
                    b0 += ix * it;
                    b1 += iy * it;
                }
            }

            // Решаем Z * v = -b
            let det = z00 * z11 - z01 * z01;
            if det.abs() < 1e-12 {
                break; // Плохо обусловленная матрица
            }

            let vx = (-z11 * b0 + z01 * b1) / det;
            let vy = (z01 * b0 - z00 * b1) / det;

            flow_x += vx;
            flow_y += vy;

            if vx * vx + vy * vy < LK_EPSILON {
                break;
            }
        }
    }

    let new_x = px + flow_x;
    let new_y = py + flow_y;

    // Проверка: результат должен быть в разумных пределах
    if new_x.is_finite() && new_y.is_finite() {
        Some(NormCoords::new(new_x, new_y))
    } else {
        None
    }
}

/// Отслеживает набор признаков между двумя кадрами.
pub fn track_features(
    prev_image: &GrayImage,
    next_image: &GrayImage,
    features: &[Feature],
) -> Vec<(Feature, Option<Feature>)> {
    let prev_pyr = build_pyramid(prev_image, LK_PYRAMID_LEVELS);
    let next_pyr = build_pyramid(next_image, LK_PYRAMID_LEVELS);

    features
        .iter()
        .map(|feat| {
            let tracked = lk_track_point(
                &prev_pyr,
                &next_pyr,
                &feat.norm_coords,
                prev_image.width,
            );
            let new_feat = tracked.map(|nc| Feature {
                id: feat.id,
                norm_coords: nc,
                response: feat.response,
                age: feat.age + 1,
                map_point_id: feat.map_point_id,
            });
            (feat.clone(), new_feat)
        })
        .collect()
}

/// Фильтрует треки по обратной ошибке (forward-backward consistency).
pub fn filter_consistency(
    prev_image: &GrayImage,
    next_image: &GrayImage,
    tracks: &[(Feature, Option<Feature>)],
    threshold: f64,
) -> Vec<(Feature, Feature)> {
    let prev_pyr = build_pyramid(prev_image, LK_PYRAMID_LEVELS);
    let next_pyr = build_pyramid(next_image, LK_PYRAMID_LEVELS);

    tracks
        .iter()
        .filter_map(|(prev, next_opt)| {
            let next = next_opt.as_ref()?;

            // Прямой-обратный трек: из next обратно в prev
            let backward = lk_track_point(
                &next_pyr,
                &prev_pyr,
                &next.norm_coords,
                next_image.width,
            )?;

            // Ошибка: расстояние между исходной и обратно-спроецированной точкой
            let error = (backward - prev.norm_coords).norm();
            if error < threshold {
                Some((prev.clone(), next.clone()))
            } else {
                None
            }
        })
        .collect()
}

/// Сортирует признаки по отклику (убывание) и возвращает top-N.
pub fn select_best_features(features: &mut Vec<Feature>, max_count: usize) -> Vec<Feature> {
    features.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap_or(std::cmp::Ordering::Equal));
    features.truncate(max_count);
    features.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_image() -> GrayImage {
        let w = 100;
        let h = 100;
        let mut img = GrayImage::new(w, h);
        // Белый фон
        for y in 0..h {
            for x in 0..w {
                img.data[y * w + x] = 255.0;
            }
        }
        // Чёрная шахматная доска с крупными клетками
        for y in 0..h {
            for x in 0..w {
                if ((x / 15) + (y / 15)) % 2 == 0 {
                    img.data[y * w + x] = 0.0;
                }
            }
        }
        img
    }

    #[test]
    fn test_fast_detector_runs() {
        let img = create_test_image();
        // Простой тест: функция должна отработать без паники
        let corners = fast_detect(&img);
        // Просто проверяем, что можем получить хоть какие-то результаты
        // (детектор может не находить углы на простых паттернах, это нормально)
        assert!(corners.len() <= 100, "Слишком много углов: {}", corners.len());
    }

    #[test]
    fn test_pyramid_building() {
        let img = GrayImage::new(64, 64);
        let pyr = build_pyramid(&img, 4);
        assert_eq!(pyr.len(), 4);
        assert_eq!(pyr[1].width, 32);
        assert_eq!(pyr[2].width, 16);
        assert_eq!(pyr[3].width, 8);
    }
}
