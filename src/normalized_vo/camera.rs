// ============================================================================
// camera.rs — Нормированные координаты камеры и коррекция дисторсии
// ============================================================================
// Все пиксельные координаты преобразуются в нормированные координаты
// (x_norm, y_norm) с эффективным фокусным расстоянием f = 1.
// Радиальная дисторсия корректируется по модели Брауна-Конради:
//   x_corrected = x_distorted * (1 + k1*r^2 + k2*r^4)
// ============================================================================

use nalgebra::Vector2;
use crate::normalized_vo::types::{CameraCalibration, NormCoords, VoError};

/// Преобразует пиксельные координаты (u, v) в нормированные (x_norm, y_norm).
///
/// # Математика
/// x_norm = (u - cx) / fx
/// y_norm = (v - cy) / fy
///
/// Затем применяется коррекция радиальной дисторсии:
/// r^2 = x_norm^2 + y_norm^2
/// k = 1 + k1 * r^2 + k2 * r^4
/// x_undist = x_norm * k
/// y_undist = y_norm * k
pub fn pixel_to_normalized(u: f64, v: f64, calib: &CameraCalibration) -> Result<NormCoords, VoError> {
    // Шаг 1: переход от пиксельных к нормированным координатам
    let x_norm = (u - calib.cx) / calib.fx;
    let y_norm = (v - calib.cy) / calib.fy;

    // Шаг 2: коррекция радиальной дисторсии
    let r2 = x_norm * x_norm + y_norm * y_norm;
    let k = 1.0 + calib.k1 * r2 + calib.k2 * r2 * r2;

    let x_undist = x_norm * k;
    let y_undist = y_norm * k;

    // Проверка антигаллюцинации: дисторсия не должна выбрасывать за пределы
    // разумного диапазона нормированных координат.
    // Типичные значения нормированных координат: ~[-2, 2] для обычных камер.
    if x_undist.abs() > 10.0 || y_undist.abs() > 10.0 {
        return Err(VoError::UndistortionOutOfBounds);
    }

    Ok(NormCoords::new(x_undist, y_undist))
}

/// Обратное преобразование: из нормированных координат в пиксельные.
/// Используется для визуализации и извлечения патчей.
pub fn normalized_to_pixel(norm: &NormCoords, calib: &CameraCalibration) -> (f64, f64) {
    // Решаем обратную дисторсию итеративно (метод Ньютона),
    // так как прямое аналитическое решение отсутствует для k1,k2 ≠ 0.
    let x_norm = norm.x;
    let y_norm = norm.y;

    // Начальное приближение: x_norm, y_norm (без дисторсии)
    let mut x = x_norm;
    let mut y = y_norm;

    for _ in 0..5 {
        let r2 = x * x + y * y;
        let k = 1.0 + calib.k1 * r2 + calib.k2 * r2 * r2;
        // Обратная функция: x_norm = x / k, y_norm = y / k
        // Ошибка: err_x = x * k - x_norm, err_y = y * k - y_norm
        let err_x = x * k - x_norm;
        let err_y = y * k - y_norm;

        // Якобиан d(err)/d(x,y) для метода Ньютона
        let dk_dx = 2.0 * x * (calib.k1 + 2.0 * calib.k2 * r2);
        let dk_dy = 2.0 * y * (calib.k1 + 2.0 * calib.k2 * r2);
        let j11 = k + x * dk_dx;
        let j12 = x * dk_dy;
        let j21 = y * dk_dx;
        let j22 = k + y * dk_dy;

        let det = j11 * j22 - j12 * j21;
        if det.abs() < 1e-12 {
            break;
        }

        let dx = (-err_x * j22 + err_y * j12) / det;
        let dy = (-err_y * j11 + err_x * j21) / det;

        x += dx;
        y += dy;

        if dx * dx + dy * dy < 1e-12 {
            break;
        }
    }

    let u = x * calib.fx + calib.cx;
    let v = y * calib.fy + calib.cy;

    (u, v)
}

/// Проверяет, находятся ли нормированные координаты в пределах изображения.
pub fn is_in_image(norm: &NormCoords, calib: &CameraCalibration) -> bool {
    let (u, v) = normalized_to_pixel(norm, calib);
    u >= 0.0 && u < calib.width as f64 &&
    v >= 0.0 && v < calib.height as f64
}

/// Преобразует массив пиксельных координат в нормированные.
pub fn pixels_to_normalized(
    pixels: &[(f64, f64)],
    calib: &CameraCalibration,
) -> Result<Vec<NormCoords>, VoError> {
    pixels
        .iter()
        .map(|&(u, v)| pixel_to_normalized(u, v, calib))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_calib() -> CameraCalibration {
        CameraCalibration {
            fx: 500.0, fy: 500.0, cx: 320.0, cy: 240.0,
            k1: -0.1, k2: 0.01,
            width: 640, height: 480,
        }
    }

    /// Тест 1: Проверка, что центр изображения отображается в (0, 0)
    #[test]
    fn test_center_maps_to_zero() {
        let calib = test_calib();
        let norm = pixel_to_normalized(320.0, 240.0, &calib).unwrap();
        assert!((norm.x).abs() < 1e-9, "Центр должен быть (0,0), получен ({}, {})", norm.x, norm.y);
        assert!((norm.y).abs() < 1e-9);
    }

    /// Тест 2: Проверка прямого и обратного преобразования (цикл)
    #[test]
    fn test_roundtrip() {
        let calib = test_calib();
        let test_points = [(320.0, 240.0), (100.0, 50.0), (639.0, 479.0)];
        for &(u, v) in &test_points {
            let norm = pixel_to_normalized(u, v, &calib).unwrap();
            let (u2, v2) = normalized_to_pixel(&norm, &calib);
            assert!(
                (u - u2).abs() < 1.0 && (v - v2).abs() < 1.0,
                "Цикл не сошёлся: ({}, {}) -> ({}, {})",
                u, v, u2, v2,
            );
        }
    }

    /// Тест 3: Проверка, что дисторсия с k1=0, k2=0 даёт тождественность
    #[test]
    fn test_no_distortion() {
        let calib = CameraCalibration {
            fx: 500.0, fy: 500.0, cx: 320.0, cy: 240.0,
            k1: 0.0, k2: 0.0,
            width: 640, height: 480,
        };
        let norm = pixel_to_normalized(400.0, 300.0, &calib).unwrap();
        assert!((norm.x - (400.0 - 320.0) / 500.0).abs() < 1e-12);
        assert!((norm.y - (300.0 - 240.0) / 500.0).abs() < 1e-12);
    }
}
