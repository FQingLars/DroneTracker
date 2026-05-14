// ============================================================================
// types.rs — Типы данных для нормированной визуальной одометрии
// ============================================================================
// Все трёхмерные координаты хранятся в нормированном пространстве камеры,
// где эффективное фокусное расстояние равно единице. Векторы перемещения
// всегда имеют единичную длину. Метрический масштаб отсутствует.
// ============================================================================

use nalgebra::{Matrix3, Vector2, Vector3, SVector, SMatrix, Unit};

/// Нормированные координаты точки на изображении.
/// После коррекции дисторсии: (x_norm, y_norm) где f = 1.
pub type NormCoords = Vector2<f64>;

/// 3D точка в нормированном пространстве камеры.
/// Координаты X, Y, Z — безразмерные величины.
pub type NormPoint3 = Vector3<f64>;

/// Матрица поворота 3×3, принадлежащая группе SO(3).
pub type Rotation = Matrix3<f64>;

/// Вектор перемещения единичной длины (||t|| = 1).
pub type UnitTranslation = Unit<Vector3<f64>>;

/// Поза камеры: поворот R ∈ SO(3) и единичное перемещение t.
#[derive(Clone, Debug)]
pub struct Pose {
    pub rotation: Rotation,
    pub translation: UnitTranslation,
}

impl Pose {
    /// Создаёт новую позу. translation автоматически нормируется.
    pub fn new(rotation: Rotation, translation: Vector3<f64>) -> Self {
        let norm_t = Unit::try_new(translation, 1e-12)
            .expect("ОШИБКА [антигаллюцинация]: вектор перемещения нулевой длины");
        // Проверка антигаллюцинации: детерминант матрицы поворота должен быть близок к 1
        // Допуск 1e-3, как указано в правилах
        assert!(
            (rotation.determinant() - 1.0).abs() < 1e-3,
            "Матрица поворота не в SO(3), det = {}",
            rotation.determinant()
        );
        Pose { rotation, translation: norm_t }
    }

    /// Проецирует 3D точку из мирового пространства в нормированные координаты.
    /// Возвращает None, если точка за камерой (Z ≤ 0).
    pub fn project(&self, point: &NormPoint3) -> Option<NormCoords> {
        let p = self.rotation * point + self.translation.into_inner();
        if p.z <= 1e-12 {
            return None; // Точка за камерой или слишком близко
        }
        Some(NormCoords::new(p.x / p.z, p.y / p.z))
    }
}

/// Калибровка камеры (матрица внутренних параметров + дисторсия).
#[derive(Clone, Debug)]
pub struct CameraCalibration {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub k1: f64,
    pub k2: f64,
    pub width: u32,
    pub height: u32,
}

/// Признаковая точка на изображении в нормированных координатах.
#[derive(Clone, Debug)]
pub struct Feature {
    pub id: u64,
    /// Нормированные координаты (x_norm, y_norm)
    pub norm_coords: NormCoords,
    /// Отклик детектора (сила угла)
    pub response: f64,
    /// Сколько кадров живёт трек
    pub age: u32,
    /// Ссылка на точку карты, если ассоциирована
    pub map_point_id: Option<u64>,
}

/// Точка карты — 3D точка в нормированном пространстве.
#[derive(Clone, Debug)]
pub struct MapPoint {
    pub id: u64,
    /// Позиция в нормированных координатах (X, Y, Z) — безразмерные единицы
    pub position: NormPoint3,
    /// Список ID ключевых кадров, в которых наблюдается точка
    pub observations: Vec<u64>,
}

/// Ключевой кадр.
#[derive(Clone, Debug)]
pub struct KeyFrame {
    pub id: u64,
    pub pose: Pose,
    pub features: Vec<Feature>,
    pub timestamp: f64,
}

/// Обобщённый вектор состояния оптимизации позы (6 DOF).
/// [δθ_x, δθ_y, δθ_z, δt_x, δt_y, δt_z]
/// δθ — возмущение угла (3 DOF), δt — возмущение перемещения.
pub type PoseUpdate = SVector<f64, 6>;

/// Результат оценки движения между двумя кадрами.
#[derive(Clone, Debug)]
pub struct MotionEstimate {
    pub pose: Pose,
    pub inliers: Vec<(NormCoords, NormCoords)>,
    pub num_inliers: usize,
    pub reprojection_error: f64,
}

/// Ошибки, определённые правилами антигаллюцинаций.
#[derive(Debug)]
pub enum VoError {
    /// Кадры с нулевым или почти нулевым движением (вырожденная E)
    DegenerateMotion,
    /// Недостаточно сопоставленных точек: менее 8
    InsufficientMatches(usize),
    /// Детерминант R отклонился от 1 более чем на 1e-3
    RotationNotInSO3(f64),
    /// Вектор перемещения после нормализации имеет нулевую длину
    ZeroTranslation,
    /// Радиальная дисторсия привела к координатам за пределами изображения
    UndistortionOutOfBounds,
    /// Алгоритм оптимизации не сошёлся
    OptimizationFailed,
}

/// Отрезок изображения (патч) для фотометрической ошибки.
pub type ImagePatch = [[f64; 8]; 8];

/// Изображение в градациях серого (f64 для точности при вычислении градиентов).
#[derive(Clone, Debug)]
pub struct GrayImage {
    pub data: Vec<f64>,
    pub width: usize,
    pub height: usize,
}

impl GrayImage {
    pub fn new(width: usize, height: usize) -> Self {
        GrayImage {
            data: vec![0.0; width * height],
            width,
            height,
        }
    }

    /// Получить значение пикселя с билинейной интерполяцией.
    pub fn interpolate(&self, x: f64, y: f64) -> Option<f64> {
        if x < 0.0 || x >= (self.width - 1) as f64 ||
           y < 0.0 || y >= (self.height - 1) as f64 {
            return None;
        }
        let ix = x as usize;
        let iy = y as usize;
        let dx = x - ix as f64;
        let dy = y - iy as f64;
        let p00 = self.data[iy * self.width + ix];
        let p10 = self.data[iy * self.width + (ix + 1)];
        let p01 = self.data[(iy + 1) * self.width + ix];
        let p11 = self.data[(iy + 1) * self.width + (ix + 1)];
        Some(
            p00 * (1.0 - dx) * (1.0 - dy) +
            p10 * dx * (1.0 - dy) +
            p01 * (1.0 - dx) * dy +
            p11 * dx * dy,
        )
    }
}
