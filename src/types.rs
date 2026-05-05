use serde::Deserialize;

pub struct Point {
    pub x: f64,
    pub y: f64,
    pub z: f64
}
pub struct Rotation {
    pub yaw: f64,
    pub pitch: f64,
    pub roll: f64
}
pub struct Frame {
    pub id: u32,
    pub data: opencv::core::Mat,
    pub pos: Point,
    pub rot: Rotation,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CsvRow {
    pub translation_x: f64,
    pub translation_y: f64,
    pub translation_z: f64,
    pub frame_num: u32,
}
