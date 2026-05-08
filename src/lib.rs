//! Tracker - Drone trajectory estimation using computer vision
//!
//! This library provides modules for:
//! - Camera calibration and undistortion
//! - Feature detection and matching
//! - Ground plane homography for height estimation
//! - Global motion compensation (GMC) tracking

pub mod types;
pub mod height;
pub mod gmc;
pub mod visualization;
// gui module temporarily disabled due to API changes
// pub mod gui;

// Re-export commonly used types
pub use types::{CameraIntrinsics, PoseDelta, FramePair, TrajectoryRecord, GroundTruthRecord, TrackingMetrics};
pub use height::{HeightEstimator, UndistortionCache, HomographyConfig};
pub use gmc::GmcTracker;
pub use visualization::generate_all_plots;
