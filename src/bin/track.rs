//! Drone trajectory tracking binary.
//!
//! Reads video frames, performs GMC tracking with height estimation,
//! compares with ground truth, computes metrics, and outputs results.

use anyhow::{anyhow, Result};
use csv::{ReaderBuilder, WriterBuilder};
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, info, warn};
use opencv::core::Mat;
use opencv::videoio::{self, VideoCapture, VideoCaptureTrait};
use opencv::prelude::{VideoCaptureTraitConst, MatTraitConst};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;
use tracker::gmc::GmcTracker;
use tracker::height::{HeightEstimator, UndistortionCache};
use tracker::types::{
    CameraIntrinsics, GroundTruthRecord, TrackingMetrics, TrajectoryRecord,
};

/// Frame skip interval for stability (process every N-th frame)
const FRAME_SKIP: usize = 2;

/// Number of frames between CSV flushes
const FLUSH_INTERVAL: usize = 100;

/// Ground truth CSV file name (relative to working directory)
const GROUND_TRUTH_CSV: &str = "src/trajectory_norm.csv";

/// Load ground truth trajectory from CSV file (no headers, format: x,y,z,frame_idx)
fn load_ground_truth(path: &str) -> Result<Vec<GroundTruthRecord>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)?;

    let mut records = Vec::new();
    for result in reader.records() {
        let record = result?;
        if record.len() >= 4 {
            let x: f64 = record[0].parse()?;
            let y: f64 = record[1].parse()?;
            let z: f64 = record[2].parse()?;
            let frame_idx: u32 = record[3].parse()?;
            records.push(GroundTruthRecord { x, y, z, frame_idx });
        }
    }
    info!("Loaded {} ground truth records from {}", records.len(), path);
    Ok(records)
}

/// Save comparison CSV with both estimated and ground truth values
fn save_comparison_csv(
    output_path: &str,
    estimated: &[TrajectoryRecord],
    ground_truth: &[GroundTruthRecord],
) -> Result<()> {
    let mut writer = WriterBuilder::new().from_path(output_path)?;
    writer.write_record([
        "frame_idx", "est_x", "est_y", "est_z",
        "gt_x", "gt_y", "gt_z", "scale", "inliers",
    ])?;

    let gt_map: HashMap<usize, &GroundTruthRecord> = ground_truth
        .iter()
        .map(|r| (r.frame_idx as usize, r))
        .collect();

    for est in estimated {
        if let Some(&gt) = gt_map.get(&est.frame_idx) {
            writer.serialize((
                est.frame_idx,
                est.x, est.y, est.z,
                gt.x, gt.y, gt.z,
                est.scale, est.inliers,
            ))?;
        }
    }
    writer.flush()?;
    info!("Saved comparison CSV to {}", output_path);
    Ok(())
}

/// Generate trajectory comparison plots
fn generate_plots(
    estimated: &[TrajectoryRecord],
    ground_truth: &[GroundTruthRecord],
    output_prefix: &str,
) -> Result<()> {
    let gt_map: HashMap<usize, &GroundTruthRecord> = ground_truth
        .iter()
        .map(|r| (r.frame_idx as usize, r))
        .collect();
    
    // Prepare data for plotting
    let mut est_x = Vec::new();
    let mut est_y = Vec::new();
    let mut est_z = Vec::new();
    let mut gt_x = Vec::new();
    let mut gt_y = Vec::new();
    let mut gt_z = Vec::new();
    let mut frames = Vec::new();
    
    for est in estimated {
        if let Some(&gt) = gt_map.get(&est.frame_idx) {
            frames.push(est.frame_idx);
            est_x.push(est.x);
            est_y.push(est.y);
            est_z.push(est.z);
            gt_x.push(gt.x);
            gt_y.push(gt.y);
            gt_z.push(gt.z);
        }
    }
    
    if frames.is_empty() {
        warn!("No matching frames for plotting");
        return Ok(());
    }
    
    // Plot X trajectory
    plot_1d_trajectory(
        &format!("{}_x.png", output_prefix),
        "X Trajectory Comparison",
        &frames,
        &est_x,
        &gt_x,
        "Frame",
        "X position",
        "Estimated".to_string(),
        "Ground Truth".to_string(),
    )?;
    
    // Plot Y trajectory
    plot_1d_trajectory(
        &format!("{}_y.png", output_prefix),
        "Y Trajectory Comparison",
        &frames,
        &est_y,
        &gt_y,
        "Frame",
        "Y position",
        "Estimated".to_string(),
        "Ground Truth".to_string(),
    )?;
    
    // Plot Z trajectory
    plot_1d_trajectory(
        &format!("{}_z.png", output_prefix),
        "Z Trajectory Comparison",
        &frames,
        &est_z,
        &gt_z,
        "Frame",
        "Z position",
        "Estimated".to_string(),
        "Ground Truth".to_string(),
    )?;
    
    // Plot 2D XY trajectory
    plot_2d_trajectory(
        &format!("{}_xy.png", output_prefix),
        "XY Trajectory Comparison",
        &est_x, &est_y,
        &gt_x, &gt_y,
        "X position",
        "Y position",
    )?;
    
    info!("Generated plots with prefix: {}", output_prefix);
    Ok(())
}

/// Plot 1D trajectory comparison
fn plot_1d_trajectory(
    filename: &str,
    title: &str,
    x_data: &[usize],
    y_est: &[f64],
    y_gt: &[f64],
    x_label: &str,
    y_label: &str,
    legend_est: String,
    legend_gt: String,
) -> Result<()> {
    use plotters::prelude::*;
    
    let root = BitMapBackend::new(filename, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let x_min = *x_data.first().unwrap_or(&0) as f64;
    let x_max = *x_data.last().unwrap_or(&1) as f64;
    let y_all: Vec<f64> = y_est.iter().chain(y_gt.iter()).copied().collect();
    let y_min = y_all.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = y_all.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_margin = (y_max - y_min) * 0.1;
    
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(
            x_min..x_max,
            (y_min - y_margin)..(y_max + y_margin),
        )?;
    
    chart.configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .draw()?;
    
    // Ground truth line
    chart
        .draw_series(LineSeries::new(
            x_data.iter().zip(y_gt.iter()).map(|(x, y)| (*x as f64, *y)),
            RED.stroke_width(2),
        ))?
        .label(legend_gt)
        .legend(|(x, y)| PathElement::new(vec![(x, y)], RED.stroke_width(2)));
    
    // Estimated line
    chart
        .draw_series(LineSeries::new(
            x_data.iter().zip(y_est.iter()).map(|(x, y)| (*x as f64, *y)),
            BLUE.stroke_width(2),
        ))?
        .label(legend_est)
        .legend(|(x, y)| PathElement::new(vec![(x, y)], BLUE.stroke_width(2)));
    
    // Add legend
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    
    root.present()?;
    Ok(())
}

/// Plot 2D XY trajectory comparison
fn plot_2d_trajectory(
    filename: &str,
    title: &str,
    x_est: &[f64],
    y_est: &[f64],
    x_gt: &[f64],
    y_gt: &[f64],
    x_label: &str,
    y_label: &str,
) -> Result<()> {
    use plotters::prelude::*;
    
    let root = BitMapBackend::new(filename, (1000, 1000)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let x_all: Vec<f64> = x_est.iter().chain(x_gt.iter()).copied().collect();
    let y_all: Vec<f64> = y_est.iter().chain(y_gt.iter()).copied().collect();
    let x_min = x_all.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let x_max = x_all.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_min = y_all.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = y_all.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let margin_x = (x_max - x_min) * 0.1;
    let margin_y = (y_max - y_min) * 0.1;
    
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(
            (x_min - margin_x)..(x_max + margin_x),
            (y_min - margin_y)..(y_max + margin_y),
        )?;
    
    chart.configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .draw()?;
    
    // Ground truth trajectory
    chart
        .draw_series(LineSeries::new(
            x_gt.iter().zip(y_gt.iter()).map(|(x, y)| (*x, *y)),
            RED.stroke_width(2),
        ))?
        .label("Ground Truth")
        .legend(|(x, y)| PathElement::new(vec![(x, y)], RED.stroke_width(2)));
    
    // Estimated trajectory
    chart
        .draw_series(LineSeries::new(
            x_est.iter().zip(y_est.iter()).map(|(x, y)| (*x, *y)),
            BLUE.stroke_width(2),
        ))?
        .label("Estimated")
        .legend(|(x, y)| PathElement::new(vec![(x, y)], BLUE.stroke_width(2)));
    
    // Add legend
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    
    root.present()?;
    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();
    info!("Drone trajectory tracker starting...");

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <video_path> <output_csv> [ground_truth_csv]", args[0]);
        return Err(anyhow!("Missing arguments"));
    }

    let video_path = &args[1];
    let output_csv = &args[2];
    let ground_truth_path = args.get(3).map(|s| s.as_str()).unwrap_or(GROUND_TRUTH_CSV);

    if !Path::new(video_path).exists() {
        return Err(anyhow!("Video file not found: {}", video_path));
    }

    info!("Video: {}, Output: {}", video_path, output_csv);
    info!("Ground truth: {}", ground_truth_path);

    // Load ground truth if available
    let ground_truth = if Path::new(ground_truth_path).exists() {
        load_ground_truth(ground_truth_path)?
    } else {
        warn!("Ground truth file not found: {}", ground_truth_path);
        Vec::new()
    };

    let mut cap = VideoCapture::from_file(video_path, videoio::CAP_ANY)?;
    if !cap.is_opened()? {
        return Err(anyhow!("Failed to open video: {}", video_path));
    }

    let total_frames = cap.get(videoio::CAP_PROP_FRAME_COUNT)? as usize;
    let fps = cap.get(videoio::CAP_PROP_FPS)?;
    info!(
        "Video properties: {} frames, {:.2} FPS",
        total_frames, fps
    );

    let mut gmc_tracker = GmcTracker::new()?;
    let mut height_estimator = HeightEstimator::new();
    let intrinsics = CameraIntrinsics::default();

    let image_size = (
        cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32,
        cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32,
    );
    info!("Image size: {}x{}", image_size.0, image_size.1);

    let undistort_cache = UndistortionCache::new(&intrinsics, image_size)?;
    info!("Undistortion cache initialized");

    let mut csv_writer = WriterBuilder::new().from_path(output_csv)?;
    csv_writer.write_record([
        "frame_idx", "x", "y", "z", "scale", "inliers", "dt_sec",
    ])?;

    let mut frame_idx = 0usize;
    let mut processed_idx = 0usize;

    let mut x = 0.0f64;
    let mut y = 0.0f64;
    let mut z = 0.0f64;
    
    // Store all trajectory records for later comparison
    let mut all_records: Vec<TrajectoryRecord> = Vec::new();

    let pb = ProgressBar::new(total_frames as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )?
            .progress_chars("#>-"),
    );

    let mut frame_buffer = Mat::default();
    let mut undistorted = Mat::default();

    while cap.read(&mut frame_buffer)? {
        pb.set_position(frame_idx as u64);

        if !frame_idx.is_multiple_of(FRAME_SKIP) {
            frame_idx += 1;
            continue;
        }

        let start_time = Instant::now();

        if frame_buffer.empty() {
            warn!("Empty frame at index {}", frame_idx);
            frame_idx += 1;
            continue;
        }

        undistort_cache.remap(&frame_buffer, &mut undistorted)?;

        let (frame_pair_opt, _compensated) = gmc_tracker.process(&undistorted)?;

        if let Some(frame_pair) = frame_pair_opt {
            let (dx_raw, dy_raw) = gmc_tracker.get_raw_motion(&frame_pair)?;

            let (dz, scale, inliers_count, is_unreliable) =
                height_estimator.estimate_height_change(&frame_pair)?;

            let (dx, dy) = gmc_tracker.correct_by_scale(dx_raw, dy_raw, scale)?;

            x += dx;
            y += dy;
            z += dz;
            let dt = start_time.elapsed().as_secs_f64();

            if is_unreliable {
                warn!(
                    "Frame {}: unreliable height estimate, using previous scale",
                    frame_idx
                );
            }

            debug!(
                "Frame {}: x={:.3}, y={:.3}, z={:.3}, scale={:.4}, inliers={}, dz={:.6}",
                frame_idx, x, y, z, scale, inliers_count, dz
            );

            info!(
                "Frame {}: (x={:.3}, y={:.3}, z={:.3}), scale={:.4}, inliers={}",
                frame_idx, x, y, z, scale, inliers_count
            );

            let record = TrajectoryRecord {
                frame_idx: processed_idx,
                x,
                y,
                z,
                scale,
                inliers: inliers_count,
                dt_sec: dt,
            };
            
            csv_writer.serialize(&record)?;
            all_records.push(record);

            processed_idx += 1;

            if processed_idx.is_multiple_of(FLUSH_INTERVAL) {
                csv_writer.flush()?;
                info!("Flushed CSV after {} frames", processed_idx);
            }
        } else {
            debug!("Frame {}: no valid frame pair, skipping", frame_idx);
        }

        frame_idx += 1;
    }

    pb.finish_with_message("Tracking complete");
    csv_writer.flush()?;

    info!("Tracking complete: {} frames processed", processed_idx);
    info!("Final position: x={:.3}, y={:.3}, z={:.3}", x, y, z);

    // Compute metrics and generate plots if ground truth is available
    if !ground_truth.is_empty() && !all_records.is_empty() {
        info!("Computing tracking metrics...");
        let metrics = TrackingMetrics::compute(&all_records, &ground_truth);
        info!("{}", metrics);
        
        // Save comparison CSV
        let comparison_csv = output_csv.replace(".csv", "_comparison.csv");
        save_comparison_csv(&comparison_csv, &all_records, &ground_truth)?;
        
        // Generate plots
        let plot_prefix = output_csv.replace(".csv", "");
        if let Err(e) = generate_plots(&all_records, &ground_truth, &plot_prefix) {
            warn!("Failed to generate plots: {}", e);
        }
    } else if ground_truth.is_empty() {
        info!("No ground truth available, skipping metrics and plots");
    }

    Ok(())
}
