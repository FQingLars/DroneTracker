//! Visualization module for trajectory plotting and metrics display
//!
//! Uses plotters to create trajectory comparison plots and error visualizations

use crate::types::{TrajectoryRecord, GroundTruthRecord, TrackingMetrics};
use plotters::prelude::*;
use std::collections::HashMap;

/// Plot estimated vs ground truth trajectory in 2D projections
///
/// Creates a PNG file with trajectory plots (XY, XZ, YZ projections)
///
/// # Arguments
/// * `estimated` - Slice of estimated trajectory records
/// * `ground_truth` - Slice of ground truth records
/// * `output_path` - Path for output PNG file
/// * `title` - Plot title
pub fn plot_trajectory_comparison(
    estimated: &[TrajectoryRecord],
    ground_truth: &[GroundTruthRecord],
    output_path: &str,
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    // Create three subplots: XY, XZ, YZ
    let (right, left) = root.split_horizontally(600);
    let (top_left, bottom_left) = left.split_vertically(400);
    let (top_right, bottom_right) = right.split_vertically(400);

    // Build ground truth map for matching
    let gt_map: HashMap<usize, &GroundTruthRecord> = ground_truth
        .iter()
        .map(|r| (r.frame_idx as usize, r))
        .collect();

    // Prepare data points
    let mut xy_est = Vec::new();
    let mut xy_gt = Vec::new();
    let mut xz_est = Vec::new();
    let mut xz_gt = Vec::new();
    let mut yz_est = Vec::new();
    let mut yz_gt = Vec::new();

    for est in estimated {
        if let Some(&gt) = gt_map.get(&est.frame_idx) {
            xy_est.push((est.x, est.y));
            xz_est.push((est.x, est.z));
            yz_est.push((est.y, est.z));
            xy_gt.push((gt.x, gt.y));
            xz_gt.push((gt.x, gt.z));
            yz_gt.push((gt.y, gt.z));
        }
    }

    // Plot XY projection (top_left)
    if !xy_est.is_empty() && !xy_gt.is_empty() {
        let x_min = xy_est.iter().chain(xy_gt.iter()).map(|p| p.0).fold(f64::INFINITY, f64::min);
        let x_max = xy_est.iter().chain(xy_gt.iter()).map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let y_min = xy_est.iter().chain(xy_gt.iter()).map(|p| p.1).fold(f64::INFINITY, f64::min);
        let y_max = xy_est.iter().chain(xy_gt.iter()).map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        let x_pad = (x_max - x_min).max(1.0) * 0.1;
        let y_pad = (y_max - y_min).max(1.0) * 0.1;

        let mut chart = ChartBuilder::on(&top_left)
            .caption("XY Plane (Top View)", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(x_min - x_pad..x_max + x_pad, y_min - y_pad..y_max + y_pad)?;

        chart.configure_mesh()
            .x_desc("X")
            .y_desc("Y")
            .draw()?;

        chart.draw_series(LineSeries::new(xy_gt.iter().cloned(), GREEN.stroke_width(2)))?
            .label("Ground Truth")
            .legend(|(x, y)| PathElement::new(vec![(x, y)], GREEN.stroke_width(2)));
        chart.draw_series(LineSeries::new(xy_est.iter().cloned(), RED.stroke_width(2)))?
            .label("Estimated")
            .legend(|(x, y)| PathElement::new(vec![(x, y)], RED.stroke_width(2)));
        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .draw()?;
    }

    // Plot XZ projection (bottom_left)
    if !xz_est.is_empty() && !xz_gt.is_empty() {
        let x_min = xz_est.iter().chain(xz_gt.iter()).map(|p| p.0).fold(f64::INFINITY, f64::min);
        let x_max = xz_est.iter().chain(xz_gt.iter()).map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let y_min = xz_est.iter().chain(xz_gt.iter()).map(|p| p.1).fold(f64::INFINITY, f64::min);
        let y_max = xz_est.iter().chain(xz_gt.iter()).map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        let x_pad = (x_max - x_min).max(1.0) * 0.1;
        let y_pad = (y_max - y_min).max(1.0) * 0.1;

        let mut chart = ChartBuilder::on(&bottom_left)
            .caption("XZ Plane (Side View)", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(x_min - x_pad..x_max + x_pad, y_min - y_pad..y_max + y_pad)?;

        chart.configure_mesh()
            .x_desc("X")
            .y_desc("Z")
            .draw()?;

        chart.draw_series(LineSeries::new(xz_gt.iter().cloned(), GREEN.stroke_width(2)))?;
        chart.draw_series(LineSeries::new(xz_est.iter().cloned(), RED.stroke_width(2)))?;
    }

    // Plot YZ projection (top_right)
    if !yz_est.is_empty() && !yz_gt.is_empty() {
        let x_min = yz_est.iter().chain(yz_gt.iter()).map(|p| p.0).fold(f64::INFINITY, f64::min);
        let x_max = yz_est.iter().chain(yz_gt.iter()).map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let y_min = yz_est.iter().chain(yz_gt.iter()).map(|p| p.1).fold(f64::INFINITY, f64::min);
        let y_max = yz_est.iter().chain(yz_gt.iter()).map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        let x_pad = (x_max - x_min).max(1.0) * 0.1;
        let y_pad = (y_max - y_min).max(1.0) * 0.1;

        let mut chart = ChartBuilder::on(&top_right)
            .caption("YZ Plane (Side View)", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(x_min - x_pad..x_max + x_pad, y_min - y_pad..y_max + y_pad)?;

        chart.configure_mesh()
            .x_desc("Y")
            .y_desc("Z")
            .draw()?;

        chart.draw_series(LineSeries::new(yz_gt.iter().cloned(), GREEN.stroke_width(2)))?;
        chart.draw_series(LineSeries::new(yz_est.iter().cloned(), RED.stroke_width(2)))?;
    }

    // Plot error over time (bottom_right)
    let mut errors_x = Vec::new();
    let mut errors_y = Vec::new();
    let mut errors_z = Vec::new();
    for est in estimated {
        if let Some(&gt) = gt_map.get(&est.frame_idx) {
            errors_x.push((est.frame_idx as f64, est.x - gt.x));
            errors_y.push((est.frame_idx as f64, est.y - gt.y));
            errors_z.push((est.frame_idx as f64, est.z - gt.z));
        }
    }

    if !errors_x.is_empty() {
        let frame_min = errors_x.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let frame_max = errors_x.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let err_min = errors_x.iter().chain(errors_y.iter()).chain(errors_z.iter())
            .map(|p| p.1).fold(f64::INFINITY, f64::min);
        let err_max = errors_x.iter().chain(errors_y.iter()).chain(errors_z.iter())
            .map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);

        let mut chart = ChartBuilder::on(&bottom_right)
            .caption("Tracking Error Over Time", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(frame_min..frame_max, err_min..err_max)?;

        chart.configure_mesh()
            .x_desc("Frame Index")
            .y_desc("Error")
            .draw()?;

        chart.draw_series(LineSeries::new(errors_x.into_iter(), RED.stroke_width(1)))?;
        chart.draw_series(LineSeries::new(errors_y.into_iter(), GREEN.stroke_width(1)))?;
        chart.draw_series(LineSeries::new(errors_z.into_iter(), BLUE.stroke_width(1)))?;
    }

    // Add title
    root.titled(title, ("sans-serif", 30))?;

    root.present()?;
    println!("✅ Trajectory plot saved to: {}", output_path);
    Ok(())
}

/// Plot 3D trajectory (static 3D projection to 2D)
pub fn plot_3d_trajectory(
    estimated: &[TrajectoryRecord],
    ground_truth: &[GroundTruthRecord],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let gt_map: HashMap<usize, &GroundTruthRecord> = ground_truth
        .iter()
        .map(|r| (r.frame_idx as usize, r))
        .collect();

    let mut points_est = Vec::new();
    let mut points_gt = Vec::new();

    for est in estimated {
        if let Some(&gt) = gt_map.get(&est.frame_idx) {
            points_est.push((est.x, est.y, est.z));
            points_gt.push((gt.x, gt.y, gt.z));
        }
    }

    if points_est.is_empty() {
        return Ok(());
    }

    let project = |x: f64, y: f64, z: f64| -> (f64, f64) {
        let scale = 50.0;
        let x2d = (x - y) * scale;
        let y2d = (x + y) * 0.5 * scale - z * scale;
        (x2d, y2d)
    };

    let projected_est: Vec<_> = points_est
        .iter()
        .map(|(x, y, z)| project(*x, *y, *z))
        .collect();
    let projected_gt: Vec<_> = points_gt
        .iter()
        .map(|(x, y, z)| project(*x, *y, *z))
        .collect();

    let all_points: Vec<_> = projected_est
        .iter()
        .chain(projected_gt.iter())
        .collect();
    let x_min = all_points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let x_max = all_points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let y_min = all_points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let y_max = all_points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("3D Trajectory (Isometric View)", ("sans-serif", 25))
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh()
        .x_desc("X-Y Projection")
        .y_desc("Z Axis")
        .draw()?;

    chart.draw_series(LineSeries::new(projected_gt.into_iter(), GREEN.stroke_width(2)))?;
    chart.draw_series(LineSeries::new(projected_est.into_iter(), RED.stroke_width(2)))?;

    root.present()?;
    println!("✅ 3D trajectory plot saved to: {}", output_path);
    Ok(())
}

/// Save metrics to a text file
pub fn save_metrics_to_file(
    metrics: &TrackingMetrics,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(output_path)?;
    writeln!(file, "{}", metrics)?;
    println!("✅ Metrics saved to: {}", output_path);
    Ok(())
}

/// Generate all visualization outputs
pub fn generate_all_plots(
    estimated: &[TrajectoryRecord],
    ground_truth: &[GroundTruthRecord],
    output_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(output_dir)?;

    let metrics = TrackingMetrics::compute(estimated, ground_truth);
    println!("{}", metrics);

    let metrics_path = format!("{}/metrics.txt", output_dir);
    save_metrics_to_file(&metrics, &metrics_path)?;

    let comparison_path = format!("{}/trajectory_comparison.png", output_dir);
    plot_trajectory_comparison(
        estimated,
        ground_truth,
        &comparison_path,
        "Trajectory Comparison: Estimated vs Ground Truth",
    )?;

    let trajectory_3d_path = format!("{}/trajectory_3d.png", output_dir);
    plot_3d_trajectory(estimated, ground_truth, &trajectory_3d_path)?;

    Ok(())
}
