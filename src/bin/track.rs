use opencv::core::Mat;
use opencv::prelude::*;
use opencv::videoio::{VideoCapture, VideoCaptureTrait, CAP_ANY};
use std::env;
use std::error::Error;

use Tracker::height::HeightEstimator;
use Tracker::gmc::GmcTracker;
use Tracker::types::CsvRow;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <video_path> <csv_path>", args[0]);
        return Ok(());
    }
    let video_path = &args[1];
    let csv_path = &args[2];

    eprintln!("Loading CSV: {}", csv_path);
    let mut csv_data = Vec::new();
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(csv_path)?;
    for result in rdr.deserialize() {
        let row: CsvRow = result?;
        csv_data.push(row);
    }
    eprintln!("Loaded {} CSV rows", csv_data.len());

    eprintln!("Opening video: {}", video_path);
    let mut cap = VideoCapture::from_file(video_path, CAP_ANY)?;
    if !cap.is_opened()? {
        eprintln!("Failed to open video");
        return Ok(());
    }

    let mut tracker = GmcTracker::new()?;
    let mut height_est = HeightEstimator::new(800.0, 960.0, 540.0, 12.8)?;

    let mut frame = Mat::default();
    let mut frame_num = 0u32;
    let mut pos = (0.0f64, 0.0f64, 100.0f64);
    let mut prev_frame: Option<Mat> = None;

    eprintln!("Processing video...");
    while cap.read(&mut frame)? {
        frame_num += 1;
        if frame.empty() {
            break;
        }

        match tracker.process(&frame) {
            Ok((Some(h), _)) => {
                let dx = *h.at_2d::<f64>(0, 2).unwrap_or(&0.0);
                let dy = *h.at_2d::<f64>(1, 2).unwrap_or(&0.0);
                let h_norm = (dx*dx + dy*dy).sqrt();
                // Инвертируем, так как движение камеры обратно движению точек
                let scale = 5.0; // увеличенный масштаб для теста
                pos.0 -= dx * scale;
                pos.1 -= dy * scale;

                eprintln!(
                    "Frame {}: dx={:.2}, dy={:.2}, norm={:.2}, scale={:.2}",
                    frame_num, dx, dy, h_norm, scale
                );
                eprintln!(
                    "         H=[[{:.3}, {:.3}, {:.3}], [{:.3}, {:.3}, {:.3}], [{:.3}, {:.3}, {:.3}]]",
                    *h.at_2d::<f64>(0, 0).unwrap_or(&0.0),
                    *h.at_2d::<f64>(0, 1).unwrap_or(&0.0),
                    *h.at_2d::<f64>(0, 2).unwrap_or(&0.0),
                    *h.at_2d::<f64>(1, 0).unwrap_or(&0.0),
                    *h.at_2d::<f64>(1, 1).unwrap_or(&0.0),
                    *h.at_2d::<f64>(1, 2).unwrap_or(&0.0),
                    *h.at_2d::<f64>(2, 0).unwrap_or(&0.0),
                    *h.at_2d::<f64>(2, 1).unwrap_or(&0.0),
                    *h.at_2d::<f64>(2, 2).unwrap_or(&0.0),
                );
            }
            Ok((None, _)) => {
                eprintln!("Frame {}: no homography", frame_num);
            }
            Err(e) => {
                eprintln!("Frame {}: error: {}", frame_num, e);
            }
        }

        if let Some(ref prev) = prev_frame {
            match height_est.estimate(prev, &frame) {
                Ok(z) => {
                    pos.2 = z;
                    eprintln!("Frame {}: height updated to {:.2}", frame_num, z);
                }
                Err(e) => {
                    eprintln!("Frame {}: height error: {}", frame_num, e);
                }
            }
        }
        prev_frame = Some(frame.clone());

        if frame_num <= csv_data.len() as u32 {
            let gt = &csv_data[(frame_num - 1) as usize];
            println!(
                "{} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2}",
                frame_num,
                pos.0, pos.1, pos.2,
                gt.translation_x, gt.translation_y, gt.translation_z
            );
        } else {
            println!("{} {:.2} {:.2} {:.2} 0.0 0.0 0.0", frame_num, pos.0, pos.1, pos.2);
        }

        if frame_num % 10 == 0 {
            eprintln!("Processed {} frames", frame_num);
        }
    }

    eprintln!("Done. Processed {} frames", frame_num);
    Ok(())
}
