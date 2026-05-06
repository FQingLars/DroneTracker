use eframe::{self, App, Frame as EFrame};
use egui::{CentralPanel, Grid, TopBottomPanel};
use egui_plot::{Legend, Line, Plot, PlotPoints};
use opencv::core::Mat;
use opencv::prelude::*;
use opencv::videoio::{VideoCapture, VideoCaptureTrait, VideoCaptureTraitConst, CAP_ANY};
use rfd::FileDialog;
use std::sync::mpsc::{channel, Sender, Receiver};

use crate::gmc::GmcTracker;
use crate::types::CsvRow;

pub struct TrackerApp {
    video_path: Option<String>,
    csv_data: Vec<CsvRow>,
    estimated: Vec<(f64, f64, f64, u32)>,
    status: String,
    rx: Option<Receiver<(f64, f64, f64, u32)>>,
    processing: bool,
}

impl Default for TrackerApp {
    fn default() -> Self {
        Self {
            video_path: None,
            csv_data: Vec::new(),
            estimated: Vec::new(),
            status: "Загрузите CSV и видео".to_string(),
            rx: None,
            processing: false,
        }
    }
}

impl App for TrackerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut EFrame) {
        // Проверяем новые результаты (все доступные на данный момент)
        if let Some(ref rx) = self.rx {
            let mut count = 0;
            loop {
                match rx.try_recv() {
                    Ok(result) => {
                        self.estimated.push(result);
                        count += 1;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        // Поток завершился
                        self.rx = None;
                        self.processing = false;
                        self.status = format!(
                            "Завершено: {} кадров обработано",
                            self.estimated.len()
                        );
                        eprintln!("[GUI] Обработка завершена: {} кадров", self.estimated.len());
                        break;
                    }
                }
            }
            if count > 0 {
                eprintln!("[GUI] Получено {} новых результатов, всего: {}", count, self.estimated.len());
                // Запрашиваем перерисовку для обновления графиков
                ctx.request_repaint();
            }
        }

        TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Загрузить CSV").clicked() {
                    if let Some(path) = FileDialog::new().add_filter("CSV", &["csv"]).pick_file() {
                        self.load_csv(&path.to_string_lossy());
                    }
                }
                if ui.button("Загрузить видео").clicked() {
                    if let Some(path) = FileDialog::new().add_filter("Video", &["mp4", "avi", "mov", "mkv"]).pick_file() {
                        self.video_path = Some(path.to_string_lossy().to_string());
                        self.status = format!("Видео: {}", self.video_path.as_ref().unwrap());
                    }
                }
                let processing = self.processing;
                if ui.add_enabled(!processing, egui::Button::new("Обработать")).clicked() && !processing {
                    self.start_processing();
                }
            });
            ui.label(&self.status);
        });

        CentralPanel::default().show(ctx, |ui| {
            if self.csv_data.is_empty() && self.estimated.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label("Загрузите CSV и видео, затем нажмите 'Обработать'");
                });
                return;
            }

            // Используем ui.columns(2, ...) для создания двух колонок равной ширины
            ui.columns(2, |cols| {
                // Левая колонка (индекс 0)
                cols[0].vertical(|ui| {
                    ui.label("Translation X");
                    Plot::new("plot_x")
                        .legend(Legend::default())
                        .height(280.0)
                        .show(ui, |plot_ui| {
                            if !self.csv_data.is_empty() {
                                let gt: PlotPoints = self.csv_data.iter()
                                    .map(|r| [r.frame_num as f64, r.translation_x])
                                    .collect();
                                plot_ui.line(Line::new(gt).name("GT X"));
                            }
                            if !self.estimated.is_empty() {
                                let est: PlotPoints = self.estimated.iter()
                                    .map(|(x, _, _, f)| [*f as f64, *x])
                                    .collect();
                                plot_ui.line(Line::new(est).name("Est X"));
                            }
                        });

                    ui.label("Translation Z");
                    Plot::new("plot_z")
                        .legend(Legend::default())
                        .height(280.0)
                        .show(ui, |plot_ui| {
                            if !self.csv_data.is_empty() {
                                let gt: PlotPoints = self.csv_data.iter()
                                    .map(|r| [r.frame_num as f64, r.translation_z])
                                    .collect();
                                plot_ui.line(Line::new(gt).name("GT Z"));
                            }
                            if !self.estimated.is_empty() {
                                let est: PlotPoints = self.estimated.iter()
                                    .map(|(_, _, z, f)| [*f as f64, *z])
                                    .collect();
                                plot_ui.line(Line::new(est).name("Est Z"));
                            }
                        });
                });

                // Правая колонка (индекс 1)
                cols[1].vertical(|ui| {
                    ui.label("Translation Y");
                    Plot::new("plot_y")
                        .legend(Legend::default())
                        .height(280.0)
                        .show(ui, |plot_ui| {
                            if !self.csv_data.is_empty() {
                                let gt: PlotPoints = self.csv_data.iter()
                                    .map(|r| [r.frame_num as f64, r.translation_y])
                                    .collect();
                                plot_ui.line(Line::new(gt).name("GT Y"));
                            }
                            if !self.estimated.is_empty() {
                                let est: PlotPoints = self.estimated.iter()
                                    .map(|(_, y, _, f)| [*f as f64, *y])
                                    .collect();
                                plot_ui.line(Line::new(est).name("Est Y"));
                            }
                        });

                    ui.label("XY Trajectory");
                    Plot::new("plot_xy")
                        .legend(Legend::default())
                        .height(280.0)
                        .show(ui, |plot_ui| {
                            if !self.csv_data.is_empty() {
                                let gt: PlotPoints = self.csv_data.iter()
                                    .map(|r| [r.translation_x, r.translation_y])
                                    .collect();
                                plot_ui.line(Line::new(gt).name("GT XY"));
                            }
                            if !self.estimated.is_empty() {
                                let est: PlotPoints = self.estimated.iter()
                                    .map(|(x, y, _, _)| [*x, *y])
                                    .collect();
                                plot_ui.line(Line::new(est).name("Est XY"));
                            }
                        });
                });
            });
        });
    }
}

impl TrackerApp {
    fn load_csv(&mut self, path: &str) {
        self.csv_data.clear();
        let mut rdr = match csv::ReaderBuilder::new().has_headers(false).from_path(path) {
            Ok(r) => r,
            Err(e) => {
                self.status = format!("Ошибка CSV: {}", e);
                return;
            }
        };
        for result in rdr.deserialize() {
            if let Ok(row) = result {
                self.csv_data.push(row);
            }
        }
        self.status = format!("CSV загружен: {} строк", self.csv_data.len());
        eprintln!("[GUI] CSV загружен: {} строк", self.csv_data.len());
    }

    fn start_processing(&mut self) {
        let path = match self.video_path.clone() {
            Some(p) => p,
            None => {
                self.status = "Сначала загрузите видео".to_string();
                return;
            }
        };
        self.estimated.clear();
        self.processing = true;
        self.status = "Обработка...".to_string();

        let (tx, rx) = channel();
        self.rx = Some(rx);

        eprintln!("[GUI] Запуск обработки видео: {}", path);

        std::thread::spawn(move || {
            let count = Self::process_video(&path, &tx);
            eprintln!("[PROC] Поток обработки завершён, отправлено {} кадров", count);
        });
    }

    fn process_video(path: &str, tx: &Sender<(f64, f64, f64, u32)>) -> u32 {
        eprintln!("[PROC] Открытие видео: {}", path);
        let mut cap = match VideoCapture::from_file(path, CAP_ANY) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[PROC] Ошибка открытия видео: {}", e);
                return 0;
            }
        };
        if !cap.is_opened().unwrap_or(false) {
            eprintln!("[PROC] Видео не открылось");
            return 0;
        }

        eprintln!("[PROC] Создание GmcTracker");
        let mut tracker = match GmcTracker::new() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[PROC] Ошибка создания GmcTracker: {}", e);
                return 0;
            }
        };

        let mut frame = Mat::default();
        let mut frame_num = 0u32;
        let mut pos = (0.0f64, 0.0f64, 0.0f64);
        let mut sent = 0u32;

        eprintln!("[PROC] Начало обработки кадров");
        while let Ok(true) = cap.read(&mut frame) {
            if frame.empty() {
                eprintln!("[PROC] Кадр {} пустой, завершение", frame_num);
                break;
            }
            frame_num += 1;

            eprintln!("[PROC] Кадр {}: обработка", frame_num);
            match tracker.process(&frame) {
                Ok((Some(h), _)) => {
                    let dx = *h.at_2d::<f64>(0, 2).unwrap_or(&0.0);
                    let dy = *h.at_2d::<f64>(1, 2).unwrap_or(&0.0);
                    let scale = 0.001;
                    pos.0 += dx * scale;
                    pos.1 += dy * scale;
                    eprintln!(
                        "[PROC] Кадр {}: dx={:.4}, dy={:.4}, pos=({:.4}, {:.4}, {:.4})",
                        frame_num, dx, dy, pos.0, pos.1, pos.2
                    );
                }
                Ok((None, _)) => {
                    eprintln!("[PROC] Кадр {}: гомография не вычислена", frame_num);
                }
                Err(e) => {
                    eprintln!("[PROC] Кадр {}: ошибка: {}", frame_num, e);
                }
            }
            let result = (pos.0, pos.1, pos.2, frame_num);
            match tx.send(result) {
                Ok(_) => {
                    sent += 1;
                    if sent % 10 == 0 {
                        eprintln!("[PROC] Отправлено {} результатов", sent);
                    }
                }
                Err(e) => {
                    eprintln!("[PROC] Ошибка отправки результата: {}", e);
                    break;
                }
            }
        }

        eprintln!("[PROC] Обработка завершена: {} кадров обработано, {} результатов отправлено", frame_num, sent);
        sent
    }
}

pub fn run_gui() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Tracker GUI",
        options,
        Box::new(|_cc| Ok(Box::new(TrackerApp::default()))),
    )
}
