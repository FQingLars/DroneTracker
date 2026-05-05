mod types;
mod height;
mod gmc;
mod gui;

fn main() {
    if let Err(e) = gui::run_gui() {
        eprintln!("GUI error: {}", e);
    }
}
