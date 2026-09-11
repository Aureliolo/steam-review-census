mod cli;
mod ui;

/// One binary, two front ends: a window when it is opened like an application, the pipeline
/// when it is given arguments. Neither is the real one. Anything a person can do in the
/// window can be scripted, and anything a script can do can be watched happening.
fn main() -> anyhow::Result<()> {
    if std::env::args_os().nth(1).is_some() {
        return tokio::runtime::Runtime::new()?.block_on(cli::run());
    }
    ui::run()
}
