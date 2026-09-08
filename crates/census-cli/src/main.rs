use std::{
    io::{IsTerminal, Write},
    path::PathBuf,
    time::Duration,
};

use anyhow::Result;
use census_core::{CrawlOptions, CrawlReport, SteamClient, crawl};
use clap::{Parser, Subcommand};

/// How often to emit a progress line when stderr is not a terminal.
const PROGRESS_EVERY_PAGES: u32 = 25;

#[derive(Parser, Debug)]
#[command(
    name = "census",
    version,
    about = "Count what Steam reviewers actually say, rather than what the loudest ones say."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Download every review Valve will serve for an app into an immutable Parquet capture.
    Crawl {
        /// Steam app ID, as it appears in the store URL.
        app_id: u32,
        /// Directory to write the capture into.
        #[arg(short, long, default_value = "data")]
        out: PathBuf,
        /// Milliseconds to wait between pages.
        #[arg(long, default_value_t = 250)]
        pace_ms: u64,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Crawl {
            app_id,
            out,
            pace_ms,
        } => run_crawl(app_id, out, pace_ms).await,
    }
}

async fn run_crawl(app_id: u32, out: PathBuf, pace_ms: u64) -> Result<()> {
    let client = SteamClient::new()?;
    let options = CrawlOptions {
        out_dir: out,
        page_interval: Duration::from_millis(pace_ms),
    };

    // A crawl of a large corpus runs for hours, so it is often piped to a log, where a
    // carriage-returned progress line becomes one unreadable smear.
    let interactive = std::io::stderr().is_terminal();

    eprintln!("crawling app {app_id}");
    let report = crawl(&client, app_id, &options, |progress| {
        let line = format!(
            "  page {:<5} {} of {} reviews",
            progress.pages,
            thousands(progress.unique),
            thousands(progress.valve_total)
        );
        if interactive {
            let mut err = std::io::stderr();
            let _ = write!(err, "\r{line}");
            let _ = err.flush();
        } else if progress.pages.is_multiple_of(PROGRESS_EVERY_PAGES) {
            eprintln!("{line}");
        }
    })
    .await?;
    if interactive {
        eprintln!();
    }

    print_report(&report);
    Ok(())
}

fn print_report(report: &CrawlReport) {
    let coverage = report
        .coverage()
        .map_or_else(|| "unknown".to_owned(), |c| format!("{:.2}%", c * 100.0));

    println!("app {}  ({})", report.app_id, report.review_score_desc);
    println!("  pages        {}", report.pages);
    println!("  fetched      {}", thousands(report.fetched));
    println!("  unique       {}", thousands(report.unique));
    println!("  duplicates   {}", thousands(report.duplicates()));
    println!(
        "  valve total  {}  ({} up / {} down)",
        thousands(report.valve_total),
        thousands(report.valve_positive),
        thousands(report.valve_negative)
    );
    println!("  coverage     {coverage}");
    println!("  stopped      {:?}", report.stopped_because);
    println!("  elapsed      {:.1}s", report.elapsed.as_secs_f64());
    println!("  capture      {}", report.path.display());

    if report.stopped_because != census_core::StopReason::Exhausted {
        eprintln!(
            "\nwarning: the walk did not reach the end of the corpus, so coverage is a floor."
        );
    }
}

fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::thousands;

    #[test]
    fn thousands_groups_digits() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(2909), "2,909");
        assert_eq!(thousands(986_295), "986,295");
        assert_eq!(thousands(1_700_000), "1,700,000");
    }
}
