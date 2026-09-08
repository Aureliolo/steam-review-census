use std::{
    io::{IsTerminal, Write},
    path::PathBuf,
    time::Duration,
};

use anyhow::Result;
use census_core::{
    ClassifyOptions, CrawlOptions, CrawlReport, DEFAULT_BATCH_SIZE, DEFAULT_SHARD_TARGET,
    SteamClient, crawl,
};
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Calibrate {
    Search,
    On,
    Off,
}

impl From<Calibrate> for census_core::anchors::Calibration {
    fn from(value: Calibrate) -> Self {
        match value {
            Calibrate::Search => Self::Search,
            Calibrate::On => Self::Always,
            Calibrate::Off => Self::Never,
        }
    }
}

/// How often to emit a progress line when stderr is not a terminal.
const PROGRESS_EVERY_SHARDS: usize = 5;
const PROGRESS_EVERY_TEXTS: u64 = 5_000;

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
        /// Directory to write captures and crawl state into.
        #[arg(short, long, default_value = "data")]
        out: PathBuf,
        /// Milliseconds between requests, enforced globally across all shards.
        #[arg(long, default_value_t = 250)]
        pace_ms: u64,
        /// Shards to crawl at once. Pacing is global, so this reorders work rather than
        /// increasing load on Valve.
        #[arg(long, default_value_t = 4)]
        concurrency: usize,
        /// Reviews per shard before a date window is split further.
        #[arg(long, default_value_t = DEFAULT_SHARD_TARGET)]
        shard_target: u64,
        /// Plan a fresh crawl instead of continuing an unfinished one.
        #[arg(long)]
        restart: bool,
        /// Fetch only reviews newer than the last completed crawl.
        #[arg(long)]
        top_up: bool,
    },

    /// Embed the captured reviews locally, so nothing is sent anywhere.
    Embed {
        /// Steam app ID whose most recent capture should be embedded.
        app_id: u32,
        /// Directory holding the capture.
        #[arg(short, long, default_value = "data")]
        out: PathBuf,
        /// Reviews per forward pass.
        #[arg(long, default_value_t = DEFAULT_BATCH_SIZE)]
        batch_size: usize,
        /// Where to cache the model. Defaults to the platform cache directory.
        #[arg(long)]
        model_dir: Option<PathBuf>,
    },

    /// Sort embedded reviews into the core-spine categories and report what players say.
    Classify {
        /// Steam app ID whose most recent capture should be classified.
        app_id: u32,
        /// Directory holding the capture and its embeddings.
        #[arg(short, long, default_value = "data")]
        out: PathBuf,
        /// How close to the best match a category must score to count as mentioned.
        /// Defaults to the margin the anchors were fitted with.
        #[arg(long)]
        mention_margin: Option<f32>,
        /// How many of the most-upvoted reviews count as "the top of the pile".
        #[arg(long, default_value_t = census_core::classify::DEFAULT_TOP_HELPFUL)]
        top_helpful: usize,
        /// Anchors fitted by `census fit`. Falls back to the written category descriptions.
        #[arg(long)]
        anchors: Option<PathBuf>,
        /// Ignore any fitted anchors and use the written category descriptions.
        #[arg(long, conflicts_with = "anchors")]
        descriptions: bool,
        /// Where to cache the model. Defaults to the platform cache directory.
        #[arg(long)]
        model_dir: Option<PathBuf>,
    },

    /// Fit category anchors from a reference set, so categories are represented by reviews
    /// people wrote rather than by descriptions of the topic.
    Fit(FitArgs),

    /// Compare stored classifications against a reference set.
    Evaluate {
        /// Steam app ID to evaluate.
        app_id: u32,
        /// Directory holding the capture and its classifications.
        #[arg(short, long, default_value = "data")]
        out: PathBuf,
        /// Reference set directory, holding manifest.json and labels.json.
        #[arg(long)]
        reference: Option<PathBuf>,
    },
}

#[derive(clap::Args, Debug)]
struct FitArgs {
    /// Steam app ID whose reference labels should be fitted.
    app_id: u32,
    /// Directory holding the capture and its embeddings.
    #[arg(short, long, default_value = "data")]
    out: PathBuf,
    /// Reference set directory, holding manifest.json and labels.json.
    #[arg(long)]
    reference: Option<PathBuf>,
    /// Subset held back from fitting entirely, so it can still measure the result.
    #[arg(long, default_value = "random")]
    holdout: String,
    /// Cross-validation folds used to choose the blend and the mention margin.
    #[arg(long, default_value_t = 5)]
    folds: usize,
    /// Score categories by how far above their own corpus average a review sits.
    /// `search` lets cross-validation decide, which favours the largest categories.
    #[arg(long, default_value = "search")]
    calibrate: Calibrate,
    /// Where to write the fitted anchors. Defaults to the reference directory.
    #[arg(long)]
    anchors_out: Option<PathBuf>,
    /// Where to cache the model. Defaults to the platform cache directory.
    #[arg(long)]
    model_dir: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Crawl {
            app_id,
            out,
            pace_ms,
            concurrency,
            shard_target,
            restart,
            top_up,
        } => {
            let options = CrawlOptions {
                out_dir: out,
                concurrency,
                shard_target,
                resume: !restart,
                top_up,
            };
            run_crawl(app_id, &options, Duration::from_millis(pace_ms)).await
        }
        Command::Embed {
            app_id,
            out,
            batch_size,
            model_dir,
        } => run_embed(app_id, &out, batch_size, model_dir).await,
        Command::Classify {
            app_id,
            out,
            mention_margin,
            top_helpful,
            anchors,
            descriptions,
            model_dir,
        } => {
            run_classify(
                app_id,
                out,
                mention_margin,
                top_helpful,
                anchors,
                descriptions,
                model_dir,
            )
            .await
        }
        Command::Fit(args) => run_fit(args).await,
        Command::Evaluate {
            app_id,
            out,
            reference,
        } => run_evaluate(app_id, &out, reference),
    }
}

fn run_evaluate(app_id: u32, out: &std::path::Path, reference: Option<PathBuf>) -> Result<()> {
    let dir = reference.unwrap_or_else(|| census_core::evaluate::default_reference_dir(app_id));
    let set = census_core::ReferenceSet::load(&dir)?;

    let unknown = set.unknown_categories();
    if !unknown.is_empty() {
        eprintln!(
            "warning: reference set names categories the taxonomy does not have: {unknown:?}"
        );
    }
    if set.spine_version != census_core::CORE_SPINE_VERSION {
        eprintln!(
            "warning: reference set was labelled against taxonomy {} but this build is {}; \
             the comparison is not meaningful.",
            set.spine_version,
            census_core::CORE_SPINE_VERSION
        );
    }

    let report = census_core::compare(&set, out, app_id)?;
    print_agreement(&report, set.human_verified);
    Ok(())
}

fn print_agreement(report: &census_core::AgreementReport, human_verified: bool) {
    let measure = if human_verified {
        "accuracy"
    } else {
        "agreement"
    };
    println!("app {}", report.app_id);
    println!("  reference    {}", report.produced_by);
    println!("  compared     {}", thousands(report.compared));
    if report.unmatched > 0 {
        println!("  unmatched    {}", thousands(report.unmatched));
    }
    println!(
        "  primary {measure:<4} {}",
        report
            .primary_agreement
            .map_or_else(|| "n/a".to_owned(), |a| format!("{:.1}%", a * 100.0))
    );
    println!(
        "  macro F1     {}",
        report
            .macro_f1()
            .map_or_else(|| "n/a".to_owned(), |f| format!("{f:.3}"))
    );
    match report.anchors.as_slice() {
        [] => {}
        [one] => println!("  anchors      {one}"),
        many => println!(
            "  anchors      {} (MIXED; re-run `census classify`)",
            many.join(", ")
        ),
    }
    print_slices(&report.slices, measure);

    let mut categories = report.categories.clone();
    categories.sort_by_key(|c| std::cmp::Reverse(c.reference_mentions));
    println!(
        "\n{:<26} {:>8} {:>10} {:>8} {:>7}",
        "category", "in ref", "precision", "recall", "F1"
    );
    println!("{}", "-".repeat(64));
    for stat in &categories {
        let fmt = |v: Option<f64>| v.map_or_else(|| "    -".to_owned(), |x| format!("{x:.2}"));
        println!(
            "{:<26} {:>8} {:>10} {:>8} {:>7}",
            stat.label,
            stat.reference_mentions,
            fmt(stat.precision()),
            fmt(stat.recall()),
            fmt(stat.f1())
        );
    }

    if human_verified {
        return;
    }
    println!(
        "\nThese are AGREEMENT figures, not accuracy. The reference labels were produced by a\n\
         model, so this measures consistency between two models rather than correctness.\n\
         Two models can agree and both be wrong, most likely on sarcasm and on reviews that\n\
         sit between categories, which is exactly where this classifier is weakest."
    );
}

/// Where `census fit` writes its result, and where `census classify` looks for it.
fn default_anchor_path(app_id: u32) -> PathBuf {
    census_core::evaluate::default_reference_dir(app_id).join("anchors.json")
}

/// Loads fitted anchors if there are any, falling back to embedding the descriptions.
///
/// Only the fallback needs the embedding model, so classifying with a fitted set never
/// downloads one.
async fn load_anchors(
    app_id: u32,
    explicit: Option<PathBuf>,
    force_descriptions: bool,
    model_dir: Option<PathBuf>,
) -> Result<census_core::Anchors> {
    let path = explicit
        .clone()
        .unwrap_or_else(|| default_anchor_path(app_id));
    if !force_descriptions && (explicit.is_some() || path.exists()) {
        let anchors = census_core::Anchors::load(&path)?;
        if anchors.fitted_from != Some(app_id) {
            eprintln!(
                "warning: these anchors were fitted on app {}, not {app_id}. Categories carry \
                 the vocabulary of the game they were fitted on, and whether that transfers \
                 has not been measured.",
                anchors
                    .fitted_from
                    .map_or_else(|| "nothing".to_owned(), |id| id.to_string())
            );
        }
        eprintln!("anchors      {}", path.display());
        return Ok(anchors);
    }

    let cache = model_dir.unwrap_or_else(census_core::model::default_cache_dir);
    census_core::model::ensure(&cache, |_| {}).await?;
    let mut embedder = census_core::Embedder::load(&cache)?;
    eprintln!(
        "anchors      written descriptions, on {}",
        embedder.device()
    );
    Ok(census_core::Anchors::from_descriptions(&mut embedder)?)
}

async fn run_classify(
    app_id: u32,
    out: PathBuf,
    mention_margin: Option<f32>,
    top_helpful: usize,
    anchors: Option<PathBuf>,
    descriptions: bool,
    model_dir: Option<PathBuf>,
) -> Result<()> {
    let anchors = load_anchors(app_id, anchors, descriptions, model_dir).await?;
    let options = ClassifyOptions {
        out_dir: out,
        mention_margin: mention_margin.unwrap_or_else(|| anchors.mention_margin()),
        top_helpful,
    };

    eprintln!("classifying app {app_id}");
    let report = census_core::classify_corpus(&anchors, app_id, &options, |_| {})?;
    print_classification(&report);
    Ok(())
}

async fn run_fit(args: FitArgs) -> Result<()> {
    let FitArgs {
        app_id,
        out,
        reference,
        holdout,
        folds,
        calibrate,
        anchors_out,
        model_dir,
    } = args;
    let holdout = holdout.as_str();
    let dir = reference.unwrap_or_else(|| census_core::evaluate::default_reference_dir(app_id));
    let set = census_core::ReferenceSet::load(&dir)?;
    if set.spine_version != census_core::CORE_SPINE_VERSION {
        anyhow::bail!(
            "reference set was labelled against taxonomy {} but this build is {}; fitting \
             anchors from it would move every category boundary towards a taxonomy that no \
             longer exists.",
            set.spine_version,
            census_core::CORE_SPINE_VERSION
        );
    }

    let vectors = census_core::anchors::corpus_vectors(&out, app_id)?;
    let training: Vec<census_core::evaluate::ReferenceLabel> = set
        .labels
        .iter()
        .filter(|label| label.subset != holdout)
        .cloned()
        .collect();
    let examples = census_core::anchors::to_examples(&training, &vectors);
    if examples.is_empty() {
        anyhow::bail!(census_core::Error::NoTrainingExamples { app_id });
    }
    // Calibration reads review vectors and no labels, so the whole corpus is fair game and
    // is also what the classifier will actually be run over.
    let corpus: Vec<Vec<f32>> = vectors.into_values().collect();

    let cache = model_dir.unwrap_or_else(census_core::model::default_cache_dir);
    census_core::model::ensure(&cache, |_| {}).await?;
    let mut embedder = census_core::Embedder::load(&cache)?;
    let descriptions = census_core::Anchors::from_descriptions(&mut embedder)?;

    eprintln!(
        "fitting {} labels from app {app_id} against {} reviews, holding back the {holdout} subset",
        examples.len(),
        corpus.len()
    );
    let outcome =
        census_core::anchors::search(&descriptions, &examples, &corpus, folds, calibrate.into());
    let mut fitted = descriptions.fit(&examples, app_id, outcome.params);
    if outcome.params.calibrate {
        fitted.calibrate(&corpus);
    }

    let path = anchors_out.unwrap_or_else(|| dir.join("anchors.json"));
    fitted.save(&path)?;
    print_fit(
        &fitted,
        &outcome,
        &path,
        holdout,
        set.labels.len() - training.len(),
    );
    Ok(())
}

fn print_fit(
    anchors: &census_core::Anchors,
    outcome: &census_core::FitOutcome,
    path: &std::path::Path,
    holdout: &str,
    held_back: usize,
) {
    println!("app {}", anchors.fitted_from.unwrap_or_default());
    println!("  fitted from  {} labels", outcome.examples);
    println!("  held back    {held_back} ({holdout} subset, never seen by the fit)");
    println!("  smoothing    {}", outcome.params.smoothing);
    println!("  secondary    {}", outcome.params.secondary_weight);
    println!("  margin       {}", outcome.params.mention_margin);
    println!(
        "  calibration  {}",
        if outcome.params.calibrate {
            "on"
        } else {
            "off"
        }
    );
    println!("  anchors      {}", path.display());
    println!(
        "\n  {}-fold cross-validation, on training reviews each fold did not see:",
        outcome.folds
    );
    let baseline_note = if outcome.baseline_calibrated {
        "descriptions alone, also calibrated"
    } else {
        "descriptions alone"
    };
    println!(
        "    primary agreement  {:.1}%  ({baseline_note}: {:.1}%)",
        outcome.primary_agreement * 100.0,
        outcome.baseline_agreement * 100.0
    );
    println!(
        "    mention macro F1   {:.3}   ({baseline_note}: {:.3})",
        outcome.mention_macro_f1, outcome.baseline_macro_f1
    );

    println!(
        "\n{:<30} {:>10} {:>16}",
        "category", "evidence", "learned share"
    );
    println!("{}", "-".repeat(58));
    for anchor in &anchors.categories {
        let label =
            census_core::taxonomy::by_id(&anchor.id).map_or(anchor.id.as_str(), |c| c.label);
        println!(
            "{label:<30} {:>10.1} {:>15.0}%",
            anchor.evidence,
            anchor.learned_share * 100.0
        );
    }
    println!(
        "\nEvidence is labelled reviews behind an anchor, secondary mentions counted at {}.\n\
         Learned share is how far the anchor moved off its written description; a category\n\
         with no labels keeps its description exactly and is unchanged by fitting.",
        outcome.params.secondary_weight
    );
    println!(
        "\nCross-validated figures are measured on the fitting data and are optimistic: the\n\
         blend and the margin were both chosen against them. Classify with these anchors and\n\
         run `census evaluate` to read the held-out {holdout} subset, which is the honest number."
    );
}

fn print_classification(report: &census_core::ClassifyReport) {
    println!("app {}", report.app_id);
    println!("  reviews      {}", thousands(report.reviews));
    if report.unmatched > 0 {
        println!(
            "  unmatched    {} (no embedding; re-run `census embed`)",
            thousands(report.unmatched)
        );
    }
    println!("  taxonomy     {}", report.spine_version);
    println!("  model        {}", report.model);
    println!(
        "  anchors      {}",
        report.anchors_fitted_from.map_or_else(
            || "written descriptions (unfitted)".to_owned(),
            |id| format!("fitted on app {id}")
        )
    );
    println!("  elapsed      {:.1}s", report.elapsed.as_secs_f64());
    println!("  assignments  {}", report.path.display());

    let mut categories = report.categories.clone();
    categories.sort_by(|a, b| {
        b.mention_rate(report.reviews)
            .total_cmp(&a.mention_rate(report.reviews))
    });

    println!(
        "\n{:<26} {:>9} {:>9} {:>9} {:>7}",
        "category", "mention%", "primary%", "top50%", "bias"
    );
    println!("{}", "-".repeat(64));
    for stat in &categories {
        let bias = stat
            .bias_factor(report.reviews, report.top_helpful)
            .map_or_else(|| "    -".to_owned(), |b| format!("{b:.2}x"));
        println!(
            "{:<26} {:>8.1}% {:>8.1}% {:>8.1}% {:>7}",
            stat.label,
            stat.mention_rate(report.reviews) * 100.0,
            stat.primary_share(report.reviews) * 100.0,
            stat.top_mention_rate(report.top_helpful) * 100.0,
            bias
        );
    }
    println!(
        "\nmention% counts every review that says anything about a category, so it sums to \
         more than 100%.\nprimary% counts each review once and sums to 100%.\nbias is how \
         much the {} most-upvoted reviews overstate a category.",
        report.top_helpful
    );
    println!(
        "\nAccuracy of these assignments is UNMEASURED. No gold set exists yet, so treat \
         every figure above as provisional."
    );
}

async fn run_embed(
    app_id: u32,
    out: &std::path::Path,
    batch_size: usize,
    model_dir: Option<PathBuf>,
) -> Result<()> {
    let cache = model_dir.unwrap_or_else(census_core::model::default_cache_dir);
    let interactive = std::io::stderr().is_terminal();

    eprintln!("model cache: {}", cache.display());
    let mut announced = String::new();
    census_core::model::ensure(&cache, |progress| {
        if announced != progress.file {
            progress.file.clone_into(&mut announced);
            eprintln!("  downloading {}", progress.file);
        }
        if interactive {
            let mut err = std::io::stderr();
            let _ = write!(
                err,
                "\r    {} MB{}   ",
                progress.downloaded / 1_048_576,
                progress
                    .total
                    .map_or(String::new(), |t| format!(" of {} MB", t / 1_048_576))
            );
            let _ = err.flush();
        }
    })
    .await?;
    if interactive && !announced.is_empty() {
        eprintln!();
    }

    let mut embedder = census_core::Embedder::load(&cache)?;
    eprintln!("embedding app {app_id} on {}", embedder.device());
    let mut last_line = 0;
    let report = census_core::embed_corpus(&mut embedder, out, app_id, batch_size, |progress| {
        let line = format!(
            "  {} of {} distinct texts",
            thousands(progress.embedded),
            thousands(progress.unique_texts)
        );
        if interactive {
            let mut err = std::io::stderr();
            let _ = write!(err, "\r{line}   ");
            let _ = err.flush();
        } else if progress.embedded - last_line >= PROGRESS_EVERY_TEXTS {
            last_line = progress.embedded;
            eprintln!("{line}");
        }
    })?;
    if interactive {
        eprintln!();
    }

    println!("app {}", report.app_id);
    println!("  device         {}", report.device);
    println!("  reviews        {}", thousands(report.reviews));
    println!("  distinct texts {}", thousands(report.unique_texts));
    println!(
        "  deduped        {}",
        report
            .dedupe_rate()
            .map_or_else(|| "n/a".to_owned(), |r| format!("{:.1}%", r * 100.0))
    );
    println!("  dimensions     {}", report.dim);
    println!("  elapsed        {:.1}s", report.elapsed.as_secs_f64());
    println!("  size           {} MB", report.bytes / 1_048_576);
    println!("  embeddings     {}", report.path.display());
    println!("\njoin to the capture on sha256(review) = text_sha256");
    Ok(())
}

async fn run_crawl(app_id: u32, options: &CrawlOptions, pace: Duration) -> Result<()> {
    let client = SteamClient::new(pace)?;

    // A crawl of a large corpus runs for hours, so it is often piped to a log, where a
    // carriage-returned progress line becomes one unreadable smear.
    let interactive = std::io::stderr().is_terminal();

    eprintln!("crawling app {app_id}");
    let report = crawl(&client, app_id, options, |progress| {
        let line = format!(
            "  shard {}/{}  {} of {} reviews",
            progress.shards_done,
            progress.shards_total,
            thousands(progress.unique),
            thousands(progress.valve_total)
        );
        if interactive {
            let mut err = std::io::stderr();
            let _ = write!(err, "\r{line}   ");
            let _ = err.flush();
        } else if progress.shards_done.is_multiple_of(PROGRESS_EVERY_SHARDS) {
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
    println!("app {}  ({})", report.app_id, report.review_score_desc);
    if let Some(from) = report.top_up_from {
        println!("  mode         top-up, reviews created after {from}");
    } else if report.resumed {
        println!("  mode         resumed an unfinished crawl");
    }
    println!("  shards       {}", report.shards);
    println!("  pages        {}", report.pages);
    println!("  unique       {}", thousands(report.unique));
    println!(
        "  duplicates   {} (this run)",
        thousands(report.duplicates_this_run)
    );
    println!(
        "  valve total  {}  ({} up / {} down)",
        thousands(report.valve_total),
        thousands(report.valve_positive),
        thousands(report.valve_negative)
    );
    println!(
        "  coverage     {}",
        report.coverage().map_or_else(
            || "not applicable".to_owned(),
            |c| format!("{:.2}%", c * 100.0)
        )
    );
    println!("  elapsed      {:.1}s", report.elapsed.as_secs_f64());
    println!("  capture      {}", report.dir.display());

    if !report.complete {
        eprintln!(
            "\nwarning: some shards did not finish. Completed shards are on disk; \
             run the same command again to continue."
        );
    }
}

/// Prints each sampling subset with its contested and clear-cut slices indented beneath it.
///
/// Every rate carries a 95% interval, because the subsets are small enough that the naked
/// percentages invite conclusions the counts do not support.
fn print_slices(slices: &[census_core::Slice], measure: &str) {
    if slices.is_empty() {
        return;
    }
    println!("\n  primary {measure} by slice, with 95% intervals:");
    for slice in slices {
        let name = match slice.contested {
            None => slice.subset.clone(),
            Some(true) => "  contested".to_owned(),
            Some(false) => "  clear-cut".to_owned(),
        };
        let rate = slice
            .agreement()
            .map_or_else(|| "  n/a".to_owned(), |a| format!("{:.1}%", a * 100.0));
        let interval = slice.interval().map_or_else(String::new, |(low, high)| {
            format!("  [{:.1}, {:.1}]", low * 100.0, high * 100.0)
        });
        let note = match (slice.subset.as_str(), slice.contested) {
            ("random", None) => "  <- the only corpus-representative figure",
            ("stratified", None) => "  <- enriched for rare categories; also fitting data",
            _ => "",
        };
        println!(
            "    {name:<14} n={:<5} {rate:>6}{interval:<16}{note}",
            slice.compared
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
