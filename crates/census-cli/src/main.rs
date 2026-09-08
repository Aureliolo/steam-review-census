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

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum Precision {
    /// Smallest download, 112 MB. Slowest on a GPU, and its vectors shift with batching.
    Int8,
    /// 224 MB. Indistinguishable from fp32 and the fastest where a GPU exists.
    #[default]
    Fp16,
    /// 448 MB. The graph as exported, and the fastest of the three on CPU.
    Fp32,
}

impl From<Precision> for census_core::model::Precision {
    fn from(value: Precision) -> Self {
        match value {
            Precision::Int8 => Self::Int8,
            Precision::Fp16 => Self::Float16,
            Precision::Fp32 => Self::Float32,
        }
    }
}

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
        /// Which build of the model to run. fp16 matches the full graph and is fastest on
        /// a GPU; int8 is the smallest download but its vectors shift with batching.
        #[arg(long, default_value = "fp16")]
        precision: Precision,
        /// Where to cache the model. Defaults to the platform cache directory.
        #[arg(long)]
        model_dir: Option<PathBuf>,
    },

    /// Sort embedded reviews into the core-spine categories and report what players say.
    Classify(ClassifyArgs),

    /// Fit category anchors from a reference set, so categories are represented by reviews
    /// people wrote rather than by descriptions of the topic.
    Fit(FitArgs),

    /// Draw the reviews a reference set will be labelled from, reproducibly.
    Sample {
        /// Steam app IDs to sample across. Categories are pooled over all of them, so a
        /// category one game never discusses is supplied by a game that does.
        #[arg(required = true, num_args = 1..)]
        app_ids: Vec<u32>,
        /// Directory holding the captures and their classifications.
        #[arg(short, long, default_value = "data")]
        out: PathBuf,
        /// Uniformly drawn reviews per app. The only subset figures may be quoted from.
        #[arg(long, default_value_t = 100)]
        random: usize,
        /// Reviews per category, pooled across every app being sampled.
        #[arg(long, default_value_t = 40)]
        per_category: usize,
        /// Changing this draws a different sample. The same seed always draws the same one.
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Also write the category sheet labellers should work from.
        #[arg(long)]
        brief: bool,
        /// Split each sample into batches of this many reviews, ready to hand out.
        #[arg(long, default_value_t = 35)]
        batch_size: usize,
    },

    /// Merge returned labels into a reference set, checking them against the drawn sample.
    Ingest {
        /// Steam app ID whose labels are being merged.
        app_id: u32,
        /// Directory holding the returned label files, one JSON array per batch.
        #[arg(long)]
        from: PathBuf,
        /// Reference set directory. Defaults to the one for this app.
        #[arg(long)]
        reference: Option<PathBuf>,
        /// Write labels.json even when reviews are missing or labels are unusable.
        #[arg(long)]
        allow_incomplete: bool,
    },

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
struct ClassifyArgs {
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
    /// Anchors fitted by `census fit`. Defaults to a set fitted for this game if one exists,
    /// then to reference/anchors.json, then to the written category descriptions.
    #[arg(long)]
    anchors: Option<PathBuf>,
    /// Ignore any fitted anchors and use the written category descriptions.
    #[arg(long, conflicts_with = "anchors")]
    descriptions: bool,
    /// Which build of the model to run when descriptions must be embedded.
    #[arg(long, default_value = "fp16")]
    precision: Precision,
    /// Where to cache the model. Defaults to the platform cache directory.
    #[arg(long)]
    model_dir: Option<PathBuf>,
}

#[derive(clap::Args, Debug)]
struct FitArgs {
    /// Steam app IDs whose reference labels should be fitted, pooled together. Anchors built
    /// from several games carry less of any one game's vocabulary.
    #[arg(required = true, num_args = 1..)]
    app_ids: Vec<u32>,
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
    /// Where to write the fitted anchors. Defaults to reference/anchors.json.
    #[arg(long)]
    anchors_out: Option<PathBuf>,
    /// Measure transfer instead of writing anchors: fit on every game but one and report
    /// agreement on the game left out, for each game in turn.
    #[arg(long)]
    leave_one_out: bool,
    /// Where to cache the model. Defaults to the platform cache directory.
    #[arg(long)]
    model_dir: Option<PathBuf>,
    /// Which build of the model to run when descriptions must be embedded.
    #[arg(long, default_value = "fp16")]
    precision: Precision,
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
            precision,
        } => run_embed(app_id, &out, batch_size, model_dir, precision.into()).await,
        Command::Classify(args) => run_classify(args).await,
        Command::Fit(args) => run_fit(args).await,
        Command::Ingest {
            app_id,
            from,
            reference,
            allow_incomplete,
        } => run_ingest(app_id, &from, reference, allow_incomplete),
        Command::Sample {
            app_ids,
            out,
            random,
            per_category,
            seed,
            brief,
            batch_size,
        } => run_sample(
            &app_ids,
            &out,
            &census_core::SampleOptions {
                random_per_app: random,
                stratified_per_category: per_category,
                seed,
            },
            brief,
            batch_size,
        ),
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
        "\n{:<30} {:>8} {:>10} {:>8} {:>7}",
        "category", "in ref", "precision", "recall", "F1"
    );
    println!("{}", "-".repeat(68));
    for stat in &categories {
        let fmt = |v: Option<f64>| v.map_or_else(|| "    -".to_owned(), |x| format!("{x:.2}"));
        println!(
            "{:<30} {:>8} {:>10} {:>8} {:>7}",
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
/// Where anchors are looked for, best first: a set fitted for this game, then the set
/// shipped with the tool, which is fitted across several games and is what a game nobody
/// has labelled gets.
fn anchor_paths(app_id: u32) -> [PathBuf; 2] {
    [
        census_core::evaluate::default_reference_dir(app_id).join("anchors.json"),
        PathBuf::from("reference").join("anchors.json"),
    ]
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
    precision: census_core::model::Precision,
) -> Result<census_core::Anchors> {
    let found = anchor_paths(app_id).into_iter().find(|path| path.exists());
    let path = explicit.clone().or(found).unwrap_or_default();
    if !force_descriptions && (explicit.is_some() || path.exists()) {
        let anchors = census_core::Anchors::load(&path)?;
        if !anchors.fitted_from.is_empty() && !anchors.fitted_from.contains(&app_id) {
            eprintln!(
                "note: these anchors were fitted on {}, which does not include {app_id}. \
                 Categories carry the vocabulary of the games behind them; `census fit \
                 --leave-one-out` measures what that costs on a game left out.",
                anchors
                    .fitted_from
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        eprintln!("anchors      {}", path.display());
        return Ok(anchors);
    }

    let cache = model_dir.unwrap_or_else(census_core::model::default_cache_dir);
    census_core::model::ensure(&cache, precision, |_| {}).await?;
    let mut embedder = census_core::Embedder::load(&cache, precision)?;
    eprintln!(
        "anchors      written descriptions, on {}",
        embedder.device()
    );
    Ok(census_core::Anchors::from_descriptions(&mut embedder)?)
}

async fn run_classify(args: ClassifyArgs) -> Result<()> {
    let ClassifyArgs {
        app_id,
        out,
        mention_margin,
        top_helpful,
        anchors,
        descriptions,
        precision,
        model_dir,
    } = args;
    let anchors = load_anchors(app_id, anchors, descriptions, model_dir, precision.into()).await?;
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

fn run_ingest(
    app_id: u32,
    from: &std::path::Path,
    reference: Option<PathBuf>,
    allow_incomplete: bool,
) -> Result<()> {
    let dir = reference.unwrap_or_else(|| census_core::evaluate::default_reference_dir(app_id));
    let (labels, report) = census_core::sample::ingest(&dir, from)?;

    println!("app {app_id}");
    println!("  accepted     {}", report.accepted);
    let complain = |name: &str, items: &[String]| {
        if items.is_empty() {
            return;
        }
        let shown: Vec<&str> = items.iter().take(6).map(String::as_str).collect();
        let more = if items.len() > shown.len() {
            format!(" and {} more", items.len() - shown.len())
        } else {
            String::new()
        };
        println!("  {name:<12} {} ({}{more})", items.len(), shown.join(", "));
    };
    complain("missing", &report.missing);
    complain("not drawn", &report.unknown_reviews);
    complain("bad category", &report.unknown_categories);
    complain("duplicated", &report.duplicates);

    if !report.is_clean() && !allow_incomplete {
        anyhow::bail!(
            "labels do not cleanly cover the drawn sample; fix them, or pass              --allow-incomplete to write the set as it stands"
        );
    }
    let path = census_core::sample::write_labels(&dir, &labels)?;
    println!("  labels       {}", path.display());
    Ok(())
}

fn run_sample(
    app_ids: &[u32],
    out: &std::path::Path,
    options: &census_core::SampleOptions,
    brief: bool,
    batch_size: usize,
) -> Result<()> {
    let (drawn, reports) = census_core::sample::draw(out, app_ids, options)?;

    for report in &reports {
        let dir = census_core::evaluate::default_reference_dir(report.app_id);
        census_core::sample::write_for_app(&dir, report.app_id, &drawn)?;
        let batches = census_core::sample::write_batches(&dir, report.app_id, &drawn, batch_size)?;

        let thin: Vec<&str> = report
            .per_category
            .iter()
            .filter(|(_, n)| *n == 0)
            .map(|(id, _)| *id)
            .collect();
        println!(
            "app {:<9} corpus {:>9}  random {:>4}  stratified {:>4}  -> {}",
            report.app_id,
            thousands(report.corpus),
            report.random,
            report.stratified,
            dir.join("sample.json").display()
        );
        if batches > 0 {
            println!("    {batches} batches in {}", dir.join("batches").display());
        }
        if !thin.is_empty() {
            println!("    supplies nothing for: {}", thin.join(", "));
        }
    }

    if brief {
        let path = std::path::Path::new("reference").join("labelling-brief.txt");
        std::fs::write(&path, census_core::taxonomy::labelling_brief())?;
        println!("brief    {}", path.display());
    }

    let short: Vec<String> = census_core::CORE_SPINE
        .iter()
        .filter_map(|category| {
            let total: usize = reports
                .iter()
                .flat_map(|r| &r.per_category)
                .filter(|(id, _)| *id == category.id)
                .map(|(_, n)| *n)
                .sum();
            (total < options.stratified_per_category).then(|| format!("{} ({total})", category.id))
        })
        .collect();
    if !short.is_empty() {
        println!(
            "\nBelow target across every corpus, so these stay anchored on their written\n\
             descriptions rather than on reviews: {}",
            short.join(", ")
        );
    }
    Ok(())
}

/// Everything one game contributes to a fit: the examples it trains on, the ones held back
/// from it, and what an average review of it looks like.
struct GameLabels {
    app_id: u32,
    training: Vec<census_core::anchors::Example>,
    holdout: Vec<census_core::anchors::Example>,
    centroid: Vec<f32>,
}

fn load_game(
    app_id: u32,
    out: &std::path::Path,
    reference: Option<&PathBuf>,
    holdout: &str,
) -> Result<GameLabels> {
    let dir = reference
        .cloned()
        .unwrap_or_else(|| census_core::evaluate::default_reference_dir(app_id));
    let set = census_core::ReferenceSet::load(&dir)?;
    if set.spine_version != census_core::CORE_SPINE_VERSION {
        anyhow::bail!(
            "app {app_id} was labelled against taxonomy {} but this build is {}; fitting from \
             it would move every category boundary towards a taxonomy that no longer exists.",
            set.spine_version,
            census_core::CORE_SPINE_VERSION
        );
    }

    // Only the labelled reviews are fetched. Pulling every vector in the corpus to use a few
    // hundred of them costs gigabytes on a million-review game and buys nothing.
    let wanted: std::collections::HashSet<String> =
        set.labels.iter().map(|label| label.id.clone()).collect();
    let vectors = census_core::embed::vectors_for(out, app_id, &wanted)?;

    let split = |held: bool| -> Vec<census_core::evaluate::ReferenceLabel> {
        set.labels
            .iter()
            .filter(|label| (label.subset == holdout) == held)
            .cloned()
            .collect()
    };
    Ok(GameLabels {
        app_id,
        training: census_core::anchors::to_examples(&split(false), &vectors),
        holdout: census_core::anchors::to_examples(&split(true), &vectors),
        centroid: census_core::embed::corpus_centroid(out, app_id)?,
    })
}

/// The mean of the per-game centroids.
///
/// Each game counts once rather than once per review, so a corpus of a million does not
/// decide by itself what an average review looks like to a set meant to serve all of them.
fn pooled_centroid(games: &[GameLabels]) -> Vec<f32> {
    let mut total = vec![0.0_f32; census_core::EMBEDDING_DIM];
    for game in games {
        for (slot, value) in total.iter_mut().zip(&game.centroid) {
            *slot += *value;
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "a handful of games")]
    let count = games.len().max(1) as f32;
    for value in &mut total {
        *value /= count;
    }
    total
}

async fn run_fit(args: FitArgs) -> Result<()> {
    let FitArgs {
        app_ids,
        out,
        reference,
        holdout,
        folds,
        calibrate,
        anchors_out,
        model_dir,
        precision,
        leave_one_out,
    } = args;
    let holdout = holdout.as_str();
    if reference.is_some() && app_ids.len() > 1 {
        anyhow::bail!("--reference names one directory, so it cannot be used with several apps");
    }

    let games: Vec<GameLabels> = app_ids
        .iter()
        .map(|&app_id| load_game(app_id, &out, reference.as_ref(), holdout))
        .collect::<Result<_>>()?;
    if games.iter().all(|game| game.training.is_empty()) {
        anyhow::bail!(census_core::Error::NoTrainingExamples { app_id: app_ids[0] });
    }

    let cache = model_dir.unwrap_or_else(census_core::model::default_cache_dir);
    let precision = precision.into();
    census_core::model::ensure(&cache, precision, |_| {}).await?;
    let mut embedder = census_core::Embedder::load(&cache, precision)?;
    let descriptions = census_core::Anchors::from_descriptions(&mut embedder)?;

    if leave_one_out {
        report_transfer(&descriptions, &games, folds, calibrate, holdout);
        return Ok(());
    }

    let examples: Vec<census_core::anchors::Example> = games
        .iter()
        .flat_map(|game| game.training.clone())
        .collect();
    let centroid = pooled_centroid(&games);
    eprintln!(
        "fitting {} labels from {} game(s), holding back the {holdout} subset",
        examples.len(),
        games.len()
    );
    let outcome =
        census_core::anchors::search(&descriptions, &examples, &centroid, folds, calibrate.into());
    let mut fitted = descriptions.fit(&examples, &app_ids, outcome.params);
    if outcome.params.calibrate {
        fitted.calibrate(&centroid);
    }

    let path = anchors_out.unwrap_or_else(|| PathBuf::from("reference").join("anchors.json"));
    fitted.save(&path)?;
    let held: usize = games.iter().map(|game| game.holdout.len()).sum();
    print_fit(&fitted, &outcome, &path, holdout, held);
    Ok(())
}

/// Fits on every game but one and measures on the game left out, for each in turn.
///
/// This is the only figure that says whether anchors carry to a game they were never built
/// from. Fitting and measuring on the same title, even across a held-back subset, still
/// shares that title's vocabulary between the two halves.
fn report_transfer(
    descriptions: &census_core::Anchors,
    games: &[GameLabels],
    folds: usize,
    calibrate: Calibrate,
    holdout: &str,
) {
    println!("leave-one-game-out, measured on each game's {holdout} subset\n");
    println!(
        "{:<10}{:>7}{:>14}{:>12}{:>13}{:>16}",
        "held out", "n", "descriptions", "same game", "unseen game", "gained / lost"
    );
    println!("{}", "-".repeat(72));

    let mut totals = [0_u64; 6];
    for (index, game) in games.iter().enumerate() {
        if game.holdout.is_empty() {
            continue;
        }
        let others: Vec<census_core::anchors::Example> = games
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != index)
            .flat_map(|(_, game)| game.training.clone())
            .collect();
        let all: Vec<census_core::anchors::Example> = games
            .iter()
            .flat_map(|game| game.training.clone())
            .collect();

        let fit_on = |examples: &[census_core::anchors::Example]| {
            let outcome = census_core::anchors::search(
                descriptions,
                examples,
                &game.centroid,
                folds,
                calibrate.into(),
            );
            let mut anchors = descriptions.fit(examples, &[game.app_id], outcome.params);
            if outcome.params.calibrate {
                anchors.calibrate(&game.centroid);
            }
            anchors
        };

        let written = census_core::anchors::agreements(descriptions, &game.holdout);
        let same = census_core::anchors::agreements(&fit_on(&all), &game.holdout);
        let unseen = census_core::anchors::agreements(&fit_on(&others), &game.holdout);
        let n = game.holdout.len() as u64;
        let count = |verdicts: &[bool]| verdicts.iter().filter(|hit| **hit).count() as u64;
        let (gained, lost) = swapped(&written, &unseen);

        totals[0] += count(&written);
        totals[1] += count(&same);
        totals[2] += count(&unseen);
        totals[3] += n;
        totals[4] += gained;
        totals[5] += lost;
        println!(
            "{:<10}{n:>7}{:>14}{:>12}{:>13}{:>16}",
            game.app_id,
            share(count(&written), n),
            share(count(&same), n),
            share(count(&unseen), n),
            format!("+{gained} / -{lost}")
        );
    }

    println!("{}", "-".repeat(72));
    println!(
        "{:<10}{:>7}{:>14}{:>12}{:>13}{:>16}",
        "all",
        totals[3],
        share(totals[0], totals[3]),
        share(totals[1], totals[3]),
        share(totals[2], totals[3]),
        format!("+{} / -{}", totals[4], totals[5])
    );
    println!(
        "\n\"same game\" fits on every game including this one, so it shares this title's\n\
         vocabulary between fitting and measuring. \"unseen game\" fits on the other games\n\
         only, and is the figure that says whether anchors transfer."
    );
    match census_core::evaluate::mcnemar_exact(totals[4], totals[5]) {
        Some(p) => println!(
            "Of the {} reviews the written descriptions and the unseen-game fit place \
             differently,\n{} move to the category the labels give and {} move away. \
             Exact McNemar {}.",
            totals[4] + totals[5],
            totals[4],
            totals[5],
            if p < 0.0001 {
                "p < 0.0001".to_owned()
            } else {
                format!("p = {p:.4}")
            }
        ),
        None => println!("Neither set placed a single review differently from the other."),
    }
}

/// Reviews the second set gets right and the first does not, and the reverse.
fn swapped(before: &[bool], after: &[bool]) -> (u64, u64) {
    let mut gained = 0;
    let mut lost = 0;
    for (was, now) in before.iter().zip(after) {
        match (was, now) {
            (false, true) => gained += 1,
            (true, false) => lost += 1,
            _ => {}
        }
    }
    (gained, lost)
}

fn apps(ids: &[u32]) -> String {
    if ids.is_empty() {
        return "no labels at all".to_owned();
    }
    ids.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

#[expect(clippy::cast_precision_loss, reason = "reference sets are hundreds")]
fn share(part: u64, whole: u64) -> String {
    if whole == 0 {
        return "n/a".to_owned();
    }
    format!("{:.1}%", part as f64 / whole as f64 * 100.0)
}

fn print_fit(
    anchors: &census_core::Anchors,
    outcome: &census_core::FitOutcome,
    path: &std::path::Path,
    holdout: &str,
    held_back: usize,
) {
    println!("anchors from {}", apps(&anchors.fitted_from));
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
        if report.anchors_fitted_from.is_empty() {
            "written descriptions (unfitted)".to_owned()
        } else {
            format!("fitted on {}", apps(&report.anchors_fitted_from))
        }
    );
    println!("  elapsed      {:.1}s", report.elapsed.as_secs_f64());
    println!("  assignments  {}", report.path.display());

    let mut categories = report.categories.clone();
    categories.sort_by(|a, b| {
        b.mention_rate(report.reviews)
            .total_cmp(&a.mention_rate(report.reviews))
    });

    println!(
        "\n{:<30} {:>9} {:>9} {:>9} {:>7}",
        "category", "mention%", "primary%", "top50%", "bias"
    );
    println!("{}", "-".repeat(68));
    for stat in &categories {
        let bias = stat
            .bias_factor(report.reviews, report.top_helpful)
            .map_or_else(|| "    -".to_owned(), |b| format!("{b:.2}x"));
        println!(
            "{:<30} {:>8.1}% {:>8.1}% {:>8.1}% {:>7}",
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
    precision: census_core::model::Precision,
) -> Result<()> {
    let cache = model_dir.unwrap_or_else(census_core::model::default_cache_dir);
    let interactive = std::io::stderr().is_terminal();

    eprintln!("model cache: {}", cache.display());
    let mut announced = String::new();
    census_core::model::ensure(&cache, precision, |progress| {
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

    let mut embedder = census_core::Embedder::load(&cache, precision)?;
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
    if report.shards_restarted > 0 {
        println!(
            "  restarted    {} (steam stopped serving early; walked again)",
            report.shards_restarted
        );
    }

    if report.shards_short > 0 {
        eprintln!(
            "\nwarning: {} window(s) stayed short of Valve's own count after every attempt \
             and were left unfinished rather than counted. Run the same command again to \
             walk them.",
            report.shards_short
        );
    } else if !report.complete {
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
