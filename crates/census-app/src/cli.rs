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
    /// Indistinguishable from fp32 and the fastest where a GPU exists. Half the download.
    #[default]
    Fp16,
    /// The graph as exported, and the faster of the two on CPU.
    Fp32,
}

impl From<Precision> for census_core::model::Precision {
    fn from(value: Precision) -> Self {
        match value {
            Precision::Fp16 => Self::Float16,
            Precision::Fp32 => Self::Float32,
        }
    }
}

/// Which encoder turns a review into a vector. Vectors from two encoders are not
/// comparable, so changing this means re-embedding the corpus.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum Model {
    /// multilingual-e5-small. 384 dimensions, MIT, the smallest and fastest of the four.
    E5Small,
    /// multilingual-e5-base. 768 dimensions, MIT.
    E5Base,
    /// snowflake-arctic-embed-m-v2.0. 768 dimensions, Apache-2.0.
    ArcticM2,
    /// gte-multilingual-base. 768 dimensions, Apache-2.0. The most accurate of the four,
    /// measured leave-one-game-out against the reference sets.
    #[default]
    GteBase,
}

impl From<Model> for census_core::Encoder {
    fn from(value: Model) -> Self {
        match value {
            Model::E5Small => Self::E5Small,
            Model::E5Base => Self::E5Base,
            Model::ArcticM2 => Self::ArcticMediumV2,
            Model::GteBase => Self::GteBase,
        }
    }
}

/// What the corpus mean is used for before categories compete for a review.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum Scoring {
    /// Let cross-validation pick, which is what a corpus nobody has swept by hand wants.
    Search,
    /// Plain cosine similarity to each anchor.
    Raw,
    /// Each category scored by how far above its own corpus average a review sits.
    Bias,
    /// The corpus mean removed from both sides before they are compared.
    Centred,
}

impl From<Scoring> for census_core::anchors::Calibration {
    fn from(value: Scoring) -> Self {
        match value {
            Scoring::Search => Self::Search,
            Scoring::Raw => Self::Only(census_core::anchors::Scoring::Raw),
            Scoring::Bias => Self::Only(census_core::anchors::Scoring::Bias),
            Scoring::Centred => Self::Only(census_core::anchors::Scoring::Centred),
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Select {
    Mentions,
    Primary,
}

impl From<Select> for census_core::anchors::Objective {
    fn from(value: Select) -> Self {
        match value {
            Select::Mentions => Self::Mentions,
            Select::Primary => Self::Primary,
        }
    }
}

/// How often to emit a progress line when stderr is not a terminal.
/// What a pass works on. Named separately from the library's own type because clap owns the
/// spelling a person types and the library owns the one the code reads.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum Grain {
    /// One point a review makes.
    #[default]
    Claim,
    /// A whole review, averaged over everything it says.
    Review,
}

impl From<Grain> for census_core::taxonomy::Unit {
    fn from(value: Grain) -> Self {
        match value {
            Grain::Claim => Self::Claim,
            Grain::Review => Self::Review,
        }
    }
}

const PROGRESS_EVERY_SHARDS: usize = 5;
const PROGRESS_EVERY_TEXTS: u64 = 5_000;

#[derive(Parser, Debug)]
#[command(
    name = "census",
    version,
    about = "Count what Steam reviewers actually say, rather than what the loudest ones say.",
    after_help = "The first four commands are the pipeline, in order: crawl a game, embed it, \
                  classify it, read it. The rest are how the classifier is measured and \
                  improved, and none of them is needed to get a report out.\n\n\
                  Nothing leaves this machine. Reviews are downloaded from Valve and \
                  everything after that happens locally, including the model."
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

    /// Split every review into the separate points it makes.
    ///
    /// A review is not one opinion, and a whole-review vector is the average of the ones it
    /// holds. This writes claims.parquet beside the capture, carrying offsets rather than
    /// text so the corpus is not stored twice.
    Claims {
        /// Steam app ID, as it appears in the store URL.
        app_id: u32,
        /// Directory holding the capture.
        #[arg(short, long, default_value = "data")]
        out: PathBuf,
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
        /// Which encoder to embed with. A corpus records this, and vectors from two
        /// encoders are never compared.
        #[arg(long, default_value = "gte-base")]
        model: Model,
        /// Which build of the graph to run. fp16 matches the full graph and is fastest on a
        /// GPU; fp32 is the faster of the two on CPU.
        #[arg(long, default_value = "fp16")]
        precision: Precision,
        /// Where to cache the model. Defaults to the platform cache directory.
        #[arg(long)]
        model_dir: Option<PathBuf>,
        /// What to embed: whole reviews, or the separate points they make.
        #[arg(long, default_value = "claim")]
        unit: Grain,
    },

    /// Draw reviews at random, split them into claims, and write batches to be labelled.
    ///
    /// Every claim of every drawn review is labelled, never a subset of them: a review
    /// labelled in part cannot say what share of a corpus names no aspect at all, which is
    /// the first thing worth knowing about one.
    SampleClaims {
        /// Steam app IDs to draw from. Each gets its own set.
        #[arg(required = true, num_args = 1..)]
        app_ids: Vec<u32>,
        /// Directory holding the capture.
        #[arg(short, long, default_value = "data")]
        out: PathBuf,
        /// Reviews to draw.
        #[arg(long, default_value_t = 700)]
        reviews: usize,
        /// Reviews per batch file. Around forty is roughly a hundred and thirty claims.
        #[arg(long, default_value_t = 40)]
        batch_size: usize,
        /// Changing this draws a different sample. The same seed always draws the same one.
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Share of each game's draw written in English. The rest is drawn from whatever
        /// else the corpus holds, so the model is trained on more than one language even
        /// though reports default to English.
        #[arg(long, default_value_t = 0.7)]
        english: f64,
        /// Where to write the sets. Each game gets a directory under it.
        #[arg(long, default_value = "reference/claims")]
        to: PathBuf,
    },

    /// Merge returned claim labels into a reference set.
    IngestClaims {
        /// Steam app ID whose labels are being merged.
        app_id: u32,
        /// Directory holding the returned label files, one JSON array per batch.
        #[arg(long)]
        from: PathBuf,
        /// The reference set. Defaults to reference/claims/<app id>.
        #[arg(long)]
        to: Option<PathBuf>,
    },

    /// Write every labelled claim, with its text, as JSONL for training.
    ///
    /// The file holds review text and is not for publishing. What gets published is the
    /// label set: ids, offsets and labels, which anyone can rehydrate with this tool.
    ExportTraining {
        /// Where the claim reference sets live.
        #[arg(long, default_value = "reference/claims")]
        from: PathBuf,
        /// Where to write the JSONL.
        #[arg(long, default_value = "training/data/claims.jsonl")]
        to: PathBuf,
    },

    /// Read every claim in a corpus with the trained model.
    Read {
        /// Steam app ID whose most recent capture should be read.
        app_id: u32,
        /// Directory holding the capture.
        #[arg(short, long, default_value = "data")]
        out: PathBuf,
        /// Directory holding model.onnx, tokenizer.json and reader.json. Defaults to
        /// models/claim-reader in the working tree if there is one, else the platform cache.
        #[arg(long)]
        model: Option<PathBuf>,
        /// Claims per forward pass.
        #[arg(long, default_value_t = census_core::read::DEFAULT_READ_BATCH)]
        batch_size: usize,
        /// Count only reviews written in this language. The capture stays whole either way.
        #[arg(long)]
        language: Option<String>,
        /// How many of the most-helpful reviews count as the top of the pile.
        #[arg(long, default_value_t = census_core::classify::DEFAULT_TOP_HELPFUL)]
        top_helpful: usize,
    },

    /// Sort embedded reviews into the core-spine categories and report what players say.
    Classify(ClassifyArgs),

    /// Render a self-contained page from what a classification found.
    Report {
        /// Steam app IDs to report on. Several become one page, a section each.
        #[arg(required = true, num_args = 1..)]
        app_ids: Vec<u32>,
        /// Directory holding the captures and their classifications.
        #[arg(short, long, default_value = "data")]
        out: PathBuf,
        /// Where to write the page.
        #[arg(long, default_value = "census-report.html")]
        to: PathBuf,
        /// Reviews quoted per category, as the evidence behind its rate.
        #[arg(long, default_value_t = census_core::report::DEFAULT_EXAMPLES)]
        examples: usize,
        /// Changing this quotes different reviews. The same seed always quotes the same ones.
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },

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
        /// Split each sample into batches of this many reviews, ready to hand out.
        #[arg(long, default_value_t = 35)]
        batch_size: usize,
    },

    /// Write the category sheet labellers work from, generated from the taxonomy.
    ///
    /// Separate from `sample` because a boundary rule can change without anything needing to
    /// be drawn again, and regenerating the sheet should never mean redrawing a sample.
    Brief {
        /// Where to write it. Both sheets are written when this is a directory.
        #[arg(long, default_value = "reference")]
        to: PathBuf,
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

    /// Fit category anchors from labelled reviews, so a category is what people wrote about it
    ///
    /// A written description of a topic is prose about that topic and reviews about it are
    /// not, so a category with labelled examples is defined by them and one without falls
    /// back to its description.
    Fit(FitArgs),

    /// Compare stored classifications against a reference set.
    Evaluate {
        /// Steam app IDs to evaluate. Several are reported one by one and then pooled,
        /// which is the figure worth quoting: one game's hundred held-out reviews carry a
        /// band twenty points wide.
        #[arg(required = true, num_args = 1..)]
        app_ids: Vec<u32>,
        /// Directory holding the capture and its classifications.
        #[arg(short, long, default_value = "data")]
        out: PathBuf,
        /// Reference set directory, holding manifest.json and labels.json.
        #[arg(long)]
        reference: Option<PathBuf>,
        /// Write the comparison as JSON instead of tables, so a manifest can quote figures
        /// nobody retyped.
        #[arg(long)]
        json: bool,
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
    /// How many of the reviews Steam ranks most helpful count as "the top of the pile".
    #[arg(long, default_value_t = census_core::classify::DEFAULT_TOP_HELPFUL)]
    top_helpful: usize,
    /// Anchors fitted by `census fit`. Defaults to a set fitted for this game if one exists,
    /// then to reference/anchors.json, then to the written category descriptions.
    #[arg(long)]
    anchors: Option<PathBuf>,
    /// Ignore any fitted anchors and use the written category descriptions.
    #[arg(long, conflicts_with = "anchors")]
    descriptions: bool,
    /// Which encoder to embed the descriptions with. Must be the encoder the corpus was
    /// embedded with, and is checked against it.
    #[arg(long, default_value = "gte-base")]
    model: Model,
    /// Which build of the graph to run when descriptions must be embedded.
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
    /// What the corpus mean is used for before categories compete. `search` lets
    /// cross-validation decide, and records what it chose beside the anchors.
    #[arg(long, default_value = "search")]
    scoring: Scoring,
    /// What the blend is chosen to win. `mentions` weighs every category the same, which is
    /// what a report giving each one a row needs; `primary` maximises how often the main
    /// subject is right, which the largest few categories decide.
    #[arg(long, default_value = "mentions")]
    select: Select,
    /// Where to write the fitted anchors. Defaults to reference/anchors.json.
    #[arg(long)]
    anchors_out: Option<PathBuf>,
    /// Measure transfer instead of writing anchors: fit on every game but one and report
    /// agreement on the game left out, for each game in turn.
    #[arg(long)]
    leave_one_out: bool,
    /// Run the transfer measurement a second time under another encoder and compare the two
    /// review by review, which is the only way to tell a real difference between encoders
    /// from two intervals that happen to overlap.
    #[arg(long, requires = "leave_one_out")]
    compare: Option<Model>,
    /// Where to cache the model. Defaults to the platform cache directory.
    #[arg(long)]
    model_dir: Option<PathBuf>,
    /// Which encoder to fit under. Naming an encoder the corpus was not embedded with makes
    /// the fit embed the labelled reviews itself, so a candidate encoder can be measured
    /// without re-embedding millions of reviews first.
    #[arg(long, default_value = "gte-base")]
    model: Model,
    /// Which build of the graph to run.
    #[arg(long, default_value = "fp16")]
    precision: Precision,
}

pub async fn run() -> Result<()> {
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
        Command::Claims { app_id, out } => run_claims(app_id, &out),
        Command::SampleClaims {
            app_ids,
            out,
            reviews,
            batch_size,
            seed,
            english,
            to,
        } => run_sample_claims(&app_ids, &out, reviews, batch_size, seed, english, &to),
        Command::IngestClaims { app_id, from, to } => run_ingest_claims(app_id, &from, to),
        Command::Read {
            app_id,
            out,
            model,
            batch_size,
            language,
            top_helpful,
        } => run_read(
            app_id,
            &model.unwrap_or_else(census_core::reader::default_dir),
            &census_core::read::ReadOptions {
                out_dir: out,
                top_helpful,
                batch_size,
                language,
            },
        ),
        Command::ExportTraining { from, to } => {
            let written = census_core::claimset::export_training(&from, &to)?;
            println!("{written} labelled claims -> {}", to.display());
            Ok(())
        }
        Command::Embed {
            app_id,
            out,
            batch_size,
            model,
            model_dir,
            precision,
            unit,
        } => {
            run_embed(
                app_id,
                &out,
                batch_size,
                model_dir,
                model.into(),
                precision.into(),
                unit.into(),
            )
            .await
        }
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
            batch_size,
        } => run_sample(
            &app_ids,
            &out,
            &census_core::SampleOptions {
                random_per_app: random,
                stratified_per_category: per_category,
                seed,
            },
            batch_size,
        ),
        Command::Brief { to } => run_brief(&to),
        Command::Report {
            app_ids,
            out,
            to,
            examples,
            seed,
        } => run_report(&app_ids, &out, &to, examples, seed),
        Command::Evaluate {
            app_ids,
            out,
            reference,
            json,
        } => run_evaluate(&app_ids, &out, reference.as_ref(), json),
    }
}

fn run_report(
    app_ids: &[u32],
    out: &std::path::Path,
    to: &std::path::Path,
    examples: usize,
    seed: u64,
) -> Result<()> {
    // Every game before any of them: a page of six corpora that dies on the sixth has read
    // five of them and quoted a thousand reviews to say what one look would have said.
    for &app_id in app_ids {
        census_core::embed::latest_snapshot(out, app_id)?;
    }
    let options = census_core::report::ReportOptions {
        out_dir: out.to_path_buf(),
        examples,
        seed,
    };
    let report = census_core::report::build(app_ids, &options)?;
    let page = census_core::html::render(&report);
    std::fs::write(to, page.as_bytes())?;

    println!("report      {}", to.display());
    println!("  games      {}", report.apps.len());
    println!("  size       {} KB", page.len() / 1024);
    for app in &report.apps {
        let quoted: usize = app.examples.iter().map(|(_, e)| e.len()).sum();
        println!(
            "  {:<10} {} reviews, {quoted} quoted",
            app.app_id(),
            thousands(app.classification.reviews)
        );
    }
    println!(
        "\nSelf-contained: open it from disk, send it as one file, print it. Nothing in it is\n\
         fetched from anywhere, and no review left this machine to produce it."
    );
    Ok(())
}

fn run_evaluate(
    app_ids: &[u32],
    out: &std::path::Path,
    reference: Option<&PathBuf>,
    json: bool,
) -> Result<()> {
    if reference.is_some() && app_ids.len() > 1 {
        anyhow::bail!("--reference names one directory, so it cannot be used with several apps");
    }
    // Every game is compared before any of them is printed. Printing as they arrived meant a
    // set missing its fourth reference left three tables on screen and no pooled figure,
    // which is the one worth quoting and the reason for naming several games at once.
    let mut reports = Vec::with_capacity(app_ids.len());
    let mut verified = true;
    for &app_id in app_ids {
        let (report, human_verified) = evaluate_one(app_id, out, reference)?;
        verified &= human_verified;
        reports.push((report, human_verified));
    }
    if !json {
        for (index, (report, human_verified)) in reports.iter().enumerate() {
            if index > 0 {
                println!();
            }
            print_agreement(report, *human_verified);
        }
    }
    let reports: Vec<census_core::AgreementReport> =
        reports.into_iter().map(|(report, _)| report).collect();
    let pooled = (reports.len() > 1).then(|| census_core::evaluate::pooled(&reports));
    if json {
        // A manifest quoting figures read off a terminal block is a manifest with a typo in
        // it, and one has already reached a shipped file. Thirty-six of them is thirty-six
        // chances at the same mistake.
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "human_verified": verified,
                "apps": reports,
                "pooled": pooled,
            }))?
        );
        return Ok(());
    }
    if let Some(pooled) = pooled {
        println!("\npooled over {} games", reports.len());
        print_agreement(&pooled, verified);
    }
    Ok(())
}

fn evaluate_one(
    app_id: u32,
    out: &std::path::Path,
    reference: Option<&PathBuf>,
) -> Result<(census_core::AgreementReport, bool)> {
    let dir = reference
        .cloned()
        .unwrap_or_else(|| census_core::evaluate::default_reference_dir(app_id));
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

    Ok((census_core::compare(&set, out, app_id)?, set.human_verified))
}

fn print_agreement(report: &census_core::AgreementReport, human_verified: bool) {
    let measure = if human_verified {
        "accuracy"
    } else {
        "agreement"
    };
    println!("app {}", apps(&report.apps));
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
        "\n{:<30} {:>8} {:>10} {:>8} {:>7}   most often read as",
        "category", "in ref", "precision", "recall", "F1"
    );
    println!("{}", "-".repeat(96));
    for stat in &categories {
        let fmt = |v: Option<f64>| v.map_or_else(|| "    -".to_owned(), |x| format!("{x:.2}"));
        // Trimmed, because most rows have nothing in the last column and a line of trailing
        // spaces is something anyone piping this to a file has to strip back out.
        let row = format!(
            "{:<30} {:>8} {:>10} {:>8} {:>7}   {}",
            stat.label,
            stat.reference_mentions,
            fmt(stat.precision()),
            fmt(stat.recall()),
            fmt(stat.f1()),
            stat.mistaken_for()
                .map_or_else(String::new, |(label, count)| format!("{label} ({count})"))
        );
        println!("{}", row.trim_end());
    }
    println!(
        "\n\"Most often read as\" counts only the main subject, and only where the classifier\n\
         and the labels disagree. A category read as one particular other category is a\n\
         boundary the taxonomy has not settled, which no amount of fitting will settle for it."
    );

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

/// Loads fitted anchors if there are any, falling back to embedding the descriptions.
///
/// Only the fallback needs the embedding model, so classifying with a fitted set never
/// downloads one.
async fn load_anchors(
    app_id: u32,
    explicit: Option<PathBuf>,
    force_descriptions: bool,
    model_dir: Option<PathBuf>,
    encoder: census_core::Encoder,
    precision: census_core::model::Precision,
) -> Result<census_core::Anchors> {
    if !force_descriptions {
        let anchors = match explicit {
            Some(ref path) => census_core::Anchors::load(path)?,
            None => census_core::Anchors::for_app(app_id)?,
        };
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
        eprintln!(
            "anchors      {}",
            explicit.as_ref().map_or_else(
                || format!("fitted on {} games, compiled in", anchors.fitted_from.len()),
                |path| path.display().to_string()
            )
        );
        return Ok(anchors);
    }

    let cache = model_dir.unwrap_or_else(census_core::model::default_cache_dir);
    census_core::model::ensure(&cache, encoder, precision, |_| {}).await?;
    let mut embedder = census_core::Embedder::load(&cache, encoder, precision)?;
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
        model,
        precision,
        model_dir,
    } = args;
    // Before the encoder, which is several hundred megabytes and a wait, and which a game
    // that was never crawled has no use for.
    census_core::embed::latest_snapshot(&out, app_id)?;
    let anchors = load_anchors(
        app_id,
        anchors,
        descriptions,
        model_dir,
        model.into(),
        precision.into(),
    )
    .await?;
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
    complain("unjudged", &report.unjudged);

    if !report.is_clean() && !allow_incomplete {
        anyhow::bail!(
            "labels do not cleanly cover the drawn sample; fix them, or pass \
             --allow-incomplete to write the set as it stands"
        );
    }
    let path = census_core::sample::write_labels(&dir, &labels)?;
    println!("  labels       {}", path.display());
    let was = census_core::sample::record_taxonomy(&dir, app_id)?;
    if was != census_core::CORE_SPINE_VERSION {
        println!(
            "  taxonomy     {} (was {})",
            census_core::CORE_SPINE_VERSION,
            if was.is_empty() { "unset" } else { &was }
        );
    }
    Ok(())
}

fn run_sample(
    app_ids: &[u32],
    out: &std::path::Path,
    options: &census_core::SampleOptions,
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
    /// What to call it in a table. Thirty-six rows of nine digits is not a table anyone
    /// reads, and the name is already in the capture.
    name: String,
    training: Vec<census_core::anchors::Example>,
    holdout: Vec<census_core::anchors::Example>,
    /// Whether the labeller called each held-back review contested, in the same order.
    ///
    /// Carried alongside rather than inside the example, because fitting has no use for it:
    /// it is a property of the measurement, and the measurement is the only thing that
    /// splits by it.
    contested: Vec<bool>,
    centroid: Vec<f32>,
}

/// Where a reference set's vectors come from.
///
/// The corpus's own vectors are free and are what classification uses. Embedding the labels
/// afresh is for judging an encoder the corpus was never built with, and it gives up the
/// corpus centroid: a thousand labelled reviews are not what an average review looks like,
/// so the scoring falls back to plain similarity rather than being fed a mean that is not
/// the one it wants.
enum Vectors<'a> {
    Stored,
    Fresh(&'a mut census_core::Embedder),
}

fn load_game(
    app_id: u32,
    out: &std::path::Path,
    reference: Option<&PathBuf>,
    holdout: &str,
    source: Vectors<'_>,
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
    let (vectors, centroid) = match source {
        Vectors::Stored => (
            census_core::embed::vectors_for(out, app_id, &wanted)?,
            census_core::embed::corpus_centroid(out, app_id)?,
        ),
        Vectors::Fresh(embedder) => (
            census_core::embed::embed_reviews(embedder, out, app_id, &wanted, DEFAULT_BATCH_SIZE)?,
            Vec::new(),
        ),
    };

    let split = |held: bool| -> Vec<census_core::evaluate::ReferenceLabel> {
        set.labels
            .iter()
            .filter(|label| (label.subset == holdout) == held)
            .cloned()
            .collect()
    };
    let holdout_labels = split(true);
    let held = census_core::anchors::labelled(&holdout_labels, &vectors);
    Ok(GameLabels {
        app_id,
        // A capture always carries one, but a fit that cannot read a name is still a fit.
        name: census_core::report::crawl_facts(out, app_id)
            .map_or_else(|_| format!("app {app_id}"), |facts| facts.title()),
        training: census_core::anchors::to_examples(&split(false), &vectors),
        contested: held
            .iter()
            .map(|(_, label)| label.ambiguous)
            .collect::<Vec<bool>>(),
        holdout: held.into_iter().map(|(example, _)| example).collect(),
        centroid,
    })
}

/// The mean of the per-game centroids.
///
/// Each game counts once rather than once per review, so a corpus of a million does not
/// decide by itself what an average review looks like to a set meant to serve all of them.
fn pooled_centroid(games: &[GameLabels]) -> Vec<f32> {
    let width = games.iter().map(|game| game.centroid.len()).max();
    let Some(width) = width.filter(|width| *width > 0) else {
        return Vec::new();
    };
    let mut total = vec![0.0_f32; width];
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

/// Everything a fit needs that does not depend on which encoder is running.
struct FitInputs<'a> {
    app_ids: &'a [u32],
    out: &'a std::path::Path,
    reference: Option<&'a PathBuf>,
    holdout: &'a str,
    cache: &'a std::path::Path,
    precision: census_core::model::Precision,
    scoring: Scoring,
    select: Select,
}

/// A reference set as one encoder sees it.
struct Under {
    encoder: census_core::Encoder,
    descriptions: census_core::Anchors,
    games: Vec<GameLabels>,
    scoring: Scoring,
    select: Select,
}

async fn load_under(model: Model, inputs: &FitInputs<'_>) -> Result<Under> {
    let encoder: census_core::Encoder = model.into();
    census_core::model::ensure(inputs.cache, encoder, inputs.precision, |_| {}).await?;
    let mut embedder = census_core::Embedder::load(inputs.cache, encoder, inputs.precision)?;
    let descriptions = census_core::Anchors::from_descriptions(&mut embedder)?;

    // A corpus embedded with another encoder holds vectors that cannot be compared with
    // these anchors at all, so its labelled reviews are embedded again rather than read.
    // Asked per game rather than once: a set of games embedded under different encoders
    // would otherwise have some of them silently measured in the wrong space.
    let mut foreign: Vec<u32> = Vec::new();
    for &app_id in inputs.app_ids {
        if census_core::embed::corpus_encoder(inputs.out, app_id)? != encoder.id() {
            foreign.push(app_id);
        }
    }
    let fresh = !foreign.is_empty();
    if fresh {
        eprintln!(
            "{} of {} corpora were embedded with another encoder, so every game's labelled \
             reviews are being embedded again under {}.\nCalibration is off for this fit: a \
             reference set is not what an average review looks like, and a stored centroid \
             belongs to the encoder that built it.",
            foreign.len(),
            inputs.app_ids.len(),
            encoder.id()
        );
    }

    let mut games: Vec<GameLabels> = Vec::with_capacity(inputs.app_ids.len());
    for &app_id in inputs.app_ids {
        // One foreign corpus re-embeds all of them: every game in a fit has to arrive in
        // the same space, whichever space that turns out to be.
        let source = if fresh {
            Vectors::Fresh(&mut embedder)
        } else {
            Vectors::Stored
        };
        games.push(load_game(
            app_id,
            inputs.out,
            inputs.reference,
            inputs.holdout,
            source,
        )?);
    }
    if games.iter().all(|game| game.training.is_empty()) {
        anyhow::bail!(census_core::Error::NoTrainingExamples {
            app_id: inputs.app_ids[0]
        });
    }
    Ok(Under {
        encoder,
        descriptions,
        games,
        scoring: if fresh { Scoring::Raw } else { inputs.scoring },
        select: inputs.select,
    })
}

async fn run_fit(args: FitArgs) -> Result<()> {
    let FitArgs {
        app_ids,
        out,
        reference,
        holdout,
        folds,
        scoring,
        anchors_out,
        model_dir,
        model,
        precision,
        leave_one_out,
        compare,
        select,
    } = args;
    let holdout = holdout.as_str();
    if reference.is_some() && app_ids.len() > 1 {
        anyhow::bail!("--reference names one directory, so it cannot be used with several apps");
    }

    // Before the model, and for every game rather than the first: a fit across six corpora
    // that dies on the fifth has spent the download and the load to say what one look at the
    // directory would have said.
    for &app_id in &app_ids {
        census_core::embed::latest_snapshot(&out, app_id)?;
    }

    let cache = model_dir.unwrap_or_else(census_core::model::default_cache_dir);
    let inputs = FitInputs {
        app_ids: &app_ids,
        out: &out,
        reference: reference.as_ref(),
        holdout,
        cache: &cache,
        precision: precision.into(),
        scoring,
        select,
    };
    let base = load_under(model, &inputs).await?;

    if let Some(other) = compare {
        let other = load_under(other, &inputs).await?;
        return report_comparison(&base, &other, folds, holdout);
    }
    if leave_one_out {
        report_transfer(
            &base.descriptions,
            &base.games,
            folds,
            base.scoring,
            base.select,
            holdout,
        );
        return Ok(());
    }

    let Under {
        descriptions,
        games,
        scoring,
        select,
        ..
    } = base;
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
    let search = |objective| {
        let outcome = census_core::anchors::search(
            &descriptions,
            &examples,
            &centroid,
            folds,
            scoring.into(),
            objective,
        );
        let mut anchors = descriptions.fit(&examples, &app_ids, outcome.params);
        anchors.prepare(outcome.params.scoring, &centroid);
        (outcome, anchors)
    };
    let (outcome, fitted) = search(select.into());

    let path = anchors_out.unwrap_or_else(|| PathBuf::from("reference").join("anchors.json"));
    fitted.save(&path)?;
    let held: Vec<census_core::anchors::Example> =
        games.iter().flat_map(|game| game.holdout.clone()).collect();
    let honest = census_core::anchors::score(&fitted, &held, outcome.params.mention_margin);
    print_fit(&fitted, &outcome, &path, holdout, honest);
    print_cost_of_the_other_choice(&search(other(select.into())), &fitted, &held);
    Ok(())
}

const fn other(objective: census_core::anchors::Objective) -> census_core::anchors::Objective {
    match objective {
        census_core::anchors::Objective::Mentions => census_core::anchors::Objective::Primary,
        census_core::anchors::Objective::Primary => census_core::anchors::Objective::Mentions,
    }
}

/// What choosing the other objective would have bought and cost, on the reviews neither saw.
///
/// A flag that silently decides how good the numbers in the report are is worth measuring
/// rather than arguing about. The two anchor sets are judged on the same held-back reviews,
/// so the ones they place the same way carry no information and are left out of the test.
fn print_cost_of_the_other_choice(
    (outcome, anchors): &(census_core::FitOutcome, census_core::Anchors),
    chosen: &census_core::Anchors,
    held: &[census_core::anchors::Example],
) {
    if held.is_empty() {
        return;
    }
    let alternative = census_core::anchors::score(anchors, held, outcome.params.mention_margin);
    let mine = census_core::anchors::agreements(chosen, held);
    let theirs = census_core::anchors::agreements(anchors, held);
    let gained = mine
        .iter()
        .zip(&theirs)
        .filter(|(a, b)| **a && !**b)
        .count() as u64;
    let lost = mine
        .iter()
        .zip(&theirs)
        .filter(|(a, b)| !**a && **b)
        .count() as u64;

    println!(
        "\nChoosing for {} instead would have scored {:.1}% and {:.3} on the same reviews.",
        match outcome.objective {
            census_core::anchors::Objective::Mentions => "mentions",
            census_core::anchors::Objective::Primary => "the main subject",
        },
        alternative.primary_agreement() * 100.0,
        alternative.mention_macro_f1
    );
    match census_core::evaluate::mcnemar_exact(gained, lost) {
        Some(p) => println!(
            "On the main subject {gained} reviews went one way and {lost} the other, \
             p = {p:.4} (exact McNemar)."
        ),
        None => {
            println!("Neither placed a single review differently, so there is nothing to test.");
        }
    }
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
    scoring: Scoring,
    select: Select,
    holdout: &str,
) {
    let measured = measure_transfer(descriptions, games, folds, scoring, select);
    println!("leave-one-game-out, measured on each game's {holdout} subset");
    transfer_table(&measured, None, "every held-back review");
    // Split because the two answer different questions. A game the anchors genuinely cannot
    // reach is worse on the reviews the labeller found easy; a game whose taxonomy does not
    // fit it is worse only on the ones the labeller could not place either.
    transfer_table(
        &measured,
        Some(false),
        "only the reviews the labeller called clear-cut",
    );
    transfer_table(&measured, Some(true), "only the ones it called contested");
}

/// The transfer table over every held-back review, or over one side of the contested split.
fn transfer_table(measured: &[HeldOut], contested: Option<bool>, what: &str) {
    println!("\n{what}\n");
    println!(
        "{:<28}{:>7}{:>14}{:>12}{:>13}{:>16}",
        "held out", "n", "descriptions", "same game", "unseen game", "gained / lost"
    );
    println!("{}", "-".repeat(90));

    let mut totals = [0_u64; 6];
    for held in measured {
        let held = held.only(contested);
        let (written, same, unseen) = (&held.written, &held.same, &held.unseen);
        let n = written.len() as u64;
        let count = |verdicts: &[bool]| verdicts.iter().filter(|hit| **hit).count() as u64;
        let (gained, lost) = swapped(written, unseen);

        totals[0] += count(written);
        totals[1] += count(same);
        totals[2] += count(unseen);
        totals[3] += n;
        totals[4] += gained;
        totals[5] += lost;
        println!(
            "{:<28}{n:>7}{:>14}{:>12}{:>13}{:>16}",
            shortened(&held.name),
            share(count(written), n),
            share(count(same), n),
            share(count(unseen), n),
            format!("+{gained} / -{lost}")
        );
    }
    print_transfer_totals(&totals);
}

/// Runs the transfer measurement under two encoders and compares them review by review.
///
/// Two encoders judged on the same reviews are a paired comparison, and the reviews they
/// both place the same way say nothing about which is better. Reading two overlapping
/// intervals instead would throw away exactly the information that decides it.
fn report_comparison(base: &Under, other: &Under, folds: usize, holdout: &str) -> Result<()> {
    let left = measure_transfer(
        &base.descriptions,
        &base.games,
        folds,
        base.scoring,
        base.select,
    );
    let right = measure_transfer(
        &other.descriptions,
        &other.games,
        folds,
        other.scoring,
        other.select,
    );
    if left.len() != right.len()
        || left
            .iter()
            .zip(&right)
            .any(|(l, r)| l.name != r.name || l.unseen.len() != r.unseen.len())
    {
        anyhow::bail!(
            "the two encoders were measured on different reviews, so pairing them would \
             compare unlike with unlike"
        );
    }

    let one = base.encoder.as_str();
    let two = other.encoder.as_str();
    println!("leave-one-game-out under two encoders, on each game's {holdout} subset\n");
    println!(
        "{:<28}{:>7}{one:>14}{two:>14}{:>16}",
        "held out", "n", "gained / lost"
    );
    println!("{}", "-".repeat(79));

    let count = |verdicts: &[bool]| verdicts.iter().filter(|hit| **hit).count() as u64;
    let mut totals = [0_u64; 5];
    for (l, r) in left.iter().zip(&right) {
        let n = l.unseen.len() as u64;
        let (gained, lost) = swapped(&l.unseen, &r.unseen);
        totals[0] += count(&l.unseen);
        totals[1] += count(&r.unseen);
        totals[2] += n;
        totals[3] += gained;
        totals[4] += lost;
        println!(
            "{:<28}{n:>7}{:>14}{:>14}{:>16}",
            shortened(&l.name),
            share(count(&l.unseen), n),
            share(count(&r.unseen), n),
            format!("+{gained} / -{lost}")
        );
    }
    println!("{}", "-".repeat(79));
    println!(
        "{:<28}{:>7}{:>14}{:>14}{:>16}",
        "all",
        totals[2],
        share(totals[0], totals[2]),
        share(totals[1], totals[2]),
        format!("+{} / -{}", totals[3], totals[4])
    );

    println!(
        "\nBoth columns are the unseen-game fit: anchors built from the other five games and \
         never\nfrom the one being measured. Gained and lost are {two} against {one}."
    );
    match census_core::evaluate::mcnemar_exact(totals[3], totals[4]) {
        Some(p) => println!(
            "Of the {} reviews the two place differently, {} move to the category the labels \
             give\nunder {two} and {} move away. Exact McNemar {}.",
            totals[3] + totals[4],
            totals[3],
            totals[4],
            if p < 0.0001 {
                "p < 0.0001".to_owned()
            } else {
                format!("p = {p:.4}")
            }
        ),
        None => println!("The two encoders placed every held-out review alike."),
    }
    Ok(())
}

/// A Steam title cut to what a column can hold, ending in an ellipsis where it was cut.
fn shortened(name: &str) -> String {
    const ROOM: usize = 26;
    if name.chars().count() <= ROOM {
        return name.to_owned();
    }
    let kept: String = name.chars().take(ROOM - 1).collect();
    format!("{}\u{2026}", kept.trim_end())
}

/// One game held out, and whether each of its held-back reviews landed where the labels put
/// it under each anchor set it is judged with.
struct HeldOut {
    name: String,
    written: Vec<bool>,
    same: Vec<bool>,
    unseen: Vec<bool>,
    contested: Vec<bool>,
}

impl HeldOut {
    /// The same measurement over the reviews the labeller called one way or the other.
    ///
    /// A game can look like it refuses anchors built elsewhere and turn out to be a game
    /// whose held-back reviews the labeller could not place either, which is a fact about
    /// the taxonomy rather than about transfer.
    fn only(&self, contested: Option<bool>) -> Self {
        let keep = |verdicts: &[bool]| -> Vec<bool> {
            verdicts
                .iter()
                .zip(&self.contested)
                .filter(|(_, flagged)| contested.is_none_or(|want| **flagged == want))
                .map(|(hit, _)| *hit)
                .collect()
        };
        Self {
            name: self.name.clone(),
            written: keep(&self.written),
            same: keep(&self.same),
            unseen: keep(&self.unseen),
            contested: keep(&self.contested),
        }
    }
}

fn measure_transfer(
    descriptions: &census_core::Anchors,
    games: &[GameLabels],
    folds: usize,
    scoring: Scoring,
    select: Select,
) -> Vec<HeldOut> {
    let mut measured = Vec::with_capacity(games.len());
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
                scoring.into(),
                select.into(),
            );
            let mut anchors = descriptions.fit(examples, &[game.app_id], outcome.params);
            anchors.prepare(outcome.params.scoring, &game.centroid);
            anchors
        };

        measured.push(HeldOut {
            name: game.name.clone(),
            written: census_core::anchors::agreements(descriptions, &game.holdout),
            same: census_core::anchors::agreements(&fit_on(&all), &game.holdout),
            unseen: census_core::anchors::agreements(&fit_on(&others), &game.holdout),
            contested: game.contested.clone(),
        });
    }
    measured
}

fn print_transfer_totals(totals: &[u64; 6]) {
    println!("{}", "-".repeat(90));
    println!(
        "{:<28}{:>7}{:>14}{:>12}{:>13}{:>16}",
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
    honest: census_core::anchors::Scored,
) {
    println!("anchors from {}", apps(&anchors.fitted_from));
    println!("  fitted from  {} labels", outcome.examples);
    println!(
        "  held back    {} ({holdout} subset, never seen by the fit)",
        honest.compared
    );
    println!("  smoothing    {}", outcome.params.smoothing);
    println!("  secondary    {}", outcome.params.secondary_weight);
    println!("  margin       {}", outcome.params.mention_margin);
    println!(
        "  scoring      {}",
        match outcome.params.scoring {
            census_core::anchors::Scoring::Raw => "raw similarity",
            census_core::anchors::Scoring::Bias => "each category against its own average",
            census_core::anchors::Scoring::Centred => "the corpus mean removed from both sides",
        }
    );
    println!(
        "  chosen for   {}",
        match outcome.objective {
            census_core::anchors::Objective::Mentions =>
                "mentions, every category weighed the same",
            census_core::anchors::Objective::Primary =>
                "the main subject, which the largest few decide",
        }
    );
    println!("  anchors      {}", path.display());
    println!(
        "\n  {}-fold cross-validation, on training reviews each fold did not see:",
        outcome.folds
    );
    let baseline_note = "descriptions alone, at their best";
    println!(
        "    primary agreement  {:.1}%  ({baseline_note}: {:.1}%)",
        outcome.primary_agreement * 100.0,
        outcome.baseline_agreement * 100.0
    );
    println!(
        "    mention macro F1   {:.3}   ({baseline_note}: {:.3})",
        outcome.mention_macro_f1, outcome.baseline_macro_f1
    );

    if honest.compared > 0 {
        println!(
            "\n  on the {} held-back {holdout} reviews, which chose nothing:",
            honest.compared
        );
        let agreement = honest.primary_agreement();
        match census_core::evaluate::wilson(honest.agreed, honest.compared) {
            Some((low, high)) => println!(
                "    primary agreement  {:.1}%  [{:.1}, {:.1}]",
                agreement * 100.0,
                low * 100.0,
                high * 100.0
            ),
            None => println!("    primary agreement  {:.1}%", agreement * 100.0),
        }
        println!("    mention macro F1   {:.3}", honest.mention_macro_f1);
    }

    println!(
        "\n{:<30} {:>10} {:>16}",
        "category", "evidence", "learned share"
    );
    println!("{}", "-".repeat(58));
    // Thinnest last, the way `census evaluate` orders its own table. What a reader comes to
    // this one for is which categories were built from almost nothing, and in taxonomy order
    // those are five rows scattered through twenty-one.
    let mut built: Vec<&census_core::anchors::Anchor> = anchors.categories.iter().collect();
    built.sort_by(|a, b| b.evidence.total_cmp(&a.evidence));
    for anchor in built {
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
         blend and the margin were both chosen against them. The held-back figures above are\n\
         the honest ones, on reviews no setting was chosen against. `census evaluate` reads\n\
         the same subset after a full classify, and adds the per-category breakdown."
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
    println!("  elapsed      {}", elapsed(report.elapsed));
    println!("  assignments  {}", report.path.display());

    let mut categories = report.categories.clone();
    categories.sort_by(|a, b| {
        b.mention_rate(report.reviews)
            .total_cmp(&a.mention_rate(report.reviews))
    });

    // The size of the top of the pile is a setting, so the column that reports it is named
    // after what actually ran rather than after the default.
    println!(
        "\n{:<30} {:>9} {:>9} {:>9} {:>7}",
        "category",
        "mention%",
        "primary%",
        format!("top{}%", report.top_helpful),
        "bias"
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
         much the {} reviews Steam ranks most helpful overstate a category.",
        report.top_helpful
    );
    println!(
        "\nAccuracy of these assignments is UNMEASURED. No gold set exists yet, so treat \
         every figure above as provisional."
    );
}

fn run_sample_claims(
    app_ids: &[u32],
    out: &std::path::Path,
    reviews: usize,
    batch_size: usize,
    seed: u64,
    english: f64,
    to: &std::path::Path,
) -> Result<()> {
    let mut languages: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let (mut all_reviews, mut all_claims, mut all_batches) = (0, 0, 0);

    for &app_id in app_ids {
        let dir = to.join(app_id.to_string());
        let drawn = census_core::claimset::draw(out, app_id, reviews, english, seed)?;
        let report = census_core::claimset::write_set(&dir, &drawn, batch_size)?;
        for review in &drawn {
            *languages.entry(review.language.clone()).or_default() += 1;
        }
        all_reviews += report.reviews;
        all_claims += report.claims;
        all_batches += report.batches;
        println!(
            "{app_id:<9} {:>4} reviews  {:>5} claims  {:>3} batches  {:.2} per review",
            report.reviews,
            report.claims,
            report.batches,
            report.per_review()
        );
    }

    let mut ranked: Vec<(String, usize)> = languages.into_iter().collect();
    ranked.sort_by_key(|(name, count)| (std::cmp::Reverse(*count), name.clone()));

    println!("\ngames        {}", app_ids.len());
    println!("drawn        {all_reviews} reviews");
    println!("claims       {all_claims}");
    println!("batches      {all_batches} under {}", to.display());
    println!(
        "languages    {}",
        ranked
            .iter()
            .take(8)
            .map(|(name, count)| format!("{name} {count}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(())
}

fn run_read(
    app_id: u32,
    model_dir: &std::path::Path,
    options: &census_core::read::ReadOptions,
) -> Result<()> {
    census_core::embed::latest_snapshot(&options.out_dir, app_id)?;
    let mut model = census_core::reader::ClaimReader::load(model_dir)?;
    eprintln!("model        {} on {}", model_dir.display(), model.device());
    eprintln!("threshold    {:.2}", model.provenance().threshold);
    eprintln!("reading app {app_id}");

    // Printed whether or not anyone is watching a terminal: this is the pass that takes
    // hours, and a log with nothing in it is indistinguishable from a hang.
    let mut announced = 0;
    let report = census_core::read::read_corpus(&mut model, app_id, options, |progress| {
        if progress.done / 25_000 <= announced {
            return;
        }
        announced = progress.done / 25_000;
        let what = if progress.reading_claims {
            "distinct claims read"
        } else {
            "reviews counted"
        };
        eprintln!("  {} {what}", thousands(progress.done));
    })?;
    let path = census_core::embed::latest_snapshot(&options.out_dir, app_id)?.join("reading.json");
    report.save(&path)?;

    println!("app          {}", report.app_id);
    if let Some(language) = &report.language {
        println!(
            "counted      {} {language} reviews of {} in the corpus",
            thousands(report.reviews),
            thousands(report.corpus_reviews)
        );
    } else {
        println!("counted      {} reviews", thousands(report.reviews));
    }
    println!("claims       {}", thousands(report.claims));
    if let Some(share) = report.unclassified_share() {
        println!(
            "no subject   {} claims ({:.1}%), and {} reviews that name nothing at all",
            thousands(report.unclassified_claims),
            share * 100.0,
            thousands(report.silent_reviews)
        );
    }
    println!("took         {}", elapsed(report.elapsed));
    println!("written to   {}\n", path.display());

    let mut ranked: Vec<&census_core::read::SubjectCount> = report
        .subjects
        .iter()
        .filter(|subject| subject.mention_reviews > 0)
        .collect();
    ranked.sort_by_key(|subject| std::cmp::Reverse(subject.mention_reviews));
    println!(
        "{:<26} {:>9} {:>8} {:>8} {:>8} {:>8}",
        "subject", "reviews", "rate", "praise", "gripe", "mixed"
    );
    for subject in ranked {
        println!(
            "{:<26} {:>9} {:>8} {:>8} {:>8} {:>8}",
            subject.label,
            thousands(subject.mention_reviews),
            share(subject.mention_reviews, report.reviews),
            thousands(subject.praised),
            thousands(subject.criticised),
            thousands(subject.mixed),
        );
    }
    Ok(())
}

fn run_ingest_claims(app_id: u32, from: &std::path::Path, to: Option<PathBuf>) -> Result<()> {
    let dir = to.unwrap_or_else(|| {
        PathBuf::from("reference")
            .join("claims")
            .join(app_id.to_string())
    });
    let (labels, report) = census_core::claimset::ingest(&dir, from)?;

    let mut subjects: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    let mut polarity: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    let mut contested = 0;
    let mut miscut = 0;
    for label in &labels {
        *subjects.entry(label.subject.as_str()).or_default() += 1;
        *polarity.entry(label.polarity.as_str()).or_default() += 1;
        contested += usize::from(label.ambiguous);
        miscut += usize::from(label.split_wrong);
    }
    let mut ranked: Vec<(&str, usize)> = subjects.into_iter().collect();
    ranked.sort_by_key(|(name, count)| (std::cmp::Reverse(*count), *name));

    println!("app          {app_id}");
    println!("accepted     {} claims", report.accepted);
    println!("contested    {contested}");
    println!("mis-split    {miscut}");
    println!(
        "polarity     {}",
        ["praise", "complaint", "neutral"]
            .iter()
            .map(|name| format!("{name} {}", polarity.get(name).copied().unwrap_or(0)))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "commonest    {}",
        ranked
            .iter()
            .take(6)
            .map(|(name, count)| format!("{name} {count}"))
            .collect::<Vec<_>>()
            .join(", ")
    );

    if !report.is_clean() {
        println!("\nnot clean:");
        if !report.missing.is_empty() {
            println!("  {} claims drawn but never labelled", report.missing.len());
            for one in report.missing.iter().take(8) {
                println!("    {one}");
            }
        }
        if !report.unknown.is_empty() {
            println!("  {} labels naming a claim nobody drew", report.unknown.len());
        }
        if !report.rejected.is_empty() {
            println!("  {} labels the sheet does not offer", report.rejected.len());
            for one in report.rejected.iter().take(8) {
                println!("    {one}");
            }
        }
    }
    Ok(())
}

fn run_brief(to: &std::path::Path) -> Result<()> {
    use census_core::taxonomy::{Unit, labelling_brief};

    let sheets = [
        ("labelling-brief.txt", Unit::Review),
        ("claim-brief.txt", Unit::Claim),
    ];
    // A file path names one sheet, which is what a caller who wants only the review sheet
    // asks for. A directory gets both, because the two drift apart the moment one is
    // regenerated without the other.
    if to.extension().is_some() {
        std::fs::write(to, labelling_brief(Unit::Review))?;
        println!("brief    {}", to.display());
        return Ok(());
    }
    std::fs::create_dir_all(to)?;
    for (name, unit) in sheets {
        let path = to.join(name);
        std::fs::write(&path, labelling_brief(unit))?;
        println!("brief    {}", path.display());
    }
    Ok(())
}

fn run_claims(app_id: u32, out: &std::path::Path) -> Result<()> {
    eprintln!("splitting app {app_id}");
    let mut announced = 0;
    let report = census_core::claims::extract_corpus(out, app_id, |reviews, claims| {
        if reviews / 100_000 > announced {
            announced = reviews / 100_000;
            eprintln!("  {reviews} reviews, {claims} claims");
        }
    })?;

    println!("app          {}", report.app_id);
    println!("reviews      {}", report.reviews);
    println!("claims       {}", report.claims);
    println!("per review   {:.2}", report.per_review());
    println!("distinct     {}", report.distinct);
    if let Some(share) = report.repeated() {
        println!("repeated     {:.1}% of claims are said in the same words elsewhere", share * 100.0);
    }
    println!("empty        {} reviews split into nothing", report.empty);
    Ok(())
}

async fn run_embed(
    app_id: u32,
    out: &std::path::Path,
    batch_size: usize,
    model_dir: Option<PathBuf>,
    encoder: census_core::Encoder,
    precision: census_core::model::Precision,
    unit: census_core::taxonomy::Unit,
) -> Result<()> {
    // Before the model, which on a first run is a download, and which a game that was never
    // crawled has no use for.
    census_core::embed::latest_snapshot(out, app_id)?;
    let cache = model_dir.unwrap_or_else(census_core::model::default_cache_dir);
    let interactive = std::io::stderr().is_terminal();

    eprintln!("model:       {} {}", encoder.id(), precision.as_str());
    eprintln!("model cache: {}", cache.display());
    let mut announced = String::new();
    census_core::model::ensure(&cache, encoder, precision, |progress| {
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

    let mut embedder = census_core::Embedder::load(&cache, encoder, precision)?;
    eprintln!("embedding app {app_id} on {}", embedder.device());
    let mut last_line = 0;
    let report = census_core::embed_corpus(&mut embedder, out, app_id, batch_size, unit, |progress| {
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
    println!("  elapsed        {}", elapsed(report.elapsed));
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
            census_core::report::coverage
        )
    );
    println!("  elapsed      {}", elapsed(report.elapsed));
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
        let row = format!(
            "    {name:<14} n={:<5} {rate:>6}{interval:<16}{note}",
            slice.compared
        );
        println!("{}", row.trim_end());
    }
}

/// A duration at the length a real run takes.
///
/// A million-review crawl is several hours, and "11455.3s" is a number the reader has to do
/// arithmetic on before it means anything.
fn elapsed(taken: std::time::Duration) -> String {
    let seconds = taken.as_secs();
    if seconds < 60 {
        return format!("{:.1}s", taken.as_secs_f64());
    }
    let (hours, minutes, seconds) = (seconds / 3600, (seconds % 3600) / 60, seconds % 60);
    if hours > 0 {
        format!("{hours}h {minutes:02}m {seconds:02}s")
    } else {
        format!("{minutes}m {seconds:02}s")
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
    use clap::Parser as _;

    use super::{Cli, Command, Model, elapsed, thousands};

    /// A crawl of a million reviews runs for hours, and a figure in seconds is arithmetic
    /// the reader has to do before the number means anything.
    #[test]
    fn a_run_reports_its_length_the_way_a_person_would_say_it() {
        use std::time::Duration;

        assert_eq!(elapsed(Duration::from_millis(1_500)), "1.5s");
        assert_eq!(elapsed(Duration::from_secs(59)), "59.0s");
        assert_eq!(elapsed(Duration::from_secs(60)), "1m 00s");
        assert_eq!(elapsed(Duration::from_secs(1_147)), "19m 07s");
        assert_eq!(elapsed(Duration::from_secs(3_600)), "1h 00m 00s");
        assert_eq!(elapsed(Duration::from_secs(11_455)), "3h 10m 55s");
    }

    #[test]
    fn the_flag_default_is_the_encoder_the_library_would_have_picked() {
        // Two defaults that must agree: what `--model` falls back to, and what a corpus is
        // embedded with when nothing says otherwise. Drift between them would classify a
        // corpus with anchors from another encoder, which is refused, loudly, much later.
        let parsed = Cli::parse_from(["census", "embed", "1"]);
        let Command::Embed { model, .. } = parsed.command else {
            panic!("embed did not parse as embed");
        };
        assert_eq!(
            census_core::Encoder::from(model),
            census_core::Encoder::default()
        );
        assert_eq!(census_core::Encoder::from(Model::default()), model.into());
    }

    #[test]
    fn thousands_groups_digits() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(2909), "2,909");
        assert_eq!(thousands(986_295), "986,295");
        assert_eq!(thousands(1_700_000), "1,700,000");
    }
}
