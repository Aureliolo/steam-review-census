use std::{
    io::{IsTerminal, Write},
    path::PathBuf,
    time::Duration,
};

use anyhow::Result;
use census_core::{
    CrawlOptions, CrawlReport, DEFAULT_BATCH_SIZE, DEFAULT_SHARD_TARGET, SteamClient, crawl,
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

    /// Score stored readings against a claim reference set.
    MeasureClaims {
        /// Steam app IDs to score. Each is reported separately, because a model that reads
        /// one game well and another badly is not a model with one accuracy.
        #[arg(required = true, num_args = 1..)]
        app_ids: Vec<u32>,
        /// Directory holding the captures.
        #[arg(short, long, default_value = "data")]
        out: PathBuf,
        /// Where the claim reference sets live.
        #[arg(long, default_value = "reference/claims")]
        reference: PathBuf,
    },

    /// Draw the share of a labelled set that a second labeller should read.
    ///
    /// A set labelled once cannot say how reliable it is. This writes the same reviews again,
    /// as fresh batches with no labels in them, for a different labeller to work from blind.
    SecondOpinion {
        /// Steam app IDs to draw a second opinion on. Every labelled set when none are named.
        #[arg(num_args = 0..)]
        app_ids: Vec<u32>,
        /// Where the claim reference sets live.
        #[arg(long, default_value = "reference/claims")]
        reference: PathBuf,
        /// Share of each set to read again.
        #[arg(long, default_value_t = 0.1)]
        share: f64,
        /// Reviews per batch, as with the first draw.
        #[arg(long, default_value_t = 35)]
        batch_size: usize,
        /// Changing this asks about different reviews. The same seed asks about the same ones.
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },

    /// Compare two labellings of the same claims, field by field.
    CompareLabels {
        /// Steam app IDs to compare. Every set with a second opinion when none are named.
        #[arg(num_args = 0..)]
        app_ids: Vec<u32>,
        /// Where the claim reference sets live.
        #[arg(long, default_value = "reference/claims")]
        reference: PathBuf,
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
        /// Steam app IDs whose most recent captures should be read.
        #[arg(required = true, num_args = 1..)]
        app_ids: Vec<u32>,
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
        #[arg(long, default_value_t = census_core::capture::DEFAULT_TOP_HELPFUL)]
        top_helpful: usize,
    },

    /// Render a self-contained page from what the reading pass found.
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
        /// Claims quoted per subject, as the evidence behind its rate.
        #[arg(long, default_value_t = census_core::report::DEFAULT_EXAMPLES)]
        examples: usize,
        /// Changing this quotes different claims. The same seed always quotes the same ones.
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },

    /// Write the category sheet labellers work from, generated from the taxonomy.
    ///
    /// Separate from `sample-claims` because a boundary rule can change without anything
    /// needing to be drawn again, and regenerating the sheet should never mean redrawing a
    /// sample.
    Brief {
        /// Where to write it. Both sheets are written when this is a directory.
        #[arg(long, default_value = "reference")]
        to: PathBuf,
    },
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
        Command::MeasureClaims {
            app_ids,
            out,
            reference,
        } => run_measure_claims(&app_ids, &out, &reference),
        Command::SecondOpinion {
            app_ids,
            reference,
            share,
            batch_size,
            seed,
        } => run_second_opinion(&app_ids, &reference, share, batch_size, seed),
        Command::CompareLabels { app_ids, reference } => run_compare_labels(&app_ids, &reference),
        Command::Read {
            app_ids,
            out,
            model,
            batch_size,
            language,
            top_helpful,
        } => run_read(
            &app_ids,
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
        Command::Brief { to } => run_brief(&to),
        Command::Report {
            app_ids,
            out,
            to,
            examples,
            seed,
        } => run_report(&app_ids, &out, &to, examples, seed),
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
            "  {:<10} {} reviews, {} claims, {quoted} quoted",
            app.app_id(),
            thousands(app.reading.reviews),
            thousands(app.reading.claims)
        );
    }
    println!(
        "\nSelf-contained: open it from disk, send it as one file, print it. Nothing in it is\n\
         fetched from anywhere, and no review left this machine to produce it."
    );
    Ok(())
}

/// Everything one game contributes to a fit: the examples it trains on, the ones held back
/// from it, and what an average review of it looks like.
#[expect(clippy::cast_precision_loss, reason = "reference sets are hundreds")]
fn share(part: u64, whole: u64) -> String {
    if whole == 0 {
        return "n/a".to_owned();
    }
    format!("{:.1}%", part as f64 / whole as f64 * 100.0)
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
    app_ids: &[u32],
    model_dir: &std::path::Path,
    options: &census_core::read::ReadOptions,
) -> Result<()> {
    // Loaded once for the whole slate. Building the session takes longer than reading a
    // small corpus, so doing it per game would be most of the time for a list of them.
    let mut model = census_core::reader::ClaimReader::load(model_dir)?;
    eprintln!("model        {} on {}", model_dir.display(), model.device());
    eprintln!("threshold    {:.2}", model.provenance().threshold);
    for &app_id in app_ids {
        read_one(&mut model, app_id, options)?;
    }
    Ok(())
}

fn read_one(
    model: &mut census_core::reader::ClaimReader,
    app_id: u32,
    options: &census_core::read::ReadOptions,
) -> Result<()> {
    census_core::embed::latest_snapshot(&options.out_dir, app_id)?;
    eprintln!("reading app {app_id}");

    // Printed whether or not anyone is watching a terminal: this is the pass that takes
    // hours, and a log with nothing in it is indistinguishable from a hang.
    let mut announced = 0;
    let report = census_core::read::read_corpus(model, app_id, options, |progress| {
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

/// Every app id under a reference root that has been labelled, in order.
fn labelled_sets(reference: &std::path::Path) -> Result<Vec<u32>> {
    let mut found: Vec<u32> = std::fs::read_dir(reference)?
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.path().join("labels.json").is_file())
        .filter_map(|entry| entry.file_name().to_str()?.parse::<u32>().ok())
        .collect();
    found.sort_unstable();
    Ok(found)
}

fn run_second_opinion(
    app_ids: &[u32],
    reference: &std::path::Path,
    share: f64,
    batch_size: usize,
    seed: u64,
) -> Result<()> {
    let wanted = if app_ids.is_empty() {
        labelled_sets(reference)?
    } else {
        app_ids.to_vec()
    };
    if wanted.is_empty() {
        anyhow::bail!("no labelled sets under {}", reference.display());
    }

    let (mut reviews, mut claims, mut batches) = (0, 0, 0);
    for app_id in wanted {
        let dir = reference.join(app_id.to_string());
        let drawn = census_core::claimset::draw_second(&dir, share, seed)?;
        let report =
            census_core::claimset::write_set(&dir.join("second"), &drawn, batch_size)?;
        reviews += report.reviews;
        claims += report.claims;
        batches += report.batches;
        println!(
            "{app_id:<9} {:>4} reviews  {:>5} claims  {:>3} batches",
            report.reviews, report.claims, report.batches
        );
    }

    println!("\ndrawn        {reviews} reviews, {claims} claims, {batches} batches");
    println!(
        "\nHand these to a different labeller from the one that did the first pass, and give \
         it\nthe same sheet and nothing else. A second opinion that can see the first is not a\n\
         second opinion. Ingest with --to <set>/second, then `census compare-labels`."
    );
    Ok(())
}

fn run_compare_labels(app_ids: &[u32], reference: &std::path::Path) -> Result<()> {
    let wanted = if app_ids.is_empty() {
        labelled_sets(reference)?
            .into_iter()
            .filter(|app_id| {
                reference
                    .join(app_id.to_string())
                    .join("second")
                    .join("labels.json")
                    .is_file()
            })
            .collect()
    } else {
        app_ids.to_vec()
    };
    if wanted.is_empty() {
        anyhow::bail!(
            "no set under {} has a second labelling yet; run `census second-opinion` first",
            reference.display()
        );
    }

    let pct =
        |value: Option<f64>| value.map_or_else(|| "-".to_owned(), |v| format!("{:.1}%", v * 100.0));
    for app_id in wanted {
        let dir = reference.join(app_id.to_string());
        let found = census_core::reliability::compare(
            &dir.join("labels.json"),
            &dir.join("second").join("labels.json"),
        )?;

        println!("\napp {app_id}");
        println!(
            "  {} claims read twice, {} only once",
            thousands(found.overlap),
            thousands(found.only_first + found.only_second)
        );
        println!("\n  {:<14} {:>9} {:>8} {:>18}", "field", "agreed", "kappa", "commonest split");
        for field in &found.fields {
            let split = field.commonest_split.as_ref().map_or_else(
                String::new,
                |(a, b, count)| format!("{a} / {b} ({count})"),
            );
            println!(
                "  {:<14} {:>9} {:>8} {split:>18}",
                field.field,
                pct(field.rate()),
                field
                    .kappa
                    .map_or_else(|| "-".to_owned(), |k| format!("{k:.2}"))
            );
        }
    }

    println!(
        "\nKappa is agreement beyond what these two labellers' own habits would produce by \
         chance.\nA corpus is mostly `verdict` and `offtopic`, so two labellers who never read \
         a claim\nwould still agree most of the time; the percentage alone cannot tell you that \
         apart\nfrom reading. Below about 0.4 the two are not labelling the same thing."
    );
    Ok(())
}

fn run_measure_claims(
    app_ids: &[u32],
    out: &std::path::Path,
    reference: &std::path::Path,
) -> Result<()> {
    let pct = |value: Option<f64>| value.map_or_else(|| "-".to_owned(), |v| format!("{:.1}%", v * 100.0));

    for &app_id in app_ids {
        let found = census_core::measure::agreement(
            out,
            app_id,
            &reference.join(app_id.to_string()),
        )?;

        println!("\napp {app_id}");
        println!(
            "  {} labelled claims found in the readings, {} answered, {} declined ({})",
            thousands(found.matched),
            thousands(found.answered),
            thousands(found.declined),
            pct(found.declined_share())
        );
        println!(
            "  {} agreement on what it answered, macro F1 {}",
            pct(found.rate()),
            found
                .macro_f1()
                .map_or_else(|| "-".to_owned(), |v| format!("{v:.3}"))
        );
        println!("  {} of polarity", pct(found.polarity_rate()));
        #[expect(
            clippy::cast_precision_loss,
            reason = "reference sets are thousands of claims"
        )]
        {
            let share = |part: u64, whole: u64| {
                (whole > 0).then(|| part as f64 / whole as f64)
            };
            println!(
                "  split by how the labeller called it: {} on {} clear-cut, {} on {} contested",
                pct(share(found.clear_agreed, found.clear_answered)),
                thousands(found.clear_answered),
                pct(share(found.contested_agreed, found.contested_answered)),
                thousands(found.contested_answered)
            );
        }

        let mut ranked: Vec<&census_core::measure::SubjectAgreement> = found
            .subjects
            .iter()
            .filter(|subject| subject.labelled > 0)
            .collect();
        ranked.sort_by(|left, right| {
            right
                .f1()
                .unwrap_or(0.0)
                .total_cmp(&left.f1().unwrap_or(0.0))
        });
        println!(
            "\n  {:<26} {:>8} {:>10} {:>8} {:>7}  most often read as",
            "subject", "labelled", "precision", "recall", "F1"
        );
        for subject in ranked {
            println!(
                "  {:<26} {:>8} {:>10} {:>8} {:>7}  {}",
                subject.label,
                thousands(subject.labelled),
                pct(subject.precision()),
                pct(subject.recall()),
                subject
                    .f1()
                    .map_or_else(|| "-".to_owned(), |v| format!("{v:.2}")),
                subject
                    .mistaken_for
                    .map_or_else(String::new, |(label, count)| format!("{label} ({count})"))
            );
        }
    }

    println!(
        "\nThese are AGREEMENT figures, not accuracy. The labels were produced by a model, so\n\
         this measures consistency between two models rather than correctness. Two models can\n\
         agree and both be wrong, most easily on sarcasm and on the boundaries between\n\
         subjects, which is exactly where this one is weakest."
    );
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
        // embedded with when nothing says otherwise. Drift between them would leave a corpus
        // holding vectors from an encoder nothing else expects, which is refused much later.
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
