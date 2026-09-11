//! The pipeline from a capture on disk to a page, without a model or a network.
//!
//! Every stage after embedding reads what the stage before it wrote, and the joins between
//! them are by column name and by hash. Those are exactly the places a change breaks
//! something three commands away with an error that names neither, so this builds a small
//! corpus by hand and walks the whole way through it.

use std::{collections::HashMap, path::Path, sync::Arc};

use arrow::{
    array::{
        ArrayRef, BooleanBuilder, FixedSizeListBuilder, Float32Builder, Float64Builder,
        Int64Builder, StringBuilder, UInt32Builder,
    },
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use census_core::taxonomy::{CORE_SPINE, CORE_SPINE_VERSION};
use parquet::arrow::ArrowWriter;

/// One axis per category, so a review's nearest anchor is the one this test names and not
/// whichever of several tied categories the iterator happened to end on.
const DIM: usize = CORE_SPINE.len();
const MODEL: &str = "test-encoder";

/// One review, and which category it should land in.
struct Seed {
    id: &'static str,
    text: &'static str,
    category: usize,
    votes_up: u32,
    voted_up: bool,
    language: &'static str,
}

fn seeds() -> Vec<Seed> {
    // Deliberately includes a repeated text, a blank one, and a review whose votes make it
    // the top of the pile while being about something the corpus barely discusses.
    vec![
        Seed {
            id: "1",
            text: "crashes on launch",
            category: 1,
            votes_up: 900,
            voted_up: false,
            language: "english",
        },
        Seed {
            id: "2",
            text: "crashes on launch",
            category: 1,
            votes_up: 3,
            voted_up: false,
            language: "english",
        },
        Seed {
            id: "3",
            text: "runs at four frames",
            category: 0,
            votes_up: 1,
            voted_up: false,
            language: "english",
        },
        Seed {
            id: "4",
            text: "great game",
            category: 4,
            votes_up: 0,
            voted_up: true,
            language: "schinese",
        },
        Seed {
            id: "5",
            text: "great game too",
            category: 4,
            votes_up: 0,
            voted_up: true,
            language: "english",
        },
        Seed {
            id: "6",
            text: "   ",
            category: 4,
            votes_up: 0,
            voted_up: true,
            language: "english",
        },
    ]
}

/// A unit vector pointing at one category's axis, so the nearest anchor is never in doubt.
fn vector(category: usize) -> Vec<f32> {
    let mut vector = vec![0.0_f32; DIM];
    vector[category] = 1.0;
    vector
}

fn write_capture(snapshot: &Path) {
    let schema = census_core::capture::schema();
    let mut ids = StringBuilder::new();
    let mut appids = UInt32Builder::new();
    let mut authors = StringBuilder::new();
    let mut languages = StringBuilder::new();
    let mut texts = StringBuilder::new();
    let mut created = Int64Builder::new();
    let mut recommended = BooleanBuilder::new();
    let mut votes = UInt32Builder::new();
    let mut funny = UInt32Builder::new();
    let mut weighted = Float64Builder::new();
    let mut raw = StringBuilder::new();

    for seed in seeds() {
        ids.append_value(seed.id);
        appids.append_value(1);
        authors.append_value(format!("7656{}", seed.id));
        languages.append_value(seed.language);
        texts.append_value(seed.text);
        created.append_value(1_700_000_000);
        recommended.append_value(seed.voted_up);
        votes.append_value(seed.votes_up);
        funny.append_value(0);
        weighted.append_value(f64::from(seed.votes_up));
        raw.append_value("{}");
    }

    let mut columns: HashMap<&str, ArrayRef> = HashMap::new();
    columns.insert("recommendationid", Arc::new(ids.finish()));
    columns.insert("appid", Arc::new(appids.finish()));
    columns.insert("author_steamid", Arc::new(authors.finish()));
    columns.insert("language", Arc::new(languages.finish()));
    columns.insert("review", Arc::new(texts.finish()));
    columns.insert("timestamp_created", Arc::new(created.finish()));
    columns.insert("voted_up", Arc::new(recommended.finish()));
    columns.insert("votes_up", Arc::new(votes.finish()));
    columns.insert("votes_funny", Arc::new(funny.finish()));
    columns.insert("weighted_vote_score", Arc::new(weighted.finish()));
    columns.insert("raw_json", Arc::new(raw.finish()));

    // Anything the capture schema has and this test does not fill is written as nulls, so a
    // new column never silently becomes this test's problem.
    let rows = seeds().len();
    let ordered: Vec<ArrayRef> = schema
        .fields()
        .iter()
        .map(|field| {
            columns
                .remove(field.name().as_str())
                .unwrap_or_else(|| arrow::array::new_null_array(field.data_type(), rows))
        })
        .collect();

    let batch = RecordBatch::try_new(Arc::clone(&schema), ordered).unwrap();
    let file = std::fs::File::create(snapshot.join("shard-0000.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn write_embeddings(snapshot: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("text_sha256", DataType::Utf8, false),
        Field::new("appid", DataType::UInt32, false),
        Field::new("n_reviews", DataType::UInt32, false),
        Field::new("model", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                i32::try_from(DIM).unwrap(),
            ),
            false,
        ),
    ]));

    let mut hashes = StringBuilder::new();
    let mut appids = UInt32Builder::new();
    let mut counts = UInt32Builder::new();
    let mut models = StringBuilder::new();
    let mut vectors = FixedSizeListBuilder::new(Float32Builder::new(), i32::try_from(DIM).unwrap());

    let mut seen: Vec<String> = Vec::new();
    for seed in seeds() {
        if seed.text.trim().is_empty() {
            continue;
        }
        let hash = sha256_hex(seed.text);
        if seen.contains(&hash) {
            continue;
        }
        seen.push(hash.clone());
        hashes.append_value(&hash);
        appids.append_value(1);
        counts.append_value(1);
        models.append_value(MODEL);
        vectors.values().append_slice(&vector(seed.category));
        vectors.append(true);
    }

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(hashes.finish()),
            Arc::new(appids.finish()),
            Arc::new(counts.finish()),
            Arc::new(models.finish()),
            Arc::new(vectors.finish()),
        ],
    )
    .unwrap();

    let file = std::fs::File::create(snapshot.join("embeddings.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    std::fs::write(
        snapshot.join("embeddings.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "model": MODEL,
            "precision": "fp32",
            "dimensions": DIM,
            "batch_size": 8,
        }))
        .unwrap(),
    )
    .unwrap();
}

fn sha256_hex(text: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hasher
        .finalize()
        .iter()
        .fold(String::new(), |mut out, byte| {
            use std::fmt::Write as _;
            let _ = write!(out, "{byte:02x}");
            out
        })
}

/// The readings a model would have produced, written by hand.
///
/// Every review here makes one claim, which is the only shape that lets a test assert exact
/// counts without also asserting how the splitter happens to cut prose today. The point of
/// this file is the join between the stages, not the arithmetic inside any one of them.
fn write_readings(snapshot: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("recommendationid", DataType::Utf8, false),
        Field::new("claim_index", DataType::UInt16, false),
        Field::new("subject", DataType::Utf8, true),
        Field::new("confidence", DataType::Float32, false),
        Field::new("polarity", DataType::Utf8, false),
    ]));

    let mut ids = StringBuilder::new();
    let mut indexes = arrow::array::UInt16Builder::new();
    let mut subjects = StringBuilder::new();
    let mut confidences = Float32Builder::new();
    let mut polarities = StringBuilder::new();

    let mut counted = 0_u64;
    let mut positive = 0_u64;
    let mut per_subject = vec![0_u64; CORE_SPINE.len()];
    let mut top = seeds();
    top.sort_by_key(|seed| std::cmp::Reverse(seed.votes_up));
    let loudest: Vec<&str> = top.iter().take(2).map(|seed| seed.id).collect();
    let mut top_per_subject = vec![0_u64; CORE_SPINE.len()];

    for seed in seeds() {
        if seed.text.trim().is_empty() {
            continue;
        }
        counted += 1;
        positive += u64::from(seed.voted_up);
        per_subject[seed.category] += 1;
        if loudest.contains(&seed.id) {
            top_per_subject[seed.category] += 1;
        }
        ids.append_value(seed.id);
        indexes.append_value(0);
        subjects.append_value(CORE_SPINE[seed.category].id);
        confidences.append_value(0.9);
        polarities.append_value(if seed.voted_up { "praise" } else { "complaint" });
    }

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(ids.finish()),
            Arc::new(indexes.finish()),
            Arc::new(subjects.finish()),
            Arc::new(confidences.finish()),
            Arc::new(polarities.finish()),
        ],
    )
    .unwrap();
    let file = std::fs::File::create(snapshot.join("readings.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let subjects: Vec<serde_json::Value> = CORE_SPINE
        .iter()
        .enumerate()
        .map(|(slot, category)| {
            serde_json::json!({
                "id": category.id,
                "label": category.label,
                "mention_reviews": per_subject[slot],
                "primary_reviews": per_subject[slot],
                "claims": per_subject[slot],
                "praised": 0,
                "criticised": per_subject[slot],
                "mixed": 0,
                "top_mention_reviews": top_per_subject[slot],
                "positive_mentions": 0,
            })
        })
        .collect();

    std::fs::write(
        snapshot.join("reading.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "app_id": 1,
            "reviews": counted,
            "corpus_reviews": counted,
            "language": serde_json::Value::Null,
            "claims": counted,
            "unclassified_claims": 0,
            "silent_reviews": 0,
            "positive": positive,
            "top_helpful": 2,
            "model": MODEL,
            "spine_version": CORE_SPINE_VERSION,
            "splitter": census_core::claims::SPLITTER_VERSION,
            "threshold": 0.5,
            "device": "cpu",
            "subjects": subjects,
            "languages": [["english", 4], ["schinese", 1]],
            "months": [],
        }))
        .unwrap(),
    )
    .unwrap();
}

/// A capture, its embeddings and an anchor set, all on disk and all real files.
fn build_corpus(name: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let root = std::env::temp_dir().join(format!("census-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    let snapshot = root.join("appid=1").join("snapshot=1700000000");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(
        snapshot.join("crawl.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "app_id": 1,
            "name": "Test Game",
            "review_score_desc": "Mixed",
            "rows_unique": 6,
            "valve_total_reviews": 6,
            "valve_total_positive": 3,
            "valve_total_negative": 3,
            "coverage": 1.0,
            "snapshot_unix": 1_700_000_000_i64,
            "shards": 1,
        }))
        .unwrap(),
    )
    .unwrap();
    write_capture(&snapshot);
    write_embeddings(&snapshot);
    write_readings(&snapshot);
    (root, snapshot)
}

fn reporting(root: &Path) -> census_core::report::ReportOptions {
    census_core::report::ReportOptions {
        out_dir: root.to_path_buf(),
        examples: 4,
        seed: 1,
    }
}

#[test]
fn a_capture_becomes_a_page_without_a_model_or_a_network() {
    let (root, _snapshot) = build_corpus("pipeline");

    let rendered = census_core::report::build(&[1], &reporting(&root)).unwrap();
    let app = &rendered.apps[0];

    // Six reviews in the capture. The blank one is in no rate at all, not even the
    // denominator, which is what makes the others mean what they say.
    assert_eq!(
        app.reading.reviews, 5,
        "a review with no text is not a review here"
    );

    let by_id = |id: &str| {
        app.reading
            .subjects
            .iter()
            .find(|s| s.id == id)
            .unwrap_or_else(|| panic!("no subject {id}"))
    };
    assert_eq!(
        by_id("bugs").mention_reviews,
        2,
        "the repeated text counts twice"
    );
    assert_eq!(by_id("performance").mention_reviews, 1);

    // The top of the pile is two reviews, and the 900-vote crash report is one of them, so
    // bugs are far more of the top than of the corpus.
    assert_eq!(app.reading.top_helpful, 2);
    assert!(app.bias(by_id("bugs")).unwrap() > 1.0);

    // The evidence has to survive the join from readings to capture: a claim is quoted by
    // review id and claim index, and the text it names is read back out of the shards.
    let quoted: usize = app.examples.iter().map(|(_, shown)| shown.len()).sum();
    assert!(quoted > 0, "no claim reached the page as evidence");

    let page = census_core::html::render(&rendered);
    assert!(page.contains("Test Game"), "the game's name is missing");
    assert!(
        page.contains("crashes on launch"),
        "the evidence is missing"
    );
    assert!(!page.contains(">   <"), "a blank review was quoted");
    assert!(
        page.contains("https://steamcommunity.com/profiles/76561/recommended/1/"),
        "the link back to the source is missing or malformed"
    );

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn counts_the_corpus_no_longer_supports_are_refused_rather_than_drawn() {
    // A reading made against another taxonomy counts subjects this build does not have. It
    // would render perfectly and every number in it would be about something else.
    let (root, snapshot) = build_corpus("stale");

    let sidecar = snapshot.join("reading.json");
    let mut stored: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&sidecar).unwrap()).unwrap();
    stored["spine_version"] = serde_json::Value::String("core-1".to_owned());
    std::fs::write(&sidecar, serde_json::to_vec_pretty(&stored).unwrap()).unwrap();

    let message = census_core::report::build(&[1], &reporting(&root))
        .expect_err("a reading against another taxonomy must not render")
        .to_string();
    assert!(
        message.contains("taxonomy"),
        "the refusal should name what disagrees, got: {message}"
    );

    std::fs::remove_dir_all(&root).ok();
}
