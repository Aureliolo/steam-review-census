# Decisions

What has been settled about this tool, and where each thing actually stands. It exists because
a long build forgets: a decision made once and not written down gets quietly dropped, or worse,
asked again as though it were open. Nothing here is a proposal. Every row was chosen.

State means: **done** is built and in use; **partial** is built for one case and not the rest;
**not built** is exactly that, however long ago it was agreed.

## The product

| Decided | State |
|---|---|
| Rust core, **Tauri desktop shell**, and a CLI beside it | shell **not built** |
| Compiled binary for Windows, Linux and macOS; the user downloads it, double-clicks it, and works entirely in the UI | **not built** |
| The UI is first class, not a wrapper over the pipeline, and has to be good enough to look at | **not built** |
| Results also export as one self-contained HTML page that fetches nothing | done |
| Name: `steam-review-census` | done |

## What the analysis does

| Decided | State |
|---|---|
| Headline figure is the **mention rate**, always labelled as such | done |
| Every other percentage says which denominator it uses | done |
| **Deep, claim-level by default**: a review is split into the points it makes, and each point carries a category | **not built**, classification is whole-review |
| Shallow is the opt-out, and neither depth drops a review | **not built** |
| Taxonomy is a **fixed core spine plus induced game-specific extras** | spine done, induction **not built** |
| Extras are discovered by an **LLM reading an embedding-diverse sample** | **not built** |
| Categories are assigned across the full corpus by a **linear probe over embeddings** | drifted to a prototype/anchor classifier with shrinkage |
| **Summarise, per category, what people praise and complain about** | **not built** |
| **Build an overall picture of the game from those summaries** | **not built** |
| **Click through from any number to the reviews behind it** | **not built**; the page quotes a few examples per category |
| Helpfulness bias ships as a column on every category | done |
| Irony and ratings that disagree with the text are flagged, not filed away | flagged on labelled reviews only |

## Models and setup

| Decided | State |
|---|---|
| A complete census with **zero setup and no API key**; hosted models are an upgrade, never a requirement | done |
| Embeddings run locally, on ONNX Runtime | done |
| `ort` with DirectML, CoreML and CPU | done |
| Embeddings carry dedupe, taxonomy, search and classification, not just one of them | dedupe and classification done, search partial |
| A hosted model does the reading when one is configured | **not built**, there is no API client in the tree |
| The user chooses which model does the sorting, and how closely it reads | **not built** |

## Data and crawling

| Decided | State |
|---|---|
| Raw Parquet capture, kept, because a re-crawl cannot recover edited or deleted reviews | done |
| SQLite for crawl state | done |
| **DuckDB for querying the corpus** | **not built** |
| `author_steamid` kept for every review, as public data | done |
| Adaptive, date-sharded crawl with capped concurrency | done |
| Watermark top-up so a re-crawl does not re-pull old reviews | done |
| **Periodic sweep by last-edit date**, to catch reviews edited since the crawl | order exists, **not wired to a command** |
| Valve's default filters overridden, because they hide 17.3% of negative reviews against 9.6% of positive | done |

## Distribution

| Decided | State |
|---|---|
| Apache-2.0 | done |
| Signed releases with cosign, plus build provenance attestation | done |
| Immutable release artefacts with checksums | done |
| No money spent: self-signed on macOS, and an extra step there is acceptable | accepted |
| Supply-chain hardening in proportion to the project, not the full enterprise set | done |

## Reference sets

| Decided | State |
|---|---|
| Reviews labelled by Fable 5.1 agents reading batches in parallel, each shown the text alone | done |
| **Opus spot-checks the labels** | **not done** |
| Roughly 400 labels to start | grew to ~5,000 across 36 games |
| 30 to 35 mid-size games, mixed sentiment, small corpora acceptable | done, 36 games |
| Stratified subset trains, random subset measures, and the two are never merged | done |
| Measured error **corrects the reported prevalence** | measured, **never applied as a correction** |
| The sets are a silver standard, and the README says so rather than calling them gold | done |

## What this leaves

Four things stand between the current tool and what was agreed, in the order they matter:

1. **The desktop app.** Decided second of everything and never started.
2. **Claim-level reading**, which is both the promised depth default and the only honest way to
   compute the mention rate that every headline figure uses.
3. **Summaries and drill-down**, which is the difference between reporting that 24.7% of
   players mention difficulty and telling a reader what they said about it.
4. **Induced per-game categories**, so a game's own subjects appear rather than only the
   twenty-four every game shares.
