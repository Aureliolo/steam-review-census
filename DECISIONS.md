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

## The classifier

Settled after the shipped classifier was opened up in the app and found to be filing "gfg",
"game" and "☺" under Graphics and art. Prototype anchors are not being tuned, they are being
removed.

| Decided | State |
|---|---|
| The unit is a **claim**, not a review. A review is split into the points it makes and each point carries one subject | splitter written and tested |
| A **fine-tuned multilingual encoder** replaces prototype similarity, distilled from Fable labels, exported to ONNX, run on the existing runtime | built, training on each new wave of labels |
| The **backbone is chosen by bake-off**, not by reputation: several candidates, identical labels, identical frozen split, judged on per-category F1 and throughput together | to build; judged on area under the risk-coverage curve as well, since a backbone that knows when it does not know is worth more here than one point of accuracy |
| **Calibrated abstention**: a claim below threshold is recorded as unclassified and counted, never folded into `verdict` | built, and the threshold is chosen by **the most coverage available at a promised accuracy**, never by maximising accuracy times coverage, which collapses to answering everything |
| **Polarity is predicted per claim**, and reported per review per subject as praised, criticised or **mixed** | to build |
| **Mention rate stays the headline** because it is verbosity-proof; claim share is deep-reading only and always labelled as verbosity-weighted | rule |
| Model is **multilingual**, reports default to **English** with a working language switch | to build |
| Labelling is **cluster-stratified in round one**, then **active learning** (uncertainty crossed with diversity) | to build |
| **One game first**, about 2,000 claims, then decide whether to commit the rest | to build |
| The 26 games already labelled at review level get **re-labelled at claim level**, so old and new are the same sample | to build |
| Full MLOps, **reproducible in this repository**: soft targets from labeller confidence, automated label auditing, frozen test set, per-language and per-length slice metrics, calibration, ONNX parity assertion, a CI gate that fails on F1 regression, hash-versioned label sets, plus self-training and multi-seed ensemble as measured experiments | to build |
| Run tracking is a local JSON log carrying git sha, data hash, config and metrics. No external service | to build |
| Model and dataset published to **Hugging Face**. The dataset is review ids, claim offsets and labels, **never review text**, because this repository holds no review data | to build |

## What the pilot found

One game, Frostpunk 2, 2,760 claims labelled by three agents against core-4. Everything here
is measured rather than argued.

| Question | Answer |
|---|---|
| Does the chain work end to end? | Yes: labels, training, ONNX export with a parity assertion at 1.7e-05 and zero answers changed, and a corpus pass |
| What do more labels buy? | 430 training claims gave accuracy 0.25 and macro F1 0.17; 1,629 gave **0.46 and 0.41**, with polarity F1 0.21 to 0.62. Still climbing |
| What does labelling cost? | ~305 tokens a claim, so 20,000 claims is about 6M tokens |
| How often is a claim mis-split? | 12% to 17% by three labellers independently, which drove three rounds of splitter fixes |
| What did labellers have nowhere to put? | Atmosphere and feel, by all three, and by six review-level labellers before them |

Then eleven games, 5,574 claims, measured with games held out whole:

| Question | Answer |
|---|---|
| How good is it on a game it has never seen? | Accuracy **0.318**, macro F1 **0.269**, on the frozen games. The pilot's 0.46 was within one game, which is the easy question; the validation games say 0.427, and they helped build the model |
| How much can it answer at a promised 75% accuracy? | **24% of claims** at threshold 0.42, on the games that chose the threshold |
| Does that promise transfer? | **No.** On the two frozen games the same threshold answers **13%** of claims at **0.620**, not 0.756. It was chosen on two validation games and overfits them |
| Is it calibrated? | Better where it matters least: calibration error 0.066 on the frozen games against 0.132 on validation, but area under the risk-coverage curve 0.543 there against 0.386, so its confidence ranks claims worse on a game it has not seen |
| What does it still not know? | `gameplay` has 257 labelled claims in one frozen game and the model answers **none** of them. `controls`, `difficulty` and `content` likewise. `accessibility`, `community`, `compatibility` and `language` score zero F1 for want of labels |

Then the set was read a second time, a tenth of it, by a different labeller working blind:

| Question | Answer |
|---|---|
| Do two labellers agree on the subject? | **86.4%, kappa 0.85**, over 456 claims read twice across ten games. That is a set two readers agree about, not one reader's habit |
| On polarity? | 93.0%, kappa 0.89 |
| On whether a claim is contested? | **60.3%, kappa 0.28.** The first labeller flagged a fifth of claims, the second three fifths. They are not applying the same bar |
| Does the flag still mean something? | Yes. On the 181 claims neither flagged, the two agree on the subject **100%** of the time; where either flagged, 77.5%. The flag finds the right claims; what differs is how readily each labeller reaches for it |
| Where do they disagree on subject? | `genre` against `verdict` most of all ("great platformer"), then `gameplay` against `genre`, `content` against `gameplay`, `updates` against `verdict`. Every one is a boundary already in the gap notes |

So a contested rate is a fact about a labeller as much as about a game, and the reports say so
rather than comparing it across games as though it were the same measure. The per-field
figures are what `census compare-labels` prints, and the contested check runs every time.

The chain is verified end to end against that last row. Training measures the frozen games in
Python, on the full-precision weights, from the claim text as labelled. The tool measures them
in Rust, on the half-precision ONNX graph, over a corpus it split itself and joined back to the
labels by review id and claim position. The two agree on **221 claims answered and 0.620
agreement**, to the claim. That is the splitter, the export, the join and the reader all
reproducing one number, which is the only way to know that none of them is quietly wrong.

So the model card and every report now quote the **frozen** games, never the validation ones.
A card is read by somebody deciding whether to run this on a game of their own, and the
validation figure answers a different question: how well it does on the games that chose its
settings. The two differ by eighteen points, and quoting the flattering one would be the same
kind of lie as a classifier that never abstains.

The first honest measurement of abstention came from fixing how the threshold is picked. Under
the old objective the model answered every claim at 39% accuracy and declined nothing, which
is the same failure as the prototype wearing a trained model's clothes.

The taxonomy is now **core-5**: `atmosphere` added on that evidence, plus rules for comparing a
game with its own predecessor, for praise or blame aimed at the studio, and for a game that
will not start at all. Every earlier reference set and the shipped anchors are core-4 and are
refused by this build, which is the guard working rather than failing.

## What this leaves

In the order they matter:

1. **Labels.** Eleven games of the thirty-six are labelled. At 20% coverage the model cannot
   carry a report yet, and nothing else on this list changes that: it is the one input every
   other number depends on.
2. **The report page**, which still renders the deleted classifier's counts and has to be
   rebuilt on readings.
3. **Summaries and drill-down**, which is the difference between reporting that 24.7% of
   players mention difficulty and telling a reader what they said about it.
4. **Induced per-game categories**, so a game's own subjects appear rather than only the
   twenty-five every game shares.
5. **Publishing** the model and the ids-and-offsets dataset, and fetching the model by checksum
   so a user who downloads the binary is not asked to train one.
