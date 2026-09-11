# Decisions

What has been settled about this tool, and where each thing actually stands. It exists because
a long build forgets: a decision made once and not written down gets quietly dropped, or worse,
asked again as though it were open. Nothing here is a proposal. Every row was chosen.

State means: **done** is built and in use; **partial** is built for one case and not the rest;
**not built** is exactly that, however long ago it was agreed.

## The product

| Decided | State |
|---|---|
| Rust core, **Tauri desktop shell**, and a CLI beside it | done: one binary, a window when opened with no arguments and the pipeline when given any |
| Compiled binary for Windows, Linux and macOS; the user downloads it, double-clicks it, and works entirely in the UI | built for all three by the release workflow, DirectML on Windows and CoreML on Apple Silicon; the claim reader is fetched by checksum on first use once one is published |
| The UI is first class, not a wrapper over the pipeline, and has to be good enough to look at | the window crawls, reads, shows the counts with their measured error, opens every row onto its claims and every word that stands out onto the claims that use it. Not yet: the report page's timeline, languages and induced subjects, and a language switch |
| Results also export as one self-contained HTML page that fetches nothing | done |
| Name: `steam-review-census` | done |

## What the analysis does

| Decided | State |
|---|---|
| Headline figure is the **mention rate**, always labelled as such | done |
| Every other percentage says which denominator it uses | done |
| **Deep, claim-level by default**: a review is split into the points it makes, and each point carries a category | done; the splitter is `claims-4` and the reading pass counts per claim |
| Shallow is the opt-out, and neither depth drops a review | done; `--depth shallow`, recorded in the reading and named on the page as not comparable |
| Taxonomy is a **fixed core spine plus induced game-specific extras** | spine done (`core-5`); the induction chain runs end to end: `census distinct` draws the sample, an agent reads it against `reference/induction-brief.txt`, `census ingest-induced` refuses any subject without three real reviews behind it, and the report shows what survives under the table with its evidence and no invented rate. First run on 296970 (Renowned Explorers): nine subjects returned, nine kept, every one refining a spine row or naming something the spine cannot (mood combat, the explorer roster, the oddball enemies, save-scumming), each with four to fourteen reviews behind it |
| Extras are discovered by an **LLM reading an embedding-diverse sample** | the sample is farthest-point traversal over a hash-drawn pool of four thousand vectors, measured to put a mechanics review, a localisation joke, a crash report and a difficulty complaint in its first eight picks. The reading is one agent call per game, 70k tokens on Opus for a 120-review handout, run one at a time behind the labellers |
| Categories are assigned across the full corpus by a **linear probe over embeddings** | superseded: a fine-tuned encoder with abstention, which is a probe that can say no |
| **Corrected prevalence**: the measured error corrects the rate rather than sitting beside it | done, per subject, where the model finds it better than chance |
| **Summarise, per category, what people praise and complain about** | built without a paraphrase: each side of a subject shows the words it uses that the other does not, counted by reviewers and ranked by log-odds z-score against the other side, and every word opens onto the claims it was counted from. Counted during the reading pass in bounded memory. A written summary by a hosted model is the upgrade, when one is configured |
| **Build an overall picture of the game from those summaries** | **not built** |
| **Click through from any number to the reviews behind it** | done in the app, every subject opens onto its claims a page at a time; the report quotes eight per subject |
| Helpfulness bias ships as a column on every category | done |
| Irony and ratings that disagree with the text are flagged, not filed away | flagged on labelled reviews only |

## Models and setup

| Decided | State |
|---|---|
| A complete census with **zero setup and no API key**; hosted models are an upgrade, never a requirement | done |
| Embeddings run locally, on ONNX Runtime | done |
| `ort` with DirectML, CoreML and CPU | done |
| Embeddings carry dedupe, taxonomy, search and classification, not just one of them | dedupe done, search partial; classification moved off embeddings onto the trained reader, which is the right call because embeddings are dominated by sentiment and length rather than subject |
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
| Watermark top-up so a re-crawl does not re-pull old reviews | superseded by the sweep below, which finds arrivals and edits in one walk. The top-up wrote a new snapshot holding only the new reviews, and every pass read the newest snapshot, so a topped-up game counted only what had arrived since |
| **Periodic sweep by last-edit date**, to catch reviews edited since the crawl | done: `census sweep`, and "Bring it up to date" in the window. One walk in `updated` order, newest first, stopping a day past the watermark (the crawl, or the last sweep). Rows land in `sweep-<unix>.parquet` beside the crawl's shards, never over them, and `newest.json` records which copy of each swept id counts; every reader of the capture goes through one walker that skips the rest. The readings record when the capture last changed, and the page and the window say so when a sweep has landed since they were made |
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
| Claims labelled by Fable 5.1 agents, one game each, shown the text alone | done for 33 of 36 games, one labeller at a time on the user's instruction |
| **Opus spot-checks the labels** | done as a blind second reading of a tenth: 1,400 claims over thirty games, subject kappa 0.85 |
| Roughly 400 labels to start | 17,249 claims and counting, target 20,000 |
| **The test set becomes gold: the user adjudicates it by hand**, a random sample of about a thousand claims labelled blind for a representative accuracy figure, then the roughly four hundred the two labellers disagreed on, to settle the boundaries | decided 2026-09-11. Not started: it waits on `core-6`, because adjudicating against a sheet about to change spends the one resource that cannot be spent twice. The tool for it is the app's own claim view with the labels hidden |
| 30 to 35 mid-size games, mixed sentiment, small corpora acceptable | done, 36 games drawn |
| Stratified subset trains, random subset measures, and the two are never merged | superseded at claim level: **whole games** are held out and the frozen ones choose nothing. A game's role is fixed by a hash of its own id, so adding games moves none; over the 36 drawn that is 8 frozen (214490, 620980, 774361, 1057090, 1274570, 1466860, 1809540, 2881650), 4 validation (275850, 1295660, 1465360, 1601580), 24 train. The earlier shuffle reassigned every role on every run, which was found when fifteen games froze a different pair from eleven |
| Measured error **corrects the reported prevalence** | done, on the report page, per subject where the model finds it better than chance |
| The sets are a silver standard, and the README says so rather than calling them gold | done |

## The classifier

Settled after the shipped classifier was opened up in the app and found to be filing "gfg",
"game" and "☺" under Graphics and art. Prototype anchors are not being tuned, they are being
removed.

| Decided | State |
|---|---|
| The unit is a **claim**, not a review. A review is split into the points it makes and each point carries one subject | splitter written and tested |
| A **fine-tuned multilingual encoder** replaces prototype similarity, distilled from Fable labels, exported to ONNX, run on the existing runtime | built, training on each new wave of labels |
| The **backbone is chosen by bake-off**, not by reputation: several candidates, identical labels, identical frozen split, judged on per-category F1 and throughput together | done, below: `gte-multilingual-base` wins on every measure but speed, and the speed it gives up is a quarter, not a multiple |
| **Calibrated abstention**: a claim below threshold is recorded as unclassified and counted, never folded into `verdict` | built, and the threshold is chosen by **the most coverage available at a promised accuracy**, never by maximising accuracy times coverage, which collapses to answering everything |
| **Polarity is predicted per claim**, and reported per review per subject as praised, criticised or **mixed** | done: a second head on the same trunk, and the report counts praise, complaint and mixed per subject |
| **Mention rate stays the headline** because it is verbosity-proof; claim share is deep-reading only and always labelled as verbosity-weighted | rule |
| Model is **multilingual**, reports default to **English** with a working language switch | done: the model reads every language the labels cover, the window reads English by default with "every language" a choice beside the button and a one-click switch to the other reading, the CLI takes `--language`. A switch is a re-read, because the model has to read the claims it skipped |
| Labelling is **cluster-stratified in round one**, then **active learning** (uncertainty crossed with diversity) | round one is stratified and running; active learning waits on the first full pass |
| **One game first**, about 2,000 claims, then decide whether to commit the rest | done, the Frostpunk 2 pilot below, and the rest was committed |
| The 26 games already labelled at review level get **re-labelled at claim level**, so old and new are the same sample | in progress: they are among the 36 in the claim-level run |
| Full MLOps, **reproducible in this repository**: soft targets from labeller confidence, automated label auditing, frozen test set, per-language and per-length slice metrics, calibration, ONNX parity assertion, a CI gate that fails on F1 regression, hash-versioned label sets, plus self-training and multi-seed ensemble as measured experiments | frozen set, calibration, parity assertion and the second-reading audit are done; slice metrics, the CI gate, soft targets and the experiments are **not built** |
| Run tracking is a local JSON log carrying git sha, data hash, config and metrics. No external service | done, `run.json` in every training run |
| Model and dataset published to **Hugging Face**. The dataset is review ids, claim offsets and labels, **never review text**, because this repository holds no review data | script and fetch path built; nothing published yet, so the pin in `reader.rs` is empty |

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

Then fifteen games, 7,680 claims, on a frozen set fixed by hash so it cannot move again:

| Question | Answer |
|---|---|
| How good is it on a game it has never seen? | Accuracy **0.476**, macro F1 **0.385**, polarity F1 0.700, on three frozen games |
| How much does it answer, and how well? | **27% of claims at 0.747**, in [0.696, 0.792] over 316 claims. Four games earlier it was 13% at 0.620 |
| Does the promise transfer now? | **Within a point**: 0.757 promised on validation, 0.747 delivered frozen |
| Why did it start transferring? | Area under the risk-coverage curve fell from 0.543 to **0.351**. The model learned when it does not know, which is the whole of what lets a threshold carry to a new game |

The frozen games are different games from the eleven-game run, because the old split
reassigned them; so the before and after are not over the same reviews. What is not in doubt
is the direction, and that labels are what moved it.

Verified in the tool as before: 315 claims answered at 0.746 against training's 316 at
0.747, one claim apart, which is the half-precision export reordering a single tie. And the
per-game spread says something the pooled figure hides: on the labelled claims, 1057090
declined 61% and agreed on 76.5%; 1466860 declined **80%** and agreed on 70.2%. 1466860 is
the game whose labeller called modding its dominant theme and had nowhere to file it.

Over the whole corpora the picture is softer: 66.5%, 72.9% and 83.4% declined for 1057090,
1466860 and 1601580, against a usual 73.2%. The labelled samples are stratified towards
claims that name a subject, so they show a gap more sharply than the corpus does. The
reader now carries its usual decline rate and the tool says when a corpus is declined at 1.2
times that, which none of these three reach; 1601580 at 1.14 is the nearest, and its labeller
reported a hard seam rather than a missing row.

Then the set was read a second time, a tenth of it, by a different labeller working blind:

| Question | Answer |
|---|---|
| Do two labellers agree on the subject? | **86.4%, kappa 0.85**, over 456 claims read twice across ten games. That is a set two readers agree about, not one reader's habit |
| On polarity? | 93.0%, kappa 0.89 |
| On whether a claim is contested? | **60.3%, kappa 0.28.** The first labeller flagged a fifth of claims, the second three fifths. They are not applying the same bar |
| Does the flag still mean something? | Yes. On the 181 claims neither flagged, the two agree on the subject **100%** of the time; where either flagged, 77.5%. The flag finds the right claims; what differs is how readily each labeller reaches for it |
| Where do they disagree on subject? | `genre` against `verdict` most of all ("great platformer"), then `gameplay` against `genre`, `content` against `gameplay`, `updates` against `verdict`. Every one is a boundary already in `reference/GAPS.md` |

So a contested rate is a fact about a labeller as much as about a game, and the reports say so
rather than comparing it across games as though it were the same measure. The per-field
figures are what `census compare-labels` prints, and the contested check runs every time.

At thirty games read twice, **1,400 claims**, every figure held: subject 86.9% at kappa
**0.85**, polarity 93.4% at 0.90, contested 74.1% at **0.48** with the first labeller flagging
31% and the second 49%. On the 657 claims neither flagged they agree on the subject 99.2% of
the time, and on the 743 either flagged, 76.0%. Their commonest disagreement is now
`difficulty` against `gameplay`, twenty-five claims of it, ahead of `genre` against `verdict`
(14) and `updates` against `verdict` (10); every one of them is a boundary
`reference/GAPS.md` already holds wording for. Three times the claims, twenty more games, and
the same numbers to a point: that is a silver standard behaving like one.

Then the backbone was put to the bake-off it was always going to face, on sixteen games and
8,608 claims, every candidate on the same split and schedule, judged on the validation games
only so the frozen ones stay unread by the thing choosing:

| Backbone, five epochs | Macro F1 | Accuracy | AURC | Answers at 75% | Claims/s |
|---|---|---|---|---|---|
| `xlm-roberta-base`, what shipped | 0.375 | 0.443 | 0.378 | 23% | 1,431 |
| `microsoft/mdeberta-v3-base` | 0.276 | 0.354 | 0.442 | 16% | 830 |
| `intfloat/multilingual-e5-base` | 0.423 | 0.482 | 0.317 | 32% | 1,428 |
| **`Alibaba-NLP/gte-multilingual-base`** | **0.449** | **0.518** | **0.295** | **34%** | 1,048 |

At three epochs the order was the same and the gaps wider, so it is not a schedule effect.
`gte` answers half as many claims again as the shipped backbone at the same promised
accuracy, its confidence ranks claims better (AURC 0.295 against 0.378), and it costs a
quarter of the throughput, which on a corpus of thirteen million claims is an hour on the
card this was measured on. `e5` is the runner-up and as fast as `xlm-roberta`; `mdeberta`
is slower and worse at everything, whatever its reputation. `gte` exports to the same ONNX
graph shape with parity intact (largest drift 8.3e-03 at half precision, no answers changed),
which was the condition for being a candidate at all.

Trained on its own on the same 5,988 training claims and measured on the same four frozen
games (1,800 claims), `gte` answers **37% at 0.803** where `xlm-roberta` answers 28% at
0.762; frozen accuracy 0.608 against 0.507, macro F1 0.512 against 0.405, AURC 0.236 against
0.322. The chain is verified against it as before: the tool answers **666 claims at 0.803**
across the four frozen games, which is training's figure to the claim and to the decimal,
on a different tokenizer and a different architecture from the last time this was checked.
Per game it runs from 74.5% (1466860, the modding game) to 84.2% (1809540), and over the
whole corpora it declines between 54% and 71% against a usual 63%.

Then twenty-three games, 12,011 claims, five frozen (214490 joined them), 2,495 frozen claims:

| Question | Answer |
|---|---|
| How much does it answer, and how well? | **43% of claims at 0.808**, 1,071 claims, at a threshold of 0.78 chosen on the validation games |
| How good is it on a game it has never seen? | Accuracy 0.606, macro F1 0.479 over the five, polarity F1 0.760, calibration error 0.082, AURC **0.217** |
| Does the tool agree? | 1,070 answered at 0.807 across the five corpora, one claim from training's 1,071 at 0.808, which is the half-precision tie again |
| Per game? | 76.3% on 1466860 to 87.0% on 1057090; declined 47% to 63% of labelled claims, and 57% is now the usual figure the reader carries |

Then twenty-seven games, 14,163 claims, six frozen (2881650 joined them), 2,797 frozen claims,
and this is the installed reader:

| Question | Answer |
|---|---|
| How much does it answer, and how well? | **47% of claims at 0.798**, 1,315 claims, at a threshold of 0.77 chosen on the validation games |
| How good is it on a game it has never seen? | Accuracy 0.601, macro F1 0.487 over the six, polarity F1 0.766, calibration error 0.109, AURC **0.210** |
| Does the tool agree? | 1,218 answered at 0.804 over the 2,614 labels it could join. Not to the claim this time, and why is below |
| Per game? | 71.6% on 1274570 to 87.6% on 1057090; declined 42% to 59% of labelled claims, and 53% is now the usual figure the reader carries |

Nine games earlier it answered 37%; the labels are still what moves it. Coverage rose four
points and agreement fell one, which is the threshold moving down the same risk-coverage curve
rather than a better or worse model: AURC, the figure that does not depend on where the
threshold sits, improved from 0.217 to 0.210.

### The splitter changed under the labels, and the labels held

`claims-4` shipped mid-run after all, which the plan above said not to do. What made it
safe is that a label had been carrying the byte span of its claim since the reference sets
were redrawn, so the join in `census measure-claims` could be moved from claim index to span:
a label whose span the new splitter still cuts as one claim finds its reading wherever that
claim now sits, and a label whose span it no longer cuts is counted as **unjoined** and said,
on the terminal and on the report page, rather than scored against whatever sentence now has
its old index. The span join reproduced the index join to the claim before the splitter
moved, which is what licensed moving it.

The rules, each from a labeller's report in `reference/GAPS.md`: three or more short
comma-separated parts are that many claims; a short answer joins the question before it; a
heading tag opened mid-sentence is emphasis; a line that ends on a comma continues on the
next; an emoticon belongs to the sentence before it; "i.e." and "z.B." are abbreviations
with their dots in; a bare `[list]` line is not a piece; a tagged heading weighs what its
words weigh and keeps its colon.

Measured with the wave5 reader on the same five frozen games: **1,003 claims answered at
0.810** over the 2,361 labels the new splitter still cuts as labelled, against 1,070 at 0.807
over 2,495 under `claims-3`. The 134 labels it no longer cuts, 5.4%, are the comma lists and
the fragments the labellers reported, split or joined as they asked; the agreement on the
rest did not move. So the splitter can improve between labelling runs without a relabel,
at the price of the labels it improves past.

A reading now records the splitter that cut it, and a reading cut by an older one is refused
wherever a claim would be quoted or scored by its index, in the report, the measure and the
window, with the counts themselves left standing, until the game is read again. Before this
the measure would have joined a stale reading silently and reported a number that was wrong
by however many indexes had shifted, which is how it was first run and why the guard exists.

The chain was first verified end to end at eleven games and re-verified at every wave since.
Training measures the frozen games in Python, on the full-precision weights, from the claim
text as labelled. The tool measures them in Rust, on the half-precision ONNX graph, over a
corpus it split itself and joined back to the labels by review id and the span each label
names. At eleven games the two agreed on **221 claims answered and 0.620 agreement**, to the
claim; at twenty-three, on 1,070 against 1,071 at 0.807 against 0.808. That is the splitter,
the export, the join and the reader all reproducing one number, which is the only way to know
that none of them is quietly wrong.

At twenty-seven it is 1,218 at 0.804 against 1,315 at 0.798, and the gap is the splitter
rather than a fault: training reads the claim text as the labeller was shown it, cut by
`claims-3`, while the tool reads the corpus as `claims-4` cuts it, and 183 of the 2,797
labelled claims name spans it no longer cuts. Every figure the model card quotes still comes
from training on the labelled text, which is the measurement that does not depend on the
splitter at all. The two come back into exact agreement when the sets are redrawn under
`claims-4` for `core-6`, and until then the difference is reported rather than smoothed.

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

## Where this is going

Settled 2026-09-11: **the dataset is the thing, and the benchmark comes before the polish.**
The aim is the reference people cite for what players say about games, which is a higher bar
than a tool that works. Measured against that bar, four of the seven things such a dataset
needs are already exceeded or met: it is reproducible without redistributing a word anybody
wrote, every label names the splitter and taxonomy it was made under, two labellers read a
tenth of it blind and agree at kappa 0.85, and the test games are fixed by hash and chose
nothing. Two are missing, and no amount of further labelling closes either.

1. **Human-adjudicated labels.** Everything so far is a model agreeing with a model, which the
   README says plainly and which no citation can rest on. The user adjudicates: a random
   thousand from the frozen games, labelled blind, for an accuracy figure that means what it
   says; then the four hundred the two labellers split on, shown both answers, to settle the
   boundaries. After `core-6`, never before.
2. **A comparison against the alternatives.** Nobody has shown this beats asking a frontier
   model directly, or a trivial baseline. Both run over the same frozen claims, all three
   figures reported together, whichever way it falls.

Then, in order:

3. **`core-6` and `claims-5` together, once.** `reference/GAPS.md` holds the wording for every
   rule, each traced to a labeller who could not see the others. The contested rate of 29% and
   the `difficulty` against `gameplay` confusion say the sheet is the ceiling now, not the
   model. Measure the relabel cost on one game before paying it for thirty-six.
4. **Games chosen for the rows that are starved**, not more games at random. `licensing` has
   32 claims over four games, `vr` 40, `accessibility` 48, and all three score zero. A random
   game costs the same as a chosen one and buys almost none of them.
5. **Publishing**, which is the user's decision and not near.
6. **Induced per-game categories** for the remaining games, one agent call of about 70k tokens
   each, behind everything else.

Built since this list was first written: the report page on readings, the polarity split,
corrected prevalence, the second reading and its comparison, the fetch-by-checksum path, the
words that stand out on each side of a subject, the paragraph, the bake-off, the timeline,
languages and induced subjects in the window, the sweep, the language switch, `claims-4` and
the span join that let it ship mid-run.
