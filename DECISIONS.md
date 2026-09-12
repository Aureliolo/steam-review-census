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
| Name: **SteamGauge**, binary `steamgauge` | done 2026-09-11, renamed from `steam-review-census`. A census counts heads; this reads opinions, and "to gauge opinion" is the phrase for it. `gauge` alone was left to ThoughtWorks' test framework and npm's progress bar |

## What the analysis does

| Decided | State |
|---|---|
| Headline figure is the **mention rate**, always labelled as such | done |
| Every other percentage says which denominator it uses | done |
| **Deep, claim-level by default**: a review is split into the points it makes, and each point carries a category | done; the splitter is `claims-4` and the reading pass counts per claim |
| Shallow is the opt-out, and neither depth drops a review | done; `--depth shallow`, recorded in the reading and named on the page as not comparable |
| Taxonomy is a **fixed core spine plus induced game-specific extras** | spine done (`core-5`); the induction chain runs end to end: `steamgauge distinct` draws the sample, an agent reads it against `reference/induction-brief.txt`, `steamgauge ingest-induced` refuses any subject without three real reviews behind it, and the report shows what survives under the table with its evidence and no invented rate. First run on 296970 (Renowned Explorers): nine subjects returned, nine kept, every one refining a spine row or naming something the spine cannot (mood combat, the explorer roster, the oddball enemies, save-scumming), each with four to fourteen reviews behind it |
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
| **Periodic sweep by last-edit date**, to catch reviews edited since the crawl | done: `steamgauge sweep`, and "Bring it up to date" in the window. One walk in `updated` order, newest first, stopping a day past the watermark (the crawl, or the last sweep). Rows land in `sweep-<unix>.parquet` beside the crawl's shards, never over them, and `newest.json` records which copy of each swept id counts; every reader of the capture goes through one walker that skips the rest. The readings record when the capture last changed, and the page and the window say so when a sweep has landed since they were made |
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
| The **backbone is chosen by bake-off**, not by reputation: several candidates, identical labels, identical frozen split, judged on per-category F1 and throughput together | done, below: `gte-multilingual-base` wins every measure but speed among models of its size, and the first model tried at twice that size beats it by more than every other setting in the sweep combined |
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
figures are what `steamgauge compare-labels` prints, and the contested check runs every time.

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

**Every candidate in that table is about 278M parameters, which is the question the bake-off
did not ask.** It compared four encoders of one size and concluded which family reads claims
best. Asked on 2026-09-12 what a bigger one does, `intfloat/multilingual-e5-large` at 560M
answered 84.9% of validation claims against `gte`'s 68.1% on the same labels, and that is four
times the spread between two seeds. The runner-up family at twice the size beats the winner at
one size by more than every other setting in the sweep put together. A bake-off is only ever an
answer about the axis it varied.

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

### Eighty per cent of what?

`steamgauge ceiling` answers the question every agreement figure in this file begs. A model
trained on one labeller's reading cannot be more right than two labellers manage with each
other, so 80% against one labeller is a score out of that ceiling and not out of a hundred.
Over the claims read twice **and** answered by the model, split by what each game was to it:

| The model | games | claims | two labellers agree | model, where they did | where they split |
|---|---|---|---|---|---|
| **never saw** | 6 | 103 | **93.2%** | **81.2%** [0.723, 0.878] | 85.7% of 7 |
| chose its threshold on | 3 | 64 | 92.2% | 76.3% | 80.0% of 5 |
| trained on | 17 | 456 | 89.0% | 94.6% | 94.0% of 50 |

Only the first row is a measurement. The third is the model reciting labels it was trained
on, and it is in the table precisely because the gap between 94.6% and 81.2% is what holding
whole games back is for: a tool that pooled all three would report 90.4% and mean nothing by
it. The first pooled figure this was run on did exactly that, before the placement the trainer
uses was ported into the tool and a test pinned the eight frozen games and the four validation
ones against it.

So the honest sentence about the current reader is: **on games it has never seen, over claims
it commits to, where two independent labellers reached the same subject, it agrees with them
81.2% of the time**, against a ceiling of 93.2%. Ninety-six claims is a thin plank, and the
interval says so. Widening it is what the hand-adjudicated set is for; more model-written
second opinions would only move the ceiling, not the floor.

### The splitter changed under the labels, and the labels held

`claims-4` shipped mid-run after all, which the plan above said not to do. What made it
safe is that a label had been carrying the byte span of its claim since the reference sets
were redrawn, so the join in `steamgauge measure-claims` could be moved from claim index to span:
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

## The model was being asked a question the labeller never had to answer

Every labeller read each claim inside the review it came from. Every model read it alone. "it
doesn't", "same here", "this too" are unanswerable on their own, and every one of them in the
training set was a label the model was asked to reach from text that cannot reach it.

Measured on the validation games, all three runs identical but for the input:

| input | budget | accuracy | macro F1 | polarity | answers at 75% |
|---|---|---|---|---|---|
| claim alone | 128 | 0.577 | 0.550 | 0.777 | 54% |
| claim alone | 256 | 0.586 | 0.566 | 0.773 | 54% |
| claim in review | 256 | 0.634 | 0.616 | 0.808 | **66%** |

The middle row is the control, and it is the point: the same token budget spent on the claim
alone buys nothing, so the twelve points of coverage come from the context and not from the
larger window. One training run, no labelling, no quota.

The budget is spent **centred on the claim** rather than from the start of the review.
Truncating a pair from the end gives a claim at the foot of a long review the opening
paragraph and nothing it is near, at exactly the same price.

What centring is worth depends entirely on how tight the budget is, which is obvious once
stated and was not obvious before it was measured:

| window | budget | answers at 75% |
|---|---|---|
| from the start of the review | 96 | 57% |
| centred on the claim | 96 | **66%** |
| from the start of the review | 256 | 66% |
| centred on the claim | 128 | 68% |
| centred on the claim | 256 | 66% |

At 96 tokens the head of a review usually misses the claim entirely and centring is worth nine
points. At 256 the head usually reaches it anyway, and centring is worth nothing measurable.
So centring does not beat a long window, it **buys the same answer at half the budget**, and
that is a cost saving rather than an accuracy gain. Between 96 and 256 tokens, centred, every
window scores the same; 128 ships because it is the cheapest of the ones that cannot be told
apart.

What it costs took three measurements to get right, and the first two were wrong in
instructive ways.

1. **On a drawn sample: "about 2x".** Wrong, and wrong in the way sampling is always wrong
   about a corpus. A few hundred drawn claims are distinct by construction, so they said
   deduplication was buying nothing. Over Cyberpunk's 1.55M English claims, 87% are distinct:
   real, but only an eighth of the forward passes.

   Most of that eighth is bought back rather than lost. A claim read in context is a different
   question in a different review, but the same question in every **copy** of the same review,
   because the window is cut from the text: the same text at the same index gives the same
   window and the same answer. So a context reading files its answers by the review's text
   rather than by its id, and "Great game." written as a whole review a thousand times is one
   forward pass again. Every reading records how many claims it counted and how many it
   actually asked about, so the saving is a number in the file rather than an argument.
2. **By token arithmetic, then on a CPU: "9x", then "7.8x".** A claim is 21 tokens and a claim
   with its review is 168, so the work must be eight times greater. On a CPU it is: 51.7
   claims a second against 6.6.
3. **On the machine that does the reading: 2.35x.** The same corpus, the same graph, the same
   pipeline, only the flag differing: 20.3 seconds against 47.8. A GPU running 21-token
   batches is mostly idle, waiting on launches rather than arithmetic, so longer sequences
   cost far less than their token count. A full library read goes from ninety minutes to about
   three and a half hours, once per model version.

The lesson is the one the project keeps relearning: measure the thing itself, on the hardware
that will do it, not a proxy for it.

## Most of a sweep is the seed

Forty configurations were trained on one set of labels overnight on 2026-09-11, every one of
them scored on the validation games only. Ranking them and taking the top row is how a sweep is
usually read, and here it would be wrong almost every time.

Train one configuration twice, changing nothing but the seed, and it lands **0.5 to 4.3 points
apart** on how many claims it answers: `win-128` on three seeds gives 64.5%, 68.1% and 68.8%.
The interval printed beside a coverage figure is about 1.8 points, and it is the wrong bar: it
is the noise in scoring one finished model on 2,615 claims and says nothing about training the
same thing again. A configuration has to clear **4.3 points** before it has changed anything.

Against that bar, almost nothing the small model was asked changed anything. Everything here
lands inside the band and bought nothing at all: batch 16, every window length from 96 to 256,
marking the claim inside its window, a polarity weight of 0.2, down-weighting the contested and
the mis-cut claims, and eight or twelve epochs against five.

Five settings are genuinely worse. A learning rate of 1e-5 answers 47.8%, three epochs 52.3%, a
batch of 64 54.3%, and a polarity weight of 1.0 59.7%. The fifth is the interesting one:
**weighting the rare subjects up makes macro F1 worse, which is the figure it exists to fix.**
At exponents of 0.3, 0.5 and 0.7 on the inverse frequency, coverage falls to 55%, 47% and 42%
and macro F1 falls with it, 0.612, 0.605 and 0.581 against the baseline's 0.637. A rare row is
rare because the corpus rarely says it, and shouting its few examples louder adds no evidence
about it while costing the rows that had some.

Two settings are genuinely better, and only one of them is small. **5e-5 answers 73.5%**, 4.7
points clear of the best of the three baseline seeds, and it is a plateau rather than a peak:
7e-5 and 1e-4 answer 73.0% and 73.3%, and three seeds of 5e-5 land between 72.5% and 73.5%. And
**a backbone of twice the size answers 84.9%**, which is eleven points clear of that and four
times the seed spread, on the same labels and the same schedule:

| `multilingual-e5-large`, 560M | answers at 75% | macro F1 | AURC | ECE |
|---|---|---|---|---|
| five epochs, 2e-5 | 84.9% | **0.669** | 0.141 | 0.181 |
| the same on another seed | 84.1% | 0.662 | 0.140 | 0.182 |
| five epochs, 3e-5 | **86.5%** | 0.659 | **0.138** | 0.190 |
| five epochs, 1e-5 | 76.9% | 0.646 | 0.156 | 0.164 |
| three epochs, 2e-5 | 78.2% | 0.656 | 0.150 | **0.135** |
| `gte-multilingual-base`, 278M, the same schedule | 68.1% | 0.637 | 0.183 | 0.133 |

Two seeds land 0.9 points apart where the small model's land 4.3 apart, and every rate tried
beats every configuration of the small model, so this is the backbone and not a lucky run. What
it costs to read a library with is the open question, and cost is the whole argument for this
project over a frontier model, so it is measured on the card that does the reading before it is
chosen. Its calibration is the other: worse than the small model's at every rate, and
calibration is what carries a threshold from the validation games to the frozen ones. Three
epochs is the fallback that keeps the calibration and gives up six points of coverage.

`training/sweep.py` prints that table and that spread from the run records, and it is the first
thing to run before calling any configuration a winner.

The sweep cannot say anything about the frozen games, because no sweep run ever reads them, so
the one question it left open was whether the worse calibration would cost the transfer: the
expected calibration error climbs from 0.066 at 1e-5 to 0.203 at 5e-5, and calibration is
exactly what makes a threshold chosen on the validation games hold on the frozen ones. That has
failed before. **It did not fail here.** Trained again with the frozen games evaluated, the
chosen configuration promised 75.0% and delivered 76.3% on them, and the small model at 5e-5
promised 75.0% and delivered 75.1%. Worse calibration inside the validation games turned out to
be compatible with a threshold that carries; it is still the thing to check, not the thing to
assume.

## What it has to beat

A number with nothing beside it says only that the thing runs. Three baselines now run over
the same claims, the same split and the same selective-prediction protocol, on the **frozen**
games that chose nothing:

| | accuracy | macro F1 | AURC | answers at 75% |
|---|---|---|---|---|
| commonest subject | 0.233 | 0.015 | 0.731 | never reaches the promise |
| bag of words (TF-IDF, word and character n-grams) | 0.441 | 0.347 | 0.339 | 32% |
| nearest subject centroid, untuned backbone | 0.423 | 0.372 | 0.442 | **5%** |
| the trained reader, claim alone, 278M | 0.605 | 0.504 | 0.216 | 58% |
| the same, claim in review, at the rate that suits it | 0.667 | 0.555 | 0.160 | 77% |
| the same again, on a backbone of twice the size | **0.709** | **0.611** | **0.127** | **84%** |

The last three rows are one change at a time, each measured on the frozen games, each promising
75% and delivering it: 0.749, 0.751 and 0.763. Reading the claim inside its review and training
at the learning rate that suits that is worth nineteen points of coverage; the bigger backbone
is worth seven more on top and most of the macro F1.

The third row is what this project did before it trained anything, and the last column is why
it stopped. Cosine distance to a prototype has no way to say "this is about nothing", so its
confidences carry almost no ordering: asked to be right three times in four, it can answer one
claim in twenty. The bag of words is the honest floor, it takes seconds to fit, and a
278M-parameter encoder that could not clear it would not be earning its electricity.

### Things tried that bought nothing, so nobody tries them again

**A better uncertainty score.** The reader abstains on the largest softmax probability, which
is the obvious choice and not usually the best one. Two others cost nothing from the same
forward pass: the margin between the best class and the runner-up, and the entropy of the
whole distribution. Over 2,615 validation claims they are indistinguishable, AURC 0.2226
against 0.2242 and 0.2231, and all three answer 54.5% at the accuracy they promise. Max
probability stays. `training/confidence.py` re-runs it against any reader.

**Three epochs instead of five.** 0.588 accuracy and 52% coverage against 0.639 and 67%. The
model was not overfitting at five; it was underfitting at three.

### The frontier model wins, and that is the finding

Measured 2026-09-11, with Claude Opus 5 as the frontier model. It was given the category sheet
the labellers work from,
471 frozen claims stratified twenty to a subject, each inside the review it came from, and the
same right to abstain the reader has. The shipped reader was then run over **exactly those
claims**, because the reader's usual frozen figure is over the natural distribution, which is
a quarter `verdict`, and two numbers from two distributions are not a comparison:

| on the same 471 claims | answers | accuracy where it answers | macro F1 | polarity |
|---|---|---|---|---|
| Claude Opus 5, zero-shot, given the sheet | **99.6%** | **87.0%** [83.6, 89.7] | **0.873** | 94.1% |
| the reader, claim alone, 278M (2026-09-11) | 60.7% | 74.8% [69.5, 79.5] | 0.525 | 87.4% |
| the reader, claim in review, 560M (2026-09-12) | 85.4% | 73.6% [69.1, 77.7] | 0.648 | 87.2% |

It is not close, and the gap that closed is coverage rather than accuracy. The bigger backbone
answers a quarter more of the claims at the same accuracy, which the intervals say is the same
accuracy and not a worse one, and reaches two thirds of the frontier model's macro F1 where the
small one reached three fifths. What has not moved is the frontier model answering essentially
every claim, more accurately, on a sheet it was handed once.

Two things to hold onto rather than explain away:

- **The labels were made by a frontier model.** Two labellers agree with each other 93.2% of
  the time on frozen claims, so 87% is close to that ceiling but below it. Some of this figure
  is models of a kind agreeing with each other, and the user's hand-adjudicated gold set is
  what will say how much.
- **What the small model buys is not quality, it is scale.** The 471 claims cost 405,000
  tokens. Cyberpunk alone holds 3.2M claims, which at that rate is about 2.7 billion tokens:
  tens of thousands of dollars, for one game. The reader does the same corpus in about two and
  a half hours on one desktop GPU, for the electricity, with no account and nothing leaving the
  machine. That is the trade, and it should be stated plainly rather than buried.

So the claim this project can honestly make is not that it beats a frontier model. It is that
it gets most of the way there at four orders of magnitude less cost, offline, and that it can
say how far short it falls, because it measured it. The gap is now thirteen points of accuracy
and fifteen of coverage, from twelve and thirty-nine, and it is the number to close.

## Where the reader loses, and what it says about where labels should go

A gap averaged over everything says to work harder. Split by subject, on the same 471
stratified frozen claims, it says what to work on. Measured 2026-09-11 against the 278M reader
that was shipped then, with how much of the set each subject has:

| subject | labels | Opus 5 | the reader | the reader declined |
|---|---|---|---|---|
| licensing | 32 | 90% | **0%** | 65% |
| vr | 57 | 94% | 17% | 44% |
| accessibility | 62 | 76% | **0%** | 71% |
| **atmosphere** | **579** | 85% | **5%** | **70%** |
| language | 64 | 83% | 83% | 17% |
| mods | 152 | 95% | 80% | 15% |
| audio | 226 | 100% | 90% | 5% |

**The failure is abstention, not error.** On the rows it loses, the reader is not answering
wrongly; it is declining 44 to 71% of the claims. That is the design working: it knows it does
not know. It also means the coverage figure, not the accuracy figure, is what more labels buy.

Twenty claims a subject is too few to say anything about a subject, though, and reading that
table as if it could was a mistake made first: it said `atmosphere` was broken. Over all 3,769
frozen claims, where `atmosphere` has 205 of them, it scores F1 0.55, which is the middle of
the pack. The per-subject picture from that larger set is the one to trust:

| | labels | frozen claims | F1, 278M | F1, 560M |
|---|---|---|---|---|
| licensing | 32 | 24 | **0.00** | 0.21 |
| accessibility | 62 | 17 | 0.08 | **0.10** |
| community | 98 | 5 | 0.09 | 0.17 |
| vr | 57 | 17 | 0.25 | 0.34 |
| language | 64 | 6 | 0.37 | 0.50 |
| **mods** | **152** | 35 | **0.92** | 0.80 |
| atmosphere | 579 | 209 | 0.55 | 0.65 |
| gameplay | 3,162 | 604 | 0.55 | **0.67** |
| price | 422 | 77 | 0.77 | 0.86 |
| verdict | 4,114 | 877 | 0.70 | **0.81** |

**The two halves of that table have different cures, and the bigger backbone proved it.** Every
row improved with capacity alone, on the same labels, but not equally: the rows with thousands
of labels gained ten to eleven points (`gameplay` 0.55 to 0.67, `verdict` 0.70 to 0.81), and the
starved rows are still broken (`accessibility` 0.08 to 0.10, `community` 0.09 to 0.17). So a
diffuse row was never short of labels, it was short of a model able to tell two overlapping
things apart; and a starved row is short of labels and no backbone will invent them. `mods`
falling from 0.92 to 0.80 on 35 claims is the width of that sample, not a regression.

**Label count predicts the bottom of that table and nothing above it.** Every subject under
about 150 labels is broken, and every one of them is a row the fifteen new games were chosen
to feed. Above 150 the count stops mattering: `mods` scores 0.80 on 152 labels while
`gameplay` scores 0.67 on 3,162, because `mods` announces itself and `gameplay` is where
everything lands that is not something else. `price` on 422 beats `verdict` on 4,114.

So there are two different problems wearing the same face. Starved rows are fixed by labelling
the games that raise them, which is in progress. Diffuse rows were never short of labels, and
capacity moved them where labels had not: what remains for them is boundary text that says what
they are not.

**The cheapest fix for a starved row is not a new game, it is the claims already in the set.**
`mods` went from nothing to 0.92 by being drawn out of `content` and `updates` by its own
words: 213 claims asked, 155 moved. The same draw for the four broken rows, using the words
`licence`, `adaptation`, `faithful`, `accessibility`, `subtitles`, `remap`, `vr`, `headset`,
`toxic` and their neighbours, finds **458 claims across 34 games** already labelled as
something else. At the rate the `mods` pass moved, that is roughly three times the current
labels for `licensing`, `accessibility`, `vr` and `community` combined, without crawling or
drawing a single new review. It is the first thing to spend labelling quota on.

## The corpus stopped being a corpus of games people like

Measured 2026-09-11 over all 51 captures, 7.5M reviews. Before the fifteen chosen games
landed, 24 of 36 were Very or Overwhelmingly Positive, the median was 86%, and **nothing in
the set was Overwhelmingly Negative**. A classifier trained on that has never read a corpus
where the complaint is the point.

| positive share | games |
|---|---|
| below 40% | 4 |
| 40 to 69% | 10 |
| 70 to 84% | 11 |
| 85% and up | 26 |

The median is still 85%, which is what Steam is: a store where most reviewed games are liked.
What changed is the tail. Fourteen games now sit below 70% and seven below 50%, against
almost none before, and the floor is The Day Before at **15.7%** across 23,177 reviews, which
is the only Overwhelmingly Negative corpus of any size on the platform.

That matters for two different reasons and they are worth keeping apart. A model that has
only read praise has never learned what a complaint about `story` looks like as against a
complaint about `bugs`. And a tool that reports a mention rate has to work on the game that is
one long argument, not only on the game everyone agrees about.

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
   says; then the claims the two labellers split on, shown both answers, to settle the
   boundaries. After `core-6`, never before.

   **The tool exists as of 2026-09-11.** `steamgauge gold` writes one self-contained page that
   fetches nothing and sends nothing, holding 1,000 blind claims from the eight frozen games
   and the 27 the two labellers split on there. A letter picks a subject, a digit the polarity,
   and a claim with both moves on by itself; answers are kept in the browser as they are made,
   because fourteen hundred claims is not one sitting. `steamgauge ingest-gold` reads the
   export back and prints the share that matches the labeller already on record, which is the
   first figure in this project that may be called accuracy. Chrome drives the page in CI.

   Frozen-only gives 27 splits rather than the four hundred first estimated, because a tenth
   of each set is read twice and 271 claims at 93.2% agreement leaves 27. A disagreement
   settles a boundary and measures nothing, so it may come from any game, and the default
   draws from all of them: 1,000 blind and 192 split, out of 1,400 claims read twice.
2. **A comparison against the alternatives. Done, and it does not flatter this project.**
   Four baselines run over the same frozen claims and the same abstention protocol, in "What
   it has to beat" above, including Claude Opus 5 asked the same question directly. It answers
   99.6% of them at 87.0% where this reader answers 60.7% at 74.8%. The honest claim is not
   that this beats a frontier model but that it gets most of the way there for the electricity
   rather than for tens of thousands of dollars a game, and can say how far short it falls.

Then, in order:

3. **`core-6` and `claims-5` together, once.** `reference/GAPS.md` holds the wording for every
   rule, each traced to a labeller who could not see the others. The contested rate of 29% and
   the `difficulty` against `gameplay` confusion say the sheet is the ceiling now, not the
   model. Measure the relabel cost on one game before paying it for thirty-six.
4. **Games chosen for the rows that are starved**, not more games at random. `licensing` has
   32 claims over four games, `vr` 40, `accessibility` 48, and all three score zero. A random
   game costs the same as a chosen one and buys almost none of them. Fifteen such games are
   captured and drawn as of 2026-09-11: two VR games, two about accessibility, five licensed,
   and four at the negative end of a distribution that was 24 Very Positive games out of 36.
   One is labelled (546560, 675 claims, 16 of them `vr`, which is 40% more `vr` than the whole
   set held). The other fourteen are drawn and waiting: 14,036 claims against the 19,582 the
   set holds. What stopped it was the labelling quota, not the plan.
5. **Draw the claims the reader cannot answer, not more claims at random. Built 2026-09-12.**
   Every set so far is a random draw, which is what makes prevalence measurable and is the
   right default. But once a reader exists, the claims it abstains on are worth several times a
   random claim to train from. `steamgauge declined` draws them, uniformly rather than from the
   least confident, because the bottom of a confidence ordering is mostly text with nothing in
   it. Three things keep it from contaminating anything: every row carries `subset: declined`
   and no prevalence figure counts one, the draw refuses any game the model does not already
   train on, and a review already in that game's random set is never drawn twice. It waits on a
   reading made by the current splitter, which the finalise pass produces.
6. **Publishing**, which is the user's decision and not near.
7. **Induced per-game categories** for the remaining games, one agent call of about 70k tokens
   each, behind everything else.

Built since this list was first written: the report page on readings, the polarity split,
corrected prevalence, the second reading and its comparison, the fetch-by-checksum path, the
words that stand out on each side of a subject, the paragraph, the bake-off, the timeline,
languages and induced subjects in the window, the sweep, the language switch, `claims-4` and
the span join that let it ship mid-run, the adjudication page and the ingest behind it, the
frontier comparison, the configuration sweep and the tool that reads it (`training/sweep.py`),
and the teaching draw.
