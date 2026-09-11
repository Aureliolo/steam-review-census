# SteamGauge

> **Unreleased and in development.** There is no installer and no release yet. This file
> describes what the tool is and how it is meant to be judged, not a running product. What
> currently works, and what every figure is currently worth, is measured by the tool itself
> and recorded next to the data it was measured from: see the reference sets under
> `reference/` and the release notes when releases begin.

A tool for finding out what players of a game actually think, rather than what the loudest
reviews say.

## The problem it solves

Steam sorts reviews by helpfulness. If you read the top forty to judge a game, you are
reading the reviews other people upvoted, which are also the longest and angriest ones. That
is a measure of agreement, not of how common an opinion is.

The gap is large enough to change conclusions. Measured on real corpora, a theme can appear
in half or more of the fifty Steam ranks most helpful and in a fifth of all of them. Both numbers
are true. Only the second is a fact about players. The tool reports the ratio between the two
as a **bias factor**, so the distortion is a number you can read rather than an argument you
have to have.

The only way to remove the argument is to hold every review and count.

## What it does

- Take one Steam app ID, or a whole list of them.
- Download **every** review for those games, not a sample, and bring the capture up to date
  later without downloading it again.
- Split each review into the points it makes and sort each point into a subject, with a
  measured error and an honest "cannot tell".
- Show, per subject, the words the praise uses and the complaints use, counted by reviewer.
- Build a picture of the game in a paragraph from those counts.
- Click through from any number, anywhere, to the actual reviews behind it.

Results come out as one self-contained page: every rate, the reviews behind it, and what the
classifier is measured to get wrong, in a single file you can open from disk, send to someone,
or print. It fetches nothing, because a corpus that never left your machine should not start
leaving it the moment somebody looks at it.

You choose which model does the sorting, and how closely it reads.

### What a percentage means here

The headline figure for a category is a **mention rate**: the share of all reviews in the
corpus that say something about it. A review that covers two topics counts towards both, so
mention rates across categories add up to more than 100%, and they are meant to. This is the
number people usually have in mind when they ask how common a complaint is.

Where a figure is not a mention rate, it is labelled. Two others appear:

- **Primary share**, the share of reviews whose single main subject is that category. These
  are exhaustive and add up to the number of reviews.
- **Claim share**, used in deep reading, where the unit counted is an individual opinion
  rather than a whole review.

No percentage is ever shown without saying which of the three it is. The point of counting
everything is lost if the denominator is ambiguous.

### The unit is a claim, not a review

A review is not one opinion. "Looks incredible, runs like a slideshow, and the story is the
best in the series" is three, about three different things, and a single vector for the whole
review is their average: a point that belongs to none of them. So every review is first split
into the separate points it makes, and a subject belongs to a point rather than to a review.

That is what keeps the arithmetic honest at both ends. A review that makes twelve points
contributes to twelve subjects instead of being flattened into one. A review that makes one
point can carry exactly one subject, which is the part that matters more than it sounds: a
two-word review cannot be filed under four topics, because it does not contain four.

A review's subjects are the union of its claims' subjects, and it still counts once towards
each of them. The headline is a mention rate, a share of reviews, and it stays that way
deliberately: counting claims instead would let whoever writes most set the numbers, which is
the same distortion the tool exists to expose at the top of the pile.

### What reads a claim

A model trained on labelled claims, and nothing else. It says which subject a claim is about,
whether it is praise, a complaint or neither, and how sure it is. Below a calibrated threshold
it says nothing at all, and those claims are reported as unclassified rather than filed under
whichever category happened to be nearest.

That last sentence is the whole of what changed. The previous classifier compared a review to
twenty-four category prototypes and kept the nearest ones. A prototype comparison has no way
to express "this is about nothing", so every string got a subject: a review reading "gfg" was
filed under graphics and art, and one reading "this game is a lot of fun" under community and
players. Those are not edge cases. Steam is full of two-word reviews, and each one of them was
adding a fraction of a percent to a rate that was supposed to be a fact about players.

The model is fine-tuned from a multilingual encoder and shipped as an ONNX graph, so it runs
on the same local runtime as everything else. No API key, no network, no account. Which
encoder it starts from is decided by measurement across candidates on identical labels and an
identical split, judged on accuracy, throughput, and how well its confidence tracks whether it
is right, because a model that cannot tell when it is guessing cannot be allowed to abstain.

**The threshold is chosen by what it promises, not by how much it answers.** The obvious way
to pick one, maximising accuracy times coverage, has a degenerate optimum on a model that is
not yet good: coverage rises faster than accuracy falls all the way down, so the sweep settles
at its own floor and the model is told to answer everything. Measured, it chose 0.05, which
across twenty-five subjects is barely above the 0.04 a uniform guess scores, and a corpus of
17,596 claims came back with nothing declined at all. That is the prototype's failure wearing
a trained model's clothes. The threshold is now the one giving the most coverage at a promised
accuracy, and when no threshold reaches that accuracy the model is recorded as not good enough
rather than quietly lowered to whatever it can manage.

Some categories are defined by what a claim does *not* say. A bare verdict and a claim that
says nothing about the game are both statements that no aspect was named, and on a corpus of
real reviews they are the commonest labels there are.

Where two categories genuinely overlap, the taxonomy settles it with a written rule rather
than leaving each labeller to decide: replayability and repetitiveness are amount-of-content,
balance is difficulty, animation speed is graphics, port requests are compatibility, sequel
requests are a verdict, a system nobody explains is tutorial however good the system is,
calling a game co-op is naming a kind of game and belongs to genre, and a protest about
anything the publisher did belongs to policy even when it names no particular term. The rules
ship with the taxonomy and generate the sheet every labeller works from, so a boundary can
only be defined in one place.

Categories are added when labellers report having nowhere to put something, not when
somebody thinks of one. Every category in the current spine was asked for by the people
labelling against the previous one.

### Depth is how closely each review is read

Every review is analysed. Depth does not decide how many are included, it decides how finely
each one is taken apart. **Deep** is the default and is described above: a review becomes the
points it makes. **Shallow** treats the whole review as one point, which is faster and
systematically understates anyone who wrote more than a sentence. Neither setting drops a
review.

### Praised, criticised, or both

A thumb is attached to a review, not to a subject. Somebody who loves the art and despairs of
the framerate has one thumb and two opposite opinions, and crediting both subjects with the
same verdict is a straightforward misreading of what they wrote.

So polarity belongs to the claim, and is reported per review per subject: of the reviews that
discuss performance, the share that criticise it, the share that praise it, and the share that
do both. **Mixed** is a real answer and appears as one. It is the most interesting thing a long
review has to say, and any tool that forces it to a single sign is throwing that away.

Claim-level polarity is available underneath, for reading rather than for headlines, and is
labelled as what it is: a count of opinions, which the most talkative reviewers dominate.

### What they said about it

A fifth of reviewers complaining about performance is a count. Whether they mean stutter,
crashes or load times is the thing a reader opened the row for, and no local model is asked to
paraphrase anybody to say it. Instead, each side of a subject shows the **words that stand
out**: the terms its complaints use far more than its praise does, and the reverse, each with
the number of reviewers who used it. "stutter 84, crashes 61, memory leak 23" under
performance is what was said, counted, and every term opens onto the exact claims it was
counted from.

The comparison is complaint against praise within one subject rather than against the corpus,
because against the corpus a subject's vocabulary is mostly its own name: "fps" stands out in
every performance claim and tells nobody anything. The ranking is a log-odds z-score with a
half-count prior, so a word three reviewers used and nobody on the other side did does not
outrank one three hundred used against ten, and a side with a handful of reviews shows nothing
rather than promoting whatever those few happened to write. Counts are by reviewer, once per
review however often it repeats itself, for the same reason the headline is a mention rate.

### Ratings that disagree with the text

A thumbs-down is not always a complaint. "0/10, haven't slept in three days" is praise wearing
a costume, and counting it as negative quietly poisons every number downstream. The reverse is
just as common: a recommendation that is really a warning not to buy yet.

These are **flagged, not filed away**. What a labeller flags is what the text does, judged
from the words alone: they are never shown whether the reviewer recommended the game, so that
they cannot be led by it, and so they are in no position to report that the two disagree.
Putting that flag next to the rating is what finds the disagreement, and that is arithmetic
rather than a judgement. The review still sorts by what it is actually about, so the praise
buried in a joke review still counts as praise of whatever it praises.

The flag lives on labelled reviews and nowhere else, which is deliberate rather than an
omission waiting to be filled. Reading it off a whole corpus needs a classifier measured to
find irony, and nothing here has measured that yet. What the reference sets buy in the
meantime is the rate: how often, in reviews drawn at random, the text and the rating point
opposite ways at all.

## How it knows whether it is right

A census that cannot say how often it is wrong is just an opinion with decimal places.

The model is measured against **reference sets**: claims labelled one at a time, stored under
`reference/claims/<app id>/` with the drawn sample beside them. Whole games are held out rather
than whole claims, because two claims from one review are not independent evidence and a score
that mixes them is a score for how well the model repeats itself.

Three things are reported together, and separating them is what makes the number mean anything:

- **Agreement**, over the claims the model was willing to answer for.
- **Abstention**, the share it declined. A score that quietly drops those is a score for a
  classifier nobody is running.
- **Contest**, the share the labeller marked as genuinely ambiguous, reported apart from the
  rest. Disagreement there says as much about the taxonomy as about the model.

### What it is worth against the alternatives

Measured 2026-09-11 over frozen games, which chose nothing about the model. Every row abstains
where it is unsure, and every row is scored only on what it answered, because a score that
quietly drops the declined claims is a score for a classifier nobody is running.

| | answers | accuracy where it answers |
|---|---|---|
| the commonest subject | everything | 23.3% |
| TF-IDF bag of words | 32% | 44.1% |
| nearest subject centroid over an untuned encoder | 5% | 42.3% |
| **this reader** | **61%** | **74.8%** |
| Claude Opus 5, given the same sheet | 99.6% | 87.0% |

The third row is what this project did before it trained anything, and it is why the rebuild
happened: cosine distance to a prototype cannot say "this is about nothing", so at the accuracy
it promises it can answer one claim in twenty.

The last row is the one worth being honest about. **A frontier model asked directly is better
than this, by twelve points of accuracy and thirty-nine of coverage.** What it is not is
affordable: that comparison cost 405,000 tokens for 471 claims, and a single large game holds
three million claims. This reader does that game in about two and a half hours on one desktop
GPU, offline, for the electricity. The claim being made is not that a 278M-parameter model
beats a frontier one. It is that it gets most of the way there at four orders of magnitude less
cost, and that it can tell you exactly how far short it falls.

### How a person turns silver into gold

`steamgauge gold` writes one page, holding a blind random sample of claims from the games the
model never saw and the claims two labellers answered differently. Blind means blind: a claim
drawn for measurement carries no answer, because an answer on the page is an answer in the
reader's head, and a figure produced by agreeing with a suggestion is a ratification rather
than a measurement.

The page is a single file that fetches nothing and sends nothing, marks each claim inside the
review it came from, and keeps answers in the browser as they are made. A letter picks a
subject, a digit picks the polarity, and a claim with both moves on by itself; a thousand
claims is not one sitting and a closed tab must not cost a night's work.

`steamgauge ingest-gold` reads the export back, files it beside the labels already there
rather than over them, and prints the share that agrees. That share is the first number this
project can call accuracy rather than agreement.

### How the reference sets are made

They are a **silver standard**, not a gold one, and the distinction decides what every number
downstream may be called. A gold standard is adjudicated by people. These labels are written by
a language model reading one claim at a time, which makes the set good enough to train a model
on and to measure against, and never good enough to quote as truth. Every manifest records
`human_verified: false`, and until that changes the tool reports **agreement** and refuses the
word accuracy.

Which model wrote them is recorded per set, in `produced_by`, and printed with every result.
A set labelled by one model and a set labelled by another are not the same evidence and must
not be pooled without saying so; the sets shipped here were written by Claude Fable 5.1.

What the set spends its size on is games rather than depth. A hundred reviews of one title
would say nothing about whether a category survives contact with a corpus it was not built
from. So the set runs to thousands of labels spread across dozens of games of different genres
and different overall sentiment, and the figure it exists to produce is the one measured on a
game the model never saw. A tenth of it is labelled twice by different labellers, which is what
lets the set report its own reliability rather than only its agreement with a classifier.

The mix of languages is chosen rather than inherited. A corpus is whatever languages its
players happen to write in, and drawing straight from it would train the model mostly on
whichever one that is. Roughly seven claims in ten are English and the rest are drawn from
everything else the corpus holds, so the model holds up in the languages the reports do not
default to.

The protocol is fixed so it can be repeated, and so a disagreement with it is about the method
rather than about somebody's afternoon:

- **The sample is drawn before anyone reads anything.** `steamgauge sample-claims` takes a seed and
  draws reviews per game with a fixed share of English, then splits each into its claims. The
  same seed against the same capture draws the same reviews, so a set can be rebuilt without
  being stored.
- **Every claim of a drawn review is labelled, never a subset of them.** A review labelled in
  part cannot say what share of a corpus names no aspect at all, which is the first thing worth
  knowing about one.
- **A set drawn to teach the model is marked as one and measures nothing.** A random draw
  spends most of its budget on claims the reader already gets right, so `steamgauge declined`
  draws instead from the claims it abstained on, uniformly rather than from the least confident
  of them, because the bottom of a confidence ordering is mostly text with nothing in it. Every
  row lands with `subset: declined`, which keeps it out of every prevalence figure; only the
  claims drawn are asked about, though the whole review is still handed over, because a claim
  reading "it doesn't" cannot be labelled without the sentence before it; and only games the
  model already trains on may be drawn, because a held-out game taught from is not held out.
- **Labellers are shown the claim inside the review it came from, and nothing else.** Not the
  game, not whether the reviewer recommended it, not what the model guessed. The prediction is
  withheld because anyone shown a proposed answer agrees with it more than someone reading
  cold. The rating and the game are withheld for a different reason: the model does not see
  them either, so a label made from more than the tool can read would measure the gap in what
  the two were shown. The surrounding review is shown because a claim reading "it doesn't" is
  not interpretable alone.
- **The sheet every labeller works from is generated from the taxonomy**, so a boundary rule
  exists in exactly one place and every labeller is given the same one. It is never changed
  mid-run: half a set labelled against a revised sheet is half a set nobody can compare.
- **Labelling runs in parallel, one labeller per game**, each working batch by batch and writing
  each batch out before opening the next.
- **Every label carries six fields**: one subject, whether the claim is praise, a complaint or
  neither, whether the text is ironic, how sure the labeller was, whether the call was genuinely
  contested, and whether the claim was cut in the wrong place. The last two are read back:
  agreement is reported separately over the contested claims, and the mis-split rate is what
  drives the splitting rules. Three rounds of them came from labellers reporting it.
- **A tenth is read again by a different labeller, blind.** `steamgauge second-opinion` draws the
  same reviews as fresh batches with no labels in them, and `steamgauge compare-labels` reads the
  two labellings together. It reports each field apart from the others, because they fail
  differently: subject is a judgement about the claim, and `ambiguous` is a judgement about the
  taxonomy. Beside every percentage is Cohen's kappa, which is what the percentage cannot tell
  you: a corpus is mostly `verdict` and `offtopic`, so two labellers who never read a claim
  would still agree most of the time by landing on the commonest subject.
- **What comes back is checked rather than trusted.** `steamgauge ingest-claims` refuses a set that
  does not cover the drawn sample exactly: claims nobody labelled, labels naming claims nobody
  drew, subjects the taxonomy does not have, claims labelled twice, and labels whose judgements
  were never made. A judgement left out is dropped rather than defaulted, because a `false`
  nobody wrote is a figure nobody stood behind.

These rules keep those figures honest:

- **Agreement is not accuracy.** Where the labels were written by a model, what gets measured
  is consistency between two models, and the tool says so on every report until a person has
  checked part of the set. Two models can be wrong together, most easily on sarcasm and on the
  boundaries between categories, which is exactly where classifiers are weakest.
- **Every rate carries a 95% interval.** Reference sets are small. At fifty reviews the honest
  band around a rate is roughly twenty-five points wide, and a bare percentage invites
  conclusions the counts cannot support.
- **A category that is wrong says what it is wrong about.** Knowing a category scores badly
  says nothing about what to do; knowing it is read as one particular other category is a
  boundary the taxonomy has not settled, and no amount of fitting will settle it. Every
  evaluation names the category each one is most often mistaken for, where that is a pattern
  rather than a single review.
- **Every row says how well it is measured, not just the page.** One agreement figure for a
  whole taxonomy hides the shape of the error: the same run finds nine mentions in ten of one
  category and one in five of another. Each category carries its own precision and recall
  from the reference set, and a category the classifier is measured to miss most of is marked
  as a floor rather than a count.
- **A difference the reader cannot see is not marked as one.** The page draws two kinds of
  claim on top of its numbers, and each is earned rather than automatic. Across games, a
  category is outlined as belonging to one of them only where that game clears every other by
  more than the rounding in the printed figures. Within a game, a category is coloured as warm
  or cold only where the interval on the reviews raising it clears the game's own baseline, so
  eight reviews all recommending the game is left as the 100% it is instead of dressed as the
  finding a thousand reviews at 98% would be.
- **Reference sets span several games, and whole games are held out.** A model trained on one
  game carries that game's vocabulary. Sets are labelled across games of different genres and
  different overall sentiment, and the split holds out whole games rather than whole claims,
  so the figure says whether a subject travels rather than whether the model can repeat itself.
  Numbers from a single corpus describe that corpus and nothing else. The same figure is
  reported again over the claims the labeller called clear-cut and over the ones it called
  contested, because those answer different questions: a model that genuinely cannot reach a
  game is worse where the reading was easy, and a taxonomy that does not fit a game is worse
  only where the labeller could not place the claim either.

## Honest limits

- **Reports default to English, so the headline is a fact about players who write English.**
  The capture is always the whole census, every language in it, and the filter is applied when
  the counting happens rather than when the crawling does, so the choice is reversible and no
  corpus has to be downloaded twice. But a mention rate over English reviews is not a mention
  rate over players: on some corpora English is under half of what was written. Every figure
  says which set it is over, and switching the language recounts from the same capture. The
  reason for the default is that evidence nobody can read is evidence nobody can check, and
  being able to open a rate and read what is behind it is the whole design.

- **The model declines claims it is not sure about, and those are counted rather than hidden.**
  A claim below the threshold gets no subject and is reported as unclassified. That is a real
  answer and an honest one, but it means a mention rate is a rate over the claims the model
  would commit to. The share it declined is printed beside it, and a large one is a finding
  about the corpus rather than a footnote.

  **Right now that share is most of the corpus.** Trained on twenty-seven games, the model
  answers 47% of the labelled claims in games it has never seen and agrees with a labeller on
  80% of those, so half of the claims come back unclassified and a mention rate is a floor
  rather than a count. Every report says so on its face. The only cure is more labelled
  claims, which is what the reference sets are for; a threshold moved to make the number look
  better would be the old classifier again. Sixteen games earlier the figures were an eighth
  and 62%, and the backbone has since been chosen by bake-off rather than by reputation, so
  the labels and the measurements are doing what they are for.

- **A threshold chosen on a few games may not transfer to a new one.** The threshold promises
  an accuracy, and that promise is measured on the games that chose it. On eleven games the
  frozen ones delivered eighteen points less than promised; on twenty-seven they deliver four
  points more. So every figure in the model card comes from the frozen games, and the
  validation figures stay in the run record where they belong. Which games are frozen is fixed
  by a hash of each game's id, so adding games never moves one across the line. And the tool
  reproduced the training measurement to the claim while the two were reading the same cut of
  the reviews: 1,071 answered at 80.8% in Python on the full-precision weights, 1,070 at 80.7%
  in Rust on the half-precision graph over five corpora it split and joined back itself. Since
  the splitter moved under the labels the two read different cuts, and the tool now answers
  1,218 at 80.4% where training says 1,315 at 79.8%, a difference `steamgauge measure-claims`
  reports claim by claim rather than smoothing.

- **Claim share is verbosity-weighted and never a headline.** Counting opinions instead of
  people lets whoever writes most set the numbers, which is the same distortion this tool
  exists to expose at the top of the pile. The headline is always the share of reviews.

- **A tenth of the reference set is labelled twice, and the rest is labelled once.** Reliability
  is measured on that tenth and assumed for the rest. It is a far better position than having
  no second reading at all, and it is not the same as a set where every label was adjudicated.
  The silver standard's own error is estimated rather than known, and the word accuracy still
  does not apply: the labels were written by a model, so what is measured is consistency
  between two models.

  Measured so far, over 1,400 claims on thirty games: two labellers agree on the subject
  87% of the time, kappa 0.85, and on polarity 93%, kappa 0.90. Those are figures a set can
  stand on, and they did not move when the set grew from ten games to thirty.

- **The contested flag measures the labeller as much as the claim.** Two labellers given the
  same definition reached for it on three tenths and on half of the same claims, kappa 0.48.
  The flag does find the right claims: where neither reached for it the two agree on the
  subject 99% of the time, and where either did they agree three times in four. What differs is the
  bar. So a game's contested rate is not compared with another game's, and agreement is
  reported over the contested claims as a floor on how hard the taxonomy is rather than as a
  property of the corpus.

- **A labeller is not told which game it is, but is given one game at a time.** Withholding the
  game keeps the label answerable from the same text the model reads. Handing over a whole
  game's batches undermines that where a corpus has a strong accent: a hundred claims about
  tracking and room scale identify a headset game whatever the sheet says. The effect runs one
  way, towards labels the model cannot reproduce, so it understates the model rather than
  flattering it. Shuffling reviews from several games into each batch would remove it, at the
  cost of routing the labels back per game before they can be ingested.

- **A claim is split mechanically, and the splitting is sometimes wrong.** Across thirty
  labelled games it is **15.8%** of claims, between 8.9% and 26.2% depending on the game, and
  every rule in the splitter came from one of those reports. The commonest failure was a
  sentence that names three subjects at once: "stunning visuals, calm music, epic story" was
  one claim carrying three, so two of them went uncounted; a list of short comma-separated
  parts is now that many claims. The rate is measured rather than assumed, because it is the
  one error in this pipeline that no amount of training fixes.

  A label names the span of the review it was written about, not a position in a list, so the
  splitter can change under a labelled set: a label whose span the new splitter no longer cuts
  as one claim is counted as unjoined and said, rather than scored against whatever sentence
  now sits at its old index. A reading records the splitter that cut it, and a reading cut by
  an older one is refused wherever a claim would be quoted or scored by its index, until the
  game is read again.

- **How often labellers find a claim genuinely contested varies more than the claims do.**
  28.9% overall, but from 13.5% on one game to **52.1%** on another. Some of that is the games,
  and some of it is labellers reading "two subjects both fit" more or less strictly. It is the
  clearest argument for the double-labelled tenth: contested is the one field with no way to
  check itself.

- **"Every review" means every review Valve will serve, and the request parameters decide how
  many that is.** Two API defaults quietly remove a large and biased slice. `purchase_type`
  defaults to Steam purchases only, dropping activated keys, and `filter_offtopic_activity`
  defaults to excluding review bombs. On a million-review title, asking for everything can
  return a tenth more reviews than the defaults do, and the loss is not even: substantially
  more negative reviews disappear than positive ones. A census that accepts the defaults
  understates exactly what it is trying to measure. This one always sets both.

- **The default pagination is worse than the defaults above.** Steam's `filter=all` is
  helpfulness-ranked and stalls almost immediately, returning a fraction of a percent of a
  corpus drawn entirely from the top of the pile before it starts repeating its cursor.
  Reading that would reproduce precisely the bias this tool exists to remove. The census
  paginates by creation date instead, and reports the coverage it achieved against Valve's own
  stated total as a figure you can read rather than a claim you have to trust.

- **Some reviews are genuinely unreachable, and a shortfall must never be blamed on that
  without checking.** Reviews from deleted and private accounts are counted in Valve's totals
  and cannot be retrieved by anyone. Steam separately, and intermittently, just stops serving
  a window early: pages arrive full, then one arrives empty long before the window is
  exhausted, which looks exactly like reaching the end. Walking the same window again returns
  everything. A crawler that accepts the first answer therefore undercounts by a quarter of a
  window at a time while reporting every shard as finished, so this one compares each window
  against Valve's count for it, walks it again when it lands short, and refuses to mark a
  window complete while it stays short. Coverage is reported per corpus either way, because
  the first number is the one worth doubting.

- **The target moves.** New reviews arrive constantly and old ones are edited, so a corpus is
  a snapshot with a timestamp, and it is brought up to date rather than rebuilt: one walk in
  last-edit order fetches everything written or changed since, and stops there. Nothing
  already captured is overwritten. An edited review is held in both forms, with a record of
  which one counts, so a capture never loses what a review used to say. Deleted reviews and
  drifting vote counts are what a sweep cannot see, and only a full re-crawl catches those.

- **A category with no labelled examples is only as good as its description.** Fitting cannot
  invent evidence. Where a game barely discusses something, that category stays weak for that
  game, and the fix is labelled examples from a game that does discuss it, not a cleverer
  scoring rule.

- **Counting words is not understanding them.** A category assignment is a useful summary and
  a starting point for reading, never a substitute for it. That is why drilling down to the
  underlying reviews is a first-class feature and not an afterthought.

## Data and privacy

This repository holds no review data. Reviews belong to the people who wrote them and to
Valve, and they are downloadable by anyone with the app ID, so there is nothing to gain from
redistributing them here.

By default nothing you pull leaves your machine. The crawler talks to Valve, and
categorisation runs on a local embedding model, so a complete census is possible with no
account, no key and no network beyond Steam itself. Reports are the same: one file with no
fonts, scripts or stylesheets fetched from anywhere, so reading a result is not a way of
publishing it.

Configuring a hosted model changes that in a specific and bounded way: the text of a sampled
subset of reviews is sent to that provider to induce categories, label examples and write
summaries. The full corpus is never sent. Which provider, and whether to use one at all, is
your choice, and the tool records what ran against every number it reports.

Steam review text, author IDs and profiles are public. A local corpus of millions of accounts
is still personal data, so it stays on your disk, and exporting a corpus offers
de-identification.

## Licence

Apache License 2.0. See [LICENSE](LICENSE).
