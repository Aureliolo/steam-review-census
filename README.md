# Steam Review Census

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
in half or more of the fifty most-upvoted reviews and in a fifth of all of them. Both numbers
are true. Only the second is a fact about players. The tool reports the ratio between the two
as a **bias factor**, so the distortion is a number you can read rather than an argument you
have to have.

The only way to remove the argument is to hold every review and count.

## What it does

- Take one Steam app ID, or a whole list of them.
- Download **every** review for those games, not a sample.
- Sort each review into a category. Use the built-in set of categories or define your own.
- Summarise, per category, what people are praising or complaining about.
- Build an overall picture of the game from those category summaries.
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

### One category, and only more when a review earns it

Every review gets a single primary category. That keeps the primary shares clean: they add up
to the number of reviews, and the figure means what it appears to mean.

A review that genuinely covers more than one thing also records the other topics it touches,
so "great simulation, awful interface" is not filed under one of those and stripped of the
other. A review about a single thing gets a single category and nothing else. Secondary
topics are recorded when they exist, never invented to fill a slot.

### Categories are built from reviews, not from definitions

A category has to be described to a machine before it can be counted. Describing it is not
enough: a written definition of a topic is prose *about* that topic, and reviews about it are
not. "amauzing" is nowhere near a paragraph on audio mixing.

So each category is anchored by the reviews that belong to it, taken from labelled reference
sets, and falls back to its written description only in proportion to how few of those exist.
A category with plenty of labelled examples is defined by them; one with none is defined by
its description alone and behaves exactly as it would without any of this. Fitting can
therefore improve a category but never leaves one worse off for lack of data.

A review usually raises several subjects and is mostly about one of them, and those are two
different questions to be good at. Fitting for the second one alone is a trap: the setting
that best identifies a review's main subject is to ignore secondary subjects entirely, which
leaves every category that is usually somebody's *second* subject, graphics and audio and
price among them, with almost nothing to be built from. Anchors are therefore fitted so that
every category counts the same however rare it is, and the fit reports what the other choice
would have scored on the same held-back reviews, so the decision is a measurement rather than
an opinion.

Some categories are defined by what a review does *not* say, and a prototype built only from
its own examples cannot tell those apart: a bare verdict and a joke are both short and about
nothing in particular. So each anchor is also pushed away from the reviews it is measured to
take off other categories, by an amount the fit chooses and can choose to be nothing.

Where two categories genuinely overlap, the taxonomy settles it with a written rule rather
than leaving each labeller to decide: replayability and repetitiveness are amount-of-content,
balance is difficulty, animation speed is graphics, port requests are compatibility, sequel
requests are a verdict. The rules ship with the taxonomy and generate the sheet every
labeller works from, so a boundary can only be defined in one place.

### Depth is how closely each review is read

Every review is analysed. Depth does not decide how many are included, it decides how finely
each one is taken apart:

- **Shallow** treats a review as one opinion about one thing.
- **Deep** breaks it into the separate points it makes, so a long review contributes several
  distinct opinions instead of one blurred average of them.

Deep reading is the default, because it is also what makes a mention rate correct: a review
counts towards a category when any of its points belongs there. Shallow is faster and
systematically understates multi-topic reviews. Neither setting drops a review.

### Ratings that disagree with the text

A thumbs-down is not always a complaint. "0/10, haven't slept in three days" is praise wearing
a costume, and counting it as negative quietly poisons every number downstream. The reverse is
just as common: a recommendation that is really a warning not to buy yet.

These are **flagged, not filed away**. The flag records that the rating and the text disagree;
the review still sorts by what it is actually about, so the praise buried in a joke review
still counts as praise of whatever it praises. You can then include, exclude or inspect the
flagged ones deliberately, instead of discovering later that they were silently miscounted.

## How it knows whether it is right

A census that cannot say how often it is wrong is just an opinion with decimal places.

Every classifier here is measured against **reference sets**: reviews labelled one at a time,
stored under `reference/<app id>/` with a manifest saying exactly what produced them. Each set
is split at sampling time. A subset stratified by predicted category supplies the labelled
examples categories are built from. A separate, randomly drawn subset is held back from that
entirely and is the only part any figure is ever quoted from, because a stratified sample
deliberately over-represents whatever the classifier rarely picks.

Three rules keep those figures honest:

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
- **Reference sets span several games.** Anchors built from one game carry that game's
  vocabulary. Sets are labelled across games of different genres and different overall
  sentiment, so agreement can be measured on a game the anchors were never fitted to:
  `census fit --leave-one-out` fits on every game but one and reports the one left out, which
  is the only figure that says whether a category travels. Numbers from a single corpus
  describe that corpus and nothing else.

## Honest limits

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

- **The target moves.** New reviews arrive constantly, so a corpus is a snapshot with a
  timestamp, and it can be topped up rather than rebuilt. Reviews are also edited and deleted
  after the fact, and vote counts drift, so a top-up is periodically followed by a full
  re-crawl.

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
