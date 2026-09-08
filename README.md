# Steam Review Census

> ## ⚠️ IN DEVELOPMENT: DO NOT TRUST THE NUMBERS YET
>
> There is a working command-line tool. There is no release, no installer and no interface.
>
> The pipeline runs end to end: it downloads every review Valve will serve, embeds them
> locally, sorts them into categories and reports what it found. How often the sorting is
> right is now measured, on **one game**, against labels **written by a language model
> rather than a person**: it agrees with those labels on 72% of held-out reviews the
> labeller called clear-cut, and on 31% of the ones it called contested. That is a real
> number with a real error bar, and it is not accuracy.
>
> Everything is subject to change, including the whole approach. Do not depend on this, and
> do not quote its output as fact.

A tool for finding out what players of a game actually think, rather than what the loudest
reviews say.

## The problem it solves

Steam sorts reviews by helpfulness. If you read the top forty to judge a game, you are
reading the reviews other people upvoted, which are also the longest and angriest ones.
That is a measure of agreement, not of how common an opinion is.

The gap is large enough to change conclusions. In one game measured in earlier work, a
complaint appeared in 64% of the fifty most-upvoted negative reviews and in 24.7% of all of
them. Both numbers are true. Only the second is a fact about players. Reading the top of
the pile overstated that theme by a factor of 2.6.

The only way to remove the argument is to hold every review and count.

## What it does

- Take one Steam app ID, or a whole list of them.
- Download **every** review for those games, not a sample.
- Sort each review into a category. Use the built-in set of categories or define your own.
- Summarise, per category, what people are praising or complaining about.
- Build an overall picture of the game from those category summaries.
- Click through from any number, anywhere, to the actual reviews behind it.

You choose which model does the sorting, and how closely it reads.

Today the first two work, and the third works only against the built-in categories. The rest
is described here because it is what the tool is for, not because it exists; Status, at the
bottom, says exactly what runs.

### What a percentage means here

The headline figure for a category is a **mention rate**: the share of all reviews in the
corpus that say something about it. A review that covers two topics counts towards both, so
mention rates across categories add up to more than 100%, and they are meant to. This is
the number the 24.7% above refers to, and it is the one people usually have in mind when
they ask how common a complaint is.

Where a figure is not a mention rate, it is labelled. Two others appear:

- **Primary share**, the share of reviews whose single main subject is that category. These
  are exhaustive and add up to the number of reviews.
- **Claim share**, used in deep reading, where the unit counted is an individual opinion
  rather than a whole review.

No percentage is ever shown without saying which of the three it is. The point of counting
everything is lost if the denominator is ambiguous.

### One category, and only more when a review earns it

Every review gets a single primary category. That keeps the primary shares clean: they add
up to the number of reviews, and the figure means what it appears to mean.

A review that genuinely covers more than one thing also records the other topics it
touches, so "great simulation, awful interface" is not filed under one of those and
stripped of the other. A review about a single thing gets a single category and nothing
else. Secondary topics are recorded when they exist, never invented to fill a slot.

### Depth is how closely each review is read

Every review is analysed. Depth does not decide how many are included, it decides how
finely each one is taken apart:

- **Shallow** treats a review as one opinion about one thing.
- **Deep** breaks it into the separate points it makes, so a long review contributes
  several distinct opinions instead of one blurred average of them.

Deep reading is the default, because it is also what makes a mention rate correct: a review
counts towards a category when any of its points belongs there. Shallow is faster and
systematically understates multi-topic reviews. Neither setting drops a review.

### Ratings that disagree with the text

A thumbs-down is not always a complaint. "0/10, haven't slept in three days" is praise
wearing a costume, and counting it as negative quietly poisons every number downstream.
The reverse is just as common: a recommendation that is really a warning not to buy yet.

These are **flagged, not filed away**. The flag records that the rating and the text
disagree; the review still sorts by what it is actually about, so the praise buried in a
joke review still counts as praise of whatever it praises. You can then include, exclude
or inspect the flagged ones deliberately, instead of discovering later that they were
silently miscounted.

## Honest limits

- **"Every review" means every review Valve will serve, and the request parameters decide
  how many that is.** Two API defaults quietly remove a large and biased slice.
  `purchase_type` defaults to Steam purchases only, dropping activated keys, and
  `filter_offtopic_activity` defaults to excluding review bombs. On Cyberpunk 2077, asking
  for everything returns 986,295 reviews where the defaults return 881,295. That is 105,000
  missing, 10.6% of the corpus, and the loss is not even: **17.3% of all negative reviews
  disappear against 9.6% of positive ones.** A census that accepts the defaults understates
  exactly what it is trying to measure. This one always sets both.

- **The default pagination is worse than the defaults above.** Steam's `filter=all` is
  helpfulness-ranked and stalls almost immediately. On a 2,909-review title it returned 21
  unique reviews before repeating its cursor: 0.7% coverage, drawn from the top of the pile.
  Reading that would reproduce precisely the bias this tool exists to remove. The census
  paginates by creation date instead, which on that same title returned all 2,909 reviews
  with no duplicates.

- **Some reviews are genuinely unreachable.** Reviews from deleted and private accounts are
  counted in Valve's totals but cannot be retrieved by anyone. Every corpus therefore
  reports its own coverage, fetched against Valve's stated total, as a figure you can read
  rather than a claim you have to trust.

- **The target moves.** New reviews arrive constantly, so a corpus is a snapshot with a
  timestamp, and it can be topped up rather than rebuilt. Reviews are also edited and
  deleted after the fact, and vote counts drift, so a top-up is periodically followed by a
  full re-crawl.

- **A category has to be described to a machine before it can be counted, and describing it
  is not enough.** Comparing each review against a written description of a category is the
  version that needs no setup, and on the one game measured it agreed with the reference
  labels on 51% of the clear-cut reviews. Descriptions are prose about a topic; reviews
  about that topic are not. Rebuilding each category out of reviews people actually wrote
  took the same measurement to 72%. It also needs labelled examples, which is why the
  description path remains the default for any game that has none.

- **Fitted categories are fitted to a game.** The anchors in this repository were built from
  one title's reviews and measured on held-back reviews of that same title. Whether they
  carry to a different game, a different genre or a different language is **not measured**,
  and the tool says so out loud when you point them at another app.

- **Counting words is not understanding them.** A category assignment is a useful summary
  and a starting point for reading, never a substitute for it. That is why drilling down to
  the underlying reviews is a first-class feature and not an afterthought.

## Data and privacy

This repository holds no review data. Reviews belong to the people who wrote them and to
Valve, and they are downloadable by anyone with the app ID, so there is nothing to gain
from redistributing them here.

By default nothing you pull leaves your machine. The crawler talks to Valve, and
categorisation runs on a local embedding model, so a complete census is possible with no
account, no key and no network beyond Steam itself.

Configuring a hosted model changes that in a specific and bounded way: the text of a
sampled subset of reviews is sent to that provider to induce categories, label training
examples and write summaries. The full corpus is never sent. Which provider, and whether to
use one at all, is your choice, and the tool records what ran against every number it
reports.

Steam review text, author IDs and profiles are public. A local corpus of millions of
accounts is still personal data, so it stays on your disk, and exporting a corpus offers
de-identification.

## Status

Five commands work, from a checkout, with no account and no API key:

```sh
census crawl 296970      # every review Valve will serve, into Parquet
census embed 296970      # vectors, computed locally
census fit 296970        # category anchors, rebuilt from labelled reviews
census classify 296970   # categories, mention rates and bias factors
census evaluate 296970   # agreement against a reference set, with intervals
```

**What works.** Crawling is sharded by date, resumable after an interruption, and can top up
an existing corpus rather than rebuilding it. Each crawl reports the share of Valve's own
stated total it actually retrieved. Embedding runs on the machine, on the GPU where one is
available, at roughly 280 reviews a second. Categories come from a fixed seventeen-item core
spine, and every percentage says which denominator it uses. `fit` rebuilds those categories
from labelled reviews, choosing how far to trust them by cross-validation, and leaves a
category with no labels exactly as written. `evaluate` reports agreement split by how the
reviews were sampled and by whether the labeller found the call contested, each with a 95%
interval, because the subsets are small enough that bare percentages mislead.

**What does not.** There is no graphical interface, no installer and no release. The
game-specific categories, the written summaries and the flag for reviews whose rating
disagrees with their text all need a model and are not built. Agreement is measured against
**one** game and against labels a model wrote, so it is consistency between two models
rather than correctness; until a person checks part of that set, no figure here should be
called accuracy.

Prior work sits behind this: the approach has been run against 1.7 million reviews across 62
games, which is where the 64% and 24.7% figures above come from. That work predates the
parameter findings recorded under Honest limits, so those corpora were gathered with filters
that omit roughly a tenth of the reviews and a sixth of the negative ones. Both figures will
be re-measured against freshly crawled corpora before they are presented as anything more
than indicative.

There is no release, no version, and no timeline.

## Licence

Apache License 2.0. See [LICENSE](LICENSE).
