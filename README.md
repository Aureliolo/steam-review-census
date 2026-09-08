# Steam Review Lab

A tool for finding out what players of a game actually think, rather than what the loudest
reviews say.

## The problem it solves

Steam sorts reviews by helpfulness. If you read the top forty to judge a game, you are
reading the reviews other people upvoted, which are also the longest and angriest ones.
That is a measure of agreement, not of how common an opinion is.

The gap is large enough to change conclusions. In one game we measured, a complaint
appeared in 64% of the fifty most-upvoted negative reviews and in 24.7% of all of them.
Both numbers are true. Only the second is a fact about players. Reading the top of the pile
overstates themes by roughly two to two and a half times.

The only way to remove the argument is to hold every review and count.

## What it does

- Take one Steam app ID, or a whole list of them.
- Download **every** review for those games, not a sample.
- Sort each review into a category. Use the built-in set of categories or define your own.
- Summarise, per category, what people are praising or complaining about.
- Build an overall picture of the game from those category summaries.
- Click through from any number, anywhere, to the actual reviews behind it.

You choose which model does the sorting, and how closely it reads.

### One category, and only more when a review earns it

Every review gets a single primary category. That keeps the counts clean: the categories
add up to the number of reviews, and a percentage means what it appears to mean.

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

Deep costs more time and finds more. Neither setting drops a review.

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

- **"Every review" means every review Valve will serve.** Reviews from deleted and private
  accounts are counted in Valve's totals but cannot be retrieved by anyone. In practice
  this leaves a fraction of a percent unreachable.
- **The target moves.** New reviews arrive constantly, so a corpus is a snapshot with a
  timestamp, and it can be topped up rather than rebuilt.
- **Counting words is not understanding them.** A category assignment is a useful summary
  and a starting point for reading, never a substitute for it. That is why drilling down to
  the underlying reviews is a first-class feature and not an afterthought.

## Data

This repository holds no review data. Reviews belong to the people who wrote them and to
Valve, and they are downloadable by anyone with the app ID, so there is nothing to gain
from redistributing them here. Everything you pull stays on your machine.

## Status

Early. The pipeline behind it has been run in anger against a corpus of 1.7 million reviews
across 62 games, but the tool itself is being assembled from that work.
