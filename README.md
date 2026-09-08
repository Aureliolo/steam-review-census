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

You choose which model does the sorting, and how deep the analysis goes.

### Categories that reflect how people really write

A thumbs-down review is not always a complaint. "0/10, haven't slept in three days" is
praise wearing a costume, and counting it as negative quietly poisons every number
downstream. The reverse happens too: recommendations that are really warnings. Both get
their own handling rather than being silently miscounted.

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
