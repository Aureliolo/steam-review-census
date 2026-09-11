# Gaps the claim labellers reported

Collected as they finish, the same way the core-3 gaps became core-4. Nothing here is acted on
mid-run: changing the sheet while half a set is labelled leaves the other half labelled against
a taxonomy it never saw. These get applied in one revision, `core-6` and `claims-4` together,
after the 36-game run is labelled and measured, and the sets are redrawn against both.

Ordered by how many labellers reported the same thing without being able to see each other's
work, which is the only evidence any of it has. The second reading adds a different kind of
evidence: where two labellers given the same sheet disagree on the subject, the sheet is what
failed. Their commonest disagreements, over 456 claims read twice, were `genre` against
`verdict` (8), `gameplay` against `genre` (5), `content` against `gameplay` (4), `updates`
against `verdict` (4) and `atmosphere` against `verdict` (3). Every one of those is below.

## Defects in the sheet, to fix in the next revision

### `verdict` contradicts itself about money: found in the core-5 run

The `verdict` description offers "worth every penny" and "waste of money" as examples, and the
`verdict` RULE says "super fun, worth every penny" is `price`. A labeller cannot follow both.
`price` owns "value for money" by its own description, so the examples are the mistake: both
phrases judge value and belong to `price`. The rule's example is also poorly chosen, since
"super fun" is gameplay rather than price.

Not fixed mid-run, because changing the sheet while half a set is labelled leaves the other
half labelled against a different one. Labellers are marking these `ambiguous`, so the data
records the uncertainty rather than hiding it.

The revision, ready to apply in `taxonomy.rs` when the run ends: drop "worth every penny" and
"waste of money" from the `verdict` examples; change the RULE's example to "super fun, and
the story is great" belongs to `story`; and add "worth every penny", "waste of money" and
"refunded it" to the `price` examples, which is where the second reading found labellers
already putting them when they disagreed.

## Needs a rule, not a category

### Comparison with the predecessor: three reports, two eras

"Not as good as the first game", "FP1 was a class", "completely different from the original",
"best Anno yet". Reported by both claim labellers and by the review-level labellers before them.
It currently scatters between `genre` (comparison to other games), `verdict` (better or worse)
and `offtopic` (a statement purely about the older game). Labellers marked nearly all of it
contested, which is the definition of a boundary the taxonomy has not settled.

A one-line RULE on `genre` would absorb most of it: a comparison with the game's own
predecessor is `genre` when the difference itself is the point, and `verdict` when it is only
better or worse.

### Praise or blame for the studio that is not about patches: three reports

"Applaud the devs for taking a risk", "hope Bandai sells the IP", "director replacement is
urgent", "fix this 11 bit". `updates` is written around post-release patching and `policy`
around the publisher and the platform, so a judgement of the studio itself lands in neither.

### A game that will not start, with no reason given: two reports

"Bought it, installed it, cannot run it". Splits between `bugs` and `compatibility` with
nothing to choose between them.

## Might need a category

### Lost immersion, emotional distance: three reports, the commonest complaint in one set

"They are just numbers now", "lost its soul", "you cannot feel the people any more", "death is
a statistic". Sits between `story` and `gameplay` and the rules do not settle it. This was the
single most frequent complaint in the Frostpunk 2 pilot, so it is not a rare shape.

Related to the atmosphere and fear gap the review-level labellers reported six times across
two horror games. Both are about what a game makes a player feel rather than about any part
that produces it. One category might take both.

### Mods and user content: four reports, review era

Reported for Blade and Sorcery, Beat Saber, Helldivers 2, Cyberpunk 2077 and SnowRunner. Falls
between `content`, `updates` and `community` with no rule choosing.

### Playing with friends: one report, but it dominated that game

"Better with friends", "fun with mates", "does not grab you alone". Not servers or matchmaking
(`multiplayer`), not calling it a co-op game (`genre`), and not a bare verdict. It was the
commonest qualified claim in DEVOUR's set.

## Confirmed by the core-5 run

`atmosphere` works. On the first horror game labelled against it, it took 40 of 118
aspect-naming claims, nearly every "scary" and "I peed myself" among them. Before it existed
those fell to `verdict` or were scattered across `graphics` and `audio`.

### A verdict that names the genre as a noun: three reports

"Excellent city builder", "among the best platformers", "great platformer in every way". The
`verdict` rule says any named aspect wins and `genre` owns naming what kind of game it is, so
these are forced to `genre` while reading as bare verdicts. Labellers flagged nearly all of
them contested. The sheet should say whether a genre noun counts as naming an aspect.

### Addictive, could not stop playing: one report, a dozen claims

Filed under `atmosphere` by its "pulls you in, lose whole evenings" example, but it reads as a
verdict to many. A single line would settle it either way.

### Pure emotional reaction with nothing named: two reports

"Made me cry", "the feels", "I'm scared", "AHHH". Sits between `atmosphere` and `story`, and
the polarity is about the reviewer rather than the game, which the sheet says polarity is not.

## Smaller, one report each

- Pre-order regret ("I who pre-ordered deluxe am a clown") fits neither `price` nor
  `monetisation` squarely.
- "I refunded it" is listed under `price` but reads as a verdict.
- Complaints about world plausibility ("how does XIX-century tech forecast a storm 88 weeks
  out") fit neither `story` nor `gameplay`.
- A reviewer's own hardware ("using an RTX 3070") sits between `compatibility` and
  `performance`.
- Neutral narrative lines inside a long bug report have nowhere to go but `bugs`.

## Splitter, not taxonomy

Both labellers flagged these through `split_wrong`, and they are fixed in `claims-2` and
`claims-3`: bullet markers, numbered list markers, headings ending in a colon, semicolons,
quotations, parenthetical asides, Steam markup, and web addresses.

### The comma list, reported by five labellers and counting

"Stunning visual, calm music, epic story". "Great story and the sound design is top notch".
"Music 9/10 Buildings 9/10". "Runs well, isn't misrepresented, just not for me". Three subjects
in one claim, so the labeller picks one and marks it contested, and two subjects go uncounted.

This looks fixable after all, and precisely: split on a comma only when the sentence is three
or more comma-separated parts that are each *short*, measured in the same weight the splitter
already uses so it works in scripts without spaces. "Stunning visual, calm music, epic story"
is three parts of weight 14, 10 and 10 and splits; "The combat, which took a while to click, is
superb" has a long middle and does not.

**Not done during the run.** Every game's batches are already drawn under `claims-3`, and a
label points at a claim index. Changing the splitter now would mean the indices in the
reference sets no longer name the same text the reader produces, which silently breaks the
join that `census measure-claims` depends on. It goes in `claims-4`, after this set is
labelled and measured, and the sets are redrawn together.

### Abbreviations the list does not have, reported on 1466860

"Since I've been playing the Def. Editions for a while" cut at "Def.", and "The biggest
improvement might be" left dangling by a cut on the line after it. A capitalised word after a
full stop that follows a three-letter capitalised token is a weaker signal than the list, but
"Def." is not in ABBREVIATIONS and "Ed." is not either. Add both, and consider treating a
one-to-four-letter capitalised token before a full stop, followed by a lowercase word, as an
abbreviation regardless of the list. Goes in `claims-4` with the comma list.

### Two shapes that may not be fixable mechanically

- A crash log or a poem cut line by line into fragments that say nothing alone.
- One sentence carrying three subjects ("beautiful art and story", "runs well, isn't
  misrepresented, just not for me"). Splitting on commas would shatter ordinary prose, so
  these keep one subject and the model learns the rest from context.
