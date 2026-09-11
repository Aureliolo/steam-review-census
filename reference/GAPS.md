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

### Praise or blame for the studio that is not about patches: five reports

"Applaud the devs for taking a risk", "hope Bandai sells the IP", "director replacement is
urgent", "fix this 11 bit", and on 2357570 and 2669320 a wall of bare insults at Blizzard and
EA. `updates` is written around post-release patching and `policy` around the publisher and
the platform, so a judgement of the studio itself lands in neither, and both labellers filed
it under `updates` and flagged every one. The revision gives `updates` a second sentence:
praise or blame aimed at the studio as a whole, with no patch, decision or term named, is
`updates`; a named decision is `policy`.

### A patch that removed content: one report, and it was most of that game

Skullgirls' 2023 censorship patch (245170) is the bulk of its set: "not what I paid for",
"content removed", "woke devs". `updates` (a patch changed the game) against `policy` ("terms
that changed after people had bought it"), and the labeller flagged every one that framed it
as the owner's decision. The RULE proposed above for the studio settles this the same way: a
patch is `updates`; the decision behind it, where the claim names it, is `policy`.

### Cannot log in: one report

"Can't log in", "needs a phone number to play", on 2357570. `bugs` when the login is broken,
`policy` when the requirement is the complaint, and the labeller could not always tell which
the reviewer meant. Related to the "will not start" gap below and settled by the same rule.

### A publisher's decision reversed, celebrated in memes: one report, a tenth of that game

Helldivers 2 (553850) after Sony withdrew the account requirement: "DEMOCRACY HAS
PREVAILED", "we won", "For Democracy". The words say nothing about the game and the context
says `policy`, praise; the labeller filed most `offtopic` at low confidence and marked them
ambiguous, which is the honest reading of a sheet that has no line for it. The revision adds
one to `policy`: a celebration or a protest of something the publisher or platform did is
`policy`, praise or complaint by which it was, however it is phrased. The same set held
anti-cheat complaints between `policy` (the kernel driver is the objection) and
`multiplayer` (it blocks joining), settled by the existing line: what a decision is is
`policy`, what it does when playing is where it does it.

### Luck that is sold: one report

FC 25's pack luck (2669320) is `monetisation` (loot boxes) and `difficulty` (luck deciding the
outcome) at once, and "scripting" (the game decides you lose) is `difficulty` by the
fairness clause and reads as a `bugs` accusation. The labeller filed both under `difficulty`
and flagged them. A RULE on `monetisation`: what a purchase gives you is `monetisation`, how
it plays once you have it is whatever it is about.

### A balance change blamed on a patch: one report, and it dominated that game

Tekken 8's set (1778820) is mostly Season 2 complaints, where one sentence names a balance
problem (`difficulty`), the mechanic it broke (`gameplay`) and the patch that did it
(`updates`). The labeller filed the balance under `difficulty` and the blame under `updates`
and marked most of them contested. A RULE on `updates` would settle it: a complaint about what
a patch changed is about the change (`difficulty`, `gameplay`, `content`); `updates` is for
the patching itself, its pace, and whether the developers listen.

### The alien that cannot be fought, and other enemies: two reports

"You can't fight back", the alien's AI on 214490, parry windows, enemy spam and damage
sponges on 1809540: `gameplay` against `difficulty` every time, and the sheet's `difficulty`
rule ("how hard, and whether it is fair") does not say whether an enemy's behaviour is a
mechanic or a difficulty. One line would: an enemy's design is `gameplay`; how hard it is to
beat is `difficulty`.

### Faithful to the film: one report, a third of that game's contested claims

"Captures the feeling of the original film", "true to the movie", on 214490. Between
`licensing` (faithful to the thing it adapts), `atmosphere` (the feeling itself) and
`verdict`. The `licensing` description already says "how faithful the game is to the thing it
is adapting", so the sheet has an answer and the labeller did not trust it against
`atmosphere`. A RULE on `atmosphere` pointing at `licensing` for faithfulness would do.

### Cutscenes that cannot be skipped: one report

Filed under `gameplay` at low confidence on 1809540. It is a `controls` question by the
sheet's "how many clicks it takes to do anything", and nothing says so.

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

### Mods and user content: the most-reported gap there is

Five games in the review era (Blade and Sorcery, Beat Saber, Helldivers 2, Cyberpunk 2077,
SnowRunner) and now three more at claim level, on 1385380, 1465360 and 1592190, where one
labeller called modding "the game's dominant theme" and had to file it under `content` at low
confidence with every claim marked contested. Falls between `content`, `updates` and
`community` with no rule choosing.

The model agrees, without having been asked. On the labelled claims of the three frozen games
under the fifteen-game reader, 1466860 is the one it declines most of, 80% against 61% on the
least, and it is the one where modding is the dominant theme. A subject the spine lacks is a
subject the model cannot commit to. Over the whole corpus the gap is softer, 72.9% against a
usual 73.2%, because the labelled sample is stratified towards subject-bearing claims and
shows it more sharply. Still two independent measurements pointing at one missing row.

Eight games is not a boundary dispute, it is a subject. The revision adds `mods`: "Mods and
user-made content. Whether the game supports them, how easy they are to install, what the
community has made, and whether the game is worth playing without them. A game that is
better modded than shipped is a statement about mods, not about content." With a RULE on
`content` sending anything user-made here.

### Playing with friends: three reports, and it dominated two games

"Better with friends", "fun with mates", "does not grab you alone". Not servers or matchmaking
(`multiplayer`), not calling it a co-op game (`genre`), and not a bare verdict. It was the
commonest qualified claim in DEVOUR's set, BeamNG's labeller (228380) hit the same three
ways of filing it, and Deep Rock Galactic's (548430) split it between `multiplayer` when the
point is solo against co-op and `verdict` when the friends are incidental. "The game is dead"
is its cousin, between `multiplayer` (nobody online) and `updates` (nobody developing), and
the same labeller marked it contested.

### A community's salute on its own: one report, forty claims

"Rock and Stone!", about forty times in Deep Rock Galactic's set, sometimes as the whole
review. It works as an endorsement and passes no judgement in its words, so the labeller
filed it `offtopic`, neutral, and marked every one ambiguous. A reader who knows the game
reads it as praise; the sheet says `offtopic` is for what a reader learns nothing from. The
revision decides it in one line under `verdict`: a community's own catchphrase used as a
salute is a verdict, and it is praise. Twenty games have one ("Praise the Sun", "For Rock and
Stone", "Ave Nex Alea"), and `offtopic` is the wrong bucket for a reviewer declaring
themselves a fan.

### Destruction and physics: one report, and it is what that game is

BeamNG's damage deformation is a system, and reviewers praise how it looks. Sits on the
`gameplay` and `graphics` line, filed `gameplay` and marked contested nearly every time. A
spine row will not fix a game whose whole point is one mechanic; this is what the induced
subjects are for, and 228380 is a good second game to run the induction on.

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

**Done in `claims-4`**, mid-run after all, because a label now names a span of its review
rather than an index, and `census measure-claims` joins by span: a labelled claim the new
splitter cuts differently is counted as unjoined rather than silently scored against the
wrong sentence. Three or more comma-separated parts, each of weight 6 to 18, split; the cut
happens after the fragments are joined, or the short parts would be joined straight back.
Readings record the splitter that cut them, and a reading cut by an older one is refused
wherever a claim is quoted or scored by its index, until the game is read again.

### A question and its answer are one claim, reported on 1601580

The worst-split game so far, 165 of 697. Two shapes the splitter has never seen. Rhetorical
question-and-answer pairs: "Is this Frost Punk?" / "No. Is this a phenomenal game?" and "Want
to see your objectives?" / "Top left of the screen." And the inline scorecard: "Visuals?
fantastic!" / ", Atmosphere?" / "fantastic!, Story?" / "mid at best, Balance?", one review
shredded into fragments that mean nothing apart.

The rule: a question mark followed by a short answer, a few words ending in a full stop or an
exclamation mark, is one claim with the question, not two. The scorecard is the comma list
again with a question mark in it, and the same short-parts rule that fixes the comma list
covers it once "?" is allowed to end a part.

**Done in `claims-4`** for the question and its answer: a piece of weight 30 or under after
a piece ending in "?" joins it. The scorecard is not: "?" still ends a piece, so "Visuals?
fantastic!, Atmosphere? fantastic!" comes out as one claim per question through the
question-and-answer rule, which is the right count, and the tail "Story? mid at best,
Balance? bad" stays one claim of two.

### Abbreviations the list does not have, reported on 1466860

"Since I've been playing the Def. Editions for a while" cut at "Def.". A capitalised word
after a full stop that follows a three-letter capitalised token is a weaker signal than the
list, but "Def." is not in ABBREVIATIONS and "Ed." is not either. Add both, and consider
treating a one-to-four-letter capitalised token before a full stop, followed by a lowercase
word, as an abbreviation regardless of the list. Goes in `claims-4` with the comma list.

**Done in `claims-4`** as list entries: "def", "ed", "ver", "vers", "esp", "resp", "orig"
and "hrs", and the list is now matched with the dots taken out, so "i.e." and "z.B." are
caught when a capital follows them (the 3551340 labeller found two "i.e." cuts). The
capitalised-token rule is not in, because a lowercase word after "Fun." is how most
reviewers start their next sentence. The same labeller reported "the old Champ." cut before
"Manager", which "champ" would cover if it were an abbreviation anybody else used; it is not
added on one report.

### A heading tag used as bold, reported on 1466860 and 1716740

"The biggest improvement might be [h1]EVERYONE WALK FAST[/h1], really nice" came back as
three claims, the middle one shouting and the other two saying nothing. The Chinese
long-form review on 1716740 does it eleven times: `[h2]` opened after a colon or a comma and
closed before one, so the labeller received claims beginning with "，" and "？". The
splitter treats every heading tag as a break because on Steam a heading is a block, but a
reviewer who opens one mid-sentence wanted emphasis, and the text runs straight through it.

The rule: a heading tag breaks only at a line boundary. An opening tag with nothing but
whitespace between it and the previous line break, or the start of the review, begins a
point; a closing tag with nothing but whitespace between it and the next line break, or the
end, ends one. Anywhere else the tag is styling and is skipped like `[b]`. Every real
heading in the same review still splits, since they all sit on lines of their own. Goes in
`claims-4`.

**Done in `claims-4`**, decided at the opening tag: it is a heading where nothing but markup
has been written since the line began or the last point ended, or where the previous
sentence has ended; otherwise it and its closing tag are styling. A heading then ends with
its closing tag wherever that sits, and weighs what its words weigh, so "Cons" joins the
point under it exactly as "Cons" on a bare line does. In the drawn sets 147 of 150 closing
heading tags are followed by a line break, so the line rule and this one agree everywhere
but in the synthetic cases.

### A line that ends on a comma has not finished, reported on 1809540

A Japanese review of the economy (184812704) came back as thirteen claims, four of them
ending on "、" and the next line starting mid-thought: the writer wrapped long sentences by
hand and never used "。" at all. A line break ends a claim, which is right for lists and
headings and wrong here. The rule: a line ending in a comma (",", "，" or "、") joins the line
after it. A comma is never how anybody ends a point, in any script the splitter handles.
Goes in `claims-4`.

**Done in `claims-4`** as written.

### An emoticon after a sentence joins the wrong neighbour

":D", ":(", ";)" and ":|" sit after the sentence they colour, but a fragment below the
minimum weight joins forward, so "Great fun. :D If you like Vermintide..." hands the smile to
the next claim. Five claims across the drawn sets begin this way. A fragment with no letter
in it, on a line that has one before it, should join backward. Goes in `claims-4`.

**Done in `claims-4`** in both shapes: a token of up to four characters with at most one
letter in it, and at least one that is not a letter, following a sentence end on the same
line is run into that sentence's boundary, and a piece that is only such a token joins the
piece before it. "xD" is the one two-letter emoticon named. A bare letter never counts,
because "A" after a full stop is the next sentence starting. Not covered: an emoji that
opens a line and is followed by words on it (548430 reported a pickaxe and a heart handed
to the line after), because on the next line it is as often that line's bullet.

### A bold heading before a list stood alone, reported on 548430

"The Good:" and "Pro Tips:" came back as claims of their own with their items after them,
though a heading ending in a colon has joined what it introduces since `claims-2`. The
reviews wrote them as `[b][u]The Good[/u][/b]:` followed by `[list]` on a line of its own,
and that bare `[list]` line was a piece: it weighed six characters of markup, took the held
heading, and the two together weighed enough to stand. **Done in `claims-4`**: a run with no
words in it is never a piece, markup weighs nothing, and a heading is read for its colon
with the tags taken out.

### Two shapes that may not be fixable mechanically

- A crash log or a poem cut line by line into fragments that say nothing alone.
- One sentence carrying three subjects ("beautiful art and story", "runs well, isn't
  misrepresented, just not for me"). The comma list in `claims-4` takes the ones written as a
  list; a sentence that names three things in ordinary grammar keeps one subject and the model
  learns the rest from context.

### A review with no punctuation at all, reported on 553850 and 3551340

"屎 闪退bug 各种奇怪bug 游戏内容太少 难度高无脑堆大怪", four complaints separated by spaces and
nothing else, and the same shape in Turkish, Polish and Czech, plus three English rants of two
hundred words with one full stop between them. The splitter has nothing to cut on: no
terminator, no comma, no line break. Splitting on spaces would shatter every ordinary
sentence, and splitting long runs at a word count would cut mid-thought at a place decided by
arithmetic. This is the shape a splitter cannot reach, and the labellers who reported it
labelled the whole run under its first subject, which is the honest reading.
