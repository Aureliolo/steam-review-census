//! The fixed core spine of categories.
//!
//! These are the categories that stay the same across every game, so numbers from different
//! corpora can be compared. Game-specific categories are induced separately and layered on
//! top; nothing here is meant to describe any particular game well.
//!
//! Each category carries a *description written the way a review would say it*, not an
//! abstract label. The classifier compares review vectors against these descriptions, and
//! embedding models place "the frame rate tanks in cities" much nearer to a sentence about
//! stuttering than to the word "performance".
//!
//! A category also carries a **boundary rule**, which is written for whoever is labelling
//! and is deliberately never embedded. Rules are about how to choose between two categories
//! and read like instructions; feeding "prefer difficulty when the complaint is about
//! balance" to an embedding model would drag the anchor towards the vocabulary of
//! instructions and away from the vocabulary of reviews.

/// One category in the taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Category {
    /// Stable identifier. Appears in output files, so changing one invalidates comparisons.
    pub id: &'static str,
    pub label: &'static str,
    /// Embedded verbatim. Must read like the reviews it is meant to attract.
    pub description: &'static str,
    /// How to choose when this category competes with another. Never embedded.
    pub boundary: Option<&'static str>,
    /// Whether a review in this category can be in no other.
    ///
    /// A claim about absence cannot share a review with a claim about presence. "Says
    /// nothing about the game" and "a verdict with no reason given" are both statements that
    /// no aspect was named, so pairing either with an aspect is a contradiction rather than a
    /// second subject, and the labellers read them that way without being told to: across
    /// the six reference sets `offtopic` is alone in 191 labels of 191 and `verdict` in 364
    /// of 374.
    ///
    /// Measured on 2026-09-09, this trades one kind of error for another and leaves the
    /// pooled macro F1 where it found it, at 0.585 against 0.584. What it buys is precision:
    /// `offtopic` goes from 0.47 to 0.56 and `verdict` from 0.74 to 0.84, because neither can
    /// be hedged onto a review whose subject the classifier has already named. What it costs
    /// is the credit those hedges used to earn: `verdict` recall falls from 0.70 to 0.57,
    /// which is the honest figure for how often a bare verdict is actually identified as one.
    /// Kept for the reason it was added rather than for the score: a classifier that can
    /// report a review as naming an aspect and naming none is wrong whatever it scores.
    pub alone: bool,
}

/// Version of the core spine. Any change to the categories or their descriptions changes
/// what the numbers mean, so this is recorded alongside every classification.
pub const CORE_SPINE_VERSION: &str = "core-6";

pub const CORE_SPINE: &[Category] = &[
    Category {
        id: "performance",
        label: "Performance",
        description: "The game runs badly. Low frame rate, stuttering, frame drops, poor \
                      optimisation, long loading times, and it struggles even on good hardware.",
        boundary: Some(
            "Frame rate and stuttering belong here. How the animation itself looks, and how \
             fast it plays out, belong to graphics.",
        ),
        alone: false,
    },
    Category {
        id: "bugs",
        label: "Bugs and crashes",
        description: "The game is broken and buggy. It crashes to desktop, freezes, corrupts \
                      or loses save files, and is full of glitches that block progress.",
        boundary: Some(
            "Something that worked and now does not belongs here. Being unable to log in is \
             here when the login itself is broken and policy when needing an account at all \
             is the complaint. A game that never starts, with no reason given, is \
             compatibility.",
        ),
        alone: false,
    },
    // Narrow on purpose. Phrased as "how the game plays and whether it is fun" this became
    // the nearest match for any general discussion and took 47.7% of primaries, which is a
    // property of the wording rather than a finding about any game.
    Category {
        id: "gameplay",
        label: "Gameplay and mechanics",
        description: "The systems and mechanics themselves. Combat, movement, crafting, \
                      building, exploration, progression systems, and whether the mechanics \
                      are deep or shallow.",
        boundary: Some(
            "Naming the genre or comparing the game to another one belongs to genre. Whether \
             options are balanced against each other belongs to difficulty, and so does how \
             hard an enemy is to beat; how that enemy behaves and what it does is a mechanic \
             and belongs here. How much game there is belongs to content, and anything \
             user-made belongs to mods. A system the game never explains belongs to tutorial, \
             however good the system itself is.",
        ),
        alone: false,
    },
    // 35% of reviews in the first corpus measured named a genre or compared the game to
    // another one. With nowhere to put that, all of it landed in gameplay, which is most of
    // why gameplay held half of all primaries.
    Category {
        id: "genre",
        label: "Genre and comparisons",
        description: "What kind of game this is. A roguelike, turn-based tactics, a \
                      management sim, a deckbuilder, a soulslike. It plays like FTL, it is \
                      XCOM meets Darkest Dungeon, if you liked Slay the Spire you will like \
                      this, it reminds me of the old games in the genre.",
        boundary: Some(
            "Naming what kind of game it is, or which games it resembles, belongs here, and \
             that includes calling it a co-op game or comparing it to other co-op games. As \
             soon as a review says a mechanic is deep, shallow, satisfying or broken, that \
             part is gameplay, and whether the co-op itself works is multiplayer. A \
             comparison with this game's own predecessor belongs here when the difference \
             itself is the point, and to verdict when it is only better or worse.",
        ),
        alone: false,
    },
    // Reviews that give a verdict and name no aspect are common, and without a home they
    // contaminate whichever category happens to sit nearest in the embedding space.
    Category {
        id: "verdict",
        label: "Overall verdict only",
        description: "A verdict with no specific reason given. Great game, terrible game, \
                      ten out of ten, would recommend, do not buy, best game ever, an \
                      excellent city builder, I find this game childish, please make a \
                      sequel.",
        boundary: Some(
            "Only when no aspect is named at all. A review that gives a verdict and then \
             names one specific thing belongs to that thing: \"super fun, and the story is \
             great\" is story. A judgement with only the kind of game attached, \"excellent \
             city builder\", is a verdict; genre is for when what kind of game it is, or \
             which game it resembles, is the point being made. A community's own catchphrase \
             used as a salute, \"Rock and Stone\", is a verdict and it is praise. Asking for \
             a sequel belongs here; asking for a port belongs to compatibility.",
        ),
        alone: true,
    },
    // Joke and meme reviews were being counted as verdicts, which inflates the one category
    // whose whole purpose is to be the honest home of reviews that say nothing specific.
    Category {
        id: "offtopic",
        label: "Says nothing about the game",
        description: "The review is not about the game. A joke, a meme, a copypasta, a story \
                      about the reviewer's day, an argument with another reviewer, a \
                      complaint about Steam or the shop page, or a protest about something \
                      the publisher did elsewhere.",
        boundary: Some(
            "A joke that still makes a point about the game belongs to whatever it is joking \
             about, most often gameplay or difficulty, and a protest about anything this \
             game's publisher or platform did belongs to policy. This is only for reviews from \
             which a reader would learn nothing at all, and that includes a review with \
             nothing in it: a full stop, a row of emoji, a keyboard mash. A single word that \
             does pass judgement, good or terrible or 10/10, is a verdict.",
        ),
        alone: true,
    },
    Category {
        id: "story",
        label: "Story and writing",
        description: "The story, plot and characters. The writing, the dialogue, the humour, \
                      the ending, the world and its lore, the characters players grew fond \
                      of, and whether the narrative is worth following.",
        boundary: Some(
            "Anecdotes from a playthrough belong here when they are about a character or an \
             event, and to gameplay when they are about a mechanic. Humour in the writing is \
             here; a charming art style is graphics.",
        ),
        alone: false,
    },
    // Asked for by nine labellers across two eras and two genres, which is more independent
    // evidence than any other category in this spine had. Every labeller of Alien: Isolation
    // named it, the DEVOUR labeller reached it from a different corpus, and all three claim
    // labellers on a city builder called it the largest gap in the sheet. Without it, "you
    // never FEEL it" and "the deaths are just numbers now" fall to story or gameplay at low
    // confidence, and a report about a horror game cannot say that players found it
    // frightening.
    Category {
        id: "atmosphere",
        label: "Atmosphere and feel",
        description: "What the game makes you feel while you are playing it. Genuinely \
                      terrifying, the tension never lets up, put headphones on and turn the \
                      lights off, it pulls you in and you lose whole evenings, the mood is \
                      oppressive and lonely, it has lost the soul the first one had, the \
                      people are just numbers to you now, you never really feel it.",
        boundary: Some(
            "This is the feeling itself, for reviews that name nothing that produces it. \
             Being unable to stop playing, losing whole evenings and calling it addictive are \
             here, and so is a reaction with nothing named behind it: it made me cry, the \
             feels, I was terrified. Where a review says what creates the mood, that part \
             belongs to what creates it: a frightening creature design is graphics, a \
             soundtrack that unsettles is audio, a mechanic that keeps you on edge is \
             gameplay, faithfulness to the film or book it adapts is licensing, and immersion \
             broken by a crash is bugs.",
        ),
        alone: false,
    },
    Category {
        id: "graphics",
        label: "Graphics and art",
        description: "How the game looks. The visuals, the art style, the animation and how \
                      quickly it plays out, the environments and character models, whether \
                      it is beautiful or ugly or charming.",
        boundary: Some(
            "Animation quality and animation speed belong here. Frame rate belongs to \
             performance.",
        ),
        alone: false,
    },
    Category {
        id: "audio",
        label: "Audio and music",
        description: "How the game sounds. The soundtrack, the music, the sound effects, the \
                      voice acting and the audio mixing.",
        boundary: None,
        alone: false,
    },
    Category {
        id: "controls",
        label: "Controls and interface",
        description: "The controls and the user interface. Clunky or responsive controls, \
                      keybindings, controller support, menus, the HUD, the inventory screens, \
                      and how many clicks it takes to do anything.",
        boundary: Some(
            "Whether a controller is supported belongs here, and so does anything the game \
             makes you sit through or click past: a cutscene that cannot be skipped is a \
             question of how many clicks it takes to do anything. Whether the game runs on a \
             given device belongs to compatibility.",
        ),
        alone: false,
    },
    Category {
        id: "difficulty",
        label: "Difficulty and balance",
        description: "How hard the game is and whether it is fair. Difficulty spikes, \
                      grinding, whether the options are balanced against each other, luck and \
                      randomness deciding the outcome, and whether it is too easy or punishing.",
        boundary: Some(
            "Balance complaints belong here rather than to gameplay, including when they name \
             a specific mechanic as overpowered or useless. How hard an enemy is to beat is \
             here; how it is designed and what it does is gameplay. A complaint about what a \
             patch changed is about the change and belongs here where the change was to \
             balance, with updates for the patching itself. Being lost because nothing was \
             explained belongs to tutorial. What a purchase gives you is monetisation, and \
             how it plays once you have it is here.",
        ),
        alone: false,
    },
    Category {
        id: "content",
        label: "Amount of content",
        description: "How much game there is. Length, how many hours it lasts, replay value, \
                      whether runs differ from one another, whether it gets repetitive, and \
                      whether it feels finished or thin and runs out of things to do.",
        boundary: Some(
            "Replayability and repetitiveness are two ends of one axis and both belong here, \
             never to gameplay. What the players made rather than the developers belongs to \
             mods, however much of the game it accounts for.",
        ),
        alone: false,
    },
    // The most-reported gap in this taxonomy's history: five games at review level, three more
    // at claim level, and one labeller calling modding a game's dominant theme while filing it
    // under `content` at low confidence with every claim marked contested. The model said the
    // same thing independently: the game it declined most of, 80% against 61% on the least, was
    // that one. Across the labelled sets a claim naming mods lands in ten different rows.
    Category {
        id: "mods",
        label: "Mods and user content",
        description: "Mods and what players have made. Whether the game supports modding, how \
                      easy mods are to install, the workshop, what the community has built, \
                      whether the game is worth playing unmodded, and mods breaking when the \
                      game updates.",
        boundary: Some(
            "Anything made by players rather than by the developers belongs here, including \
             how much of the game's life it accounts for. Whether the developers support \
             modding is here; whether a patch broke the mods is here too, because the subject \
             is the mods. Paid mods and creator programmes are monetisation, and what the \
             people who make them are like is community.",
        ),
        alone: false,
    },
    Category {
        id: "price",
        label: "Price and value",
        description: "What it costs and whether it is worth the money. Worth every penny, a \
                      waste of money, I refunded it, full price versus sale, value for money, \
                      and whether to wait for a discount.",
        boundary: Some(
            "What the base game costs belongs here, and so does judging its value in money: \
             worth every penny and waste of money are both about what it was worth paying. \
             What is sold on top of it belongs to monetisation.",
        ),
        alone: false,
    },
    Category {
        id: "monetisation",
        label: "Monetisation and DLC",
        description: "Paid extras and how they are sold. Microtransactions, battle passes, \
                      loot boxes, paywalls, season passes and content cut out to sell as DLC.",
        boundary: Some(
            "What is sold on top of the game, and how, belongs here, including what a purchase \
             gives you: a pack whose contents are decided by chance is sold here, and whether \
             luck then decides the match is difficulty. Wanting more of the game belongs to \
             content even when the review asks for it as DLC: that is a review saying it ran \
             out, not one about how the game is sold.",
        ),
        alone: false,
    },
    Category {
        id: "multiplayer",
        label: "Multiplayer and online",
        description: "Playing with or against other people. Matchmaking, servers, lag and ping, \
                      co-op, player counts, whether lobbies are dead and whether cheaters ruin it.",
        boundary: Some(
            "Servers, matchmaking and connection quality belong here, and so does what playing \
             with other people is like: better with friends, dull on your own, worth it only \
             in a group. What the other players are like belongs to community, and calling \
             the game co-op as a kind of game belongs to genre. \"The game is dead\" is here \
             when nobody is playing it and updates when nobody is developing it.",
        ),
        alone: false,
    },
    Category {
        id: "community",
        label: "Community and players",
        description: "The people who play it. Whether the community is welcoming or toxic, \
                      griefing and harassment, and what the playerbase is like.",
        boundary: Some(
            "What the people are like belongs here. What they have made belongs to mods, and \
             whether there are enough of them online belongs to multiplayer.",
        ),
        alone: false,
    },
    Category {
        id: "updates",
        label: "Updates and developer support",
        description: "What the developers do after release. Patches, roadmaps, early access \
                      progress, communication with players, whether promises were kept, and \
                      whether the game is abandoned.",
        boundary: Some(
            "What the developers do to the game belongs here, and so does praise or blame \
             aimed at the studio itself, including thanking them and telling them to fix it. \
             A complaint about what a patch changed is about the change: balance to \
             difficulty, a mechanic to gameplay, content removed to content. This is for the \
             patching itself, its pace, and whether they listen. \"The game is dead\" is here \
             when nobody is developing it and multiplayer when nobody is playing it. What the \
             publisher or the platform requires of the player belongs to policy.",
        ),
        alone: false,
    },
    // Review bombs over publisher decisions are the reviews Steam's own default filter hides,
    // and this tool exists partly to count them. On one corpus measured they are 16% of
    // 1.16 million reviews, with nowhere to go: offtopic means a reader learns nothing about
    // the game, and "you need a second account to play" is not nothing.
    Category {
        id: "policy",
        label: "Publisher and platform policy",
        description: "Decisions taken by the publisher or the platform rather than by the \
                      game. Needing a second account to play, region locks and delistings, \
                      DRM and kernel-level anti-cheat, launcher requirements, price rises, \
                      and terms that changed after people had bought it.",
        boundary: Some(
            "Anything the publisher or the platform decided belongs here, including a protest \
             that is angry and brief and names no particular term, and including celebrating \
             one that was reversed: \"we won\" about a withdrawn account requirement is this, \
             and it is praise. What a decision is belongs here; what it does while playing \
             belongs where it happens, so a launcher that will not sign you in is here and an \
             anti-cheat that keeps you out of a match is multiplayer. Only a protest about \
             something with no connection to this game at all belongs to offtopic.",
        ),
        alone: false,
    },
    Category {
        id: "compatibility",
        label: "Hardware and compatibility",
        description: "Whether it runs on your setup at all. System requirements, Steam Deck, \
                      Linux and Proton, ultrawide and multi-monitor support, drivers and \
                      hardware, and asking for it on a console or handheld.",
        boundary: Some(
            "Asking for a port to another platform belongs here, and so does a game that \
             will not start at all when no reason is given. A game that starts and then \
             crashes is bugs. Asking for a sequel belongs to verdict, and anything about \
             playing it in a headset belongs to VR.",
        ),
        alone: false,
    },
    Category {
        id: "accessibility",
        label: "Accessibility and options",
        description: "Settings and accommodations. Subtitles, colourblind modes, remappable \
                      controls, difficulty options, text size, and the settings players need \
                      in order to play at all.",
        boundary: Some("Which languages the game is available in belongs to language, not here."),
        alone: false,
    },
    // Bundled into accessibility until it was measured: language complaints were most of
    // that category's mass while having nothing to do with accommodation, and they are the
    // single most common thing non-English reviews are about.
    Category {
        id: "language",
        label: "Language and localisation",
        description: "Which languages the game is in and how good the translation is. No \
                      English, no Chinese, machine translation, a language dropped in an \
                      update, subtitles only in some languages, playing anyway with a \
                      dictionary.",
        boundary: Some(
            "Whether a language exists and how well it reads belongs here. Subtitles as an \
             accommodation, in a language that is already supported, belong to accessibility.",
        ),
        alone: false,
    },
    // Asked for by three labellers independently across the core-3 sets, which is the
    // strongest signal any reference set has produced. It was being split between gameplay
    // and difficulty, and "the tutorial explains nothing" is neither a mechanic nor a
    // question of balance.
    Category {
        id: "tutorial",
        label: "Tutorial and learning",
        description: "How the game teaches itself. The tutorial, the first hour, systems that \
                      are never explained, the learning curve, reading a wiki to understand \
                      anything, and whether a new player is left to work it out alone.",
        boundary: Some(
            "Whether the game explains itself belongs here. Whether it is hard once you do \
             understand it belongs to difficulty, and whether the systems themselves are any \
             good belongs to gameplay.",
        ),
        alone: false,
    },
    // A licence is most visible when it lapses: a team, a driver or a song that was in last
    // year's release and is not in this one. That had been landing in content, which is about
    // how much game there is, and in monetisation, which is about what is sold on top.
    Category {
        id: "licensing",
        label: "Licensed content",
        description: "Whether the game has the real names. Real teams, players, clubs and kits, \
                      real cars and tracks, licensed music, a licence lost or gained between \
                      releases, and how faithful the game is to the thing it is adapting.",
        boundary: Some(
            "A real name being present, missing or wrong belongs here. What is sold on top of \
             the game belongs to monetisation, whether an adaptation is well written belongs \
             to story, and a protest about a licence agreement, terms of service or an account \
             belongs to policy.",
        ),
        alone: false,
    },
    // Kept out of compatibility, which is about whether a game runs at all. A review of a
    // headset game is about comfort, tracking and standing inside the thing, and none of that
    // is a system requirement.
    Category {
        id: "vr",
        label: "VR and headsets",
        description: "Playing it in a headset. Room scale and seated play, motion sickness and \
                      comfort options, tracking and the controllers in your hands, whether it \
                      was built for a headset or is a flat game bolted into one, and which \
                      headsets it works with.",
        boundary: Some(
            "Anything about being in a headset belongs here, including whether a given headset \
             is supported. A flat game called immersive is not this: immersion belongs to \
             whatever creates it, most often graphics or story. How the game looks on a \
             monitor belongs to graphics.",
        ),
        alone: false,
    },
];

/// Text handed to the embedding model for a category. Boundary rules are deliberately absent.
#[must_use]
pub fn embedding_text(category: &Category) -> String {
    format!("{}. {}", category.label, category.description)
}

#[must_use]
pub fn by_id(id: &str) -> Option<&'static Category> {
    CORE_SPINE.iter().find(|c| c.id == id)
}

/// The category sheet handed to whoever is labelling, generated rather than retyped.
///
/// The first reference set was produced from instructions written by hand for each batch of
/// labellers, and they disagreed about the same boundary in near-identical cases because the
/// instructions never settled it. Generating the sheet from the taxonomy means a rule can
/// only be fixed in one place, and every labeller sees the same one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Unit {
    /// A whole review, which is what the first reference sets labelled.
    Review,
    /// One point a review makes. What the model is trained on, because a review is not one
    /// opinion and a vector for the whole of it is the average of the ones it holds.
    #[default]
    Claim,
}

#[must_use]
pub fn labelling_brief(unit: Unit) -> String {
    use std::fmt::Write as _;

    let mut brief = match unit {
        Unit::Review => format!(
            "Categories (spine {CORE_SPINE_VERSION}). Each review gets exactly one primary \
             category, plus any others it genuinely also covers.\n\n\
             You are shown the text of a review and nothing else: not which game it is, not \
             whether the reviewer recommended it, not what the classifier guessed. The tool \
             being measured sorts reviews from their text alone, so a label made from more \
             than that would measure what you were told rather than how well it reads.\n\n"
        ),
        Unit::Claim => format!(
            "Categories (spine {CORE_SPINE_VERSION}). You are labelling CLAIMS: the separate \
             points a review makes. Each claim gets exactly one subject.\n\n\
             A review arrives split into numbered claims, and you label every one of them. \
             The review is there so a claim like \"it doesn't\" or \"same here\" can be read \
             in context; the label is about the claim, not about the review around it. A \
             review that makes one point has one claim, and a review that makes twelve has \
             twelve, which is the whole reason for labelling this way: the tool used to \
             average a long review into a single vector and then file a two-word review under \
             four subjects.\n\n\
             You are not told which game it is, whether the reviewer recommended it, or what \
             the model guessed. The model reads the text alone, so a label made from more \
             than that measures what you were told rather than how well it reads.\n\n\
             Most claims name no aspect at all. \"Great game\", \"10/10\", \"gfg\", a row of \
             emoji: these are not graphics, not gameplay and not story, and filing them as \
             any of those is the exact failure this set exists to fix. They are `verdict` \
             when they judge the game and `offtopic` when they say nothing about it. Expect \
             to use those two more than anything else, and do not go looking for a subject \
             that is not there.\n\n"
        ),
    };
    for category in CORE_SPINE {
        let _ = writeln!(
            brief,
            "{} ({})\n  {}",
            category.label, category.id, category.description
        );
        if let Some(rule) = category.boundary {
            let _ = writeln!(brief, "  RULE: {rule}");
        }
        if category.alone && unit == Unit::Review {
            let _ = writeln!(
                brief,
                "  ONLY: this is the whole label. A review here covers nothing else, so it \
                 never takes a second category."
            );
        }
        brief.push('\n');
    }
    let fields = match unit {
        Unit::Review => FIELDS,
        Unit::Claim => CLAIM_FIELDS,
    };
    brief.push_str(
        &fields
            .replace(CONFIDENCE_SLOT, &CONFIDENCE.join(", "))
            .replace(POLARITY_SLOT, &POLARITY.join(", ")),
    );
    brief
}

/// The sheet a model works from when asked what a game's players talk about that the spine
/// has no row for.
///
/// Generated from the spine for the same reason the labelling sheet is: the reader has to know
/// every fixed subject to know what is not one, and a hand-written list of them would drift
/// the first time a subject was added.
#[must_use]
pub fn induction_brief() -> String {
    use std::fmt::Write as _;

    let mut brief = format!(
        "You are reading reviews of one game, chosen to be as unlike each other as the corpus \
         allows, and naming what its players talk about that no game shares.\n\n\
         Every game's reviews are sorted into the same {} subjects (spine {CORE_SPINE_VERSION}), \
         listed below. Those are the floor. What you are looking for is above it: a subject \
         this game's players return to that is not one of them, or is one of them in a form \
         so specific to this game that it deserves its own row. Mud physics in a truck \
         simulator. Jump scares in a horror game. Deck archetypes in a card game. The AI \
         opponent in a strategy game. Mods, where a game is played modded.\n\n\
         You are not looking for what people think of the game. Praise and complaint are \
         recorded elsewhere. You are looking for what they think about, and only what several \
         of them think about: a subject one review raises is an anecdote, and a subject you \
         cannot point at three reviews for does not exist.\n\n\
         The fixed subjects, which you must not name again:\n\n",
        CORE_SPINE.len()
    );
    for category in CORE_SPINE {
        let _ = writeln!(
            brief,
            "  {} ({}): {}",
            category.label, category.id, category.description
        );
    }
    brief.push_str(
        "\nReturn one JSON object and nothing else:\n\n\
         {\n  \"induced_by\": \"<the model you \
         are>\",\n  \"subjects\": [\n    {\n      \"id\": \"<short, lowercase, hyphens>\",\n      \
         \"label\": \"<what the row is called>\",\n      \"description\": \"<one sentence: \
         what belongs here and what does not>\",\n      \"refines\": \"<a fixed subject id, \
         or null>\",\n      \"evidence\": [\"<review_id>\", \"<review_id>\", \"<review_id>\"]\n    \
         }\n  ]\n}\n\n\
         `refines` names the fixed subject this one is a specific form of, when it is one, \
         and is null when it is not. `evidence` is review ids from the handout, as given, at \
         least three and every one of them a review that raises the subject. A subject whose \
         evidence names a review the handout did not hold is refused whole, and so is one \
         with fewer than three. Better to return four subjects that survive than twelve that \
         do not.\n\n\
         Between five and twelve subjects is the usual shape. Zero is a legitimate answer for \
         a game whose players talk about nothing the spine does not already name, and is \
         better than an invented one.\n",
    );
    brief
}

/// The answers `confidence` may take.
///
/// A judgement asked for as free text drifts: "Medium", "fairly", "8/10", nothing at all. The
/// sheet asks for these three and ingest refuses anything else, from the same list, so the
/// column stays something a reader can count.
pub const CONFIDENCE: [&str; 3] = ["high", "medium", "low"];

/// Where the sheet names them, so the words are written down once.
const CONFIDENCE_SLOT: &str = "{confidence}";

/// The answers `polarity` may take.
///
/// Three rather than two, because a claim can state a fact about the game without judging it,
/// and forcing "it is a roguelike deckbuilder" to be praise or complaint would put a verdict
/// in a reviewer's mouth. Mixed is deliberately absent: a claim that both praises and
/// complains is two claims the splitter failed to separate, and `split_wrong` records that
/// instead of hiding it in a fourth value.
pub const POLARITY: [&str; 3] = ["praise", "complaint", "neutral"];

const POLARITY_SLOT: &str = "{polarity}";

/// What every claim label carries.
///
/// The review-level sheet asks for a primary and a secondary category. A claim takes exactly
/// one subject, and that is the point of the unit: where two genuinely fit, the split was
/// wrong, and saying so is worth more than a second category. `split_wrong` is how the
/// splitter gets measured by the people best placed to see it fail.
const CLAIM_FIELDS: &str = "\
Every claim label is six fields.

subject
  The one category this claim is about. Exactly one, always. Most claims name no aspect at
  all: those are `verdict` if they judge the game and `offtopic` if they do not.

polarity
  What the claim does about its subject, in one of these words: {polarity}. Praise and
  complaint are about the game, not about the reviewer's mood. Neutral is for a claim that
  states something without judging it, which is common and is not a failure to decide.

ironic
  The text says the opposite of what it appears to say. \"0/10, I have not slept in three
  days\" is praise; \"10/10 would lose my save file again\" is a complaint. Judge this from
  the words alone. You are not told whether the reviewer recommended the game, so that you
  cannot be led by it. Where a claim is ironic, `polarity` is what the reviewer MEANS.

confidence
  How sure you are of the subject, in one of these words: {confidence}. This one is about
  you, and it is used: the model is trained against your uncertainty rather than against a
  flattened guess, so a truthful \"low\" is worth more than a confident wrong answer.

ambiguous
  Whether the call is genuinely contested: two subjects fit and the rules above do not settle
  which. This is about the claim and the taxonomy rather than about you, and it is read back.
  Agreement is reported separately over the claims marked here.

split_wrong
  Whether this claim is really two points stuck together, or half of one that was cut in the
  wrong place. The splitting is mechanical and it will be wrong sometimes; this is the only
  signal that it was, and it is what improves it. Leave it false unless the text in front of
  you is genuinely mis-cut.

Return every one of these for every claim, in the order the claims are given. A judgement left
out is not a judgement, and a label missing one is refused rather than filled in with a guess.
";

/// What every label carries besides its categories.
///
/// Generated with the categories and for the same reason. The boundaries were written down
/// because labellers disagreed about them; these were left to whatever each batch was told,
/// and one of them decides how a headline figure is reported. `ambiguous` is what splits
/// agreement into clear-cut and contested, and across the core-3 sets it marked between 21%
/// and 35% of a game depending on who labelled it, which is a spread no property of the
/// reviews explains.
///
/// `ironic` is a claim about the words, not about the rating. The labeller is never shown
/// whether the reviewer recommended the game, so that they cannot be led by it, which means
/// they cannot report that the two disagree either. They say what the text does; joining
/// that to the rating is arithmetic and belongs to the tool.
const FIELDS: &str = "\
Every label is five fields. The first two say which categories the review covers; the other
three are judgements about the review, and none of them changes which category it belongs to.

primary
  The one category the review is most about. Exactly one, always.

secondary
  Every other category the review genuinely also covers. Leave it empty when the review is
  about one thing. Never add a category to fill space.

ironic
  The text says the opposite of what it appears to say. \"0/10, I have not slept in three
  days\" is praise; \"10/10 would lose my save file again\" is a complaint. Judge this from
  the words alone. You are not told whether the reviewer recommended the game, so that you
  cannot be led by it.

confidence
  How sure you are of the primary category, in one of these words: {confidence}.
  This one is about you.

ambiguous
  Whether the call is genuinely contested: two categories fit and the rules above do not
  settle which. This one is about the review and the taxonomy rather than about you, and it
  is the judgement that is read back. Agreement is reported separately for the reviews
  marked here, because disagreement on them says as much about the taxonomy as about the
  classifier.

Return every one of these for every review. A judgement left out is not a judgement, and a
label missing one is refused rather than filled in with a guess: ironic and ambiguous are
claims about the review, and defaulting them to false would put words in your mouth on the
figure that decides how agreement is reported.
";

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn category_ids_are_unique_and_stable_looking() {
        let ids: HashSet<&str> = CORE_SPINE.iter().map(|c| c.id).collect();
        assert_eq!(ids.len(), CORE_SPINE.len(), "duplicate category id");
        for category in CORE_SPINE {
            assert!(
                category
                    .id
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c == '_'),
                "{} is not a stable-looking id",
                category.id
            );
        }
    }

    #[test]
    fn descriptions_are_written_as_sentences_not_labels() {
        for category in CORE_SPINE {
            assert!(
                category.description.len() > 60,
                "{} has too thin a description to embed usefully",
                category.id
            );
            assert!(
                category.description.ends_with('.'),
                "{} should read as prose",
                category.id
            );
        }
    }

    #[test]
    fn boundary_rules_never_reach_the_embedding_model() {
        // A rule is written to a labeller and reads like an instruction. Embedding one would
        // pull the anchor towards the language of instructions and away from reviews.
        for category in CORE_SPINE {
            let embedded = embedding_text(category);
            if let Some(rule) = category.boundary {
                assert!(
                    !embedded.contains(rule),
                    "{} embeds its boundary rule",
                    category.id
                );
            }
        }
    }

    #[test]
    fn every_boundary_rule_names_a_category_that_exists() {
        let ids: HashSet<&str> = CORE_SPINE.iter().map(|c| c.id).collect();
        for category in CORE_SPINE {
            let Some(rule) = category.boundary else {
                continue;
            };
            let named = ids
                .iter()
                .filter(|id| **id != category.id && rule.contains(*id))
                .count();
            assert!(
                named > 0,
                "{}'s rule resolves a conflict with nothing: {rule}",
                category.id
            );
        }
    }

    #[test]
    fn the_labelling_brief_carries_every_category_and_every_rule() {
        for unit in [Unit::Review, Unit::Claim] {
            let brief = labelling_brief(unit);
            assert!(brief.contains(CORE_SPINE_VERSION));
            for category in CORE_SPINE {
                assert!(brief.contains(category.id), "{} missing", category.id);
                assert!(brief.contains(category.description));
                if let Some(rule) = category.boundary {
                    assert!(brief.contains(rule), "{}'s rule missing", category.id);
                }
            }
        }
    }

    /// The claim sheet asks for one subject and a polarity, and the failure it exists to stop
    /// is a labeller hunting for a topic in "Great game".
    #[test]
    fn the_claim_brief_asks_for_one_subject_and_a_polarity() {
        let brief = labelling_brief(Unit::Claim);
        for field in [
            "subject",
            "polarity",
            "ironic",
            "confidence",
            "ambiguous",
            "split_wrong",
        ] {
            assert!(
                brief.contains(&format!("\n{field}\n")),
                "{field} goes unexplained"
            );
        }
        for word in POLARITY {
            assert!(
                brief.contains(word),
                "polarity {word} is asked for but not named"
            );
        }
        assert!(!brief.contains(POLARITY_SLOT));
        assert!(
            brief.contains("Return every one of these for every claim"),
            "the sheet does not say that every judgement is required"
        );
        assert!(
            !brief.contains("secondary"),
            "a claim takes one subject; offering a secondary invites the averaging this unit exists to remove"
        );
    }

    /// The judgements besides the categories were left to whatever each batch of labellers
    /// was told, and one of them decides how a headline figure is reported.
    #[test]
    fn the_brief_says_what_the_other_judgements_mean() {
        let brief = labelling_brief(Unit::Review);
        for field in ["primary", "secondary", "ironic", "confidence", "ambiguous"] {
            assert!(
                brief.contains(&format!("\n{field}\n")),
                "{field} goes unexplained"
            );
        }
        // The two are asked in the same breath and mean opposite things: one is a fact about
        // the labeller, the other a fact about the taxonomy. Only the second is read back.
        assert!(
            brief.contains("This one is about you.")
                && brief.contains("about the review and the taxonomy rather than about you"),
            "confidence and ambiguous are not told apart, which is the whole difficulty"
        );
        // Ingest drops a label that leaves one out, so the sheet is the only place a labeller
        // can find out that it will, and the only place the accepted words are written down.
        assert!(
            brief.contains("Return every one of these for every review."),
            "the sheet does not say that every judgement is required"
        );
        for word in CONFIDENCE {
            assert!(
                brief.contains(word),
                "the sheet asks for a confidence it does not name: {word}"
            );
        }
        assert!(
            !brief.contains(CONFIDENCE_SLOT),
            "the sheet still holds the placeholder instead of the words"
        );
    }

    #[test]
    fn the_checked_in_brief_is_the_one_this_build_would_generate() {
        // The sheet labellers work from is committed so it can be read without building the
        // tool, which makes it capable of drifting from the taxonomy it claims to describe.
        // A reference set labelled against a stale sheet is silently mislabelled, and the
        // first set produced by this project lost consistency exactly that way.
        for (file, unit) in [
            ("labelling-brief.txt", Unit::Review),
            ("claim-brief.txt", Unit::Claim),
        ] {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../reference")
                .join(file);
            let committed = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("{} is missing: {e}", path.display()));
            assert_eq!(
                committed.replace("\r\n", "\n"),
                labelling_brief(unit),
                "reference/{file} is stale; regenerate it with `census brief`"
            );
        }
    }

    #[test]
    fn lookup_finds_categories_and_rejects_unknown_ones() {
        assert_eq!(by_id("bugs").map(|c| c.label), Some("Bugs and crashes"));
        assert_eq!(
            by_id("genre").map(|c| c.label),
            Some("Genre and comparisons")
        );
        assert!(by_id("not-a-category").is_none());
    }
}
