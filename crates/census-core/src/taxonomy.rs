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
}

/// Version of the core spine. Any change to the categories or their descriptions changes
/// what the numbers mean, so this is recorded alongside every classification.
pub const CORE_SPINE_VERSION: &str = "core-3";

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
    },
    Category {
        id: "bugs",
        label: "Bugs and crashes",
        description: "The game is broken and buggy. It crashes to desktop, freezes, corrupts \
                      or loses save files, and is full of glitches that block progress.",
        boundary: None,
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
             options are balanced against each other belongs to difficulty. How much game \
             there is belongs to content.",
        ),
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
            "Naming what kind of game it is, or which games it resembles, belongs here. As \
             soon as a review says a mechanic is deep, shallow, satisfying or broken, that \
             part is gameplay.",
        ),
    },
    // Reviews that give a verdict and name no aspect are common, and without a home they
    // contaminate whichever category happens to sit nearest in the embedding space.
    Category {
        id: "verdict",
        label: "Overall verdict only",
        description: "A verdict with no specific reason given. Great game, terrible game, \
                      ten out of ten, would recommend, worth every penny, do not buy, \
                      waste of money, best game ever, I find this game childish, please \
                      make a sequel.",
        boundary: Some(
            "Only when no aspect is named at all. A review that gives a verdict and then \
             names one specific thing belongs to that thing: \"super fun, worth every penny\" \
             is price. Asking for a sequel belongs here; asking for a port belongs to \
             compatibility.",
        ),
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
             about, most often gameplay or difficulty. This is only for reviews from which a \
             reader would learn nothing about the game at all.",
        ),
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
    },
    Category {
        id: "audio",
        label: "Audio and music",
        description: "How the game sounds. The soundtrack, the music, the sound effects, the \
                      voice acting and the audio mixing.",
        boundary: None,
    },
    Category {
        id: "controls",
        label: "Controls and interface",
        description: "The controls and the user interface. Clunky or responsive controls, \
                      keybindings, controller support, menus, the HUD, the inventory screens, \
                      and how many clicks it takes to do anything.",
        boundary: Some(
            "Whether a controller is supported belongs here. Whether the game runs on a given \
             device belongs to compatibility.",
        ),
    },
    Category {
        id: "difficulty",
        label: "Difficulty and balance",
        description: "How hard the game is and whether it is fair. Difficulty spikes, \
                      grinding, whether the options are balanced against each other, luck and \
                      randomness deciding the outcome, and whether it is too easy or punishing.",
        boundary: Some(
            "Balance complaints belong here rather than to gameplay, including when they name \
             a specific mechanic as overpowered or useless.",
        ),
    },
    Category {
        id: "content",
        label: "Amount of content",
        description: "How much game there is. Length, how many hours it lasts, replay value, \
                      whether runs differ from one another, whether it gets repetitive, and \
                      whether it feels finished or thin and runs out of things to do.",
        boundary: Some(
            "Replayability and repetitiveness are two ends of one axis and both belong here, \
             never to gameplay.",
        ),
    },
    Category {
        id: "price",
        label: "Price and value",
        description: "What it costs and whether it is worth the money. Full price versus sale, \
                      value for money, refunds, and whether to wait for a discount.",
        boundary: Some(
            "What the base game costs belongs here. What is sold on top of it belongs to \
             monetisation.",
        ),
    },
    Category {
        id: "monetisation",
        label: "Monetisation and DLC",
        description: "Paid extras and how they are sold. Microtransactions, battle passes, \
                      loot boxes, paywalls, season passes and content cut out to sell as DLC.",
        boundary: None,
    },
    Category {
        id: "multiplayer",
        label: "Multiplayer and online",
        description: "Playing with or against other people. Matchmaking, servers, lag and ping, \
                      co-op, player counts, whether lobbies are dead and whether cheaters ruin it.",
        boundary: Some(
            "Servers, matchmaking and connection quality belong here. What the other players \
             are like belongs to community.",
        ),
    },
    Category {
        id: "community",
        label: "Community and players",
        description: "The people who play it. Whether the community is welcoming or toxic, \
                      griefing and harassment, and what the playerbase is like.",
        boundary: None,
    },
    Category {
        id: "updates",
        label: "Updates and developer support",
        description: "What the developers do after release. Patches, roadmaps, early access \
                      progress, communication with players, whether promises were kept, and \
                      whether the game is abandoned.",
        boundary: None,
    },
    Category {
        id: "compatibility",
        label: "Hardware and compatibility",
        description: "Whether it runs on your setup at all. System requirements, Steam Deck, \
                      Linux and Proton, ultrawide and multi-monitor support, drivers and \
                      hardware, and asking for it on a console or handheld.",
        boundary: Some(
            "Asking for a port to another platform belongs here. Asking for a sequel belongs \
             to verdict.",
        ),
    },
    Category {
        id: "accessibility",
        label: "Accessibility and options",
        description: "Settings and accommodations. Subtitles, colourblind modes, remappable \
                      controls, difficulty options, text size, and the settings players need \
                      in order to play at all.",
        boundary: Some("Which languages the game is available in belongs to language, not here."),
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
#[must_use]
pub fn labelling_brief() -> String {
    use std::fmt::Write as _;

    let mut brief = format!(
        "Categories (spine {CORE_SPINE_VERSION}). Each review gets exactly one primary \
         category, plus any others it genuinely also covers.\n\n"
    );
    for category in CORE_SPINE {
        let _ = writeln!(
            brief,
            "{} ({})\n  {}",
            category.label, category.id, category.description
        );
        if let Some(rule) = category.boundary {
            let _ = writeln!(brief, "  RULE: {rule}");
        }
        brief.push('\n');
    }
    brief
}

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
        let brief = labelling_brief();
        assert!(brief.contains(CORE_SPINE_VERSION));
        for category in CORE_SPINE {
            assert!(brief.contains(category.id), "{} missing", category.id);
            assert!(brief.contains(category.description));
            if let Some(rule) = category.boundary {
                assert!(brief.contains(rule), "{}'s rule missing", category.id);
            }
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
