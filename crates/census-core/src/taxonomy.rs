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

/// One category in the taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Category {
    /// Stable identifier. Appears in output files, so changing one invalidates comparisons.
    pub id: &'static str,
    pub label: &'static str,
    pub description: &'static str,
}

/// Version of the core spine. Any change to the categories or their descriptions changes
/// what the numbers mean, so this is recorded alongside every classification.
pub const CORE_SPINE_VERSION: &str = "core-1";

pub const CORE_SPINE: &[Category] = &[
    Category {
        id: "performance",
        label: "Performance",
        description: "The game runs badly. Low frame rate, stuttering, frame drops, poor \
                      optimisation, long loading times, and it struggles even on good hardware.",
    },
    Category {
        id: "bugs",
        label: "Bugs and crashes",
        description: "The game is broken and buggy. It crashes to desktop, freezes, corrupts \
                      or loses save files, and is full of glitches that block progress.",
    },
    Category {
        id: "gameplay",
        label: "Gameplay and mechanics",
        description: "How the game actually plays. The core loop, the mechanics, the combat \
                      and systems, whether it is fun to play moment to moment or repetitive.",
    },
    Category {
        id: "story",
        label: "Story and writing",
        description: "The story, plot and characters. The writing, the dialogue, the ending, \
                      the world and its lore, and whether the narrative is worth following.",
    },
    Category {
        id: "graphics",
        label: "Graphics and art",
        description: "How the game looks. The visuals, the art style, the animation, the \
                      environments and character models, whether it is beautiful or ugly.",
    },
    Category {
        id: "audio",
        label: "Audio and music",
        description: "How the game sounds. The soundtrack, the music, the sound effects, the \
                      voice acting and the audio mixing.",
    },
    Category {
        id: "controls",
        label: "Controls and interface",
        description: "The controls and the user interface. Clunky or responsive controls, \
                      keybindings, controller support, menus, the HUD and the inventory screens.",
    },
    Category {
        id: "difficulty",
        label: "Difficulty and balance",
        description: "How hard the game is and whether it is fair. Difficulty spikes, grinding, \
                      balance between options, and whether it is too easy or punishing.",
    },
    Category {
        id: "content",
        label: "Amount of content",
        description: "How much game there is. Length, replay value, how many hours it lasts, \
                      whether it feels finished or thin and runs out of things to do.",
    },
    Category {
        id: "price",
        label: "Price and value",
        description: "What it costs and whether it is worth the money. Full price versus sale, \
                      value for money, refunds, and whether to wait for a discount.",
    },
    Category {
        id: "monetisation",
        label: "Monetisation and DLC",
        description: "Paid extras and how they are sold. Microtransactions, battle passes, \
                      loot boxes, paywalls, season passes and content cut out to sell as DLC.",
    },
    Category {
        id: "multiplayer",
        label: "Multiplayer and online",
        description: "Playing with or against other people. Matchmaking, servers, lag and ping, \
                      co-op, player counts, whether lobbies are dead and whether cheaters ruin it.",
    },
    Category {
        id: "community",
        label: "Community and players",
        description: "The people who play it. Whether the community is welcoming or toxic, \
                      griefing and harassment, and what the playerbase is like.",
    },
    Category {
        id: "updates",
        label: "Updates and developer support",
        description: "What the developers do after release. Patches, roadmaps, early access \
                      progress, communication with players, and whether the game is abandoned.",
    },
    Category {
        id: "compatibility",
        label: "Hardware and compatibility",
        description: "Whether it runs on your setup at all. System requirements, Steam Deck, \
                      Linux and Proton, ultrawide and multi-monitor support, drivers and hardware.",
    },
    Category {
        id: "accessibility",
        label: "Accessibility and options",
        description: "Settings and accommodations. Subtitles, colourblind modes, remappable \
                      controls, difficulty options, language and localisation support.",
    },
];

/// Text handed to the embedding model for a category.
#[must_use]
pub fn embedding_text(category: &Category) -> String {
    format!("{}. {}", category.label, category.description)
}

#[must_use]
pub fn by_id(id: &str) -> Option<&'static Category> {
    CORE_SPINE.iter().find(|c| c.id == id)
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
    fn lookup_finds_categories_and_rejects_unknown_ones() {
        assert_eq!(by_id("bugs").map(|c| c.label), Some("Bugs and crashes"));
        assert!(by_id("not-a-category").is_none());
    }
}
