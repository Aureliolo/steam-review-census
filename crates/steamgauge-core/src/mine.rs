//! Fishing lines for the subjects the labelled set has almost none of.
//!
//! Eight of the twenty-six subjects have under 250 labels between them and `licensing` has
//! thirty-two. A random draw will not fix that: it lands on the same distribution the corpus
//! has, so another thousand random claims buys another two `vr` labels. Reweighting the loss
//! does not fix it either, because reweighting redistributes a gradient that thirty-two claims
//! do not contain; every weighting scheme tried here made macro F1 worse.
//!
//! What does fix it is going and finding them. Twenty million claims have been read and never
//! labelled, and the starved subjects are in there at their natural rate, which is small but
//! not zero: a corpus of seven million reviews holds tens of thousands of claims about
//! headsets.
//!
//! **A probe is a fishing line, not a definition.** It says where claims about a subject tend
//! to be found, which is a different thing from what the subject means: the taxonomy says what
//! `vr` is and a labeller decides whether a claim is one. Matching a probe only puts a claim
//! in front of the labeller. It follows that a probe may be loose, and several here are: the
//! cost of a false positive is one labelled claim that turns out to be `gameplay`, which is
//! still a labelled claim.
//!
//! **This is lexical, and lexical is the weaker half of the method.** The published form of
//! this selects by retrieval, embedding the pool and taking neighbours of the claims already
//! labelled, which finds the paraphrases no probe lists. Retrieval also travels across
//! languages, and these probes mostly do not: they lean on borrowed tokens that survive
//! translation ("VR", "Denuvo", "Proton", "mod") and on the one or two scripts where a term is
//! written the same way everywhere. A Russian review complaining about subtitles is not caught
//! here. That gap is the reason to build the retrieval pass, and it is not a reason to skip
//! this one: the borrowed tokens alone raise the hit rate on `vr` and `policy` by more than an
//! order of magnitude over a random draw, and they cost a scan rather than a GPU.
//!
//! **Point the draw at games that could plausibly hold the subject.** Two of these lines are
//! about a property of the game rather than a thing reviewers say: `licensing` fires on
//! adaptation vocabulary, and in a game that adapts nothing it catches film comparisons, which
//! are `genre`; `vr` fires on headset vocabulary, and a flat game's reviews mention headsets
//! only to say it has none. A run over a sports game, a tie-in and a headset game is worth
//! several over whatever happens to be captured.

/// Where claims about one starved subject tend to be found.
#[derive(Debug)]
pub struct Probe {
    /// Subject id in [`crate::taxonomy::CORE_SPINE`].
    pub subject: &'static str,
    /// Matched case-insensitively against the claim. An ASCII term must fall on a word
    /// boundary, so "mod" does not match "modern"; a term with non-ASCII characters matches
    /// as a bare substring, because the scripts that need it do not write word boundaries.
    pub terms: &'static [&'static str],
}

/// The eight subjects with the fewest labels, and what to look for.
///
/// Ordered by how starved the subject is, because the draw fills its quotas in this order and
/// a claim that matches two subjects is counted once, for the first of them.
pub const PROBES: &[Probe] = &[
    Probe {
        // Real names, not licence agreements: a protest about an EULA is `policy`, and this
        // is the row that gets the sports and adaptation vocabulary.
        subject: "licensing",
        terms: &[
            // Not bare "licence": in a games corpus it is usually an in-game mechanic (a
            // mining licence, a pilot licence) or an end-user agreement, which is `policy`.
            "licensed",
            "unlicensed",
            "licensing",
            "official licence",
            "official license",
            "music licence",
            "music license",
            "lost the licence",
            "lost the license",
            "real names",
            "real teams",
            "real cars",
            "real players",
            "fake names",
            "official teams",
            "fictional",
            "faithful to the",
            "adaptation",
            "the books",
            "the comics",
            "the anime",
            "the manga",
            "the movie",
            "the film",
            "the show",
            "canon",
            "lore accurate",
            "lore-accurate",
        ],
    },
    Probe {
        subject: "vr",
        terms: &[
            "vr",
            "pcvr",
            "steamvr",
            "openxr",
            "headset",
            "hmd",
            "oculus",
            "meta quest",
            "quest 2",
            "quest 3",
            // Not bare "vive": it is "long live" in French and Spanish, and a co-op game's
            // reviews are full of it.
            "htc vive",
            "the vive",
            "valve index",
            "psvr",
            "pimax",
            "pico 4",
            "wmr",
            "room scale",
            "roomscale",
            "room-scale",
            "6dof",
            "vr头显",
            "头显",
            // Not "motion sickness": a flat game induces it through camera shake and field of
            // view, which is `graphics` or `accessibility`, and DRG alone offered dozens.
        ],
    },
    Probe {
        subject: "accessibility",
        terms: &[
            // Not bare "accessible": in a review it nearly always means easy to get into,
            // which is `difficulty` or `verdict`, and it drowned this line when it was here.
            "accessibility option",
            "accessibility setting",
            "accessibility feature",
            "accessibility menu",
            "accessibility support",
            "colorblind",
            "colourblind",
            "color blind",
            "colour blind",
            "subtitle",
            "subtitles",
            "closed caption",
            "text size",
            "font size",
            "remap",
            "remappable",
            "rebind",
            "keybind",
            "key binding",
            "key bindings",
            "screen reader",
            "photosensitiv",
            "epilepsy",
            "epileptic",
            "one handed",
            "one-handed",
            "deaf players",
            "for the deaf",
            "hard of hearing",
            // Not bare "deaf": "tone deaf" is a complaint about a publisher, not a subtitle.
        ],
    },
    Probe {
        subject: "language",
        terms: &[
            "localization",
            "localisation",
            "localized",
            "localised",
            "translation",
            "translated",
            "mistranslat",
            "machine translat",
            "google translate",
            "no english",
            "english please",
            "english support",
            "add english",
            "中文",
            "简体",
            "繁體",
            "русский",
            "перевод",
            "日本語",
            "한국어",
            "português",
            "español",
            "deutsch",
            "français",
            "türkçe",
        ],
    },
    Probe {
        subject: "community",
        terms: &[
            "community",
            "playerbase",
            "player base",
            "the players are",
            "other players are",
            "toxic",
            "toxicity",
            "elitist",
            "gatekeep",
            "welcoming",
            "friendly community",
            "helpful community",
            "griefer",
            "griefing",
            "discord",
            "the forums",
            "steam forums",
            "subreddit",
        ],
    },
    Probe {
        subject: "mods",
        terms: &[
            // Not bare "mod", "модов" or "模组": a great many games call their own weapon
            // upgrades mods, and in Deep Rock Galactic that sense was most of what this line
            // caught. The forms below only name the user-made kind.
            "modded",
            "modding",
            "modder",
            "moddable",
            "workshop",
            "steam workshop",
            "nexusmods",
            "nexus mods",
            "script extender",
            "total conversion",
            "custom maps",
            "custom levels",
            "user created",
            "user-created",
            "мастерская",
            "创意工坊",
        ],
    },
    Probe {
        subject: "compatibility",
        terms: &[
            "steam deck",
            "steamdeck",
            "deck verified",
            "proton",
            "linux",
            "steamos",
            "macos",
            "mac os",
            "macbook",
            "ultrawide",
            "ultra wide",
            "21:9",
            "32:9",
            "multi monitor",
            "multi-monitor",
            "system requirements",
            "minimum requirements",
            "won't launch",
            "wont launch",
            "will not launch",
            "won't start",
            "wont start",
            "console port",
            "switch port",
            "handheld",
            "rog ally",
        ],
    },
    Probe {
        subject: "policy",
        terms: &[
            "denuvo",
            "drm",
            "anti-cheat",
            "anticheat",
            "anti cheat",
            "battleye",
            "easy anti-cheat",
            "kernel level",
            "kernel-level",
            "ring 0",
            // Not bare "launcher": a grenade launcher is not a games launcher, and in a
            // shooter it is most of what the word means.
            "game launcher",
            "another launcher",
            "separate launcher",
            "third party launcher",
            "third-party launcher",
            "launcher required",
            "requires a launcher",
            "epic games launcher",
            "epic games account",
            "ubisoft connect",
            "ea app",
            "rockstar social club",
            "second account",
            "separate account",
            "third party account",
            "region lock",
            "region-lock",
            "region locked",
            "delisted",
            "eula",
            "terms of service",
            "always online",
            "always-online",
            "requires an account",
        ],
    },
];

/// Which subject's line a claim took, if any.
///
/// First match wins, in [`PROBES`] order, so the most starved subject gets the claim when two
/// lines cross. A claim saying "the VR mod" is offered as `vr` and not as `mods`, which is the
/// right way round: `vr` has forty-one labels and `mods` has a hundred and forty-six.
#[must_use]
pub fn hooked(claim: &str) -> Option<&'static str> {
    let haystack = claim.to_lowercase();
    PROBES
        .iter()
        .find(|probe| {
            probe
                .terms
                .iter()
                .any(|term| contains_term(&haystack, term))
        })
        .map(|probe| probe.subject)
}

/// Whether a lowercased claim holds this term, on a word boundary when the term is ASCII.
///
/// Without the boundary "mod" matches "modern" and "model", and "vr" matches nothing useful at
/// all in a language that happens to spell a common word with those two letters together.
fn contains_term(haystack: &str, term: &str) -> bool {
    if !term.is_ascii() {
        return haystack.contains(term);
    }
    let bytes = haystack.as_bytes();
    let mut from = 0;
    while let Some(at) = haystack[from..].find(term) {
        let start = from + at;
        let end = start + term.len();
        let before_is_word = start > 0 && is_word_byte(bytes[start - 1]);
        let after_is_word = end < bytes.len() && is_word_byte(bytes[end]);
        if !before_is_word && !after_is_word {
            return true;
        }
        // Advance by one byte rather than by the term, so overlapping positions are seen; the
        // haystack is lowercase ASCII-searchable but may hold multi-byte characters, and
        // `find` returns a char boundary, so stepping one byte and letting `find` resynchronise
        // is safe.
        from = start + 1;
        if from >= haystack.len() {
            break;
        }
    }
    false
}

fn is_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || byte >= 0x80
}

#[cfg(test)]
mod tests {
    use super::{PROBES, hooked};

    /// A term listed twice is a term that was edited twice and reconciled neither time, and
    /// nothing else in the file would notice.
    #[test]
    fn no_term_is_listed_twice() {
        let mut seen = std::collections::HashSet::new();
        for probe in PROBES {
            for term in probe.terms {
                assert!(seen.insert(*term), "{term:?} is listed twice");
                assert_eq!(
                    *term,
                    term.to_lowercase(),
                    "{term:?} is matched against lowercase and will never fire"
                );
            }
        }
    }

    #[test]
    fn every_probe_names_a_subject_the_spine_has() {
        for probe in PROBES {
            assert!(
                crate::taxonomy::CORE_SPINE
                    .iter()
                    .any(|category| category.id == probe.subject),
                "{} is not a subject",
                probe.subject
            );
        }
    }

    #[test]
    fn a_term_inside_a_longer_word_is_not_a_match() {
        assert_eq!(hooked("a thoroughly modern shooter"), None);
        assert_eq!(hooked("the workshopping of ideas"), None);
        assert_eq!(hooked("the steam workshop is full of them"), Some("mods"));
    }

    #[test]
    fn the_starved_subject_wins_a_claim_that_matches_two() {
        assert_eq!(hooked("the VR modding scene is incredible"), Some("vr"));
    }

    #[test]
    fn a_term_that_is_not_ascii_matches_without_a_boundary() {
        assert_eq!(hooked("求求你们加个中文吧"), Some("language"));
        assert_eq!(hooked("创意工坊的模组很多"), Some("mods"));
    }

    /// Each of these was caught by a probe that has since been narrowed, and each was the
    /// commonest thing its line returned rather than a curiosity: a shooter says "launcher"
    /// about a weapon, a game with an upgrade tree says "mods" about it, "accessible" is how
    /// reviewers say a game is easy to get into, and a publisher is called tone deaf.
    #[test]
    fn the_wrong_sense_of_a_word_does_not_take_a_line() {
        assert_eq!(hooked("the grenade launcher is devastating"), None);
        assert_eq!(hooked("interesting weapon mods to unlock"), None);
        assert_eq!(hooked("I found it entertaining and accessible"), None);
        assert_eq!(hooked("a tone deaf announcement from the publisher"), None);
        assert_eq!(hooked("upgrade your bear license, drink beer"), None);
        assert_eq!(hooked("vive la DRG, longue vie a eux"), None);
    }

    #[test]
    fn punctuation_and_case_do_not_hide_a_term() {
        assert_eq!(hooked("(VR) is unplayable"), Some("vr"));
        assert_eq!(hooked("DENUVO. again."), Some("policy"));
        assert_eq!(hooked("no Steam Deck support"), Some("compatibility"));
    }

    #[test]
    fn a_claim_about_nothing_on_the_list_is_not_hooked() {
        assert_eq!(hooked("the combat feels weightless"), None);
    }
}
