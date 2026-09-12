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

use std::path::Path;

use crate::{Error, Result};

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

/// The labelled claims of the starved subjects, embedded, to fish with.
///
/// This is the other half of the method, and the stronger one. A probe lists words; a labelled
/// `policy` claim about Denuvo sits near every other complaint about copy protection whatever
/// words it uses and whatever language it is in, and a Russian review complaining about
/// subtitles is near an English one. The published form of this selects by retrieval for
/// exactly that reason ([arXiv:2307.14899](https://arxiv.org/pdf/2307.14899)).
///
/// Every labelled claim of a starved subject is a query, not a centroid of them. `policy`
/// spans DRM, region locks, account requirements and delistings, and the mean of those points
/// is near none of them.
#[derive(Debug)]
pub struct Lines {
    /// Subject id, one per probe, in [`PROBES`] order.
    subjects: Vec<&'static str>,
    /// Which line each query belongs to, and its unit vector.
    queries: Vec<(usize, Vec<f32>)>,
}

impl Lines {
    /// Embeds every labelled claim under `reference_root` whose subject has a probe.
    ///
    /// # Errors
    ///
    /// Fails if the reference sets cannot be read or the forward pass fails.
    pub fn cast(
        embedder: &mut crate::embed::Embedder,
        reference_root: &Path,
        batch_size: usize,
    ) -> Result<Self> {
        let subjects: Vec<&'static str> = PROBES.iter().map(|probe| probe.subject).collect();
        let labelled = crate::claimset::labelled_claims(reference_root)?;

        let mut texts: Vec<String> = Vec::new();
        let mut owners: Vec<usize> = Vec::new();
        for claim in labelled {
            let Some(at) = subjects
                .iter()
                .position(|subject| *subject == claim.label.subject)
            else {
                continue;
            };
            // A claim the labeller called contested is one two subjects fit, and a query
            // that fits two subjects fishes for both.
            if claim.label.ambiguous || claim.text.trim().is_empty() {
                continue;
            }
            texts.push(claim.text);
            owners.push(at);
        }
        if texts.is_empty() {
            return Err(Error::NoReferenceSet {
                path: reference_root.to_path_buf(),
            });
        }

        let mut queries = Vec::with_capacity(texts.len());
        for (chunk, who) in texts
            .chunks(batch_size.max(1))
            .zip(owners.chunks(batch_size.max(1)))
        {
            for (vector, owner) in embedder.embed(chunk)?.into_iter().zip(who) {
                queries.push((*owner, vector));
            }
        }
        Ok(Self { subjects, queries })
    }

    /// How many queries each line holds, in [`PROBES`] order.
    #[must_use]
    pub fn cast_count(&self) -> Vec<(&'static str, usize)> {
        let mut counts = vec![0; self.subjects.len()];
        for (owner, _) in &self.queries {
            counts[*owner] += 1;
        }
        self.subjects.iter().copied().zip(counts).collect()
    }

    /// The line a claim's vector sits nearest, and how near.
    ///
    /// Nearest to any single query of the line rather than to their mean, for the reason the
    /// queries are kept separately: the subjects being fished for are the ones with several
    /// unrelated senses.
    fn nearest(&self, vector: &[f32]) -> Option<(usize, f32)> {
        let mut best: Option<(usize, f32)> = None;
        for (owner, query) in &self.queries {
            let similarity: f32 = vector.iter().zip(query).map(|(a, b)| a * b).sum();
            if best.is_none_or(|(_, found)| similarity > found) {
                best = Some((*owner, similarity));
            }
        }
        best
    }
}

/// What a retrieval draw found: the claims, and how near each line's catch was.
#[derive(Debug)]
pub struct Retrieved {
    pub drawn: Vec<crate::claimset::DrawnReview>,
    /// Per line: how many were taken, the nearest similarity and the furthest, in [`PROBES`]
    /// order. The furthest is the figure to read. A line whose two-hundredth catch sits at
    /// 0.6 is scraping the floor of the corpus for a subject it does not hold, and those two
    /// hundred labels will mostly say `gameplay`.
    pub by_line: Vec<(&'static str, usize, f32, f32)>,
    pub claims_seen: u64,
}

/// Draws the claims nearest to the labelled claims of each starved subject.
///
/// Embeds the corpus as it goes rather than reading stored vectors, for two reasons. Stored
/// claim vectors exist for no game yet, and the splitter changes: a vector file cut by one
/// splitter names claims another does not, which is the failure `check-draws` exists to catch,
/// and a draw that re-embeds cannot suffer it. The cost is a forward pass over the corpus per
/// draw, minutes on a card for a large game, which a draw of two hundred claims can afford.
///
/// A claim goes to the line it sits nearest, and each line keeps its nearest `wanted`. There
/// is deliberately no similarity floor: a floor that suits `vr` starves `licensing`, and the
/// per-line furthest similarity in [`Retrieved::by_line`] is what says whether a line was worth
/// casting in this game.
///
/// Every review carries `subset: "retrieved"`, and no prevalence figure may count one.
///
/// # Errors
///
/// Fails if there is no capture, no reading, the reading was cut by another splitter, or the
/// forward pass fails.
#[expect(
    clippy::too_many_arguments,
    reason = "a draw is its corpus, its reference set and its budget, and each is a separate thing"
)]
pub fn draw_by_neighbour(
    embedder: &mut crate::embed::Embedder,
    lines: &Lines,
    out_dir: &Path,
    app_id: u32,
    dir: &Path,
    wanted: usize,
    batch_size: usize,
    mut on_progress: impl FnMut(u64),
) -> Result<Retrieved> {
    let snapshot = crate::embed::latest_snapshot(out_dir, app_id)?;
    let reading: crate::read::ReadReport =
        serde_json::from_slice(&std::fs::read(snapshot.join("reading.json")).map_err(|_| {
            crate::Error::NoClassifications {
                path: snapshot.join("reading.json"),
            }
        })?)?;
    reading.cut_as_this_build()?;
    let depth = reading.depth;

    let already = crate::claimset::already_drawn(dir);

    // The key is the distance, quantised, so the bounded keeper's "smallest" is "nearest".
    let mut kept: Vec<crate::bounded::Smallest<u32, (String, u16, f32)>> = lines
        .subjects
        .iter()
        .map(|_| crate::bounded::Smallest::new(wanted))
        .collect();
    let mut claims_seen = 0_u64;

    let mut pending: Vec<(String, u16, String)> = Vec::with_capacity(batch_size);
    let mut flush = |pending: &mut Vec<(String, u16, String)>,
                     kept: &mut Vec<crate::bounded::Smallest<u32, (String, u16, f32)>>|
     -> Result<()> {
        if pending.is_empty() {
            return Ok(());
        }
        let texts: Vec<String> = pending.iter().map(|(_, _, text)| text.clone()).collect();
        let vectors = embedder.embed(&texts)?;
        for ((id, index, _), vector) in pending.drain(..).zip(vectors) {
            let Some((line, similarity)) = lines.nearest(&vector) else {
                continue;
            };
            kept[line].offer(distance_key(similarity), (id, index, similarity));
        }
        Ok(())
    };

    crate::capture::for_each_body(&snapshot, |id, _, text| {
        if already.contains(id) {
            return Ok(());
        }
        for (index, claim) in depth.claims_of(text).into_iter().enumerate() {
            let trimmed = claim.trim();
            if trimmed.is_empty() {
                continue;
            }
            claims_seen += 1;
            pending.push((
                id.to_owned(),
                u16::try_from(index).unwrap_or(u16::MAX),
                trimmed.to_owned(),
            ));
            if pending.len() >= batch_size {
                flush(&mut pending, &mut kept)?;
                on_progress(claims_seen);
            }
        }
        Ok(())
    })?;
    flush(&mut pending, &mut kept)?;
    on_progress(claims_seen);

    let caught: Vec<Vec<(String, u16, f32)>> = kept
        .into_iter()
        .map(crate::bounded::Smallest::take)
        .collect();
    let by_line = lines
        .subjects
        .iter()
        .zip(&caught)
        .map(|(subject, line)| {
            let nearest = line.first().map_or(0.0, |(_, _, s)| *s);
            let furthest = line.last().map_or(0.0, |(_, _, s)| *s);
            (*subject, line.len(), nearest, furthest)
        })
        .collect();
    let picks: Vec<Vec<(String, u16)>> = caught
        .into_iter()
        .map(|line| line.into_iter().map(|(id, index, _)| (id, index)).collect())
        .collect();
    let (picks, _) = crate::claimset::round_robin(&picks, wanted);
    Ok(Retrieved {
        drawn: crate::claimset::handouts(&snapshot, app_id, depth, &picks, "retrieved")?,
        by_line,
        claims_seen,
    })
}

/// A cosine similarity as a key the bounded keeper sorts ascending, nearest first.
///
/// Quantised to a millionth, which is finer than an fp16 forward pass can tell two claims
/// apart by, so nothing is lost in the ordering.
fn distance_key(similarity: f32) -> u32 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to [0, 2] and scaled to fit"
    )]
    let key = ((1.0 - similarity.clamp(-1.0, 1.0)) * 1_000_000.0) as u32;
    key
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
