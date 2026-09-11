//! Subjects a game's own players raise that the fixed taxonomy has no row for.
//!
//! The spine is the floor, not the ceiling. A truck simulator's players talk about mud
//! physics and a horror game's about jump scares, and neither is a row every game shares. A
//! language model reads the diverse handout for a game and names what it finds; this is the
//! shape of what it hands back, and the checks that stop a hallucinated subject reaching a
//! report.
//!
//! The check is evidence. Every induced subject has to name the reviews in the handout that
//! raise it, by id, and a subject that names reviews not in the handout, or fewer than a
//! handful, is refused. A model asked for subjects will produce subjects; the reviews are what
//! make them a finding about the game rather than a list.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::Result;

/// A subject that needs this many distinct reviews behind it before it is kept.
///
/// Three, because one review is an anecdote, two is a coincidence, and three of a hundred and
/// twenty chosen to be as unlike each other as possible is a subject the corpus keeps
/// returning to.
pub const ENOUGH_REVIEWS: usize = 3;

/// One subject the model found, as it hands it back.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Induced {
    /// Short, lowercase, hyphenated. What a row is called in a table.
    pub id: String,
    /// What appears in the report.
    pub label: String,
    /// One sentence a labeller could work from: what belongs here, and what does not.
    pub description: String,
    /// Which fixed subject this is a refinement of, when it is one. A subject that is really
    /// `gameplay` for this genre is still worth its own row, and saying so is what lets a
    /// report show it under the row it refines.
    #[serde(default)]
    pub refines: Option<String>,
    /// Review ids from the handout that raise this subject.
    pub evidence: Vec<String>,
}

/// What the model was handed and what came back, together, so the subjects can be checked
/// against the reviews they claim to rest on.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InducedSet {
    pub app_id: u32,
    /// Which draw this was induced from, so the same seed can rebuild the handout.
    pub seed: u64,
    pub handout_size: usize,
    /// Which model read the handout. A subject list is only as good as its reader.
    pub induced_by: String,
    pub subjects: Vec<Induced>,
}

/// Why an induced subject was refused.
#[derive(Debug, Clone, Serialize)]
pub struct Refused {
    pub id: String,
    pub reason: String,
}

/// Checks a returned subject list against the handout it was induced from.
///
/// Returns the subjects that survive and why each of the others did not. Nothing is fixed up:
/// a subject with two pieces of evidence is refused, not kept with a note, because a report
/// row resting on two reviews is the thing the check exists to prevent.
#[must_use]
pub fn check(returned: &InducedSet, handout_ids: &[String]) -> (Vec<Induced>, Vec<Refused>) {
    let known: std::collections::HashSet<&str> = handout_ids.iter().map(String::as_str).collect();
    let spine: std::collections::HashSet<&str> =
        crate::taxonomy::CORE_SPINE.iter().map(|c| c.id).collect();

    let mut kept = Vec::new();
    let mut refused = Vec::new();
    let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

    for subject in &returned.subjects {
        let id = subject.id.trim().to_lowercase();
        if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
            refused.push(Refused {
                id: subject.id.clone(),
                reason: "id must be lowercase letters, digits and hyphens".to_owned(),
            });
            continue;
        }
        if spine.contains(id.as_str()) {
            refused.push(Refused {
                id,
                reason: "already a fixed subject; a refinement needs its own id".to_owned(),
            });
            continue;
        }
        if !seen_ids.insert(id.clone()) {
            refused.push(Refused {
                id,
                reason: "named twice".to_owned(),
            });
            continue;
        }
        if let Some(parent) = &subject.refines
            && !spine.contains(parent.as_str())
        {
            refused.push(Refused {
                id,
                reason: format!("refines `{parent}`, which is not a fixed subject"),
            });
            continue;
        }

        // A review cited twice is one review, and citing it twice is untidy rather than
        // dishonest; citing one the handout never held is the thing that refuses a subject.
        let mut cited: Vec<&String> = subject.evidence.iter().collect();
        cited.sort();
        cited.dedup();
        let evidence: Vec<String> = cited
            .iter()
            .filter(|review| known.contains(review.as_str()))
            .map(|review| (*review).clone())
            .collect();
        let invented = cited.len() - evidence.len();
        if invented > 0 {
            refused.push(Refused {
                id,
                reason: format!("{invented} of its evidence names reviews not in the handout"),
            });
            continue;
        }
        if evidence.len() < ENOUGH_REVIEWS {
            refused.push(Refused {
                id,
                reason: format!(
                    "{} review(s) is not enough; {ENOUGH_REVIEWS} distinct ones are needed",
                    evidence.len()
                ),
            });
            continue;
        }

        kept.push(Induced {
            id,
            label: subject.label.trim().to_owned(),
            description: subject.description.trim().to_owned(),
            refines: subject.refines.clone(),
            evidence,
        });
    }
    (kept, refused)
}

/// Where a game's induced subjects live.
#[must_use]
pub fn default_path(app_id: u32) -> std::path::PathBuf {
    std::path::PathBuf::from("reference")
        .join("induced")
        .join(format!("{app_id}.json"))
}

/// Reads a game's induced subjects, if any were kept.
///
/// # Errors
///
/// Fails only if the file exists and is malformed; a missing file is `None`.
pub fn load(path: &Path) -> Result<Option<InducedSet>> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        Err(_) => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn returned(subjects: Vec<Induced>) -> InducedSet {
        InducedSet {
            app_id: 1,
            seed: 1,
            handout_size: 4,
            induced_by: "test".to_owned(),
            subjects,
        }
    }

    fn subject(id: &str, evidence: &[&str]) -> Induced {
        Induced {
            id: id.to_owned(),
            label: id.to_owned(),
            description: "a subject".to_owned(),
            refines: None,
            evidence: evidence.iter().map(|s| (*s).to_owned()).collect(),
        }
    }

    fn handout() -> Vec<String> {
        ["1", "2", "3", "4"].iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn a_subject_with_enough_real_evidence_is_kept() {
        let (kept, refused) = check(&returned(vec![subject("mud", &["1", "2", "3"])]), &handout());
        assert_eq!(kept.len(), 1);
        assert!(refused.is_empty());
    }

    #[test]
    fn evidence_the_handout_never_held_refuses_the_subject() {
        // A model that cites review 99 in a handout of four is making it up, and a subject
        // that rests on made-up reviews is not kept with the real ones subtracted.
        let (kept, refused) =
            check(&returned(vec![subject("mud", &["1", "2", "3", "99"])]), &handout());
        assert!(kept.is_empty());
        assert!(refused[0].reason.contains("not in the handout"), "{}", refused[0].reason);
    }

    #[test]
    fn two_reviews_are_a_coincidence_not_a_subject() {
        let (kept, refused) = check(&returned(vec![subject("mud", &["1", "2", "2"])]), &handout());
        assert!(kept.is_empty(), "a duplicated id is one review");
        assert!(refused[0].reason.contains("not enough"));
    }

    #[test]
    fn a_fixed_subject_cannot_be_induced_again() {
        let (kept, refused) =
            check(&returned(vec![subject("gameplay", &["1", "2", "3"])]), &handout());
        assert!(kept.is_empty());
        assert!(refused[0].reason.contains("already a fixed subject"));
    }

    #[test]
    fn a_refinement_must_refine_something_that_exists() {
        let mut fine = subject("mud-physics", &["1", "2", "3"]);
        fine.refines = Some("gameplay".to_owned());
        let mut broken = subject("jump-scares", &["1", "2", "3"]);
        broken.refines = Some("scariness".to_owned());
        let (kept, refused) = check(&returned(vec![fine, broken]), &handout());
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].id, "mud-physics");
        assert!(refused[0].reason.contains("not a fixed subject"));
    }

    #[test]
    fn ids_are_made_safe_for_a_table_and_a_url() {
        let (kept, refused) =
            check(&returned(vec![subject("Mud Physics", &["1", "2", "3"])]), &handout());
        assert!(kept.is_empty(), "a space is not allowed");
        assert_eq!(refused.len(), 1);

        let (kept, _) = check(&returned(vec![subject("Mud-Physics", &["1", "2", "3"])]), &handout());
        assert_eq!(kept[0].id, "mud-physics", "case is folded rather than refused");

        // Folding case means two ids that differ only in case are the same id.
        let (kept, refused) = check(
            &returned(vec![
                subject("mud-physics", &["1", "2", "3"]),
                subject("MUD-PHYSICS", &["1", "2", "3"]),
            ]),
            &handout(),
        );
        assert_eq!(kept.len(), 1);
        assert!(refused[0].reason.contains("named twice"));
    }
}
