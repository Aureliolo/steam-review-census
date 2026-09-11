//! Drawing a sample that covers a corpus rather than one that resembles it.
//!
//! A random sample of a million reviews is mostly "great game" and "10/10", because that is
//! what a million reviews mostly are, and a reader hoping to find what this game's players
//! talk about that other games' do not would see none of it. What that reader wants is the
//! opposite of representative: a handful of reviews chosen so that each says something the
//! others do not.
//!
//! Farthest-point traversal does that. Start anywhere, then repeatedly take the review whose
//! vector is farthest from everything already taken. The exact version is quadratic in the
//! corpus, which for a million reviews is a week; this one walks a bounded pool of candidates
//! drawn by hash, which is the same draw for the same corpus and seed, and takes the farthest
//! from that. The pool is a few thousand, which is enough to hold every neighbourhood a
//! corpus has and small enough that the traversal is seconds.

use std::path::Path;

use crate::{Result, bounded::Smallest};

/// How many candidates are drawn before the farthest are chosen among them.
///
/// Sized so that a subject raised by one review in a thousand still has a few candidates in
/// the pool, and so that the traversal over it takes seconds rather than minutes.
const POOL: usize = 4_096;

/// One review chosen for what it adds.
#[derive(Debug, Clone)]
pub struct Distinct {
    pub text_hash: String,
    /// How far this was from everything chosen before it, as one minus the cosine to its
    /// nearest chosen neighbour. The first pick has no neighbours and is recorded as 1.
    pub novelty: f32,
}

/// Draws `wanted` distinct texts from a corpus's stored vectors.
///
/// Returns them in the order they were chosen, which is also the order of decreasing novelty:
/// the first few are the corners of the corpus, and the last are filling in between.
///
/// # Errors
///
/// Fails if the embeddings are missing or malformed.
pub fn draw(out_dir: &Path, app_id: u32, wanted: usize, seed: u64) -> Result<Vec<Distinct>> {
    let snapshot = crate::embed::latest_snapshot(out_dir, app_id)?;

    // The candidate pool, by hash of the text hash so the same corpus gives the same pool.
    let mut pool: Smallest<[u8; 32], (String, Vec<f32>)> = Smallest::new(POOL);
    crate::embed::for_each_vector(&snapshot, |hash, vector| {
        pool.offer(
            crate::bounded::rank(seed, "diverse", hash),
            (hash.to_owned(), vector.to_vec()),
        );
        Ok(())
    })?;
    let candidates = pool.take();
    Ok(farthest_first(&candidates, wanted))
}

/// One drawn review with its text, ready to be read by whatever induces subjects from it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Handout {
    pub review_id: String,
    pub language: String,
    pub novelty: f32,
    pub review: String,
}

/// Draws a diverse sample and resolves it to review text, in the order it was chosen.
///
/// Reviews whose text is already in the sample are skipped, so five hundred copies of one
/// copypasta contribute at most one review however far from everything else it sits.
///
/// # Errors
///
/// Fails if the capture or the embeddings cannot be read.
pub fn handout(out_dir: &Path, app_id: u32, wanted: usize, seed: u64) -> Result<Vec<Handout>> {
    let snapshot = crate::embed::latest_snapshot(out_dir, app_id)?;
    let drawn = draw(out_dir, app_id, wanted, seed)?;
    let novelty: std::collections::HashMap<&str, f32> = drawn
        .iter()
        .map(|pick| (pick.text_hash.as_str(), pick.novelty))
        .collect();

    // One walk of the capture for all of them. Whichever review with a given text comes
    // first is the one shown, which is arbitrary and stable.
    let mut found: std::collections::HashMap<String, Handout> = std::collections::HashMap::new();
    crate::capture::for_each_body(&snapshot, |id, language, body| {
        let hash = crate::embed::sha256_hex(body);
        if let Some(&novelty) = novelty.get(hash.as_str())
            && !found.contains_key(&hash)
        {
            found.insert(
                hash,
                Handout {
                    review_id: id.to_owned(),
                    language: language.to_owned(),
                    novelty,
                    review: body.to_owned(),
                },
            );
        }
        Ok(())
    })?;

    Ok(drawn
        .into_iter()
        .filter_map(|pick| found.remove(&pick.text_hash))
        .collect())
}

/// Greedy farthest-point traversal over unit vectors.
///
/// Every vector's distance to the chosen set is kept as one number, the distance to its
/// nearest chosen neighbour, and updated with a single dot product each time something new is
/// chosen. That makes the whole traversal `wanted × pool` dot products rather than the
/// `pool²` of doing it naively.
fn farthest_first(candidates: &[(String, Vec<f32>)], wanted: usize) -> Vec<Distinct> {
    let mut chosen = Vec::with_capacity(wanted.min(candidates.len()));
    if candidates.is_empty() || wanted == 0 {
        return chosen;
    }

    // Distance to the nearest chosen point, starting at "infinitely far" for all so the first
    // pick is whichever is first: any point is as good a corner as any other to start from,
    // and the second pick is the true opposite of it.
    let mut nearest: Vec<f32> = vec![f32::INFINITY; candidates.len()];
    let mut taken = vec![false; candidates.len()];

    let mut pick = 0;
    for _ in 0..wanted.min(candidates.len()) {
        taken[pick] = true;
        let novelty = if nearest[pick].is_finite() {
            nearest[pick]
        } else {
            1.0
        };
        chosen.push(Distinct {
            text_hash: candidates[pick].0.clone(),
            novelty,
        });

        let just_chosen = &candidates[pick].1;
        let mut farthest = None::<(usize, f32)>;
        for (index, (_, vector)) in candidates.iter().enumerate() {
            if taken[index] {
                continue;
            }
            let distance = 1.0 - dot(vector, just_chosen);
            if distance < nearest[index] {
                nearest[index] = distance;
            }
            if farthest.is_none_or(|(_, best)| nearest[index] > best) {
                farthest = Some((index, nearest[index]));
            }
        }
        let Some((next, _)) = farthest else {
            break;
        };
        pick = next;
    }
    chosen
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(x: f32, y: f32) -> Vec<f32> {
        let norm = (x * x + y * y).sqrt();
        vec![x / norm, y / norm]
    }

    #[test]
    fn the_second_pick_is_the_farthest_point_from_the_first() {
        // Three near each other, one opposite. Whatever comes first, the opposite one is
        // second: that is the whole point of the traversal.
        let candidates = vec![
            ("a".to_owned(), unit(1.0, 0.0)),
            ("b".to_owned(), unit(1.0, 0.1)),
            ("c".to_owned(), unit(1.0, -0.1)),
            ("far".to_owned(), unit(-1.0, 0.0)),
        ];
        let chosen = farthest_first(&candidates, 2);
        assert_eq!(chosen[1].text_hash, "far");
        assert!(
            chosen[1].novelty > 1.9,
            "opposite unit vectors are two apart, got {}",
            chosen[1].novelty
        );
    }

    #[test]
    fn near_duplicates_are_chosen_last_if_at_all() {
        let candidates = vec![
            ("a".to_owned(), unit(1.0, 0.0)),
            ("a-again".to_owned(), unit(1.0, 0.001)),
            ("b".to_owned(), unit(0.0, 1.0)),
            ("c".to_owned(), unit(-1.0, 0.0)),
        ];
        let chosen: Vec<String> = farthest_first(&candidates, 3)
            .into_iter()
            .map(|d| d.text_hash)
            .collect();
        assert!(
            !chosen.iter().any(|hash| hash == "a-again"),
            "a near copy of something chosen adds nothing and should lose to anything else: \
             {chosen:?}"
        );
    }

    #[test]
    fn asking_for_more_than_there_is_returns_what_there_is() {
        let candidates = vec![("only".to_owned(), unit(1.0, 0.0))];
        assert_eq!(farthest_first(&candidates, 10).len(), 1);
        assert!(farthest_first(&[], 10).is_empty());
        assert!(farthest_first(&candidates, 0).is_empty());
    }

    #[test]
    fn novelty_never_rises_along_the_draw() {
        let candidates: Vec<(String, Vec<f32>)> = (0_u8..40)
            .map(|i| {
                let angle = f32::from(i) * 0.157;
                (i.to_string(), unit(angle.cos(), angle.sin()))
            })
            .collect();
        let chosen = farthest_first(&candidates, 12);
        for pair in chosen.windows(2).skip(1) {
            assert!(
                pair[1].novelty <= pair[0].novelty + 1e-6,
                "each pick is the farthest left, so novelty can only fall: {} then {}",
                pair[0].novelty,
                pair[1].novelty
            );
        }
    }
}
