//! Keeping the best few rows of a corpus without ever holding the corpus.
//!
//! Several passes here want the same shape: the handful of rows with the smallest keys out
//! of millions, in memory that does not grow with the corpus. Collecting everything and
//! sorting it is the obvious way to get that and the reason a million-review game used to
//! need gigabytes to answer a question about fifty reviews.
//!
//! Pruning on a doubling threshold costs an occasional sort of twice the limit instead of
//! one sort of everything, and the result is identical to sorting the corpus and taking the
//! front of it, which is what the tests assert.

/// Ranks a row deterministically for a given seed and purpose.
///
/// Hashing rather than shuffling means the choice depends only on the row, so a corpus that
/// gains reviews does not renumber the ones already drawn, and two runs against the same
/// corpus quote the same evidence. The purpose keeps two draws over the same corpus from
/// being the same draw.
#[must_use]
pub(crate) fn rank(seed: u64, purpose: &str, id: &str) -> [u8; 32] {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(seed.to_le_bytes());
    hasher.update(purpose.as_bytes());
    hasher.update(id.as_bytes());
    hasher.finalize().into()
}

/// Keeps the `limit` items with the smallest keys offered so far.
///
/// Smallest rather than largest because every caller here ranks by a hash, where "smallest"
/// is an arbitrary but fixed choice that makes selection reproducible. A caller wanting the
/// largest can offer a reversed key.
pub(crate) struct Smallest<K: Ord, T> {
    limit: usize,
    kept: Vec<(K, T)>,
}

impl<K: Ord, T> Smallest<K, T> {
    pub(crate) fn new(limit: usize) -> Self {
        Self {
            limit,
            kept: Vec::new(),
        }
    }

    pub(crate) fn offer(&mut self, key: K, value: T) {
        if self.limit == 0 {
            return;
        }
        self.kept.push((key, value));
        if self.kept.len() >= self.limit.saturating_mul(2) {
            self.prune();
        }
    }

    fn prune(&mut self) {
        // Stable, so two rows with equal keys keep the order the corpus offered them in and
        // a re-run of the same pass draws the same rows.
        self.kept.sort_by(|(left, _), (right, _)| left.cmp(right));
        self.kept.truncate(self.limit);
    }

    /// The kept items, smallest key first.
    pub(crate) fn take(self) -> Vec<T> {
        self.take_with_keys()
            .into_iter()
            .map(|(_, value)| value)
            .collect()
    }

    /// The kept items with their keys, for a caller that has to order them again later
    /// against items from another pass.
    pub(crate) fn take_with_keys(mut self) -> Vec<(K, T)> {
        self.prune();
        self.kept
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeping_as_it_goes_picks_what_sorting_everything_would_have_picked() {
        let keys: Vec<u32> = (0..1000)
            .map(|n: u32| n.wrapping_mul(2_654_435_761))
            .collect();

        let mut bounded = Smallest::new(7);
        for (index, key) in keys.iter().enumerate() {
            bounded.offer(*key, index);
        }

        let mut sorted: Vec<(u32, usize)> = keys.iter().copied().zip(0..).collect();
        sorted.sort_unstable();
        let expected: Vec<usize> = sorted.into_iter().take(7).map(|(_, index)| index).collect();

        assert_eq!(bounded.take(), expected);
    }

    #[test]
    fn a_limit_of_none_keeps_nothing_rather_than_everything() {
        let mut bounded = Smallest::new(0);
        for index in 0..100_u32 {
            bounded.offer(index, index);
        }
        assert!(bounded.take().is_empty());
    }

    #[test]
    fn fewer_rows_than_the_limit_come_back_in_key_order() {
        let mut bounded = Smallest::new(10);
        for key in [5_u32, 1, 9] {
            bounded.offer(key, key);
        }
        assert_eq!(bounded.take(), vec![1, 5, 9]);
    }
}
