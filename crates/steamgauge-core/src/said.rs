//! What stands out in what was said about each subject.
//!
//! A count says a fifth of reviewers complain about performance. It does not say whether they
//! mean stutter, crashes or load times, and the reader who opened the row came for that. The
//! honest answer without a model paraphrasing anybody is the words themselves: the terms that
//! turn up in a subject's complaints far more than in its praise, and the other way round,
//! counted by how many reviewers used them.
//!
//! Compared side against side within one subject rather than against the corpus, because
//! against the corpus a subject's vocabulary is mostly the subject's name: "fps" and "runs"
//! stand out in every performance claim, praise or complaint alike, and tell a reader
//! nothing the row label did not. Complaint against praise is the comparison that separates
//! "stutter" from "smooth".
//!
//! Counted by reviews, once per review however often it repeated itself, for the same reason
//! the headline is a mention rate: nobody's verbosity moves it.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::reader::Polarity;

/// Terms shown per side of a subject.
pub const TERMS_SHOWN: usize = 8;

/// A term used by fewer reviewers than this is a coincidence rather than a finding.
pub const FEWEST_REVIEWS: u64 = 5;

/// A side with fewer reviews than this shows nothing. Below it every word the few used clears
/// the bar against the many on the other side, and "but" was the finding about a subject
/// three people praised.
pub const FEWEST_ON_A_SIDE: u64 = 20;

/// Standard deviations of log-odds a term needs before it is shown.
const CLEARLY: f64 = 2.0;

/// How much likelier a term has to be on this side than on the other, as log-odds. Twice.
/// Significance alone is not enough: over a thousand reviews "and" is significantly commoner
/// in praise than in complaint, because praise runs longer, and it is not what anybody said.
const AT_LEAST_TWICE: f64 = std::f64::consts::LN_2;

/// The share of a term's every counted use that one side of one subject must hold. A word
/// spread across every subject is a word of the language rather than of the subject, however
/// unevenly the two sides of one subject happen to use it.
const OWNED: f64 = 0.25;

/// Distinct terms kept per side of a subject while counting. Twice this is the most the map
/// ever holds, so fifty sides over a million reviews stay within tens of megabytes whatever
/// the vocabulary does.
const KEPT: usize = 8_192;

/// A term and how many reviews used it on one side of a subject.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Term {
    pub text: String,
    /// An undercount by at most a few where rare terms had to be forgotten to stay in memory,
    /// never an overcount.
    pub reviews: u64,
}

/// The terms that separate a subject's praise from its complaints.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SaidAbout {
    pub subject: String,
    /// Reviews with at least one praising claim about the subject.
    pub praising: u64,
    /// Reviews with at least one complaining claim about it.
    pub complaining: u64,
    pub praised: Vec<Term>,
    pub criticised: Vec<Term>,
}

/// Counts terms per side of every subject over a corpus, in bounded memory.
pub(crate) struct Said {
    sides: Vec<[Counter; 2]>,
    /// What the review in hand has said so far, each term once per side of a subject.
    heard: HashSet<(usize, usize, String)>,
    scratch: Terms,
}

impl Said {
    pub(crate) fn new(subjects: usize) -> Self {
        Self {
            sides: (0..subjects)
                .map(|_| [Counter::default(), Counter::default()])
                .collect(),
            heard: HashSet::new(),
            scratch: Terms::default(),
        }
    }

    /// Notes one claim of the review in hand. Neutral claims take no side and say nothing here.
    pub(crate) fn note(&mut self, subject: usize, polarity: Polarity, claim: &str) {
        let side = match polarity {
            Polarity::Praise => 0,
            Polarity::Complaint => 1,
            Polarity::Neutral => return,
        };
        if subject >= self.sides.len() {
            return;
        }
        let heard = &mut self.heard;
        self.scratch.each_in(claim, |term| {
            heard.insert((subject, side, term.to_owned()));
        });
    }

    /// Closes the review in hand and counts what it said.
    pub(crate) fn next_review(&mut self) {
        let mut sides_taken: HashSet<(usize, usize)> = HashSet::new();
        for (subject, side, term) in self.heard.drain() {
            let counter = &mut self.sides[subject][side];
            if sides_taken.insert((subject, side)) {
                counter.reviews += 1;
            }
            counter.add(term);
        }
    }

    /// What stands out on each side of every subject, in the order the subjects were given
    /// as `(id, label)`. A subject's own name is never one of its words: "graphics" under
    /// graphics and "tutorial" under tutorial say what the row label already said.
    pub(crate) fn finish(self, subjects: &[(&str, &str)]) -> Vec<SaidAbout> {
        let mut everywhere: HashMap<&str, u64> = HashMap::new();
        for [praise, complaint] in &self.sides {
            for counter in [praise, complaint] {
                for (term, count) in &counter.terms {
                    *everywhere.entry(term.as_str()).or_default() += count;
                }
            }
        }
        self.sides
            .iter()
            .zip(subjects)
            .map(|([praise, complaint], (id, label))| {
                let mut own: Vec<String> = label
                    .split(|ch: char| !ch.is_alphanumeric())
                    .filter(|word| !word.is_empty())
                    .map(str::to_lowercase)
                    .collect();
                own.push((*id).to_lowercase());
                SaidAbout {
                    subject: (*id).to_owned(),
                    praising: praise.reviews,
                    complaining: complaint.reviews,
                    praised: distinctive(praise, complaint, &everywhere, &own),
                    criticised: distinctive(complaint, praise, &everywhere, &own),
                }
            })
            .collect()
    }
}

/// Whether a claim uses a term, by the same cut that counted it, so the claims a term opens
/// onto are exactly the ones it was counted from.
#[must_use]
pub fn mentions(claim: &str, term: &str) -> bool {
    let mut found = false;
    Terms::default().each_in(claim, |candidate| {
        if candidate == term {
            found = true;
        }
    });
    found
}

/// Term counts for one side of one subject, forgetting the rarest when it has to.
#[derive(Debug, Default)]
struct Counter {
    reviews: u64,
    terms: HashMap<String, u64>,
    /// The largest count ever dropped. A term absent from `terms` was used by at most this
    /// many reviews, which is what the other side is charged with when it lacks a term.
    forgotten: u64,
}

impl Counter {
    fn add(&mut self, term: String) {
        if let Some(count) = self.terms.get_mut(&term) {
            *count += 1;
            return;
        }
        self.terms.insert(term, 1);
        if self.terms.len() >= KEPT * 2 {
            self.forget();
        }
    }

    /// Keeps the commonest half. A term that comes back after being dropped starts again
    /// from one, so what survives is an undercount rather than an estimate, and a term used
    /// by many reviewers is never in the half that goes.
    fn forget(&mut self) {
        let mut kept: Vec<(String, u64)> = self.terms.drain().collect();
        kept.sort_unstable_by(|(left_term, left), (right_term, right)| {
            right.cmp(left).then_with(|| left_term.cmp(right_term))
        });
        if let Some((_, dropped)) = kept.get(KEPT) {
            self.forgotten = self.forgotten.max(*dropped);
        }
        kept.truncate(KEPT);
        self.terms = kept.into_iter().collect();
    }
}

/// The terms far commoner on `this` side than on `other`, and belonging to this subject
/// rather than to the language, most clearly so first.
#[expect(
    clippy::cast_precision_loss,
    reason = "review counts are far below 2^53"
)]
fn distinctive(
    this: &Counter,
    other: &Counter,
    everywhere: &HashMap<&str, u64>,
    own_words: &[String],
) -> Vec<Term> {
    if this.reviews < FEWEST_ON_A_SIDE {
        return Vec::new();
    }
    let mut scored: Vec<(f64, &str, u64)> = this
        .terms
        .iter()
        .filter(|(_, here)| **here >= FEWEST_REVIEWS)
        // The row's own name says nothing new alone; in a phrase ("no bugs") it does.
        .filter(|(term, _)| !own_words.iter().any(|word| word == *term))
        .filter(|(term, here)| {
            let counted = everywhere.get(term.as_str()).copied().unwrap_or(**here);
            **here as f64 >= OWNED * counted as f64
        })
        .filter_map(|(term, &here)| {
            let there = other.terms.get(term).copied().unwrap_or(other.forgotten);
            let (delta, z) = log_odds(here, this.reviews, there, other.reviews);
            (delta >= AT_LEAST_TWICE && z >= CLEARLY).then_some((z, term.as_str(), here))
        })
        .collect();
    // On a dead heat the phrase beats the word: "frame drops" and "drops" seen by exactly the
    // same reviewers are one finding, and the phrase is the one that names it.
    scored.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| right.2.cmp(&left.2))
            .then_with(|| right.1.len().cmp(&left.1.len()))
            .then_with(|| left.1.cmp(right.1))
    });

    let mut shown: Vec<Term> = Vec::with_capacity(TERMS_SHOWN);
    for (_, term, reviews) in scored {
        if shown.len() == TERMS_SHOWN {
            break;
        }
        // "full price" and "full" say one thing. The word outranks the phrase whenever a few
        // reviewers used it in some other phrase, and the phrase is still what was said, so
        // it takes the word's place unless it was noticeably rarer.
        if let Some(kept) = shown
            .iter_mut()
            .find(|kept| shares_a_word(&kept.text, term))
        {
            if term.contains(' ') && !kept.text.contains(' ') && nearly(reviews, kept.reviews) {
                term.clone_into(&mut kept.text);
                kept.reviews = reviews;
            }
            continue;
        }
        // 中文, 文配 and 配音 used by the same reviewers are one word, 中文配音, cut into the
        // pairs the counter works in. Joined back up where the pairs overlap.
        if let Some(kept) = shown
            .iter_mut()
            .find(|kept| nearly(reviews, kept.reviews) && overlaps(&kept.text, term))
        {
            kept.text = joined(&kept.text, term);
            kept.reviews = kept.reviews.min(reviews);
            continue;
        }
        shown.push(Term {
            text: term.to_owned(),
            reviews,
        });
    }
    shown
}

/// Whether two counts are within a twentieth of each other, which is what "the same
/// reviewers" means once a few of them have phrased a thing two ways.
fn nearly(left: u64, right: u64) -> bool {
    left * 20 >= right * 19 && right * 20 >= left * 19
}

/// Whether one term is a whole word of the other. "price tag" and "tag" are one finding;
/// "full price" and "price tag" are two, and a reader wants both.
fn shares_a_word(left: &str, right: &str) -> bool {
    left.split(' ').any(|word| word == right) || right.split(' ').any(|word| word == left)
}

/// Whether a pair of characters from a script without spaces continues or precedes a run.
fn overlaps(run: &str, pair: &str) -> bool {
    let mut chars = pair.chars();
    let (Some(first), Some(second), None) = (chars.next(), chars.next(), chars.next()) else {
        return false;
    };
    if !crate::claims::writes_without_spaces(first) || !crate::claims::writes_without_spaces(second)
    {
        return false;
    }
    run.ends_with(first) || run.starts_with(second)
}

/// The run with the pair joined on at whichever end it overlaps.
fn joined(run: &str, pair: &str) -> String {
    let mut chars = pair.chars();
    let (first, second) = (chars.next().unwrap_or('\0'), chars.next().unwrap_or('\0'));
    if run.ends_with(first) {
        let mut out = run.to_owned();
        out.push(second);
        out
    } else {
        let mut out = String::from(first);
        out.push_str(run);
        out
    }
}

/// The log-odds of a term between two sides, and how many standard deviations that is.
///
/// Half a count added to every cell so a term absent from one side is very unlikely rather
/// than infinitely so, which is the usual prior and keeps a term seen five times against
/// none from outranking one seen three hundred times against ten.
#[expect(
    clippy::cast_precision_loss,
    reason = "review counts are far below 2^53"
)]
fn log_odds(here: u64, of: u64, there: u64, of_other: u64) -> (f64, f64) {
    let a = here as f64 + 0.5;
    let b = of.saturating_sub(here) as f64 + 0.5;
    let c = there as f64 + 0.5;
    let d = of_other.saturating_sub(there) as f64 + 0.5;
    let delta = (a / b).ln() - (c / d).ln();
    let sigma = (1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d).sqrt();
    (delta, delta / sigma)
}

/// Cuts a claim into the terms worth counting: words, pairs of adjacent words, and pairs of
/// adjacent characters in the scripts that write without spaces.
///
/// Reused across claims so a corpus of millions does not allocate a buffer per word.
#[derive(Debug, Default)]
struct Terms {
    word: String,
    previous: String,
    pair: String,
}

impl Terms {
    fn each_in(&mut self, claim: &str, mut visit: impl FnMut(&str)) {
        self.word.clear();
        self.previous.clear();
        let mut last_dense: Option<char> = None;

        for ch in claim.chars() {
            if crate::claims::writes_without_spaces(ch) {
                self.close_word(&mut visit);
                self.previous.clear();
                if let Some(before) = last_dense {
                    self.pair.clear();
                    self.pair.push(before);
                    self.pair.push(ch);
                    // A filler pair breaks the chain as a filler word does, so 没有中文 yields
                    // 中文 and not the bridge 有中 as well.
                    if filler().contains(self.pair.as_str()) {
                        last_dense = None;
                        continue;
                    }
                    visit(&self.pair);
                }
                last_dense = Some(ch);
                continue;
            }
            last_dense = None;
            if ch.is_alphanumeric() {
                for lower in ch.to_lowercase() {
                    self.word.push(lower);
                }
            } else if (ch == '\'' || ch == '\u{2019}') && !self.word.is_empty() {
                // "don't" is one word, and the curly apostrophe phones type is the same one.
                self.word.push('\'');
            } else {
                self.close_word(&mut visit);
                // A pair of words straddling a comma or a bracket is not a phrase anybody
                // used; only a space carries the chain on, and a hyphen, because "full-price"
                // and "full price" are the same two words.
                if !ch.is_whitespace() && ch != '-' {
                    self.previous.clear();
                }
            }
        }
        self.close_word(&mut visit);
    }

    /// Emits the word in hand, and its pair with the word before it, then remembers it.
    fn close_word(&mut self, visit: &mut impl FnMut(&str)) {
        let word = self.word.trim_matches('\'');
        if word.is_empty() {
            return;
        }
        // "runs at 60 fps" must not yield "at fps": a word not worth counting still breaks
        // the chain of pairs.
        if word.chars().count() < 2 || word.chars().all(|ch| ch.is_ascii_digit()) {
            self.word.clear();
            self.previous.clear();
            return;
        }
        let kind = kind_of(word);
        if kind == Word::Content {
            visit(word);
            if !self.previous.is_empty() {
                self.pair.clear();
                self.pair.push_str(&self.previous);
                self.pair.push(' ');
                self.pair.push_str(word);
                visit(&self.pair);
            }
        }
        // A pair ends on a content word and starts on one, or on the few function words that
        // change what follows: "not worth" and "on sale" are findings, "worth it" and "is
        // great" are not, and "it on" was the best a thousand price claims had to offer.
        self.previous.clear();
        if kind != Word::Filler {
            self.previous.push_str(word);
        }
        self.word.clear();
    }
}

/// What a word is worth on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Word {
    /// Says something: counted alone and in pairs.
    Content,
    /// Says nothing alone but turns what follows: "not", "no", "too", "on".
    Modifier,
    /// Says nothing and is counted nowhere: "the", "is", "game".
    Filler,
}

fn kind_of(word: &str) -> Word {
    if MODIFIERS.contains(&word) {
        Word::Modifier
    } else if filler().contains(word) {
        Word::Filler
    } else {
        Word::Content
    }
}

/// Function words that carry meaning into the word after them.
const MODIFIERS: [&str; 12] = [
    "no", "not", "never", "without", "too", "on", "off", "less", "more", "only", "still", "always",
];

/// Function words, and the handful of words that are function words in a Steam review:
/// every claim is about a game somebody played. English, Chinese and a short list each for
/// the next six languages Steam is written in; the rest mostly fail the ownership bar on
/// their own, and a list for each of forty languages would be a maintenance burden with no
/// measured return. Sentiment words are not here: "great" fails the ownership bar by itself,
/// and "worth" is what the price subject is made of. The Chinese entries are pairs, because
/// pairs are what the counter cuts that script into.
const FILLER: &str = "\
the a an and or but if so as of to in at by for from with into onto about over under than \
then that this these those there here it its it's is are was were be been being am i i'm i've \
i'd i'll you your you're he she they them their we our us me my mine him his her who whom \
whose which what when where why how have has had having do does did doing done will would can \
could should may might must shall get got gets getting make makes made go goes going went \
come comes came also just very really quite pretty even much many most some any all each \
every both few such own other another same one ones thing things something anything nothing \
everything someone anyone everyone yes yeah yep nope ok okay well like lot lots bit way \
because cause while though although since until after before again out up down back through \
around now ever yet far game games play played playing plays player players steam review \
reviews \
游戏 玩家 这个 那个 一个 自己 不是 就是 可以 什么 但是 因为 所以 如果 还是 已经 觉得 知道 一下 \
一些 非常 比较 这样 那么 然后 而且 或者 虽然 不过 真的 感觉 我们 你们 他们 没有 有点 这种 那种 \
时候 东西 的话 是的 不能 不会 不要 应该 可能 现在 之后 之前 以及 对于 关于 需要 只是 只有 而已 \
的时 我的 你的 它的 他的 这些 那些 一样 一直 一点 一定 一起 还有 也是 都是 就会 就能 不了 不太 \
太多 很多 玩了 玩的 玩过 玩到 小时 多小 个小 \
и в не на что это как но а то же он она они мы вы я ты у из за для по от о до при или \
если бы был была было были есть нет очень так только уже еще ещё все всё игра игры игру \
игре этот эта это эти его её их мне меня тебе вас нам них там тут здесь \
der das und ist nicht ein eine einer einen dem den des ich du er sie es wir ihr \
mit von zu auf für aus bei nach über auch nur noch schon sehr aber oder wenn dass wie \
wo da hier dort wird sind habe haben kann spiel spiele spielen \
el la los las un una unos unas y o pero de del en con por para que es son está están \
muy más menos también ya no sí este esta esto ese esa eso lo le les se me te su sus mi \
juego juegos jugar jugando \
o a os as um uma e é são está estão de do da dos das em no na nos nas com por para que \
não sim muito mais menos também já este esta isto esse essa isso ele ela eles elas eu tu \
jogo jogos jogar jogando \
le la les un une des et ou mais de du au aux en dans sur pour par avec sans que qui ne \
pas plus moins très aussi est sont c'est il elle ils elles je tu on nous vous ce cette \
ces ça jeu jeux jouer \
i w na z do nie się jest są to tak jak ale co za od po dla przez ten ta te tego tej tym \
gra gry grę grze grać \
ve bir bu şu o de da için ile çok daha ama ya en gibi kadar var yok mi mı mu mü değil \
oyun oyunu oyunda oyna";

/// The filler words as a set, built once: the tokeniser asks about every word of every claim.
fn filler() -> &'static HashSet<&'static str> {
    static SET: std::sync::OnceLock<HashSet<&'static str>> = std::sync::OnceLock::new();
    SET.get_or_init(|| FILLER.split_whitespace().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(claim: &str) -> Vec<String> {
        let mut found = Vec::new();
        Terms::default().each_in(claim, |term| found.push(term.to_owned()));
        found
    }

    #[test]
    fn words_and_their_pairs_stop_at_punctuation_and_skip_bare_numbers() {
        assert_eq!(
            terms("Frame drops in the city, 60fps elsewhere. 10/10"),
            [
                "frame",
                "drops",
                "frame drops",
                "city",
                "60fps",
                "elsewhere",
                "60fps elsewhere",
            ]
        );
    }

    #[test]
    fn apostrophes_stay_inside_a_word_and_case_is_folded() {
        assert_eq!(terms("Don’t BUY it"), ["don't", "buy", "don't buy"]);
    }

    #[test]
    fn a_function_word_leads_a_pair_only_when_it_turns_the_word_after_it() {
        assert_eq!(
            terms("not worth it on sale, and the story is short"),
            ["worth", "not worth", "sale", "on sale", "story", "short"]
        );
    }

    #[test]
    fn a_dense_script_counts_pairs_of_characters() {
        assert_eq!(terms("画面很好 lags 卡"), ["画面", "面很", "很好", "lags"]);
    }

    #[test]
    fn a_claim_mentions_a_term_by_the_cut_that_counted_it() {
        assert!(mentions("Constant frame DROPS in town", "frame drops"));
        assert!(mentions("Constant frame DROPS in town", "town"));
        assert!(!mentions("frame rate drops", "frame drops"));
        assert!(!mentions("framedrops", "drops"));
        assert!(mentions("画面很好", "面很"));
    }

    #[test]
    fn a_review_counts_a_term_once_per_side_however_often_it_repeats_it() {
        let mut said = Said::new(1);
        said.note(0, Polarity::Complaint, "stutter everywhere");
        said.note(0, Polarity::Complaint, "constant stutter");
        said.note(0, Polarity::Praise, "stutter aside, it looks great");
        said.next_review();
        assert_eq!(said.sides[0][1].terms["stutter"], 1);
        assert_eq!(said.sides[0][1].reviews, 1);
        assert_eq!(said.sides[0][0].terms["stutter"], 1);
        assert_eq!(said.sides[0][0].reviews, 1);
    }

    #[test]
    fn what_stands_out_is_what_one_side_says_and_the_other_does_not() {
        let mut said = Said::new(1);
        for review in 0..100 {
            said.note(0, Polarity::Praise, "the game runs smooth for me");
            if review < 40 {
                said.note(0, Polarity::Complaint, "the game stutters in the city");
            } else {
                said.note(0, Polarity::Complaint, "the game crashed once");
            }
            said.next_review();
        }
        let found = said.finish(&[("performance", "Performance")]);
        let praised: Vec<&str> = found[0].praised.iter().map(|t| t.text.as_str()).collect();
        let criticised: Vec<&str> = found[0]
            .criticised
            .iter()
            .map(|t| t.text.as_str())
            .collect();

        assert!(praised.iter().any(|t| t.contains("smooth")), "{praised:?}");
        assert!(
            criticised.iter().any(|t| t.contains("stutters")),
            "{criticised:?}"
        );
        assert!(
            criticised.iter().any(|t| t.contains("crashed")),
            "{criticised:?}"
        );
        // Said equally often on both sides, so it says nothing about either.
        for common in ["the", "game", "the game"] {
            assert!(!praised.contains(&common), "{praised:?}");
            assert!(!criticised.contains(&common), "{criticised:?}");
        }
        assert_eq!(found[0].praising, 100);
        assert_eq!(found[0].complaining, 100);
        let stutters = found[0]
            .criticised
            .iter()
            .find(|t| t.text.contains("stutters"))
            .unwrap();
        assert_eq!(stutters.reviews, 40);
        // The commoner complaint stands out more clearly than the rarer one.
        let crashed = found[0]
            .criticised
            .iter()
            .position(|t| t.text.contains("crashed"))
            .unwrap();
        let stuttered = found[0]
            .criticised
            .iter()
            .position(|t| t.text.contains("stutters"))
            .unwrap();
        assert!(crashed < stuttered);
    }

    #[test]
    fn a_side_with_too_few_reviews_shows_nothing_rather_than_its_accidents() {
        let mut said = Said::new(1);
        said.note(0, Polarity::Complaint, "laggy menus");
        said.next_review();
        said.note(0, Polarity::Complaint, "laggy menus");
        said.next_review();
        for _ in 0..50 {
            said.note(0, Polarity::Praise, "buttery smooth");
            said.next_review();
        }
        let found = said.finish(&[("performance", "Performance")]);
        assert!(found[0].criticised.is_empty(), "{:?}", found[0].criticised);
    }

    #[test]
    fn a_phrase_and_a_word_inside_it_are_shown_once() {
        let mut said = Said::new(1);
        for _ in 0..60 {
            said.note(0, Polarity::Complaint, "frame drops");
            said.note(0, Polarity::Praise, "looks lovely");
            said.next_review();
        }
        let found = said.finish(&[("performance", "Performance")]);
        let criticised: Vec<&str> = found[0]
            .criticised
            .iter()
            .map(|t| t.text.as_str())
            .collect();
        assert_eq!(criticised.len(), 1, "{criticised:?}");
    }

    #[test]
    fn forgetting_keeps_the_commonest_and_records_the_most_it_dropped() {
        let mut counter = Counter::default();
        for _ in 0..5 {
            counter.add("common".to_owned());
        }
        for n in 0..(KEPT * 2) {
            counter.add(format!("rare{n}"));
        }
        assert!(counter.terms.len() < KEPT * 2);
        assert_eq!(counter.terms["common"], 5);
        assert_eq!(counter.forgotten, 1);
    }

    #[test]
    fn log_odds_prefer_the_well_attested_over_the_merely_absent_elsewhere() {
        let (_, few) = log_odds(3, 100, 0, 100);
        let (_, many) = log_odds(300, 1000, 10, 1000);
        assert!(many > few, "{many} vs {few}");
        assert!(log_odds(90, 100, 85, 100).1 < CLEARLY);
    }

    #[test]
    fn a_word_of_the_language_is_not_a_finding_however_significant_its_lean() {
        // "and" in 42% of a thousand praising reviews against 30% of five hundred complaining
        // ones is a real difference and says nothing: praise runs longer. It fails the twice
        // bar. A word used across every subject fails the ownership bar even where it leans.
        let (delta, z) = log_odds(420, 1000, 150, 500);
        assert!(z >= CLEARLY, "{z}");
        assert!(delta < AT_LEAST_TWICE, "{delta}");

        let subjects = [
            ("gameplay", "Gameplay"),
            ("audio", "Audio"),
            ("story", "Story"),
            ("graphics", "Graphics"),
            ("controls", "Controls"),
        ];
        let praised_for = ["combat", "music", "plot", "colours", "keybinds"];
        let mut said = Said::new(subjects.len());
        for review in 0..200 {
            for (slot, aspect) in praised_for.iter().enumerate() {
                said.note(
                    slot,
                    Polarity::Praise,
                    &format!("you will love the {aspect}"),
                );
                if review % 4 == 0 {
                    said.note(slot, Polarity::Complaint, "you might hate it though");
                }
            }
            said.next_review();
        }
        let found = said.finish(&subjects);
        let praised: Vec<&str> = found[0].praised.iter().map(|t| t.text.as_str()).collect();
        assert!(praised.iter().any(|t| t.contains("combat")), "{praised:?}");
        // "love" leans entirely to praise and clears every bar but one: it is a fifth of
        // every subject's praise, so it belongs to none of them.
        assert!(!praised.iter().any(|t| t.contains("love")), "{praised:?}");
        assert!(!praised.iter().any(|t| t.contains("you")), "{praised:?}");
    }

    #[test]
    fn a_subject_never_stands_out_for_its_own_name_unless_the_name_is_in_a_phrase() {
        let mut said = Said::new(1);
        for review in 0..100 {
            said.note(0, Polarity::Praise, "no bugs at all, the graphics shine");
            if review < 30 {
                said.note(0, Polarity::Complaint, "bugs, bugs, bugs");
            }
            said.next_review();
        }
        let found = said.finish(&[("bugs", "Bugs and crashes")]);
        let praised: Vec<&str> = found[0].praised.iter().map(|t| t.text.as_str()).collect();
        assert!(praised.contains(&"no bugs"), "{praised:?}");
        assert!(!praised.contains(&"bugs"), "{praised:?}");
        assert!(found[0].criticised.is_empty(), "{:?}", found[0].criticised);
    }

    #[test]
    fn the_phrase_takes_the_place_of_a_word_nearly_always_said_inside_it() {
        let mut said = Said::new(1);
        for review in 0..100 {
            said.note(0, Polarity::Praise, "cheap enough");
            said.note(
                0,
                Polarity::Complaint,
                if review < 97 {
                    "full price is a joke"
                } else {
                    "the full game is a joke"
                },
            );
            said.next_review();
        }
        let found = said.finish(&[("price", "Price and value")]);
        let criticised: Vec<&str> = found[0]
            .criticised
            .iter()
            .map(|t| t.text.as_str())
            .collect();
        assert!(criticised.contains(&"full price"), "{criticised:?}");
        assert!(!criticised.contains(&"full"), "{criticised:?}");
    }

    #[test]
    fn character_pairs_used_by_the_same_reviewers_are_joined_back_into_the_word() {
        let mut said = Said::new(1);
        for _ in 0..30 {
            said.note(0, Polarity::Complaint, "没有中文配音");
            said.note(0, Polarity::Praise, "very good indeed");
            said.next_review();
        }
        let found = said.finish(&[("language", "Language and localisation")]);
        let criticised: Vec<&str> = found[0]
            .criticised
            .iter()
            .map(|t| t.text.as_str())
            .collect();
        // 没有 is "there is no", a function word, and the pair bridging it to the noun goes
        // with it: what is left is the noun, joined back up from its pairs.
        assert_eq!(criticised, ["中文配音"]);
    }
}
