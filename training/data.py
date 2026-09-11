"""Loading labelled claims, and splitting them so a number means something.

The split is the part worth reading twice. Claims from one game are not independent of each
other, so a random split would put a reviewer's own words on both sides of it and report a
score the model cannot repeat on a game it has never seen. Games are therefore split whole.

Three parts, and they have different jobs:

  train        what the model learns from
  validation   what every choice is made against: thresholds, epochs, backbone
  test         frozen before the first run and read once, at the end

Anything tuned against the test part stops being a measurement, so nothing here returns it
except `held_out`, and nothing calls that until there is a number to publish.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

SUBJECTS: list[str] = []  # filled from the data, then asserted against the spine
POLARITIES = ["praise", "complaint", "neutral"]
CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.7, "low": 0.4}


@dataclass(frozen=True)
class Claim:
    text: str
    review: str
    subject: str
    polarity: str
    confidence: str
    ambiguous: bool
    ironic: bool
    split_wrong: bool
    language: str
    app_id: int
    review_id: str
    claim_index: int
    subset: str

    @property
    def weight(self) -> float:
        """How much this label is trusted, from the labeller's own confidence.

        Training against a flattened id throws away the one thing the labeller said about
        their own uncertainty. A truthful "low" is worth more than a confident wrong answer,
        so it is worth less in the loss rather than being dropped.
        """
        return CONFIDENCE_WEIGHT.get(self.confidence, 0.7)


def load(path: str | Path) -> list[Claim]:
    """Reads the JSONL that `census export-training` writes."""
    claims = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            claims.append(
                Claim(
                    text=row["text"],
                    review=row.get("review") or row["text"],
                    subject=row["subject"],
                    polarity=row.get("polarity", "neutral"),
                    confidence=row.get("confidence", "medium"),
                    ambiguous=bool(row.get("ambiguous")),
                    ironic=bool(row.get("ironic")),
                    split_wrong=bool(row.get("split_wrong")),
                    language=row.get("language", ""),
                    app_id=int(row["app_id"]),
                    review_id=str(row["review_id"]),
                    claim_index=int(row["claim_index"]),
                    subset=row.get("subset", "stratified"),
                )
            )
    if not claims:
        raise SystemExit(f"{path} holds no labels; run `census export-training` first")
    return claims


def fingerprint(claims: list[Claim]) -> str:
    """A hash of the label set, recorded with every run.

    Two runs reporting different numbers from "the same data" is the most expensive kind of
    confusion, and the only defence is for the data to be able to say which it was.
    """
    digest = hashlib.sha256()
    for claim in sorted(claims, key=lambda c: (c.app_id, c.review_id, c.claim_index)):
        digest.update(
            f"{claim.app_id}/{claim.review_id}/{claim.claim_index}/{claim.subject}/"
            f"{claim.polarity}/{claim.confidence}\n".encode()
        )
    return digest.hexdigest()[:16]


def split_by_game(
    claims: list[Claim], seed: int = 1, test_share: float = 0.2, validation_share: float = 0.15
) -> tuple[list[Claim], list[Claim], list[Claim]]:
    """Splits whole games, never claims, so a score is about a game nobody trained on.

    With fewer than three games there is no such split to make, and whole reviews are held out
    instead. That is a weaker guarantee and the number it produces is worth less: it says the
    model generalises across reviews of one game, not across games. Anything reported from it
    has to say so.
    """
    games = sorted({claim.app_id for claim in claims})
    if len(games) < 3:
        return _split_by_review(claims, seed, test_share, validation_share)

    # Each game's role comes from its own id and the seed, never from which other games are
    # labelled. Shuffling the list of games would reassign every role each time a game was
    # added, and the "games the model never saw" would quietly be different games on every
    # run. Measured before this: eleven games froze {1295660, 1361210}; fifteen froze three
    # others and put 1361210 into training.
    #
    # So a game is frozen if its hash falls in the lowest fifth of the range, validates if in
    # the next fifteenth, and trains otherwise, against fixed thresholds rather than a ranking.
    # A ranking would still move the boundary as the count grew. The shares hold only in
    # expectation; a set this small can land short, and the floor below keeps it from landing
    # at nothing, which is the one outcome worse than an uneven split.
    placed = {app_id: place(app_id, seed) for app_id in games}
    test_games = {app_id for app_id, at in placed.items() if at < test_share}
    validation_games = {
        app_id
        for app_id, at in placed.items()
        if test_share <= at < test_share + validation_share
    }
    ranked = sorted(games, key=lambda app_id: placed[app_id])
    if not test_games:
        test_games = {ranked[0]}
        validation_games.discard(ranked[0])
    if not validation_games:
        for app_id in ranked:
            if app_id not in test_games:
                validation_games = {app_id}
                break

    train, validation, test = [], [], []
    for claim in claims:
        if claim.app_id in test_games:
            test.append(claim)
        elif claim.app_id in validation_games:
            validation.append(claim)
        else:
            train.append(claim)
    return train, validation, test


def place(app_id: int, seed: int) -> float:
    """Where a game sits in [0, 1), from its id and the seed alone.

    SHA-256 rather than Python's hash, which is salted per process for strings and would put a
    game somewhere different on every run.
    """
    digest = hashlib.sha256(f"{seed}:{app_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _split_by_review(
    claims: list[Claim], seed: int, test_share: float, validation_share: float
) -> tuple[list[Claim], list[Claim], list[Claim]]:
    reviews = sorted({(claim.app_id, claim.review_id) for claim in claims})
    shuffled = list(reviews)
    random.Random(seed).shuffle(shuffled)

    held = max(1, round(len(shuffled) * test_share))
    checked = max(1, round(len(shuffled) * validation_share))
    test_reviews = set(shuffled[:held])
    validation_reviews = set(shuffled[held : held + checked])

    train, validation, test = [], [], []
    for claim in claims:
        key = (claim.app_id, claim.review_id)
        if key in test_reviews:
            test.append(claim)
        elif key in validation_reviews:
            validation.append(claim)
        else:
            train.append(claim)
    return train, validation, test


def subjects_in(claims: list[Claim]) -> list[str]:
    return sorted({claim.subject for claim in claims})


def distribution(claims: list[Claim]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for claim in claims:
        counts[claim.subject] = counts.get(claim.subject, 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: -pair[1]))


def summarise(claims: list[Claim]) -> str:
    languages: dict[str, int] = {}
    for claim in claims:
        languages[claim.language] = languages.get(claim.language, 0) + 1
    top = ", ".join(f"{name} {count}" for name, count in sorted(languages.items(), key=lambda p: -p[1])[:6])
    contested = sum(1 for claim in claims if claim.ambiguous)
    miscut = sum(1 for claim in claims if claim.split_wrong)
    return (
        f"{len(claims)} claims from {len({c.app_id for c in claims})} games\n"
        f"  languages: {top}\n"
        f"  contested: {contested} ({contested / len(claims):.1%})\n"
        f"  mis-split: {miscut} ({miscut / len(claims):.1%})\n"
        f"  subjects:  {len(subjects_in(claims))}"
    )
