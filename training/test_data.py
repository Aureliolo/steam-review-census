"""What the split has to guarantee, because every number reported rests on it.

    python -m pytest training/test_data.py
"""

from __future__ import annotations

import data as claimdata


def claim(app_id: int, index: int = 0) -> claimdata.Claim:
    return claimdata.Claim(
        review_id=f"r{app_id}-{index}",
        claim_index=index,
        app_id=app_id,
        language="english",
        subset="random",
        text="a claim",
        review="something before it. a claim. something after it.",
        review_offset=len("something before it. "),
        subject="verdict",
        polarity="praise",
        ironic=False,
        confidence="high",
        ambiguous=False,
        split_wrong=False,
    )


def roles(claims: list[claimdata.Claim]) -> dict[int, str]:
    train, validation, test = claimdata.split_by_game(claims, seed=1)
    found = {}
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        for c in part:
            found[c.app_id] = name
    return found


def test_adding_a_game_moves_no_other_game():
    # The property a frozen set needs and the shuffle did not have: a game's role depends
    # on nothing but itself. Measured before the fix, adding four games to eleven moved a
    # frozen game into training.
    games = list(range(1000, 1040))
    before = roles([claim(g) for g in games[:20]])
    after = roles([claim(g) for g in games])
    for g in games[:20]:
        assert before[g] == after[g], f"game {g} moved from {before[g]} to {after[g]}"


def test_every_role_is_filled_however_few_games_there_are():
    for count in range(3, 12):
        found = roles([claim(g) for g in range(5000, 5000 + count)])
        assert "test" in found.values(), f"{count} games left nothing frozen"
        assert "validation" in found.values(), f"{count} games left nothing to validate on"
        assert "train" in found.values(), f"{count} games left nothing to train on"


def test_the_shares_hold_over_many_games():
    found = roles([claim(g) for g in range(1, 2001)])
    counts = {name: sum(1 for role in found.values() if role == name) for name in ("test", "validation", "train")}
    assert 0.17 < counts["test"] / 2000 < 0.23, counts
    assert 0.12 < counts["validation"] / 2000 < 0.18, counts


def test_a_different_seed_is_a_different_split():
    games = [claim(g) for g in range(100, 140)]
    one = claimdata.split_by_game(games, seed=1)
    two = claimdata.split_by_game(games, seed=2)
    assert {c.app_id for c in one[2]} != {c.app_id for c in two[2]}
