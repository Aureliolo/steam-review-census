"""Asks a frontier model the same question the reader is asked, and scores the answers.

The other baselines are things a frontier model obviously beats. This is the one that might
beat the trained reader, and until it is run the claim that a fine-tuned 278M encoder is worth
having is an assertion. Same claims, same category sheet the labellers worked from, same
review around each claim, and the same abstention: a model that may say "unsure" is measured
on coverage as well as accuracy, exactly as the reader is.

    python frontier.py handout --out ../scratchpad/frontier/    # what the model reads
    python frontier.py score --answers ../scratchpad/frontier/answers.json

The handout deliberately carries no labels and no game names. A model told which game it is
reading can lean on what it knows about that game rather than on what the claim says, and the
reader cannot.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import data as claimdata

HERE = Path(__file__).resolve().parent
BATCH = 50


def drawn(claims, per_subject, seed):
    """A stratified draw, so the starved rows are measured rather than rounded away.

    A proportional sample of frozen claims is a quarter `verdict` and two claims of
    `licensing`, and a macro F1 over that says almost nothing about the rows that need it.
    """
    by_subject = defaultdict(list)
    for claim in claims:
        by_subject[claim.subject].append(claim)
    rng = random.Random(seed)
    picked = []
    for subject in sorted(by_subject):
        pool = by_subject[subject]
        picked.extend(rng.sample(pool, min(per_subject, len(pool))))
    rng.shuffle(picked)
    return picked


def write_handout(claims, out: Path, sheet: Path):
    out.mkdir(parents=True, exist_ok=True)
    (out / "sheet.txt").write_text(sheet.read_text(encoding="utf-8"), encoding="utf-8")

    keys = []
    for at in range(0, len(claims), BATCH):
        chunk = claims[at : at + BATCH]
        rows = []
        for index, claim in enumerate(chunk):
            claim_id = f"{at + index:04d}"
            keys.append(
                {
                    "id": claim_id,
                    "app_id": claim.app_id,
                    "review_id": claim.review_id,
                    "claim_index": claim.claim_index,
                    "subject": claim.subject,
                    "polarity": claim.polarity,
                }
            )
            rows.append(
                {
                    "id": claim_id,
                    "claim": claim.text,
                    "review": claim.review,
                    "language": claim.language,
                }
            )
        (out / f"batch-{at // BATCH:02d}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    (out / "key.json").write_text(json.dumps(keys, indent=2), encoding="utf-8")
    (out / "TASK.md").write_text(TASK, encoding="utf-8")
    return len(keys), (len(claims) + BATCH - 1) // BATCH


TASK = """# Read each claim and say what it is about

`sheet.txt` is the category sheet. Every batch file holds claims drawn from Steam reviews of
games you are not told the names of. For each claim, using the review around it as context:

1. Which category of the sheet it is about, by its `id`.
2. Whether it is `praise`, a `complaint`, or `neutral` about that subject.
3. Whether you are sure. Answer `unsure` as the subject when you genuinely cannot tell;
   that is measured as coverage, not counted against you as an error.

Answer every claim of every batch. Write one file, `answers.json`, holding a list of
`{"id": "0000", "subject": "<sheet id or unsure>", "polarity": "praise|complaint|neutral"}`.

Nothing else: no reasoning in the file, no extra fields, no claims left out.
"""


def score(answers: Path, key: Path):
    given = {row["id"]: row for row in json.loads(answers.read_text(encoding="utf-8"))}
    wanted = json.loads(key.read_text(encoding="utf-8"))

    answered = subject_right = polarity_right = 0
    missing = 0
    per_subject = defaultdict(lambda: {"found": 0, "wanted": 0, "hit": 0})
    for row in wanted:
        said = given.get(row["id"])
        if said is None:
            missing += 1
            continue
        per_subject[row["subject"]]["wanted"] += 1
        if said.get("subject") in (None, "", "unsure"):
            continue
        answered += 1
        per_subject[said["subject"]]["found"] += 1
        if said["subject"] == row["subject"]:
            subject_right += 1
            per_subject[row["subject"]]["hit"] += 1
            if said.get("polarity") == row["polarity"]:
                polarity_right += 1

    scores = []
    for counts in per_subject.values():
        if not counts["wanted"] and not counts["found"]:
            continue
        precision = counts["hit"] / max(counts["found"], 1)
        recall = counts["hit"] / max(counts["wanted"], 1)
        scores.append(
            0.0 if not counts["hit"] else 2 * precision * recall / (precision + recall)
        )

    return {
        "claims": len(wanted),
        "unanswered_rows": missing,
        "coverage": answered / max(len(wanted), 1),
        "accuracy_where_answered": subject_right / max(answered, 1),
        "polarity_where_subject_right": polarity_right / max(subject_right, 1),
        "macro_f1": sum(scores) / max(len(scores), 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    make = sub.add_parser("handout")
    make.add_argument("--data", default=str(HERE / "data" / "claims.jsonl"))
    make.add_argument("--sheet", default=str(HERE.parent / "reference" / "claim-brief.txt"))
    make.add_argument("--out", required=True)
    make.add_argument("--per-subject", type=int, default=20)
    make.add_argument("--seed", type=int, default=1)
    make.add_argument("--split-seed", type=int, default=1)

    read = sub.add_parser("score")
    read.add_argument("--answers", required=True)
    read.add_argument("--key", default=None)
    read.add_argument("--out", default=None)

    args = parser.parse_args()

    if args.mode == "handout":
        claims = claimdata.load(args.data)
        _, _, frozen = claimdata.split_by_game(claims, seed=args.split_seed)
        picked = drawn(frozen, args.per_subject, args.seed)
        count, batches = write_handout(picked, Path(args.out), Path(args.sheet))
        print(f"{count:,} claims over {batches} batches -> {args.out}")
        print(f"drawn from {len({c.app_id for c in picked})} frozen games, labels held back")
        return

    answers = Path(args.answers)
    key = Path(args.key) if args.key else answers.parent / "key.json"
    found = score(answers, key)
    print(json.dumps(found, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(found, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
