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


def wilson(hits: int, total: int, z: float = 1.96):
    """A proportion's interval, the same way every other figure in this project reports one."""
    if total == 0:
        return None
    rate = hits / total
    middle = rate + z * z / (2 * total)
    spread = z * ((rate * (1 - rate) + z * z / (4 * total)) / total) ** 0.5
    divisor = 1 + z * z / total
    return ((middle - spread) / divisor, (middle + spread) / divisor)


def score(answers: Path, key: Path, subjects: set[str] | None = None):
    given = {row["id"]: row for row in json.loads(answers.read_text(encoding="utf-8"))}
    wanted = json.loads(key.read_text(encoding="utf-8"))

    answered = subject_right = polarity_right = 0
    missing = 0
    off_sheet = defaultdict(int)
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
        # A subject the sheet does not have is a wrong answer, but it is a different kind of
        # wrong from picking the neighbouring category, and it is worth reporting apart.
        if subjects is not None and said["subject"] not in subjects:
            off_sheet[said["subject"]] += 1
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
        "accuracy_interval": wilson(subject_right, answered),
        "polarity_where_subject_right": polarity_right / max(subject_right, 1),
        "macro_f1": sum(scores) / max(len(scores), 1),
        "off_sheet_subjects": dict(off_sheet),
    }


def score_reader(model_dir: Path, key: Path, data: str):
    """The shipped reader, over exactly the claims the frontier model was given.

    Without this the comparison is a cheat. The reader's frozen figure is over every frozen
    claim, which is a quarter `verdict`; the frontier handout is stratified, twenty a subject,
    and the starved rows are the hard ones. Two numbers from two distributions are not a
    comparison, however carefully each was measured.
    """
    import numpy as np
    import onnxruntime
    from tokenizers import Tokenizer

    from train import Claims

    provenance = json.loads((model_dir / "reader.json").read_text(encoding="utf-8"))
    subjects = provenance["subjects"]
    threshold = provenance["threshold"]
    wanted = json.loads(key.read_text(encoding="utf-8"))

    held = {(c.app_id, c.review_id, c.claim_index): c for c in claimdata.load(data)}
    claims = [held[(row["app_id"], row["review_id"], row["claim_index"])] for row in wanted]

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    cut = Claims(
        claims,
        _HuggingFaceShim(tokenizer),
        subjects,
        provenance["max_tokens"],
        provenance.get("context", False),
        # A model trained to look for the marks and scored without them is scored on input it
        # has never seen, which understates it and looks like a worse model rather than a
        # worse measurement.
        mark=provenance.get("mark", False),
    )

    session = onnxruntime.InferenceSession(
        str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    outputs = [out.name for out in session.get_outputs()]

    confidence, predicted, polarity = [], [], []
    for at in range(0, len(claims), 64):
        chunk = list(range(at, min(at + 64, len(claims))))
        encoded = [
            tokenizer.encode(claims[i].text, cut.windows[i])
            if cut.context
            else tokenizer.encode(claims[i].text)
            for i in chunk
        ]
        longest = max(min(len(e.ids), provenance["max_tokens"]) for e in encoded)
        ids = np.zeros((len(chunk), longest), dtype=np.int64)
        mask = np.zeros((len(chunk), longest), dtype=np.int64)
        for row, one in enumerate(encoded):
            keep = one.ids[:longest]
            ids[row, : len(keep)] = keep
            mask[row, : len(keep)] = one.attention_mask[:longest]
        found = session.run(outputs, {"input_ids": ids, "attention_mask": mask})
        subject_logits, polarity_logits = found[0], found[1]
        for row in range(len(chunk)):
            exp = np.exp(subject_logits[row] - subject_logits[row].max())
            probabilities = exp / exp.sum()
            confidence.append(float(probabilities.max()))
            predicted.append(subjects[int(probabilities.argmax())])
            polarity.append(
                claimdata.POLARITIES[int(np.argmax(polarity_logits[row]))]
                if int(np.argmax(polarity_logits[row])) < len(claimdata.POLARITIES)
                else "neutral"
            )

    answers = [
        {
            "id": row["id"],
            "subject": predicted[at] if confidence[at] >= threshold else "unsure",
            "polarity": polarity[at],
        }
        for at, row in enumerate(wanted)
    ]
    return answers


class _HuggingFaceShim:
    """Enough of the transformers tokenizer for `Claims.window` to run on the shipped one.

    The window is cut with the tokenizer that will read the claim, and the model directory
    ships `tokenizer.json` rather than a transformers directory. Duplicating the window in a
    second place is how the tool and the trainer come to disagree about what the model saw.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text, *rest, **kwargs):
        encoded = self.tokenizer.encode(text, *rest, add_special_tokens=False)
        return {"offset_mapping": encoded.offsets, "input_ids": encoded.ids}


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
    read.add_argument("--data", default=str(HERE / "data" / "claims.jsonl"))
    read.add_argument(
        "--by",
        required=True,
        help="which model answered, as its own name. A benchmark row nobody can attribute is "
        "a row nobody can reproduce or argue with.",
    )

    mine = sub.add_parser("reader")
    mine.add_argument("--model", default=str(HERE.parent / "models" / "claim-reader"))
    mine.add_argument("--key", default=None)
    mine.add_argument(
        "--frozen",
        action="store_true",
        help="score every frozen claim, writing the key first. This is the run that has to "
        "reproduce what training reported, and `steamgauge measure-claims` over the same "
        "games is the third opinion: any two of them more than a point apart is a bug.",
    )
    mine.add_argument("--app-id", type=int, default=None, help="one game of the frozen set")
    mine.add_argument("--split-seed", type=int, default=1)
    mine.add_argument("--data", default=str(HERE / "data" / "claims.jsonl"))
    mine.add_argument("--answers", default=None, help="where to write the reader's answers")
    mine.add_argument("--out", default=None)

    args = parser.parse_args()

    if args.mode == "handout":
        claims = claimdata.load(args.data)
        _, _, frozen = claimdata.split_by_game(claims, seed=args.split_seed)
        picked = drawn(frozen, args.per_subject, args.seed)
        count, batches = write_handout(picked, Path(args.out), Path(args.sheet))
        print(f"{count:,} claims over {batches} batches -> {args.out}")
        print(f"drawn from {len({c.app_id for c in picked})} frozen games, labels held back")
        return

    if args.mode == "reader":
        model_dir = Path(args.model)
        if args.frozen:
            # Every frozen claim rather than a stratified draw, because this is the run that
            # has to reproduce what training reported, and training reported over all of them.
            claims = claimdata.load(args.data)
            _, _, frozen = claimdata.split_by_game(claims, seed=args.split_seed)
            if args.app_id:
                frozen = [claim for claim in frozen if claim.app_id == args.app_id]
            # Beside the export it is built from, which is where this project keeps the files
            # that hold review ids rather than review text and are not committed either way.
            key = Path(args.key) if args.key else HERE / "data" / "frozen-key.json"
            key.write_text(
                json.dumps(
                    [
                        {
                            "id": f"{at:05d}",
                            "app_id": claim.app_id,
                            "review_id": claim.review_id,
                            "claim_index": claim.claim_index,
                            "subject": claim.subject,
                            "polarity": claim.polarity,
                        }
                        for at, claim in enumerate(frozen)
                    ]
                ),
                encoding="utf-8",
            )
            print(f"{len(frozen):,} frozen claims -> {key}")
        else:
            key = Path(args.key)
        said = score_reader(model_dir, key, args.data)
        written = Path(args.answers) if args.answers else key.parent / "reader-answers.json"
        written.write_text(json.dumps(said, indent=2), encoding="utf-8")
        found = score(written, key, {row["subject"] for row in json.loads(key.read_text("utf-8"))})
        found["model"] = str(model_dir)
        print(json.dumps(found, indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(found, indent=2), encoding="utf-8")
        return

    answers = Path(args.answers)
    key = Path(args.key) if args.key else answers.parent / "key.json"
    found = score(answers, key, {claim.subject for claim in claimdata.load(args.data)})
    found["answered_by"] = args.by
    print(json.dumps(found, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(found, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
