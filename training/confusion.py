"""What the reader mistakes for what, which is the question a bad subject always raises.

A per-subject F1 says a subject is weak. It never says whether the cure is more labels, a
clearer boundary in the sheet, or a bigger model, and those have nothing in common. The
direction of the confusion does say:

  scattered across many subjects   the reader has not learned this subject; a starved row,
                                   and more labels are the cure
  concentrated on one other        the boundary between those two is not drawn where the
                                   sheet thinks it is, and no amount of labelling fixes a
                                   boundary the labeller cannot apply either
  right, but below the threshold   the reader knows and does not know that it knows, which
                                   is a calibration problem and not a knowledge one

Run on the validation games, never the frozen ones.

    python confusion.py --model ../models/claim-reader --subject story
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import claimdata
from confidence import logits_of, pooled_folds, softmax

HERE = Path(__file__).resolve().parent


def rates(truth, predicted, index):
    """Precision, recall and support for one subject, from the counts themselves.

    Written out rather than imported so this runs without sklearn, which the training
    environment does not otherwise need.
    """
    said = predicted == index
    is_it = truth == index
    hit = (said & is_it).sum()
    precision = hit / said.sum() if said.sum() else float("nan")
    recall = hit / is_it.sum() if is_it.sum() else float("nan")
    return float(precision), float(recall), int(is_it.sum()), int(said.sum())


def confusions(truth, predicted, index, subjects, most=4):
    """The subjects this one is mistaken for, and the ones mistaken for it."""
    became = np.bincount(predicted[(truth == index) & (predicted != index)], minlength=len(subjects))
    came_from = np.bincount(truth[(predicted == index) & (truth != index)], minlength=len(subjects))

    def top(counts):
        order = np.argsort(-counts)[:most]
        return [(subjects[at], int(counts[at])) for at in order if counts[at]]

    return top(became), top(came_from)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(HERE.parent / "models" / "claim-reader"))
    parser.add_argument("--data", default=str(HERE / "data" / "claims.jsonl"))
    parser.add_argument(
        "--oof",
        default=None,
        help="directory of .npz files written by `train.py --save-logits`. Twenty-eight games "
        "of held-out answers instead of four, which is the difference between a confusion "
        "worth acting on and one built from thirty claims.",
    )
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--subject", default=None, help="one subject to read in full")
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()

    validation: list = []
    if args.oof:
        logits, truth, _, subjects = pooled_folds(Path(args.oof).glob("*.npz"))
        where = f"{len(truth):,} out-of-fold claims, from {args.oof}"
    else:
        model_dir = Path(args.model)
        provenance = json.loads((model_dir / "reader.json").read_text(encoding="utf-8"))
        subjects = provenance["subjects"]
        claims = claimdata.load(args.data)
        _, validation, _ = claimdata.split_by_game(claims, seed=args.split_seed)
        validation = [claim for claim in validation if claim.subject in subjects]
        logits = logits_of(model_dir, validation, provenance)
        truth = np.array([subjects.index(claim.subject) for claim in validation])
        where = f"{len(validation):,} validation claims, model {model_dir}"

    probabilities = softmax(logits)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    print(f"{where}\n")
    print(f"{'subject':<16} {'true':>5} {'said':>5} {'prec':>6} {'rec':>6}   mistaken for")
    for index, name in enumerate(subjects):
        precision, recall, support, said = rates(truth, predicted, index)
        if not support and not said:
            continue
        became, _ = confusions(truth, predicted, index, subjects, most=3)
        trail = ", ".join(f"{other} {count}" for other, count in became)
        print(
            f"{name:<16} {support:>5} {said:>5} {precision:>6.2f} {recall:>6.2f}   {trail}"
        )

    if not args.subject:
        return
    if args.subject not in subjects:
        raise SystemExit(f"{args.subject!r} is not a subject this reader knows")

    index = subjects.index(args.subject)
    became, came_from = confusions(truth, predicted, index, subjects, most=8)
    precision, recall, support, said = rates(truth, predicted, index)
    print(f"\n{args.subject}: {support} true, {said} predicted, precision {precision:.2f}")
    print("  labelled it, read as:  " + ", ".join(f"{n} {c}" for n, c in became))
    print("  read as it, labelled:  " + ", ".join(f"{n} {c}" for n, c in came_from))

    # What the reader actually said it about, most confidently first, because the confident
    # mistakes are the ones a threshold cannot catch and are what a boundary problem looks
    # like from the inside.
    wrong = np.flatnonzero((predicted == index) & (truth != index))
    print(f"\n  the {min(args.examples, len(wrong))} most confident of its {len(wrong)} mistakes")
    if not validation:
        # The fold files carry the logits and the claim's name, never its text: what a review
        # says is not written into this repository, and a diagnostic is not a reason to start.
        print("    (the text is not in the fold files; re-run without --oof to read them)")
        return
    for at in wrong[np.argsort(-confidence[wrong])][: args.examples]:
        claim = validation[at]
        text = " ".join(claim.text.split())[:96]
        print(f"    {confidence[at]:.2f}  labelled {subjects[truth[at]]:<14} {text}")


if __name__ == "__main__":
    main()
