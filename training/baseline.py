"""What the trained reader has to beat, measured the same way it is.

Three baselines over the same split, the same games and the same selective-prediction
protocol, because a number with nothing beside it says only that the thing runs:

- **The commonest subject.** The floor. Anything that cannot beat it has learned nothing, and
  on a set where one subject is a quarter of the claims it is not as low a floor as it sounds.
- **Bag of words.** TF-IDF over words and character n-grams into a linear model. Multilingual
  by accident rather than by design, trains in seconds, and is the honest thing to beat: if a
  278M-parameter encoder cannot clear it, it is not earning its electricity.
- **Nearest subject centroid over the untuned backbone.** What this project did before it
  trained anything, and the reason it stopped: cosine distance to a prototype has no way to
  say "this is about nothing", so it answers every claim including "gfg".

    python baseline.py
    python baseline.py --frozen
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from transformers import AutoModel, AutoTokenizer

import claimdata
from train import area_under_risk_coverage, risk_coverage

HERE = Path(__file__).resolve().parent


def scored(name, confidence, predicted, truth, min_accuracy):
    """The same figures the trained model reports, from any predictor's confidences."""
    truth = np.asarray(truth)
    correct = (np.asarray(predicted) == truth).astype(float)
    curve, chosen = risk_coverage(np.asarray(confidence), correct, min_accuracy)
    return {
        "name": name,
        "accuracy": float(correct.mean()),
        "macro_f1": macro_f1(predicted, truth),
        "aurc": area_under_risk_coverage(np.asarray(confidence), correct),
        "threshold": chosen["threshold"],
        "threshold_coverage": chosen["coverage"],
        "threshold_accuracy": chosen["accuracy"],
        "threshold_met": chosen["met"],
        "risk_coverage": curve,
    }


def macro_f1(predicted, truth):
    """Unweighted mean F1, so a subject with forty claims counts as much as one with four
    thousand. The starved rows are the ones worth knowing about."""
    predicted, truth = np.asarray(predicted), np.asarray(truth)
    scores = []
    for subject in np.unique(truth):
        found = predicted == subject
        wanted = truth == subject
        hit = float((found & wanted).sum())
        precision = hit / max(found.sum(), 1)
        recall = hit / max(wanted.sum(), 1)
        scores.append(0.0 if hit == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def commonest(train, held, subjects):
    """Always the subject most of the training claims are about."""
    counts = np.bincount([subjects.index(c.subject) for c in train], minlength=len(subjects))
    best = int(counts.argmax())
    share = float(counts[best] / counts.sum())
    return np.full(len(held), share), np.full(len(held), best)


def bag_of_words(train, held, subjects):
    """TF-IDF over words and characters into a linear model.

    Character n-grams are what make this work at all across languages: a Russian claim shares
    no words with an English one and plenty of substrings with another Russian one.
    """
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True
                ),
            ),
        ]
    )
    x = features.fit_transform([c.text for c in train])
    y = [subjects.index(c.subject) for c in train]
    model = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    model.fit(x, y)
    probabilities = model.predict_proba(features.transform([c.text for c in held]))
    # The classes the fit actually saw, which is not every subject when one is absent here.
    order = model.classes_
    return probabilities.max(axis=1), order[probabilities.argmax(axis=1)]


def centroids(train, held, subjects, backbone, batch_size=64):
    """Nearest subject centroid over the untuned encoder: the prototype this replaced."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    encoder = AutoModel.from_pretrained(backbone, trust_remote_code=True).to(device).eval()

    def embed(claims):
        out = []
        with torch.inference_mode():
            for at in range(0, len(claims), batch_size):
                chunk = [c.text for c in claims[at : at + batch_size]]
                fed = tokenizer(
                    chunk, padding=True, truncation=True, max_length=128, return_tensors="pt"
                ).to(device)
                hidden = encoder(**fed).last_hidden_state
                mask = fed["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                out.append(torch.nn.functional.normalize(pooled, dim=1).float().cpu())
        return torch.cat(out).numpy()

    trained = embed(train)
    middles = np.zeros((len(subjects), trained.shape[1]), dtype=np.float32)
    for index, name in enumerate(subjects):
        rows = [at for at, claim in enumerate(train) if claim.subject == name]
        if rows:
            middles[index] = trained[rows].mean(axis=0)
    middles /= np.linalg.norm(middles, axis=1, keepdims=True).clip(min=1e-9)

    similarity = embed(held) @ middles.T
    # A cosine is not a probability, and reading one as though it were is how the prototype
    # came to answer everything. Softmax over the row at least orders the claims by how much
    # the nearest centroid beat the others, which is the only thing a threshold can use.
    exponent = np.exp((similarity - similarity.max(axis=1, keepdims=True)) * 20.0)
    probabilities = exponent / exponent.sum(axis=1, keepdims=True)
    return probabilities.max(axis=1), probabilities.argmax(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(HERE / "data" / "claims.jsonl"))
    parser.add_argument("--backbone", default="Alibaba-NLP/gte-multilingual-base")
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--min-accuracy", type=float, default=0.75)
    parser.add_argument(
        "--frozen",
        action="store_true",
        help="score against the frozen games rather than the validation ones. A set used to "
        "choose between models cannot also say how the chosen one does, and these baselines "
        "choose nothing, so the frozen set is the right one to quote them on.",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    claims = claimdata.load(args.data)
    train, validation, test = claimdata.split_by_game(claims, seed=args.split_seed)
    held = test if args.frozen else validation
    subjects = claimdata.subjects_in(claims)
    truth = [subjects.index(claim.subject) for claim in held]

    print(f"{len(train):,} training claims, {len(held):,} held out")
    print(f"games held out: {sorted({claim.app_id for claim in held})}")

    results = []
    for name, run in (
        ("commonest subject", commonest),
        ("bag of words", bag_of_words),
        ("nearest centroid", lambda a, b, c: centroids(a, b, c, args.backbone)),
    ):
        confidence, predicted = run(train, held, subjects)
        results.append(scored(name, confidence, predicted, truth, args.min_accuracy))
        last = results[-1]
        answered = (
            f"{last['threshold_coverage']:.0%} at {last['threshold_accuracy']:.3f}"
            if last["threshold_met"]
            else "never reaches the promise"
        )
        print(
            f"{name:<20} accuracy {last['accuracy']:.3f}  macro F1 {last['macro_f1']:.3f}  "
            f"AURC {last['aurc']:.3f}  answers {answered}"
        )

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "held_out": "frozen" if args.frozen else "validation",
                    "games": sorted({claim.app_id for claim in held}),
                    "claims": len(held),
                    "data_fingerprint": claimdata.fingerprint(claims),
                    "baselines": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
