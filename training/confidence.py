"""Which uncertainty score orders the reader's answers best.

The reader abstains on the largest softmax probability, which is the obvious choice and not
usually the best one. Two others cost nothing to compute from the same forward pass: the
margin between the best class and the second, and the entropy of the whole distribution. A
better ordering is coverage for free, with no retraining and no labels.

Measured over the validation games, never the frozen ones.

    python confidence.py --model ../models/claim-reader
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime
from tokenizers import Tokenizer

import claimdata
from train import Claims, area_under_risk_coverage, risk_coverage

HERE = Path(__file__).resolve().parent


class _Shim:
    """Enough of the transformers tokenizer for `Claims.window`, over the shipped one."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text, *rest, **kwargs):
        encoded = self.tokenizer.encode(text, *rest, add_special_tokens=False)
        return {"offset_mapping": encoded.offsets, "input_ids": encoded.ids}


def logits_of(model_dir: Path, claims, provenance):
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()

    budget = provenance["max_tokens"]
    context = provenance.get("context", False)
    cut = Claims(
        claims,
        None,
        provenance["subjects"],
        budget,
        context,
        mark=provenance.get("mark", False),
        prefix=provenance.get("prefix", False),
    )
    cut.tokenizer = _Shim(tokenizer)
    cut.windows = [cut.window(claim) for claim in claims] if context else []

    session = onnxruntime.InferenceSession(
        str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    names = [out.name for out in session.get_outputs()]

    found = []
    for at in range(0, len(claims), 64):
        chunk = range(at, min(at + 64, len(claims)))
        encoded = [tokenizer.encode(*cut.pair(i)) for i in chunk]
        longest = max(min(len(one.ids), budget) for one in encoded)
        ids = np.zeros((len(encoded), longest), dtype=np.int64)
        mask = np.zeros((len(encoded), longest), dtype=np.int64)
        for row, one in enumerate(encoded):
            keep = one.ids[:longest]
            ids[row, : len(keep)] = keep
            mask[row, : len(keep)] = one.attention_mask[:longest]
        found.append(session.run(names, {"input_ids": ids, "attention_mask": mask})[0])
    return np.concatenate(found)


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(HERE.parent / "models" / "claim-reader"))
    parser.add_argument("--data", default=str(HERE / "data" / "claims.jsonl"))
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--min-accuracy", type=float, default=0.75)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model_dir = Path(args.model)
    provenance = json.loads((model_dir / "reader.json").read_text(encoding="utf-8"))
    subjects = provenance["subjects"]

    claims = claimdata.load(args.data)
    _, validation, _ = claimdata.split_by_game(claims, seed=args.split_seed)
    validation = [claim for claim in validation if claim.subject in subjects]
    print(f"{len(validation):,} validation claims, model {model_dir}")

    logits = logits_of(model_dir, validation, provenance)
    probabilities = softmax(logits)
    predicted = probabilities.argmax(axis=1)
    truth = np.array([subjects.index(claim.subject) for claim in validation])
    correct = (predicted == truth).astype(float)
    print(f"accuracy over everything: {correct.mean():.3f}")

    ordered = np.sort(probabilities, axis=1)
    scores = {
        "max probability": probabilities.max(axis=1),
        "margin over the runner-up": ordered[:, -1] - ordered[:, -2],
        "negative entropy": -(-probabilities * np.log(probabilities + 1e-12)).sum(axis=1),
    }

    found = {}
    for name, score in scores.items():
        # The curve sweeps 0.05 to 0.99, which is a probability's range and not a margin's or
        # an entropy's, so each score is put on the same scale before it is swept.
        low, high = float(score.min()), float(score.max())
        scaled = (score - low) / max(high - low, 1e-9)
        curve, chosen = risk_coverage(scaled, correct, args.min_accuracy)
        found[name] = {
            "aurc": area_under_risk_coverage(score, correct),
            "coverage": chosen["coverage"],
            "accuracy": chosen["accuracy"],
            "met": chosen["met"],
            "threshold_scaled": chosen["threshold"],
            "curve_points": len(curve),
        }
        answered = (
            f"{chosen['coverage']:.1%} at {chosen['accuracy']:.3f}"
            if chosen["met"]
            else "never reaches the promise"
        )
        print(f"{name:<26} AURC {found[name]['aurc']:.4f}   answers {answered}")

    if args.out:
        Path(args.out).write_text(json.dumps(found, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
