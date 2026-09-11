"""Fine-tunes a multilingual encoder to read one claim.

Two heads on a shared trunk: which subject the claim is about, and whether it is praise, a
complaint, or neither. One forward pass produces both, plus the pooled vector the clustering
uses, which is the reason for fine-tuning the encoder rather than bolting a classifier onto a
frozen one.

The loss is weighted by the labeller's own confidence. A claim they called "low" moves the
model less than one they called "high", because throwing that away and training on a flattened
id discards the only thing the annotator said about their own uncertainty.

    python train.py --backbone xlm-roberta-base
    python train.py --backbone Alibaba-NLP/gte-multilingual-base --epochs 4
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

import data as claimdata

HERE = Path(__file__).resolve().parent
POLARITIES = claimdata.POLARITIES


class Claims(Dataset):
    def __init__(
        self,
        claims,
        tokenizer,
        subjects,
        max_length,
        context=False,
        ambiguous_weight=1.0,
        split_wrong_weight=1.0,
    ):
        self.claims = claims
        self.ambiguous_weight = ambiguous_weight
        self.split_wrong_weight = split_wrong_weight
        self.tokenizer = tokenizer
        self.subjects = {name: index for index, name in enumerate(subjects)}
        self.polarities = {name: index for index, name in enumerate(POLARITIES)}
        self.max_length = max_length
        # Whether the model reads the claim with the review around it, as the labeller did.
        # "it doesn't" and "same here" are unanswerable alone, and every one of them in the
        # training set is a label the model is asked to reach from text that cannot reach it.
        self.context = context
        # The window depends on the claim and the budget, neither of which moves, so it is cut
        # once rather than once per epoch. It costs a tokenisation of the whole review.
        self.windows = [self.window(claim) for claim in claims] if context else []

    def __len__(self):
        return len(self.claims)

    def window(self, claim):
        """The part of the review the budget can afford, centred on the claim.

        Truncating a pair from the end spends the whole budget on the opening of the review,
        so a claim at the foot of a long one is read beside paragraphs it is nowhere near.
        The budget is the same either way; where it is spent is not, and it is worth the most
        immediately around the claim.
        """
        at = claim.review_offset
        if at < 0:
            return claim.review
        offsets = self.tokenizer(
            claim.review, add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        if not offsets:
            return claim.review
        ends = at + len(claim.text)
        first = next((index for index, (_, end) in enumerate(offsets) if end > at), 0)
        last = next(
            (index for index, (start, _) in enumerate(offsets) if start >= ends), len(offsets)
        )
        # Four special tokens on a pair, and the claim is spent twice: once as the first
        # sequence, and again where it sits inside the window.
        spare = max(self.max_length - 2 * (last - first) - 4, 0) // 2
        opens = offsets[max(first - spare, 0)][0]
        closes = offsets[min(last + spare, len(offsets)) - 1][1]
        return claim.review[opens:closes]

    def __getitem__(self, at):
        claim = self.claims[at]
        encoded = self.tokenizer(
            claim.text,
            *([self.windows[at]] if self.context else []),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "subject": torch.tensor(self.subjects[claim.subject]),
            "polarity": torch.tensor(self.polarities.get(claim.polarity, 2)),
            "weight": torch.tensor(self.weight_of(claim), dtype=torch.float),
        }

    def weight_of(self, claim):
        """What this claim is worth in the loss.

        Three things the labeller said about their own answer, rather than one. A claim they
        called contested is one where the sheet does not settle the boundary, and a claim they
        called mis-cut is half a point or two stuck together: training on either at full
        weight teaches the model to reproduce a coin toss confidently.
        """
        weight = claim.weight
        if claim.ambiguous:
            weight *= self.ambiguous_weight
        if claim.split_wrong:
            weight *= self.split_wrong_weight
        return weight


class ClaimReader(torch.nn.Module):
    """The trunk, plus the two heads, plus the pooled vector everything else wants."""

    def __init__(self, backbone: str, subjects: int, dropout: float = 0.1):
        super().__init__()
        config = AutoConfig.from_pretrained(backbone, trust_remote_code=True)
        self.trunk = AutoModel.from_pretrained(backbone, trust_remote_code=True)
        width = getattr(config, "hidden_size", 768)
        self.drop = torch.nn.Dropout(dropout)
        self.subject = torch.nn.Linear(width, subjects)
        self.polarity = torch.nn.Linear(width, len(POLARITIES))

    def forward(self, input_ids, attention_mask):
        hidden = self.trunk(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # Mean pooling rather than the first token: the backbones being compared were not all
        # pretrained with a sentence-level CLS, and a pooling choice that only suits some of
        # them would decide the bake-off instead of the models doing it.
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        dropped = self.drop(pooled)
        return self.subject(dropped), self.polarity(dropped), pooled


def macro_f1(truth, predicted, classes):
    scores = []
    per_class = {}
    for index, name in enumerate(classes):
        hit = int(((predicted == index) & (truth == index)).sum())
        said = int((predicted == index).sum())
        was = int((truth == index).sum())
        if was == 0:
            continue
        precision = hit / said if said else 0.0
        recall = hit / was
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": was}
        scores.append(f1)
    return (sum(scores) / len(scores) if scores else 0.0), per_class


def expected_calibration_error(confidence, correct, bins=15):
    """How far the model's stated certainty is from how often it is right.

    A threshold for abstention is only meaningful if 0.6 means roughly 60%, so this is
    reported beside accuracy rather than after somebody asks.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (confidence > low) & (confidence <= high)
        if not inside.any():
            continue
        error += inside.mean() * abs(correct[inside].mean() - confidence[inside].mean())
    return float(error)


def risk_coverage(confidence, correct, min_accuracy):
    """Where to stop answering, and the whole curve the choice was made from.

    The obvious objective, accuracy times coverage, is degenerate on a model that is not yet
    good. Coverage rises faster than accuracy falls all the way down, so the product is
    maximised at the bottom of the sweep and the model is told to answer everything. Measured:
    it chose 0.05, which for twenty-four subjects is barely above the 0.042 a uniform guess
    scores, and a corpus of 17,596 claims came back with nothing declined. That is the failure
    this model was built to end, arrived at by arithmetic instead of by cosine distance.

    So the objective is the one selective prediction actually asks for: answer as much as
    possible, on the condition that what you do answer is right at least `min_accuracy` of the
    time. The lowest threshold meeting that condition is the most coverage available at the
    promised quality. When no threshold meets it the model is not good enough to promise it,
    and that is recorded rather than rounded away.
    """
    curve = []
    for floor in np.arange(0.05, 0.991, 0.01):
        sure = confidence >= floor
        if not sure.any():
            continue
        curve.append(
            {
                "threshold": round(float(floor), 3),
                "coverage": float(sure.mean()),
                "accuracy": float(correct[sure].mean()),
            }
        )

    meeting = [point for point in curve if point["accuracy"] >= min_accuracy]
    if meeting:
        best = max(meeting, key=lambda point: point["coverage"])
        return curve, {**best, "met": True}

    # Nothing reaches the bar. The most accurate point available is what there is, and the
    # `met` flag is what stops it being read as though it had cleared it.
    best = max(curve, key=lambda point: point["accuracy"]) if curve else None
    if best is None:
        return curve, {"threshold": 1.0, "coverage": 0.0, "accuracy": None, "met": False}
    return curve, {**best, "met": False}


def selective(confidence_and_predictions, truth, threshold):
    """How much a threshold answers, and how often it is right when it does."""
    confidence, predicted = confidence_and_predictions
    truth = np.asarray(truth)
    sure = confidence >= threshold
    if not sure.any():
        return {"coverage": 0.0, "accuracy": None, "answered": 0}
    correct = (predicted[sure] == truth[sure]).astype(float)
    return {
        "coverage": float(sure.mean()),
        "accuracy": float(correct.mean()),
        "answered": int(sure.sum()),
    }


@torch.no_grad()
def confidence_of(model, loader, device):
    """Each claim's best subject and how sure the model is of it."""
    model.eval()
    logits = []
    for batch in loader:
        subject, _, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        logits.append(subject.float().cpu())
    probabilities = torch.softmax(torch.cat(logits), dim=1).numpy()
    return probabilities.max(axis=1), probabilities.argmax(axis=1)


def area_under_risk_coverage(confidence, correct):
    """How good the confidence ordering is, without reference to any threshold.

    Lower is better. A threshold is a policy; this is the property the policy is drawn from,
    and it is what tells you whether a backbone knows when it does not know. Two models can
    reach the same accuracy and only one of them be usable with abstention.
    """
    order = np.argsort(-confidence)
    ranked = correct[order]
    risks = [1.0 - ranked[: size + 1].mean() for size in range(len(ranked))]
    return float(np.mean(risks)) if risks else 0.0


@torch.no_grad()
def evaluate(model, loader, device, subjects, claims, min_accuracy=0.75):
    model.eval()
    subject_logits, polarity_logits = [], []
    for batch in loader:
        subject, polarity, _ = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )
        subject_logits.append(subject.float().cpu())
        polarity_logits.append(polarity.float().cpu())

    subject_logits = torch.cat(subject_logits)
    polarity_logits = torch.cat(polarity_logits)
    probabilities = torch.softmax(subject_logits, dim=1).numpy()
    predicted = probabilities.argmax(axis=1)
    index_of = {name: index for index, name in enumerate(subjects)}
    truth = np.array([index_of[claim.subject] for claim in claims])

    confidence = probabilities.max(axis=1)
    correct = (predicted == truth).astype(float)
    macro, per_class = macro_f1(truth, predicted, subjects)

    polarity_index = {name: index for index, name in enumerate(POLARITIES)}
    polarity_truth = np.array([polarity_index.get(claim.polarity, 2) for claim in claims])
    polarity_predicted = polarity_logits.argmax(axis=1).numpy()
    polarity_macro, _ = macro_f1(polarity_truth, polarity_predicted, POLARITIES)

    by_language = defaultdict(list)
    by_length = defaultdict(list)
    for claim, hit in zip(claims, correct):
        by_language[claim.language or "unknown"].append(hit)
        bucket = "short" if len(claim.text) < 40 else "medium" if len(claim.text) < 140 else "long"
        by_length[bucket].append(hit)

    contested = np.array([claim.ambiguous for claim in claims])
    abstention = {}
    for floor in (0.3, 0.5, 0.7, 0.9):
        sure = confidence >= floor
        abstention[str(floor)] = {
            "answers_for": float(sure.mean()),
            "accuracy": float(correct[sure].mean()) if sure.any() else None,
        }

    curve, chosen = risk_coverage(confidence, correct, min_accuracy)

    return {
        "accuracy": float(correct.mean()),
        "macro_f1": macro,
        "threshold": chosen["threshold"],
        "threshold_coverage": chosen["coverage"],
        "threshold_accuracy": chosen["accuracy"],
        "threshold_met": chosen["met"],
        "min_accuracy": min_accuracy,
        "risk_coverage": curve,
        "aurc": area_under_risk_coverage(confidence, correct),
        "polarity_macro_f1": polarity_macro,
        "calibration_error": expected_calibration_error(confidence, correct),
        "on_clear_cut": float(correct[~contested].mean()) if (~contested).any() else None,
        "on_contested": float(correct[contested].mean()) if contested.any() else None,
        "per_subject": per_class,
        "per_language": {
            name: {"accuracy": float(np.mean(hits)), "claims": len(hits)}
            for name, hits in sorted(by_language.items(), key=lambda pair: -len(pair[1]))
        },
        "per_length": {
            name: {"accuracy": float(np.mean(hits)), "claims": len(hits)}
            for name, hits in by_length.items()
        },
        "abstention": abstention,
    }


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run(args) -> dict:
    # The classifier heads start from noise and the training order is shuffled, so two runs of
    # identical settings land a point or two apart. Comparing configurations means separating
    # that from the difference being measured, which means being able to repeat a run.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    claims = claimdata.load(args.data)
    train, validation, test = claimdata.split_by_game(claims, seed=args.split_seed)
    subjects = claimdata.subjects_in(claims)
    print(claimdata.summarise(claims))

    # A learning curve asks what the next game buys, and the only way to read one is against a
    # test set that does not move. Training games are dropped in a fixed hash order so that
    # ten games is the same ten in every run, and the frozen and validation games are never
    # touched: the point is to vary what the model learned from, not what it is asked about.
    if args.train_games:
        kept = sorted(
            {claim.app_id for claim in train},
            key=lambda app_id: claimdata.place(app_id, args.split_seed),
        )[: args.train_games]
        train = [claim for claim in train if claim.app_id in kept]
        print(f"training on {len(kept)} of the training games, by hash order")

    print(f"train {len(train)}  validation {len(validation)}  test {len(test)} (frozen)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    model = ClaimReader(args.backbone, len(subjects)).to(device)

    loaders = {
        name: DataLoader(
            Claims(
                part,
                tokenizer,
                subjects,
                args.max_length,
                args.context,
                # Only the training split is reweighted. Scoring a contested claim at less
                # than a whole claim would be marking the model's own exam generously.
                args.ambiguous_weight if name == "train" else 1.0,
                args.split_wrong_weight if name == "train" else 1.0,
            ),
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=0,
        )
        for name, part in (("train", train), ("validation", validation), ("test", test))
    }

    steps = len(loaders["train"]) * args.epochs
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    schedule = get_linear_schedule_with_warmup(optimiser, int(steps * 0.1), steps)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")

    # Rare subjects would otherwise be drowned by `verdict`, which is most of any corpus.
    counts = claimdata.distribution(train)
    weights = torch.tensor(
        [len(train) / (len(subjects) * max(1, counts.get(name, 0))) for name in subjects],
        dtype=torch.float,
        device=device,
    )

    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(loaders["train"]):
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=device == "cuda", dtype=torch.bfloat16):
                subject, polarity, _ = model(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device)
                )
                trust = batch["weight"].to(device)
                subject_loss = torch.nn.functional.cross_entropy(
                    subject, batch["subject"].to(device), weight=weights, reduction="none"
                )
                polarity_loss = torch.nn.functional.cross_entropy(
                    polarity, batch["polarity"].to(device), reduction="none"
                )
                loss = ((subject_loss + args.polarity_weight * polarity_loss) * trust).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            running += float(loss)
            if step % 50 == 0:
                print(f"  epoch {epoch + 1} step {step}/{len(loaders['train'])} loss {running / (step + 1):.4f}")

        metrics = evaluate(
            model, loaders["validation"], device, subjects, validation, args.min_accuracy
        )
        print(
            f"epoch {epoch + 1}: accuracy {metrics['accuracy']:.3f}  "
            f"macro F1 {metrics['macro_f1']:.3f}  polarity {metrics['polarity_macro_f1']:.3f}  "
            f"calibration {metrics['calibration_error']:.3f}"
        )
        answers = "answers nothing at that accuracy" if not metrics["threshold_met"] else (
            f"answers {metrics['threshold_coverage']:.0%} of claims at "
            f"{metrics['threshold_accuracy']:.3f}"
        )
        print(
            f"  abstains below {metrics['threshold']:.2f}: {answers}"
            f"{'' if metrics['threshold_met'] else f' (wanted {args.min_accuracy:.2f})'}"
        )

    # Scored once more outside the loop, so what `run.json` reports is measured from the same
    # weights that get saved rather than from whichever epoch happened to be last.
    metrics = evaluate(
        model, loaders["validation"], device, subjects, validation, args.min_accuracy
    )

    # The frozen games, read once, with the threshold the validation games chose. Nothing here
    # picks anything: the moment a number from this set changes a setting, the set stops being
    # able to answer the only question it exists for. That is why it can be switched off, and
    # why the bake-off switches it off: comparing four backbones on it and then reporting the
    # winner's score from it would be reporting a number the winner was chosen by.
    #
    # It is asked at all because the validation figure was not transferring. Measured on eleven
    # games, a threshold promising 0.756 on the validation games delivered 0.620 on the frozen
    # ones, and a model card quoting the first would advertise an accuracy the tool does not
    # have on a game it has never seen.
    held = None
    if getattr(args, "frozen", True):
        held = evaluate(model, loaders["test"], device, subjects, test, args.min_accuracy)
        kept = confidence_of(model, loaders["test"], device)
        truth = [subjects.index(claim.subject) for claim in test]
        at_threshold = selective(kept, truth, metrics["threshold"])
        held["at_validation_threshold"] = at_threshold
        print(
            f"frozen games: {at_threshold['coverage']:.0%} of claims answered at "
            f"{at_threshold['accuracy']:.3f}"
            if at_threshold["coverage"] > 0
            else "frozen games: nothing cleared the threshold"
        )
        if (
            metrics["threshold_met"]
            and at_threshold["accuracy"] is not None
            and at_threshold["accuracy"] < args.min_accuracy
        ):
            print(
                f"  WARNING: the threshold promised {args.min_accuracy:.2f} and delivers "
                f"{at_threshold['accuracy']:.3f} on games it has never seen. The promise was "
                f"chosen on {len({claim.app_id for claim in validation})} validation games and "
                f"does not transfer; quote the frozen figure, not the validation one."
            )

    elapsed = time.time() - started
    record = {
        "backbone": args.backbone,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "context": args.context,
        "seed": args.seed,
        "ambiguous_weight": args.ambiguous_weight,
        "split_wrong_weight": args.split_wrong_weight,
        "polarity_weight": args.polarity_weight,
        "split_seed": args.split_seed,
        "device": device,
        "git_sha": git_sha(),
        "data_fingerprint": claimdata.fingerprint(claims),
        "claims": {"train": len(train), "validation": len(validation), "test": len(test)},
        "games": {
            "train": sorted({claim.app_id for claim in train}),
            "validation": sorted({claim.app_id for claim in validation}),
            "test": sorted({claim.app_id for claim in test}),
        },
        "subjects": subjects,
        "seconds": round(elapsed),
        "validation": metrics,
    }
    if held is not None:
        record["test"] = held

    run_id = args.run_id or f"{args.backbone.replace('/', '-')}-{int(time.time())}"
    out = HERE / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    if args.save:
        torch.save(model.state_dict(), out / "model.bin")
        tokenizer.save_pretrained(out / "tokenizer")
    print(f"\nwritten to {out}")
    return record


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(HERE / "data" / "claims.jsonl"))
    parser.add_argument("--backbone", default="xlm-roberta-base")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--polarity-weight", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="seeds the head initialisation and the shuffle, so a run can be repeated. Not "
        "the split seed: which games are held out is a separate decision and never moves.",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.75,
        help="how often the model must be right on the claims it does answer. The abstention "
        "threshold is the one giving the most coverage at this accuracy; if none reaches it, "
        "that is recorded rather than lowered to whatever the model can manage.",
    )
    parser.add_argument(
        "--context",
        action="store_true",
        help="read each claim with the review around it, as the labeller did, rather than "
        "the claim alone. Costs tokens per claim and therefore throughput; the question is "
        "whether a claim that cannot be answered alone stops being one.",
    )
    parser.add_argument(
        "--train-games",
        type=int,
        default=0,
        help="train on only this many of the training games, chosen in a fixed hash order so "
        "that every run of a learning curve uses the same ones. The validation and frozen "
        "games are untouched, so the curve is read against one unmoving test set.",
    )
    parser.add_argument(
        "--ambiguous-weight",
        type=float,
        default=1.0,
        help="what a claim the labeller called genuinely contested is worth in the loss. "
        "Nearly a third of the set is flagged, and the sheet does not settle those "
        "boundaries, so training on them at full weight teaches a confident coin toss.",
    )
    parser.add_argument(
        "--split-wrong-weight",
        type=float,
        default=1.0,
        help="what a claim the labeller called mis-cut is worth in the loss. A sixth of the "
        "set is flagged: two points stuck together, or half of one.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument(
        "--no-frozen",
        dest="frozen",
        action="store_false",
        help="leave the frozen games unread. For runs that compare configurations against each "
        "other: a set used to choose between models cannot also say how the chosen one does.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse())
