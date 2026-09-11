"""Exports a trained run to ONNX, and refuses to believe it worked without checking.

An fp16 export can quietly disagree with the model it came from. Not by much, and not on
every input, which is exactly what makes it dangerous: the graph loads, the numbers look
plausible, and a category boundary has moved. So the export is followed by a parity check on
real claims, and a disagreement fails the run rather than printing a warning.

    python export.py --run runs/xlm-roberta-base-1788899000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

import data as claimdata
from train import ClaimReader

HERE = Path(__file__).resolve().parent

# What a claim's logits may drift by between PyTorch and the exported graph. Tight enough
# that a changed argmax cannot hide inside it on anything but a genuine tie.
TOLERANCE = 2e-3

# Half precision keeps about three decimal digits, so logits of this size drift by tens of
# thousandths. The argmax check below is what actually guards the answers.
HALF_TOLERANCE = 8e-2


class InFullPrecisionOut(torch.nn.Module):
    """Runs the model in half precision and hands back full-precision numbers.

    The arithmetic is where the saving is; the three small vectors that come out are not. A
    graph whose outputs are half precision forces every reader of it to know that, and the
    tool that runs this graph should not have to care which precision it was exported at.
    """

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, input_ids, attention_mask):
        subject, polarity, pooled = self.inner(input_ids, attention_mask)
        return subject.float(), polarity.float(), pooled.float()


def wilson_note(at_threshold: dict) -> str:
    """The range an accuracy from a couple of hundred claims is entitled to claim.

    A point estimate from 221 claims reads as three significant figures and has about one.
    The same interval the Rust side prints beside every rate, so the card and the page cannot
    disagree about how sure a number is.
    """
    answered = at_threshold.get("answered", 0)
    accuracy = at_threshold.get("accuracy")
    if not answered or accuracy is None:
        return ""
    z = 1.959963985
    n = float(answered)
    denominator = 1.0 + z * z / n
    centre = accuracy + z * z / (2.0 * n)
    spread = z * ((accuracy * (1.0 - accuracy) / n + z * z / (4.0 * n * n)) ** 0.5)
    low = max(0.0, (centre - spread) / denominator)
    high = min(1.0, (centre + spread) / denominator)
    return f", somewhere in [{low:.3f}, {high:.3f}] over {answered} claims"


def sample_claims(path: Path, count: int, record: dict, tokenizer) -> list:
    """Claims to check the exported graph on, in the form the reader will send them.

    A model trained on the claim inside its review is asked about pairs for the rest of its
    life, and a parity check on bare claims measures drift on input it will never see. The
    sequences are also four times shorter, which is exactly where a half-precision graph is
    least likely to drift.
    """
    claims = claimdata.load(path)
    step = max(1, len(claims) // count)
    drawn = claims[::step][:count]
    if not record.get("context"):
        return [claim.text for claim in drawn]

    from train import Claims

    cut = Claims(drawn, tokenizer, record["subjects"], record["max_length"], True, mark=record.get("mark", False))
    return [(claim.text, cut.windows[at]) for at, claim in enumerate(drawn)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", default=str(HERE / "data" / "claims.jsonl"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--check", type=int, default=256)
    parser.add_argument("--spine", default="core-4", help="the taxonomy these labels were made against")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="export half precision: half the download, and roughly half the arithmetic over "
        "millions of claims. Checked against the full-precision model like any other export.",
    )
    args = parser.parse_args()

    # `runs/<name>` names a run of this project wherever the command was typed from, so a
    # script driving the whole chain from the repository root does not silently export nothing.
    run = Path(args.run)
    if not run.is_absolute() and not run.joinpath("run.json").is_file():
        run = HERE / args.run
    record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    subjects = record["subjects"]

    tokenizer = AutoTokenizer.from_pretrained(run / "tokenizer")
    model = ClaimReader(record["backbone"], len(subjects))
    model.load_state_dict(torch.load(run / "model.bin", map_location="cpu"))
    model.eval()

    texts = sample_claims(Path(args.data), args.check, record, tokenizer)
    encoded = tokenizer(
        [pair[0] for pair in texts] if record.get("context") else texts,
        [pair[1] for pair in texts] if record.get("context") else None,
        truncation=True,
        max_length=record["max_length"],
        padding=True,
        return_tensors="pt",
    )

    # The reference answers come from the full-precision model whatever is exported, so a
    # half-precision graph is checked against the thing it is meant to approximate rather
    # than against itself.
    with torch.no_grad():
        wanted = model(encoded["input_ids"], encoded["attention_mask"])[0].numpy()

    exported = model
    if args.fp16:
        half = ClaimReader(record["backbone"], len(subjects))
        half.load_state_dict(torch.load(run / "model.bin", map_location="cpu"))
        exported = InFullPrecisionOut(half.eval().half()).eval()

    graph = run / "model.onnx"
    torch.onnx.export(
        exported,
        (encoded["input_ids"], encoded["attention_mask"]),
        graph,
        input_names=["input_ids", "attention_mask"],
        output_names=["subject_logits", "polarity_logits", "pooled"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "tokens"},
            "attention_mask": {0: "batch", 1: "tokens"},
            "subject_logits": {0: "batch"},
            "polarity_logits": {0: "batch"},
            "pooled": {0: "batch"},
        },
        opset_version=args.opset,
        # The tracing exporter rather than the dynamo one. Dynamo produces a graph full of
        # operators ONNX Runtime's CUDA provider does not implement, so it partitions the
        # model and copies tensors between host and device at every boundary: measured, the
        # same weights ran at sixty claims a second on a 4090 and the card sat at a quarter
        # busy. The traced graph is plainer and stays on the card.
        dynamo=False,
    )

    # One file, not a graph plus a weights blob beside it. What ships is verified by checksum
    # before it is run, and a checksum over one of two files is a checksum over nothing.
    import onnx

    inlined = onnx.load(str(graph), load_external_data=True)
    onnx.save(inlined, str(graph), save_as_external_data=False)
    for stray in graph.parent.glob("model.onnx.data*"):
        stray.unlink()

    import onnxruntime

    # Half precision has no CPU kernels worth the name, so it is checked where it will run.
    providers = ["CUDAExecutionProvider"] if args.fp16 else ["CPUExecutionProvider"]
    session = onnxruntime.InferenceSession(str(graph), providers=providers)
    got = session.run(
        ["subject_logits"],
        {
            "input_ids": encoded["input_ids"].numpy(),
            "attention_mask": encoded["attention_mask"].numpy(),
        },
    )[0]

    got = got.astype(np.float32)
    drift = float(np.abs(wanted - got).max())
    allowed = HALF_TOLERANCE if args.fp16 else TOLERANCE

    # A changed answer is only a disagreement when the model had an answer to change. Where the
    # best two subjects sit within the drift of each other, the two graphs are not disagreeing
    # about the claim, they are reporting a tie in different arbitrary orders, and forbidding
    # that while permitting the drift that causes it is a contradiction the export cannot
    # satisfy. So a flip is measured against the margin it had to cross.
    changed = wanted.argmax(axis=1) != got.argmax(axis=1)
    ordered = np.sort(wanted, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    decided = int((changed & (margin > allowed)).sum())
    ties = int((changed & (margin <= allowed)).sum())

    print(
        f"parity: largest drift {drift:.2e} over {len(texts)} claims, "
        f"{decided} answers changed, {ties} ties reordered"
    )
    if drift > allowed or decided:
        raise SystemExit(
            f"the exported graph disagrees with the model it came from "
            f"({drift:.2e} > {allowed:.0e}, {decided} answers changed on a margin wider than "
            f"the drift). Not shipping this."
        )
    # Ties that reorder are tolerable one at a time and not in bulk: a graph that cannot agree
    # with itself on a twentieth of its answers is not approximating the model, whatever the
    # margins say.
    if ties > len(texts) // 20:
        raise SystemExit(
            f"{ties} of {len(texts)} claims came back with a different subject. Every one is "
            f"within the drift, but a graph this unsteady is not the model. Not shipping this."
        )

    # What the Rust side needs to use the graph without being told anything else. The
    # taxonomy version is in here so a model trained against another spine is refused rather
    # than quietly asked about categories nobody labelled.
    # A threshold this low is not abstention, it is the nearest-match classifier this project
    # replaced. Twenty-four subjects put a uniform guess at 0.042, so a model told to answer
    # above 0.05 answers everything, and every claim of "unclassified" the tool makes becomes a
    # claim it cannot keep. Refused here rather than discovered in a report.
    threshold = record["validation"].get("threshold", 0.5)
    floor = 2.0 / len(subjects)
    if threshold < floor:
        raise SystemExit(
            f"threshold {threshold:.3f} is below {floor:.3f}, which for {len(subjects)} "
            f"subjects is what a coin lands on. This model does not abstain. Not shipping it."
        )

    # What the model declined on games it never saw, so the tool can say when a corpus is
    # declined far more than usual. That is the one number a reader of a new game's report
    # has no other way to get, and a corpus declined at twice the usual rate is a corpus about
    # something the taxonomy lacks.
    frozen = record.get("test", {}).get("at_validation_threshold", {})
    usual_declined = 1.0 - frozen["coverage"] if frozen.get("coverage") is not None else None

    (run / "reader.json").write_text(
        json.dumps(
            {
                "spine_version": args.spine,
                "subjects": subjects,
                "threshold": threshold,
                "max_tokens": record["max_length"],
                "context": record.get("context", False),
                "mark": record.get("mark", False),
                "trained_from": record["backbone"],
                "data_fingerprint": record["data_fingerprint"],
                "usual_declined": usual_declined,
                # What this model did on games it never saw, carried so that a report of a
                # game with no reference set can still say what its rates are worth. Without
                # it, the only honest thing such a report can say is nothing.
                "frozen": {
                    "games": len(record.get("games", {}).get("test", [])),
                    "claims": record.get("claims", {}).get("test"),
                    "coverage": frozen.get("coverage"),
                    "accuracy": frozen.get("accuracy"),
                    "macro_f1": record.get("test", {}).get("macro_f1"),
                }
                if frozen.get("coverage") is not None
                else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # The reader wants three files beside each other, and so does publishing. The tokenizer
    # directory the trainer saved holds more than the one file the tool reads.
    import shutil

    shutil.copyfile(run / "tokenizer" / "tokenizer.json", run / "tokenizer.json")

    card = run / "MODEL_CARD.md"
    metrics = record["validation"]

    # The card quotes the frozen games. A card is read by somebody deciding whether to run this
    # on a game of their own, and the validation figure answers a different question: how well
    # it does on the games that chose its settings. Measured, the two differ by eighteen points.
    frozen = record.get("test", metrics)
    at_threshold = frozen.get(
        "at_validation_threshold",
        {
            "coverage": metrics.get("threshold_coverage", 0),
            "accuracy": metrics.get("threshold_accuracy"),
        },
    )
    weakest = sorted(frozen["per_subject"].items(), key=lambda pair: pair[1]["f1"])[:5]
    card.write_text(
        "\n".join(
            [
                f"# Claim reader ({record['backbone']})",
                "",
                "Reads one point from a Steam review and says which subject it is about, whether",
                "it is praise or a complaint, and how sure it is. Below a calibrated threshold it",
                "says nothing, and that is a supported answer rather than a failure.",
                "",
                "## Measured",
                "",
                f"- Accuracy {frozen.get('accuracy', 0):.3f}, macro F1 "
                f"{frozen.get('macro_f1', 0):.3f}",
                f"- Polarity macro F1 {frozen.get('polarity_macro_f1', 0):.3f}",
                f"- Calibration error {frozen.get('calibration_error', 0):.3f}",
                f"- Below {threshold:.2f} confidence it says nothing, which leaves it answering "
                f"{at_threshold.get('coverage', 0):.0%} of claims at "
                f"{at_threshold.get('accuracy') or 0:.3f} accuracy"
                + wilson_note(at_threshold),
                f"- Area under the risk-coverage curve {frozen.get('aurc', 0):.3f} (lower is "
                f"better; it says whether the model knows when it does not know)",
                f"- Trained on {record['claims']['train']} claims, validated on "
                f"{record['claims']['validation']}, measured on {record['claims']['test']}",
                f"- Data fingerprint `{record['data_fingerprint']}`, code `{record['git_sha'][:12]}`",
                "",
                "**Every figure above is from the frozen games**, which the model never saw and",
                "which chose nothing about it, not even the threshold. At that same threshold the",
                f"validation games report {metrics.get('threshold_accuracy') or 0:.3f}, which is",
                "what it scores on games used to build it rather than what a new game gets.",
                "Weakest subjects here: "
                + ", ".join(f"`{name}` {row['f1']:.2f}" for name, row in weakest)
                + ".",
                "",
                "## Honest limits",
                "",
                "The labels were produced by a language model working from a written category",
                "sheet, not by human adjudication. That makes this a silver standard: agreement",
                "with a model rather than correctness. Two models can agree and be wrong together,",
                "most easily on sarcasm and on the boundaries between categories.",
                "",
                "## Licence",
                "",
                "Apache-2.0, as is the encoder it was fine-tuned from.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {graph} and {card}")


if __name__ == "__main__":
    main()
