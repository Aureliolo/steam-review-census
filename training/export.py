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


def sample_claims(path: Path, count: int) -> list[str]:
    claims = claimdata.load(path)
    step = max(1, len(claims) // count)
    return [claim.text for claim in claims[::step]][:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", default=str(HERE / "data" / "claims.jsonl"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--check", type=int, default=256)
    parser.add_argument("--spine", default="core-4", help="the taxonomy these labels were made against")
    args = parser.parse_args()

    run = Path(args.run)
    record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    subjects = record["subjects"]

    tokenizer = AutoTokenizer.from_pretrained(run / "tokenizer")
    model = ClaimReader(record["backbone"], len(subjects))
    model.load_state_dict(torch.load(run / "model.bin", map_location="cpu"))
    model.eval()

    texts = sample_claims(Path(args.data), args.check)
    encoded = tokenizer(
        texts, truncation=True, max_length=record["max_length"], padding=True, return_tensors="pt"
    )

    graph = run / "model.onnx"
    torch.onnx.export(
        model,
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
    )

    # One file, not a graph plus a weights blob beside it. What ships is verified by checksum
    # before it is run, and a checksum over one of two files is a checksum over nothing.
    import onnx

    inlined = onnx.load(str(graph), load_external_data=True)
    onnx.save(inlined, str(graph), save_as_external_data=False)
    for stray in graph.parent.glob("model.onnx.data*"):
        stray.unlink()

    import onnxruntime

    with torch.no_grad():
        wanted = model(encoded["input_ids"], encoded["attention_mask"])[0].numpy()

    session = onnxruntime.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    got = session.run(
        ["subject_logits"],
        {
            "input_ids": encoded["input_ids"].numpy(),
            "attention_mask": encoded["attention_mask"].numpy(),
        },
    )[0]

    drift = float(np.abs(wanted - got).max())
    moved = int((wanted.argmax(axis=1) != got.argmax(axis=1)).sum())
    print(f"parity: largest drift {drift:.2e} over {len(texts)} claims, {moved} answers changed")
    if drift > TOLERANCE or moved:
        raise SystemExit(
            f"the exported graph disagrees with the model it came from "
            f"({drift:.2e} > {TOLERANCE:.0e}, {moved} answers changed). Not shipping this."
        )

    # What the Rust side needs to use the graph without being told anything else. The
    # taxonomy version is in here so a model trained against another spine is refused rather
    # than quietly asked about categories nobody labelled.
    (run / "reader.json").write_text(
        json.dumps(
            {
                "spine_version": args.spine,
                "subjects": subjects,
                "threshold": record["validation"].get("threshold", 0.5),
                "max_tokens": record["max_length"],
                "trained_from": record["backbone"],
                "data_fingerprint": record["data_fingerprint"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    card = run / "MODEL_CARD.md"
    metrics = record["validation"]
    weakest = sorted(metrics["per_subject"].items(), key=lambda pair: pair[1]["f1"])[:5]
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
                f"- Accuracy {metrics['accuracy']:.3f}, macro F1 {metrics['macro_f1']:.3f}",
                f"- Polarity macro F1 {metrics['polarity_macro_f1']:.3f}",
                f"- Calibration error {metrics['calibration_error']:.3f}",
                f"- Trained on {record['claims']['train']} claims, validated on "
                f"{record['claims']['validation']}, held out {record['claims']['test']}",
                f"- Data fingerprint `{record['data_fingerprint']}`, code `{record['git_sha'][:12]}`",
                "",
                "Games are split whole, never claims, so these figures are about a game the model",
                "never saw. Weakest subjects here: "
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
