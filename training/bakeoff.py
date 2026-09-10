"""Decides the backbone by measurement rather than by reputation.

Every candidate sees the same claims, the same split and the same schedule, and is judged on
two things together: how well it reads a claim, and how fast it reads thirteen million of
them. A model two points better and three times slower is not better here.

The encoder this project already ships was chosen the same way, as an embedder. That is a
different job from classifying, which is why it does not get to keep the position unmeasured.

    python bakeoff.py
    python bakeoff.py --epochs 2 --only xlm-roberta-base
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

import train as trainer

HERE = Path(__file__).resolve().parent

# Multilingual, small enough to run over a corpus, and permissively licensed. Anything added
# here has to be exportable to ONNX, because the tool runs the graph and not PyTorch.
CANDIDATES = [
    "xlm-roberta-base",
    "microsoft/mdeberta-v3-base",
    "intfloat/multilingual-e5-base",
    "Alibaba-NLP/gte-multilingual-base",
]


def throughput(backbone: str, batch: int = 64, length: int = 128, rounds: int = 12) -> float:
    """Claims per second on this machine, which is half of what decides the winner."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
    model = AutoModel.from_pretrained(backbone, trust_remote_code=True).to(device).eval()
    text = ["The driving physics feel floaty and the handling in the rain is broken."] * batch
    encoded = tokenizer(
        text, truncation=True, max_length=length, padding="max_length", return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        for _ in range(3):
            model(**encoded)
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.time()
        for _ in range(rounds):
            model(**encoded)
        if device == "cuda":
            torch.cuda.synchronize()
    return (batch * rounds) / (time.time() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(HERE / "data" / "claims.jsonl"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    candidates = args.only or CANDIDATES
    results = []
    for backbone in candidates:
        print(f"\n=== {backbone} ===", flush=True)
        try:
            speed = throughput(backbone)
            record = trainer.run(
                argparse.Namespace(
                    data=args.data,
                    backbone=backbone,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=2e-5,
                    max_length=128,
                    polarity_weight=0.5,
                    split_seed=1,
                    run_id=f"bakeoff-{backbone.replace('/', '-')}",
                    save=False,
                )
            )
        except Exception as failure:  # a candidate that cannot be loaded is a result too
            print(f"  {backbone} failed: {failure}")
            results.append({"backbone": backbone, "error": str(failure)})
            continue
        record["claims_per_second"] = round(speed, 1)
        results.append(record)

    table = HERE / "runs" / "bakeoff.json"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"{'backbone':<40} {'macro F1':>9} {'accuracy':>9} {'claims/s':>10}")
    for record in results:
        if "error" in record:
            print(f"{record['backbone']:<40} {'failed':>9}")
            continue
        metrics = record["validation"]
        print(
            f"{record['backbone']:<40} {metrics['macro_f1']:>9.3f} "
            f"{metrics['accuracy']:>9.3f} {record['claims_per_second']:>10.1f}"
        )
    print(f"\nwritten to {table}")


if __name__ == "__main__":
    main()
