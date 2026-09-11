"""Publishes an exported run to Hugging Face, and pins it in the tool in the same motion.

Two things go up. The model: the graph, its tokenizer and reader.json, with the card the
export wrote. And the dataset it was trained from: review ids, claim offsets and labels, and
never the text, because this repository holds no review data and neither will that one.

Publishing without pinning would be the one thing worse than not publishing. The tool fetches
the reader by checksum, so a model uploaded and not pinned is a model nobody can use, and a
model pinned by hand is a hash somebody typed. This computes the hashes from the exact bytes
it uploaded and writes them into reader.rs.

    python publish.py --run runs/wave2 --model-repo <user>/steam-review-claim-reader \
        --data-repo <user>/steam-review-claims

Needs a Hugging Face token with write access, from `huggingface-cli login` or HF_TOKEN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
READER_RS = REPO / "crates" / "census-core" / "src" / "reader.rs"

MODEL_FILES = ["model.onnx", "tokenizer.json", "reader.json"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_card(labels: int, games: int, fingerprint: str) -> str:
    return "\n".join(
        [
            "---",
            "license: apache-2.0",
            "task_categories:",
            "- text-classification",
            "language:",
            "- multilingual",
            "pretty_name: Steam review claims",
            "---",
            "",
            "# Steam review claims",
            "",
            f"{labels:,} claims from {games} games, each labelled with the subject it is",
            "about, whether it is praise or a complaint, whether it is ironic, how sure the",
            "labeller was, whether the call was genuinely contested, and whether the claim was",
            "cut in the wrong place.",
            "",
            "## What is not here",
            "",
            "The review text. Each row carries a Steam review id and the byte offsets of one",
            "claim within that review's text, and nothing else that a reviewer wrote. The text is",
            "Valve's and the reviewer's, and it stays where it is; anyone with the review id can",
            "fetch it from Steam and cut the claim out at the offsets given. `steam-review-census`",
            "does exactly that, and so can you.",
            "",
            "## How the labels were made",
            "",
            "By a language model, working from a written category sheet, one game at a time and",
            "without being told which game. This is a silver standard: what it measures is",
            "agreement between models, not correctness. A tenth of the set is labelled a second",
            "time by a different model, and the two agree on the subject 86% of the time (Cohen's",
            "kappa 0.85) and on polarity 93% (kappa 0.89). They agree far less about whether a",
            "claim is contested (kappa 0.28), which is a fact about the labellers rather than",
            "the claims and is documented in the repository.",
            "",
            f"Data fingerprint `{fingerprint}`. The splitter that produced the offsets and the",
            "taxonomy the subjects come from are both versioned in the repository; a row names",
            "which of each it was made under.",
            "",
            "## Licence",
            "",
            "Apache-2.0. The labels are ours to give; the reviews are not, and are not here.",
        ]
    )


def pin(repo: str, hashes: dict[str, str]) -> None:
    """Writes the repository and hashes into reader.rs, where the tool reads them from."""
    source = READER_RS.read_text(encoding="utf-8")

    def replace_once(pattern: str, replacement: str, text: str) -> str:
        found = re.subn(pattern, replacement, text, count=1, flags=re.S)
        if found[1] != 1:
            raise SystemExit(f"could not find where to pin {pattern!r} in {READER_RS}")
        return found[0]

    source = replace_once(
        r'(pub const PUBLISHED: Published = Published \{\s*repository: )"[^"]*"',
        rf'\1"{repo}"',
        source,
    )
    for name, digest in hashes.items():
        source = replace_once(
            rf'(remote: "{re.escape(name)}",\s*local: "{re.escape(name)}",\s*sha256: )"[^"]*"',
            rf'\1"{digest}"',
            source,
        )
    READER_RS.write_text(source, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--data-repo", required=True)
    parser.add_argument("--reference", default=str(REPO / "reference" / "claims"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="hash and pin without uploading, to see what would change",
    )
    args = parser.parse_args()

    run = Path(args.run)
    for name in MODEL_FILES:
        if not (run / name).is_file():
            raise SystemExit(f"{run / name} is missing; export the run first")
    if not (run / "MODEL_CARD.md").is_file():
        raise SystemExit(f"{run} has no model card; export writes one")

    hashes = {name: sha256(run / name) for name in MODEL_FILES}
    for name, digest in hashes.items():
        print(f"{name:<16} {digest}")

    # The dataset is built from the reference sets, not from the training export. The export
    # carries each claim's text because training needs it; the reference sets carry the
    # claim's offsets within its review because that is all a label needs to be joined back.
    # Only the second shape leaves this machine.
    rows = []
    for labels in sorted(Path(args.reference).glob("*/labels.json")):
        rows.extend(json.loads(labels.read_text(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"no labelled sets under {args.reference}")

    # Checked rather than trusted, because a text field arriving in a later version of the
    # reference format would be published before anyone noticed.
    forbidden = {"text", "review", "claim", "body"}
    leaking = sorted(forbidden & set().union(*(row.keys() for row in rows)))
    if leaking:
        raise SystemExit(f"the reference sets carry review text in {leaking}; not publishing that")
    for name in ("review_id", "start", "end", "subject"):
        if any(name not in row for row in rows):
            raise SystemExit(f"a row has no {name}; nothing can be joined back from it")

    data = run / "claims.jsonl"
    data.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    games = len({row["app_id"] for row in rows})
    record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    card = dataset_card(len(rows), games, record["data_fingerprint"])

    if args.dry_run:
        print(f"\nwould upload {len(rows):,} rows from {games} games to {args.data_repo}")
        print(f"would upload {', '.join(MODEL_FILES)} to {args.model_repo}")
        pin(args.model_repo, hashes)
        print(f"pinned in {READER_RS.relative_to(REPO)}")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.model_repo, repo_type="model", exist_ok=True)
    api.create_repo(args.data_repo, repo_type="dataset", exist_ok=True)

    for name in MODEL_FILES:
        api.upload_file(
            path_or_fileobj=str(run / name),
            path_in_repo=name,
            repo_id=args.model_repo,
            repo_type="model",
        )
    api.upload_file(
        path_or_fileobj=str(run / "MODEL_CARD.md"),
        path_in_repo="README.md",
        repo_id=args.model_repo,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=str(data),
        path_in_repo="claims.jsonl",
        repo_id=args.data_repo,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.data_repo,
        repo_type="dataset",
    )

    # Hashed before the upload, from the same files the upload read. If anything between here
    # and the hub changes a byte, the pin will not match what comes down, and the tool refuses
    # it at fetch time, which is the failure this whole arrangement exists to make loud.
    pin(args.model_repo, hashes)
    print(f"\npublished to {args.model_repo} and {args.data_repo}")
    print(f"pinned in {READER_RS.relative_to(REPO)}; rebuild and commit")


if __name__ == "__main__":
    main()
